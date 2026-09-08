
import argparse

import csv

import json

import os

import shutil

import time

import matplotlib.pyplot as plt

import numpy as np

import torch

from benchmark import ICUSepsisOfflineBenchmark

from models import OfflineActorCriticNet, soft_update

from trainers import (
    ensure_dir,
    mean_ci95,
    plain_softmax_from_logits,
    plain_log_softmax_from_logits,
    masked_softmax_from_logits,
    masked_log_softmax_from_logits,
    greedy_policy_from_logits,
    masked_greedy_policy_from_logits,
    cql_regularizer,
    masked_action_mse,
    evaluate_policy_set,
    policy_numpy,
    save_model_run,
)

PLOT_DPI = 600

PLOT_PALETTE = {
    "Full LAADAN-AC": "#2171b5",
    "LAADAN without action mask": "#6baed6",
    "LAADAN without conservative critic": "#9ecae1",
    "LAADAN without expert KL": "#4292c6",
    "LAADAN without smoothness proxy": "#bdd7e7",
    "LAADAN without Lagrangian cost control": "#08306b",
    "No-mask no-Lagrangian": "#c6dbef",
    "Masking-only actor-critic": "#08519c",
    "Post-hoc Masked VOAC": "#3182bd",
}


# lagrangian analysis
def colour_for(name):

    if name in PLOT_PALETTE:
        return PLOT_PALETTE[name]

    if "budget" in name.lower():
        return "#2171b5"

    return "#3182bd"

BASE_LAADAN_CONFIG = {

    "epochs": 1000,

    "actor_lr": 5e-4,

    "critic_lr": 1e-3,

    "dropout": 0.10,

    "hidden_dim": 128,

    "latent_dim": 128,

    "eval_every": 10,

    "gamma": 1.0,

    "tau": 0.01,

    "entropy_coef": 0.001,

    "conservative_alpha": 0.5,

    "expert_kl_weight": 0.005,

    "smoothness_weight": 0.001,

    "cost_budget": 0.0,

    "lagrange_lr": 0.0002,

    "lagrange_init": 0.0,

    "weight_decay": 1e-5,
}


# BudgetedLagrangianExperiment
class BudgetedLagrangianExperiment:

    def __init__(self, benchmark, results_dir, config):

        self.benchmark = benchmark

        self.results_dir = results_dir

        self.config = dict(config)

    def build_model_pair(self, use_cost_head):

        model = OfflineActorCriticNet(
            self.benchmark.feature_dim,
            self.benchmark.num_actions,
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            latent_dim=int(self.config.get("latent_dim", 128)),
            dropout=float(self.config.get("dropout", 0.10)),
            use_cost_head=bool(use_cost_head),
        ).to(self.benchmark.device)

        target_model = OfflineActorCriticNet(
            self.benchmark.feature_dim,
            self.benchmark.num_actions,
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            latent_dim=int(self.config.get("latent_dim", 128)),
            dropout=float(self.config.get("dropout", 0.10)),
            use_cost_head=bool(use_cost_head),
        ).to(self.benchmark.device)

        target_model.load_state_dict(model.state_dict())

        target_model.eval()

        return model, target_model

    def policy_probs(self, logits, admissible, use_action_mask):

        if use_action_mask:
            return masked_softmax_from_logits(logits, admissible)

        return plain_softmax_from_logits(logits)

    def policy_log_probs(self, logits, admissible, use_action_mask):

        if use_action_mask:
            return masked_log_softmax_from_logits(logits, admissible)

        return plain_log_softmax_from_logits(logits)

    def greedy_policy(self, logits, admissible, use_action_mask):

        if use_action_mask:
            return masked_greedy_policy_from_logits(logits, admissible)

        return greedy_policy_from_logits(logits)

    def train_variant(self, seed, variant):

        self.benchmark.set_seed(seed)

        use_action_mask = bool(variant.get("use_action_mask", True))

        use_conservative = bool(variant.get("use_conservative", True))

        use_expert_kl = bool(variant.get("use_expert_kl", True))

        use_smoothness = bool(variant.get("use_smoothness", True))

        use_lagrangian = bool(variant.get("use_lagrangian", True))

        train_cost_head = bool(variant.get("train_cost_head", use_lagrangian))

        model, target_model = self.build_model_pair(use_cost_head=train_cost_head)

        actor_params = list(model.actor_encoder.parameters()) + list(model.actor_head.parameters())

        critic_params = (
            list(model.critic_encoder.parameters())
            + list(model.q1_head.parameters())
            + list(model.q2_head.parameters())
        )

        if train_cost_head:
            critic_params += list(model.cost_head.parameters())

        actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=float(self.config.get("actor_lr", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
        )

        critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=float(self.config.get("critic_lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
        )

        gamma = float(self.config.get("gamma", 1.0))
        tau = float(self.config.get("tau", 0.01))
        entropy_coef = float(self.config.get("entropy_coef", 0.001))
        conservative_alpha = float(variant.get("conservative_alpha", self.config.get("conservative_alpha", 0.5)))
        expert_kl_weight = float(variant.get("expert_kl_weight", self.config.get("expert_kl_weight", 0.005)))
        smoothness_weight = float(variant.get("smoothness_weight", self.config.get("smoothness_weight", 0.001)))
        cost_budget = float(variant.get("cost_budget", self.config.get("cost_budget", 0.0)))
        lagrange_lr = float(variant.get("lagrange_lr", self.config.get("lagrange_lr", 0.0002)))
        lagrange_value = float(variant.get("lagrange_init", self.config.get("lagrange_init", 0.0)))
        epochs = int(self.config.get("epochs", 1000))
        eval_every = int(self.config.get("eval_every", 10))

        x = self.benchmark.state_features_t

        admissible = self.benchmark.admissible_mask_t

        expert = self.benchmark.expert_safe_t

        reward_sa = self.benchmark.reward_sa_t

        transition = self.benchmark.transition_t

        terminal = self.benchmark.terminal_mask_t

        immediate_cost = self.benchmark.immediate_cost_t

        smoothness_cost = self.benchmark.smoothness_cost_t

        critic_fit_mask = admissible if use_action_mask else None

        conservative_mask = admissible if use_action_mask else None

        history = []

        best_survival = -1.0

        best_state = None

        best_metrics = None

        start_time = time.time()

        for epoch in range(1, epochs + 1):

            model.train()

            with torch.no_grad():

                target_outputs = target_model(x)

                next_probs = self.policy_probs(target_outputs["logits"], admissible, use_action_mask)

                next_log_probs = self.policy_log_probs(target_outputs["logits"], admissible, use_action_mask)

                q1_next = target_outputs["q1"]

                q2_next = target_outputs["q2"]

                min_q_next = torch.min(q1_next, q2_next)

                if train_cost_head:
                    cost_next = torch.relu(target_outputs["cost"])

                else:
                    cost_next = immediate_cost



                next_v = torch.sum(
                    next_probs * (min_q_next - entropy_coef * next_log_probs - lagrange_value * cost_next),
                    dim=1,
                ) * (1.0 - terminal)

                next_c = torch.sum(next_probs * cost_next, dim=1) * (1.0 - terminal)


                q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)


                c_target = immediate_cost + gamma * torch.einsum("san,n->sa", transition, next_c)

            critic_optimizer.zero_grad()

            outputs = model(x)

            q1 = outputs["q1"]

            q2 = outputs["q2"]

            q1_loss = masked_action_mse(q1, q_target, action_mask=critic_fit_mask)


            q2_loss = masked_action_mse(q2, q_target, action_mask=critic_fit_mask)

            critic_loss = q1_loss + q2_loss

            if train_cost_head:
                cost_pred = torch.relu(outputs["cost"])
                cost_loss = masked_action_mse(cost_pred, c_target, action_mask=critic_fit_mask)
                critic_loss = critic_loss + cost_loss

            else:
                cost_pred = immediate_cost
                cost_loss = torch.tensor(0.0, device=self.benchmark.device)



            if use_conservative:
                cql_q1 = cql_regularizer(q1, expert, admissible_mask=conservative_mask)
                cql_q2 = cql_regularizer(q2, expert, admissible_mask=conservative_mask)
                critic_loss = critic_loss + conservative_alpha * (cql_q1 + cql_q2)


            else:
                cql_q1 = torch.tensor(0.0, device=self.benchmark.device)
                cql_q2 = torch.tensor(0.0, device=self.benchmark.device)

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)

            critic_optimizer.step()

            actor_optimizer.zero_grad()

            outputs = model(x)

            logits = outputs["logits"]

            probs = self.policy_probs(logits, admissible, use_action_mask)

            log_probs = self.policy_log_probs(logits, admissible, use_action_mask)

            q1 = outputs["q1"]

            q2 = outputs["q2"]

            min_q = torch.min(q1, q2)

            if train_cost_head:
                cost_pred = torch.relu(outputs["cost"])

            else:
                cost_pred = immediate_cost

            expected_q = torch.sum(probs * min_q, dim=1)

            expected_cost_to_go = torch.sum(probs * cost_pred, dim=1)

            expected_immediate_cost = torch.sum(probs * immediate_cost, dim=1)

            entropy = -torch.sum(probs * log_probs, dim=1)

            actor_loss = torch.mean(-expected_q - entropy_coef * entropy)

            if use_lagrangian:
                actor_loss = actor_loss + lagrange_value * torch.mean(expected_cost_to_go)

            if use_expert_kl:
                expert_kl = torch.sum(expert * (torch.log(expert + 1e-8) - log_probs), dim=1)
                actor_loss = actor_loss + expert_kl_weight * torch.mean(expert_kl)

            else:
                expert_kl = torch.zeros_like(expected_q)

            if use_smoothness:
                expected_smoothness = torch.sum(probs * smoothness_cost, dim=1)
                actor_loss = actor_loss + smoothness_weight * torch.mean(expected_smoothness)

            else:
                expected_smoothness = torch.zeros_like(expected_q)

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)

            actor_optimizer.step()

            soft_update(target_model, model, tau)

            target_model.eval()

            mean_immediate_cost = float(torch.mean(expected_immediate_cost).item())

            mean_cost_to_go = float(torch.mean(expected_cost_to_go).item())

            if use_lagrangian:
                lagrange_value = max(0.0, lagrange_value + lagrange_lr * (mean_immediate_cost - cost_budget))

            row = {
                "epoch": epoch,
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "q1_loss": float(q1_loss.item()),
                "q2_loss": float(q2_loss.item()),
                "cost_loss": float(cost_loss.item()),
                "cql_q1": float(cql_q1.item()),
                "cql_q2": float(cql_q2.item()),
                "expected_immediate_cost": mean_immediate_cost,
                "expected_cost_to_go": mean_cost_to_go,
                "cost_budget": float(cost_budget),
                "lagrange": float(lagrange_value),
                "expert_kl": float(torch.mean(expert_kl).item()),
                "smoothness": float(torch.mean(expected_smoothness).item()),
            }

            if epoch % eval_every == 0 or epoch == epochs:

                model.eval()

                with torch.no_grad():
                    logits_eval = model(x)["logits"]
                    primary_policy = self.greedy_policy(logits_eval, admissible, use_action_mask)
                    soft_policy = self.policy_probs(logits_eval, admissible, use_action_mask)

                metrics = evaluate_policy_set(
                    self.benchmark,
                    policy_numpy(primary_policy),
                    soft_policy=policy_numpy(soft_policy),
                )

                simulation = self.benchmark.simulate_policy(
                    policy_numpy(primary_policy),
                    num_episodes=int(variant.get("num_trajectory_episodes", 2000)),
                    seed=123 + int(seed),
                )

                metrics["trajectory_mean_action_jump"] = float(simulation["mean_action_jump"])

                metrics["trajectory_normalised_action_jump"] = float(
                    simulation["mean_action_jump"] / max(1, self.benchmark.num_actions - 1)
                )

                row.update(metrics)

                if metrics["survival_rate"] > best_survival:
                    best_survival = metrics["survival_rate"]
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    best_metrics = dict(metrics)

            history.append(row)

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()

        with torch.no_grad():
            logits_final = model(x)["logits"]
            final_policy = self.greedy_policy(logits_final, admissible, use_action_mask)
            final_soft_policy = self.policy_probs(logits_final, admissible, use_action_mask)

        if best_metrics is None:
            best_metrics = evaluate_policy_set(
                self.benchmark,
                policy_numpy(final_policy),
                soft_policy=policy_numpy(final_soft_policy),
            )

        best_metrics["variant_folder"] = variant["folder"]
        best_metrics["variant_name"] = variant["name"]
        best_metrics["uses_action_mask"] = use_action_mask
        best_metrics["uses_lagrangian"] = use_lagrangian
        best_metrics["cost_budget"] = float(cost_budget)

        seed_dir = os.path.join(self.results_dir, "lagrangian_frontier", variant["folder"], "seed_" + str(seed))

        save_model_run(seed_dir, model, history, best_metrics)

        return {
            "name": variant["name"],
            "folder": variant["folder"],
            "seed": seed,
            "model": model,
            "policy": policy_numpy(final_policy),
            "analysis_policy": policy_numpy(final_soft_policy),
            "history": history,
            "metrics": best_metrics,
            "train_time_seconds": float(time.time() - start_time),
        }

    def run_variants(self, variants, seeds):

        all_runs = {}

        for variant in variants:

            runs = []

            for seed in seeds:

                print("[TRAIN]", variant["name"], "seed", seed)

                runs.append(self.train_variant(seed, variant))

            all_runs[variant["name"]] = runs

        return all_runs


# FrontierReporter
class FrontierReporter:

    def __init__(self, benchmark, results_dir):

        self.benchmark = benchmark

        self.root = os.path.join(results_dir, "lagrangian_frontier")

        self.figures_dir = os.path.join(self.root, "figures")

        self.final_dir = os.path.join(results_dir, "final_lagrangian_models")

        ensure_dir(self.root)

        ensure_dir(self.figures_dir)

        ensure_dir(self.final_dir)

    def aggregate_runs(self, all_runs):

        summary = {}

        for model_name, runs in all_runs.items():

            metrics = {}

            metric_names = []

            for run in runs:
                for key in run["metrics"].keys():
                    if key not in metric_names and isinstance(run["metrics"][key], (int, float, bool)):
                        metric_names.append(key)

            for metric_name in metric_names:

                values = []

                for run in runs:
                    value = run["metrics"].get(metric_name, np.nan)

                    if isinstance(value, bool):
                        value = float(value)

                    values.append(value)

                values = np.asarray(values, dtype=float)

                valid = values[np.isfinite(values)]

                if valid.size > 0:
                    stats = mean_ci95(valid)
                    stats["min"] = float(np.min(valid))
                    stats["max"] = float(np.max(valid))
                    metrics[metric_name] = stats

            summary[model_name] = metrics

        return summary

    def write_summary_csv(self, summary):

        path = os.path.join(self.root, "budgeted_lagrangian_summary.csv")

        columns = [
            "model",
            "n",
            "survival_rate_mean",
            "survival_rate_ci95_half",
            "inadmissibility_rate_mean",
            "inadmissibility_rate_ci95_half",
            "expert_argmax_match_mean",
            "expert_argmax_match_ci95_half",
            "mean_kl_to_expert_mean",
            "mean_kl_to_expert_ci95_half",
            "mean_action_deviation_from_expert_mean",
            "mean_action_deviation_from_expert_ci95_half",
            "trajectory_normalised_action_jump_mean",
            "trajectory_normalised_action_jump_ci95_half",
        ]

        with open(path, "w", newline="", encoding="utf-8") as handle:

            writer = csv.DictWriter(handle, fieldnames=columns)

            writer.writeheader()

            for model_name, metrics in summary.items():

                row = {"model": model_name}

                if "survival_rate" in metrics:
                    row["n"] = metrics["survival_rate"]["n"]

                for key in columns:
                    if key in ["model", "n"]:
                        continue

                    metric_name = key.replace("_mean", "").replace("_ci95_half", "")

                    if key.endswith("_mean") and metric_name in metrics:
                        row[key] = metrics[metric_name]["mean"]

                    if key.endswith("_ci95_half") and metric_name in metrics:
                        row[key] = metrics[metric_name]["ci95_half"]

                writer.writerow(row)

        return path

    def save_summary_json(self, summary):

        path = os.path.join(self.root, "budgeted_lagrangian_summary.json")

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        return path

    def copy_if_exists(self, source, destination):

        if os.path.exists(source):
            shutil.copy2(source, destination)

    def select_final_models(self, all_runs):

        rows = []

        for model_name, runs in all_runs.items():

            best_run = None

            for run in runs:

                if best_run is None:
                    best_run = run
                    continue

                survival = float(run["metrics"].get("survival_rate", -1e9))
                inad = float(run["metrics"].get("inadmissibility_rate", 1e9))

                best_survival = float(best_run["metrics"].get("survival_rate", -1e9))
                best_inad = float(best_run["metrics"].get("inadmissibility_rate", 1e9))

                if survival > best_survival or (survival == best_survival and inad < best_inad):
                    best_run = run

            folder = best_run["folder"]

            seed_folder = "seed_" + str(best_run["seed"])

            src_dir = os.path.join(self.root, folder, seed_folder)

            dst_dir = os.path.join(self.final_dir, folder)

            ensure_dir(dst_dir)

            for filename in ["model.pt", "metrics.json", "history.csv"]:
                self.copy_if_exists(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))

            rows.append({
                "folder": folder,
                "model": model_name,
                "best_seed": seed_folder,
                "survival_rate": best_run["metrics"].get("survival_rate", ""),
                "inadmissibility_rate": best_run["metrics"].get("inadmissibility_rate", ""),
                "expert_argmax_match": best_run["metrics"].get("expert_argmax_match", ""),
                "cost_budget": best_run["metrics"].get("cost_budget", ""),
            })

        csv_path = os.path.join(self.final_dir, "final_lagrangian_model_selection.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "folder",
                    "model",
                    "best_seed",
                    "survival_rate",
                    "inadmissibility_rate",
                    "expert_argmax_match",
                    "cost_budget",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        json_path = os.path.join(self.final_dir, "final_lagrangian_model_selection.json")

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)

        return csv_path, json_path

    def plot_pareto_frontier(self, summary):

        points = []


        if "LAADAN without action mask" in summary:
            metrics = summary["LAADAN without action mask"]
            if "survival_rate" in metrics and "inadmissibility_rate" in metrics:
                points.append((
                    "d=0.000",
                    metrics["inadmissibility_rate"]["mean"],
                    metrics["survival_rate"]["mean"],
                    metrics["inadmissibility_rate"]["ci95_half"],
                    metrics["survival_rate"]["ci95_half"],
                ))


        for model_name, metrics in summary.items():
            if "No-mask budget" not in model_name:
                continue
            if "survival_rate" not in metrics or "inadmissibility_rate" not in metrics:
                continue
            label = "d=" + model_name.replace("No-mask budget ", "")
            points.append((
                label,
                metrics["inadmissibility_rate"]["mean"],
                metrics["survival_rate"]["mean"],
                metrics["inadmissibility_rate"]["ci95_half"],
                metrics["survival_rate"]["ci95_half"],
            ))

        if not points:
            return None


        points.sort(key=lambda item: item[1])


        x = [p[1] for p in points]

        y = [p[2] for p in points]

        xerr = [p[3] for p in points]

        yerr = [p[4] for p in points]

        plt.figure(figsize=(8.8, 5.6), constrained_layout=True)

        plt.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color="#2171b5",
        )

        for label, px, py, _, _ in points:
            plt.annotate(label, (px, py), textcoords="offset points", xytext=(6, 5), fontsize=8)


        plt.xlabel("Inadmissible action rate")

        plt.ylabel("Survival / return")

        plt.title("Budgeted Lagrangian return-safety frontier")

        plt.grid(True, alpha=0.25)

        path = os.path.join(self.figures_dir, "budgeted_lagrangian_pareto_frontier.png")

        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        plt.close()

        return path

    def plot_mechanism_comparison(self, summary):

        preferred_order = [
            "No-mask no-Lagrangian",
            "LAADAN without action mask",
            "Masking-only actor-critic",
            "Full LAADAN-AC",
            "Post-hoc Masked VOAC",
        ]

        selected = [name for name in preferred_order if name in summary]

        if not selected:
            return None

        x = np.arange(len(selected))

        survival = [summary[name]["survival_rate"]["mean"] for name in selected]
        survival_ci = [summary[name]["survival_rate"]["ci95_half"] for name in selected]

        inad = [summary[name]["inadmissibility_rate"]["mean"] for name in selected]
        inad_ci = [summary[name]["inadmissibility_rate"]["ci95_half"] for name in selected]

        expert = [summary[name]["expert_argmax_match"]["mean"] for name in selected]
        expert_ci = [summary[name]["expert_argmax_match"]["ci95_half"] for name in selected]

        labels = [name.replace(" actor-critic", "\nactor-critic").replace("without action mask", "without\naction mask").replace("no-Lagrangian", "no\nLagrangian") for name in selected]

        width = 0.24

        plt.figure(figsize=(12.2, 6.2), constrained_layout=True)

        plt.bar(x - width, survival, width=width, yerr=survival_ci, capsize=4, label="Survival / return", color="#2171b5")

        plt.bar(x, inad, width=width, yerr=inad_ci, capsize=4, label="Inadmissibility", color="#9ecae1")

        plt.bar(x + width, expert, width=width, yerr=expert_ci, capsize=4, label="Expert argmax match", color="#08306b")

        plt.xticks(x, labels, rotation=0)

        plt.ylabel("Score")

        plt.ylim(0.0, 1.05)

        plt.title("Safety mechanism comparison")

        plt.grid(True, axis="y", alpha=0.25)

        plt.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))

        path = os.path.join(self.figures_dir, "safety_mechanism_comparison.png")

        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        plt.close()

        return path

    def plot_component_ablation_comparison(self, summary):

        selected = []
        for name in [
            "Full LAADAN-AC",
            "LAADAN without action mask",
            "LAADAN without conservative critic",
            "LAADAN without expert KL",
            "LAADAN without smoothness proxy",
            "LAADAN without Lagrangian cost control",
        ]:
            if name in summary:
                selected.append(name)

        if len(selected) < 3:
            return None

        y = np.arange(len(selected))

        survival = [summary[name]["survival_rate"]["mean"] for name in selected]
        survival_ci = [summary[name]["survival_rate"]["ci95_half"] for name in selected]

        inad = [summary[name]["inadmissibility_rate"]["mean"] for name in selected]
        inad_ci = [summary[name]["inadmissibility_rate"]["ci95_half"] for name in selected]

        labels = [
            name.replace("LAADAN ", "")
                .replace("without ", "No ")
                .replace(" cost control", "")
            for name in selected
        ]

        height = 0.35

        plt.figure(figsize=(10.8, 6.4), constrained_layout=True)

        plt.barh(y - height / 2, survival, height=height, xerr=survival_ci, capsize=4, label="Survival / return", color="#2171b5")

        plt.barh(y + height / 2, inad, height=height, xerr=inad_ci, capsize=4, label="Inadmissibility", color="#9ecae1")

        plt.yticks(y, labels)

        plt.xlabel("Score")

        plt.xlim(0.0, 1.05)

        plt.title("LAADAN-AC component ablation comparison")

        plt.grid(True, axis="x", alpha=0.25)

        plt.legend(frameon=False, loc="lower right")

        path = os.path.join(self.figures_dir, "laadan_component_ablation_comparison.png")

        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        plt.close()

        return path

    def plot_lagrange_dynamics(self, all_runs):

        candidate_name = "LAADAN without action mask"

        if candidate_name not in all_runs:
            return None

        best_run = None
        for run in all_runs[candidate_name]:
            if best_run is None:
                best_run = run
                continue
            if run["metrics"].get("survival_rate", -1.0) > best_run["metrics"].get("survival_rate", -1.0):
                best_run = run

        rows = best_run["history"]

        epochs = []
        lagrange = []
        expected_cost = []

        survival_epochs = []
        survival = []
        inad = []


        for row in rows:
            epochs.append(row["epoch"])
            lagrange.append(row.get("lagrange", np.nan))
            expected_cost.append(row.get("expected_immediate_cost", np.nan))
            if "survival_rate" in row:
                survival_epochs.append(row["epoch"])
                survival.append(row["survival_rate"])
                inad.append(row["inadmissibility_rate"])


        plt.figure(figsize=(10.5, 6.2), constrained_layout=True)

        plt.plot(epochs, lagrange, linewidth=2.0, label="Lagrange multiplier")

        plt.plot(epochs, expected_cost, linewidth=2.0, label="Expected immediate cost")

        plt.plot(survival_epochs, survival, linewidth=2.0, label="Survival / return")

        plt.plot(survival_epochs, inad, linewidth=2.0, label="Evaluated inadmissibility")

        plt.xlabel("Epoch")

        plt.ylabel("Value")

        plt.title("Adaptive Lagrangian dynamics without hard action masking")

        plt.grid(True, alpha=0.25)

        plt.legend(frameon=False, loc="best")

        path = os.path.join(self.figures_dir, "no_mask_lagrangian_training_dynamics.png")

        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        plt.close()

        return path

    def write_manifest(self, files):

        path = os.path.join(self.root, "lagrangian_manifest.json")

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(files, handle, indent=2)

        return path


def unpack_state_dict(loaded_object):

    if isinstance(loaded_object, dict):

        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in loaded_object and isinstance(loaded_object[key], dict):
                return loaded_object[key]

        raw_state = True
        for value in loaded_object.values():
            if not torch.is_tensor(value):
                raw_state = False
                break

        if raw_state:
            return loaded_object

    raise ValueError("Could not find model weights inside checkpoint.")


def clean_state_dict_keys(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            cleaned[key[7:]] = value

        else:
            cleaned[key] = value

    return cleaned


# evaluate posthoc masked voac
def evaluate_posthoc_masked_voac(benchmark, results_dir, config):

    model_path = os.path.join(results_dir, "final_models", "voac", "model.pt")

    if not os.path.exists(model_path):
        print("[SKIP] Post-hoc Masked VOAC was skipped because this file was not found:", model_path)
        return None

    model = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
        use_cost_head=False,
    ).to(benchmark.device)


    checkpoint = torch.load(model_path, map_location=benchmark.device)

    state_dict = clean_state_dict_keys(unpack_state_dict(checkpoint))

    model.load_state_dict(state_dict)

    model.eval()

    with torch.no_grad():
        logits = model(benchmark.state_features_t)["logits"]
        primary_policy = masked_greedy_policy_from_logits(logits, benchmark.admissible_mask_t)
        soft_policy = masked_softmax_from_logits(logits, benchmark.admissible_mask_t)

    metrics = evaluate_policy_set(
        benchmark,
        policy_numpy(primary_policy),
        soft_policy=policy_numpy(soft_policy),
    )

    simulation = benchmark.simulate_policy(
        policy_numpy(primary_policy),
        num_episodes=2000,
        seed=777,
    )

    metrics["trajectory_mean_action_jump"] = float(simulation["mean_action_jump"])

    metrics["trajectory_normalised_action_jump"] = float(
        simulation["mean_action_jump"] / max(1, benchmark.num_actions - 1)
    )

    metrics["variant_folder"] = "posthoc_masked_voac"
    metrics["variant_name"] = "Post-hoc Masked VOAC"
    metrics["uses_action_mask"] = True
    metrics["uses_lagrangian"] = False
    metrics["cost_budget"] = 0.0

    output_dir = os.path.join(results_dir, "lagrangian_frontier", "posthoc_masked_voac", "seed_final_voac")

    ensure_dir(output_dir)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    history_path = os.path.join(output_dir, "history.csv")

    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    return {
        "name": "Post-hoc Masked VOAC",
        "folder": "posthoc_masked_voac",
        "seed": "final_voac",
        "model": model,
        "policy": policy_numpy(primary_policy),
        "analysis_policy": policy_numpy(soft_policy),
        "history": [metrics],
        "metrics": metrics,
        "train_time_seconds": 0.0,
    }


# select device
def choose_device(requested):

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


# build variants
def build_variants(args):

    variants = [
        {
            "folder": "full_laadan_ac",
            "name": "Full LAADAN-AC",
            "use_action_mask": True,
            "use_conservative": True,
            "use_expert_kl": True,
            "use_smoothness": True,
            "use_lagrangian": True,
            "train_cost_head": True,
            "cost_budget": 0.0,
        },
        {
            "folder": "masking_only_actor_critic",
            "name": "Masking-only actor-critic",
            "use_action_mask": True,
            "use_conservative": False,
            "use_expert_kl": False,
            "use_smoothness": False,
            "use_lagrangian": False,
            "train_cost_head": False,
            "cost_budget": 0.0,
        },
        {
            "folder": "no_mask_lagrangian",
            "name": "LAADAN without action mask",
            "use_action_mask": False,
            "use_conservative": True,
            "use_expert_kl": True,
            "use_smoothness": True,
            "use_lagrangian": True,
            "train_cost_head": True,
            "cost_budget": 0.0,
        },
        {
            "folder": "no_mask_no_lagrangian",
            "name": "No-mask no-Lagrangian",
            "use_action_mask": False,
            "use_conservative": True,
            "use_expert_kl": True,
            "use_smoothness": True,
            "use_lagrangian": False,
            "train_cost_head": True,
            "cost_budget": 0.0,
        },
    ]

    if not args.skip_budget_sweep:
        for budget in [0.005, 0.01, 0.02, 0.05]:

            safe_name = str(budget).replace(".", "_")

            variants.append({
                "folder": "budget_" + safe_name,
                "name": "No-mask budget " + str(budget),
                "use_action_mask": False,
                "use_conservative": True,
                "use_expert_kl": True,
                "use_smoothness": True,
                "use_lagrangian": True,
                "train_cost_head": True,
                "cost_budget": budget,
            })

    if args.include_leave_one_out:
        variants += [
            {
                "folder": "no_lagrangian",
                "name": "LAADAN without Lagrangian cost control",
                "use_action_mask": True,
                "use_conservative": True,
                "use_expert_kl": True,
                "use_smoothness": True,
                "use_lagrangian": False,
                "train_cost_head": True,
                "cost_budget": 0.0,
            },
            {
                "folder": "no_conservative",
                "name": "LAADAN without conservative critic",
                "use_action_mask": True,
                "use_conservative": False,
                "use_expert_kl": True,
                "use_smoothness": True,
                "use_lagrangian": True,
                "train_cost_head": True,
                "cost_budget": 0.0,
            },
            {
                "folder": "no_expert_kl",
                "name": "LAADAN without expert KL",
                "use_action_mask": True,
                "use_conservative": True,
                "use_expert_kl": False,
                "use_smoothness": True,
                "use_lagrangian": True,
                "train_cost_head": True,
                "cost_budget": 0.0,
            },
            {
                "folder": "no_smoothness",
                "name": "LAADAN without smoothness proxy",
                "use_action_mask": True,
                "use_conservative": True,
                "use_expert_kl": True,
                "use_smoothness": False,
                "use_lagrangian": True,
                "train_cost_head": True,
                "cost_budget": 0.0,
            },
        ]

    return variants


# arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Budgeted Lagrangian LAADAN-AC experiments.")

    parser.add_argument("--data-dir", required=True, help="Path to ICU-Sepsis benchmark files.")

    parser.add_argument("--results-dir", default="results", help="Output directory.")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--mode", choices=["debug", "main"], default="main")

    parser.add_argument("--skip-budget-sweep", action="store_true")

    parser.add_argument("--include-leave-one-out", action="store_true")

    parser.add_argument("--skip-masked-voac", action="store_true")

    parser.add_argument("--num-trajectory-episodes", type=int, default=2000)

    return parser.parse_args()


def main():

    args = parse_args()

    device = choose_device(args.device)

    config = dict(BASE_LAADAN_CONFIG)

    seeds = [42, 43, 44, 45, 46]

    if args.mode == "debug":
        seeds = [42]
        config["epochs"] = 40
        config["eval_every"] = 5

    ensure_dir(args.results_dir)


    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=50,
        device=device,
        use_one_hot_states=False,
    )


    benchmark.save_benchmark_description(
        os.path.join(args.results_dir, "lagrangian_benchmark_description.json")
    )

    variants = build_variants(args)

    for variant in variants:
        variant["num_trajectory_episodes"] = int(args.num_trajectory_episodes)

    experiment = BudgetedLagrangianExperiment(benchmark, args.results_dir, config)

    all_runs = experiment.run_variants(variants, seeds)

    if not args.skip_masked_voac:
        masked_voac_run = evaluate_posthoc_masked_voac(benchmark, args.results_dir, config)
        if masked_voac_run is not None:
            all_runs["Post-hoc Masked VOAC"] = [masked_voac_run]

    reporter = FrontierReporter(benchmark, args.results_dir)

    summary = reporter.aggregate_runs(all_runs)

    files = {}

    files["summary_csv"] = reporter.write_summary_csv(summary)

    files["summary_json"] = reporter.save_summary_json(summary)

    files["final_models_csv"], files["final_models_json"] = reporter.select_final_models(all_runs)

    files["pareto_frontier_png"] = reporter.plot_pareto_frontier(summary)

    files["mechanism_comparison_png"] = reporter.plot_mechanism_comparison(summary)

    files["component_ablation_png"] = reporter.plot_component_ablation_comparison(summary)

    files["lagrange_dynamics_png"] = reporter.plot_lagrange_dynamics(all_runs)

    files["manifest_json"] = reporter.write_manifest(files)

    print("[DONE] Lagrangian outputs saved to:", reporter.root)

    print("[DONE] Final selected models saved to:", reporter.final_dir)

    for key in sorted(files.keys()):
        print("[FILE]", key + ":", files[key])


if __name__ == "__main__":
    main()
