#!/usr/bin/env python3
"""Shared LAADAN-AC ablation engine used by the fixed-schedule experiments."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

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

                # Computing the next-state value that combines Q-value, entropy
                # and Lagrangian cost pressure.
                next_v = torch.sum(
                    next_probs * (min_q_next - entropy_coef * next_log_probs - lagrange_value * cost_next),
                    dim=1,
                ) * (1.0 - terminal)

                next_c = torch.sum(next_probs * cost_next, dim=1) * (1.0 - terminal)

                # Building the Bellman target for reward/value Q-learning.
                q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)

                # Building the Bellman target for cost prediction.
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

            # If conservative regularisation is enabled, penalise Q-values that
            # inflate unsupported or non-expert actions.
            if use_conservative:
                cql_q1 = cql_regularizer(q1, expert, admissible_mask=conservative_mask)
                cql_q2 = cql_regularizer(q2, expert, admissible_mask=conservative_mask)
                critic_loss = critic_loss + conservative_alpha * (cql_q1 + cql_q2)

            # Otherwise, use zero placeholders for logging.
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
