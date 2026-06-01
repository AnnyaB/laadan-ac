
# lagrangian_frontier.py

# This script runs the LAADAN-AC Lagrangian frontier experiments.
# This script is intended to be self-contained and reproducible

# Libraries used in this script:

# argparse is used to read command-line arguments such as --data-dir,
# --results-dir, --device, --mode and the experiment flags.
import argparse

# csv is used to write editable result tables, especially the compact
# summary table and the final model selection table.
import csv

#  used to save structured summaries, metrics, manifests, and benchmark
# descriptions in a machine-readable format.
import json

# os is used to build file paths, check whether files/folders exist, and create
# output paths in a platform-independent way.
import os

# shutil is used to copy selected model files into a clean final-model directory.
import shutil

# time is used to measure how long each seed run takes.
import time

# matplotlib is used to create and save plots.
import matplotlib.pyplot as plt

# numpy is used for numeric arrays, metric aggregation, finite-value filtering,
# confidence-interval handling and plotting preparation.
import numpy as np

# torch is used for PyTorch model training, tensors, optimisers, gradients and
# checkpoint loading.
import torch

# Importing the ICU-Sepsis benchmark loader/evaluator used throughout the project.
from benchmark import ICUSepsisOfflineBenchmark

# Importing the shared actor-critic architecture and target-network update helper.
from models import OfflineActorCriticNet, soft_update

# Importing helper functions from trainers.py 
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


# Saving all figures at 600 dpi
PLOT_DPI = 600


# Fixed colour palette used across the generated plots.

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


def colour_for(name):
    """
    Returning a consistent colour for a model name.
    """

    # If the model name is explicitly included in the palette, use that colour.
    if name in PLOT_PALETTE:
        return PLOT_PALETTE[name]

    # If the model is one of the budget-sweep models, use the main LAADAN colour.
    if "budget" in name.lower():
        return "#2171b5"

    # Otherwise, return a safe default blue.
    return "#3182bd"


# Base training configuration for the LAADAN-AC variants.
# These values are copied into a config dictionary inside main(), so debug mode
# can modify them without changing the original constant.
BASE_LAADAN_CONFIG = {
    # Number of training epochs for main-mode variants.
    "epochs": 1000,

    # Actor learning rate.
    "actor_lr": 5e-4,

    # Critic learning rate.
    "critic_lr": 1e-3,

    # Dropout used inside the shared actor-critic network encoders.
    "dropout": 0.10,

    # Hidden layer width for the MLP encoders.
    "hidden_dim": 128,

    # Latent representation size produced by the encoders.
    "latent_dim": 128,

    # Evaluation interval. The script evaluates every 10 epochs.
    "eval_every": 10,

    # Discount factor used in the finite-horizon benchmark setting.
    "gamma": 1.0,

    # Target-network soft-update coefficient.
    "tau": 0.01,

    # Entropy coefficient used to avoid an overly collapsed actor during training.
    "entropy_coef": 0.001,

    # Strength of the conservative critic regulariser.
    "conservative_alpha": 0.5,

    # Strength of the expert-safe KL regularisation term.
    "expert_kl_weight": 0.005,

    # Strength of the state-action smoothness proxy term.
    "smoothness_weight": 0.001,

    # Default cost budget for the Lagrangian constraint.
    "cost_budget": 0.0,

    # Learning rate for the Lagrange multiplier update.
    "lagrange_lr": 0.0002,

    # Initial Lagrange multiplier value.
    "lagrange_init": 0.0,

    # Adam weight decay used for actor and critic optimisers.
    "weight_decay": 1e-5,
}


class BudgetedLagrangianExperiment:
    """
    Runs LAADAN-AC variants using the same ICU-Sepsis MDP.
    
    """

    def __init__(self, benchmark, results_dir, config):
        """
        Store the benchmark, output directory and base training configuration.
        """

        # Saving the loaded ICU-Sepsis benchmark object so all training/evaluation
        # functions can access its tensors, masks, rewards and transition matrix.
        self.benchmark = benchmark

        # Saving the root results directory where all outputs will go.
        self.results_dir = results_dir

        # Copying the configuration dictionary so this class owns its own version.
        self.config = dict(config)

    def build_model_pair(self, use_cost_head):
        """
        Build the main model and target model.

        The architecture is the existing OfflineActorCriticNet:
        state features -> actor encoder/policy head
        state features -> critic encoder -> Q1, Q2, optional cost head
        """

        # Creating the trainable actor-critic model for the current variant.
        model = OfflineActorCriticNet(
            self.benchmark.feature_dim,
            self.benchmark.num_actions,
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            latent_dim=int(self.config.get("latent_dim", 128)),
            dropout=float(self.config.get("dropout", 0.10)),
            use_cost_head=bool(use_cost_head),
        ).to(self.benchmark.device)

        # Creating a target model with the same architecture.
        # The target model is used to make Bellman targets more stable.
        target_model = OfflineActorCriticNet(
            self.benchmark.feature_dim,
            self.benchmark.num_actions,
            hidden_dim=int(self.config.get("hidden_dim", 128)),
            latent_dim=int(self.config.get("latent_dim", 128)),
            dropout=float(self.config.get("dropout", 0.10)),
            use_cost_head=bool(use_cost_head),
        ).to(self.benchmark.device)

        # Copying the initial parameters from the main model into the target model.
        target_model.load_state_dict(model.state_dict())

        # Putting the target model in evaluation mode because it is not trained
        # directly by backpropagation.
        target_model.eval()

        # Returning both the trainable model and target model.
        return model, target_model

    def policy_probs(self, logits, admissible, use_action_mask):
        """
        Convert logits to probabilities.

        If use_action_mask is True, the policy can only place probability on
        benchmark-admissible actions. If False, the policy is free to use all
        25 actions, so any safety must be learned through the objective.
        """

        # If this variant uses hard action masking, apply masked softmax.
        if use_action_mask:
            return masked_softmax_from_logits(logits, admissible)

        # Otherwise, apply a normal softmax over all actions.
        return plain_softmax_from_logits(logits)

    def policy_log_probs(self, logits, admissible, use_action_mask):
        """
        Convert logits to log-probabilities using the same masking choice as the policy.
        """

        # If this variant uses hard action masking, apply masked log-softmax.
        if use_action_mask:
            return masked_log_softmax_from_logits(logits, admissible)

        # Otherwise, apply normal log-softmax over all actions.
        return plain_log_softmax_from_logits(logits)

    def greedy_policy(self, logits, admissible, use_action_mask):
        """
        Build the deterministic primary policy used for exact benchmark evaluation.
        """

        # If this variant uses hard action masking, choose the best admissible action.
        if use_action_mask:
            return masked_greedy_policy_from_logits(logits, admissible)

        # Otherwise, choose the best action from all 25 action bins.
        return greedy_policy_from_logits(logits)

    def train_variant(self, seed, variant):
        """
        Train one variant for one random seed.

        Variant flags:
        - use_action_mask: hard admissibility mask during learning and evaluation
        - use_conservative: include CQL-style critic regularisation
        - use_expert_kl: include expert-safe KL regularisation
        - use_smoothness: include state-action smoothness proxy
        - use_lagrangian: include adaptive Lagrangian cost pressure
        - train_cost_head: train the cost critic even if the actor does not use lambda
        """

        # Setting the benchmark seed so this seed run is reproducible.
        self.benchmark.set_seed(seed)

        # Reading whether this variant uses hard action masking.
        use_action_mask = bool(variant.get("use_action_mask", True))

        # Reading whether this variant uses conservative critic regularisation.
        use_conservative = bool(variant.get("use_conservative", True))

        # Reading whether this variant uses expert-safe KL regularisation.
        use_expert_kl = bool(variant.get("use_expert_kl", True))

        # Reading whether this variant uses the smoothness proxy term.
        use_smoothness = bool(variant.get("use_smoothness", True))

        # Reading whether this variant uses adaptive Lagrangian cost pressure.
        use_lagrangian = bool(variant.get("use_lagrangian", True))

        # Reading whether this variant trains a cost head.
        # By default, a cost head is used if Lagrangian control is used.
        train_cost_head = bool(variant.get("train_cost_head", use_lagrangian))

        # Building the trainable model and the target model for this variant.
        model, target_model = self.build_model_pair(use_cost_head=train_cost_head)

        # Collecting actor parameters, which include the actor encoder and policy head.
        actor_params = list(model.actor_encoder.parameters()) + list(model.actor_head.parameters())

        # Collecting critic parameters, which include the critic encoder and two Q heads.
        critic_params = (
            list(model.critic_encoder.parameters())
            + list(model.q1_head.parameters())
            + list(model.q2_head.parameters())
        )

        # If this variant has a cost head, include the cost-head parameters in
        # the critic optimiser because the cost head is value-like.
        if train_cost_head:
            critic_params += list(model.cost_head.parameters())

        # Creating the Adam optimiser for the actor branch.
        actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=float(self.config.get("actor_lr", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
        )

        # Creating the Adam optimiser for the critic branch and optional cost head.
        critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=float(self.config.get("critic_lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
        )

        # Reading scalar hyperparameters from the base config or variant override.
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

        # Loading the full benchmark state-feature matrix.
        x = self.benchmark.state_features_t

        # Loading the benchmark admissibility mask M(s, a).
        admissible = self.benchmark.admissible_mask_t

        # Loading the expert-safe policy distribution used for KL regularisation.
        expert = self.benchmark.expert_safe_t

        # Loading the state-action reward table.
        reward_sa = self.benchmark.reward_sa_t

        # Loading the transition tensor P(s' | s, a).
        transition = self.benchmark.transition_t

        # Loading the terminal-state mask.
        terminal = self.benchmark.terminal_mask_t

        # Loading the immediate inadmissibility cost table.
        immediate_cost = self.benchmark.immediate_cost_t

        # Loading the state-action smoothness proxy cost table.
        smoothness_cost = self.benchmark.smoothness_cost_t

    
        critic_fit_mask = admissible if use_action_mask else None

        conservative_mask = admissible if use_action_mask else None

        # Creating a list that will store one history row per epoch.
        history = []

        # Initialising the best survival score seen during training.
        best_survival = -1.0

        # best_state will store the model weights for the best evaluated epoch.
        best_state = None

        # best_metrics will store the metrics for the best evaluated epoch.
        best_metrics = None

        # Recording the start time for this seed run.
        start_time = time.time()

        # Starting the main training loop.
        for epoch in range(1, epochs + 1):

            # Putting the model in training mode so dropout is active.
            model.train()

            # Computing Bellman targets without tracking gradients.
            with torch.no_grad():

                # Running the target model on all benchmark states.
                target_outputs = target_model(x)

                # Converting target actor logits into action probabilities.
                next_probs = self.policy_probs(target_outputs["logits"], admissible, use_action_mask)

                # Converting target actor logits into log-probabilities.
                next_log_probs = self.policy_log_probs(target_outputs["logits"], admissible, use_action_mask)

                # Reading Q1 values from the target model.
                q1_next = target_outputs["q1"]

                # Reading Q2 values from the target model.
                q2_next = target_outputs["q2"]

                # Taking the smaller of Q1 and Q2 to reduce over-optimistic value estimates.
                min_q_next = torch.min(q1_next, q2_next)

                # If a cost head is trained, use its non-negative cost estimates.
                if train_cost_head:
                    cost_next = torch.relu(target_outputs["cost"])

                # Otherwise, use the fixed immediate benchmark cost table.
                else:
                    cost_next = immediate_cost

                # Computing the next-state value that combines Q-value, entropy
                # and Lagrangian cost pressure.
                next_v = torch.sum(
                    next_probs * (min_q_next - entropy_coef * next_log_probs - lagrange_value * cost_next),
                    dim=1,
                ) * (1.0 - terminal)

                # Computing the next-state expected cost-to-go.
                next_c = torch.sum(next_probs * cost_next, dim=1) * (1.0 - terminal)

                # Building the Bellman target for reward/value Q-learning.
                q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)

                # Building the Bellman target for cost prediction.
                c_target = immediate_cost + gamma * torch.einsum("san,n->sa", transition, next_c)

            # Clearing old critic gradients.
            critic_optimizer.zero_grad()

            # Running the main model on all benchmark states.
            outputs = model(x)

            # Reading the first Q-value table.
            q1 = outputs["q1"]

            # Reading the second Q-value table.
            q2 = outputs["q2"]

            # Computing the Q1 mean-squared Bellman error.
            q1_loss = masked_action_mse(q1, q_target, action_mask=critic_fit_mask)

            # Computing the Q2 mean-squared Bellman error.
            q2_loss = masked_action_mse(q2, q_target, action_mask=critic_fit_mask)

            # Combining both critic losses.
            critic_loss = q1_loss + q2_loss

            # If this variant trains a cost head, fit it to the cost Bellman target.
            if train_cost_head:
                cost_pred = torch.relu(outputs["cost"])
                cost_loss = masked_action_mse(cost_pred, c_target, action_mask=critic_fit_mask)
                critic_loss = critic_loss + cost_loss

            # Otherwise, use a zero placeholder for logging.
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

            # Backpropagating the critic loss.
            critic_loss.backward()

            # Clipping critic gradients to avoid unstable updates.
            torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)

            # Applying one critic optimiser step.
            critic_optimizer.step()

            # Clearing old actor gradients.
            actor_optimizer.zero_grad()

            # Running the model again after the critic update.
            outputs = model(x)

            # Reading actor logits.
            logits = outputs["logits"]

            # Converting actor logits into policy probabilities.
            probs = self.policy_probs(logits, admissible, use_action_mask)

            # Converting actor logits into log-probabilities.
            log_probs = self.policy_log_probs(logits, admissible, use_action_mask)

            # Reading current Q1 values.
            q1 = outputs["q1"]

            # Reading current Q2 values.
            q2 = outputs["q2"]

            # Taking the minimum Q estimate for conservative actor learning.
            min_q = torch.min(q1, q2)

            # If the cost head exists, use its non-negative predictions.
            if train_cost_head:
                cost_pred = torch.relu(outputs["cost"])

            # Otherwise, use the fixed immediate cost table.
            else:
                cost_pred = immediate_cost

            # Computing the expected Q-value under the current policy.
            expected_q = torch.sum(probs * min_q, dim=1)

            # Computing the expected learned cost-to-go under the current policy.
            expected_cost_to_go = torch.sum(probs * cost_pred, dim=1)

            # Computing the expected immediate benchmark-defined inadmissibility cost.
            expected_immediate_cost = torch.sum(probs * immediate_cost, dim=1)

            # Computing the entropy of the current policy.
            entropy = -torch.sum(probs * log_probs, dim=1)

            # Starting the actor loss by maximising Q and encouraging some entropy.
            actor_loss = torch.mean(-expected_q - entropy_coef * entropy)

            # If Lagrangian control is active, penalise expected cost-to-go.
            if use_lagrangian:
                actor_loss = actor_loss + lagrange_value * torch.mean(expected_cost_to_go)

            # If expert KL is active, keep the learned policy close to the expert-safe policy.
            if use_expert_kl:
                expert_kl = torch.sum(expert * (torch.log(expert + 1e-8) - log_probs), dim=1)
                actor_loss = actor_loss + expert_kl_weight * torch.mean(expert_kl)

            # Otherwise, use a zero tensor for logging.
            else:
                expert_kl = torch.zeros_like(expected_q)

            # If smoothness is active, add the state-action smoothness proxy penalty.
            if use_smoothness:
                expected_smoothness = torch.sum(probs * smoothness_cost, dim=1)
                actor_loss = actor_loss + smoothness_weight * torch.mean(expected_smoothness)

            # Otherwise, use a zero tensor for logging.
            else:
                expected_smoothness = torch.zeros_like(expected_q)

            # Backpropagating the actor loss.
            actor_loss.backward()

            # Clipping actor gradients to avoid unstable updates.
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)

            # Applying one actor optimiser step.
            actor_optimizer.step()

            # Soft-updating the target model toward the current model.
            soft_update(target_model, model, tau)

            # Returning the target model to evaluation mode after the update.
            target_model.eval()

            # Converting expected immediate cost to a Python float for logging.
            mean_immediate_cost = float(torch.mean(expected_immediate_cost).item())

            # Converting expected cost-to-go to a Python float for logging.
            mean_cost_to_go = float(torch.mean(expected_cost_to_go).item())

            # Updating the Lagrange multiplier if this variant uses Lagrangian control.
            # If expected cost is above the allowed budget, lambda increases.
            if use_lagrangian:
                lagrange_value = max(0.0, lagrange_value + lagrange_lr * (mean_immediate_cost - cost_budget))

            # Creating the per-epoch logging row.
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

            # Evaluating the current policy every eval_every epochs and at the final epoch.
            if epoch % eval_every == 0 or epoch == epochs:

                # Putting the model in evaluation mode.
                model.eval()

                # Building deterministic and soft policies without gradients.
                with torch.no_grad():
                    logits_eval = model(x)["logits"]
                    primary_policy = self.greedy_policy(logits_eval, admissible, use_action_mask)
                    soft_policy = self.policy_probs(logits_eval, admissible, use_action_mask)

                # Evaluating the deterministic and soft policy using benchmark metrics.
                metrics = evaluate_policy_set(
                    self.benchmark,
                    policy_numpy(primary_policy),
                    soft_policy=policy_numpy(soft_policy),
                )

                # Sampling trajectories to compute trajectory-level action-change behaviour.
                simulation = self.benchmark.simulate_policy(
                    policy_numpy(primary_policy),
                    num_episodes=int(variant.get("num_trajectory_episodes", 2000)),
                    seed=123 + int(seed),
                )

                # Saving raw mean action jump.
                metrics["trajectory_mean_action_jump"] = float(simulation["mean_action_jump"])

                # Saving normalised mean action jump by dividing by the largest possible
                # action-bin distance.
                metrics["trajectory_normalised_action_jump"] = float(
                    simulation["mean_action_jump"] / max(1, self.benchmark.num_actions - 1)
                )

                # Adding evaluation metrics into this epoch's history row.
                row.update(metrics)

                # If this evaluated policy has the best survival so far, store it.
                if metrics["survival_rate"] > best_survival:
                    best_survival = metrics["survival_rate"]
                    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    best_metrics = dict(metrics)

            # Appending the epoch row to the history list.
            history.append(row)

        # Restoring the model weights from the best survival checkpoint, if available.
        if best_state is not None:
            model.load_state_dict(best_state)

        # Putting the final restored model in evaluation mode.
        model.eval()

        # Building final deterministic and soft policies from the restored model.
        with torch.no_grad():
            logits_final = model(x)["logits"]
            final_policy = self.greedy_policy(logits_final, admissible, use_action_mask)
            final_soft_policy = self.policy_probs(logits_final, admissible, use_action_mask)

        # If no evaluation metrics were collected, evaluate the final policy now.
        if best_metrics is None:
            best_metrics = evaluate_policy_set(
                self.benchmark,
                policy_numpy(final_policy),
                soft_policy=policy_numpy(final_soft_policy),
            )

        # Adding variant metadata to the final metrics file.
        best_metrics["variant_folder"] = variant["folder"]
        best_metrics["variant_name"] = variant["name"]
        best_metrics["uses_action_mask"] = use_action_mask
        best_metrics["uses_lagrangian"] = use_lagrangian
        best_metrics["cost_budget"] = float(cost_budget)

        # Building the seed-specific output directory.
        seed_dir = os.path.join(self.results_dir, "lagrangian_frontier", variant["folder"], "seed_" + str(seed))

        # Saving model.pt, history.csv and metrics.json for this seed run.
        save_model_run(seed_dir, model, history, best_metrics)

        # Returning the trained run information so the reporter can aggregate it.
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
        """
        Train every requested variant across all requested random seeds.
        """

        # Creating a dictionary that will store all runs grouped by model name.
        all_runs = {}

        # Looping through every requested variant.
        for variant in variants:

            # Creating a list that will hold all seed runs for this variant.
            runs = []

            # Running the current variant for each requested random seed.
            for seed in seeds:

                # Printing progress so Kaggle/terminal output shows which run is active.
                print("[TRAIN]", variant["name"], "seed", seed)

                # Training one seed and appending the result.
                runs.append(self.train_variant(seed, variant))

            # Storing all seed runs for this variant under its readable name.
            all_runs[variant["name"]] = runs

        # Returning all trained runs.
        return all_runs


class FrontierReporter:
    """
    Converts runs into CSV, JSON, final checkpoints and figures.
    """

    def __init__(self, benchmark, results_dir):
        """
        Store benchmark reference and create output directories.
        """

        # Storing the benchmark object for any figure/table function that may need it.
        self.benchmark = benchmark

        # Building the root folder for the Lagrangian outputs.
        self.root = os.path.join(results_dir, "lagrangian_frontier")

        # Building the figures folder inside the Lagrangian output folder.
        self.figures_dir = os.path.join(self.root, "figures")

        # Building the final selected model folder for  variants.
        self.final_dir = os.path.join(results_dir, "final_lagrangian_models")

        # Creating the root folder if needed.
        ensure_dir(self.root)

        # Creating the figures folder if needed.
        ensure_dir(self.figures_dir)

        # Creating the final selected model folder if needed.
        ensure_dir(self.final_dir)

    def aggregate_runs(self, all_runs):
        """
        Aggregate each metric across seeds using mean and 95% confidence intervals.
        """

        # Creating the dictionary that will store aggregated metrics for every model.
        summary = {}

        # Looping through each model name and its list of seed runs.
        for model_name, runs in all_runs.items():

            # Creating a dictionary for this model's aggregated metrics.
            metrics = {}

            # Creating a list of numeric metric names found in the run metrics.
            metric_names = []

            # Inspecting every run to collect all numeric metric keys.
            for run in runs:
                for key in run["metrics"].keys():
                    if key not in metric_names and isinstance(run["metrics"][key], (int, float, bool)):
                        metric_names.append(key)

            # Aggregating each numeric metric across seeds.
            for metric_name in metric_names:

                # Creating a list that will store this metric's value for each seed.
                values = []

                # Reading the metric value from each run.
                for run in runs:
                    value = run["metrics"].get(metric_name, np.nan)

                    # Converting boolean metadata into numeric values if needed.
                    if isinstance(value, bool):
                        value = float(value)

                    # Appending the value for aggregation.
                    values.append(value)

                # Converting the collected values into a NumPy array.
                values = np.asarray(values, dtype=float)

                # Keeping only finite numeric values.
                valid = values[np.isfinite(values)]

                # If at least one valid value exists, compute mean and confidence interval.
                if valid.size > 0:
                    stats = mean_ci95(valid)
                    stats["min"] = float(np.min(valid))
                    stats["max"] = float(np.max(valid))
                    metrics[metric_name] = stats

            # Saving the aggregated metrics for this model.
            summary[model_name] = metrics

        # Returning the full multi-model summary.
        return summary

    def write_summary_csv(self, summary):
        """
        Save a compact table as editable CSV.
        """

        # Building the CSV output path.
        path = os.path.join(self.root, "budgeted_lagrangian_summary.csv")

        # Defining the exact column order for the summary table.
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

        # Opening the CSV file for writing.
        with open(path, "w", newline="", encoding="utf-8") as handle:

            # Creating a CSV writer with the fixed column order.
            writer = csv.DictWriter(handle, fieldnames=columns)

            # Writing the header row.
            writer.writeheader()

            # Writing one row for each model in the summary.
            for model_name, metrics in summary.items():

                # Starting the row with the readable model name.
                row = {"model": model_name}

                # Adding the number of seeds/runs if survival was reported.
                if "survival_rate" in metrics:
                    row["n"] = metrics["survival_rate"]["n"]

                # Filling the mean and CI columns for each metric.
                for key in columns:
                    if key in ["model", "n"]:
                        continue

                    # Recovering the metric name from the column name.
                    metric_name = key.replace("_mean", "").replace("_ci95_half", "")

                    # If this is a mean column, write the metric mean.
                    if key.endswith("_mean") and metric_name in metrics:
                        row[key] = metrics[metric_name]["mean"]

                    # If this is a CI column, write the 95% CI half-width.
                    if key.endswith("_ci95_half") and metric_name in metrics:
                        row[key] = metrics[metric_name]["ci95_half"]

                # Writing this model's row to the CSV file.
                writer.writerow(row)

        # Returning the path of the written CSV file.
        return path

    def save_summary_json(self, summary):
        """
        Save the full statistical summary as JSON.
        """

        # Building the JSON output path.
        path = os.path.join(self.root, "budgeted_lagrangian_summary.json")

        # Writing the complete summary dictionary to JSON.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        # Returning the path of the written JSON file.
        return path

    def copy_if_exists(self, source, destination):
        """
        Copy one file only if it exists.
        """

        # Copying the file only when the source path exists.
        if os.path.exists(source):
            shutil.copy2(source, destination)

    def select_final_models(self, all_runs):
        """
        Pick one final checkpoint per variant.

        """

        # Creating a list that will store final model selection rows.
        rows = []

        # Looping through each model variant and its seed runs.
        for model_name, runs in all_runs.items():

            # Initialising the best run for this model.
            best_run = None

            # Searching through every seed run for this model.
            for run in runs:

                # If this is the first run, use it as the current best.
                if best_run is None:
                    best_run = run
                    continue

                # Reading this run's survival and inadmissibility.
                survival = float(run["metrics"].get("survival_rate", -1e9))
                inad = float(run["metrics"].get("inadmissibility_rate", 1e9))

                # Reading the current best run's survival and inadmissibility.
                best_survival = float(best_run["metrics"].get("survival_rate", -1e9))
                best_inad = float(best_run["metrics"].get("inadmissibility_rate", 1e9))

                # Prefer the run with higher survival.
                # If survival is exactly tied, prefer lower inadmissibility.
                if survival > best_survival or (survival == best_survival and inad < best_inad):
                    best_run = run

            # Reading the selected run's folder name.
            folder = best_run["folder"]

            # Building the seed folder name, for example seed_42.
            seed_folder = "seed_" + str(best_run["seed"])

            # Building the source directory for the selected seed run.
            src_dir = os.path.join(self.root, folder, seed_folder)

            # Building the destination directory for the selected final model.
            dst_dir = os.path.join(self.final_dir, folder)

            # Creating the destination directory if needed.
            ensure_dir(dst_dir)

            # Copying the model checkpoint, metrics and history if they exist.
            for filename in ["model.pt", "metrics.json", "history.csv"]:
                self.copy_if_exists(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))

            # Adding one row to the final model selection table.
            rows.append({
                "folder": folder,
                "model": model_name,
                "best_seed": seed_folder,
                "survival_rate": best_run["metrics"].get("survival_rate", ""),
                "inadmissibility_rate": best_run["metrics"].get("inadmissibility_rate", ""),
                "expert_argmax_match": best_run["metrics"].get("expert_argmax_match", ""),
                "cost_budget": best_run["metrics"].get("cost_budget", ""),
            })

        # Building the final model selection CSV path.
        csv_path = os.path.join(self.final_dir, "final_lagrangian_model_selection.csv")

        # Writing the final model selection CSV file.
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

        # Building the final model selection JSON path.
        json_path = os.path.join(self.final_dir, "final_lagrangian_model_selection.json")

        # Writing the same final selection data as JSON.
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)

        # Returning both final model selection file paths.
        return csv_path, json_path

    def plot_pareto_frontier(self, summary):
        """
        Plot survival against inadmissibility for the budgeted no-mask variants.

        The d=0 point is taken from LAADAN without action mask. The remaining
        points come from the no-mask budget sweep. This avoids retraining the
        exact same zero-budget variant twice and keeps the figure easier to read.
        """

        # Creating a list that will store each budget point for the plot.
        points = []

        # Adding the zero-budget no-mask Lagrangian point from the ablation model.
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

        # Adding each non-zero budget-sweep point.
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

        # If no budget points exist, skip the figure.
        if not points:
            return None

        # Sorting points from lowest to highest inadmissibility.
        points.sort(key=lambda item: item[1])

        # Extracting x values, which are inadmissibility means.
        x = [p[1] for p in points]

        # Extracting y values, which are survival/return means.
        y = [p[2] for p in points]

        # Extracting horizontal error bars, which are inadmissibility CI half-widths.
        xerr = [p[3] for p in points]

        # Extracting vertical error bars, which are survival CI half-widths.
        yerr = [p[4] for p in points]

        # Creating the figure.
        plt.figure(figsize=(8.8, 5.6), constrained_layout=True)

        # Plotting mean 95% CI for each budget point.
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

        # Adding text labels near each point.
        for label, px, py, _, _ in points:
            plt.annotate(label, (px, py), textcoords="offset points", xytext=(6, 5), fontsize=8)

        # Labelling the x-axis.
        plt.xlabel("Inadmissible action rate")

        # Labelling the y-axis.
        plt.ylabel("Survival / return")

        # Adding the figure title.
        plt.title("Budgeted Lagrangian return-safety frontier")

        # Adding light grid lines for readability.
        plt.grid(True, alpha=0.25)

        # Building the output path for the Pareto frontier figure.
        path = os.path.join(self.figures_dir, "budgeted_lagrangian_pareto_frontier.png")

        # Saving the figure at 600 dpi.
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        # Closing the figure to avoid overlap with later plots.
        plt.close()

        # Returning the saved figure path.
        return path

    def plot_mechanism_comparison(self, summary):
        """
        Plot the main safety mechanism comparison.
        """

        # Defining the order of mechanisms in the plot.
        preferred_order = [
            "No-mask no-Lagrangian",
            "LAADAN without action mask",
            "Masking-only actor-critic",
            "Full LAADAN-AC",
            "Post-hoc Masked VOAC",
        ]

        # Keeping only models that actually exist in the summary.
        selected = [name for name in preferred_order if name in summary]

        # If none of the selected models exist, skip this figure.
        if not selected:
            return None

        # Creating x-axis group positions.
        x = np.arange(len(selected))

        # Reading survival means and CIs.
        survival = [summary[name]["survival_rate"]["mean"] for name in selected]
        survival_ci = [summary[name]["survival_rate"]["ci95_half"] for name in selected]

        # Reading inadmissibility means and CIs.
        inad = [summary[name]["inadmissibility_rate"]["mean"] for name in selected]
        inad_ci = [summary[name]["inadmissibility_rate"]["ci95_half"] for name in selected]

        # Reading expert argmax match means and CIs.
        expert = [summary[name]["expert_argmax_match"]["mean"] for name in selected]
        expert_ci = [summary[name]["expert_argmax_match"]["ci95_half"] for name in selected]

        # Shortening long x-axis labels so they fit inside the figure.
        labels = [name.replace(" actor-critic", "\nactor-critic").replace("without action mask", "without\naction mask").replace("no-Lagrangian", "no\nLagrangian") for name in selected]

        # Setting the bar width inside each group.
        width = 0.24

        # Creating the grouped bar chart figure.
        plt.figure(figsize=(12.2, 6.2), constrained_layout=True)

        # Plotting survival bars.
        plt.bar(x - width, survival, width=width, yerr=survival_ci, capsize=4, label="Survival / return", color="#2171b5")

        # Plotting inadmissibility bars.
        plt.bar(x, inad, width=width, yerr=inad_ci, capsize=4, label="Inadmissibility", color="#9ecae1")

        # Plotting expert argmax match bars.
        plt.bar(x + width, expert, width=width, yerr=expert_ci, capsize=4, label="Expert argmax match", color="#08306b")

        # Applying x-axis labels.
        plt.xticks(x, labels, rotation=0)

        # Labelling the y-axis.
        plt.ylabel("Score")

        # Keeping all three metrics on a consistent 0-1 scale.
        plt.ylim(0.0, 1.05)

        # Adding the plot title.
        plt.title("Safety mechanism comparison")

        # Adding light horizontal grid lines.
        plt.grid(True, axis="y", alpha=0.25)

        # Adding the legend above the plot.
        plt.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))

        # Building the output path.
        path = os.path.join(self.figures_dir, "safety_mechanism_comparison.png")

        # Saving the figure at 600 dpi.
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        # Closing the figure.
        plt.close()

        # Returning the saved figure path.
        return path

    def plot_component_ablation_comparison(self, summary):
        """
        Plot leave-one-out LAADAN component ablations in a compact horizontal figure.

        """

        # Creating the list of LAADAN component variants in the desired order.
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

        # If too few ablations exist, skip this figure.
        if len(selected) < 3:
            return None

        # Creating y-axis positions for horizontal bars.
        y = np.arange(len(selected))

        # Reading survival means and CIs.
        survival = [summary[name]["survival_rate"]["mean"] for name in selected]
        survival_ci = [summary[name]["survival_rate"]["ci95_half"] for name in selected]

        # Reading inadmissibility means and CIs.
        inad = [summary[name]["inadmissibility_rate"]["mean"] for name in selected]
        inad_ci = [summary[name]["inadmissibility_rate"]["ci95_half"] for name in selected]

        # Shortening labels for the y-axis.
        labels = [
            name.replace("LAADAN ", "")
                .replace("without ", "No ")
                .replace(" cost control", "")
            for name in selected
        ]

        # Setting the height of each horizontal bar.
        height = 0.35

        # Creating the figure.
        plt.figure(figsize=(10.8, 6.4), constrained_layout=True)

        # Plotting survival bars.
        plt.barh(y - height / 2, survival, height=height, xerr=survival_ci, capsize=4, label="Survival / return", color="#2171b5")

        # Plotting inadmissibility bars.
        plt.barh(y + height / 2, inad, height=height, xerr=inad_ci, capsize=4, label="Inadmissibility", color="#9ecae1")

        # Applying y-axis labels.
        plt.yticks(y, labels)

        # Labelling the x-axis.
        plt.xlabel("Score")

        # Keeping the x-axis on a consistent 0-1 scale.
        plt.xlim(0.0, 1.05)

        # Adding the title.
        plt.title("LAADAN-AC component ablation comparison")

        # Adding light vertical grid lines.
        plt.grid(True, axis="x", alpha=0.25)

        # Adding the legend.
        plt.legend(frameon=False, loc="lower right")

        # Building the output path.
        path = os.path.join(self.figures_dir, "laadan_component_ablation_comparison.png")

        # Saving the figure at 600 dpi.
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        # Closing the figure.
        plt.close()

        # Returning the saved figure path.
        return path

    def plot_lagrange_dynamics(self, all_runs):
        """
        Plot lambda, expected cost, survival and inadmissibility through training
        for the best no-mask Lagrangian run.
        """

        # Choosing the no-mask Lagrangian variant for the dynamics plot.
        candidate_name = "LAADAN without action mask"

        # If this variant does not exist, skip this figure.
        if candidate_name not in all_runs:
            return None

        # Finding the best seed run for this variant by survival.
        best_run = None
        for run in all_runs[candidate_name]:
            if best_run is None:
                best_run = run
                continue
            if run["metrics"].get("survival_rate", -1.0) > best_run["metrics"].get("survival_rate", -1.0):
                best_run = run

        # Reading the saved epoch history for the best run.
        rows = best_run["history"]

        # Creating lists for full-epoch training quantities.
        epochs = []
        lagrange = []
        expected_cost = []

        # Creating lists for evaluation-only quantities.
        survival_epochs = []
        survival = []
        inad = []

        # Extracting values from the history rows.
        for row in rows:
            epochs.append(row["epoch"])
            lagrange.append(row.get("lagrange", np.nan))
            expected_cost.append(row.get("expected_immediate_cost", np.nan))
            if "survival_rate" in row:
                survival_epochs.append(row["epoch"])
                survival.append(row["survival_rate"])
                inad.append(row["inadmissibility_rate"])

        # Creating the dynamics figure.
        plt.figure(figsize=(10.5, 6.2), constrained_layout=True)

        # Plotting the Lagrange multiplier over training.
        plt.plot(epochs, lagrange, linewidth=2.0, label="Lagrange multiplier")

        # Plotting the expected immediate cost over training.
        plt.plot(epochs, expected_cost, linewidth=2.0, label="Expected immediate cost")

        # Plotting evaluated survival at evaluation epochs.
        plt.plot(survival_epochs, survival, linewidth=2.0, label="Survival / return")

        # Plotting evaluated inadmissibility at evaluation epochs.
        plt.plot(survival_epochs, inad, linewidth=2.0, label="Evaluated inadmissibility")

        # Labelling the x-axis.
        plt.xlabel("Epoch")

        # Labelling the y-axis.
        plt.ylabel("Value")

        # Adding the title.
        plt.title("Adaptive Lagrangian dynamics without hard action masking")

        # Adding grid lines.
        plt.grid(True, alpha=0.25)

        # Adding the legend.
        plt.legend(frameon=False, loc="best")

        # Building the output path.
        path = os.path.join(self.figures_dir, "no_mask_lagrangian_training_dynamics.png")

        # Saving the figure at 600 dpi.
        plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")

        # Closing the figure.
        plt.close()

        # Returning the saved figure path.
        return path

    def write_manifest(self, files):
        """
        Save a manifest listing the generated outputs.
        """

        # Building the manifest path.
        path = os.path.join(self.root, "lagrangian_manifest.json")

        # Writing the dictionary of generated output files to JSON.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(files, handle, indent=2)

        # Returning the manifest path.
        return path


def unpack_state_dict(loaded_object):
    """
    Extract a raw PyTorch state_dict from a checkpoint.

    This keeps post-hoc masked VOAC evaluation independent from test_final_models.py.
    """

    # If the loaded checkpoint is a dictionary, inspect its structure.
    if isinstance(loaded_object, dict):

        # Checking common checkpoint keys that may contain the actual state_dict.
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in loaded_object and isinstance(loaded_object[key], dict):
                return loaded_object[key]

        # Checking whether the loaded object is already a raw state_dict.
        raw_state = True
        for value in loaded_object.values():
            if not torch.is_tensor(value):
                raw_state = False
                break

        # Returning the object directly if every value is a tensor.
        if raw_state:
            return loaded_object

    # Raising an error if no valid weights were found.
    raise ValueError("Could not find model weights inside checkpoint.")


def clean_state_dict_keys(state_dict):
    """
    Remove DataParallel 'module.' prefixes if they exist.
    """

    # Creating a new dictionary for cleaned key names.
    cleaned = {}

    # Looping through every parameter tensor in the state dictionary.
    for key, value in state_dict.items():

        # Removing the DataParallel prefix if present.
        if key.startswith("module."):
            cleaned[key[7:]] = value

        # Otherwise, keeping the original key.
        else:
            cleaned[key] = value

    # Returning the cleaned state dictionary.
    return cleaned


def evaluate_posthoc_masked_voac(benchmark, results_dir, config):
    """
    Evaluate an already-trained VOAC checkpoint with the admissibility mask applied only at evaluation time. 
    
    """

    # Building the expected path to the already-selected final VOAC checkpoint.
    model_path = os.path.join(results_dir, "final_models", "voac", "model.pt")

    # If the checkpoint is missing, skip this analysis instead of crashing.
    if not os.path.exists(model_path):
        print("[SKIP] Post-hoc Masked VOAC was skipped because this file was not found:", model_path)
        return None

    # Creating a VOAC-compatible actor-critic model without a cost head.
    model = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
        use_cost_head=False,
    ).to(benchmark.device)

    # Loading the checkpoint from disk.
    checkpoint = torch.load(model_path, map_location=benchmark.device)

    # Extracting the raw model weights from the checkpoint.
    state_dict = clean_state_dict_keys(unpack_state_dict(checkpoint))

    # Loading the checkpoint weights into the model.
    model.load_state_dict(state_dict)

    # Putting the model in evaluation mode.
    model.eval()

    # Building a masked greedy policy and masked soft policy from VOAC logits.
    with torch.no_grad():
        logits = model(benchmark.state_features_t)["logits"]
        primary_policy = masked_greedy_policy_from_logits(logits, benchmark.admissible_mask_t)
        soft_policy = masked_softmax_from_logits(logits, benchmark.admissible_mask_t)

    # Evaluating the post-hoc masked VOAC policy.
    metrics = evaluate_policy_set(
        benchmark,
        policy_numpy(primary_policy),
        soft_policy=policy_numpy(soft_policy),
    )

    # Sampling trajectories to compute trajectory-level action-change behaviour.
    simulation = benchmark.simulate_policy(
        policy_numpy(primary_policy),
        num_episodes=2000,
        seed=777,
    )

    # Saving raw trajectory action-jump metric.
    metrics["trajectory_mean_action_jump"] = float(simulation["mean_action_jump"])

    # Saving normalised trajectory action-jump metric.
    metrics["trajectory_normalised_action_jump"] = float(
        simulation["mean_action_jump"] / max(1, benchmark.num_actions - 1)
    )

    # Adding metadata identifying this evaluation-only variant.
    metrics["variant_folder"] = "posthoc_masked_voac"
    metrics["variant_name"] = "Post-hoc Masked VOAC"
    metrics["uses_action_mask"] = True
    metrics["uses_lagrangian"] = False
    metrics["cost_budget"] = 0.0

    # Building the output folder for this post-hoc evaluation.
    output_dir = os.path.join(results_dir, "lagrangian_frontier", "posthoc_masked_voac", "seed_final_voac")

    # Creating the output folder if needed.
    ensure_dir(output_dir)

    # Writing the post-hoc masked VOAC metrics file.
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Building the one-row history path.
    history_path = os.path.join(output_dir, "history.csv")

    # Writing the one-row history CSV for consistency with trained variants.
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # Returning a run dictionary so this result can be aggregated with trained variants.
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


def choose_device(requested):
    """
    Choose CPU or CUDA.
    """

    # If the user explicitly asked for CPU, use CPU.
    if requested == "cpu":
        return "cpu"

    # If the user explicitly asked for CUDA, check that CUDA exists.
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    # In auto mode, use CUDA when available, otherwise use CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_variants(args):
    """
    Build the variants.

    Core variants are always included. Budget sweep and leave-one-out variants
    can be included or skipped from the command line.
    """

    # Creating the core variants that are always trained.
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

    # Adding the no-mask Lagrangian cost-budget sweep unless the user skipped it.
    if not args.skip_budget_sweep:
        for budget in [0.005, 0.01, 0.02, 0.05]:

            # Creating a file-safe budget name such as 0_005.
            safe_name = str(budget).replace(".", "_")

            # Adding one no-mask Lagrangian variant for this budget value.
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

    # Adding leave-one-out component ablations only when requested.
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

    # Returning the final list of variants to train/evaluate.
    return variants


def parse_args():
    """
    Read command-line arguments.
    """

    # Creating the argument parser for this script.
    parser = argparse.ArgumentParser(description="Budgeted Lagrangian LAADAN-AC experiments.")

    # Required path to the ICU-Sepsis benchmark data folder.
    parser.add_argument("--data-dir", required=True, help="Path to ICU-Sepsis benchmark files.")

    # Output directory where results are saved.
    parser.add_argument("--results-dir", default="results", help="Output directory.")

    # Device selection: auto, cpu, or cuda.
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Run mode: debug uses one seed and fewer epochs; main uses the full setup.
    parser.add_argument("--mode", choices=["debug", "main"], default="main")

    # Optional flag to skip the budget sweep.
    parser.add_argument("--skip-budget-sweep", action="store_true")

    # Optional flag to include leave-one-out component ablations.
    parser.add_argument("--include-leave-one-out", action="store_true")

    # Optional flag to skip post-hoc masked VOAC evaluation.
    parser.add_argument("--skip-masked-voac", action="store_true")

    # Number of simulated trajectories used for trajectory-level action-jump metrics.
    parser.add_argument("--num-trajectory-episodes", type=int, default=2000)

    # Returning parsed arguments.
    return parser.parse_args()


def main():
    """
    Run the complete enhancement experiment.
    """

    # Reading command-line arguments.
    args = parse_args()

    # Choosing CPU or CUDA based on the requested device and availability.
    device = choose_device(args.device)

    # Copying the base configuration so this run can modify it safely.
    config = dict(BASE_LAADAN_CONFIG)

    # Using five seeds for the main run.
    seeds = [42, 43, 44, 45, 46]

    # If debug mode is selected, use one seed and fewer epochs for quick testing.
    if args.mode == "debug":
        seeds = [42]
        config["epochs"] = 40
        config["eval_every"] = 5

    # Creating the results directory if it does not exist.
    ensure_dir(args.results_dir)

    # Loading the ICU-Sepsis benchmark using the same state features as the main project.
    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=50,
        device=device,
        use_one_hot_states=False,
    )

    # Saving a benchmark description file for reproducibility.
    benchmark.save_benchmark_description(
        os.path.join(args.results_dir, "lagrangian_benchmark_description.json")
    )

    # Building the list of variants requested by the command-line flags.
    variants = build_variants(args)

    # Storing the requested trajectory simulation count inside every variant.
    for variant in variants:
        variant["num_trajectory_episodes"] = int(args.num_trajectory_episodes)

    # Creating the experiment runner object.
    experiment = BudgetedLagrangianExperiment(benchmark, args.results_dir, config)

    # Training every requested variant across all selected seeds.
    all_runs = experiment.run_variants(variants, seeds)

    # Evaluating post-hoc masked VOAC unless skipped.
    if not args.skip_masked_voac:
        masked_voac_run = evaluate_posthoc_masked_voac(benchmark, args.results_dir, config)
        if masked_voac_run is not None:
            all_runs["Post-hoc Masked VOAC"] = [masked_voac_run]

    # Creating the reporter object for tables, plots and final model selection.
    reporter = FrontierReporter(benchmark, args.results_dir)

    # Aggregating all runs into mean ± 95% CI summaries.
    summary = reporter.aggregate_runs(all_runs)

    # Creating a dictionary that records the generated output files.
    files = {}

    # Writing the compact CSV summary.
    files["summary_csv"] = reporter.write_summary_csv(summary)

    # Writing the full JSON summary.
    files["summary_json"] = reporter.save_summary_json(summary)

    # Selecting one final checkpoint per variant.
    files["final_models_csv"], files["final_models_json"] = reporter.select_final_models(all_runs)

    # Creating the budgeted Lagrangian return-safety figure.
    files["pareto_frontier_png"] = reporter.plot_pareto_frontier(summary)

    # Creating the safety mechanism comparison figure.
    files["mechanism_comparison_png"] = reporter.plot_mechanism_comparison(summary)

    # Creating the LAADAN component ablation comparison figure.
    files["component_ablation_png"] = reporter.plot_component_ablation_comparison(summary)

    # Creating the no-mask Lagrangian training dynamics figure.
    files["lagrange_dynamics_png"] = reporter.plot_lagrange_dynamics(all_runs)

    # Writing a manifest that lists all generated output files.
    files["manifest_json"] = reporter.write_manifest(files)

    # Printing the main output folder.
    print("[DONE] Lagrangian outputs saved to:", reporter.root)

    # Printing the final selected model folder.
    print("[DONE] Final selected models saved to:", reporter.final_dir)

    # Printing every generated output file path.
    for key in sorted(files.keys()):
        print("[FILE]", key + ":", files[key])


# Running main() only when this file is executed directly.
if __name__ == "__main__":
    main()
    
