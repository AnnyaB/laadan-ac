"""Training and evaluation utilities for BC, CQL, VOAC, and LAADAN-AC."""


# trainers.py

# Training loops for the four-model offline ICU-Sepsis, eicu and other experiments.

# Libraries used in this file

# CSV is used for writing training histories to disk.
import csv

# for saving metrics in a structured reproducible format.
import json

# for filesystem paths and folder creation.
import os

# to measure total training time for each run.
import time

# for metric aggregation and summary statistics.
import numpy as np

# PyTorch is used for all tensor operations, neural-network training,
# optimisation, gradient updates, and checkpointing.
import torch

# Importing all network classes and the target-network update helper
# from models.py.
from models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet, soft_update


def ensure_dir(path):

    """
    Creating a folder if it does not already exist.
    """

    # Checking whether the target folder already exists.
    if not os.path.exists(path):
        # If not, create it.
        os.makedirs(path)


# Precomputed critical t-values for a 95% confidence interval.

_T_CRIT_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def mean_ci95(values):

    """
    Returning mean, sample standard deviation, and 95% t-based confidence interval.

    """
    # Converting the input to a NumPy array of floats. for instance, [0.792, 0.793, 0.794]
    values = np.asarray(values, dtype=float)

    # Removing invalid NaN or infinite values before computing statistics.
    values = values[np.isfinite(values)]

    # Counting the number of valid observations. For five seeds: n = 5
    n = int(values.size)

    # If there are no valid values, return NaNs so downstream code can detect
    # the missing summary cleanly.
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_half": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    # Computing the arithmetic mean, (the average).
    mean = float(np.mean(values))

    # If only one value exists, the spread and CI are effectively zero because
    # there is no multi-run variation to estimate, so std = 0, CI = 0
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_half": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }

    # Computing sample standard deviation with ddof=1 for an unbiased
    # sample-based estimate.
    std = float(np.std(values, ddof=1))

    # Look up the t critical value; if n is outside the small dictionary, use
    # 1.96 as a reasonable large-sample approximation.
    t_crit = _T_CRIT_95.get(n, 1.96)

    # Computing the half-width of the 95% confidence interval.
    ci95_half = float(t_crit * std / np.sqrt(n))


    # Returning the full summary dictionary.
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_half": ci95_half,
        "ci95_low": mean - ci95_half,
        "ci95_high": mean + ci95_half,
    }


def plain_softmax_from_logits(logits):

    """
    Standard softmax with no admissibility masking. Used when the policy is allowed to place probability mass over all actions.
    """

    # Converting raw action logits to probabilities over actions.
    return torch.softmax(logits, dim=1)


def plain_log_softmax_from_logits(logits):

    # gives log-probabilities instead of probabilities.
    # Because they are more numerically stable for:
    # entropy
    # KL divergence
    # policy losses

    # Converting raw logits to log-probabilities.
    return torch.log_softmax(logits, dim=1)


def masked_softmax_from_logits(logits, admissible_mask):

    """
    Softmax after hiding inadmissible actions with a large negative logit.
    """

    # Creating a tensor with a very large negative value matching the logits shape.
    large_negative = torch.full_like(logits, -1e9) # -1,000,000,000

    # Keeping original logits where actions are admissible; replace inadmissible
    # actions with the large negative value.
    masked = torch.where(admissible_mask > 0.5, logits, large_negative)

    # Converting the masked logits into a valid probability distribution.
    return torch.softmax(masked, dim=1) # So LAADAN-AC cannot select blocked actions through the masked policy.


def masked_log_softmax_from_logits(logits, admissible_mask):

    """
    Log-softmax after hiding inadmissible actions with a large negative logit.

    """
    # Creating the same large negative tensor used to suppress forbidden actions.
    large_negative = torch.full_like(logits, -1e9)

    # Replacing inadmissible-action logits with a very negative number.
    masked = torch.where(admissible_mask > 0.5, logits, large_negative)

    # Returning log-probabilities after masking.
    return torch.log_softmax(masked, dim=1)


def greedy_policy_from_logits(logits):

    """
    Deterministic one-hot policy from raw logits.

    This chooses the single highest-logit action in each state.
    """
    # Finding the index of the largest logit in each row.
    best = torch.argmax(logits, dim=1)

    # Creating a zero matrix with the same shape as the logits.
    policy = torch.zeros_like(logits) # [0, 0, 0, 0, ...]

    # Writing a 1.0 into the best-action column for each row.
    policy.scatter_(1, best.unsqueeze(1), 1.0)

    # Returning the resulting one-hot deterministic policy.
    return policy


def masked_greedy_policy_from_logits(logits, admissible_mask):

    """
    Deterministic one-hot policy from logits after admissibility masking.

    This ensures the greedy action is selected only from admissible actions.
    """

    # Building the large negative filler used to suppress forbidden actions.
    large_negative = torch.full_like(logits, -1e9)

    # Replacing inadmissible logits with the large negative constant.
    masked = torch.where(admissible_mask > 0.5, logits, large_negative)

    # Choosing the best admissible action in each state.
    best = torch.argmax(masked, dim=1)

    # Creating an all-zero policy tensor.
    policy = torch.zeros_like(logits)

    # Placing 1.0 on the chosen action index for each row.
    policy.scatter_(1, best.unsqueeze(1), 1.0)

    # Returning the masked greedy policy.
    return policy


def weighted_policy_kl(logits, target_policy, admissible_mask=None):

    # Choosing either masked or unmasked log-probabilities depending on whether
    # The action space should be restricted.
    if admissible_mask is None:
        log_probs = plain_log_softmax_from_logits(logits)
    else:
        log_probs = masked_log_softmax_from_logits(logits, admissible_mask)


    # Normalizing the target policy to ensure each row sums to 1 even if the input
    # arrives with minor numerical mismatch.
    target = target_policy / torch.clamp(torch.sum(target_policy, dim=1, keepdim=True), min=1e-8)

    # Computing KL(target || model) row-wise:
    # sum target * (log target - log model)
    kl = torch.sum(target * (torch.log(target + 1e-8) - log_probs), dim=1)

    # Returning the mean KL over all states.
    return torch.mean(kl)


def cql_regularizer(q_values, expert_policy, admissible_mask=None):

    """
    Conservative Q-Learning like penalty used for this experiment.
    """
    # If no mask is provided, applying logsumexp over all actions.
    if admissible_mask is None:
        conservative_term = torch.logsumexp(q_values, dim=1)


    else:
        # Otherwise hide inadmissible actions before logsumexp.
        masked = torch.where(admissible_mask > 0.5, q_values, torch.full_like(q_values, -1e9))

        conservative_term = torch.logsumexp(masked, dim=1)


    # Computing the value of expert-supported actions under the current Q-values.
    data_term = torch.sum(expert_policy * q_values, dim=1)

    # Returning the mean conservative penalty.
    return torch.mean(conservative_term - data_term)


def masked_action_mse(pred, target, action_mask=None):

    """
    Mean squared error over all actions or over a masked subset.

    """
    # Computing elementwise squared error.
    squared = (pred - target) ** 2

    # If no mask is supplied, just return the ordinary mean squared error.
    if action_mask is None:
        return torch.mean(squared)

    # Converting the mask to float so it can be used as weights.
    weights = action_mask.float()

    # Computing the sum of weights; clamp to at least 1.0 so division is safe.
    denom = torch.clamp(torch.sum(weights), min=1.0)

    # Returning the weighted masked MSE.
    return torch.sum(squared * weights) / denom


def save_history_csv(path, rows):

    """
    Writing a list of dictionaries to CSV, allowing rows to have different keys.

    """
    # If there is no history to save, do nothing.
    if not rows:
        return

    # Collecting the full union of field names across all rows.
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    # Opening the destination CSV file.
    with open(path, "w", newline="", encoding="utf-8") as handle:
        # Building the CSV writer using the discovered fieldnames.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")

        # Writing the header row first.
        writer.writeheader()

        # Writing each history row, filling missing keys with empty strings.
        for row in rows:
            full_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(full_row)


def summarize_history(history):

    """
    Creating a compact summary from a history whose rows may have different keys.

    """
    # If no history exists, return an empty summary.
    if not history:
        return {}

    # Building a list of all unique keys seen across rows.
    all_keys = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    # Preparing the output dictionary.
    summary = {}

    # Processing each field independently.
    for key in all_keys:
        values = []

        # Collecting all finite numeric values for the current key.
        for row in history:
            value = row.get(key, np.nan)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                values.append(float(value))

        # If at least one valid value exists, store last, min, and max summaries.
        if values:
            summary[key + "_last"] = values[-1]
            summary[key + "_best_min"] = float(np.min(values))
            summary[key + "_best_max"] = float(np.max(values))

    # Returning the compact history summary.
    return summary


def epoch_to_fraction_of_best(history, metric_name, fraction=0.95):

    """
    Finding the first epoch that reaches a fraction of the best value.

    """
    # Collecting all finite (epoch, value) pairs for the requested metric.
    metric_pairs = []
    for row in history:
        value = row.get(metric_name, np.nan)
        if np.isfinite(value):
            metric_pairs.append((int(row["epoch"]), float(value)))

    # If no valid metric values exist, convergence cannot be estimated.
    if not metric_pairs:
        return None

    # Finding the best metric value achieved during training.
    best_value = max(value for _, value in metric_pairs)

    # Setting the convergence threshold as the chosen fraction of the best value.
    threshold = fraction * best_value

    # Returning the first epoch that reaches or exceeds that threshold.
    for epoch, value in metric_pairs:
        if value >= threshold:
            return int(epoch)

    # If never reached, return None.
    return None


def save_model_run(seed_dir, model, history, metrics):

    """
    Saving the trained model, CSV history, and metrics JSON for one seed.

    """

    # Making sure the output folder exists.
    ensure_dir(seed_dir)

    # Saving the model weights.
    torch.save(model.state_dict(), os.path.join(seed_dir, "model.pt"))

    # Saving the training history as CSV.
    save_history_csv(os.path.join(seed_dir, "history.csv"), history)

    # Saving the metric dictionary as JSON.
    with open(os.path.join(seed_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


@torch.no_grad()
def evaluate_policy_set(benchmark, policy, soft_policy=None):

    """
    Evaluating the primary policy and optionally a secondary soft policy.

    """

    # Evaluating the primary policy exactly on the benchmark.
    metrics = benchmark.exact_policy_evaluation(policy)

    # If a soft-policy version is also provided, evaluate it too and append
    # those values as auxiliary metrics.
    if soft_policy is not None:
        soft_metrics = benchmark.exact_policy_evaluation(soft_policy)
        metrics["soft_survival_rate"] = soft_metrics["survival_rate"]
        metrics["soft_inadmissibility_rate"] = soft_metrics["inadmissibility_rate"]
        metrics["soft_avg_return"] = soft_metrics["avg_return"]
        metrics["soft_policy_entropy"] = soft_metrics["policy_entropy"]
        metrics["soft_mean_kl_to_expert"] = soft_metrics["mean_kl_to_expert"]


    # Returning the combined metric dictionary.
    return metrics


@torch.no_grad()
def policy_numpy(tensor_policy):

    """
    Converting a PyTorch policy tensor to a NumPy array.

    """
    return tensor_policy.detach().cpu().numpy()


def train_behavior_cloning(benchmark, seed, results_dir, config):

    """
    Training the clinician-imitation baseline.
    """

    # Setting Python, NumPy, and Torch seeds for reproducibility.
    benchmark.set_seed(seed)

    # Recording the training device for convenience.
    device = benchmark.device

    # Building the Behavior Cloning network and moving it to the target device.
    model = BehaviorCloningNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)


    # Creating the Adam optimizer for all model parameters. Adam updates the model weights using gradients.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    # Reading high-level training settings from the configuration.
    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))

    # Shortcut references to benchmark tensors.
    x = benchmark.state_features_t
    target = benchmark.expert_safe_t

    # Preparing per-epoch logging structures.
    history = []
    best_survival = -1.0
    best_state = None
    best_metrics = None

    # Recording wall-clock start time.
    start_time = time.time()

    # Main epoch loop.
    for epoch in range(1, epochs + 1):
        # Putting the model in training mode so dropout is active as intended.
        model.train()

        # Clearing old gradients. affects in a detrimental way if not resulting in slow convergence.
        optimizer.zero_grad()

        # Forward pass: get policy logits for every state.
        logits = model(x)

        # Computing imitation loss against the benchmark expert-safe policy.
        kl_loss = weighted_policy_kl(logits, target, admissible_mask=None) # Small KL = close to expert.

        # Building the soft policy distribution for entropy regularisation.
        probs = plain_softmax_from_logits(logits)

        # Building log-probabilities for entropy calculation.
        log_probs = plain_log_softmax_from_logits(logits)

        # Computing average policy entropy.
        entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))


        # Total loss:
        # imitation loss minus optional entropy bonus.
        loss = kl_loss - float(config.get("entropy_bonus", 0.0)) * entropy

        # Backpropagate.
        loss.backward() # Which weights caused the loss, and how should they change?

        # Updating parameters.
        optimizer.step()

        # Starting the row with always-available training values. Save the current epoch’s training values.
        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "kl_loss": float(kl_loss.item()),
            "entropy": float(entropy.item()),
        }

        # Running evaluation at the chosen interval or at the final epoch.
        if epoch % eval_every == 0 or epoch == epochs:   # Evaluate every 10 epochs and at the final epoch.

            # Switching to evaluation mode so dropout is disabled.
            model.eval()

            with torch.no_grad():  # No gradient tracking during evaluation.

                # Recomputing logits for evaluation.
                logits_eval = model(x)

                # Building the greedy one-hot policy used for primary comparison.
                greedy_policy = greedy_policy_from_logits(logits_eval)

                # Also keeping the soft policy for auxiliary analysis.
                soft_policy = plain_softmax_from_logits(logits_eval)




            # Evaluating both policies through the benchmark evaluator.
            metrics = evaluate_policy_set(
                benchmark,
                policy_numpy(greedy_policy),
                soft_policy=policy_numpy(soft_policy),
            )

            # Adding evaluation metrics into the current history row.
            row.update(metrics)

            # Saving checkpoint state if this is the best survival so far. The best checkpoint is selected by survival rate.
            if metrics["survival_rate"] > best_survival:
                best_survival = metrics["survival_rate"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_metrics = metrics

        # Appending the row to the training history.
        history.append(row)


    # After training:
    # Restoring the best checkpoint if one was captured.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Computing final policy outputs from the restored best model.
    model.eval()
    with torch.no_grad():
        logits_final = model(x)
        final_policy = greedy_policy_from_logits(logits_final)
        final_soft_policy = plain_softmax_from_logits(logits_final)

    # If for some reason no evaluation was recorded, compute metrics now.
    if best_metrics is None:
        best_metrics = evaluate_policy_set(
            benchmark,
            policy_numpy(final_policy),
            soft_policy=policy_numpy(final_soft_policy),
        )

    # Building the per-seed output folder.
    seed_dir = os.path.join(results_dir, "bc", "seed_" + str(seed))

    # Saving checkpoint, history, and metrics.
    save_model_run(seed_dir, model, history, best_metrics)

    # Returning a structured result dictionary.
    return {
        "name": "Behavior Cloning",
        "seed": seed,
        "model": model,
        "policy": policy_numpy(final_policy),
        "analysis_policy": policy_numpy(final_soft_policy),
        "history": history,
        "metrics": best_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }


def train_cql(benchmark, seed, results_dir, config): # CQL is value-learning.

    """
    Training the conservative offline value-learning baseline.

    Primary evaluation uses greedy action selection from the learned Q-values.
    """
    # Setting random seeds for reproducibility.
    benchmark.set_seed(seed)

    # Recording training device.
    device = benchmark.device

    # Building the main Q-network.
    model = ConservativeQNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)

    # Building the target Q-network.
    target_model = ConservativeQNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)

    # Starting target model with identical weights.
    target_model.load_state_dict(model.state_dict())

    # Keeping target model in evaluation mode for stable targets.
    target_model.eval()

    # Creating the optimizer for the main model.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    # Reading algorithm hyperparameters.
    gamma = float(config.get("gamma", 1.0))
    cql_alpha = float(config.get("cql_alpha", 1.0))
    tau = float(config.get("tau", 0.02))
    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))


    # Shortcut references to benchmark tensors.
    x = benchmark.state_features_t
    expert = benchmark.expert_safe_t
    reward_sa = benchmark.reward_sa_t
    transition = benchmark.transition_t
    terminal = benchmark.terminal_mask_t

    # Preparing tracking variables.
    history = []
    best_survival = -1.0
    best_state = None
    best_metrics = None
    start_time = time.time()

    # Main training loop.
    for epoch in range(1, epochs + 1):

        # Enabling training mode for the main model.
        model.train()

        # Clearing previous gradients.
        optimizer.zero_grad()

        # Forward pass through the current Q-network.
        q_values = model(x)  # Main Q-network predicts Q-values. Shape: [716, 25]

        # Building Bellman targets using the slowly updated target network.
        with torch.no_grad():
            q_target_now = target_model(x)

            # Value of next state = max Q over actions, masked by non-terminal states.
            next_v = torch.max(q_target_now, dim=1).values * (1.0 - terminal)

            # Bellman target for every state-action pair.
            bellman_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)


        # Temporal-difference loss over all state-action values.
        td_loss = torch.mean((q_values - bellman_target) ** 2)  # TD loss. It trains Q-values to match Bellman targets.

        # Conservative penalty discouraging unsupported value inflation.
        conservative_loss = cql_regularizer(q_values, expert, admissible_mask=None)

        # Total CQL objective.
        loss = td_loss + cql_alpha * conservative_loss # Bellman error + conservative penalty

        # Backpropagate.
        loss.backward()

        # Clipping gradients for stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        # Optimizer step.
        optimizer.step()

        # Soft-update the target network.
        soft_update(target_model, model, tau)

        # Keepping the target network in eval mode after update.
        target_model.eval()

        # Log training losses.
        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(conservative_loss.item()),
        }

        # Periodic evaluation.
        if epoch % eval_every == 0 or epoch == epochs:
            # Disable dropout in the main model.
            model.eval()

            with torch.no_grad():
                # Getting current Q-values as NumPy.
                q_now = model(x).detach().cpu().numpy()

                # Converting Q-values to a deterministic greedy policy.
                greedy_policy = benchmark.greedy_policy_from_q_unmasked(q_now)

                # CQL chooses the highest Q-value action. it can choose inadmissible actions

            # Evaluating the greedy policy.
            metrics = benchmark.exact_policy_evaluation(greedy_policy)

            # Merging metrics into the row.
            row.update(metrics)

            # Saving best checkpoint if survival improved.
            if metrics["survival_rate"] > best_survival:
                best_survival = metrics["survival_rate"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_metrics = metrics

        # Appending row to history.
        history.append(row)

    # Restoring best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final policy extraction from best restored model.
    model.eval()
    with torch.no_grad():
        q_final = model(x).detach().cpu().numpy()
        final_policy = benchmark.greedy_policy_from_q_unmasked(q_final)

    # Fallback in case best metrics were never set.
    if best_metrics is None:
        best_metrics = benchmark.exact_policy_evaluation(final_policy)

    # Output folder for this seed.
    seed_dir = os.path.join(results_dir, "cql", "seed_" + str(seed))

    # Saving run outputs.
    save_model_run(seed_dir, model, history, best_metrics)

    # Returning structured run result.
    return {
        "name": "Conservative Q-Learning",
        "seed": seed,
        "model": model,
        "policy": final_policy,
        "analysis_policy": final_policy,
        "history": history,
        "metrics": best_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }


def _actor_critic_common_setup(benchmark, config, use_cost_head):

    """ This builds shared setup for VOAC and LAADAN-AC. Because both use OfflineActorCriticNet """

    # Reading device.
    device = benchmark.device

    # Building the main actor-critic model.
    model = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
        use_cost_head=use_cost_head,
    ).to(device)

    # Building the target actor-critic model.
    target_model = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
        use_cost_head=use_cost_head,
    ).to(device)


    # Copying weights from main to target.
    target_model.load_state_dict(model.state_dict())

    # Keeping target network in eval mode.
    target_model.eval()

    # Bundle frequently used tensors for convenience.
    tensors = {
        "x": benchmark.state_features_t,
        "admissible": benchmark.admissible_mask_t,
        "expert": benchmark.expert_safe_t,
        "reward_sa": benchmark.reward_sa_t,
        "transition": benchmark.transition_t,
        "terminal": benchmark.terminal_mask_t,
        "immediate_cost": benchmark.immediate_cost_t,
        "smoothness_cost": benchmark.smoothness_cost_t,
    }

    # Returning the built models and tensor bundle.
    return model, target_model, tensors


def train_voac(benchmark, seed, results_dir, config):

    """ VOAC = Vanilla Offline Actor-Critic. It has:
    actor
    critic Q1
    critic Q2
    target network

    It does not have since it's the ablation:

    action mask
    cost head
    expert KL
    smoothness penalty
    Lagrangian safety """


    # Setting random seeds.
    benchmark.set_seed(seed)

    # Building model, target model, and tensor bundle.
    model, target_model, tensors = _actor_critic_common_setup(benchmark, config, use_cost_head=False)

    # Actor parameters are only the actor encoder and actor head.
    actor_params = list(model.actor_encoder.parameters()) + list(model.actor_head.parameters())

    # Actor parameters are:
    # actor encoder weights
    # actor head weights

    # Critic parameters are the critic encoder and the two Q heads.
    critic_params = (
        list(model.critic_encoder.parameters())
        + list(model.q1_head.parameters())
        + list(model.q2_head.parameters())
    )

    # Critic parameters are:
    # critic encoder weights
    # Q1 head weights
    # Q2 head weights


    # VOAC USES TWO OPTIMISERS

    # Separating optimizer for the actor.
    actor_optimizer = torch.optim.Adam(
        actor_params,
        lr=float(config.get("actor_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    # Separating optimizer for the critics.
    critic_optimizer = torch.optim.Adam(
        critic_params,
        lr=float(config.get("critic_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    # Reading algorithm settings.
    gamma = float(config.get("gamma", 1.0))
    tau = float(config.get("tau", 0.02))
    entropy_coef = float(config.get("entropy_coef", 0.02))
    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))

    # Shortcuts to benchmark tensors.
    x = tensors["x"]
    reward_sa = tensors["reward_sa"]
    transition = tensors["transition"]
    terminal = tensors["terminal"]

    # Preparing tracking structures.
    history = []
    best_survival = -1.0
    best_state = None
    best_metrics = None
    start_time = time.time()

    # Main epoch loop.
    for epoch in range(1, epochs + 1):
        # Setting main model to training mode.
        model.train()


        # Critic update

        with torch.no_grad():
            # Getting target-network outputs.
            target_outputs = target_model(x) # Run target actor-critic.

            # Building the target policy distribution from target actor logits.
            next_probs = plain_softmax_from_logits(target_outputs["logits"])
            next_log_probs = plain_log_softmax_from_logits(target_outputs["logits"])

            # Build target actor’s soft policy. No mask.

            # Reading both target critics.
            q1_next = target_outputs["q1"]
            q2_next = target_outputs["q2"]

            # Using the minimum of the two critics for a more conservative target.
            min_q_next = torch.min(q1_next, q2_next)

            """ Take the lower of Q1 and Q2. This reduces over-optimistic value estimates."""

            # Computing soft value target for next state.
            next_v = torch.sum(
                next_probs * (min_q_next - entropy_coef * next_log_probs),
                dim=1,
            ) * (1.0 - terminal) # Average the future Q-values under the target policy, including entropy regularisation.

            # Bellman target for every state-action pair.
            q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)

        # Resetting critic gradients.
        critic_optimizer.zero_grad()

        # Forward pass through current model.
        outputs = model(x)
        q1 = outputs["q1"]
        q2 = outputs["q2"]

        # Current critic predictions.

        # Computing critic losses against the shared target.
        q1_loss = torch.mean((q1 - q_target) ** 2)
        q2_loss = torch.mean((q2 - q_target) ** 2)

        # Total critic loss is the sum of both critic errors.
        critic_loss = q1_loss + q2_loss

        # Train both critics to match Bellman target.

        # Backpropagate critic loss.
        critic_loss.backward()

        # Gradient clipping for critic stability.
        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)

        # Updating critic parameters.
        critic_optimizer.step()

        # Update critic weights.

        # Actor update

        # Resetting actor gradients.
        actor_optimizer.zero_grad()

        # Forward pass again because parameters changed after critic step.
        outputs = model(x)
        logits = outputs["logits"]
        probs = plain_softmax_from_logits(logits)
        log_probs = plain_log_softmax_from_logits(logits)

        # Actor produces an unmasked soft policy.

        q1 = outputs["q1"]
        q2 = outputs["q2"]

        # Again use the smaller critic estimate.
        min_q = torch.min(q1, q2)  # Use conservative lower Q estimate.

        # Actor objective:

        # maximising value while accounting for entropy regularisation.
        actor_loss = torch.mean(torch.sum(probs * (entropy_coef * log_probs - min_q), dim=1))

        # Because optimiser minimises loss, this means: maximise Q-value and encourage entropy

        # Backpropagate actor loss.
        actor_loss.backward()

        # Gradient clipping for actor stability.
        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)

        # Updating actor parameters.
        actor_optimizer.step()

        # Updating target network toward current model.
        soft_update(target_model, model, tau)

        # Keeping target network in eval mode.
        target_model.eval()

        # Preparing training row.
        row = {
            "epoch": epoch,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
        }

        # Periodic evaluation.
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x)["logits"]
                greedy_policy = greedy_policy_from_logits(logits_eval)
                soft_policy = plain_softmax_from_logits(logits_eval)

            metrics = evaluate_policy_set(
                benchmark,
                policy_numpy(greedy_policy),
                soft_policy=policy_numpy(soft_policy),
            )

            row.update(metrics)

            # Saving best checkpoint.
            if metrics["survival_rate"] > best_survival:
                best_survival = metrics["survival_rate"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_metrics = metrics

        # Storing row.
        history.append(row)

    # Restoring best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final policy extraction.
    model.eval()
    with torch.no_grad():
        logits_final = model(x)["logits"]
        final_policy = greedy_policy_from_logits(logits_final)
        final_soft_policy = plain_softmax_from_logits(logits_final)

    # Fallback metric computation
    if best_metrics is None:
        best_metrics = evaluate_policy_set(
            benchmark,
            policy_numpy(final_policy),
            soft_policy=policy_numpy(final_soft_policy),
        )

    # Saving run output
    seed_dir = os.path.join(results_dir, "voac", "seed_" + str(seed))
    save_model_run(seed_dir, model, history, best_metrics)

    # Returning structured result.
    return {
        "name": "Vanilla Offline Actor-Critic",
        "seed": seed,
        "model": model,
        "policy": policy_numpy(final_policy),
        "analysis_policy": policy_numpy(final_soft_policy),
        "history": history,
        "metrics": best_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }


def train_laadan_ac(benchmark, seed, results_dir, config):

    """
    Train the proposed LAADAN-AC model.

    LAADAN-AC = Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic

    Design features beyond VOAC:
    - admissibility-aware action masking
    - conservative critic regularisation
    - expert-policy regularisation
    - smoothness penalty
    - cost critic and Lagrangian safety control

    Primary evaluation uses masked greedy action selection for a fair comparison
    against deterministic CQL testing. Soft masked metrics are stored as extras.
    """

    # Setting random seeds.
    benchmark.set_seed(seed)

    # Building model, target model, and benchmark tensors.
    model, target_model, tensors = _actor_critic_common_setup(benchmark, config, use_cost_head=True)

    # Actor parameters: actor encoder + actor head.
    actor_params = list(model.actor_encoder.parameters()) + list(model.actor_head.parameters())

    # Critic parameters: critic encoder + both reward critics + cost head.
    critic_params = (
        list(model.critic_encoder.parameters())
        + list(model.q1_head.parameters())
        + list(model.q2_head.parameters())
        + list(model.cost_head.parameters())
    )
    # Actor and critic/cost are trained separately.

    # two adam optimizers
    # Actor optimizer.
    actor_optimizer = torch.optim.Adam(
        actor_params,
        lr=float(config.get("actor_lr", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    # Critic optimizer.
    critic_optimizer = torch.optim.Adam(
        critic_params,
        lr=float(config.get("critic_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    # Reading LAADAN hyperparameters.
    gamma = float(config.get("gamma", 1.0)) # 1.0 means future survival is fully considered across the finite horizon.
    tau = float(config.get("tau", 0.01)) # Target network update speed. Smaller tau = slower, more stable target updates.
    entropy_coef = float(config.get("entropy_coef", 0.001)) # This prevents policy from becoming too rigid too early.
    conservative_alpha = float(config.get("conservative_alpha", 0.25)) # Higher value = critic becomes more cautious. too high can reduce return
    expert_kl_weight = float(config.get("expert_kl_weight", 0.005)) # Strength of expert-policy regularisation. Higher = behave more like expert/BC, Lower = more freedom for RL optimisation.
    smoothness_weight = float(config.get("smoothness_weight", 0.001)) # Penalty for action choices far from expert-supported mean action. Higher = smoother/more conservative action patterns.
    cost_budget = float(config.get("cost_budget", 0.0)) # expected unsafe cost should be zero
    lagrange_lr = float(config.get("lagrange_lr", 0.0002))
    lagrange_value = float(config.get("lagrange_init", 0.0))
    # Lagrangian multiplier controls how strongly unsafe cost is penalised.
    # If expected cost rises above budget, lagrange value increases.

    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))

    # Tensor shortcuts.
    x = tensors["x"]
    admissible = tensors["admissible"]
    expert = tensors["expert"]
    reward_sa = tensors["reward_sa"]
    transition = tensors["transition"]
    terminal = tensors["terminal"]
    immediate_cost = tensors["immediate_cost"]
    smoothness_cost = tensors["smoothness_cost"]

    # Tracking structures.
    history = []
    best_survival = -1.0
    best_state = None
    best_metrics = None
    start_time = time.time()

    # Main epoch loop.
    for epoch in range(1, epochs + 1):
        # Enable training mode for main model.
        model.train()

        # Critic target computation

        with torch.no_grad():
            target_outputs = target_model(x) # Use target network to compute stable future targets.

            # Target actor probabilities with admissibility mask applied.
            next_probs = masked_softmax_from_logits(target_outputs["logits"], admissible)
            next_log_probs = masked_log_softmax_from_logits(target_outputs["logits"], admissible)

            # Target reward critics.
            q1_next = target_outputs["q1"]
            q2_next = target_outputs["q2"]

            # Target cost critic, forced non-negative through ReLU.
            cost_next = torch.relu(target_outputs["cost"])

            # Conservative reward estimate from the smaller reward critic.
            min_q_next = torch.min(q1_next, q2_next)

            # Soft next-state value including entropy and cost pressure.
            next_v = torch.sum(
                next_probs * (min_q_next - entropy_coef * next_log_probs - lagrange_value * cost_next),
                dim=1,
            ) * (1.0 - terminal)

            # Next-state expected cost.
            next_c = torch.sum(next_probs * cost_next, dim=1) * (1.0 - terminal)

            # Reward Bellman target.
            q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)

            # Cost Bellman target.
            c_target = immediate_cost + gamma * torch.einsum("san,n->sa", transition, next_c)


        # Critic update

        # Get current Q and cost predictions.

        critic_optimizer.zero_grad()

        outputs = model(x)
        q1 = outputs["q1"]
        q2 = outputs["q2"]

        # Current predicted cost values, again passed through ReLU.
        cost_pred = torch.relu(outputs["cost"])

        # Reward critics are fitted only over admissible actions.
        q1_loss = masked_action_mse(q1, q_target, action_mask=admissible)
        q2_loss = masked_action_mse(q2, q_target, action_mask=admissible)

        # Cost critic is also fitted only over admissible actions.
        cost_loss = masked_action_mse(cost_pred, c_target, action_mask=admissible)

        # Conservative penalties over admissible actions only.
        cql_q1 = cql_regularizer(q1, expert, admissible_mask=admissible)
        cql_q2 = cql_regularizer(q2, expert, admissible_mask=admissible)

        # Total critic loss.
        critic_loss = q1_loss + q2_loss + cost_loss + conservative_alpha * (cql_q1 + cql_q2)

        # Backpropagate critic objective.
        critic_loss.backward()

        # Gradient clipping for critic stability.
        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)

        # Critic step.
        critic_optimizer.step()

        # Calculate gradients, clip them, update critic/cost weights.

        # Actor update

        actor_optimizer.zero_grad()

        outputs = model(x)
        logits = outputs["logits"]

        # Start actor update.

        # Actor policy after masking.
        probs = masked_softmax_from_logits(logits, admissible)
        log_probs = masked_log_softmax_from_logits(logits, admissible)

        # The actor only assigns probability to admissible actions.

        q1 = outputs["q1"]
        q2 = outputs["q2"]
        cost_pred = torch.relu(outputs["cost"])

        # Conservative reward estimate from the smaller reward critic.
        min_q = torch.min(q1, q2)

        # Expected cost under the current masked policy.
        expected_cost = torch.sum(probs * cost_pred, dim=1)

        # Expected reward value under the current masked policy.
        expected_q = torch.sum(probs * min_q, dim=1)

        # Policy entropy. Higher entropy = more spread-out policy.
        entropy = -torch.sum(probs * log_probs, dim=1)

        # Expert-regularisation term. Measures how far LAADAN policy is from expert-safe policy. Lower is closer to expert.
        expert_kl = torch.sum(expert * (torch.log(expert + 1e-8) - log_probs), dim=1)

        # Smoothness proxy term. Higher means action choice is farther from expert-supported mean action.
        expected_smoothness = torch.sum(probs * smoothness_cost, dim=1)

        # Actor objective:
        # maximise reward while discouraging cost, expert deviation, and rough
        # action behaviour.
        actor_loss = torch.mean(
            -expected_q # maximise reward/survival value
            - entropy_coef * entropy # encourage some policy spread/exploration
            + lagrange_value * expected_cost # penalise unsafe expected cost
            + expert_kl_weight * expert_kl # stay close to expert policy
            + smoothness_weight * expected_smoothness # avoid rough/unusual action choices
        )

        # Backpropagate actor objective.
        actor_loss.backward()

        # Gradient clipping for actor stability.
        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)

        # Actor step.
        actor_optimizer.step()

        # Update actor weights.

        # Update target network.
        soft_update(target_model, model, tau)

        # Keeping target model in eval mode.
        target_model.eval()

        # Updating the Lagrange multiplier with projected gradient ascent.
        mean_expected_cost = float(torch.mean(expected_cost).item())
        lagrange_value = max(0.0, lagrange_value + lagrange_lr * (mean_expected_cost - cost_budget))

        # Training row.
        row = {
            "epoch": epoch,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "cost_loss": float(cost_loss.item()),
            "expected_cost": mean_expected_cost,
            "lagrange": float(lagrange_value),
            "expert_kl": float(torch.mean(expert_kl).item()),
            "smoothness": float(torch.mean(expected_smoothness).item()),
        }

        # Periodic evaluation.
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x)["logits"]
                greedy_policy = masked_greedy_policy_from_logits(logits_eval, admissible)
                soft_policy = masked_softmax_from_logits(logits_eval, admissible)

            metrics = evaluate_policy_set(
                benchmark,
                policy_numpy(greedy_policy),
                soft_policy=policy_numpy(soft_policy),
            )

            row.update(metrics)

            # Saving best checkpoint if survival improves.
            if metrics["survival_rate"] > best_survival:
                best_survival = metrics["survival_rate"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                best_metrics = metrics

        # Storing row in history.
        history.append(row)

    # Restoring best checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final policy extraction.
    model.eval()
    with torch.no_grad():
        logits_final = model(x)["logits"]
        final_policy = masked_greedy_policy_from_logits(logits_final, admissible)
        final_soft_policy = masked_softmax_from_logits(logits_final, admissible)

    # Fallback metric computation if none was stored.
    if best_metrics is None:
        best_metrics = evaluate_policy_set(
            benchmark,
            policy_numpy(final_policy),
            soft_policy=policy_numpy(final_soft_policy),
        )

    # Save the run.
    seed_dir = os.path.join(results_dir, "laadan_ac", "seed_" + str(seed))
    save_model_run(seed_dir, model, history, best_metrics)

    # Returning structured result dictionary.
    return {
        "name": "LAADAN-AC",
        "seed": seed,
        "model": model,
        "policy": policy_numpy(final_policy),
        "analysis_policy": policy_numpy(final_soft_policy),
        "history": history,
        "metrics": best_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }


def aggregate_seed_metrics(run_list):

    """
    Aggregate per-seed metrics into mean/std/95% CI/min/max summaries.

    This is the main multi-seed summary used for tables and bar charts.
    """

    # If no runs exist, return an empty summary.
    if not run_list:
        return {}

    # Taking metric names from the first run's metric dictionary.
    metric_names = list(run_list[0]["metrics"].keys())

    # Preparing summary output.
    summary = {}

    # Aggregating each metric separately across seeds.
    for metric_name in metric_names:
        values = np.asarray([run["metrics"][metric_name] for run in run_list], dtype=float)
        stats = mean_ci95(values)
        stats["min"] = float(np.min(values))
        stats["max"] = float(np.max(values))
        summary[metric_name] = stats

    # Aggregating wall-clock training times.
    times = np.asarray([run["train_time_seconds"] for run in run_list], dtype=float)
    time_stats = mean_ci95(times)
    time_stats["min"] = float(np.min(times))
    time_stats["max"] = float(np.max(times))
    summary["train_time_seconds"] = time_stats

    # Aggregating convergence epochs when available.
    convergence_values = [
        run["convergence_epoch_95"]
        for run in run_list
        if run["convergence_epoch_95"] is not None
    ]
    if convergence_values:
        convergence_values = np.asarray(convergence_values, dtype=float)
        conv_stats = mean_ci95(convergence_values)
        conv_stats["min"] = float(np.min(convergence_values))
        conv_stats["max"] = float(np.max(convergence_values))
        summary["convergence_epoch_95"] = conv_stats

    # Returning aggregated summary.
    return summary


def aggregate_histories(run_list, metric_names):

    """
    Building mean/std/95% CI history curves across seeds for selected metrics.

    Missing values remain NaN so plotting code can skip them cleanly.
    """

    # If there are no runs, return an empty result.
    if not run_list:
        return {}

    # Output structure.
    result = {}

    # Epoch indices are taken from the first run.
    epochs = np.asarray([row["epoch"] for row in run_list[0]["history"]], dtype=int)
    result["epoch"] = epochs.tolist()

    # Process each requested metric separately.
    for metric_name in metric_names:
        stacked = []

        # Building a seed-by-epoch matrix for this metric.
        for run in run_list:
            values = []
            for row in run["history"]:
                value = row.get(metric_name, np.nan)
                values.append(float(value) if np.isfinite(value) else np.nan)
            stacked.append(values)

        # Converting to NumPy for column-wise aggregation.
        stacked = np.asarray(stacked, dtype=float)

        # If no values exist, return empty arrays for this metric.
        if stacked.size == 0:
            result[metric_name] = {
                "mean": [],
                "std": [],
                "ci95_half": [],
                "ci95_low": [],
                "ci95_high": [],
                "n": [],
            }
            continue

        # Preparing lists for per-epoch summaries.
        mean_values = []
        std_values = []
        ci95_half_values = []
        ci95_low_values = []
        ci95_high_values = []
        n_values = []

        # Aggregating each epoch column independently across seeds.
        for column_index in range(stacked.shape[1]):
            stats = mean_ci95(stacked[:, column_index])
            mean_values.append(stats["mean"])
            std_values.append(stats["std"])
            ci95_half_values.append(stats["ci95_half"])
            ci95_low_values.append(stats["ci95_low"])
            ci95_high_values.append(stats["ci95_high"])
            n_values.append(stats["n"])

        # Saving aggregated history arrays for this metric.
        result[metric_name] = {
            "mean": mean_values,
            "std": std_values,
            "ci95_half": ci95_half_values,
            "ci95_low": ci95_low_values,
            "ci95_high": ci95_high_values,
            "n": n_values,
        }

    # Returning the full aggregated history bundle.
    return result


def best_policy_from_runs(run_list, use_analysis_policy=False):

    """
    Returning the policy from the seed with the highest survival rate.
    """
    # If run list is empty, no best policy exists.
    if not run_list:
        return None

    # Starting by assuming the first run is best.
    best_index = 0
    best_value = run_list[0]["metrics"]["survival_rate"]

    # Searching for the run with highest survival.
    for index in range(1, len(run_list)):
        value = run_list[index]["metrics"]["survival_rate"]
        if value > best_value:
            best_value = value
            best_index = index

    # Returning either the auxiliary soft policy or the primary policy.
    if use_analysis_policy:
        return run_list[best_index].get("analysis_policy", run_list[best_index]["policy"])
    return run_list[best_index]["policy"]


