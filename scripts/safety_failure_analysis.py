# safety_failure_analysis.py

# This script performs the safety-failure analysis for the
# ICU-Sepsis LAADAN-AC.

# Libraries used in this script:

# Importing argparse so the script can receive command-line arguments such as
# --data-dir, --results-dir, --device and --num-trajectories.
import argparse
# Importing csv so the script can write editable table files for the paper.
import csv

# Importing json so the script can read run_config.json and write the final
# manifest file listing all generated outputs.
import json

# Importing os so the script can build file paths, check whether files exist,
# and create output folders.
import os

# Importing matplotlib so the script can create the analysis figure and save
# it as a high-resolution PNG.
import matplotlib.pyplot as plt

# Importing NumPy for arrays, softmax calculations, PCA projection, trajectory
# sampling, and table calculations.
import numpy as np

# Importing PyTorch so final saved model checkpoints can be loaded and evaluated.
import torch

from trainers import (
    greedy_policy_from_logits,
    masked_greedy_policy_from_logits,
    plain_softmax_from_logits,
    masked_softmax_from_logits,
    policy_numpy,
)

# Importing the ICU-Sepsis benchmark wrapper used by the project.
# This gives access to state features, transition probabilities, admissibility
# masks, expert-safe policy, rewards and terminal states.
from benchmark import ICUSepsisOfflineBenchmark

# Importing the exact model classes used during training.
# These are needed so the saved model.pt files can be loaded into matching
# architectures before analysis.
from models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet


# Setting the figure resolution.
PLOT_DPI = 600

# Setting a tiny numerical constant used to avoid division-by-zero in softmax and
# trajectory probability normalisation.
EPS = 1e-8

# Defining a fixed colour palette so plots remain visually consistent with the
# rest of the project figures.
PALETTE = {
    "Behavior Cloning": "#6baed6",
    "Conservative Q-Learning": "#08519c",
    "Vanilla Offline Actor-Critic": "#9ecae1",
    "Post-hoc Masked VOAC": "#4292c6",
    "LAADAN-AC": "#2171b5",
}


def ensure_dir(path):
    """
    Create a folder if it does not already exist.
    """

    # Checking whether the requested folder already exists.
    if not os.path.exists(path):

        # Creating the folder when it is missing.
        os.makedirs(path)


def choose_device(requested):
    """
    Select CPU or CUDA in the same style as the existing project scripts.
    """

    # If the user explicitly requested CPU, use CPU.
    if requested == "cpu":
        return "cpu"

    # If the user explicitly requested CUDA, check that CUDA is available.
    if requested == "cuda":

        # Raising a clear error if CUDA was requested but no GPU is available.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        # Returning CUDA when it was requested and available.
        return "cuda"

    # If the user selected auto, use CUDA when possible and otherwise fall back
    # to CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_json_if_exists(path):
    """
    Load a JSON file if it exists, otherwise return an empty dictionary.
    """

    # Checking whether the JSON file exists.
    if not os.path.exists(path):

        # Returning an empty dictionary keeps the script usable even if
        # run_config.json is missing.
        return {}

    # Opening the JSON file using UTF-8 encoding.
    with open(path, "r", encoding="utf-8") as handle:

        # Loading and returning the JSON content as a Python dictionary.
        return json.load(handle)


def get_config_block(config, name):
    """
    Read architecture settings from run_config.json with safe defaults.
    """

    # Reading the configuration block for one model family.
    #
    # Example:
    # config["voac"]
    block = config.get(name, {})

    # Returning the model architecture settings.
    #
    # If the config file does not contain these values, the same default values
    # used by the project are used here.
    return {
        "hidden_dim": int(block.get("hidden_dim", 128)),
        "latent_dim": int(block.get("latent_dim", 128)),
        "dropout": float(block.get("dropout", 0.10)),
    }


def find_model_path(final_models_dir, candidates):
    """
    Find the first existing model checkpoint for a list of possible folder names.
    """

    # Looping through possible folder names because some outputs may use slightly
    # different naming conventions, for example "laadan_ac" or "laadan-ac".
    for folder in candidates:

        # Building the expected checkpoint path for the current candidate folder.
        path = os.path.join(final_models_dir, folder, "model.pt")

        # Returning the first checkpoint path that actually exists.
        if os.path.exists(path):
            return path, folder

    # Returning None values when none of the candidate folders contain model.pt.
    return None, None


def unpack_state_dict(loaded_object):
    """
    Extract a PyTorch state_dict from common checkpoint formats.
    """

    # Checking whether the loaded checkpoint is a dictionary.
    if isinstance(loaded_object, dict):

        # Some training scripts save weights inside a nested key.
        # This loop checks the most common names used for that nested dictionary.
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:

            # Returning the nested state dictionary when found.
            if key in loaded_object and isinstance(loaded_object[key], dict):
                return loaded_object[key]

        # If the dictionary itself maps parameter names directly to tensors, then
        # it is already a raw PyTorch state_dict.
        raw_state = True

        # Checking every value to confirm that all values are tensors.
        for value in loaded_object.values():
            if not torch.is_tensor(value):
                raw_state = False
                break

        # Returning the object directly when it is already a raw state_dict.
        if raw_state:
            return loaded_object

    # Raising a clear error when the checkpoint format is not recognised.
    raise ValueError("Could not find model weights inside checkpoint.")


def clean_state_dict_keys(state_dict):
    """
    Remove DataParallel 'module.' prefixes if present.
    """

    # Creating a new dictionary for cleaned parameter names.
    cleaned = {}

    # Looping through every saved parameter name and tensor.
    for key, value in state_dict.items():

        # Removing the "module." prefix if the model was saved from DataParallel.
        if key.startswith("module."):
            cleaned[key[7:]] = value

        # Keeping the key unchanged when no prefix exists.
        else:
            cleaned[key] = value

    # Returning the cleaned state dictionary.
    return cleaned


def load_checkpoint(model, model_path, device):
    """
    Load model weights and switch the model to evaluation mode.
    """

    # Loading the checkpoint onto the requested device.
    checkpoint = torch.load(model_path, map_location=device)

    # Extracting and cleaning the saved model weights.
    state_dict = clean_state_dict_keys(unpack_state_dict(checkpoint))

    # Loading the saved weights into the matching model architecture.
    model.load_state_dict(state_dict)

    # Switching the model to evaluation mode so dropout is disabled.
    model.eval()

    # Returning the loaded model.
    return model


def tensor_to_numpy(value):
    """
    Convert a tensor to a NumPy array.
    """

    # If the value is a PyTorch tensor, detach it from the graph, move it to CPU,
    # and convert it to NumPy.
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    # If the value is already array-like, convert it to a NumPy array.
    return np.asarray(value)


def softmax(scores):
    """
    Numerically stable softmax for NumPy arrays.
    """
    x = np.asarray(scores, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(np.sum(exp_x, axis=1, keepdims=True), EPS)


def masked_softmax(scores, mask):
    """
    Softmax restricted to benchmark-admissible actions.
    """
    safe_scores = np.where(mask > 0, scores, -1e9)
    return softmax(safe_scores)


def greedy_policy(scores):
    """
    Convert action scores to a deterministic one-hot greedy policy.
    """
    actions = np.argmax(scores, axis=1)
    policy = np.zeros_like(scores, dtype=np.float64)
    policy[np.arange(scores.shape[0]), actions] = 1.0
    return policy


def masked_greedy_policy(scores, mask):
    """
    Convert action scores to a deterministic policy after admissibility filtering.
    """
    safe_scores = np.where(mask > 0, scores, -1e9)
    return greedy_policy(safe_scores)


def policy_actions(policy):
    """
    Return the selected action index for each state.
    """
    return np.argmax(policy, axis=1)


def get_benchmark_array(benchmark, name):
    """
    Read a NumPy array from the benchmark object.
    """
    value = getattr(benchmark, name)
    return tensor_to_numpy(value)


def load_final_models(benchmark, final_models_dir, config, device):
    """
    Load the saved final checkpoints needed for the safety-failure analysis.
    """

    # Creating an empty dictionary to store the loaded model objects.

    # The keys will be readable model names and the values will be PyTorch models.
    loaded = {}

    # Looking for the final Behavior Cloning checkpoint.
    bc_path, _ = find_model_path(final_models_dir, ["bc"])

    # Loading Behavior Cloning only if the checkpoint exists.
    if bc_path is not None:
        cfg = get_config_block(config, "bc")
        model = BehaviorCloningNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        loaded["Behavior Cloning"] = load_checkpoint(model, bc_path, device)

    # Looking for the final Conservative Q-Learning checkpoint.
    cql_path, _ = find_model_path(final_models_dir, ["cql"])

    # Loading CQL only if the checkpoint exists.
    if cql_path is not None:
        cfg = get_config_block(config, "cql")
        model = ConservativeQNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        ).to(device)
        loaded["Conservative Q-Learning"] = load_checkpoint(model, cql_path, device)

    # Looking for the final Vanilla Offline Actor-Critic checkpoint.
    voac_path, _ = find_model_path(final_models_dir, ["voac"])

    # Loading VOAC only if the checkpoint exists.
    if voac_path is not None:
        cfg = get_config_block(config, "voac")
        model = OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
            use_cost_head=False,
        ).to(device)
        loaded["Vanilla Offline Actor-Critic"] = load_checkpoint(model, voac_path, device)

    # Looking for the final LAADAN-AC checkpoint.

    # Two folder-name options are checked because projects sometimes use either
    # an underscore or a hyphen.
    laadan_path, _ = find_model_path(final_models_dir, ["laadan_ac", "laadan-ac"])

    # Loading LAADAN-AC only if the checkpoint exists.
    if laadan_path is not None:
        cfg = get_config_block(config, "laadan_ac")
        model = OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
            use_cost_head=True,
        ).to(device)
        loaded["LAADAN-AC"] = load_checkpoint(model, laadan_path, device)

    return loaded


def build_policies(benchmark, models):
    """
    Build analysis policies from the loaded final model checkpoints.

    """
    policies = {}

    x = benchmark.state_features_t
    admissible = benchmark.admissible_mask_t

    for name, model in models.items():
        model.eval()

        with torch.no_grad():
            outputs = model(x)

            if name == "Conservative Q-Learning":
                q_values = outputs

                raw_primary = greedy_policy_from_logits(q_values)
                raw_soft = plain_softmax_from_logits(q_values)

                masked_primary = masked_greedy_policy_from_logits(q_values, admissible)
                masked_soft = masked_softmax_from_logits(q_values, admissible)

                policies[name] = {
                    "primary": policy_numpy(raw_primary),
                    "soft": policy_numpy(raw_soft),
                    "raw_primary": policy_numpy(raw_primary),
                    "raw_soft": policy_numpy(raw_soft),
                    "masked_primary": policy_numpy(masked_primary),
                    "masked_soft": policy_numpy(masked_soft),
                    "q_values": q_values.detach().cpu().numpy(),
                    "scores": q_values.detach().cpu().numpy(),
                }

            else:
                if isinstance(outputs, dict):
                    logits = outputs["logits"]

                    if outputs.get("q1") is not None and outputs.get("q2") is not None:
                        q_values = torch.min(outputs["q1"], outputs["q2"])
                    elif outputs.get("q1") is not None:
                        q_values = outputs["q1"]
                    else:
                        q_values = logits
                else:
                    logits = outputs
                    q_values = outputs

                raw_primary = greedy_policy_from_logits(logits)
                raw_soft = plain_softmax_from_logits(logits)

                masked_primary = masked_greedy_policy_from_logits(logits, admissible)
                masked_soft = masked_softmax_from_logits(logits, admissible)

                if name == "LAADAN-AC":
                    primary = masked_primary
                    soft = masked_soft
                else:
                    primary = raw_primary
                    soft = raw_soft

                policies[name] = {
                    "primary": policy_numpy(primary),
                    "soft": policy_numpy(soft),
                    "raw_primary": policy_numpy(raw_primary),
                    "raw_soft": policy_numpy(raw_soft),
                    "masked_primary": policy_numpy(masked_primary),
                    "masked_soft": policy_numpy(masked_soft),
                    "q_values": q_values.detach().cpu().numpy(),
                    "scores": logits.detach().cpu().numpy(),
                }

    if "Vanilla Offline Actor-Critic" in policies:
        voac_payload = policies["Vanilla Offline Actor-Critic"]

        policies["Post-hoc Masked VOAC"] = {
            "primary": voac_payload["masked_primary"],
            "soft": voac_payload["masked_soft"],
            "raw_primary": voac_payload["raw_primary"],
            "raw_soft": voac_payload["raw_soft"],
            "masked_primary": voac_payload["masked_primary"],
            "masked_soft": voac_payload["masked_soft"],
            "q_values": voac_payload["q_values"],
            "scores": voac_payload["scores"],
        }

    return policies

def per_state_inadmissibility(policy, admissible_mask):
    """
    Compute inadmissibility probability for each state under a policy.
    """
    cost = 1.0 - np.asarray(admissible_mask, dtype=np.float64)
    return np.sum(np.asarray(policy, dtype=np.float64) * cost, axis=1)


def pca_two_components(features):
    """
    Compute a two-component PCA projection using only NumPy.
    """
    x = np.asarray(features, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def get_terminal_mask(benchmark):
    """
    Return the benchmark terminal-state mask as a Boolean NumPy array.
    """
    if hasattr(benchmark, "terminal_mask"):
        return np.asarray(benchmark.terminal_mask, dtype=bool)
    return tensor_to_numpy(benchmark.terminal_mask_t).astype(bool)


def get_expert_policy(benchmark):
    """
    Return the expert-safe policy as a NumPy array.
    """
    if hasattr(benchmark, "expert_safe"):
        return np.asarray(benchmark.expert_safe, dtype=np.float64)
    return tensor_to_numpy(benchmark.expert_safe_t).astype(np.float64)


def write_state_table(path, benchmark, policies):
    """
    Save state-level inadmissibility and selected-action information.
    """

    # Reading the admissibility mask.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    # Reading which states are terminal states.
    terminal = get_terminal_mask(benchmark)

    # Reading the benchmark expert-safe policy.
    expert = get_expert_policy(benchmark)

    # Converting the expert policy into one selected action per state.
    expert_actions = policy_actions(expert)

    columns = ["state_id", "terminal", "expert_action"]
    for name in policies.keys():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        columns += [safe_name + "_action", safe_name + "_inadmissibility"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for state_id in range(benchmark.num_states):
            row = {
                "state_id": state_id,
                "terminal": int(terminal[state_id]),
                "expert_action": int(expert_actions[state_id]),
            }
            for name, payload in policies.items():
                safe_name = name.lower().replace(" ", "_").replace("-", "_")
                policy = payload["primary"]
                row[safe_name + "_action"] = int(policy_actions(policy)[state_id])
                row[safe_name + "_inadmissibility"] = float(per_state_inadmissibility(policy, mask)[state_id])
            writer.writerow(row)


def write_feature_distribution_table(path, benchmark, voac_policy):
    """
    Compare 47-dimensional feature means for VOAC-safe and VOAC-unsafe states.
    """

    # Reading the 47-dimensional state feature matrix.
    features = get_benchmark_array(benchmark, "state_features")

    # Reading the admissibility mask.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    # Reading terminal-state flags so terminal states can be excluded from this
    # feature-distribution comparison.
    terminal = get_terminal_mask(benchmark)

    # Computing one inadmissibility value per state under VOAC.
    state_cost = per_state_inadmissibility(voac_policy, mask)

    unsafe = (state_cost > 0.5) & (~terminal)
    safe = (state_cost <= 0.5) & (~terminal)

    if int(np.sum(unsafe)) == 0:
        threshold = np.quantile(state_cost[~terminal], 0.90)
        unsafe = (state_cost >= threshold) & (~terminal)
        safe = (state_cost < threshold) & (~terminal)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_index",
                "safe_state_mean",
                "unsafe_state_mean",
                "difference_unsafe_minus_safe",
                "absolute_difference",
            ],
        )
        writer.writeheader()

        for index in range(features.shape[1]):
            safe_mean = float(np.mean(features[safe, index])) if np.any(safe) else float("nan")
            unsafe_mean = float(np.mean(features[unsafe, index])) if np.any(unsafe) else float("nan")
            diff = unsafe_mean - safe_mean
            writer.writerow({
                "feature_index": index,
                "safe_state_mean": safe_mean,
                "unsafe_state_mean": unsafe_mean,
                "difference_unsafe_minus_safe": diff,
                "absolute_difference": abs(diff),
            })


def write_action_tables(action_path, confusion_path, benchmark, policies):
    """
    Save action-level inadmissibility counts and expert/VOAC/LAADAN confusion counts.
    """

    # Reading the admissibility mask.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    # Reading terminal-state flags.
    terminal = get_terminal_mask(benchmark)

    # Reading and converting the expert-safe policy to selected actions.
    expert = get_expert_policy(benchmark)
    expert_actions = policy_actions(expert)

    with open(action_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "action", "selected_count", "inadmissible_selected_count", "inadmissible_selected_rate"],
        )
        writer.writeheader()

        for name, payload in policies.items():
            actions = policy_actions(payload["primary"])
            for action in range(benchmark.num_actions):
                selected = (actions == action) & (~terminal)
                inad_selected = selected & (mask[:, action] < 0.5)
                selected_count = int(np.sum(selected))
                writer.writerow({
                    "model": name,
                    "action": action,
                    "selected_count": selected_count,
                    "inadmissible_selected_count": int(np.sum(inad_selected)),
                    "inadmissible_selected_rate": float(np.sum(inad_selected) / max(1, selected_count)),
                })

    if "Vanilla Offline Actor-Critic" not in policies or "LAADAN-AC" not in policies:
        return

    voac_actions = policy_actions(policies["Vanilla Offline Actor-Critic"]["primary"])
    laadan_actions = policy_actions(policies["LAADAN-AC"]["primary"])
    counts = {}

    for state_id in range(benchmark.num_states):
        if terminal[state_id]:
            continue
        key = (int(expert_actions[state_id]), int(voac_actions[state_id]), int(laadan_actions[state_id]))
        counts[key] = counts.get(key, 0) + 1

    with open(confusion_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["expert_action", "voac_action", "laadan_action", "state_count"])
        writer.writeheader()
        for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
            writer.writerow({
                "expert_action": key[0],
                "voac_action": key[1],
                "laadan_action": key[2],
                "state_count": count,
            })


def representative_unsafe_states(benchmark, voac_policy, k):
    """
    Pick representative non-terminal states where VOAC selects inadmissible actions.
    """

    # Reading the admissibility mask.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    # Reading terminal-state flags.
    terminal = get_terminal_mask(benchmark)

    # Computing the per-state VOAC inadmissibility score.
    state_cost = per_state_inadmissibility(voac_policy, mask)

    order = np.argsort(-state_cost)
    selected = []
    for state_id in order:
        if terminal[state_id]:
            continue
        if state_cost[state_id] <= 0:
            continue
        selected.append(int(state_id))
        if len(selected) >= k:
            break

    if not selected:
        non_terminal_ids = np.where(~terminal)[0]
        selected = [int(x) for x in non_terminal_ids[:k]]

    return selected


def write_q_value_table(path, benchmark, policies, state_ids):
    """
    Save Q-value rows for representative states and all action bins.
    """

    # Reading the admissibility mask so each action can be labelled as admissible
    # or inadmissible in each representative state.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "state_id", "action", "admissible", "q_value"])
        writer.writeheader()

        for name, payload in policies.items():
            if "q_values" not in payload or payload["q_values"] is None:
                continue
            q_values = payload["q_values"]
            for state_id in state_ids:
                for action in range(benchmark.num_actions):
                    writer.writerow({
                        "model": name,
                        "state_id": int(state_id),
                        "action": action,
                        "admissible": int(mask[state_id, action] > 0.5),
                        "q_value": float(q_values[state_id, action]),
                    })


def transition_array(benchmark):
    """
    Return transition probabilities as a NumPy array.
    """
    if hasattr(benchmark, "transition"):
        return np.asarray(benchmark.transition, dtype=np.float64)
    return tensor_to_numpy(benchmark.transition_t).astype(np.float64)


def initial_state_distribution(benchmark):
    """
    Return the initial-state distribution as a NumPy array.
    """
    if hasattr(benchmark, "initial_state_dist"):
        return np.asarray(benchmark.initial_state_dist, dtype=np.float64)
    return tensor_to_numpy(benchmark.initial_state_dist_t).astype(np.float64)


def expected_reward_table(benchmark):
    """
    Return expected reward per state-action pair if available.
    """
    if hasattr(benchmark, "reward_sa"):
        return np.asarray(benchmark.reward_sa, dtype=np.float64)
    if hasattr(benchmark, "reward_sa_t"):
        return tensor_to_numpy(benchmark.reward_sa_t).astype(np.float64)
    return np.zeros((benchmark.num_states, benchmark.num_actions), dtype=np.float64)


def sample_trajectory(benchmark, policy, seed, initial_state=None):
    """
    Sample one trajectory from the benchmark MDP under a fixed policy.
    """

    # Creating a reproducible NumPy random generator.
    rng = np.random.default_rng(seed)

    # Reading transition probabilities from the benchmark.
    transition = transition_array(benchmark)

    # Reading the initial-state distribution.
    initial = initial_state_distribution(benchmark)

    # Reading terminal-state flags.
    terminal = get_terminal_mask(benchmark)

    # Reading the admissibility mask.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    # Reading the expected reward for each state-action pair.
    reward_sa = expected_reward_table(benchmark)

    if initial_state is None:
        state = int(rng.choice(benchmark.num_states, p=initial / np.sum(initial)))
    else:
        state = int(initial_state)

    rows = []
    for step in range(int(benchmark.horizon)):
        if terminal[state]:
            break

        action = int(rng.choice(benchmark.num_actions, p=policy[state] / np.sum(policy[state])))
        probs = transition[state, action]
        probs = probs / max(EPS, float(np.sum(probs)))
        next_state = int(rng.choice(benchmark.num_states, p=probs))

        rows.append({
            "step": step,
            "state": state,
            "action": action,
            "admissible": int(mask[state, action] > 0.5),
            "reward": float(reward_sa[state, action]),
            "next_state": next_state,
        })

        state = next_state

    return rows


def find_voac_failure_trajectory(benchmark, voac_policy, seed, attempts):
    """
    Find a sampled VOAC trajectory containing at least one inadmissible action.
    """
    rng = np.random.default_rng(seed)
    initial = initial_state_distribution(benchmark)

    for attempt in range(attempts):
        initial_state = int(rng.choice(benchmark.num_states, p=initial / np.sum(initial)))
        rows = sample_trajectory(benchmark, voac_policy, seed + attempt + 1, initial_state=initial_state)
        for row in rows:
            if int(row["admissible"]) == 0:
                return initial_state, rows

    initial_state = int(np.argmax(initial))
    return initial_state, sample_trajectory(benchmark, voac_policy, seed + 999, initial_state=initial_state)


def write_trajectory_tables(trajectory_path, timepoint_path, benchmark, policies, seed, attempts):
    """
    Save representative trajectory rows and timepoint-level inadmissibility counts.
    """

    # If either VOAC or LAADAN-AC is missing, this trajectory comparison cannot
    # be created.
    if "Vanilla Offline Actor-Critic" not in policies or "LAADAN-AC" not in policies:
        return None

    voac_policy = policies["Vanilla Offline Actor-Critic"]["primary"]
    laadan_policy = policies["LAADAN-AC"]["primary"]
    initial_state, voac_rows = find_voac_failure_trajectory(benchmark, voac_policy, seed, attempts)
    laadan_rows = sample_trajectory(benchmark, laadan_policy, seed + 5000, initial_state=initial_state)

    with open(trajectory_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "initial_state", "step", "state", "action", "admissible", "reward", "next_state"],
        )
        writer.writeheader()
        for model_name, rows in [("Vanilla Offline Actor-Critic", voac_rows), ("LAADAN-AC", laadan_rows)]:
            for row in rows:
                out = dict(row)
                out["model"] = model_name
                out["initial_state"] = int(initial_state)
                writer.writerow(out)

    time_counts = {}
    for model_name, policy in [("Vanilla Offline Actor-Critic", voac_policy), ("LAADAN-AC", laadan_policy)]:
        for episode in range(attempts):
            rows = sample_trajectory(benchmark, policy, seed + 10000 + episode, initial_state=None)
            for row in rows:
                key = (model_name, int(row["step"]))
                if key not in time_counts:
                    time_counts[key] = {"visited": 0, "inadmissible": 0}
                time_counts[key]["visited"] += 1
                if int(row["admissible"]) == 0:
                    time_counts[key]["inadmissible"] += 1

    with open(timepoint_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "step", "visited_count", "inadmissible_count", "inadmissible_rate"])
        writer.writeheader()
        for key, counts in sorted(time_counts.items()):
            writer.writerow({
                "model": key[0],
                "step": key[1],
                "visited_count": counts["visited"],
                "inadmissible_count": counts["inadmissible"],
                "inadmissible_rate": float(counts["inadmissible"] / max(1, counts["visited"])),
            })

    return initial_state, voac_rows, laadan_rows


def plot_four_panel_figure(output_path, benchmark, policies, representative_states, trajectory_payload):
    """
    Create the four-panel figure requested for the manuscript.
    """

    # Reading the benchmark admissibility mask for all states and actions.
    mask = get_benchmark_array(benchmark, "admissible_mask")

    # Reading the 47-dimensional state feature matrix.
    features = get_benchmark_array(benchmark, "state_features")

    # Reading terminal-state flags so the PCA panel focuses on non-terminal states.
    terminal = get_terminal_mask(benchmark)

    voac_policy = policies["Vanilla Offline Actor-Critic"]["primary"]
    laadan_policy = policies["LAADAN-AC"]["primary"]
    masked_voac_policy = policies.get("Post-hoc Masked VOAC", policies["LAADAN-AC"])["primary"]

    voac_cost = per_state_inadmissibility(voac_policy, mask)
    laadan_cost = per_state_inadmissibility(laadan_policy, mask)
    masked_voac_cost = per_state_inadmissibility(masked_voac_policy, mask)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    heatmap_states = representative_states[:15]
    heatmap = np.vstack([
        voac_cost[heatmap_states],
        masked_voac_cost[heatmap_states],
        laadan_cost[heatmap_states],
    ])
    image = axes[0, 0].imshow(heatmap, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("(a) Per-state inadmissibility")
    axes[0, 0].set_yticks([0, 1, 2])
    axes[0, 0].set_yticklabels(["VOAC", "Masked VOAC", "LAADAN-AC"])
    axes[0, 0].set_xticks(np.arange(len(heatmap_states)))
    axes[0, 0].set_xticklabels(heatmap_states, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_xlabel("Representative state id")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.88, label="Inadmissibility")

    coords = pca_two_components(features)
    non_terminal = ~terminal
    scatter = axes[0, 1].scatter(
        coords[non_terminal, 0],
        coords[non_terminal, 1],
        c=voac_cost[non_terminal],
        cmap="Blues",
        s=22,
        alpha=0.90,
        edgecolors="none",
    )
    axes[0, 1].set_title("(b) PCA state projection coloured by VOAC violation")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].grid(True, alpha=0.20)
    fig.colorbar(scatter, ax=axes[0, 1], shrink=0.88, label="VOAC inadmissibility")

    state_for_q = representative_states[0]
    q_labels = []
    q_values = []
    for model_name in ["Vanilla Offline Actor-Critic", "Conservative Q-Learning", "LAADAN-AC"]:
        if model_name not in policies or "q_values" not in policies[model_name]:
            continue
        q = policies[model_name]["q_values"][state_for_q]
        adm = mask[state_for_q] > 0.5
        q_labels += [model_name.replace("Vanilla Offline Actor-Critic", "VOAC") + " adm"]
        q_values += [q[adm]]
        if np.any(~adm):
            q_labels += [model_name.replace("Vanilla Offline Actor-Critic", "VOAC") + " inad"]
            q_values += [q[~adm]]

    if q_values:
        axes[1, 0].boxplot(q_values, labels=q_labels, showfliers=False)
        axes[1, 0].tick_params(axis="x", rotation=25, labelsize=8)
        axes[1, 0].set_ylabel("Q value")
        axes[1, 0].set_title("(c) Q-values for admissible and inadmissible actions, state " + str(state_for_q))
        axes[1, 0].grid(True, axis="y", alpha=0.25)
    else:
        axes[1, 0].text(0.5, 0.5, "Q-values unavailable", ha="center", va="center")
        axes[1, 0].set_axis_off()

    if trajectory_payload is not None:
        _, voac_rows, laadan_rows = trajectory_payload
        for model_name, rows in [("VOAC", voac_rows), ("LAADAN-AC", laadan_rows)]:
            steps = [row["step"] for row in rows]
            actions = [row["action"] for row in rows]
            color = PALETTE["Vanilla Offline Actor-Critic"] if model_name == "VOAC" else PALETTE["LAADAN-AC"]
            axes[1, 1].plot(steps, actions, marker="o", linewidth=2.0, label=model_name, color=color)
            bad_steps = [row["step"] for row in rows if int(row["admissible"]) == 0]
            bad_actions = [row["action"] for row in rows if int(row["admissible"]) == 0]
            if bad_steps:
                axes[1, 1].scatter(bad_steps, bad_actions, s=70, marker="x", color="#08306b", label=model_name + " inadmissible")

        axes[1, 1].set_title("(d) Sampled action sequence")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Action bin")
        axes[1, 1].grid(True, alpha=0.25)
        axes[1, 1].legend(frameon=False, fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "Trajectory unavailable", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.suptitle("Safety-failure analysis of unrestricted and admissibility-aware policies", fontsize=14)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path, files):
    """
    Save a manifest of generated analysis outputs.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(files, handle, indent=2)


def parse_args():
    """
    Read command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Safety-failure analysis for LAADAN-AC figures.")
    parser.add_argument("--data-dir", required=True, help="Path to ICU-Sepsis benchmark files.")
    parser.add_argument("--results-dir", default="results", help="Root results directory.")
    parser.add_argument("--final-models-dir", default="", help="Folder containing final model subfolders. Defaults to results/final_models.")
    parser.add_argument("--output-dir", default="", help="Output folder. Defaults to results/safety_failure_analysis.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-trajectories", type=int, default=2000, help="Number of trajectories for timepoint analysis.")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    """
    Run the full safety-failure analysis without retraining.
    """

    # Reading command-line arguments.
    args = parse_args()

    # Selecting the device requested by the user.
    device = choose_device(args.device)

    final_models_dir = args.final_models_dir
    if final_models_dir == "":
        final_models_dir = os.path.join(args.results_dir, "final_models")

    output_dir = args.output_dir
    if output_dir == "":
        output_dir = os.path.join(args.results_dir, "safety_failure_analysis")

    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    config = read_json_if_exists(os.path.join(args.results_dir, "run_config.json"))

    print("[INFO] Device:", device, flush=True)
    print("[INFO] Final models directory:", final_models_dir, flush=True)
    print("[INFO] Output directory:", output_dir, flush=True)
    print("[INFO] Number of trajectories:", args.num_trajectories, flush=True)

    print("[STEP 1] Loading ICU-Sepsis benchmark...", flush=True)
    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=50,
        device=device,
        use_one_hot_states=False,
    )

    print("[STEP 2] Loading final model checkpoints...", flush=True)
    models = load_final_models(benchmark, final_models_dir, config, device)
    required = ["Vanilla Offline Actor-Critic", "LAADAN-AC"]
    for name in required:
        if name not in models:
            raise FileNotFoundError("Required final model was not loaded: " + name)

    print("[STEP 3] Building policies from loaded models...", flush=True)
    policies = build_policies(benchmark, models)

    print("[STEP 4] Selecting representative unsafe VOAC states...", flush=True)
    representative_states = representative_unsafe_states(
        benchmark,
        policies["Vanilla Offline Actor-Critic"]["primary"],
        k=20,
    )

    files = {}

    print("[STEP 5] Writing state-level inadmissibility table...", flush=True)
    files["state_inadmissibility_csv"] = os.path.join(tables_dir, "state_level_inadmissibility.csv")
    write_state_table(files["state_inadmissibility_csv"], benchmark, policies)

    print("[STEP 6] Writing safe-vs-unsafe feature table...", flush=True)
    files["feature_distribution_csv"] = os.path.join(tables_dir, "safe_vs_unsafe_feature_distribution.csv")
    write_feature_distribution_table(
        files["feature_distribution_csv"],
        benchmark,
        policies["Vanilla Offline Actor-Critic"]["primary"],
    )

    print("[STEP 7] Writing action-level and confusion tables...", flush=True)
    files["action_counts_csv"] = os.path.join(tables_dir, "action_level_inadmissibility_counts.csv")
    files["action_confusion_csv"] = os.path.join(tables_dir, "expert_voac_laadan_action_confusion.csv")
    write_action_tables(files["action_counts_csv"], files["action_confusion_csv"], benchmark, policies)

    print("[STEP 8] Writing representative Q-value table...", flush=True)
    files["q_values_csv"] = os.path.join(tables_dir, "representative_state_q_values.csv")
    write_q_value_table(files["q_values_csv"], benchmark, policies, representative_states[:8])

    print("[STEP 9] Sampling trajectories and writing trajectory tables...", flush=True)
    files["trajectory_csv"] = os.path.join(tables_dir, "sample_trajectory_voac_vs_laadan.csv")
    files["timepoint_csv"] = os.path.join(tables_dir, "trajectory_timepoint_inadmissibility.csv")
    trajectory_payload = write_trajectory_tables(
        files["trajectory_csv"],
        files["timepoint_csv"],
        benchmark,
        policies,
        seed=int(args.seed),
        attempts=int(args.num_trajectories),
    )

    print("[STEP 10] Creating four-panel safety-failure figure at 600 dpi...", flush=True)
    files["four_panel_figure_png"] = os.path.join(figures_dir, "safety_failure_four_panel_analysis.png")
    plot_four_panel_figure(files["four_panel_figure_png"], benchmark, policies, representative_states, trajectory_payload)

    print("[STEP 11] Writing manifest...", flush=True)
    files["manifest_json"] = os.path.join(output_dir, "safety_failure_analysis_manifest.json")
    write_manifest(files["manifest_json"], files)

    print("[DONE] Safety-failure analysis saved to:", output_dir, flush=True)
    for key in sorted(files.keys()):
        print("[FILE]", key + ":", files[key], flush=True)
        
# Running the script only when this file is executed directly.
if __name__ == "__main__":
    main()