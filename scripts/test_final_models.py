# test_final_models.py


# tester for the final ICU-Sepsis, eicu and langrange frontier models.

# Libraries used in this file:

# argparse is used to read command-line arguments for data paths, device, output
# file location, and numerical tolerances.
import argparse

# used to read saved configuration/metrics files and to write the final
# testing summary.
import json

# used for file/folder existence checks and path construction.
import os

# NumPy is used for array handling, masking, numeric checks, and converting
# tensors into CPU-based arrays for evaluation.
import numpy as np

#  used to load checkpoints, move models to the correct device,
# and perform forward passes during evaluation.
import torch

# Import the benchmark loader/evaluator used during training.
from scripts.benchmark import ICUSepsisOfflineBenchmark

# Import the three model-building classes needed to reconstruct the saved models.
from scripts.models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet


# AGENTS defines the four model families that this script expects to test.
AGENTS = [
    {"folder": "bc", "name": "Behavior Cloning", "config_key": "bc", "kind": "bc"},
    {"folder": "cql", "name": "Conservative Q-Learning", "config_key": "cql", "kind": "cql"},
    {"folder": "voac", "name": "Vanilla Offline Actor-Critic", "config_key": "voac", "kind": "voac"},
    {"folder": "laadan_ac", "name": "LAADAN-AC", "config_key": "laadan_ac", "kind": "laadan"},
]

# DEFAULT_ARCH acts as a fallback architecture in case run_config.json is
# missing a field. This improves robustness and prevents fragile failures.
DEFAULT_ARCH = {
    "hidden_dim": 128,
    "latent_dim": 128,
    "dropout": 0.10,
}

# CORE_KEYS are the most important benchmark metrics.

CORE_KEYS = [
    "avg_return",
    "survival_rate",
    "mortality_rate",
    "avg_length",
    "inadmissibility_rate",
]

# AUX_KEYS are extra policy-analysis metrics.

AUX_KEYS = [
    "expert_argmax_match",
    "mean_kl_to_expert",
    "policy_entropy",
    "mean_action_deviation_from_expert",
]


def read_json(path):
    
    """
    Reading a JSON file safely.
    """
    
    # If the file does not exist, return an empty dictionary instead of failing.
    if not os.path.exists(path):
        return {}

    # Opening and decoding the JSON file using UTF-8.
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_device(requested):
    
    """
    Choosing the execution device.

    """
    # Forcing CPU when requested.
    if requested == "cpu":
        return "cpu"

    # Forcing CUDA when requested, but fail clearly if unavailable.
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available on this machine.")
        return "cuda"

    # Auto-select CUDA if possible, otherwise fall back to CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_architecture(run_config, config_key):
    
    """
    Recovering architecture hyperparameters for one agent from run_config.json.

    """
    # Starting from the default fallback values.
    cfg = dict(DEFAULT_ARCH)

    # Extracting the relevant agent-specific section from the run config.
    section = run_config.get(config_key, {})

    # Overriding the defaults with the saved architecture if present.
    cfg["hidden_dim"] = int(section.get("hidden_dim", cfg["hidden_dim"]))
    cfg["latent_dim"] = int(section.get("latent_dim", cfg["latent_dim"]))
    cfg["dropout"] = float(section.get("dropout", cfg["dropout"]))

    # Returning the final architecture dictionary.
    return cfg


def build_model(agent_folder, benchmark, arch_cfg):
    
    
    """ This rebuilds the correct neural-network class for each saved model.
        It needs: agent folder name, benchmark feature/action dimensions and architecture config """
    
    # Rebuilding the Behaviour Cloning model.
    if agent_folder == "bc":
        return BehaviorCloningNet(
            benchmark.feature_dim,               # input feature dimension from benchmark
            benchmark.num_actions,               # number of discrete actions
            hidden_dim=arch_cfg["hidden_dim"],   # hidden layer width
            latent_dim=arch_cfg["latent_dim"],   # latent representation width
            dropout=arch_cfg["dropout"],         # dropout probability
        ).to(benchmark.device)                   # move the model to CPU/GPU

    # Rebuilding the Conservative Q-Learning model.
    if agent_folder == "cql":
        return ConservativeQNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
        ).to(benchmark.device)

    # Rebuilding the VOAC ablation.
    if agent_folder == "voac":
        return OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
            use_cost_head=False,                 # VOAC has no cost critic head
        ).to(benchmark.device)

    # Rebuilding the LAADAN-AC model.
    if agent_folder == "laadan_ac":
        return OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
            use_cost_head=True,                  # LAADAN includes the cost head
        ).to(benchmark.device)

    # Failing clearly if an unknown folder name was supplied.
    raise ValueError("Unknown agent folder: " + str(agent_folder))


def strip_module_prefix(state_dict):
    
    
    """
    Removing the 'module.' prefix from state_dict keys when needed.

    """
    
    # Creating an empty dictionary and storing the cleaned key/value pairs here.
    cleaned = {}

    # Looping through every saved parameter tensor.
    for key, value in state_dict.items():
        
        # key is the parameter name. value is the tensor of weights.
        
        # If the key starts with 'module.', remove that prefix.
        cleaned[key[7:] if key.startswith("module.") else key] = value

    # Returning the cleaned state dictionary.
    return cleaned


def unpack_state_dict(loaded_object):
    
    """
    Extracting a usable state_dict from a checkpoint object.
    """
    
    # Continuing only if the loaded checkpoint is dictionary-like.
    if isinstance(loaded_object, dict):
        # Trying common nested locations first.
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in loaded_object and isinstance(loaded_object[key], dict):
                return strip_module_prefix(loaded_object[key])

        # If not nested, check whether the whole dictionary already looks like
        # a raw state_dict, meaning all values are tensors.
        plain_state_dict = True
        for value in loaded_object.values():
            if not torch.is_tensor(value):
                plain_state_dict = False
                break

        # If it is a raw state_dict, clean and return it.
        if plain_state_dict:
            return strip_module_prefix(loaded_object)

    # If no valid state_dict could be found, fail with a clear message.
    raise ValueError("Could not find a valid state_dict inside the checkpoint file.")
    # If no valid weights were found, stop clearly.

def load_model_weights(model, model_path, device):
    
    """
    Loading checkpoint weights into a reconstructed model.
    
    """
    # Reading the checkpoint onto the requested device.
    loaded = torch.load(model_path, map_location=device)

    # Extracting the actual state dictionary from the loaded object.
    state_dict = unpack_state_dict(loaded)

    # Loading the weights into the reconstructed network, this only works if the architecture matches training.
    model.load_state_dict(state_dict)

    # Ensuring dropout/batchnorm behave in evaluation mode, because dropout must be disabled during testing.
    model.eval()

    # Returning the fully loaded model.
    
    return model


def tensor_to_numpy(tensor):
    
    """
    Converting a PyTorch tensor to a NumPy array on CPU.

    """
    return tensor.detach().cpu().numpy().astype(np.float64) 



def normalize_probabilities_array(probs):
    
    """ makes sure probabilities are valid: 
    no negative probabilities
    each row sums to 1
    empty rows become uniform

    """
    
    # Converting to NumPy float64 for stable numeric handling.
    probs = np.asarray(probs, dtype=np.float64)

    # Clipping negative values to zero.
    probs = np.clip(probs, 0.0, None)

    # Computing row sums for normalisation.
    row_sums = probs.sum(axis=1, keepdims=True)

    # Identifying rows that contain no probability mass.
    zero_rows = row_sums.squeeze(1) <= 0.0

    # Avoiding division-by-zero.
    row_sums[row_sums <= 0.0] = 1.0

    # Renormalising rows to sum to 1.
    probs = probs / row_sums

    # Replacing zero rows with a uniform distribution.
    if np.any(zero_rows):
        probs[zero_rows] = 1.0 / probs.shape[1]

    # Returning the cleaned probability table.
    return probs


def softmax_policy_from_logits(logits):
    
    
    """
    Converting raw logits into a stochastic policy using softmax.

    This is used for auxiliary soft-policy analysis for actor-based models.
    """
    return tensor_to_numpy(torch.softmax(logits, dim=1))


def greedy_policy_from_logits(logits):
    
    
    """
    Converting raw logits into a deterministic one-hot greedy policy.

    The highest-logit action in each state gets probability 1.0.
    """
    # Choosing the best action index in each state.
    best = torch.argmax(logits, dim=1)

    # Creating an all-zero policy table with the same shape as the logits.
    policy = torch.zeros_like(logits)

    # Placing a 1.0 in the best action column for each state.
    policy.scatter_(1, best.unsqueeze(1), 1.0)
    

    # Converting to NumPy and returning.
    return tensor_to_numpy(policy)


def masked_softmax_policy_from_logits(logits, mask):
    
    """
    Converting logits into a stochastic masked policy.

    """
    
    # Moving logits to NumPy.
    logits_np = tensor_to_numpy(logits)

    # Copying so the original tensor-derived array is not edited in place.
    masked_logits = logits_np.copy()

    # Pushing inadmissible actions to a huge negative value before exponentiation.
    masked_logits[mask <= 0.0] = -1e30

    # Subtracting the row maximum for numerical stability.
    row_max = masked_logits.max(axis=1, keepdims=True)
    stable = masked_logits - row_max

    # Exponentiating the stabilised logits.
    exp_values = np.exp(stable)

    # Forcing inadmissible actions to zero probability mass.
    exp_values[mask <= 0.0] = 0.0

    # Computing row sums for normalisation.
    row_sums = exp_values.sum(axis=1, keepdims=True)

    # Detecting rows where no admissible mass exists.
    zero_rows = row_sums.squeeze(1) <= 0.0
    
    # Preventing divide-by-zero.
    row_sums[row_sums <= 0.0] = 1.0

    # Normalising to a probability distribution.
    probs = exp_values / row_sums
    
    # Normalise each row so probabilities sum to 1.

    # If a row became empty, fall back to an ordinary softmax over raw logits.
    if np.any(zero_rows):
        fallback = softmax_policy_from_logits(logits)
        probs[zero_rows] = fallback[zero_rows]

    # Returning the masked stochastic policy.
    return probs


def masked_greedy_policy_from_logits(logits, mask):
    
    """ this creates a deterministic greedy policy, but only among admissible actions. Used for LAADAN-AC final testing. """
    
    # Converting logits to NumPy.
    logits_np = tensor_to_numpy(logits)

    # Copying so we can safely modify the working array.
    masked_logits = logits_np.copy()

    # Making inadmissible actions extremely unattractive. So they cannot be selected as best.
    masked_logits[mask <= 0.0] = -1e30

    # Detecting rows with no admissible actions.
    zero_rows = np.all(mask <= 0.0, axis=1)

    # For any fully empty row, fall back to the unmasked logits.
    if np.any(zero_rows):
        masked_logits[zero_rows] = logits_np[zero_rows]

    # Choosing the best action in each row.
    best_actions = np.argmax(masked_logits, axis=1)

    # Creating a one-hot deterministic policy table. For each state row, place 1.0 in the selected action column.
    policy = np.zeros_like(masked_logits)
    policy[np.arange(masked_logits.shape[0]), best_actions] = 1.0

    # Returning the masked greedy policy. This is why LAADAN-AC gets zero inadmissibility: its final policy is masked.
    return policy


def compare_saved_vs_recomputed(saved_metrics, recomputed_metrics):
    
    """
    Comparing stored metrics against freshly recomputed metrics.

    """
    
    # Dictionary that will store per-metric comparisons.
    compared = {}

    # Looping through all recomputed metrics.
    for key in recomputed_metrics:
        # Only compare metrics that were also present in the saved file.
        if key in saved_metrics:
            compared[key] = {
                "saved": float(saved_metrics[key]),
                "recomputed": float(recomputed_metrics[key]),
                "abs_diff": abs(float(saved_metrics[key]) - float(recomputed_metrics[key])),
            }
                 
    # Returning the comparison dictionary.
    return compared


def compute_pass_flags(metric_comparison, core_tol, aux_tol):
    

    """
    Computing pass/fail flags for the reproduction test.

    """
    
    # Tracking the maximum observed absolute difference among core metrics.
    core_max = 0.0

    # Tracking the maximum observed absolute difference among auxiliary metrics.    
    aux_max = 0.0

    # Updating the maximum difference across core metrics.
    for key in CORE_KEYS:
        if key in metric_comparison:
            core_max = max(core_max, float(metric_comparison[key]["abs_diff"]))

    # Updating the maximum difference across auxiliary metrics.
    for key in AUX_KEYS:
        if key in metric_comparison:
            aux_max = max(aux_max, float(metric_comparison[key]["abs_diff"]))

    # Returning a summary of all pass/fail checks.
    return {
        "core_max_abs_diff": core_max,
        "aux_max_abs_diff": aux_max,
        "core_pass": core_max <= core_tol,
        "aux_pass": aux_max <= aux_tol,
        "overall_pass": (core_max <= core_tol) and (aux_max <= aux_tol),
    }


def policy_check(policy):
    
    """
    Running simple sanity checks on a policy table.

    """
    # Computing one row-sum per state.
    row_sums = np.sum(policy, axis=1)

    # Returning a compact dictionary of sanity-check values.
    return {
        "has_nan": bool(np.isnan(policy).any()),
        "has_negative_prob": bool((policy < 0.0).any()),
        "min_probability": float(np.min(policy)),
        "max_probability": float(np.max(policy)),
        "min_row_sum": float(np.min(row_sums)),
        "max_row_sum": float(np.max(row_sums)),
    }


def evaluate_policy_set(benchmark, primary_policy, soft_policy=None):
    
    """ This evaluates a policy using exact benchmark evaluation. """

    # Evaluating the main policy first.
    metrics = benchmark.exact_policy_evaluation(primary_policy)

    # If a soft policy is supplied, compute extra soft-policy metrics too.
    if soft_policy is not None:
        soft_metrics = benchmark.exact_policy_evaluation(soft_policy)
        metrics["soft_survival_rate"] = soft_metrics["survival_rate"]
        metrics["soft_inadmissibility_rate"] = soft_metrics["inadmissibility_rate"]
        metrics["soft_avg_return"] = soft_metrics["avg_return"]
        metrics["soft_policy_entropy"] = soft_metrics["policy_entropy"]

    # Returning the combined metric dictionary.
    return metrics


def evaluate_agent(agent, benchmark, final_models_dir, run_config, core_tol, aux_tol): 
    
    """
    Evaluating one saved final model.

    """
    
    # Building the expected folder path for this agent.
    folder_path = os.path.join(final_models_dir, agent["folder"])
    
    # Example: results/final_models/laadan_ac

    # Expected model checkpoint path.
    model_path = os.path.join(folder_path, "model.pt")

    # Expected saved metrics path.
    metrics_path = os.path.join(folder_path, "metrics.json")

    # Fail early if the folder is missing.
    if not os.path.exists(folder_path):
        raise FileNotFoundError("Missing folder: " + folder_path)

    # Fail early if the checkpoint is missing.
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing checkpoint: " + model_path)

    # Reading the stored metrics file.
    saved_metrics = read_json(metrics_path)

    # Recovering the correct architecture for this agent, hidden size, latent size and dropout from training config.
    arch_cfg = pick_architecture(run_config, agent["config_key"])

    # Rebuilding the right model class. 
    model = build_model(agent["folder"], benchmark, arch_cfg)

    # Loading the saved weights and switching to eval mode.
    model = load_model_weights(model, model_path, benchmark.device)

    # Using the benchmark's full state-feature matrix as model input.
    x = benchmark.state_features_t

    # Extracting the admissibility mask as NumPy float64 for masked policy rules.
    admissible_mask = benchmark.admissible_mask.astype(np.float64)
    

    # Running one forward pass over all states without gradient tracking.
    with torch.no_grad():
        output = model(x)
        
   
    # Select the correct primary/soft policy interpretation per model family
   
    if agent["kind"] == "bc":
        
        # BC outputs policy logits, so primary evaluation uses greedy argmax.
        primary_policy = greedy_policy_from_logits(output)

        # Also compute the soft version for analysis.
        soft_policy = softmax_policy_from_logits(output)

        # Readable description of the policy rule used.
        policy_rule = "greedy policy from behaviour-cloning logits"
        

    elif agent["kind"] == "cql":
        
        # CQL outputs Q-values, so use the benchmark's greedy Q policy rule.
        q_values = tensor_to_numpy(output)
        primary_policy = benchmark.greedy_policy_from_q_unmasked(q_values)

        # No soft policy is used for CQL in this test script.
        soft_policy = None

        # Readable description of the test-time interpretation.
        policy_rule = "greedy policy from CQL Q-values"
        

    elif agent["kind"] == "voac":
        
        # VOAC returns a dictionary; actor logits are under output["logits"].
        logits = output["logits"]

        # Primary evaluation uses greedy actor selection.
        primary_policy = greedy_policy_from_logits(logits)

        # Soft policy is also recorded for extra analysis.
        soft_policy = softmax_policy_from_logits(logits)

        # Readable description.
        policy_rule = "greedy policy from VOAC actor logits"
        


    elif agent["kind"] == "laadan":
        
        # LAADAN also returns actor logits in output["logits"].
        logits = output["logits"]

        # Primary evaluation uses masked greedy policy, matching training.
        primary_policy = masked_greedy_policy_from_logits(logits, admissible_mask)

        # Soft masked policy is also recorded for sanity analysis.
        soft_policy = masked_softmax_policy_from_logits(logits, admissible_mask)

        # Readable description.
        policy_rule = "masked greedy policy from LAADAN actor logits"
        

    else:
        # Fail clearly if an unsupported model kind appears.
        raise ValueError("Unknown agent kind: " + str(agent["kind"]))

    # Recomputing all relevant benchmark metrics from the chosen policy/policies.
    recomputed_metrics = evaluate_policy_set(benchmark, primary_policy, soft_policy=soft_policy)

    # Comparing saved metrics against the recomputed ones.
    metric_comparison = compare_saved_vs_recomputed(saved_metrics, recomputed_metrics)

    # Turning those differences into core/auxiliary pass/fail flags.
    pass_flags = compute_pass_flags(metric_comparison, core_tol, aux_tol)

    # Returning a structured result dictionary for this one agent.
    
    return {
        "agent_name": agent["name"],
        "folder": agent["folder"],
        "checkpoint_path": model_path,
        "architecture": arch_cfg,
        "policy_rule_used": policy_rule,
        "policy_check": policy_check(primary_policy),
        "saved_metrics": saved_metrics,
        "recomputed_metrics": recomputed_metrics,
        "saved_vs_recomputed": metric_comparison,
        "pass_flags": pass_flags,
    }
    
    # This gets saved into test_summary.json.

def print_agent_report(result, core_tol, aux_tol):
    
    """
    Printing a readable console report for one evaluated agent.

    """
    
    # Printing a section divider.
    print("\n" + "=" * 88)

    # Printing the model name.
    print(result["agent_name"])

    # Printing another section divider.
    print("=" * 88)

    # Showing which checkpoint file was loaded.
    print("Checkpoint:", result["checkpoint_path"])

    # Showing which policy interpretation rule was used.
    print("Policy interpretation:", result["policy_rule_used"])

    # Printing the reconstructed model architecture.
    print(
        "Architecture: hidden_dim={0}, latent_dim={1}, dropout={2}".format(
            result["architecture"]["hidden_dim"],
            result["architecture"]["latent_dim"],
            result["architecture"]["dropout"],
        )
    )

    # Printing sanity checks on the primary policy.
    print("\nPolicy sanity check:")
    for key, value in result["policy_check"].items():
        print(key, "=", value)

    # Printing all recomputed metrics.
    print("\nRecomputed metrics:")
    for key, value in result["recomputed_metrics"].items():
        print(key + " =", round(float(value), 6))

    # Printing direct comparisons between saved and recomputed metrics.
    print("\nSaved vs recomputed:")
    for key, payload in result["saved_vs_recomputed"].items():
        print(
            key
            + " | saved="
            + str(round(float(payload["saved"]), 6))
            + " | recomputed="
            + str(round(float(payload["recomputed"]), 6))
            + " | abs_diff="
            + str(round(float(payload["abs_diff"]), 10))
        )

    # Printing pass/fail summary.   
    print("\nPass rule:")
    print("core tolerance =", core_tol)
    print("aux tolerance =", aux_tol)
    print("core max abs diff =", round(float(result["pass_flags"]["core_max_abs_diff"]), 10))
    print("aux max abs diff =", round(float(result["pass_flags"]["aux_max_abs_diff"]), 10))
    print("overall PASS =", result["pass_flags"]["overall_pass"])


def to_json_safe(value):
    
    """
    Converting NumPy types and arrays into plain Python objects for JSON saving.

    """
    
    # Recursively convert dictionaries.
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}

    # Recursively convert lists.
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]

    # Converting tuples to JSON-safe lists.
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]

    # Converting NumPy arrays to Python lists.
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Converting NumPy floating scalars to Python float.
    if isinstance(value, (np.floating,)):
        return float(value)

    # Converting NumPy integer scalars to Python int.
    if isinstance(value, (np.integer,)):
        return int(value)

    # Converting NumPy booleans to Python bool.
    if isinstance(value, (np.bool_,)):
        return bool(value)

    # Leaving already JSON-safe values unchanged.
    return value


def main(): # This runs the whole testing script.
    
    """
    Main entry point for the final-model testing script.

    """
    
    # Building the command-line parser.
    parser = argparse.ArgumentParser(description="Training-aligned tester for final ICU-Sepsis models.")

    # Path to the benchmark data folder.
    parser.add_argument("--data-dir", required=True, help="Folder containing the ICU-Sepsis benchmark files.")

    # Path to the folder containing one chosen final model per method.
    parser.add_argument(
        "--final-models-dir",
        default=os.path.join("results", "final_models"),
        help="Folder containing bc/, cql/, voac/, laadan_ac/.",
    )

    # Path to the saved run configuration from training.
    parser.add_argument(
        "--config-path",
        default=os.path.join("results", "run_config.json"),
        help="Path to run_config.json used during training.",
    )

    # Device selection.
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    # Optional explicit override of the horizon.
    parser.add_argument("--horizon", type=int, default=None)

    # Tolerance for the main benchmark metrics.
    parser.add_argument("--core-tol", type=float, default=1e-5)

    # Tolerance for auxiliary analysis metrics.
    parser.add_argument("--aux-tol", type=float, default=1e-5)

    # Output JSON file path.
    parser.add_argument(
        "--output-json",
        default=os.path.join("results", "final_models", "test_summary.json"),
        help="Where to save the testing report JSON.",
    )

    # Parsing the command-line arguments.
    args = parser.parse_args()

    # Reading the training configuration.
    run_config = read_json(args.config_path)

    # Resolving the device.
    device = choose_device(args.device)

    # Using explicit CLI horizon if given, otherwise match training config.
    horizon = int(args.horizon if args.horizon is not None else run_config.get("horizon", 20))

    # Matching the state representation used during training.
    use_one_hot_states = bool(run_config.get("use_one_hot_states", False))

    # Rebuilding the benchmark exactly as used during training/testing.
    benchmark = ICUSepsisOfflineBenchmark(
        data_dir=args.data_dir,
        horizon=horizon,
        device=device,
        use_one_hot_states=use_one_hot_states,
    )

    # Preparing the overall JSON report container.
    
    all_results = {
        "device_used": device,
        "data_dir": args.data_dir,
        "final_models_dir": args.final_models_dir,
        "config_path": args.config_path,
        "horizon": horizon,
        "use_one_hot_states": use_one_hot_states,
        "core_tolerance": args.core_tol,
        "aux_tolerance": args.aux_tol,
        "benchmark_description": {
            "num_states": benchmark.num_states,
            "num_actions": benchmark.num_actions,
            "feature_dim": benchmark.feature_dim,
            "death_state": benchmark.death_state,
            "survival_state": benchmark.survival_state,
        },
        "agents": [],
    }

    # Tracking whether all models passed the reproduction check.
    overall_pass = True

    # Evaluating each agent one by one.
    for agent in AGENTS:
        result = evaluate_agent(
            agent=agent,
            benchmark=benchmark,
            final_models_dir=args.final_models_dir,
            run_config=run_config,
            core_tol=args.core_tol,
            aux_tol=args.aux_tol,
        )

        # Saving the per-agent result.
        all_results["agents"].append(result)

        # Printing the readable console report.
        print_agent_report(result, args.core_tol, args.aux_tol)

        # Updating the overall pass flag.
        if not result["pass_flags"]["overall_pass"]:
            overall_pass = False

    # Saving the overall pass flag into the final JSON object.
    all_results["overall_pass"] = overall_pass

    # Extracting the output folder path.
    output_dir = os.path.dirname(args.output_json)

    # Creating the output folder if needed.
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Writing the final JSON summary.
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(all_results), handle, indent=2)

    # Printing a final summary banner.
    print("\n" + "*" * 88)
    print("Finished.")
    print("Overall PASS =", overall_pass)
    print("Full testing report written to:", args.output_json)
    print("*" * 88)


# Main entry point.
if __name__ == "__main__":
    main()
    