
import argparse


import json


import os


import numpy as np


import torch


from scripts.benchmark import ICUSepsisOfflineBenchmark


from scripts.models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet


AGENTS = [
    {"folder": "bc", "name": "Behavior Cloning", "config_key": "bc", "kind": "bc"},
    {"folder": "cql", "name": "Conservative Q-Learning", "config_key": "cql", "kind": "cql"},
    {"folder": "voac", "name": "Vanilla Offline Actor-Critic", "config_key": "voac", "kind": "voac"},
    {"folder": "laadan_ac", "name": "LAADAN-AC", "config_key": "laadan_ac", "kind": "laadan"},
]

DEFAULT_ARCH = {
    "hidden_dim": 128,
    "latent_dim": 128,
    "dropout": 0.10,
}


CORE_KEYS = [
    "avg_return",
    "survival_rate",
    "mortality_rate",
    "avg_length",
    "inadmissibility_rate",
]


AUX_KEYS = [
    "expert_argmax_match",
    "mean_kl_to_expert",
    "policy_entropy",
    "mean_action_deviation_from_expert",
]


# evaluation
def read_json(path):

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# select device
def choose_device(requested):


    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available on this machine.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


# select architecture
def pick_architecture(run_config, config_key):


    cfg = dict(DEFAULT_ARCH)

    section = run_config.get(config_key, {})

    cfg["hidden_dim"] = int(section.get("hidden_dim", cfg["hidden_dim"]))
    cfg["latent_dim"] = int(section.get("latent_dim", cfg["latent_dim"]))
    cfg["dropout"] = float(section.get("dropout", cfg["dropout"]))

    return cfg


# build model
def build_model(agent_folder, benchmark, arch_cfg):


    if agent_folder == "bc":
        return BehaviorCloningNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
        ).to(benchmark.device)


    if agent_folder == "cql":
        return ConservativeQNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
        ).to(benchmark.device)


    if agent_folder == "voac":
        return OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
            use_cost_head=False,
        ).to(benchmark.device)


    if agent_folder == "laadan_ac":
        return OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=arch_cfg["hidden_dim"],
            latent_dim=arch_cfg["latent_dim"],
            dropout=arch_cfg["dropout"],
            use_cost_head=True,
        ).to(benchmark.device)


    raise ValueError("Unknown agent folder: " + str(agent_folder))


def strip_module_prefix(state_dict):




    cleaned = {}


    for key, value in state_dict.items():




        cleaned[key[7:] if key.startswith("module.") else key] = value


    return cleaned


def unpack_state_dict(loaded_object):


    if isinstance(loaded_object, dict):

        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in loaded_object and isinstance(loaded_object[key], dict):
                return strip_module_prefix(loaded_object[key])



        plain_state_dict = True
        for value in loaded_object.values():
            if not torch.is_tensor(value):
                plain_state_dict = False
                break


        if plain_state_dict:
            return strip_module_prefix(loaded_object)


    raise ValueError("Could not find a valid state_dict inside the checkpoint file.")


def load_model_weights(model, model_path, device):






    loaded = torch.load(model_path, map_location=device)


    state_dict = unpack_state_dict(loaded)


    model.load_state_dict(state_dict)


    model.eval()



    return model


def tensor_to_numpy(tensor):

    return tensor.detach().cpu().numpy().astype(np.float64)



def normalize_probabilities_array(probs):



    probs = np.asarray(probs, dtype=np.float64)


    probs = np.clip(probs, 0.0, None)


    row_sums = probs.sum(axis=1, keepdims=True)


    zero_rows = row_sums.squeeze(1) <= 0.0


    row_sums[row_sums <= 0.0] = 1.0


    probs = probs / row_sums


    if np.any(zero_rows):
        probs[zero_rows] = 1.0 / probs.shape[1]


    return probs


def softmax_policy_from_logits(logits):


    return tensor_to_numpy(torch.softmax(logits, dim=1))


def greedy_policy_from_logits(logits):








    best = torch.argmax(logits, dim=1)


    policy = torch.zeros_like(logits)


    policy.scatter_(1, best.unsqueeze(1), 1.0)



    return tensor_to_numpy(policy)


def masked_softmax_policy_from_logits(logits, mask):



    logits_np = tensor_to_numpy(logits)


    masked_logits = logits_np.copy()


    masked_logits[mask <= 0.0] = -1e30


    row_max = masked_logits.max(axis=1, keepdims=True)
    stable = masked_logits - row_max


    exp_values = np.exp(stable)


    exp_values[mask <= 0.0] = 0.0


    row_sums = exp_values.sum(axis=1, keepdims=True)


    zero_rows = row_sums.squeeze(1) <= 0.0


    row_sums[row_sums <= 0.0] = 1.0


    probs = exp_values / row_sums




    if np.any(zero_rows):
        fallback = softmax_policy_from_logits(logits)
        probs[zero_rows] = fallback[zero_rows]


    return probs


def masked_greedy_policy_from_logits(logits, mask):



    logits_np = tensor_to_numpy(logits)


    masked_logits = logits_np.copy()


    masked_logits[mask <= 0.0] = -1e30


    zero_rows = np.all(mask <= 0.0, axis=1)


    if np.any(zero_rows):
        masked_logits[zero_rows] = logits_np[zero_rows]


    best_actions = np.argmax(masked_logits, axis=1)


    policy = np.zeros_like(masked_logits)
    policy[np.arange(masked_logits.shape[0]), best_actions] = 1.0


    return policy


def compare_saved_vs_recomputed(saved_metrics, recomputed_metrics):



    compared = {}


    for key in recomputed_metrics:

        if key in saved_metrics:
            compared[key] = {
                "saved": float(saved_metrics[key]),
                "recomputed": float(recomputed_metrics[key]),
                "abs_diff": abs(float(saved_metrics[key]) - float(recomputed_metrics[key])),
            }


    return compared


# compute pass flags
def compute_pass_flags(metric_comparison, core_tol, aux_tol):




    core_max = 0.0


    aux_max = 0.0


    for key in CORE_KEYS:
        if key in metric_comparison:
            core_max = max(core_max, float(metric_comparison[key]["abs_diff"]))


    for key in AUX_KEYS:
        if key in metric_comparison:
            aux_max = max(aux_max, float(metric_comparison[key]["abs_diff"]))


    return {
        "core_max_abs_diff": core_max,
        "aux_max_abs_diff": aux_max,
        "core_pass": core_max <= core_tol,
        "aux_pass": aux_max <= aux_tol,
        "overall_pass": (core_max <= core_tol) and (aux_max <= aux_tol),
    }


def policy_check(policy):






    row_sums = np.sum(policy, axis=1)


    return {
        "has_nan": bool(np.isnan(policy).any()),
        "has_negative_prob": bool((policy < 0.0).any()),
        "min_probability": float(np.min(policy)),
        "max_probability": float(np.max(policy)),
        "min_row_sum": float(np.min(row_sums)),
        "max_row_sum": float(np.max(row_sums)),
    }


# evaluate policy set
def evaluate_policy_set(benchmark, primary_policy, soft_policy=None):



    metrics = benchmark.exact_policy_evaluation(primary_policy)


    if soft_policy is not None:
        soft_metrics = benchmark.exact_policy_evaluation(soft_policy)
        metrics["soft_survival_rate"] = soft_metrics["survival_rate"]
        metrics["soft_inadmissibility_rate"] = soft_metrics["inadmissibility_rate"]
        metrics["soft_avg_return"] = soft_metrics["avg_return"]
        metrics["soft_policy_entropy"] = soft_metrics["policy_entropy"]


    return metrics


# evaluate agent
def evaluate_agent(agent, benchmark, final_models_dir, run_config, core_tol, aux_tol):



    folder_path = os.path.join(final_models_dir, agent["folder"])




    model_path = os.path.join(folder_path, "model.pt")


    metrics_path = os.path.join(folder_path, "metrics.json")


    if not os.path.exists(folder_path):
        raise FileNotFoundError("Missing folder: " + folder_path)


    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing checkpoint: " + model_path)


    saved_metrics = read_json(metrics_path)


    arch_cfg = pick_architecture(run_config, agent["config_key"])


    model = build_model(agent["folder"], benchmark, arch_cfg)


    model = load_model_weights(model, model_path, benchmark.device)


    x = benchmark.state_features_t


    admissible_mask = benchmark.admissible_mask.astype(np.float64)



    with torch.no_grad():
        output = model(x)




    if agent["kind"] == "bc":


        primary_policy = greedy_policy_from_logits(output)


        soft_policy = softmax_policy_from_logits(output)


        policy_rule = "greedy policy from behaviour-cloning logits"


    elif agent["kind"] == "cql":


        q_values = tensor_to_numpy(output)
        primary_policy = benchmark.greedy_policy_from_q_unmasked(q_values)


        soft_policy = None


        policy_rule = "greedy policy from CQL Q-values"


    elif agent["kind"] == "voac":


        logits = output["logits"]


        primary_policy = greedy_policy_from_logits(logits)


        soft_policy = softmax_policy_from_logits(logits)


        policy_rule = "greedy policy from VOAC actor logits"



    elif agent["kind"] == "laadan":


        logits = output["logits"]


        primary_policy = masked_greedy_policy_from_logits(logits, admissible_mask)


        soft_policy = masked_softmax_policy_from_logits(logits, admissible_mask)


        policy_rule = "masked greedy policy from LAADAN actor logits"


    else:

        raise ValueError("Unknown agent kind: " + str(agent["kind"]))


    recomputed_metrics = evaluate_policy_set(benchmark, primary_policy, soft_policy=soft_policy)


    metric_comparison = compare_saved_vs_recomputed(saved_metrics, recomputed_metrics)


    pass_flags = compute_pass_flags(metric_comparison, core_tol, aux_tol)



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



def print_agent_report(result, core_tol, aux_tol):







    print("\n" + "=" * 88)


    print(result["agent_name"])


    print("=" * 88)


    print("Checkpoint:", result["checkpoint_path"])


    print("Policy interpretation:", result["policy_rule_used"])


    print(
        "Architecture: hidden_dim={0}, latent_dim={1}, dropout={2}".format(
            result["architecture"]["hidden_dim"],
            result["architecture"]["latent_dim"],
            result["architecture"]["dropout"],
        )
    )


    print("\nPolicy sanity check:")
    for key, value in result["policy_check"].items():
        print(key, "=", value)


    print("\nRecomputed metrics:")
    for key, value in result["recomputed_metrics"].items():
        print(key + " =", round(float(value), 6))


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


    print("\nPass rule:")
    print("core tolerance =", core_tol)
    print("aux tolerance =", aux_tol)
    print("core max abs diff =", round(float(result["pass_flags"]["core_max_abs_diff"]), 10))
    print("aux max abs diff =", round(float(result["pass_flags"]["aux_max_abs_diff"]), 10))
    print("overall PASS =", result["pass_flags"]["overall_pass"])


def to_json_safe(value):



    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}


    if isinstance(value, list):
        return [to_json_safe(v) for v in value]


    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]


    if isinstance(value, np.ndarray):
        return value.tolist()


    if isinstance(value, (np.floating,)):
        return float(value)


    if isinstance(value, (np.integer,)):
        return int(value)


    if isinstance(value, (np.bool_,)):
        return bool(value)


    return value


def main():

    parser = argparse.ArgumentParser(description="Training-aligned tester for final ICU-Sepsis models.")


    parser.add_argument("--data-dir", required=True, help="Folder containing the ICU-Sepsis benchmark files.")

    parser.add_argument(
        "--final-models-dir",
        default=os.path.join("results", "final_models"),
        help="Folder containing bc/, cql/, voac/, laadan_ac/.",
    )

    parser.add_argument(
        "--config-path",
        default=os.path.join("results", "run_config.json"),
        help="Path to run_config.json used during training.",
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")

    parser.add_argument("--horizon", type=int, default=None)

    parser.add_argument("--core-tol", type=float, default=1e-5)

    parser.add_argument("--aux-tol", type=float, default=1e-5)

    parser.add_argument(
        "--output-json",
        default=os.path.join("results", "final_models", "test_summary.json"),
        help="Where to save the testing report JSON.",
    )

    args = parser.parse_args()

    run_config = read_json(args.config_path)

    device = choose_device(args.device)

    horizon = int(args.horizon if args.horizon is not None else run_config.get("horizon", 20))

    use_one_hot_states = bool(run_config.get("use_one_hot_states", False))

    benchmark = ICUSepsisOfflineBenchmark(
        data_dir=args.data_dir,
        horizon=horizon,
        device=device,
        use_one_hot_states=use_one_hot_states,
    )


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

    overall_pass = True

    for agent in AGENTS:
        result = evaluate_agent(
            agent=agent,
            benchmark=benchmark,
            final_models_dir=args.final_models_dir,
            run_config=run_config,
            core_tol=args.core_tol,
            aux_tol=args.aux_tol,
        )


        all_results["agents"].append(result)

        print_agent_report(result, args.core_tol, args.aux_tol)


        if not result["pass_flags"]["overall_pass"]:
            overall_pass = False


    all_results["overall_pass"] = overall_pass


    output_dir = os.path.dirname(args.output_json)


    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(all_results), handle, indent=2)


    print("\n" + "*" * 88)
    print("Finished.")
    print("Overall PASS =", overall_pass)
    print("Full testing report written to:", args.output_json)
    print("*" * 88)



if __name__ == "__main__":
    main()
