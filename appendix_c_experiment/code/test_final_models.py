
# test_final_models.py
#
# Training-aligned tester for final ICU-Sepsis models.
#
# Supports:
# - BC
# - CQL
# - VOAC
# - LAADAN-AC
# - LAADAN-AC-PD / LAADAN-AC-RC no-mask variant

import argparse
import json
import os

import numpy as np
import torch

from benchmark import ICUSepsisOfflineBenchmark
from models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet


AGENTS = [
    {"folder": "bc", "name": "Behavior Cloning", "config_key": "bc", "kind": "bc"},
    {"folder": "cql", "name": "Conservative Q-Learning", "config_key": "cql", "kind": "cql"},
    {"folder": "voac", "name": "Vanilla Offline Actor-Critic", "config_key": "voac", "kind": "voac"},
    {"folder": "laadan_ac", "name": "LAADAN-AC", "config_key": "laadan_ac", "kind": "laadan"},
    {
        "folder": "laadan_ac_pd_nomask",
        "name": "LAADAN-AC-PD / LAADAN-AC-RC",
        "config_key": "laadan_ac_pd",
        "kind": "laadan_rc",
    },
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
    "soft_survival_rate",
    "soft_inadmissibility_rate",
    "soft_avg_return",
    "soft_policy_entropy",
]


def read_json(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_device(requested):
    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_architecture(run_config, config_key):
    cfg = dict(DEFAULT_ARCH)
    section = run_config.get(config_key, {})

    cfg["hidden_dim"] = int(section.get("hidden_dim", cfg["hidden_dim"]))
    cfg["latent_dim"] = int(section.get("latent_dim", cfg["latent_dim"]))
    cfg["dropout"] = float(section.get("dropout", cfg["dropout"]))

    cfg["num_cost_heads"] = int(section.get("num_cost_heads", 1))
    cfg["use_action_mask"] = bool(section.get("use_action_mask", False))

    cfg["rho_reward"] = float(section.get("rho_reward", 1.0))
    cfg["rho_cost"] = float(section.get("rho_cost", 1.0))

    cfg["barrier_weight"] = float(section.get("barrier_weight", 25.0))
    cfg["support_weight"] = float(section.get("support_weight", 1.0))
    cfg["smoothness_weight"] = float(section.get("smoothness_weight", 0.001))
    cfg["uncertainty_weight"] = float(section.get("uncertainty_weight", 0.1))

    cfg["policy_temperature"] = float(section.get("policy_temperature", 1.0))
    cfg["energy_scale"] = float(section.get("energy_scale", 1.0))

    cfg["cost_threshold"] = float(section.get("cost_threshold", 0.05))
    cfg["uncertainty_threshold"] = float(section.get("uncertainty_threshold", 1.0))
    cfg["expert_threshold"] = float(section.get("expert_threshold", 1e-8))

    cfg["cost_budget"] = float(section.get("cost_budget", 0.0001))

    return cfg


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

    if agent_folder in ["laadan_ac", "laadan_ac_pd_nomask", "laadan_ac_pd_masked"]:
        try:
            return OfflineActorCriticNet(
                benchmark.feature_dim,
                benchmark.num_actions,
                hidden_dim=arch_cfg["hidden_dim"],
                latent_dim=arch_cfg["latent_dim"],
                dropout=arch_cfg["dropout"],
                use_cost_head=True,
                num_cost_heads=int(arch_cfg.get("num_cost_heads", 1)),
            ).to(benchmark.device)
        except TypeError:
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

    raise ValueError("Could not find a valid state_dict inside checkpoint.")


def add_cost_head_aliases_if_needed(state_dict):
    updated = dict(state_dict)

    for suffix in ["linear.weight", "linear.bias"]:
        old_key = "cost_head." + suffix
        new_key = "cost_heads.0." + suffix

        if old_key in updated and new_key not in updated:
            updated[new_key] = updated[old_key]

        if new_key in updated and old_key not in updated:
            updated[old_key] = updated[new_key]

    return updated


def load_model_weights(model, model_path, device):
    loaded = torch.load(model_path, map_location=device)
    state_dict = unpack_state_dict(loaded)
    state_dict = add_cost_head_aliases_if_needed(state_dict)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        print("Strict load failed; retrying with strict=False.")
        print("Original load error:", str(error))
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    return model


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


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


def reward_lcb_from_twin_q(q1, q2, rho_reward=1.0):
    stacked = torch.stack([q1, q2], dim=0)
    mean = torch.mean(stacked, dim=0)
    std = torch.std(stacked, dim=0, unbiased=False)
    lcb = mean - float(rho_reward) * std
    return lcb, mean, std


def cost_ucb_from_ensemble(cost_raw, rho_cost=1.0):
    if cost_raw is None:
        raise ValueError("LAADAN-AC-PD / RC evaluation requires a cost output.")

    if cost_raw.dim() == 2:
        cost_stack = torch.relu(cost_raw).unsqueeze(0)
    else:
        cost_stack = torch.relu(cost_raw)

    mean = torch.mean(cost_stack, dim=0)

    if cost_stack.shape[0] > 1:
        std = torch.std(cost_stack, dim=0, unbiased=False)
    else:
        std = torch.zeros_like(mean)

    ucb = mean + float(rho_cost) * std

    return ucb, mean, std


def risk_adjusted_logits_rc(
    logits,
    reward_lcb,
    cost_ucb,
    immediate_cost,
    expert_policy,
    smoothness_cost,
    uncertainty_cost,
    lambda_inad,
    lambda_support,
    lambda_smooth,
    lambda_uncertainty,
    barrier_weight=25.0,
    support_weight=1.0,
    smoothness_weight=0.001,
    uncertainty_weight=0.1,
    temperature=1.0,
    energy_scale=1.0,
):
    support_penalty = -torch.log(expert_policy + 1e-8)

    safety_energy = (
        reward_lcb
        - float(lambda_inad) * cost_ucb
        - float(lambda_support) * support_penalty
        - float(lambda_smooth) * smoothness_cost
        - float(lambda_uncertainty) * uncertainty_cost
        - float(barrier_weight) * immediate_cost
        - float(support_weight) * support_penalty
        - float(smoothness_weight) * smoothness_cost
        - float(uncertainty_weight) * uncertainty_cost
    )

    adjusted_logits = logits + float(energy_scale) * safety_energy

    return adjusted_logits / max(float(temperature), 1e-8)


def certified_greedy_policy_rc(
    energy_logits,
    immediate_cost,
    cost_ucb,
    expert_policy,
    uncertainty_cost,
    cost_threshold=0.05,
    uncertainty_threshold=1.0,
    expert_threshold=1e-8,
):
    safe_by_immediate = immediate_cost <= 0.5
    safe_by_cost = cost_ucb <= float(cost_threshold)
    safe_by_uncertainty = uncertainty_cost <= float(uncertainty_threshold)
    safe_by_support = expert_policy > float(expert_threshold)

    certified = safe_by_immediate & safe_by_cost & safe_by_uncertainty & safe_by_support

    very_negative = torch.full_like(energy_logits, -1e9)

    certified_logits = torch.where(certified, energy_logits, very_negative)
    has_certified = torch.sum(certified.float(), dim=1) > 0.0
    best_certified = torch.argmax(certified_logits, dim=1)

    admissible_logits = torch.where(safe_by_immediate, energy_logits, very_negative)
    has_admissible = torch.sum(safe_by_immediate.float(), dim=1) > 0.0
    best_admissible = torch.argmax(admissible_logits, dim=1)

    support_penalty = -torch.log(expert_policy + 1e-8)
    fallback_risk = immediate_cost + cost_ucb + uncertainty_cost + 0.01 * support_penalty
    lowest_risk = torch.argmin(fallback_risk, dim=1)

    chosen = torch.where(
        has_certified,
        best_certified,
        torch.where(has_admissible, best_admissible, lowest_risk),
    )

    policy = torch.zeros_like(energy_logits)
    policy.scatter_(1, chosen.unsqueeze(1), 1.0)

    return policy


def evaluate_policy_set(benchmark, primary_policy, soft_policy=None):
    metrics = benchmark.exact_policy_evaluation(primary_policy)

    if soft_policy is not None:
        soft_metrics = benchmark.exact_policy_evaluation(soft_policy)
        metrics["soft_survival_rate"] = soft_metrics["survival_rate"]
        metrics["soft_inadmissibility_rate"] = soft_metrics["inadmissibility_rate"]
        metrics["soft_avg_return"] = soft_metrics["avg_return"]
        metrics["soft_policy_entropy"] = soft_metrics["policy_entropy"]

    return metrics


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

    elif agent["kind"] == "laadan_rc":
        logits = output["logits"]

        q1 = output["q1"]
        q2 = output["q2"]

        reward_lcb, _, reward_unc = reward_lcb_from_twin_q(
            q1,
            q2,
            rho_reward=float(arch_cfg.get("rho_reward", 1.0)),
        )

        cost_ucb, _, cost_unc = cost_ucb_from_ensemble(
            output["cost"],
            rho_cost=float(arch_cfg.get("rho_cost", 1.0)),
        )

        total_uncertainty = reward_unc + cost_unc

        immediate_cost = benchmark.immediate_cost_t
        expert_policy = benchmark.expert_safe_t
        smoothness_cost = benchmark.smoothness_cost_t

        lambda_inad = float(saved_metrics.get("lambda_inad", saved_metrics.get("lagrange", 0.0)))
        lambda_support = float(saved_metrics.get("lambda_support", 0.0))
        lambda_smooth = float(saved_metrics.get("lambda_smooth", 0.0))
        lambda_uncertainty = float(saved_metrics.get("lambda_uncertainty", 0.0))

        if bool(arch_cfg.get("use_action_mask", False)):
            primary_policy = masked_greedy_policy_from_logits(logits, admissible_mask)
            soft_policy = masked_softmax_policy_from_logits(logits, admissible_mask)
            policy_rule = "masked greedy policy from LAADAN-AC-PD / RC actor logits"
        else:
            rc_logits = risk_adjusted_logits_rc(
                logits=logits,
                reward_lcb=reward_lcb,
                cost_ucb=cost_ucb,
                immediate_cost=immediate_cost,
                expert_policy=expert_policy,
                smoothness_cost=smoothness_cost,
                uncertainty_cost=total_uncertainty,
                lambda_inad=lambda_inad,
                lambda_support=lambda_support,
                lambda_smooth=lambda_smooth,
                lambda_uncertainty=lambda_uncertainty,
                barrier_weight=float(arch_cfg.get("barrier_weight", 25.0)),
                support_weight=float(arch_cfg.get("support_weight", 1.0)),
                smoothness_weight=float(arch_cfg.get("smoothness_weight", 0.001)),
                uncertainty_weight=float(arch_cfg.get("uncertainty_weight", 0.1)),
                temperature=float(arch_cfg.get("policy_temperature", 1.0)),
                energy_scale=float(arch_cfg.get("energy_scale", 1.0)),
            )

            primary_policy_tensor = certified_greedy_policy_rc(
                energy_logits=rc_logits,
                immediate_cost=immediate_cost,
                cost_ucb=cost_ucb,
                expert_policy=expert_policy,
                uncertainty_cost=total_uncertainty,
                cost_threshold=float(arch_cfg.get("cost_threshold", 0.05)),
                uncertainty_threshold=float(arch_cfg.get("uncertainty_threshold", 1.0)),
                expert_threshold=float(arch_cfg.get("expert_threshold", 1e-8)),
            )

            primary_policy = tensor_to_numpy(primary_policy_tensor)
            soft_policy = softmax_policy_from_logits(rc_logits)
            policy_rule = "certified risk-adjusted greedy policy from LAADAN-AC-PD / RC logits"

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
    parser = argparse.ArgumentParser(
        description="Training-aligned tester for final ICU-Sepsis models."
    )

    parser.add_argument("--data-dir", required=True, help="Folder containing ICU-Sepsis benchmark files.")

    parser.add_argument(
        "--final-models-dir",
        default=os.path.join("results", "final_models"),
        help="Folder containing selected final model folders.",
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
        folder_path = os.path.join(args.final_models_dir, agent["folder"])

        if not os.path.exists(folder_path):
            print("Skipping missing final model folder:", folder_path)
            continue

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

    if len(all_results["agents"]) == 0:
        raise RuntimeError(
            "No final model folders were found or tested. "
            "Run pick_final_models.py first with the correct results directory:\n"
            "!python -u pick_final_models.py --results-dir results_icu_sepsis_LAADAN_AC_PD --clear-output"
        )

    all_results["overall_pass"] = overall_pass

    output_dir = os.path.dirname(args.output_json)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(all_results), handle, indent=2)

    print("\n" + "*" * 88)
    print("Finished.")
    print("Overall PASS =", overall_pass)
    print("Number of tested models =", len(all_results["agents"]))
    print("Full testing report written to:", args.output_json)
    print("*" * 88)


if __name__ == "__main__":
    main()
