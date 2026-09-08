import argparse
import csv
import json
import os

import numpy as np
import torch

from benchmark import ICUSepsisOfflineBenchmark
from models import OfflineActorCriticNet


VARIANTS = [
    "full_rc",
    "no_lambda_updates",
    "no_barrier",
    "no_certified_nudging",
    "no_support_lambda",
    "no_cost_ucb",
    "lambda_only_no_mask",
]


# ablation evaluation
def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# save results
def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_json_safe(payload), handle, indent=2)


# select device
def choose_device(requested):
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


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

        if all(torch.is_tensor(value) for value in loaded_object.values()):
            return strip_module_prefix(loaded_object)

    raise ValueError("Could not find a valid state_dict inside checkpoint.")


def load_model_weights(model, model_path, device):
    loaded = torch.load(model_path, map_location=device)
    state_dict = unpack_state_dict(loaded)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        print("Strict load failed; retrying with strict=False.")
        print("Original load error:", str(error))
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


# build model
def build_model(benchmark, config):
    hidden_dim = int(config.get("hidden_dim", 128))
    latent_dim = int(config.get("latent_dim", 128))
    dropout = float(config.get("dropout", 0.10))
    num_cost_heads = int(config.get("num_cost_heads", 1))

    try:
        return OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            use_cost_head=True,
            num_cost_heads=num_cost_heads,
        ).to(benchmark.device)
    except TypeError:
        return OfflineActorCriticNet(
            benchmark.feature_dim,
            benchmark.num_actions,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            use_cost_head=True,
        ).to(benchmark.device)


def reward_lcb_from_twin_q(q1, q2, rho_reward=1.0):
    stacked = torch.stack([q1, q2], dim=0)
    mean = torch.mean(stacked, dim=0)
    std = torch.std(stacked, dim=0, unbiased=False)
    lcb = mean - float(rho_reward) * std
    return lcb, mean, std


def cost_ucb_from_ensemble(cost_raw, rho_cost=1.0):
    if cost_raw is None:
        raise ValueError("Cost output is required for RC ablation testing.")

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


# admissible policy
def masked_softmax_from_logits(logits, mask):
    very_negative = torch.full_like(logits, -1e9)
    masked_logits = torch.where(mask > 0.0, logits, very_negative)
    return torch.softmax(masked_logits, dim=1)


def masked_greedy_policy_from_logits(logits, mask):
    very_negative = torch.full_like(logits, -1e9)
    masked_logits = torch.where(mask > 0.0, logits, very_negative)
    best_actions = torch.argmax(masked_logits, dim=1)
    policy = torch.zeros_like(logits)
    policy.scatter_(1, best_actions.unsqueeze(1), 1.0)
    return policy


def plain_softmax_from_logits(logits):
    return torch.softmax(logits, dim=1)


def unconstrained_greedy_policy_from_logits(logits):
    best_actions = torch.argmax(logits, dim=1)
    policy = torch.zeros_like(logits)
    policy.scatter_(1, best_actions.unsqueeze(1), 1.0)
    return policy


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


# evaluate variant
def evaluate_variant(benchmark, variant_name, final_models_dir, results_dir):
    variant_dir = os.path.join(final_models_dir, variant_name)
    model_path = os.path.join(variant_dir, "model.pt")
    metrics_path = os.path.join(variant_dir, "metrics.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing checkpoint: " + model_path)

    saved_metrics = read_json(metrics_path)

    config_path = os.path.join(results_dir, variant_name, "variant_config.json")
    config = read_json(config_path)

    if not config:
        raise FileNotFoundError("Missing variant config: " + config_path)

    model = build_model(benchmark, config)
    model = load_model_weights(model, model_path, benchmark.device)

    x = benchmark.state_features_t
    admissible = benchmark.admissible_mask_t
    immediate_cost = benchmark.immediate_cost_t
    expert = benchmark.expert_safe_t
    smoothness_cost = benchmark.smoothness_cost_t

    use_action_mask = bool(config.get("use_action_mask", False))
    use_certified_nudging = bool(config.get("use_certified_nudging", True))

    rho_reward = float(config.get("rho_reward", 1.0))
    rho_cost = float(config.get("rho_cost", 1.0))

    lambda_inad = float(saved_metrics.get("lambda_inad", saved_metrics.get("lagrange", 0.0)))
    lambda_support = float(saved_metrics.get("lambda_support", 0.0))
    lambda_smooth = float(saved_metrics.get("lambda_smooth", 0.0))
    lambda_uncertainty = float(saved_metrics.get("lambda_uncertainty", 0.0))

    with torch.no_grad():
        outputs = model(x)

        logits = outputs["logits"]

        reward_lcb, _, reward_unc = reward_lcb_from_twin_q(
            outputs["q1"],
            outputs["q2"],
            rho_reward=rho_reward,
        )

        cost_ucb, _, cost_unc = cost_ucb_from_ensemble(
            outputs["cost"],
            rho_cost=rho_cost,
        )

        uncertainty_cost = reward_unc + cost_unc

        if use_action_mask:
            greedy_policy = masked_greedy_policy_from_logits(logits, admissible)
            soft_policy = masked_softmax_from_logits(logits, admissible)
            policy_rule = "masked greedy policy"
        else:
            rc_logits = risk_adjusted_logits_rc(
                logits=logits,
                reward_lcb=reward_lcb,
                cost_ucb=cost_ucb,
                immediate_cost=immediate_cost,
                expert_policy=expert,
                smoothness_cost=smoothness_cost,
                uncertainty_cost=uncertainty_cost,
                lambda_inad=lambda_inad,
                lambda_support=lambda_support,
                lambda_smooth=lambda_smooth,
                lambda_uncertainty=lambda_uncertainty,
                barrier_weight=float(config.get("barrier_weight", 25.0)),
                support_weight=float(config.get("support_weight", 1.0)),
                smoothness_weight=float(config.get("smoothness_weight", 0.001)),
                uncertainty_weight=float(config.get("uncertainty_weight", 0.1)),
                temperature=float(config.get("policy_temperature", 1.0)),
                energy_scale=float(config.get("energy_scale", 1.0)),
            )

            if use_certified_nudging:
                greedy_policy = certified_greedy_policy_rc(
                    energy_logits=rc_logits,
                    immediate_cost=immediate_cost,
                    cost_ucb=cost_ucb,
                    expert_policy=expert,
                    uncertainty_cost=uncertainty_cost,
                    cost_threshold=float(config.get("cost_threshold", 0.05)),
                    uncertainty_threshold=float(config.get("uncertainty_threshold", 1.0)),
                    expert_threshold=float(config.get("expert_threshold", 1e-8)),
                )
                policy_rule = "certified risk-adjusted greedy policy"
            else:
                greedy_policy = unconstrained_greedy_policy_from_logits(rc_logits)
                policy_rule = "uncertified risk-adjusted greedy policy"

            soft_policy = plain_softmax_from_logits(rc_logits)

    greedy_metrics = benchmark.exact_policy_evaluation(tensor_to_numpy(greedy_policy))
    soft_metrics = benchmark.exact_policy_evaluation(tensor_to_numpy(soft_policy))

    recomputed = dict(greedy_metrics)
    recomputed["soft_survival_rate"] = soft_metrics["survival_rate"]
    recomputed["soft_inadmissibility_rate"] = soft_metrics["inadmissibility_rate"]
    recomputed["soft_avg_return"] = soft_metrics["avg_return"]
    recomputed["soft_policy_entropy"] = soft_metrics["policy_entropy"]

    recomputed["lambda_inad"] = lambda_inad
    recomputed["lambda_support"] = lambda_support
    recomputed["lambda_smooth"] = lambda_smooth
    recomputed["lambda_uncertainty"] = lambda_uncertainty
    recomputed["use_certified_nudging"] = use_certified_nudging
    recomputed["uses_action_mask"] = use_action_mask

    comparison = {}
    for key, value in recomputed.items():
        if key in saved_metrics and isinstance(value, (int, float)):
            comparison[key] = {
                "saved": float(saved_metrics[key]),
                "recomputed": float(value),
                "abs_diff": abs(float(saved_metrics[key]) - float(value)),
            }

    return {
        "variant": variant_name,
        "variant_dir": variant_dir,
        "checkpoint_path": model_path,
        "config_path": config_path,
        "policy_rule": policy_rule,
        "saved_metrics": saved_metrics,
        "recomputed_metrics": recomputed,
        "saved_vs_recomputed": comparison,
    }


def to_json_safe(value):
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Test final LAADAN-AC-RC ablation models.")

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--final-models-dir",
        default=None,
        help="Defaults to <results-dir>/final_ablation_models.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Defaults to <results-dir>/final_ablation_models/test_summary.json.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Defaults to <results-dir>/final_ablation_models/test_summary.csv.",
    )

    args = parser.parse_args()

    device = choose_device(args.device)

    final_models_dir = args.final_models_dir or os.path.join(args.results_dir, "final_ablation_models")
    output_json = args.output_json or os.path.join(final_models_dir, "test_summary.json")
    output_csv = args.output_csv or os.path.join(final_models_dir, "test_summary.csv")

    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=int(args.horizon),
        device=device,
        use_one_hot_states=False,
    )

    results = []

    for variant_name in VARIANTS:
        variant_path = os.path.join(final_models_dir, variant_name)

        if not os.path.exists(variant_path):
            print("Skipping missing variant:", variant_path)
            continue

        print("\n" + "=" * 88)
        print("Testing variant:", variant_name)
        print("=" * 88)

        result = evaluate_variant(
            benchmark=benchmark,
            variant_name=variant_name,
            final_models_dir=final_models_dir,
            results_dir=args.results_dir,
        )

        results.append(result)

        m = result["recomputed_metrics"]
        print("Policy rule:", result["policy_rule"])
        print("survival_rate =", round(float(m["survival_rate"]), 6))
        print("inadmissibility_rate =", round(float(m["inadmissibility_rate"]), 6))
        print("soft_inadmissibility_rate =", round(float(m["soft_inadmissibility_rate"]), 12))
        print("lambda_inad =", round(float(m["lambda_inad"]), 6))

    if len(results) == 0:
        raise RuntimeError("No ablation final models were tested.")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    save_json(output_json, {
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "final_models_dir": final_models_dir,
        "horizon": int(args.horizon),
        "device": device,
        "variants": results,
    })

    fieldnames = [
        "variant",
        "survival_rate",
        "inadmissibility_rate",
        "soft_inadmissibility_rate",
        "avg_return",
        "soft_avg_return",
        "expert_argmax_match",
        "mean_kl_to_expert",
        "lambda_inad",
        "lambda_support",
        "lambda_smooth",
        "lambda_uncertainty",
        "uses_action_mask",
        "use_certified_nudging",
        "policy_rule",
        "checkpoint_path",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            m = result["recomputed_metrics"]
            writer.writerow({
                "variant": result["variant"],
                "survival_rate": m.get("survival_rate", ""),
                "inadmissibility_rate": m.get("inadmissibility_rate", ""),
                "soft_inadmissibility_rate": m.get("soft_inadmissibility_rate", ""),
                "avg_return": m.get("avg_return", ""),
                "soft_avg_return": m.get("soft_avg_return", ""),
                "expert_argmax_match": m.get("expert_argmax_match", ""),
                "mean_kl_to_expert": m.get("mean_kl_to_expert", ""),
                "lambda_inad": m.get("lambda_inad", ""),
                "lambda_support": m.get("lambda_support", ""),
                "lambda_smooth": m.get("lambda_smooth", ""),
                "lambda_uncertainty": m.get("lambda_uncertainty", ""),
                "uses_action_mask": m.get("uses_action_mask", ""),
                "use_certified_nudging": m.get("use_certified_nudging", ""),
                "policy_rule": result["policy_rule"],
                "checkpoint_path": result["checkpoint_path"],
            })

    print("\n" + "*" * 88)
    print("Finished ablation final-model testing.")
    print("JSON summary:", output_json)
    print("CSV summary:", output_csv)
    print("*" * 88)


if __name__ == "__main__":
    main()
