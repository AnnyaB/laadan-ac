import argparse
import copy
import csv
import json
import math
import os
import shutil

import torch

from benchmark import ICUSepsisOfflineBenchmark
from trainers import train_laadan_ac_primal_dual, ensure_dir


# ablation experiments
# select device
def choose_device(requested):
    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"


# save results
def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def metric_float(metrics, key, default):
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def constrained_run_is_better(metrics, best_metrics, tolerance=1e-8):
    if best_metrics is None:
        return True

    budget = metric_float(metrics, "cost_budget", 0.0001)
    best_budget = metric_float(best_metrics, "cost_budget", budget)

    current_greedy = metric_float(metrics, "inadmissibility_rate", math.inf)
    current_soft = metric_float(metrics, "soft_inadmissibility_rate", current_greedy)
    current_cost = max(current_greedy, current_soft)

    best_greedy = metric_float(best_metrics, "inadmissibility_rate", math.inf)
    best_soft = metric_float(best_metrics, "soft_inadmissibility_rate", best_greedy)
    best_cost = max(best_greedy, best_soft)

    current_return = metric_float(
        metrics,
        "survival_rate",
        metric_float(metrics, "avg_return", -math.inf),
    )

    best_return = metric_float(
        best_metrics,
        "survival_rate",
        metric_float(best_metrics, "avg_return", -math.inf),
    )

    current_feasible = current_cost <= budget + tolerance
    best_feasible = best_cost <= best_budget + tolerance

    if current_feasible and not best_feasible:
        return True

    if current_feasible and best_feasible:
        return current_return > best_return

    if not current_feasible and not best_feasible:
        if current_cost < best_cost - tolerance:
            return True
        if abs(current_cost - best_cost) <= tolerance and current_return > best_return:
            return True

    return False


def base_rc_config():
    return {
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

        "num_cost_heads": 3,
        "rho_reward": 1.0,
        "rho_cost": 1.0,

        "conservative_alpha": 0.5,
        "expert_kl_weight": 0.005,

        "use_action_mask": False,

        "cost_budget": 0.0001,
        "support_budget": 0.25,
        "smooth_budget": 1.0,
        "uncertainty_budget": 0.25,

        "lambda_inad_init": 0.0,
        "lambda_support_init": 0.0,
        "lambda_smooth_init": 0.0,
        "lambda_uncertainty_init": 0.0,

        "lambda_inad_lr": 0.01,
        "lambda_support_lr": 0.001,
        "lambda_smooth_lr": 0.001,
        "lambda_uncertainty_lr": 0.001,

        "lagrange_max": 100.0,
        "lambda_update_every": 10,

        "barrier_weight": 25.0,
        "support_weight": 1.0,
        "smoothness_weight": 0.001,
        "uncertainty_weight": 0.1,

        "use_certified_nudging": True,
        "policy_temperature": 1.0,
        "energy_scale": 1.0,
        "cost_threshold": 0.05,
        "uncertainty_threshold": 1.0,
        "expert_threshold": 1e-8,
    }


def make_ablation_configs(base):
    configs = {}

    configs["full_rc"] = copy.deepcopy(base)

    no_lambda = copy.deepcopy(base)
    no_lambda["lambda_inad_lr"] = 0.0
    no_lambda["lambda_support_lr"] = 0.0
    no_lambda["lambda_smooth_lr"] = 0.0
    no_lambda["lambda_uncertainty_lr"] = 0.0
    no_lambda["lambda_inad_init"] = 0.0
    no_lambda["lambda_support_init"] = 0.0
    no_lambda["lambda_smooth_init"] = 0.0
    no_lambda["lambda_uncertainty_init"] = 0.0
    configs["no_lambda_updates"] = no_lambda

    no_barrier = copy.deepcopy(base)
    no_barrier["barrier_weight"] = 0.0
    configs["no_barrier"] = no_barrier

    no_nudge = copy.deepcopy(base)
    no_nudge["use_certified_nudging"] = False
    configs["no_certified_nudging"] = no_nudge

    no_support_lambda = copy.deepcopy(base)
    no_support_lambda["lambda_support_lr"] = 0.0
    no_support_lambda["lambda_support_init"] = 0.0
    configs["no_support_lambda"] = no_support_lambda

    no_cost_ucb = copy.deepcopy(base)
    no_cost_ucb["rho_cost"] = 0.0
    configs["no_cost_ucb"] = no_cost_ucb

    lambda_only = copy.deepcopy(base)
    lambda_only["use_certified_nudging"] = False
    lambda_only["barrier_weight"] = 0.0
    lambda_only["support_weight"] = 0.0
    lambda_only["smoothness_weight"] = 0.0
    lambda_only["uncertainty_weight"] = 0.0
    lambda_only["expert_kl_weight"] = 0.0
    lambda_only["lambda_support_lr"] = 0.0
    lambda_only["lambda_smooth_lr"] = 0.0
    lambda_only["lambda_uncertainty_lr"] = 0.0
    lambda_only["lambda_support_init"] = 0.0
    lambda_only["lambda_smooth_init"] = 0.0
    lambda_only["lambda_uncertainty_init"] = 0.0
    lambda_only["lambda_inad_lr"] = 0.05
    lambda_only["lagrange_max"] = 500.0
    configs["lambda_only_no_mask"] = lambda_only

    return configs


def shrink_for_debug(config):
    config = copy.deepcopy(config)
    config["epochs"] = 40
    config["eval_every"] = 5
    config["lambda_update_every"] = 5
    return config


def copy_best_seed_to_final(variant_dir, final_dir, variant_name, best_seed):
    src = os.path.join(variant_dir, "laadan_ac_pd_nomask", "seed_" + str(best_seed))
    dst = os.path.join(final_dir, variant_name)

    if os.path.exists(dst):
        shutil.rmtree(dst)

    os.makedirs(dst, exist_ok=True)

    for filename in ["model.pt", "metrics.json", "history.csv"]:
        src_file = os.path.join(src, filename)
        dst_file = os.path.join(dst, filename)

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print("Warning: missing file:", src_file)


# run variant
def run_variant(benchmark, output_root, variant_name, config, seeds):
    variant_dir = os.path.join(output_root, variant_name)
    ensure_dir(variant_dir)

    save_json(os.path.join(variant_dir, "variant_config.json"), config)

    runs = []

    for seed in seeds:
        print("\n" + "=" * 88)
        print("Running variant:", variant_name, "| seed:", seed)
        print("=" * 88)

        run = train_laadan_ac_primal_dual(
            benchmark=benchmark,
            seed=int(seed),
            results_dir=variant_dir,
            config=config,
        )

        runs.append(run)

    best_seed = None
    best_metrics = None

    for run in runs:
        if constrained_run_is_better(run["metrics"], best_metrics):
            best_seed = int(run["seed"])
            best_metrics = dict(run["metrics"])

    return {
        "variant": variant_name,
        "best_seed": best_seed,
        "best_metrics": best_metrics,
        "runs": runs,
        "variant_dir": variant_dir,
    }


def row_from_result(result):
    metrics = result["best_metrics"] or {}

    return {
        "variant": result["variant"],
        "best_seed": result["best_seed"],
        "survival_rate": metrics.get("survival_rate", ""),
        "inadmissibility_rate": metrics.get("inadmissibility_rate", ""),
        "soft_inadmissibility_rate": metrics.get("soft_inadmissibility_rate", ""),
        "avg_return": metrics.get("avg_return", ""),
        "soft_avg_return": metrics.get("soft_avg_return", ""),
        "expert_argmax_match": metrics.get("expert_argmax_match", ""),
        "mean_kl_to_expert": metrics.get("mean_kl_to_expert", ""),
        "policy_entropy": metrics.get("policy_entropy", ""),
        "soft_policy_entropy": metrics.get("soft_policy_entropy", ""),
        "cost_budget": metrics.get("cost_budget", ""),
        "lagrange": metrics.get("lagrange", ""),
        "lambda_inad": metrics.get("lambda_inad", ""),
        "lambda_support": metrics.get("lambda_support", ""),
        "lambda_smooth": metrics.get("lambda_smooth", ""),
        "lambda_uncertainty": metrics.get("lambda_uncertainty", ""),
        "uses_action_mask": metrics.get("uses_action_mask", ""),
        "use_certified_nudging": metrics.get("use_certified_nudging", ""),
        "barrier_weight": metrics.get("barrier_weight", ""),
        "support_weight": metrics.get("support_weight", ""),
        "rho_cost": metrics.get("rho_cost", ""),
    }


def write_summary(output_root, results):
    rows = [row_from_result(result) for result in results]

    csv_path = os.path.join(output_root, "rc_ablation_summary.csv")
    json_path = os.path.join(output_root, "rc_ablation_summary.json")

    fieldnames = [
        "variant",
        "best_seed",
        "survival_rate",
        "inadmissibility_rate",
        "soft_inadmissibility_rate",
        "avg_return",
        "soft_avg_return",
        "expert_argmax_match",
        "mean_kl_to_expert",
        "policy_entropy",
        "soft_policy_entropy",
        "cost_budget",
        "lagrange",
        "lambda_inad",
        "lambda_support",
        "lambda_smooth",
        "lambda_uncertainty",
        "uses_action_mask",
        "use_certified_nudging",
        "barrier_weight",
        "support_weight",
        "rho_cost",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    safe_results = []

    for result in results:
        safe_results.append({
            "variant": result["variant"],
            "best_seed": result["best_seed"],
            "best_metrics": result["best_metrics"],
            "variant_dir": result["variant_dir"],
        })

    save_json(json_path, safe_results)

    print("\nSaved ablation CSV:", csv_path)
    print("Saved ablation JSON:", json_path)

    print("\nAblation summary:")
    for row in rows:
        print(
            row["variant"],
            "| seed=", row["best_seed"],
            "| survival=", row["survival_rate"],
            "| inad=", row["inadmissibility_rate"],
            "| soft_inad=", row["soft_inadmissibility_rate"],
            "| lambda_inad=", row["lambda_inad"],
        )


def main():
    parser = argparse.ArgumentParser(description="Run LAADAN-AC-RC mechanism ablations.")

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", default="results_rc_ablations")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["debug", "main"], default="debug")

    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of variants to run.",
    )

    args = parser.parse_args()

    device = choose_device(args.device)

    seeds = [42, 43, 44, 45, 46]

    if args.mode == "debug":
        seeds = [42]

    ensure_dir(args.results_dir)

    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=50,
        device=device,
        use_one_hot_states=False,
    )

    base = base_rc_config()

    if args.mode == "debug":
        base = shrink_for_debug(base)

    configs = make_ablation_configs(base)

    if args.variants is not None and len(args.variants) > 0:
        missing = [name for name in args.variants if name not in configs]

        if missing:
            raise ValueError("Unknown variants requested: " + str(missing))

        configs = {name: configs[name] for name in args.variants}

    save_json(os.path.join(args.results_dir, "ablation_master_config.json"), configs)

    all_results = []

    final_dir = os.path.join(args.results_dir, "final_ablation_models")
    ensure_dir(final_dir)

    for variant_name, config in configs.items():
        result = run_variant(
            benchmark=benchmark,
            output_root=args.results_dir,
            variant_name=variant_name,
            config=config,
            seeds=seeds,
        )

        all_results.append(result)

        if result["best_seed"] is not None:
            copy_best_seed_to_final(
                variant_dir=result["variant_dir"],
                final_dir=final_dir,
                variant_name=variant_name,
                best_seed=result["best_seed"],
            )

    write_summary(args.results_dir, all_results)

    print("\nFinished LAADAN-AC-RC ablations.")
    print("All results saved under:", args.results_dir)
    print("Best selected models saved under:", final_dir)


if __name__ == "__main__":
    main()
