
# pick_final_models.py
#
# Selects the best seed checkpoint for each trained model family and copies it
# into a clean final_models/ folder.
#
# Supports:
# - BC
# - CQL
# - VOAC
# - LAADAN-AC
# - LAADAN-AC-PD / LAADAN-AC-RC no-mask variant

import argparse
import csv
import json
import math
import os
import shutil


AGENTS = [
    {
        "folder": "bc",
        "name": "Behavior Cloning",
        "selection_rule": "highest_survival",
    },
    {
        "folder": "cql",
        "name": "Conservative Q-Learning",
        "selection_rule": "highest_survival",
    },
    {
        "folder": "voac",
        "name": "Vanilla Offline Actor-Critic",
        "selection_rule": "highest_survival",
    },
    {
        "folder": "laadan_ac",
        "name": "LAADAN-AC",
        "selection_rule": "highest_survival",
    },
    {
        "folder": "laadan_ac_pd_nomask",
        "name": "LAADAN-AC-PD / LAADAN-AC-RC",
        "selection_rule": "constrained",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick final selected checkpoints from multi-seed experiment results."
    )

    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root results directory containing per-agent seed folders. Example: results_icu_sepsis_LAADAN_AC_PD",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for final selected models. Defaults to <results-dir>/final_models.",
    )

    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="If set, remove the existing final_models folder before copying selected models.",
    )

    return parser.parse_args()


def looks_like_results_dir(path):
    if not os.path.isdir(path):
        return False

    if not os.path.exists(os.path.join(path, "run_config.json")):
        return False

    for agent in AGENTS:
        if os.path.isdir(os.path.join(path, agent["folder"])):
            return True

    return False


def auto_find_results_dir():
    candidates = []

    for name in sorted(os.listdir(".")):
        if not os.path.isdir(name):
            continue

        if looks_like_results_dir(name):
            candidates.append(name)

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-detect a results folder. "
            "Please run with --results-dir results_icu_sepsis_LAADAN_AC_PD"
        )

    # Prefer the most recently modified valid results folder.
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = candidates[0]

    print("Auto-detected results folder:", chosen)

    return chosen


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_float(metrics, key, default):
    try:
        return float(metrics.get(key, default))
    except Exception:
        return float(default)


def standard_seed_is_better(metrics, best_metrics):
    if best_metrics is None:
        return True

    survival = metric_float(metrics, "survival_rate", -math.inf)
    inad = metric_float(metrics, "inadmissibility_rate", math.inf)

    best_survival = metric_float(best_metrics, "survival_rate", -math.inf)
    best_inad = metric_float(best_metrics, "inadmissibility_rate", math.inf)

    if survival > best_survival:
        return True

    if survival == best_survival and inad < best_inad:
        return True

    return False


def constrained_seed_is_better(metrics, best_metrics, tolerance=1e-8):
    """
    Constrained selection for LAADAN-AC-PD / LAADAN-AC-RC.

    Priority:
    1. Feasible checkpoint first.
    2. Among feasible checkpoints, highest survival.
    3. If no feasible checkpoint exists, lowest constraint cost.
    """

    if best_metrics is None:
        return True

    budget = metric_float(metrics, "cost_budget", 0.0001)
    best_budget = metric_float(best_metrics, "cost_budget", budget)

    current_greedy_cost = metric_float(metrics, "inadmissibility_rate", math.inf)
    current_soft_cost = metric_float(metrics, "soft_inadmissibility_rate", current_greedy_cost)
    current_cost = max(current_greedy_cost, current_soft_cost)

    best_greedy_cost = metric_float(best_metrics, "inadmissibility_rate", math.inf)
    best_soft_cost = metric_float(best_metrics, "soft_inadmissibility_rate", best_greedy_cost)
    best_cost = max(best_greedy_cost, best_soft_cost)

    current_return = metric_float(metrics, "survival_rate", metric_float(metrics, "avg_return", -math.inf))
    best_return = metric_float(best_metrics, "survival_rate", metric_float(best_metrics, "avg_return", -math.inf))

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


def choose_best_seed(agent, source_dir):
    best_seed = None
    best_metrics = None

    for folder in sorted(os.listdir(source_dir)):
        if not folder.startswith("seed_"):
            continue

        seed_dir = os.path.join(source_dir, folder)
        metrics_path = os.path.join(seed_dir, "metrics.json")

        if not os.path.isdir(seed_dir):
            continue

        if not os.path.exists(metrics_path):
            continue

        metrics = read_json(metrics_path)

        if agent["selection_rule"] == "constrained":
            is_better = constrained_seed_is_better(metrics, best_metrics)
        else:
            is_better = standard_seed_is_better(metrics, best_metrics)

        if is_better:
            best_seed = folder
            best_metrics = metrics

    return best_seed, best_metrics


def copy_selected_files(src_seed_dir, dst_agent_dir):
    os.makedirs(dst_agent_dir, exist_ok=True)

    for filename in ["model.pt", "metrics.json", "history.csv"]:
        src = os.path.join(src_seed_dir, filename)
        dst = os.path.join(dst_agent_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print("Warning: missing file, not copied:", src)


def main():
    args = parse_args()

    if args.results_dir is None:
        results_dir = auto_find_results_dir()
    else:
        results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        raise FileNotFoundError("Results directory does not exist: " + results_dir)

    final_root = args.output_dir or os.path.join(results_dir, "final_models")

    if args.clear_output and os.path.exists(final_root):
        shutil.rmtree(final_root)

    os.makedirs(final_root, exist_ok=True)

    rows = []

    for agent in AGENTS:
        agent_dir = agent["folder"]
        agent_name = agent["name"]

        source_dir = os.path.join(results_dir, agent_dir)

        if not os.path.isdir(source_dir):
            print("Skipping missing agent folder:", source_dir)
            continue

        best_seed, best_metrics = choose_best_seed(agent, source_dir)

        if best_seed is None or best_metrics is None:
            print("Skipping agent with no valid seed metrics:", agent_dir)
            continue

        src_seed_dir = os.path.join(source_dir, best_seed)
        dst_agent_dir = os.path.join(final_root, agent_dir)

        copy_selected_files(src_seed_dir, dst_agent_dir)

        rows.append({
            "agent_dir": agent_dir,
            "agent_name": agent_name,
            "best_seed": best_seed,
            "selection_rule": agent["selection_rule"],
            "survival_rate": best_metrics.get("survival_rate", ""),
            "inadmissibility_rate": best_metrics.get("inadmissibility_rate", ""),
            "soft_inadmissibility_rate": best_metrics.get("soft_inadmissibility_rate", ""),
            "avg_return": best_metrics.get("avg_return", ""),
            "cost_budget": best_metrics.get("cost_budget", ""),
            "lagrange": best_metrics.get("lagrange", ""),
            "lambda_inad": best_metrics.get("lambda_inad", ""),
            "lambda_support": best_metrics.get("lambda_support", ""),
            "lambda_smooth": best_metrics.get("lambda_smooth", ""),
            "lambda_uncertainty": best_metrics.get("lambda_uncertainty", ""),
        })

    if len(rows) == 0:
        raise RuntimeError(
            "No final models were selected. "
            "You probably used the wrong --results-dir. "
            "Expected something like: --results-dir results_icu_sepsis_LAADAN_AC_PD"
        )

    csv_path = os.path.join(final_root, "final_model_selection.csv")

    fieldnames = [
        "agent_dir",
        "agent_name",
        "best_seed",
        "selection_rule",
        "survival_rate",
        "inadmissibility_rate",
        "soft_inadmissibility_rate",
        "avg_return",
        "cost_budget",
        "lagrange",
        "lambda_inad",
        "lambda_support",
        "lambda_smooth",
        "lambda_uncertainty",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved final selected models to:", final_root)
    print("Selection table:", csv_path)

    print("\nSelected models:")
    for row in rows:
        print(
            row["agent_name"],
            "| seed=",
            row["best_seed"],
            "| survival=",
            row["survival_rate"],
            "| inad=",
            row["inadmissibility_rate"],
            "| soft_inad=",
            row["soft_inadmissibility_rate"],
        )


if __name__ == "__main__":
    main()
