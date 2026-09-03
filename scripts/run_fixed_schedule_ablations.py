#!/usr/bin/env python3
"""Complete the ICU-Sepsis fixed-schedule leave-one-out ablation suite.

This script is intentionally separate from the main fixed-schedule runner so the
already frozen BC/CQL/VOAC/LAADAN headline runs are not retrained. By default it
runs only the four leave-one-out variants that were not part of the initial core
reviewer-response experiment, then rebuilds the complete seven-variant aggregate
files from metrics saved on disk.

The full LAADAN-AC row is taken directly from the already frozen main
``main_fixed_schedule/laadan_ac`` runs. It is not redundantly retrained as an
"ablation" variant.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark import ICUSepsisOfflineBenchmark  # noqa: E402
from ablation_engine import BASE_LAADAN_CONFIG, BudgetedLagrangianExperiment  # noqa: E402
from run_fixed_schedule import (  # noqa: E402
    EPOCHS,
    FINAL_ONLY_EVAL_EVERY,
    SEEDS,
    ablation_variants,
    dump_json,
    validate_final_only_history,
)
from trainers import mean_ci95  # noqa: E402

DEFAULT_NEW_VARIANTS = ("no_expert_kl", "no_smoothness", "no_lagrangian", "no_mask")
PERTURBATION_VARIANTS = {
    variant["folder"]: variant
    for variant in ablation_variants("full")
    if variant["folder"] != "full_laadan_ac"
}
DISPLAY_NAME_BY_FOLDER = {
    folder: variant["name"] for folder, variant in PERTURBATION_VARIANTS.items()
}
PERTURBATION_FOLDERS = tuple(DISPLAY_NAME_BY_FOLDER)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested


def existing_seed_dirs(ablation_root: Path, folder: str) -> List[int]:
    return [
        seed
        for seed in SEEDS
        if (ablation_root / folder / f"seed_{seed}" / "metrics.json").is_file()
    ]


def run_requested_variants(
    benchmark,
    output_root: Path,
    requested_folders: List[str],
    overwrite: bool,
) -> None:
    unknown = sorted(set(requested_folders) - set(PERTURBATION_VARIANTS))
    if unknown:
        raise ValueError(
            "Unknown/non-perturbation ablation folders: " + ", ".join(unknown)
        )

    ablation_root = output_root / "ablations" / "lagrangian_frontier"
    selected = []
    for folder in requested_folders:
        present = existing_seed_dirs(ablation_root, folder)
        if present and not overwrite:
            if present == list(SEEDS):
                print(f"[SKIP] {folder}: all five seed metrics already exist.")
                continue
            raise RuntimeError(
                f"Partial existing result for {folder}: seeds {present}. "
                "Refusing to mix partial reruns; inspect/remove that folder or use --overwrite intentionally."
            )
        selected.append(PERTURBATION_VARIANTS[folder])

    if not selected:
        print("[INFO] No ablation training required.")
        return

    cfg = deepcopy(BASE_LAADAN_CONFIG)
    cfg["epochs"] = EPOCHS
    cfg["eval_every"] = FINAL_ONLY_EVAL_EVERY
    experiment = BudgetedLagrangianExperiment(
        benchmark, str(output_root / "ablations"), cfg
    )
    groups = experiment.run_variants(selected, SEEDS)
    for runs in groups.values():
        for run in runs:
            validate_final_only_history(run)


def numeric_metrics(metrics: Dict) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key != "convergence_epoch_95"
        and isinstance(value, (int, float, np.integer, np.floating, bool))
        and np.isfinite(float(value))
    }


def rebuild_aggregates(output_root: Path) -> None:
    ablation_root = output_root / "ablations" / "lagrangian_frontier"
    rows: List[Dict] = []
    grouped_values: Dict[str, List[Dict[str, float]]] = {"Full LAADAN-AC": []}

    # The full row is the same fixed-schedule LAADAN-AC model used in the main
    # comparison; reusing these exact files avoids an unnecessary duplicate run.
    for seed in SEEDS:
        metrics_path = (
            output_root
            / "main_fixed_schedule"
            / "laadan_ac"
            / f"seed_{seed}"
            / "metrics.json"
        )
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing frozen full LAADAN metrics: {metrics_path}")
        numeric = numeric_metrics(load_json(metrics_path))
        grouped_values["Full LAADAN-AC"].append(numeric)
        rows.append({"method": "Full LAADAN-AC", "seed": int(seed), **numeric})

    for folder in PERTURBATION_FOLDERS:
        display_name = DISPLAY_NAME_BY_FOLDER[folder]
        grouped_values[display_name] = []
        for seed in SEEDS:
            metrics_path = ablation_root / folder / f"seed_{seed}" / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(
                    f"Full ablation suite incomplete: missing {metrics_path}"
                )
            numeric = numeric_metrics(load_json(metrics_path))
            grouped_values[display_name].append(numeric)
            rows.append({"method": display_name, "seed": int(seed), **numeric})

    summary = {}
    for method, seed_metrics in grouped_values.items():
        common = set.intersection(*(set(metrics) for metrics in seed_metrics))
        summary[method] = {
            metric: mean_ci95([metrics[metric] for metrics in seed_metrics])
            for metric in sorted(common)
        }

    aggregate_dir = output_root / "aggregate"
    dump_json(aggregate_dir / "ablation_summary.json", summary)
    write_csv(aggregate_dir / "ablation_per_seed_metrics.csv", rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/icu_sepsis")
    parser.add_argument(
        "--output-root", default="results/fixed_schedule_2026/icu_sepsis"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_NEW_VARIANTS),
        help=(
            "Comma-separated perturbation folder names. Defaults to the four missing leave-one-out variants: "
            + ",".join(DEFAULT_NEW_VARIANTS)
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally rerun and replace a complete existing perturbation. Off by default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    output_root = Path(args.output_root)
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]

    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir, horizon=50, device=device, use_one_hot_states=False
    )
    run_requested_variants(
        benchmark,
        output_root,
        requested_folders=requested,
        overwrite=args.overwrite,
    )
    rebuild_aggregates(output_root)
    print("PASS: complete seven-variant fixed-schedule ablation aggregates rebuilt.")


if __name__ == "__main__":
    main()
