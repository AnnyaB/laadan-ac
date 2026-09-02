#!/usr/bin/env python3
"""Rebuild fixed-schedule aggregate artifacts from saved per-seed metrics only.

Use this after a result-tree cleanup or protocol-metadata correction. It never
retrains models and never invokes the benchmark evaluator. The script removes the
non-interpretable ``convergence_epoch_95`` field from final-only runs, recomputes
five-seed summaries, recomputes the paired LAADAN-minus-post-hoc comparison, and
optionally refreshes the recursive data hash manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_fixed_schedule import (  # noqa: E402
    SEEDS,
    dump_json,
    recursive_sha256_manifest,
)
from trainers import mean_ci95  # noqa: E402

METHODS = [
    ("Behavior Cloning", "bc"),
    ("CQL-regularised fitted-Q", "cql"),
    ("Vanilla Offline Actor-Critic", "voac"),
    ("LAADAN-AC", "laadan_ac"),
]
POSTHOC_NAME = "Post-hoc Masked VOAC"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_metrics(payload: Dict) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in payload.items()
        if key != "convergence_epoch_95"
        and isinstance(value, (int, float, np.integer, np.floating, bool))
        and np.isfinite(float(value))
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(groups: Dict[str, List[Dict[str, float]]]) -> Dict:
    summary = {}
    for method, seed_metrics in groups.items():
        common = set.intersection(*(set(metrics) for metrics in seed_metrics))
        summary[method] = {
            metric: mean_ci95([metrics[metric] for metrics in seed_metrics])
            for metric in sorted(common)
        }
        for payload in summary[method].values():
            values = [metrics[next(k for k, v in summary[method].items() if v is payload)] for metrics in seed_metrics]
            payload["min"] = float(np.min(values))
            payload["max"] = float(np.max(values))
    return summary


def aggregate_clean(groups: Dict[str, List[Dict[str, float]]]) -> Dict:
    """Aggregate with the same fields used by the released trainer summaries."""
    summary = {}
    for method, seed_metrics in groups.items():
        common = sorted(set.intersection(*(set(metrics) for metrics in seed_metrics)))
        summary[method] = {}
        for metric in common:
            values = [metrics[metric] for metrics in seed_metrics]
            stats = mean_ci95(values)
            stats["min"] = float(np.min(values))
            stats["max"] = float(np.max(values))
            summary[method][metric] = stats
    return summary


def paired_summary(
    laadan_by_seed: Dict[int, Dict[str, float]],
    posthoc_by_seed: Dict[int, Dict[str, float]],
) -> Dict:
    if set(laadan_by_seed) != set(posthoc_by_seed) != set(SEEDS):
        raise RuntimeError("Paired seed sets are inconsistent")
    common = sorted(
        set.intersection(
            *(set(metrics) for metrics in list(laadan_by_seed.values()) + list(posthoc_by_seed.values()))
        )
    )
    metrics = {}
    for metric in common:
        diffs = [
            laadan_by_seed[seed][metric] - posthoc_by_seed[seed][metric]
            for seed in SEEDS
        ]
        metrics[metric] = {
            "per_seed_difference": [float(value) for value in diffs],
            **mean_ci95(diffs),
        }
    return {
        "comparison": "LAADAN-AC minus Post-hoc Masked VOAC",
        "seeds": list(SEEDS),
        "metrics": metrics,
    }


def rebuild(dataset_root: Path, data_dir: Path | None) -> None:
    groups: Dict[str, List[Dict[str, float]]] = {name: [] for name, _ in METHODS}
    groups[POSTHOC_NAME] = []
    rows: List[Dict] = []
    laadan_by_seed: Dict[int, Dict[str, float]] = {}
    posthoc_by_seed: Dict[int, Dict[str, float]] = {}

    for seed in SEEDS:
        for method_name, folder in METHODS:
            path = dataset_root / "main_fixed_schedule" / folder / f"seed_{seed}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            metrics = numeric_metrics(load_json(path))
            groups[method_name].append(metrics)
            rows.append({"method": method_name, "seed": int(seed), **metrics})
            if method_name == "LAADAN-AC":
                laadan_by_seed[int(seed)] = metrics

        path = dataset_root / "posthoc_masked_voac" / f"seed_{seed}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics = numeric_metrics(load_json(path))
        groups[POSTHOC_NAME].append(metrics)
        rows.append({"method": POSTHOC_NAME, "seed": int(seed), **metrics})
        posthoc_by_seed[int(seed)] = metrics

    aggregate_dir = dataset_root / "aggregate"
    dump_json(aggregate_dir / "main_summary.json", aggregate_clean(groups))
    write_csv(aggregate_dir / "per_seed_metrics.csv", rows)
    dump_json(
        aggregate_dir / "paired_laadan_minus_posthoc_voac.json",
        paired_summary(laadan_by_seed, posthoc_by_seed),
    )
    if data_dir is not None:
        dump_json(dataset_root / "data_sha256.json", recursive_sha256_manifest(data_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="results/fixed_schedule_2026/icu_sepsis"
    )
    parser.add_argument("--data-dir", default="data/icu_sepsis")
    parser.add_argument(
        "--skip-data-manifest",
        action="store_true",
        help="Do not refresh data_sha256.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rebuild(
        Path(args.root),
        None if args.skip_data_manifest else Path(args.data_dir),
    )
    print(f"PASS: aggregate artifacts rebuilt from saved seed metrics under {args.root}")


if __name__ == "__main__":
    main()
