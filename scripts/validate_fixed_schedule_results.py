#!/usr/bin/env python3
"""Integrity checks for LAADAN-AC fixed-schedule experiment releases.

The validator recomputes aggregate summaries from per-seed files, verifies the
pre-specified final-checkpoint protocol, checks paired-seed correspondence,
validates the recursive data manifest, and confirms that expected artifacts are
present without relying on manuscript tables or README numbers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SEEDS = (42, 43, 44, 45, 46)
MAIN_METHOD_DIRS = ("bc", "cql", "voac", "laadan_ac")
EXPECTED_METHOD_NAMES = {
    "Behavior Cloning",
    "CQL-regularised fitted-Q",
    "Vanilla Offline Actor-Critic",
    "Post-hoc Masked VOAC",
    "LAADAN-AC",
}
CORE_ABLATION_METHODS = {
    "Full LAADAN-AC",
    "Masking-only actor-critic",
    "LAADAN without conservative critic",
}
FULL_ABLATION_METHODS = CORE_ABLATION_METHODS | {
    "LAADAN without expert KL",
    "LAADAN without smoothness proxy",
    "LAADAN without Lagrangian cost control",
    "LAADAN without action mask",
}
CORE_PERTURBATION_FOLDERS = {"mask_only", "no_conservative"}
FULL_PERTURBATION_FOLDERS = CORE_PERTURBATION_FOLDERS | {
    "no_expert_kl",
    "no_smoothness",
    "no_lagrangian",
    "no_mask",
}
T_CRIT_95 = {
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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recursive_manifest(data_dir: Path) -> Dict[str, str]:
    return {
        path.relative_to(data_dir).as_posix(): sha256_file(path)
        for path in sorted(p for p in data_dir.rglob("*") if p.is_file())
    }


def numeric_row(row: Dict[str, str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in row.items():
        if key in {"method", "seed", ""} or value in {None, ""}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            raise AssertionError(f"Non-finite value in row {row}: {key}={value}")
        result[key] = number
    return result


def mean_ci95(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise AssertionError("Cannot summarize an empty metric vector")
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"Non-finite values: {arr}")
    mean = float(np.mean(arr))
    if arr.size == 1:
        std = 0.0
        half = 0.0
    else:
        std = float(np.std(arr, ddof=1))
        half = float(T_CRIT_95.get(int(arr.size), 1.96) * std / np.sqrt(arr.size))
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95_half": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-10) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=atol):
        raise AssertionError(f"{label}: {actual} != {expected}")


def validate_data_manifest(result_root: Path, data_dir: Path) -> None:
    recorded = load_json(result_root / "data_sha256.json")
    current = recursive_manifest(data_dir)
    if recorded != current:
        missing = sorted(set(recorded) - set(current))
        extra = sorted(set(current) - set(recorded))
        mismatched = sorted(
            key for key in set(recorded) & set(current) if recorded[key] != current[key]
        )
        raise AssertionError(
            "Data manifest mismatch. "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )


def validate_main_runs(root: Path) -> Dict[str, List[Dict[str, str]]]:
    rows = load_csv(root / "aggregate" / "per_seed_metrics.csv")
    methods = {row["method"] for row in rows}
    if methods != EXPECTED_METHOD_NAMES:
        raise AssertionError(
            f"Expected methods {sorted(EXPECTED_METHOD_NAMES)}, found {sorted(methods)}"
        )

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for method in sorted(EXPECTED_METHOD_NAMES):
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted(int(row["seed"]) for row in method_rows)
        if seeds != list(SEEDS):
            raise AssertionError(f"{method}: expected seeds {SEEDS}, found {seeds}")
        for row in method_rows:
            numeric_row(row)
        grouped[method] = method_rows

    for method_dir in MAIN_METHOD_DIRS:
        for seed in SEEDS:
            seed_dir = root / "main_fixed_schedule" / method_dir / f"seed_{seed}"
            for filename in ("history.csv", "metrics.json", "model.pt", "provenance.json"):
                if not (seed_dir / filename).is_file():
                    raise AssertionError(f"Missing artifact: {seed_dir / filename}")
            provenance = load_json(seed_dir / "provenance.json")
            assert provenance["checkpoint_rule"] == "pre-specified final epoch"
            assert int(provenance["checkpoint_epoch"]) == 1000
            assert provenance["benchmark_evaluation_used_for_model_selection"] is False
            assert provenance["headline_seed_selection"] is False
            evaluated = provenance.get("evaluated_epochs", [1000])
            if evaluated != [1000]:
                raise AssertionError(f"{seed_dir}: evaluated_epochs={evaluated}")

    for seed in SEEDS:
        posthoc_dir = root / "posthoc_masked_voac" / f"seed_{seed}"
        for filename in ("metrics.json", "provenance.json"):
            if not (posthoc_dir / filename).is_file():
                raise AssertionError(f"Missing artifact: {posthoc_dir / filename}")
        provenance = load_json(posthoc_dir / "provenance.json")
        assert provenance["mask_applied_during_training"] is False
        assert provenance["mask_applied_at_evaluation"] is True
        assert provenance["parameters_updated_during_posthoc_evaluation"] is False
        checkpoint = Path(provenance["source_voac_checkpoint"])
        if not checkpoint.is_file():
            raise AssertionError(f"Broken VOAC checkpoint reference: {checkpoint}")

    return grouped


def validate_zero_posthoc_inadmissibility(grouped: Dict[str, List[Dict[str, str]]]) -> None:
    for row in grouped["Post-hoc Masked VOAC"]:
        if abs(float(row["inadmissibility_rate"])) > 1e-12:
            raise AssertionError(f"Post-hoc masked VOAC is inadmissible: {row}")


def validate_aggregate_summary(root: Path, grouped: Dict[str, List[Dict[str, str]]]) -> None:
    saved = load_json(root / "aggregate" / "main_summary.json")
    if "convergence_epoch_95" in json.dumps(saved):
        raise AssertionError(
            "convergence_epoch_95 must not be reported for final-only benchmark evaluation"
        )

    for method, rows in grouped.items():
        if method not in saved:
            raise AssertionError(f"Missing method in main_summary.json: {method}")
        metric_names = sorted(set.intersection(*(set(numeric_row(row)) for row in rows)))
        for metric in metric_names:
            values = [numeric_row(row)[metric] for row in rows]
            recomputed = mean_ci95(values)
            recorded = saved[method].get(metric)
            if recorded is None:
                raise AssertionError(f"Missing {method}/{metric} in main_summary.json")
            for field, value in recomputed.items():
                if field not in recorded:
                    raise AssertionError(f"Missing {method}/{metric}/{field}")
                assert_close(recorded[field], value, f"{method}/{metric}/{field}")


def validate_paired_summary(root: Path, grouped: Dict[str, List[Dict[str, str]]]) -> None:
    saved = load_json(root / "aggregate" / "paired_laadan_minus_posthoc_voac.json")
    if saved["seeds"] != list(SEEDS):
        raise AssertionError(f"Paired summary has wrong seeds: {saved['seeds']}")

    laadan = {int(row["seed"]): numeric_row(row) for row in grouped["LAADAN-AC"]}
    posthoc = {
        int(row["seed"]): numeric_row(row) for row in grouped["Post-hoc Masked VOAC"]
    }
    common_metrics = sorted(
        set.intersection(*(set(v) for v in list(laadan.values()) + list(posthoc.values())))
    )

    for metric in common_metrics:
        diffs = [laadan[seed][metric] - posthoc[seed][metric] for seed in SEEDS]
        recomputed = mean_ci95(diffs)
        recorded = saved["metrics"].get(metric)
        if recorded is None:
            raise AssertionError(f"Paired summary missing metric: {metric}")
        recorded_diffs = recorded["per_seed_difference"]
        if len(recorded_diffs) != len(diffs):
            raise AssertionError(f"Paired diff count mismatch for {metric}")
        for idx, (actual, expected) in enumerate(zip(recorded_diffs, diffs)):
            assert_close(actual, expected, f"paired/{metric}/seed_index_{idx}")
        for field, value in recomputed.items():
            if field in recorded:
                assert_close(recorded[field], value, f"paired/{metric}/{field}")


def validate_ablations(root: Path, expected_level: str) -> None:
    if expected_level == "none":
        return
    csv_path = root / "aggregate" / "ablation_per_seed_metrics.csv"
    json_path = root / "aggregate" / "ablation_summary.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise AssertionError("Missing fixed-schedule ablation aggregate artifacts")

    rows = load_csv(csv_path)
    expected_methods = (
        FULL_ABLATION_METHODS if expected_level == "full" else CORE_ABLATION_METHODS
    )
    method_names = {row["method"] for row in rows}
    if method_names != expected_methods:
        raise AssertionError(
            f"Expected ablation methods {sorted(expected_methods)}, found {sorted(method_names)}"
        )
    for method in sorted(method_names):
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted(int(row["seed"]) for row in method_rows)
        if seeds != list(SEEDS):
            raise AssertionError(f"Ablation {method}: expected seeds {SEEDS}, found {seeds}")
        for row in method_rows:
            numeric_row(row)

    # The full LAADAN row must be exactly the frozen main LAADAN seed metrics,
    # not a separately retrained duplicate.
    main_laadan = {
        int(row["seed"]): numeric_row(row)
        for row in load_csv(root / "aggregate" / "per_seed_metrics.csv")
        if row["method"] == "LAADAN-AC"
    }
    ablation_full = {
        int(row["seed"]): numeric_row(row)
        for row in rows
        if row["method"] == "Full LAADAN-AC"
    }
    for seed in SEEDS:
        common = set(main_laadan[seed]) & set(ablation_full[seed])
        for metric in common:
            assert_close(
                ablation_full[seed][metric],
                main_laadan[seed][metric],
                f"full-ablation-reuse/seed_{seed}/{metric}",
            )

    perturbation_folders = (
        FULL_PERTURBATION_FOLDERS
        if expected_level == "full"
        else CORE_PERTURBATION_FOLDERS
    )
    ablation_root = root / "ablations" / "lagrangian_frontier"
    for folder in sorted(perturbation_folders):
        for seed in SEEDS:
            seed_dir = ablation_root / folder / f"seed_{seed}"
            for filename in ("history.csv", "metrics.json", "model.pt"):
                if not (seed_dir / filename).is_file():
                    raise AssertionError(f"Missing ablation artifact: {seed_dir / filename}")


def validate_figures(root: Path, minimum_count: int = 5) -> None:
    figure_dir = root / "figures"
    if not figure_dir.is_dir():
        raise AssertionError(f"Missing figure directory: {figure_dir}")
    figures = sorted(figure_dir.glob("*.png"))
    if len(figures) < minimum_count:
        raise AssertionError(
            f"Expected at least {minimum_count} PNG figures in {figure_dir}, found {len(figures)}"
        )
    non_png = [
        path
        for path in figure_dir.iterdir()
        if path.is_file() and path.suffix.lower() != ".png"
    ]
    if non_png:
        raise AssertionError(f"Unexpected non-PNG figures: {non_png}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/fixed_schedule_2026/icu_sepsis")
    parser.add_argument("--data-dir", default="data/icu_sepsis")
    parser.add_argument("--ablations", choices=["none", "core", "full"], default="full")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    data_dir = Path(args.data_dir)

    grouped = validate_main_runs(root)
    validate_zero_posthoc_inadmissibility(grouped)
    validate_aggregate_summary(root, grouped)
    validate_paired_summary(root, grouped)
    validate_ablations(root, args.ablations)
    validate_data_manifest(root, data_dir)
    if not args.skip_figures:
        validate_figures(root)

    print(f"PASS: fixed-schedule result integrity checks completed for {root}.")


if __name__ == "__main__":
    main()
