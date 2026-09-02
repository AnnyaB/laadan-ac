#!/usr/bin/env python3
"""Freeze a machine-verifiable LAADAN-AC fixed-schedule result release.

This script does not train models or alter metrics. It validates the already
produced experiment tree, extracts paper-facing summaries from the frozen JSON
and CSV artifacts, and writes a cryptographic manifest of every released file.
Run it only after training, diagnostics, plotting, and validation have completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

SEEDS = (42, 43, 44, 45, 46)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def summary_metric(summary: Dict, method: str, metric: str) -> Dict:
    payload = summary[method][metric]
    return {
        "mean": float(payload["mean"]),
        "ci95_half": float(payload["ci95_half"]),
        "n": int(payload["n"]),
    }


def main_table(dataset_root: Path) -> Dict:
    summary = load_json(dataset_root / "aggregate" / "main_summary.json")
    methods = [
        "Behavior Cloning",
        "CQL-regularised fitted-Q",
        "Vanilla Offline Actor-Critic",
        "Post-hoc Masked VOAC",
        "LAADAN-AC",
    ]
    metrics = [
        "survival_rate",
        "inadmissibility_rate",
        "expert_argmax_match",
        "mean_kl_to_expert",
    ]
    return {
        method: {metric: summary_metric(summary, method, metric) for metric in metrics}
        for method in methods
    }


def ablation_table(dataset_root: Path) -> Dict:
    path = dataset_root / "aggregate" / "ablation_summary.json"
    if not path.is_file():
        return {}
    summary = load_json(path)
    metrics = [
        "survival_rate",
        "inadmissibility_rate",
        "expert_argmax_match",
        "mean_kl_to_expert",
    ]
    return {
        method: {
            metric: summary_metric(summary, method, metric)
            for metric in metrics
            if metric in summary[method]
        }
        for method in summary
    }


def paired_table(dataset_root: Path) -> Dict:
    paired = load_json(
        dataset_root / "aggregate" / "paired_laadan_minus_posthoc_voac.json"
    )
    keep = {}
    for metric in (
        "survival_rate",
        "inadmissibility_rate",
        "expert_argmax_match",
        "mean_kl_to_expert",
    ):
        payload = paired["metrics"][metric]
        keep[metric] = {
            "mean": float(payload["mean"]),
            "ci95_half": float(payload["ci95_half"]),
            "n": int(payload["n"]),
            "per_seed_difference": [float(x) for x in payload["per_seed_difference"]],
        }
    return {
        "comparison": paired["comparison"],
        "seeds": paired["seeds"],
        "metrics": keep,
    }


def release_file_manifest(root: Path, excluded: Iterable[Path]) -> Dict[str, Dict]:
    excluded_resolved = {path.resolve() for path in excluded}
    manifest = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.resolve() in excluded_resolved:
            continue
        rel = path.relative_to(root).as_posix()
        manifest[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/fixed_schedule_2026")
    parser.add_argument(
        "--output", default="results/fixed_schedule_2026/release_manifest.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    repo_root = Path(__file__).resolve().parents[1]

    icu_root = root / "icu_sepsis"
    eicu_root = root / "eicu_demo"
    if not icu_root.is_dir():
        raise FileNotFoundError(f"Missing ICU fixed-schedule results: {icu_root}")

    release = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha(repo_root),
        "protocol": {
            "training_epochs": 1000,
            "checkpoint_rule": "pre-specified final epoch",
            "benchmark_evaluation_used_for_model_selection": False,
            "random_seeds": list(SEEDS),
            "confidence_interval_interpretation": "95% t-CI across five random training seeds; not patient/population uncertainty",
        },
        "icu_sepsis": {
            "main": main_table(icu_root),
            "paired_laadan_minus_posthoc_voac": paired_table(icu_root),
            "ablations": ablation_table(icu_root),
        },
    }

    if eicu_root.is_dir():
        release["eicu_demo"] = {
            "scope": "exploratory cross-source portability check",
            "main": main_table(eicu_root),
            "paired_laadan_minus_posthoc_voac": paired_table(eicu_root),
            "ablations": ablation_table(eicu_root),
        }

    release["files"] = release_file_manifest(root, excluded=[output])
    dump_json(output, release)

    print(f"PASS: release frozen at {output}")
    print(f"Files hashed: {len(release['files'])}")


if __name__ == "__main__":
    main()
