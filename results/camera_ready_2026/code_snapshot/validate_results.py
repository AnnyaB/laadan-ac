#!/usr/bin/env python3
"""Integrity checks for camera-ready LAADAN-AC results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = {42, 43, 44, 45, 46}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/camera_ready_2026/icu_sepsis")
    args = p.parse_args()
    root = Path(args.root)

    rows = list(csv.DictReader(
        (root / "aggregate" / "per_seed_metrics.csv").open("r", encoding="utf-8")
    ))

    expected = {
        "Behavior Cloning",
        "CQL-regularised fitted-Q",
        "Vanilla Offline Actor-Critic",
        "Post-hoc Masked VOAC",
        "LAADAN-AC",
    }
    methods = {r["method"] for r in rows}
    assert expected <= methods, f"Missing methods: {expected - methods}"

    for method in expected:
        method_rows = [r for r in rows if r["method"] == method]
        seeds = {int(r["seed"]) for r in method_rows}
        assert seeds == SEEDS, (method, seeds)
        for r in method_rows:
            for k, v in r.items():
                if k in {"method", "seed", ""} or v in {None, ""}:
                    continue
                try:
                    value = float(v)
                except ValueError:
                    continue
                assert np.isfinite(value), (method, r["seed"], k, v)

    posthoc = [r for r in rows if r["method"] == "Post-hoc Masked VOAC"]
    assert all(abs(float(r["inadmissibility_rate"])) < 1e-12 for r in posthoc)

    for method_dir in ["bc", "cql", "voac", "laadan_ac"]:
        for seed in sorted(SEEDS):
            prov = load_json(
                root / "main_fixed_schedule" / method_dir
                / f"seed_{seed}" / "provenance.json"
            )
            assert prov["checkpoint_rule"] == "pre-specified final epoch"
            assert prov["checkpoint_epoch"] == 1000
            assert prov["benchmark_evaluation_used_for_model_selection"] is False

    print("PASS: camera-ready result integrity checks completed successfully.")


if __name__ == "__main__":
    main()
