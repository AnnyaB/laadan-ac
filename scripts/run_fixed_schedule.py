from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark import ICUSepsisOfflineBenchmark
from experiment_config import DEFAULT_CONFIG
from trainers import (
    aggregate_seed_metrics,
    evaluate_policy_set,
    greedy_policy_from_logits,
    masked_greedy_policy_from_logits,
    masked_softmax_from_logits,
    mean_ci95,
    plain_softmax_from_logits,
    policy_numpy,
    train_behavior_cloning,
    train_cql,
    train_laadan_ac,
    train_voac,
)
from ablation_engine import (
    BASE_LAADAN_CONFIG,
    BudgetedLagrangianExperiment,
)



SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 1000
FINAL_ONLY_EVAL_EVERY = EPOCHS + 1


# experiments
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recursive_sha256_manifest(data_dir: Path) -> Dict[str, str]:

    manifest: Dict[str, str] = {}
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        manifest[path.relative_to(data_dir).as_posix()] = sha256_file(path)
    if not manifest:
        raise RuntimeError(f"No input files found under {data_dir}")
    return manifest


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def environment_manifest(device: str) -> Dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_requested": device,
    }
    if torch.cuda.is_available():
        manifest["gpu_name"] = torch.cuda.get_device_name(0)
        manifest["gpu_count"] = torch.cuda.device_count()
    return manifest


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def final_epoch_config(config: Dict) -> Dict:
    cfg = deepcopy(config)
    cfg["epochs"] = EPOCHS


    cfg["eval_every"] = FINAL_ONLY_EVAL_EVERY
    return cfg


def evaluated_epochs(run: Dict) -> List[int]:
    epochs: List[int] = []
    for row in run["history"]:
        value = row.get("survival_rate")
        if value is not None and np.isfinite(value):
            epochs.append(int(row["epoch"]))
    return epochs


# validate final only history
def validate_final_only_history(run: Dict) -> None:
    epochs = evaluated_epochs(run)
    if epochs != [EPOCHS]:
        raise RuntimeError(
            f"Protocol violation for {run['name']} seed {run['seed']}: "
            f"benchmark evaluator used at {epochs}; expected [{EPOCHS}]."
        )


def attach_provenance(
    run: Dict,
    result_root: Path,
    method_folder: str,
    dataset_name: str,
    config: Dict,
    env: Dict,
) -> None:
    seed_dir = result_root / method_folder / f"seed_{run['seed']}"
    dump_json(
        seed_dir / "provenance.json",
        {
            "dataset": dataset_name,
            "method": run["name"],
            "seed": int(run["seed"]),
            "training_epochs": EPOCHS,
            "checkpoint_rule": "pre-specified final epoch",
            "checkpoint_epoch": EPOCHS,
            "benchmark_evaluation_used_for_model_selection": False,
            "headline_seed_selection": False,
            "evaluated_epochs": evaluated_epochs(run),
            "config": config,
            "environment": env,
        },
    )


# aggregate runs
def aggregate_runs(run_groups: Dict[str, List[Dict]]) -> Dict:

    result: Dict[str, Dict] = {}
    for name, runs in run_groups.items():
        cleaned = []
        for run in runs:
            copy_run = dict(run)
            copy_run["metrics"] = {
                key: value
                for key, value in run["metrics"].items()
                if key != "convergence_epoch_95"
            }
            cleaned.append(copy_run)
        summary = aggregate_seed_metrics(cleaned)
        summary.pop("convergence_epoch_95", None)
        result[name] = summary
    return result


def raw_seed_rows(run_groups: Dict[str, List[Dict]]) -> List[Dict]:
    rows: List[Dict] = []
    for name, runs in run_groups.items():
        for run in runs:
            row = {"method": name, "seed": int(run["seed"])}
            for key, value in run["metrics"].items():
                if key == "convergence_epoch_95":
                    continue
                if isinstance(value, (int, float, np.integer, np.floating, bool)):
                    row[key] = float(value)
            rows.append(row)
    return rows




# post-hoc mask
def posthoc_masked_voac(benchmark, voac_run: Dict) -> Dict:

    model = voac_run["model"]
    model.eval()
    x = benchmark.state_features_t
    admissible = benchmark.admissible_mask_t

    with torch.no_grad():
        logits = model(x)["logits"]
        raw_soft = plain_softmax_from_logits(logits)
        raw_greedy = greedy_policy_from_logits(logits)
        masked_soft = masked_softmax_from_logits(logits, admissible)
        masked_greedy = masked_greedy_policy_from_logits(logits, admissible)

    metrics = evaluate_policy_set(
        benchmark,
        policy_numpy(masked_greedy),
        soft_policy=policy_numpy(masked_soft),
    )

    terminal = np.asarray(benchmark.terminal_mask).astype(bool)
    nonterminal = ~terminal
    raw_actions = np.argmax(policy_numpy(raw_greedy), axis=1)
    masked_actions = np.argmax(policy_numpy(masked_greedy), axis=1)
    metrics["state_mask_intervention_rate"] = float(
        np.mean(raw_actions[nonterminal] != masked_actions[nonterminal])
    )

    raw_soft_np = policy_numpy(raw_soft)
    inad_np = 1.0 - np.asarray(benchmark.admissible_mask, dtype=float)
    removed_mass = np.sum(raw_soft_np * inad_np, axis=1)
    metrics["mean_inadmissible_probability_mass_before_mask"] = float(
        np.mean(removed_mass[nonterminal])
    )
    metrics["max_inadmissible_probability_mass_before_mask"] = float(
        np.max(removed_mass[nonterminal])
    )

    return {
        "name": "Post-hoc Masked VOAC",
        "seed": int(voac_run["seed"]),
        "model": model,
        "policy": policy_numpy(masked_greedy),
        "analysis_policy": policy_numpy(masked_soft),
        "history": [],
        "metrics": metrics,
        "train_time_seconds": 0.0,
        "convergence_epoch_95": None,
    }


def paired_summary(
    a_runs: List[Dict], b_runs: List[Dict], a_name: str, b_name: str
) -> Dict:
    a = {int(run["seed"]): run for run in a_runs}
    b = {int(run["seed"]): run for run in b_runs}
    if set(a) != set(b):
        raise RuntimeError(f"Paired seeds differ: {sorted(a)} vs {sorted(b)}")

    numeric_sets = []
    for run in a_runs + b_runs:
        numeric_sets.append(
            {
                key
                for key, value in run["metrics"].items()
                if key != "convergence_epoch_95"
                and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
    metric_names = sorted(set.intersection(*numeric_sets))

    out = {
        "comparison": f"{a_name} minus {b_name}",
        "seeds": sorted(a),
        "metrics": {},
    }
    for metric in metric_names:
        diffs = [
            float(a[seed]["metrics"][metric])
            - float(b[seed]["metrics"][metric])
            for seed in sorted(a)
        ]
        out["metrics"][metric] = {
            "per_seed_difference": diffs,
            **mean_ci95(diffs),
        }
    return out


def ablation_variants(level: str) -> List[Dict]:
    variants = [
        {
            "name": "Full LAADAN-AC",
            "folder": "full_laadan_ac",
            "use_action_mask": True,
            "use_conservative": True,
            "use_expert_kl": True,
            "use_smoothness": True,
            "use_lagrangian": True,
            "train_cost_head": True,
        },
        {
            "name": "Masking-only actor-critic",
            "folder": "mask_only",
            "use_action_mask": True,
            "use_conservative": False,
            "use_expert_kl": False,
            "use_smoothness": False,
            "use_lagrangian": False,
            "train_cost_head": False,
        },
        {
            "name": "LAADAN without conservative critic",
            "folder": "no_conservative",
            "use_action_mask": True,
            "use_conservative": False,
            "use_expert_kl": True,
            "use_smoothness": True,
            "use_lagrangian": True,
            "train_cost_head": True,
        },
    ]
    if level == "full":
        variants.extend(
            [
                {
                    "name": "LAADAN without expert KL",
                    "folder": "no_expert_kl",
                    "use_action_mask": True,
                    "use_conservative": True,
                    "use_expert_kl": False,
                    "use_smoothness": True,
                    "use_lagrangian": True,
                    "train_cost_head": True,
                },
                {
                    "name": "LAADAN without smoothness proxy",
                    "folder": "no_smoothness",
                    "use_action_mask": True,
                    "use_conservative": True,
                    "use_expert_kl": True,
                    "use_smoothness": False,
                    "use_lagrangian": True,
                    "train_cost_head": True,
                },
                {
                    "name": "LAADAN without Lagrangian cost control",
                    "folder": "no_lagrangian",
                    "use_action_mask": True,
                    "use_conservative": True,
                    "use_expert_kl": True,
                    "use_smoothness": True,
                    "use_lagrangian": False,
                    "train_cost_head": True,
                },
                {
                    "name": "LAADAN without action mask",
                    "folder": "no_mask",
                    "use_action_mask": False,
                    "use_conservative": True,
                    "use_expert_kl": True,
                    "use_smoothness": True,
                    "use_lagrangian": True,
                    "train_cost_head": True,
                },
            ]
        )
    return variants




# run dataset
def run_dataset(
    data_dir: Path,
    output_dir: Path,
    dataset_name: str,
    device: str,
    run_ablations: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = environment_manifest(device)
    dump_json(output_dir / "environment.json", env)

    benchmark = ICUSepsisOfflineBenchmark(
        str(data_dir), horizon=50, device=device, use_one_hot_states=False
    )
    benchmark.save_benchmark_description(str(output_dir / "benchmark_description.json"))
    dump_json(output_dir / "data_sha256.json", recursive_sha256_manifest(data_dir))

    base = deepcopy(DEFAULT_CONFIG)
    main_cfg = {
        "bc": final_epoch_config(base["bc"]),
        "cql": final_epoch_config(base["cql"]),
        "voac": final_epoch_config(base["voac"]),
        "laadan_ac": final_epoch_config(base["laadan_ac"]),
    }
    dump_json(
        output_dir / "fixed_schedule_config.json",
        {
            "seeds": SEEDS,
            "epochs": EPOCHS,
            "checkpoint_rule": "final_epoch",
            "benchmark_evaluation_used_for_model_selection": False,
            "configs": main_cfg,
        },
    )

    groups: Dict[str, List[Dict]] = {
        "Behavior Cloning": [],
        "CQL-regularised fitted-Q": [],
        "Vanilla Offline Actor-Critic": [],
        "LAADAN-AC": [],
        "Post-hoc Masked VOAC": [],
    }

    main_root = output_dir / "main_fixed_schedule"
    for seed in SEEDS:
        print(f"\n=== {dataset_name}: seed {seed} ===")
        bc = train_behavior_cloning(benchmark, seed, str(main_root), main_cfg["bc"])
        cql = train_cql(benchmark, seed, str(main_root), main_cfg["cql"])
        voac = train_voac(benchmark, seed, str(main_root), main_cfg["voac"])
        laadan = train_laadan_ac(
            benchmark, seed, str(main_root), main_cfg["laadan_ac"]
        )

        for run in (bc, cql, voac, laadan):
            validate_final_only_history(run)

        posthoc = posthoc_masked_voac(benchmark, voac)

        groups["Behavior Cloning"].append(bc)
        groups["CQL-regularised fitted-Q"].append(cql)
        groups["Vanilla Offline Actor-Critic"].append(voac)
        groups["LAADAN-AC"].append(laadan)
        groups["Post-hoc Masked VOAC"].append(posthoc)

        attach_provenance(bc, main_root, "bc", dataset_name, main_cfg["bc"], env)
        attach_provenance(cql, main_root, "cql", dataset_name, main_cfg["cql"], env)
        attach_provenance(voac, main_root, "voac", dataset_name, main_cfg["voac"], env)
        attach_provenance(
            laadan, main_root, "laadan_ac", dataset_name, main_cfg["laadan_ac"], env
        )

        posthoc_dir = output_dir / "posthoc_masked_voac" / f"seed_{seed}"
        dump_json(posthoc_dir / "metrics.json", posthoc["metrics"])
        dump_json(
            posthoc_dir / "provenance.json",
            {
                "dataset": dataset_name,
                "seed": seed,
                "source_voac_checkpoint": str(
                    main_root / "voac" / f"seed_{seed}" / "model.pt"
                ),
                "source_voac_training_rule": "pre-specified final epoch",
                "source_voac_checkpoint_epoch": EPOCHS,
                "mask_applied_during_training": False,
                "mask_applied_at_evaluation": True,
                "parameters_updated_during_posthoc_evaluation": False,
                "environment": env,
            },
        )

    dump_json(output_dir / "aggregate" / "main_summary.json", aggregate_runs(groups))
    write_csv(output_dir / "aggregate" / "per_seed_metrics.csv", raw_seed_rows(groups))
    dump_json(
        output_dir / "aggregate" / "paired_laadan_minus_posthoc_voac.json",
        paired_summary(
            groups["LAADAN-AC"],
            groups["Post-hoc Masked VOAC"],
            "LAADAN-AC",
            "Post-hoc Masked VOAC",
        ),
    )

    if run_ablations != "none":
        ablation_cfg = deepcopy(BASE_LAADAN_CONFIG)
        ablation_cfg["epochs"] = EPOCHS
        ablation_cfg["eval_every"] = FINAL_ONLY_EVAL_EVERY
        experiment = BudgetedLagrangianExperiment(
            benchmark, str(output_dir / "ablations"), ablation_cfg
        )
        variants = ablation_variants(run_ablations)
        ablation_groups = experiment.run_variants(variants, SEEDS)
        for runs in ablation_groups.values():
            for run in runs:
                validate_final_only_history(run)
        dump_json(
            output_dir / "aggregate" / "ablation_summary.json",
            aggregate_runs(ablation_groups),
        )
        write_csv(
            output_dir / "aggregate" / "ablation_per_seed_metrics.csv",
            raw_seed_rows(ablation_groups),
        )

    print(f"\nCompleted {dataset_name}. Results: {output_dir}")


# arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--icu-data", default="data/icu_sepsis")
    parser.add_argument("--eicu-data", default="data/eicu_demo_mdp")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--datasets", choices=["icu", "eicu", "both"], default="icu")
    parser.add_argument(
        "--icu-ablations",
        choices=["none", "core", "full"],
        default="full",
        help="Ablation suite for ICU-Sepsis. Full leave-one-out is the final-paper protocol.",
    )
    parser.add_argument(
        "--eicu-ablations",
        choices=["none", "core", "full"],
        default="none",
        help="Ablation suite for eICU portability check; none is recommended.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested


def main():
    args = parse_args()
    device = resolve_device(args.device)
    out = Path(args.output)

    if args.datasets in {"icu", "both"}:
        run_dataset(
            Path(args.icu_data),
            out / "icu_sepsis",
            "ICU-Sepsis",
            device,
            args.icu_ablations,
        )

    if args.datasets in {"eicu", "both"}:
        run_dataset(
            Path(args.eicu_data),
            out / "eicu_demo",
            "eICU-CRD Demo",
            device,
            args.eicu_ablations,
        )


if __name__ == "__main__":
    main()
