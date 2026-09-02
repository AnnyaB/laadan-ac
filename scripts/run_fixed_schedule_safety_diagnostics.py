#!/usr/bin/env python3
"""Safety-failure diagnostics using only fixed-schedule checkpoints.

All quantitative diagnostic summaries are computed across the pre-specified five
seeds (42--46). Qualitative state/Q-value/trajectory panels use one predeclared
illustrative seed (42 by default) rather than selecting a visually favourable
run. No model is retrained and no benchmark metric is used for checkpoint or
seed selection in this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark import ICUSepsisOfflineBenchmark  # noqa: E402
from models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet  # noqa: E402
from safety_failure_analysis import (  # noqa: E402
    build_policies,
    get_benchmark_array,
    get_config_block,
    get_terminal_mask,
    load_checkpoint,
    pca_two_components,
    per_state_inadmissibility,
    representative_unsafe_states,
    write_action_tables,
    write_feature_distribution_table,
    write_q_value_table,
    write_state_table,
    write_trajectory_tables,
)
from trainers import mean_ci95  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
PLOT_DPI = 600

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.8,
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


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


def instantiate_models(benchmark, config: Dict, root: Path, seed: int, device: str):
    """Load the four epoch-1000 fixed-schedule checkpoints for one seed."""
    models = {}

    bc_cfg = get_config_block(config, "bc")
    bc = BehaviorCloningNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=bc_cfg["hidden_dim"],
        latent_dim=bc_cfg["latent_dim"],
        dropout=bc_cfg["dropout"],
    ).to(device)
    models["Behavior Cloning"] = load_checkpoint(
        bc, str(root / "bc" / f"seed_{seed}" / "model.pt"), device
    )

    cql_cfg = get_config_block(config, "cql")
    cql = ConservativeQNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=cql_cfg["hidden_dim"],
        latent_dim=cql_cfg["latent_dim"],
        dropout=cql_cfg["dropout"],
    ).to(device)
    models["Conservative Q-Learning"] = load_checkpoint(
        cql, str(root / "cql" / f"seed_{seed}" / "model.pt"), device
    )

    voac_cfg = get_config_block(config, "voac")
    voac = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=voac_cfg["hidden_dim"],
        latent_dim=voac_cfg["latent_dim"],
        dropout=voac_cfg["dropout"],
        use_cost_head=False,
    ).to(device)
    models["Vanilla Offline Actor-Critic"] = load_checkpoint(
        voac, str(root / "voac" / f"seed_{seed}" / "model.pt"), device
    )

    laadan_cfg = get_config_block(config, "laadan_ac")
    laadan = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=laadan_cfg["hidden_dim"],
        latent_dim=laadan_cfg["latent_dim"],
        dropout=laadan_cfg["dropout"],
        use_cost_head=True,
    ).to(device)
    models["LAADAN-AC"] = load_checkpoint(
        laadan, str(root / "laadan_ac" / f"seed_{seed}" / "model.pt"), device
    )

    return models


def seed_diagnostics(benchmark, policies: Dict, seed: int) -> Dict:
    mask = get_benchmark_array(benchmark, "admissible_mask")
    terminal = get_terminal_mask(benchmark)
    nonterminal = ~terminal

    voac = policies["Vanilla Offline Actor-Critic"]
    posthoc = policies["Post-hoc Masked VOAC"]
    laadan = policies["LAADAN-AC"]

    voac_primary_cost = per_state_inadmissibility(voac["primary"], mask)
    posthoc_primary_cost = per_state_inadmissibility(posthoc["primary"], mask)
    laadan_primary_cost = per_state_inadmissibility(laadan["primary"], mask)
    voac_soft_cost = per_state_inadmissibility(voac["soft"], mask)
    laadan_soft_cost = per_state_inadmissibility(laadan["soft"], mask)

    voac_raw_action = np.argmax(voac["raw_primary"], axis=1)
    voac_masked_action = np.argmax(voac["masked_primary"], axis=1)

    return {
        "seed": seed,
        "voac_fraction_states_with_inadmissible_greedy_action": float(
            np.mean(voac_primary_cost[nonterminal] > 0.5)
        ),
        "voac_mean_state_inadmissibility": float(np.mean(voac_primary_cost[nonterminal])),
        "posthoc_mean_state_inadmissibility": float(np.mean(posthoc_primary_cost[nonterminal])),
        "laadan_mean_state_inadmissibility": float(np.mean(laadan_primary_cost[nonterminal])),
        "posthoc_mask_intervention_rate": float(
            np.mean(voac_raw_action[nonterminal] != voac_masked_action[nonterminal])
        ),
        "voac_mean_soft_inadmissible_probability_mass": float(
            np.mean(voac_soft_cost[nonterminal])
        ),
        "laadan_mean_soft_inadmissible_probability_mass": float(
            np.mean(laadan_soft_cost[nonterminal])
        ),
    }


def aggregate_seed_rows(rows: List[Dict]) -> Dict:
    metric_names = [key for key in rows[0] if key != "seed"]
    return {
        metric: mean_ci95([float(row[metric]) for row in rows])
        for metric in metric_names
    }


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, alpha=0.42)
    ax.set_axisbelow(True)


def panel_label(ax, label: str):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def plot_fixed_diagnostic_figure(
    output_path: str,
    benchmark,
    policies: Dict,
    representative_states: List[int],
    trajectory_payload,
    illustrative_seed: int,
) -> None:
    """Create a four-panel fixed-checkpoint diagnostic with no seed cherry-picking."""
    mask = get_benchmark_array(benchmark, "admissible_mask")
    features = get_benchmark_array(benchmark, "state_features")
    terminal = get_terminal_mask(benchmark)
    nonterminal = ~terminal

    voac = policies["Vanilla Offline Actor-Critic"]
    posthoc = policies["Post-hoc Masked VOAC"]
    laadan = policies["LAADAN-AC"]

    voac_cost = per_state_inadmissibility(voac["primary"], mask)
    posthoc_cost = per_state_inadmissibility(posthoc["primary"], mask)
    laadan_cost = per_state_inadmissibility(laadan["primary"], mask)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0))

    # A: distribution over all non-terminal states.
    order = np.argsort(voac_cost[nonterminal])[::-1]
    voac_sorted = voac_cost[nonterminal][order]
    post_sorted = posthoc_cost[nonterminal][order]
    laadan_sorted = laadan_cost[nonterminal][order]
    x = np.arange(len(voac_sorted))
    axes[0, 0].plot(x, voac_sorted, linewidth=2.0, label="VOAC")
    axes[0, 0].plot(x, post_sorted, linewidth=1.8, label="Post-hoc masked VOAC")
    axes[0, 0].plot(x, laadan_sorted, linewidth=1.8, label="LAADAN-AC")
    axes[0, 0].set_xlabel("Non-terminal states sorted by VOAC inadmissibility")
    axes[0, 0].set_ylabel("Selected-action inadmissibility")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(frameon=False, loc="upper right")
    clean_axis(axes[0, 0])
    panel_label(axes[0, 0], "A")

    # B: state-space localization of VOAC violations.
    coords = pca_two_components(features)
    scatter = axes[0, 1].scatter(
        coords[nonterminal, 0],
        coords[nonterminal, 1],
        c=voac_cost[nonterminal],
        s=22,
        alpha=0.9,
        edgecolors="none",
    )
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].set_title("VOAC violations across benchmark states")
    fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04, label="VOAC inadmissibility")
    clean_axis(axes[0, 1])
    panel_label(axes[0, 1], "B")

    # C: Q-value separation at a deterministic representative VOAC-failure state.
    state_id = representative_states[0]
    groups = []
    labels = []
    admissible = mask[state_id] > 0.5
    for model_name, short in [
        ("Vanilla Offline Actor-Critic", "VOAC"),
        ("Conservative Q-Learning", "CQL"),
        ("LAADAN-AC", "LAADAN")
    ]:
        payload = policies.get(model_name)
        if payload is None or payload.get("q_values") is None:
            continue
        q = np.asarray(payload["q_values"])[state_id]
        if np.any(admissible):
            groups.append(q[admissible])
            labels.append(f"{short}\nadm")
        if np.any(~admissible):
            groups.append(q[~admissible])
            labels.append(f"{short}\ninad")
    if groups:
        axes[1, 0].boxplot(groups, tick_labels=labels, showfliers=False, widths=0.58)
        axes[1, 0].set_ylabel("Q value")
        axes[1, 0].set_title(f"Representative VOAC-failure state {state_id}")
        clean_axis(axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "Q-values unavailable", ha="center", va="center")
        axes[1, 0].set_axis_off()
    panel_label(axes[1, 0], "C")

    # D: a reproducibly sampled trajectory initialized identically for both policies.
    if trajectory_payload is not None:
        _, voac_rows, laadan_rows = trajectory_payload
        for name, rows in [("VOAC", voac_rows), ("LAADAN-AC", laadan_rows)]:
            steps = [int(row["step"]) for row in rows]
            actions = [int(row["action"]) for row in rows]
            axes[1, 1].plot(steps, actions, marker="o", markersize=4, linewidth=1.7, label=name)
            if name == "VOAC":
                bad_steps = [int(row["step"]) for row in rows if int(row["admissible"]) == 0]
                bad_actions = [int(row["action"]) for row in rows if int(row["admissible"]) == 0]
                if bad_steps:
                    axes[1, 1].scatter(
                        bad_steps,
                        bad_actions,
                        marker="x",
                        s=56,
                        linewidth=1.5,
                        label="VOAC inadmissible",
                        zorder=5,
                    )
        axes[1, 1].set_xlabel("Trajectory step")
        axes[1, 1].set_ylabel("Treatment-action bin")
        axes[1, 1].set_title("Matched-initial-state sampled trajectory")
        axes[1, 1].legend(frameon=False, loc="best")
        clean_axis(axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "Trajectory unavailable", ha="center", va="center")
        axes[1, 1].set_axis_off()
    panel_label(axes[1, 1], "D")

    fig.suptitle(
        f"Fixed-checkpoint safety-failure diagnostics — predeclared illustrative seed {illustrative_seed}",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.005,
        "Qualitative panels use a predeclared seed only; five-seed quantitative diagnostics are saved separately and are not inferred from this illustration.",
        ha="center",
        fontsize=8.9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_illustrative_outputs(
    benchmark,
    policies: Dict,
    output_dir: Path,
    seed: int,
    num_trajectories: int,
) -> Dict[str, str]:
    tables_dir = output_dir / "illustrative_seed" / f"seed_{seed}" / "tables"
    figures_dir = output_dir / "illustrative_seed" / f"seed_{seed}" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    representative_states = representative_unsafe_states(
        benchmark,
        policies["Vanilla Offline Actor-Critic"]["primary"],
        k=20,
    )

    outputs = {
        "state_level": str(tables_dir / "state_level_inadmissibility.csv"),
        "feature_distribution": str(tables_dir / "safe_vs_unsafe_feature_distribution.csv"),
        "action_counts": str(tables_dir / "action_level_inadmissibility_counts.csv"),
        "action_confusion": str(tables_dir / "expert_voac_laadan_action_confusion.csv"),
        "q_values": str(tables_dir / "representative_state_q_values.csv"),
        "trajectory": str(tables_dir / "sample_trajectory_voac_vs_laadan.csv"),
        "timepoint": str(tables_dir / "trajectory_timepoint_inadmissibility.csv"),
        "figure": str(figures_dir / "safety_failure_fixed_checkpoint.png"),
    }

    write_state_table(outputs["state_level"], benchmark, policies)
    write_feature_distribution_table(
        outputs["feature_distribution"],
        benchmark,
        policies["Vanilla Offline Actor-Critic"]["primary"],
    )
    write_action_tables(
        outputs["action_counts"], outputs["action_confusion"], benchmark, policies
    )
    write_q_value_table(outputs["q_values"], benchmark, policies, representative_states[:8])
    trajectory_payload = write_trajectory_tables(
        outputs["trajectory"],
        outputs["timepoint"],
        benchmark,
        policies,
        seed=seed,
        attempts=num_trajectories,
    )
    plot_fixed_diagnostic_figure(
        outputs["figure"],
        benchmark,
        policies,
        representative_states,
        trajectory_payload,
        illustrative_seed=seed,
    )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/icu_sepsis")
    parser.add_argument(
        "--results-root", default="results/fixed_schedule_2026/icu_sepsis"
    )
    parser.add_argument(
        "--output-dir",
        default="results/fixed_schedule_2026/icu_sepsis/safety_diagnostics",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--illustrative-seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--num-trajectories", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = load_json(results_root / "fixed_schedule_config.json")
    config = config_payload["configs"]
    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir, horizon=50, device=device, use_one_hot_states=False
    )
    checkpoint_root = results_root / "main_fixed_schedule"

    rows: List[Dict] = []
    illustrative_policies = None
    for seed in SEEDS:
        models = instantiate_models(benchmark, config, checkpoint_root, seed, device)
        policies = build_policies(benchmark, models)
        rows.append(seed_diagnostics(benchmark, policies, seed))
        if seed == args.illustrative_seed:
            illustrative_policies = policies

    write_csv(output_dir / "per_seed_diagnostics.csv", rows)
    dump_json(output_dir / "aggregate_diagnostics.json", aggregate_seed_rows(rows))

    if illustrative_policies is None:
        raise RuntimeError("Illustrative seed policies were not loaded")
    illustrative_outputs = build_illustrative_outputs(
        benchmark,
        illustrative_policies,
        output_dir,
        args.illustrative_seed,
        args.num_trajectories,
    )
    dump_json(
        output_dir / "diagnostic_protocol.json",
        {
            "seeds": list(SEEDS),
            "quantitative_summary": "all five fixed-schedule epoch-1000 seeds",
            "illustrative_seed": args.illustrative_seed,
            "illustrative_seed_selection_rule": "predeclared; not selected by performance or appearance",
            "num_trajectory_samples": args.num_trajectories,
            "checkpoint_rule": "pre-specified final epoch",
            "benchmark_evaluation_used_for_checkpoint_selection": False,
            "illustrative_outputs": illustrative_outputs,
        },
    )

    print(f"PASS: fixed-schedule safety diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
