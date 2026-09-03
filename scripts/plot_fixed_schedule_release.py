
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEEDS = (42, 43, 44, 45, 46)
METHODS = [
    ("Behavior Cloning", "BC"),
    ("CQL-regularised fitted-Q", "CQL"),
    ("Vanilla Offline Actor-Critic", "VOAC"),
    ("Post-hoc Masked VOAC", "Post-hoc\nmask"),
    ("LAADAN-AC", "LAADAN-AC"),
]
ABLATION_LABELS = {
    "Full LAADAN-AC": "Full",
    "Masking-only actor-critic": "Mask only",
    "LAADAN without conservative critic": "No conservative",
    "LAADAN without expert KL": "No expert KL",
    "LAADAN without smoothness proxy": "No smoothness",
    "LAADAN without Lagrangian cost control": "No Lagrangian",
    "LAADAN without action mask": "No mask",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.7, alpha=0.45)
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


def padded_limits(values: Sequence[float], lower_zero: bool = False) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = max(hi - lo, abs(hi) * 0.08, 1e-3)
    lo -= 0.12 * span
    hi += 0.18 * span
    if lower_zero:
        lo = 0.0
    return lo, hi


def summary_value(summary: Dict, method: str, metric: str):
    payload = summary[method][metric]
    return float(payload["mean"]), float(payload["ci95_half"])


def seed_values(rows: List[Dict[str, str]], method: str, metric: str) -> Dict[int, float]:
    return {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["method"] == method and row.get(metric, "") not in {"", None}
    }


def bar_panel(
    ax,
    summary: Dict,
    rows: List[Dict[str, str]],
    methods: Sequence[Tuple[str, str]],
    metric: str,
    ylabel: str,
    percent: bool = False,
):
    x = np.arange(len(methods))
    means = []
    cis = []
    all_values = []
    for method, _ in methods:
        mean, ci = summary_value(summary, method, metric)
        if percent:
            mean *= 100.0
            ci *= 100.0
        means.append(mean)
        cis.append(ci)
        vals = list(seed_values(rows, method, metric).values())
        if percent:
            vals = [100.0 * value for value in vals]
        all_values.extend(vals)

    bars = ax.bar(
        x,
        means,
        width=0.62,
        yerr=cis,
        capsize=3,
        edgecolor="black",
        linewidth=0.65,
        alpha=0.82,
    )
    for idx, (method, _) in enumerate(methods):
        vals = list(seed_values(rows, method, metric).values())
        if percent:
            vals = [100.0 * value for value in vals]
        offsets = np.linspace(-0.12, 0.12, len(vals)) if vals else []
        if vals:
            ax.scatter(
                np.full(len(vals), idx) + offsets,
                vals,
                s=22,
                facecolor="white",
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )

    ax.set_xticks(x, [label for _, label in methods])
    ax.set_ylabel(ylabel)
    clean_axis(ax)
    ymax_values = list(np.asarray(means) + np.asarray(cis)) + all_values
    ymin_values = list(np.asarray(means) - np.asarray(cis)) + all_values
    lower_zero = metric == "inadmissibility_rate"
    ax.set_ylim(*padded_limits(ymin_values + ymax_values, lower_zero=lower_zero))
    return bars


def plot_main(dataset_root: Path, out_dir: Path, dataset_label: str, stem: str):
    summary = load_json(dataset_root / "aggregate" / "main_summary.json")
    rows = load_csv(dataset_root / "aggregate" / "per_seed_metrics.csv")
    specs = [
        ("survival_rate", "Exact survival / return", False),
        ("inadmissibility_rate", "Selected-action inadmissibility (%)", True),
        ("expert_argmax_match", "Expert argmax match", False),
        ("mean_kl_to_expert", "KL to expert", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.25))
    for idx, (metric, ylabel, percent) in enumerate(specs):
        bar_panel(axes[idx], summary, rows, METHODS, metric, ylabel, percent)
        panel_label(axes[idx], chr(ord("A") + idx))
    fig.suptitle(
        f"{dataset_label}: fixed epoch-1000 policy comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "Bars: five-seed mean; error bars: 95% t-CI across random training seeds; white circles: individual seeds.",
        ha="center",
        fontsize=9.2,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_paired(dataset_root: Path, out_dir: Path, stem: str):
    rows = load_csv(dataset_root / "aggregate" / "per_seed_metrics.csv")
    paired = load_json(dataset_root / "aggregate" / "paired_laadan_minus_posthoc_voac.json")
    specs = [
        ("survival_rate", "Exact survival / return"),
        ("expert_argmax_match", "Expert argmax match"),
        ("mean_kl_to_expert", "KL to expert"),
        ("mean_action_deviation_from_expert", "Mean action deviation"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.2))

    for idx, (metric, ylabel) in enumerate(specs):
        post = seed_values(rows, "Post-hoc Masked VOAC", metric)
        laadan = seed_values(rows, "LAADAN-AC", metric)
        vals = []
        for seed in SEEDS:
            if seed not in post or seed not in laadan:
                raise RuntimeError(f"Missing paired seed {seed} for {metric}")
            axes[idx].plot([0, 1], [post[seed], laadan[seed]], linewidth=1.5, alpha=0.85)
            axes[idx].scatter([0, 1], [post[seed], laadan[seed]], s=25, zorder=4)
            vals.extend([post[seed], laadan[seed]])
        axes[idx].set_xticks([0, 1], ["Post-hoc\nmasked VOAC", "LAADAN-AC"])
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_ylim(*padded_limits(vals, lower_zero=False))
        delta = paired["metrics"][metric]
        axes[idx].set_title(
            f"Δ = {float(delta['mean']):+.4f} ± {float(delta['ci95_half']):.4f}"
        )
        clean_axis(axes[idx])
        panel_label(axes[idx], chr(ord("A") + idx))

    fig.suptitle(
        "Paired five-seed comparison: training under the admissibility interface vs post-hoc masking",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def ordered_ablation_methods(summary: Dict) -> List[Tuple[str, str]]:
    preferred = [
        "Full LAADAN-AC",
        "Masking-only actor-critic",
        "LAADAN without action mask",
        "LAADAN without conservative critic",
        "LAADAN without expert KL",
        "LAADAN without smoothness proxy",
        "LAADAN without Lagrangian cost control",
    ]
    missing = [method for method in preferred if method not in summary]
    if missing:
        raise RuntimeError(
            "Full fixed-schedule ablation suite is incomplete; missing: " + ", ".join(missing)
        )
    return [(method, ABLATION_LABELS[method]) for method in preferred]


def plot_full_ablation(dataset_root: Path, out_dir: Path, stem: str):
    summary = load_json(dataset_root / "aggregate" / "ablation_summary.json")
    rows = load_csv(dataset_root / "aggregate" / "ablation_per_seed_metrics.csv")
    methods = ordered_ablation_methods(summary)
    specs = [
        ("survival_rate", "Exact survival / return", False),
        ("inadmissibility_rate", "Selected-action inadmissibility (%)", True),
        ("expert_argmax_match", "Expert argmax match", False),
        ("mean_kl_to_expert", "KL to expert", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
    axes = axes.ravel()
    for idx, (metric, ylabel, percent) in enumerate(specs):
        bar_panel(axes[idx], summary, rows, methods, metric, ylabel, percent)
        axes[idx].tick_params(axis="x", rotation=24)
        for label in axes[idx].get_xticklabels():
            label.set_ha("right")
        panel_label(axes[idx], chr(ord("A") + idx))
    fig.suptitle(
        "ICU-Sepsis fixed-schedule component analysis",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.005,
        "All variants: epoch 1000, seeds 42–46. Zero inadmissibility identifies feasibility; return and expert alignment characterize the operating point inside/outside the feasible set.",
        ha="center",
        fontsize=9.0,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_cross_source(icu_root: Path, eicu_root: Path, out_dir: Path, stem: str):
    datasets = [("ICU-Sepsis", icu_root), ("eICU Demo", eicu_root)]
    specs = [
        ("survival_rate", "Exact survival / return", False),
        ("inadmissibility_rate", "Selected-action inadmissibility (%)", True),
        ("expert_argmax_match", "Expert argmax match", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    x = np.arange(len(METHODS))
    width = 0.36

    for panel_idx, (metric, ylabel, percent) in enumerate(specs):
        plotted = []
        for dataset_idx, (dataset_name, root) in enumerate(datasets):
            summary = load_json(root / "aggregate" / "main_summary.json")
            means, cis = [], []
            for method, _ in METHODS:
                mean, ci = summary_value(summary, method, metric)
                if percent:
                    mean *= 100.0
                    ci *= 100.0
                means.append(mean)
                cis.append(ci)
            positions = x + (dataset_idx - 0.5) * width
            axes[panel_idx].bar(
                positions,
                means,
                width=width,
                yerr=cis,
                capsize=2.5,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.72 if dataset_idx == 0 else 0.95,
                label=dataset_name,
            )
            plotted.extend(np.asarray(means) - np.asarray(cis))
            plotted.extend(np.asarray(means) + np.asarray(cis))
        axes[panel_idx].set_xticks(x, [label for _, label in METHODS])
        axes[panel_idx].set_ylabel(ylabel)
        axes[panel_idx].set_ylim(*padded_limits(plotted, lower_zero=metric == "inadmissibility_rate"))
        clean_axis(axes[panel_idx])
        panel_label(axes[panel_idx], chr(ord("A") + panel_idx))

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Fixed-schedule cross-source portability comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.01,
        "eICU-CRD Demo is an exploratory cross-source portability check, not external clinical validation.",
        ha="center",
        fontsize=9.2,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_policy_diagnostics(dataset_root: Path, out_dir: Path, stem: str):
    rows = load_csv(dataset_root / "aggregate" / "per_seed_metrics.csv")
    metrics = [
        (
            "state_mask_intervention_rate",
            "Post-hoc mask intervention rate (%)",
            "Post-hoc Masked VOAC",
            True,
        ),
        (
            "mean_inadmissible_probability_mass_before_mask",
            "Mean VOAC inadmissible mass before mask (%)",
            "Post-hoc Masked VOAC",
            True,
        ),
        (
            "max_inadmissible_probability_mass_before_mask",
            "Maximum VOAC inadmissible mass before mask (%)",
            "Post-hoc Masked VOAC",
            True,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1))
    for idx, (metric, ylabel, method, percent) in enumerate(metrics):
        vals = seed_values(rows, method, metric)
        ordered = [vals[seed] * (100.0 if percent else 1.0) for seed in SEEDS]
        axes[idx].bar(np.arange(len(SEEDS)), ordered, edgecolor="black", linewidth=0.6, alpha=0.82)
        axes[idx].set_xticks(np.arange(len(SEEDS)), [str(seed) for seed in SEEDS])
        axes[idx].set_xlabel("Training seed")
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_ylim(*padded_limits(ordered, lower_zero=True))
        clean_axis(axes[idx])
        panel_label(axes[idx], chr(ord("A") + idx))
    fig.suptitle(
        "Policy-distribution diagnostics before and after admissibility masking",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/fixed_schedule_2026")
    parser.add_argument(
        "--out-dir", default="results/fixed_schedule_2026/icu_sepsis/figures"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    icu_root = root / "icu_sepsis"
    if not icu_root.is_dir():
        raise FileNotFoundError(f"Missing ICU results: {icu_root}")

    plot_main(icu_root, out_dir, "ICU-Sepsis", "fixed_checkpoint_policy_comparison")
    plot_paired(icu_root, out_dir, "paired_seed_comparison")
    plot_full_ablation(icu_root, out_dir, "full_component_ablation")
    plot_policy_diagnostics(icu_root, out_dir, "policy_distribution_diagnostics")

    eicu_root = root / "eicu_demo"
    if eicu_root.is_dir():
        plot_main(eicu_root, out_dir, "eICU-CRD Demo", "eicu_fixed_checkpoint_policy_comparison")
        plot_cross_source(icu_root, eicu_root, out_dir, "cross_source_portability")

    print(f"PASS: final fixed-schedule figures written to {out_dir}")


if __name__ == "__main__":
    main()
