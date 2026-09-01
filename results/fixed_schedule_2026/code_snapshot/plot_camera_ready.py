#!/usr/bin/env python3
"""Publication-quality figures for the LAADAN-AC camera-ready experiment suite."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "Behavior Cloning": "#A9A9A9",
    "CQL-regularised fitted-Q": "#6E6E6E",
    "Vanilla Offline Actor-Critic": "#F28E8E",
    "Post-hoc Masked VOAC": "#E76F51",
    "LAADAN-AC": "#6FA8DC",
    "Full LAADAN-AC": "#6FA8DC",
    "Masking-only actor-critic": "#9EC5E5",
    "LAADAN without conservative critic": "#8E7CC3",
    "LAADAN without expert KL": "#A64AC9",
    "LAADAN without smoothness proxy": "#C79FEF",
    "LAADAN without Lagrangian cost control": "#674EA7",
    "LAADAN without action mask": "#F6B26B",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary, method, key):
    s = summary[method][key]
    return s["mean"], s["ci95_half"]


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=(0, (4, 4)), alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_main(summary, outdir):
    methods = [
        "Behavior Cloning",
        "CQL-regularised fitted-Q",
        "Vanilla Offline Actor-Critic",
        "Post-hoc Masked VOAC",
        "LAADAN-AC",
    ]
    panels = [
        ("survival_rate", "Survival / return"),
        ("inadmissibility_rate", "Selected-action inadmissibility"),
        ("expert_argmax_match", "Expert argmax match"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    x = np.arange(len(methods))
    labels = ["BC", "CQL", "VOAC", "Masked\nVOAC", "LAADAN"]
    for ax, (key, title) in zip(axes, panels):
        vals, errs = zip(*(metric(summary, m, key) for m in methods))
        ax.bar(
            x, vals, yerr=errs, capsize=4,
            color=[COLORS[m] for m in methods],
            edgecolor="black", linewidth=0.7
        )
        ax.set_xticks(x, labels)
        ax.set_title(title)
        clean_axis(ax)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig_main_confirmatory")


def plot_masking(summary, outdir):
    methods = [
        "Vanilla Offline Actor-Critic",
        "Post-hoc Masked VOAC",
        "LAADAN-AC",
    ]
    panels = [
        ("survival_rate", "Return"),
        ("inadmissibility_rate", "Inadmissibility"),
        ("expert_argmax_match", "Expert match"),
        ("mean_kl_to_expert", "KL to expert"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.7))
    x = np.arange(3)
    labels = ["VOAC", "Post-hoc\nmask", "LAADAN"]
    for ax, (key, title) in zip(axes, panels):
        vals, errs = zip(*(metric(summary, m, key) for m in methods))
        ax.bar(
            x, vals, yerr=errs, capsize=4,
            color=[COLORS[m] for m in methods],
            edgecolor="black", linewidth=0.7
        )
        ax.set_xticks(x, labels)
        ax.set_title(title)
        clean_axis(ax)
    fig.tight_layout(w_pad=2.0)
    save(fig, outdir, "fig_training_vs_posthoc_masking")


def plot_ablation(summary, outdir):
    order = [m for m in [
        "Full LAADAN-AC",
        "Masking-only actor-critic",
        "LAADAN without conservative critic",
        "LAADAN without expert KL",
        "LAADAN without smoothness proxy",
        "LAADAN without Lagrangian cost control",
        "LAADAN without action mask",
    ] if m in summary]
    if len(order) < 2:
        return

    panels = [
        ("survival_rate", "Return"),
        ("inadmissibility_rate", "Inadmissibility"),
        ("expert_argmax_match", "Expert match"),
        ("mean_kl_to_expert", "KL to expert"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8))
    x = np.arange(len(order))
    labels = [
        "Full" if m == "Full LAADAN-AC" else
        "Mask" if m == "Masking-only actor-critic" else
        "−CQL" if "conservative" in m else
        "−KL" if "expert KL" in m else
        "−Smooth" if "smoothness" in m else
        "−λ" if "Lagrangian" in m else
        "−Mask"
        for m in order
    ]
    for ax, (key, title) in zip(axes, panels):
        vals, errs = zip(*(metric(summary, m, key) for m in order))
        ax.bar(
            x, vals, yerr=errs, capsize=3,
            color=[COLORS.get(m, "#8E7CC3") for m in order],
            edgecolor="black", linewidth=0.65
        )
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(title)
        clean_axis(ax)
    fig.tight_layout(w_pad=1.8)
    save(fig, outdir, "fig_ablation_mechanisms")


def plot_paired(per_seed_csv: Path, outdir: Path):
    with per_seed_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    wanted = {"Post-hoc Masked VOAC", "LAADAN-AC"}
    by_method = {m: {} for m in wanted}
    for r in rows:
        m = r["method"]
        if m in wanted:
            by_method[m][int(r["seed"])] = r

    seeds = sorted(
        set(by_method["Post-hoc Masked VOAC"])
        & set(by_method["LAADAN-AC"])
    )
    if not seeds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for ax, metric_name, title in [
        (axes[0], "survival_rate", "Paired return by seed"),
        (axes[1], "expert_argmax_match", "Paired expert match by seed"),
    ]:
        for seed in seeds:
            a = float(by_method["Post-hoc Masked VOAC"][seed][metric_name])
            b = float(by_method["LAADAN-AC"][seed][metric_name])
            ax.plot([0, 1], [a, b], marker="o", linewidth=1.4, alpha=0.8)
        ax.set_xticks([0, 1], ["Post-hoc mask", "LAADAN"])
        ax.set_title(title)
        clean_axis(ax)
    fig.tight_layout(w_pad=2.2)
    save(fig, outdir, "fig_paired_masking_seed_comparison")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/camera_ready_2026/icu_sepsis")
    args = p.parse_args()
    root = Path(args.root)
    out = root / "figures"

    main_summary = load_json(root / "aggregate" / "main_summary.json")
    plot_main(main_summary, out)
    plot_masking(main_summary, out)
    plot_paired(root / "aggregate" / "per_seed_metrics.csv", out)

    ablation_path = root / "aggregate" / "ablation_summary.json"
    if ablation_path.exists():
        plot_ablation(load_json(ablation_path), out)

    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()
