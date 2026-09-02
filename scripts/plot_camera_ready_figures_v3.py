from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Style
# ---------------------------

def set_pub_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 18,
        "axes.titleweight": "semibold",
        "axes.labelsize": 15,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12,
        "axes.linewidth": 1.1,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.20,
        "grid.color": "#7f7f7f",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })


# restrained, publication-like palette
METHOD_COLORS = {
    "Behavior Cloning": "#B7C9DD",
    "CQL-regularised fitted-Q": "#6E9ED6",
    "Vanilla Offline Actor-Critic": "#C95B47",
    "Post-hoc Masked VOAC": "#D8923B",
    "LAADAN-AC": "#1E2F97",

    "Full LAADAN-AC": "#1E2F97",
    "Masking-only actor-critic": "#D8923B",
    "LAADAN without action mask": "#C95B47",
    "LAADAN without conservative critic": "#C22F7C",
    "LAADAN without expert KL": "#8D5CCB",
    "LAADAN without smoothness proxy": "#E6605A",
    "LAADAN without Lagrangian cost control": "#3D8B1E",
}


# ---------------------------
# I/O helpers
# ---------------------------

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wrap_label(s: str, width: int = 12) -> str:
    return textwrap.fill(s, width=width, break_long_words=False)


def method_seed_lists(rows, order, metric):
    out = []
    for method in order:
        vals = []
        for row in rows:
            if row["method"] == method and row.get(metric, "") != "":
                vals.append(float(row[metric]))
        out.append(vals)
    return out


def summary_arrays(summary, order, metric):
    means, cis = [], []
    for method in order:
        means.append(summary[method][metric]["mean"])
        cis.append(summary[method][metric].get("ci95_half", 0.0))
    return np.array(means, dtype=float), np.array(cis, dtype=float)


# ---------------------------
# Plotting helpers
# ---------------------------

def beautify_ax(ax):
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def auto_ylim(vals, start_at_zero=True):
    """
    Robustly flatten arrays/lists of unequal lengths.

    Inputs can contain:
      - mean vectors
      - CI vectors
      - per-method seed lists

    These are intentionally different lengths, so they must be
    flattened independently before concatenation.
    """
    pieces = []

    for value in vals:
        arr = np.asarray(value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]

        if arr.size:
            pieces.append(arr)

    if not pieces:
        return (0.0, 1.0)

    flat = np.concatenate(pieces)

    lo = float(np.min(flat))
    hi = float(np.max(flat))

    if hi == lo:
        pad = max(0.05 * abs(hi), 0.05)
    else:
        pad = 0.12 * (hi - lo)

    ymin = 0.0 if start_at_zero else lo - pad
    ymax = hi + pad

    # Avoid a degenerate zero-height axis.
    if ymax <= ymin:
        ymax = ymin + max(0.05, 0.05 * abs(ymin))

    return ymin, ymax


def panel_bar(ax, labels, means, cis, seed_lists, colors, ylabel, panel_title, percent=False):
    x = np.arange(len(labels))
    plot_means = means * (100.0 if percent else 1.0)
    plot_cis = cis * (100.0 if percent else 1.0)
    plot_seeds = [[v * (100.0 if percent else 1.0) for v in vals] for vals in seed_lists]

    ax.bar(
        x,
        plot_means,
        width=0.58,
        color=colors,
        edgecolor="#202020",
        linewidth=1.0,
        zorder=2,
    )
    ax.errorbar(
        x,
        plot_means,
        yerr=plot_cis,
        fmt="none",
        ecolor="#202020",
        elinewidth=1.25,
        capsize=4,
        zorder=3,
    )

    for i, vals in enumerate(plot_seeds):
        if len(vals) == 0:
            continue
        offs = np.linspace(-0.09, 0.09, len(vals))
        ax.scatter(
            np.full(len(vals), x[i]) + offs,
            vals,
            s=36,
            facecolors="white",
            edgecolors="#202020",
            linewidths=0.9,
            zorder=4,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, pad=10)
    beautify_ax(ax)

    y0, y1 = auto_ylim(
        [plot_means - plot_cis, plot_means + plot_cis] + plot_seeds,
        start_at_zero=True,
    )
    ax.set_ylim(y0, y1)


def savefig(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=600)
    plt.close(fig)
    print(f"PASS: {outpath}")


# ---------------------------
# Figure builders
# ---------------------------

def plot_main_icu(results_root: Path, outdir: Path):
    summary = load_json(results_root / "icu_sepsis" / "aggregate" / "main_summary.json")
    rows = load_csv_rows(results_root / "icu_sepsis" / "aggregate" / "per_seed_metrics.csv")

    order = [
        "Behavior Cloning",
        "CQL-regularised fitted-Q",
        "Vanilla Offline Actor-Critic",
        "Post-hoc Masked VOAC",
        "LAADAN-AC",
    ]
    labels = ["BC", "CQL", "VOAC", "Post-hoc\nmask", "LAADAN-\nAC"]
    colors = [METHOD_COLORS[m] for m in order]

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.0))
    fig.suptitle("ICU-Sepsis fixed-schedule policy comparison", y=1.04, fontsize=20, fontweight="semibold")

    metrics = [
        ("survival_rate", "Survival / return", "(a) Return", False),
        ("inadmissibility_rate", "Inadmissibility (%)", "(b) Selected-action inadmissibility", True),
        ("expert_argmax_match", "Expert argmax match", "(c) Expert alignment", False),
        ("mean_kl_to_expert", "KL to expert", "(d) Distributional deviation", False),
    ]

    for ax, (metric, ylabel, title, percent) in zip(axes, metrics):
        means, cis = summary_arrays(summary, order, metric)
        seeds = method_seed_lists(rows, order, metric)
        panel_bar(ax, labels, means, cis, seeds, colors, ylabel, title, percent=percent)

    fig.text(
        0.5, -0.02,
        "Bars: five-seed mean. Error bars: 95% CI across random training seeds. White circles: individual seeds.",
        ha="center", fontsize=12
    )
    savefig(fig, outdir / "main" / "fig2_icu_fixed_schedule_comparison.png")


def plot_ablation_icu(results_root: Path, outdir: Path):
    summary = load_json(results_root / "icu_sepsis" / "aggregate" / "ablation_summary.json")
    rows = load_csv_rows(results_root / "icu_sepsis" / "aggregate" / "ablation_per_seed_metrics.csv")

    order = [
        "Full LAADAN-AC",
        "Masking-only actor-critic",
        "LAADAN without action mask",
        "LAADAN without conservative critic",
        "LAADAN without expert KL",
        "LAADAN without smoothness proxy",
        "LAADAN without Lagrangian cost control",
    ]
    labels = [
        "Full",
        "Mask\nonly",
        "No\nmask",
        "No\nconservative",
        "No expert\nKL",
        "No\nsmoothness",
        "No\nLagrangian",
    ]
    colors = [METHOD_COLORS[m] for m in order]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))
    fig.suptitle("ICU-Sepsis fixed-schedule component analysis", y=1.01, fontsize=20, fontweight="semibold")
    axes = axes.ravel()

    metrics = [
        ("survival_rate", "Survival / return", "(a) Return", False),
        ("inadmissibility_rate", "Inadmissibility (%)", "(b) Selected-action inadmissibility", True),
        ("expert_argmax_match", "Expert argmax match", "(c) Expert alignment", False),
        ("mean_kl_to_expert", "KL to expert", "(d) Distributional deviation", False),
    ]

    for ax, (metric, ylabel, title, percent) in zip(axes, metrics):
        means, cis = summary_arrays(summary, order, metric)
        seeds = method_seed_lists(rows, order, metric)
        panel_bar(ax, labels, means, cis, seeds, colors, ylabel, title, percent=percent)

    fig.text(
        0.5, -0.01,
        "All variants use the fixed epoch-1000 protocol and seeds 42–46.",
        ha="center", fontsize=12
    )
    savefig(fig, outdir / "main" / "fig4_full_component_ablation.png")


def plot_eicu(results_root: Path, outdir: Path):
    summary = load_json(results_root / "eicu_demo" / "aggregate" / "main_summary.json")
    rows = load_csv_rows(results_root / "eicu_demo" / "aggregate" / "per_seed_metrics.csv")

    order = [
        "Behavior Cloning",
        "CQL-regularised fitted-Q",
        "Vanilla Offline Actor-Critic",
        "Post-hoc Masked VOAC",
        "LAADAN-AC",
    ]
    labels = ["BC", "CQL", "VOAC", "Post-hoc\nmask", "LAADAN-\nAC"]
    colors = [METHOD_COLORS[m] for m in order]

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.0))
    fig.suptitle("eICU Demo fixed-schedule policy comparison", y=1.04, fontsize=20, fontweight="semibold")

    metrics = [
        ("survival_rate", "Survival / return", "(a) Return", False),
        ("inadmissibility_rate", "Inadmissibility (%)", "(b) Selected-action inadmissibility", True),
        ("expert_argmax_match", "Expert argmax match", "(c) Expert alignment", False),
        ("mean_kl_to_expert", "KL to expert", "(d) Distributional deviation", False),
    ]

    for ax, (metric, ylabel, title, percent) in zip(axes, metrics):
        means, cis = summary_arrays(summary, order, metric)
        seeds = method_seed_lists(rows, order, metric)
        panel_bar(ax, labels, means, cis, seeds, colors, ylabel, title, percent=percent)

    fig.text(
        0.5, -0.02,
        "Supplementary portability view under the same fixed-schedule reporting protocol.",
        ha="center", fontsize=12
    )
    savefig(fig, outdir / "appendix" / "figA2_eicu_fixed_schedule_comparison.png")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    set_pub_style()
    plot_main_icu(args.results_root, args.outdir)
    plot_ablation_icu(args.results_root, args.outdir)
    plot_eicu(args.results_root, args.outdir)


if __name__ == "__main__":
    main()
