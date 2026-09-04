"""Render the paired, cross-source, and policy-diagnostic camera-ready figures."""



from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


BLUE = "#4A90D9"
RED = "#E84A3C"
ORANGE = "#F28E2B"
PURPLE = "#8F63C7"
INK = "#202020"
MID_GREY = "#AFAFAF"
LIGHT_GREY = "#D7D7D7"
GRID = "#C9CDD2"
WHITE = "#FFFFFF"

T95_N5 = 2.776

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,

    "axes.titlesize": 11.0,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.7,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.85,

    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "xtick.color": INK,
    "ytick.color": INK,

    "legend.fontsize": 9.0,

    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 5, f"Expected 5 seeds, got {len(values)}"

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    ci = float(T95_N5 * std / np.sqrt(len(values)))

    return mean, ci


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)

    ax.tick_params(
        direction="out",
        width=0.75,
        length=3.5,
        pad=3,
    )

    ax.grid(
        axis=grid_axis,
        linestyle=(0, (3, 3)),
        linewidth=0.65,
        alpha=0.65,
        color=GRID,
    )

    ax.set_axisbelow(True)


def verify_600dpi(path: Path):
    image = Image.open(path)
    dpi = image.info.get("dpi")

    assert dpi is not None, f"No DPI metadata: {path}"
    assert 590 <= dpi[0] <= 610
    assert 590 <= dpi[1] <= 610

    return image.size, dpi


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.035,
    )

    plt.close(fig)

    size, dpi = verify_600dpi(path)

    print(
        f"PASS: {path}\n"
        f"      {size[0]} x {size[1]} px | "
        f"{dpi[0]:.2f} DPI"
    )



METHODS = [
    ("Behavior Cloning", "BC"),
    ("CQL-regularised fitted-Q", "CQL"),
    ("Vanilla Offline Actor-Critic", "VOAC"),
    ("Post-hoc Masked VOAC", "Post-hoc\nmask"),
    ("LAADAN-AC", "LAADAN-AC"),
]


def summary_value(summary, method, metric, percent=False):
    payload = summary[method][metric]

    mean = float(payload["mean"])
    ci = float(payload["ci95_half"])

    if percent:
        mean *= 100.0
        ci *= 100.0

    return mean, ci


def plot_cross_source(root: Path, out: Path):

    icu = load_json(
        root / "icu_sepsis/aggregate/main_summary.json"
    )

    eicu = load_json(
        root / "eicu_demo/aggregate/main_summary.json"
    )

    specs = [
        (
            "survival_rate",
            "(a) Return",
            "Survival / return",
            False,
            (0.0, 0.83),
        ),
        (
            "inadmissibility_rate",
            "(b) Selected-action inadmissibility",
            "Inadmissibility (%)",
            True,
            (0.0, 85.0),
        ),
        (
            "expert_argmax_match",
            "(c) Expert alignment",
            "Expert argmax match",
            False,
            (0.0, 1.06),
        ),
    ]

    labels = [display for _, display in METHODS]
    x = np.arange(len(METHODS))
    width = 0.34

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.8, 3.35),
        constrained_layout=True,
    )

    for ax, (metric, title, ylabel, percent, ylim) in zip(
        axes, specs
    ):

        icu_mean = []
        icu_ci = []

        eicu_mean = []
        eicu_ci = []

        for method, _ in METHODS:
            m, c = summary_value(
                icu,
                method,
                metric,
                percent=percent,
            )
            icu_mean.append(m)
            icu_ci.append(c)

            m, c = summary_value(
                eicu,
                method,
                metric,
                percent=percent,
            )
            eicu_mean.append(m)
            eicu_ci.append(c)

        ax.bar(
            x - width / 2,
            icu_mean,
            width,
            yerr=icu_ci,
            capsize=2.7,
            color=BLUE,
            alpha=0.92,
            edgecolor=INK,
            linewidth=0.55,
            error_kw={
                "elinewidth": 0.9,
                "capthick": 0.9,
            },
            label="ICU-Sepsis",
        )

        ax.bar(
            x + width / 2,
            eicu_mean,
            width,
            yerr=eicu_ci,
            capsize=2.7,
            color=RED,
            alpha=0.88,
            edgecolor=INK,
            linewidth=0.55,
            error_kw={
                "elinewidth": 0.9,
                "capthick": 0.9,
            },
            label="eICU Demo",
        )

        ax.set_title(
            title,
            loc="left",
            pad=7,
        )

        ax.set_ylabel(ylabel)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_ylim(*ylim)

        style_axis(ax)

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        facecolor=WHITE,
        framealpha=1.0,
        borderpad=0.35,
        handlelength=1.9,
        columnspacing=1.8,
    )

    fig.text(
        0.5,
        -0.015,
        "Bars show five-seed means; error bars are 95% t-CIs across random training seeds. "
        "eICU Demo is an exploratory cross-source portability check.",
        ha="center",
        va="top",
        fontsize=8.1,
        color="#444444",
    )

    save_figure(fig, out)



def plot_policy_diagnostics(root: Path, out: Path):

    csv_path = (
        root
        / "icu_sepsis/aggregate/per_seed_metrics.csv"
    )

    df = pd.read_csv(csv_path)

    df = df[
        df["method"] == "Post-hoc Masked VOAC"
    ].copy()

    df = df.sort_values("seed")

    assert df["seed"].tolist() == [42, 43, 44, 45, 46]

    specs = [
        (
            "state_mask_intervention_rate",
            "(a) Greedy action changed",
            "Post-hoc intervention (%)",
        ),
        (
            "mean_inadmissible_probability_mass_before_mask",
            "(b) Mean pre-mask inadmissible mass",
            "Probability mass (%)",
        ),
        (
            "max_inadmissible_probability_mass_before_mask",
            "(c) Maximum pre-mask inadmissible mass",
            "Probability mass (%)",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.15),
        constrained_layout=True,
    )

    y = np.arange(5)

    for ax, (metric, title, xlabel) in zip(axes, specs):

        values = (
            df[metric]
            .astype(float)
            .to_numpy()
            * 100.0
        )

        mean, ci = mean_ci95(values)

        span = float(values.max() - values.min())

        padding = max(
            0.40,
            span * 2.2,
            ci * 2.2,
        )

        lo = max(
            0.0,
            float(values.min()) - padding,
        )

        hi = min(
            100.0,
            float(values.max()) + padding,
        )

        if hi - lo < 0.8:
            centre = (hi + lo) / 2
            lo = max(0.0, centre - 0.45)
            hi = min(100.0, centre + 0.45)

        ax.axvspan(
            mean - ci,
            mean + ci,
            color=BLUE,
            alpha=0.11,
            linewidth=0,
            zorder=0,
        )

        ax.axvline(
            mean,
            color=BLUE,
            linewidth=1.65,
            linestyle=(0, (4, 2)),
            zorder=1,
        )

        ax.scatter(
            values,
            y,
            s=54,
            color=RED,
            edgecolor=INK,
            linewidth=0.7,
            zorder=4,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(
            [str(seed) for seed in df["seed"]]
        )

        ax.set_xlabel(xlabel)

        ax.set_title(
            title,
            loc="left",
            pad=7,
        )

        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.55, 4.55)
        ax.invert_yaxis()

        style_axis(
            ax,
            grid_axis="x",
        )

        ax.text(
            0.97,
            0.08,
            f"mean = {mean:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.6,
            color=INK,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor=WHITE,
                edgecolor="#C8C8C8",
                linewidth=0.55,
            ),
        )

    axes[0].set_ylabel("Training seed")

    fig.text(
        0.5,
        -0.02,
        "Red markers are individual seeds; dashed blue lines are seed means; "
        "blue bands are 95% t-CIs. Panel-specific x-scales expose seed variability.",
        ha="center",
        va="top",
        fontsize=8.1,
        color="#444444",
    )

    save_figure(fig, out)


def plot_paired(root: Path, out: Path):

    csv_path = (
        root
        / "icu_sepsis/aggregate/per_seed_metrics.csv"
    )

    paired_path = (
        root
        / "icu_sepsis/aggregate/"
        "paired_laadan_minus_posthoc_voac.json"
    )

    df = pd.read_csv(csv_path)
    paired = load_json(paired_path)

    post = (
        df[
            df["method"] == "Post-hoc Masked VOAC"
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    laadan = (
        df[
            df["method"] == "LAADAN-AC"
        ]
        .sort_values("seed")
        .reset_index(drop=True)
    )

    assert post["seed"].tolist() == [42, 43, 44, 45, 46]
    assert laadan["seed"].tolist() == [42, 43, 44, 45, 46]

    specs = [
        (
            "survival_rate",
            "(a) Return",
            "Survival / return",
        ),
        (
            "expert_argmax_match",
            "(b) Expert alignment",
            "Expert argmax match",
        ),
        (
            "mean_kl_to_expert",
            "(c) Distributional deviation",
            "KL to expert",
        ),
        (
            "mean_action_deviation_from_expert",
            "(d) Action deviation",
            "Mean action deviation",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.7, 6.15),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (metric, title, ylabel) in zip(
        axes, specs
    ):

        y0 = post[metric].astype(float).to_numpy()
        y1 = laadan[metric].astype(float).to_numpy()

        for idx in range(5):
            ax.plot(
                [0, 1],
                [y0[idx], y1[idx]],
                color=MID_GREY,
                linewidth=0.9,
                alpha=0.72,
                zorder=1,
            )

        ax.scatter(
            np.zeros(5),
            y0,
            s=40,
            color=RED,
            edgecolor=INK,
            linewidth=0.65,
            zorder=4,
        )

        ax.scatter(
            np.ones(5),
            y1,
            s=40,
            color=BLUE,
            edgecolor=INK,
            linewidth=0.65,
            zorder=4,
        )

        mean0, ci0 = mean_ci95(y0)
        mean1, ci1 = mean_ci95(y1)

        ax.errorbar(
            [0],
            [mean0],
            yerr=[ci0],
            fmt="D",
            markersize=5.2,
            color=RED,
            markeredgecolor=INK,
            markeredgewidth=0.65,
            capsize=3,
            elinewidth=1.1,
            zorder=6,
        )

        ax.errorbar(
            [1],
            [mean1],
            yerr=[ci1],
            fmt="D",
            markersize=5.2,
            color=BLUE,
            markeredgecolor=INK,
            markeredgewidth=0.65,
            capsize=3,
            elinewidth=1.1,
            zorder=6,
        )

        delta = paired["metrics"][metric]

        delta_mean = float(delta["mean"])
        delta_ci = float(delta["ci95_half"])

        ax.text(
            0.97,
            0.96,
            rf"$\Delta$ = {delta_mean:+.4f} $\pm$ {delta_ci:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.6,
            bbox=dict(
                boxstyle="round,pad=0.24",
                facecolor=WHITE,
                edgecolor="#C7C7C7",
                linewidth=0.55,
            ),
        )

        all_values = np.concatenate(
            [
                y0 - ci0,
                y0 + ci0,
                y1 - ci1,
                y1 + ci1,
            ]
        )

        ymin = float(all_values.min())
        ymax = float(all_values.max())

        span = max(
            ymax - ymin,
            abs(ymax) * 0.03,
            1e-4,
        )

        ax.set_ylim(
            ymin - 0.13 * span,
            ymax + 0.23 * span,
        )

        ax.set_xlim(-0.16, 1.16)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                "Post-hoc\nmask",
                "LAADAN-AC",
            ]
        )

        ax.set_ylabel(ylabel)

        ax.set_title(
            title,
            loc="left",
            pad=7,
        )

        style_axis(ax)


    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=RED,
            markeredgecolor=INK,
            label="Post-hoc masked VOAC",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=BLUE,
            markeredgecolor=INK,
            label="LAADAN-AC",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        facecolor=WHITE,
        framealpha=1.0,
        borderpad=0.35,
        handletextpad=0.5,
        columnspacing=1.5,
    )

    fig.text(
        0.5,
        -0.015,
        "Thin gray lines connect identical training seeds. Diamonds show five-seed means "
        "with 95% t-CIs; Δ denotes LAADAN-AC minus post-hoc masked VOAC.",
        ha="center",
        va="top",
        fontsize=8.1,
        color="#444444",
    )

    save_figure(fig, out)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        default="results/fixed_schedule_2026",
    )

    parser.add_argument(
        "--out-dir",
        default="figures/camera_ready",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir)

    required = [
        root
        / "icu_sepsis/aggregate/main_summary.json",

        root
        / "icu_sepsis/aggregate/per_seed_metrics.csv",

        root
        / "icu_sepsis/aggregate/"
          "paired_laadan_minus_posthoc_voac.json",

        root
        / "eicu_demo/aggregate/main_summary.json",
    ]

    for path in required:
        assert path.is_file(), f"Missing result artifact: {path}"

    plot_cross_source(
        root,
        out
        / "appendix/"
          "figA1_cross_source_portability.png",
    )

    plot_policy_diagnostics(
        root,
        out
        / "appendix/"
          "figB1_policy_distribution_diagnostics.png",
    )

    plot_paired(
        root,
        out
        / "main/"
          "fig3_paired_training_vs_posthoc.png",
    )

    print("\n==============================================")
    print("PASS: three redesigned camera-ready figures")
    print("were generated entirely from frozen results.")
    print("==============================================")


if __name__ == "__main__":
    main()
