"""Render the fixed-schedule safety-diagnostic figure from frozen tables."""


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

BLUE = "#4A90D9"
RED = "#E84A3C"
ORANGE = "#F28E2B"
INK = "#1F1F1F"
GREY = "#AFAFAF"
GRID = "#CCD1D6"
WHITE = "#FFFFFF"
T95_N5 = 2.776

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 11.0,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.8,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.85,
    "xtick.labelsize": 8.7,
    "ytick.labelsize": 8.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 8.3,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.75, length=3.3, pad=3)
    ax.grid(
        axis=grid_axis,
        linestyle=(0, (3, 3)),
        linewidth=0.65,
        color=GRID,
        alpha=0.65,
    )
    ax.set_axisbelow(True)


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 5, f"Expected 5 seeds, found {len(values)}"
    mean = float(values.mean())
    ci = float(T95_N5 * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, ci


def verify_600dpi(path: Path):
    image = Image.open(path)
    dpi = image.info.get("dpi")
    assert dpi is not None, f"Missing DPI metadata: {path}"
    assert 590 <= dpi[0] <= 610 and 590 <= dpi[1] <= 610, dpi
    return image.size, dpi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/fixed_schedule_2026",
    )
    parser.add_argument(
        "--output",
        default="figures/camera_ready/appendix/figC1_safety_failure_diagnostics.png",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    output = Path(args.output)

    safety_root = root / "icu_sepsis/safety_diagnostics"
    tables = safety_root / "illustrative_seed/seed_42/tables"

    per_seed = pd.read_csv(safety_root / "per_seed_diagnostics.csv")
    actions = pd.read_csv(tables / "action_level_inadmissibility_counts.csv")
    qvalues = pd.read_csv(tables / "representative_state_q_values.csv")
    trajectory = pd.read_csv(tables / "sample_trajectory_voac_vs_laadan.csv")

    assert per_seed["seed"].tolist() == [42, 43, 44, 45, 46]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 6.7),
        constrained_layout=True,
    )

    ax = axes[0, 0]
    state_metrics = [
        (
            "voac_fraction_states_with_inadmissible_greedy_action",
            "VOAC",
            RED,
        ),
        (
            "posthoc_mean_state_inadmissibility",
            "Post-hoc\nmask",
            ORANGE,
        ),
        (
            "laadan_mean_state_inadmissibility",
            "LAADAN-AC",
            BLUE,
        ),
    ]

    for index, (column, label, color) in enumerate(state_metrics):
        values = per_seed[column].to_numpy(dtype=float) * 100.0
        mean, ci = mean_ci95(values)
        jitter = np.linspace(-0.08, 0.08, len(values))

        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=34,
            color=color,
            edgecolor=INK,
            linewidth=0.55,
            zorder=4,
        )
        ax.errorbar(
            index,
            mean,
            yerr=ci,
            fmt="D",
            markersize=5.2,
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.55,
            capsize=3,
            elinewidth=1.1,
            zorder=5,
        )

    ax.set_xticks(
        np.arange(len(state_metrics)),
        [item[1] for item in state_metrics],
    )
    ax.set_ylabel("States with inadmissible\ngreedy action (%)")
    ax.set_title("(a) State-level failure rate", loc="left", pad=7)
    ax.set_ylim(-1, 47)
    style_axis(ax)

    ax = axes[0, 1]
    voac_actions = actions[
        actions["model"] == "Vanilla Offline Actor-Critic"
    ].copy()
    voac_actions = voac_actions[voac_actions["selected_count"] > 0]
    voac_actions = voac_actions.sort_values("action")

    action_ids = voac_actions["action"].astype(int).to_numpy()
    selected = voac_actions["selected_count"].to_numpy(dtype=float)
    inadmissible = voac_actions["inadmissible_selected_count"].to_numpy(dtype=float)
    x = np.arange(len(action_ids))

    ax.bar(
        x,
        selected,
        width=0.62,
        color="#E8E8E8",
        edgecolor=INK,
        linewidth=0.5,
        label="Selected states",
    )
    ax.bar(
        x,
        inadmissible,
        width=0.62,
        color=RED,
        alpha=0.94,
        edgecolor=INK,
        linewidth=0.5,
        label="Inadmissible",
    )

    ax.set_xticks(x, action_ids)
    ax.set_xlabel("Treatment-action bin")
    ax.set_ylabel("Number of states")
    ax.set_title("(b) Action-level VOAC failures (seed 42)", loc="left", pad=7)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax)


    ax = axes[1, 0]
    representative_state = int(qvalues["state_id"].iloc[0])
    q = qvalues[
        (qvalues["state_id"] == representative_state)
        & qvalues["model"].isin(
            [
                "Vanilla Offline Actor-Critic",
                "Conservative Q-Learning",
                "LAADAN-AC",
            ]
        )
    ]

    display_models = [
        ("Vanilla Offline Actor-Critic", "VOAC"),
        ("Conservative Q-Learning", "CQL"),
        ("LAADAN-AC", "LAADAN-AC"),
    ]
    ypos = np.arange(len(display_models))[::-1]

    for y, (model, short_name) in zip(ypos, display_models):
        subset = q[q["model"] == model]
        admissible = subset[subset["admissible"] == 1]["q_value"].to_numpy(dtype=float)
        inadmissible_q = subset[subset["admissible"] == 0]["q_value"].to_numpy(dtype=float)

        admissible_mean = float(admissible.mean())
        inadmissible_mean = float(inadmissible_q.mean())

        ax.plot(
            [inadmissible_mean, admissible_mean],
            [y, y],
            color=GREY,
            linewidth=2.0,
            zorder=1,
        )
        ax.scatter(
            inadmissible_mean,
            y,
            s=55,
            color=RED,
            edgecolor=INK,
            linewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            admissible_mean,
            y,
            s=55,
            color=BLUE,
            edgecolor=INK,
            linewidth=0.6,
            zorder=3,
        )

        if len(inadmissible_q) > 1:
            ax.plot(
                [inadmissible_q.min(), inadmissible_q.max()],
                [y - 0.13, y - 0.13],
                color=RED,
                linewidth=1.1,
                alpha=0.70,
            )

    ax.set_yticks(ypos, [item[1] for item in display_models])
    ax.set_xlabel("Q value")
    ax.set_title(
        f"(c) Q-value separation at representative state {representative_state}",
        loc="left",
        pad=7,
    )
    style_axis(ax, grid_axis="x")

    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=BLUE,
                markeredgecolor=INK,
                label="Admissible mean",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=RED,
                markeredgecolor=INK,
                label="Inadmissible mean",
            ),
        ],
        frameon=False,
        loc="lower right",
    )

    ax = axes[1, 1]

    for model, color, label in [
        ("Vanilla Offline Actor-Critic", RED, "VOAC"),
        ("LAADAN-AC", BLUE, "LAADAN-AC"),
    ]:
        subset = trajectory[trajectory["model"] == model].sort_values("step")
        ax.plot(
            subset["step"],
            subset["action"],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=color,
            label=label,
        )

    voac_traj = trajectory[
        trajectory["model"] == "Vanilla Offline Actor-Critic"
    ].sort_values("step")
    bad = voac_traj[voac_traj["admissible"] == 0]

    ax.scatter(
        bad["step"],
        bad["action"],
        marker="x",
        s=55,
        color=INK,
        linewidth=1.5,
        label="VOAC inadmissible",
        zorder=5,
    )

    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("Treatment-action bin")
    ax.set_title(
        "(d) Matched-initial-state trajectory (seed 42)",
        loc="left",
        pad=7,
    )
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)

    fig.text(
        0.5,
        -0.01,
        "Quantitative state-level summaries use fixed-schedule seeds 42–46; "
        "panels (b)–(d) use predeclared illustrative seed 42, not a performance-selected run.",
        ha="center",
        va="top",
        fontsize=8.1,
        color="#444444",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    size, dpi = verify_600dpi(output)
    print(f"PASS: {output}")
    print(f"      {size[0]} x {size[1]} px")
    print(f"      {dpi[0]:.2f} DPI")


if __name__ == "__main__":
    main()
