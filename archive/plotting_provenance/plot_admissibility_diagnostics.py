
"""Render the fixed-schedule admissibility diagnostics from frozen tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

SEEDS = [42, 43, 44, 45, 46]
T95_N5 = 2.776

INK = "#222222"
GRID = "#D9DDE3"
WHITE = "#FFFFFF"
VOAC = "#E45756"
POSTHOC = "#F2A541"
LAADAN = "#355C9A"
CQL = "#4D86C6"
GREY = "#AFAFAF"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.4,
    "font.weight": "normal",
    "axes.titlesize": 9.2,
    "axes.titleweight": "normal",
    "axes.labelsize": 8.7,
    "axes.labelweight": "normal",
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.4,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.72,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError(f"Expected five finite seed values, got {values}")
    mean = float(values.mean())
    half = float(T95_N5 * values.std(ddof=1) / np.sqrt(5.0))
    return mean, half


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.72, length=3.0, pad=3)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.65, alpha=0.55)
    ax.set_axisbelow(True)


def validate_png(path: Path) -> None:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        if dpi is None or not (590 <= dpi[0] <= 610 and 590 <= dpi[1] <= 610):
            raise RuntimeError(f"Unexpected PNG DPI for {path}: {dpi}")


def save_figure(fig, out_dir: Path, write_pdf: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / "fig5_admissibility_failure_diagnostics.png"]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.04)
    validate_png(outputs[0])
    if write_pdf:
        outputs.append(out_dir / "fig5_admissibility_failure_diagnostics.pdf")
        fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    for path in outputs:
        print(f"PASS: {path}")
    return outputs


def load_tables(results_root: Path, sources: set[Path]):
    root = results_root / "icu_sepsis" / "safety_diagnostics"
    tables = root / "illustrative_seed" / "seed_42" / "tables"
    paths = {
        "per_seed": root / "per_seed_diagnostics.csv",
        "actions": tables / "action_level_inadmissibility_counts.csv",
        "qvalues": tables / "representative_state_q_values.csv",
        "trajectory": tables / "sample_trajectory_voac_vs_laadan.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing diagnostic artifacts:\n" + "\n".join(map(str, missing)))
    sources.update(paths.values())
    return {name: pd.read_csv(path) for name, path in paths.items()}


def plot_diagnostics(data: dict[str, pd.DataFrame], out_dir: Path, write_pdf: bool) -> list[Path]:
    per_seed = data["per_seed"].sort_values("seed").reset_index(drop=True)
    if per_seed["seed"].astype(int).tolist() != SEEDS:
        raise ValueError(f"Expected seeds {SEEDS}")

    actions = data["actions"]
    qvalues = data["qvalues"]
    trajectory = data["trajectory"]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.95, hspace=0.38, wspace=0.32)

    ax = axes[0, 0]
    state_metrics = [
        ("voac_fraction_states_with_inadmissible_greedy_action", "VOAC", VOAC),
        ("posthoc_mean_state_inadmissibility", "Post-hoc", POSTHOC),
        ("laadan_mean_state_inadmissibility", "LAADAN-AC", LAADAN),
    ]
    for i, (column, label, color) in enumerate(state_metrics):
        if column not in per_seed.columns:
            raise ValueError(f"Missing diagnostic column: {column}")
        values = per_seed[column].to_numpy(float) * 100.0
        mean, half = ci95(values)
        jitter = np.linspace(-0.07, 0.07, 5)
        ax.scatter(np.full(5, i) + jitter, values, s=25, color=color,
                   edgecolor=INK, linewidth=0.5, zorder=3)
        ax.errorbar(i, mean, yerr=half, fmt="D", markersize=4.5, color=color,
                    markeredgecolor=INK, markeredgewidth=0.5, capsize=2.6, elinewidth=1.0, zorder=4)
    ax.set_xticks(range(3), [item[1] for item in state_metrics])
    ax.set_ylabel("States with inadmissible\ngreedy action (%)")
    ax.set_title("(a) State-level inadmissibility", loc="left", pad=5)
    ax.set_ylim(-1.0, 47.0)
    style_axis(ax)

    ax = axes[0, 1]
    voac_actions = actions[actions["model"] == "Vanilla Offline Actor-Critic"].copy()
    voac_actions = voac_actions[voac_actions["selected_count"] > 0].sort_values("action")
    action_ids = voac_actions["action"].astype(int).to_numpy()
    selected = voac_actions["selected_count"].to_numpy(float)
    inadmissible = voac_actions["inadmissible_selected_count"].to_numpy(float)
    x = np.arange(len(action_ids))
    ax.bar(x, selected, width=0.62, color="#E8E8E8", edgecolor=INK, linewidth=0.45, label="Selected states")
    ax.bar(x, inadmissible, width=0.62, color=VOAC, edgecolor=INK, linewidth=0.45, label="Inadmissible")
    ax.set_xticks(x, action_ids)
    ax.set_xlabel("Treatment-action bin")
    ax.set_ylabel("Number of states")
    ax.set_title("(b) Action-level VOAC failures (seed 42)", loc="left", pad=5)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(ax)

    ax = axes[1, 0]
    state_id = int(qvalues["state_id"].iloc[0])
    q = qvalues[qvalues["state_id"] == state_id]
    display_models = [
        ("Vanilla Offline Actor-Critic", "VOAC"),
        ("Conservative Q-Learning", "CQL"),
        ("LAADAN-AC", "LAADAN-AC"),
    ]
    ypos = np.arange(len(display_models))[::-1]
    for y, (model, short) in zip(ypos, display_models):
        subset = q[q["model"] == model]
        admissible = subset[subset["admissible"] == 1]["q_value"].to_numpy(float)
        inadmissible_q = subset[subset["admissible"] == 0]["q_value"].to_numpy(float)
        if len(admissible) == 0 or len(inadmissible_q) == 0:
            raise ValueError(f"Missing admissible/inadmissible Q values for {model}")
        adm_mean = float(admissible.mean())
        inad_mean = float(inadmissible_q.mean())
        ax.plot([inad_mean, adm_mean], [y, y], color=GREY, linewidth=1.8, zorder=1)
        ax.scatter(inad_mean, y, s=40, color=VOAC, edgecolor=INK, linewidth=0.5, zorder=3)
        ax.scatter(adm_mean, y, s=40, color=LAADAN, edgecolor=INK, linewidth=0.5, zorder=3)
        if len(inadmissible_q) > 1:
            ax.plot([inadmissible_q.min(), inadmissible_q.max()], [y - 0.12, y - 0.12],
                    color=VOAC, linewidth=1.0, alpha=0.70)
    ax.set_yticks(ypos, [short for _, short in display_models])
    ax.set_xlabel("Q value")
    ax.set_title(f"(c) Q-value separation at state {state_id}", loc="left", pad=5)
    style_axis(ax, "x")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="None", markersize=5.5,
                   markerfacecolor=LAADAN, markeredgecolor=INK, label="Admissible mean"),
            Line2D([0], [0], marker="o", linestyle="None", markersize=5.5,
                   markerfacecolor=VOAC, markeredgecolor=INK, label="Inadmissible mean"),
        ],
        frameon=False, loc="best",
    )

    ax = axes[1, 1]
    for model, color, label in [
        ("Vanilla Offline Actor-Critic", VOAC, "VOAC"),
        ("LAADAN-AC", LAADAN, "LAADAN-AC"),
    ]:
        subset = trajectory[trajectory["model"] == model].sort_values("step")
        ax.plot(subset["step"], subset["action"], marker="o", markersize=3.8,
                linewidth=1.5, color=color, label=label)
    voac_traj = trajectory[trajectory["model"] == "Vanilla Offline Actor-Critic"].sort_values("step")
    bad = voac_traj[voac_traj["admissible"] == 0]
    ax.scatter(bad["step"], bad["action"], marker="x", s=42, color=INK,
               linewidth=1.3, label="VOAC inadmissible", zorder=5)
    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("Treatment-action bin")
    ax.set_title("(d) Matched-initial-state trajectory (seed 42)", loc="left", pad=5)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)

    return save_figure(fig, out_dir, write_pdf)


def make_animation(trajectory: pd.DataFrame, out_dir: Path) -> Path:
    # fixed illustrative seed
    voac = trajectory[trajectory["model"] == "Vanilla Offline Actor-Critic"].sort_values("step")
    laadan = trajectory[trajectory["model"] == "LAADAN-AC"].sort_values("step")
    max_step = int(max(voac["step"].max(), laadan["step"].max()))

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    style_axis(ax)
    ax.set_xlim(-0.25, max_step + 0.4)
    ax.set_ylim(-1, 25)
    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("Treatment-action bin")
    ax.set_title("Matched-initial-state policy trajectory (seed 42)")

    line_voac, = ax.plot([], [], color=VOAC, linewidth=1.8, marker="o", label="VOAC")
    line_laadan, = ax.plot([], [], color=LAADAN, linewidth=1.8, marker="o", label="LAADAN-AC")
    bad_scatter = ax.scatter([], [], marker="x", s=70, color=INK, linewidth=1.4, label="VOAC inadmissible")
    ax.legend(frameon=False, loc="upper left")

    def update(frame):
        vv = voac[voac["step"] <= frame]
        ll = laadan[laadan["step"] <= frame]
        line_voac.set_data(vv["step"], vv["action"])
        line_laadan.set_data(ll["step"], ll["action"])
        bad = vv[vv["admissible"] == 0]
        offsets = np.column_stack([bad["step"].to_numpy(), bad["action"].to_numpy()]) if len(bad) else np.empty((0, 2))
        bad_scatter.set_offsets(offsets)
        return line_voac, line_laadan, bad_scatter

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trajectory_animation.gif"
    animation = FuncAnimation(fig, update, frames=range(max_step + 1), interval=650, blit=False, repeat=True)
    animation.save(path, writer=PillowWriter(fps=2), dpi=120)
    plt.close(fig)
    print(f"PASS: {path}")
    return path


def write_manifest(out_dir: Path, sources: set[Path], outputs: list[Path]) -> None:
    payload = {
        "git_head": git_head(),
        "sources": {str(path): sha256(path) for path in sorted(sources)},
        "outputs": {str(path): sha256(path) for path in outputs},
        "seeds": SEEDS,
        "illustrative_seed": 42,
    }
    path = out_dir / "diagnostic_plot_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/main"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/camera_ready/main"))
    parser.add_argument("--pdf", action="store_true", help="also write a vector PDF")
    parser.add_argument("--animate", action="store_true", help="also write the seed-42 trajectory GIF")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    sources: set[Path] = set()
    data = load_tables(results_root, sources)

    outputs = plot_diagnostics(data, out_dir, args.pdf)
    if args.animate:
        outputs.append(make_animation(data["trajectory"], out_dir))
    write_manifest(out_dir, sources, outputs)


if __name__ == "__main__":
    main()
