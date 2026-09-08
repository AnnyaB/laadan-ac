
"""Render the fixed-schedule appendix figures from frozen result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

SEEDS = [42, 43, 44, 45, 46]
T95_N5 = 2.776

INK = "#222222"
GRID = "#D9DDE3"
WHITE = "#FFFFFF"
BC = "#8EB1D2"
CQL = "#4D86C6"
VOAC = "#E45756"
POSTHOC = "#F2A541"
LAADAN = "#355C9A"
ICU = "#355C9A"
EICU = "#4DB26B"

METHOD_STYLE = {
    "Behavior Cloning": (BC, "BC"),
    "CQL-regularised fitted-Q": (CQL, "CQL"),
    "Vanilla Offline Actor-Critic": (VOAC, "VOAC"),
    "Post-hoc Masked VOAC": (POSTHOC, "Post-hoc"),
    "LAADAN-AC": (LAADAN, "LAADAN-AC"),
}

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
    "legend.fontsize": 7.5,
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


def save_figure(fig, out_dir: Path, stem: str, write_pdf: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{stem}.png"]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", pad_inches=0.04)
    validate_png(outputs[0])
    if write_pdf:
        outputs.append(out_dir / f"{stem}.pdf")
        fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    for path in outputs:
        print(f"PASS: {path}")
    return outputs


def load_metrics(results_root: Path, dataset: str, sources: set[Path]) -> pd.DataFrame:
    path = results_root / dataset / "aggregate" / "per_seed_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    sources.add(path)
    df = pd.read_csv(path)
    required = {
        "method", "seed", "survival_rate", "inadmissibility_rate",
        "expert_argmax_match", "mean_kl_to_expert", "soft_inadmissibility_rate",
        "state_mask_intervention_rate", "mean_inadmissible_probability_mass_before_mask",
        "max_inadmissible_probability_mass_before_mask",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def method_rows(df: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = df[df["method"] == method].sort_values("seed").reset_index(drop=True)
    seeds = rows["seed"].astype(int).tolist()
    if seeds != SEEDS:
        raise ValueError(f"{method}: expected seeds {SEEDS}; found {seeds}")
    return rows


def draw_metric(ax, df: pd.DataFrame, methods: list[str], metric: str, ylabel: str,
                title: str, *, percent: bool = False, ylim=None, labels: bool = True) -> None:
    x = np.arange(len(methods), dtype=float)
    jitter = np.linspace(-0.08, 0.08, 5)

    for xi, method in zip(x, methods):
        values = method_rows(df, method)[metric].astype(float).to_numpy()
        if percent:
            values *= 100.0
        mean, half = ci95(values)
        color, _ = METHOD_STYLE[method]
        ax.scatter(np.full(5, xi) + jitter, values, s=16, facecolors=WHITE,
                   edgecolors=color, linewidths=0.9, zorder=3)
        ax.errorbar(xi, mean, yerr=half, fmt="o", markersize=4.6,
                    markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.55,
                    ecolor=color, elinewidth=1.0, capsize=2.6, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_STYLE[m][1] for m in methods] if labels else [])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)


def plot_a1(results_root: Path, out_dir: Path, sources: set[Path], write_pdf: bool) -> list[Path]:
    eicu = load_metrics(results_root, "eicu_demo", sources)
    methods = list(METHOD_STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.7))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.95, hspace=0.40, wspace=0.30)
    draw_metric(axes[0, 0], eicu, methods, "survival_rate", "Return", "(a) Return", ylim=(0.0, 0.79), labels=False)
    draw_metric(axes[0, 1], eicu, methods, "inadmissibility_rate", "Inadmissibility (%)",
                "(b) Inadmissibility", percent=True, ylim=(0.0, 86.0), labels=False)
    draw_metric(axes[1, 0], eicu, methods, "expert_argmax_match", "Expert argmax match",
                "(c) Expert alignment", ylim=(0.0, 1.05))
    draw_metric(axes[1, 1], eicu, methods, "mean_kl_to_expert", "KL to expert",
                "(d) KL to expert", ylim=(0.0, 20.0))
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=18)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    return save_figure(fig, out_dir, "figA1_eicu_detailed_fixed_schedule", write_pdf)


def plot_b1(results_root: Path, out_dir: Path, sources: set[Path], write_pdf: bool) -> list[Path]:
    icu = load_metrics(results_root, "icu_sepsis", sources)
    post = method_rows(icu, "Post-hoc Masked VOAC")
    specs = [
        ("state_mask_intervention_rate", "(a) Greedy action changed", "Intervention (%)"),
        ("mean_inadmissible_probability_mass_before_mask", "(b) Mean inadmissible mass", "Probability mass (%)"),
        ("max_inadmissible_probability_mass_before_mask", "(c) Maximum inadmissible mass", "Probability mass (%)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.45))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.91, wspace=0.42)
    x = np.arange(5)

    for ax, (metric, title, ylabel) in zip(axes, specs):
        values = post[metric].astype(float).to_numpy() * 100.0
        mean, half = ci95(values)
        ax.fill_between([-0.25, 4.25], [mean - half] * 2, [mean + half] * 2,
                        color=LAADAN, alpha=0.12, linewidth=0, zorder=1)
        ax.axhline(mean, color=LAADAN, linewidth=1.1, zorder=2)
        ax.plot(x, values, marker="o", markersize=4.0, linewidth=0.9,
                color=VOAC, markerfacecolor=WHITE, markeredgewidth=0.9, zorder=3)
        ax.set_xticks(x, [str(seed) for seed in SEEDS])
        ax.set_xlabel("Training seed")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=5)
        style_axis(ax)

    return save_figure(fig, out_dir, "figB1_posthoc_intervention_diagnostics", write_pdf)


def plot_b2(results_root: Path, out_dir: Path, sources: set[Path], write_pdf: bool) -> list[Path]:
    icu = load_metrics(results_root, "icu_sepsis", sources)
    eicu = load_metrics(results_root, "eicu_demo", sources)
    datasets = [(icu, ICU, "o", "ICU-Sepsis"), (eicu, EICU, "s", "eICU Demo")]

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.55), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.82, wspace=0.30)
    offsets = [-0.12, 0.12]
    jitter = np.linspace(-0.03, 0.03, 5)

    for ax, methods, labels, ylim, title in [
        (axes[0], ["Behavior Cloning", "Vanilla Offline Actor-Critic", "LAADAN-AC"],
         ["BC", "VOAC", "LAADAN-AC"], (0.0, 90.0), "(a) Full scale"),
        (axes[1], ["Behavior Cloning", "LAADAN-AC"],
         ["BC", "LAADAN-AC"], (0.0, 1.10), "(b) Near-zero regime"),
    ]:
        x = np.arange(len(methods), dtype=float)
        for offset, (df, color, marker, _) in zip(offsets, datasets):
            for xi, method in zip(x, methods):
                values = method_rows(df, method)["soft_inadmissibility_rate"].astype(float).to_numpy() * 100.0
                mean, half = ci95(values)
                xpos = xi + offset
                ax.scatter(np.full(5, xpos) + jitter, values, s=16, facecolors=WHITE,
                           edgecolors=color, linewidths=0.9, zorder=3)
                ax.errorbar(xpos, mean, yerr=half, fmt=marker, markersize=4.7,
                            markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.55,
                            ecolor=color, elinewidth=1.0, capsize=2.6, zorder=4)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Soft inadmissible mass (%)")
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", pad=5)
        style_axis(ax)

    fig.legend(
        handles=[
            Line2D([0], [0], color=ICU, marker="o", markersize=4.8, linewidth=1.1, label="ICU-Sepsis"),
            Line2D([0], [0], color=EICU, marker="s", markersize=4.8, linewidth=1.1, label="eICU Demo"),
        ],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2,
    )
    return save_figure(fig, out_dir, "figB2_policy_distribution_admissibility", write_pdf)


def load_histories(results_root: Path, dataset: str, sources: set[Path]):
    train_metrics = ["expected_cost", "lagrange", "expert_kl", "smoothness"]
    eval_cols = ["avg_return", "survival_rate", "inadmissibility_rate", "expert_argmax_match"]
    stacks = {metric: [] for metric in train_metrics}
    epochs_ref = None

    for seed in SEEDS:
        path = results_root / dataset / "main_fixed_schedule" / "laadan_ac" / f"seed_{seed}" / "history.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.add(path)
        df = pd.read_csv(path)
        required = {"epoch", *train_metrics, *eval_cols}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        epochs = df["epoch"].astype(int).to_numpy()
        if len(epochs) != 1000 or epochs[0] != 1 or epochs[-1] != 1000:
            raise ValueError(f"{path}: expected exact epoch grid 1..1000")
        if epochs_ref is None:
            epochs_ref = epochs
        elif not np.array_equal(epochs_ref, epochs):
            raise ValueError(f"{path}: epoch grid differs across seeds")

        # benchmark evaluation is final-epoch only
        pre_final = df["epoch"] < 1000
        for column in eval_cols:
            if df.loc[pre_final, column].notna().any():
                raise ValueError(f"{path}: benchmark field {column} populated before epoch 1000")
            if pd.isna(df.loc[df["epoch"] == 1000, column]).all():
                raise ValueError(f"{path}: final benchmark field {column} missing")

        for metric in train_metrics:
            values = df[metric].astype(float).to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {metric}")
            stacks[metric].append(values)

    return epochs_ref, {metric: np.vstack(values) for metric, values in stacks.items()}


def plot_c1(results_root: Path, out_dir: Path, sources: set[Path], write_pdf: bool) -> list[Path]:
    epochs, icu = load_histories(results_root, "icu_sepsis", sources)
    eicu_epochs, eicu = load_histories(results_root, "eicu_demo", sources)
    if not np.array_equal(epochs, eicu_epochs):
        raise ValueError("ICU and eICU epoch grids differ")

    specs = [
        ("expected_cost", "(a) Expected training cost", "Expected cost"),
        ("lagrange", "(b) Lagrange multiplier", r"$\lambda$"),
        ("expert_kl", "(c) Expert-policy KL", "Expert KL"),
        ("smoothness", "(d) State-action smoothness", "Smoothness"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.4), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.95, hspace=0.34, wspace=0.28)

    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        for label, color, arrays in [("ICU-Sepsis", ICU, icu), ("eICU Demo", EICU, eicu)]:
            values = arrays[metric]
            mean = values.mean(axis=0)
            half = T95_N5 * values.std(axis=0, ddof=1) / np.sqrt(5.0)
            ax.plot(epochs, mean, color=color, linewidth=1.2, marker="o", markersize=2.2,
                    markevery=100, label=label, zorder=3)
            ax.fill_between(epochs, mean - half, mean + half, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.set_title(title, loc="left", pad=5)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 1000)
        style_axis(ax, "both")
        if metric == "lagrange":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    axes[1, 0].set_xlabel("Training epoch")
    axes[1, 1].set_xlabel("Training epoch")
    axes[0, 0].legend(frameon=False, loc="best")
    return save_figure(fig, out_dir, "figC1_training_objective_dynamics", write_pdf)


def write_manifest(out_dir: Path, sources: set[Path], outputs: list[Path]) -> None:
    payload = {
        "git_head": git_head(),
        "sources": {str(path): sha256(path) for path in sorted(sources)},
        "outputs": {str(path): sha256(path) for path in outputs},
        "seeds": SEEDS,
        "interval": "95% t-CI across five random training seeds",
        "checkpoint_rule": "pre-specified final epoch 1000",
    }
    path = out_dir / "appendix_plot_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/main"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/camera_ready/appendix"))
    parser.add_argument("--pdf", action="store_true", help="also write vector PDF files")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    sources: set[Path] = set()
    outputs: list[Path] = []

    outputs += plot_a1(results_root, out_dir, sources, args.pdf)
    outputs += plot_b1(results_root, out_dir, sources, args.pdf)
    outputs += plot_b2(results_root, out_dir, sources, args.pdf)
    outputs += plot_c1(results_root, out_dir, sources, args.pdf)
    write_manifest(out_dir, sources, outputs)


if __name__ == "__main__":
    main()
