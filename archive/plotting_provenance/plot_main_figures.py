
"""Render the fixed-schedule main-result figures from frozen metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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

METHODS = [
    ("Behavior Cloning", "BC", BC),
    ("CQL-regularised fitted-Q", "CQL", CQL),
    ("Vanilla Offline Actor-Critic", "VOAC", VOAC),
    ("Post-hoc Masked VOAC", "Post-hoc", POSTHOC),
    ("LAADAN-AC", "LAADAN-AC", LAADAN),
]

ABLATIONS = [
    ("Full LAADAN-AC", "Full", LAADAN),
    ("Masking-only actor-critic", "Mask\nonly", POSTHOC),
    ("LAADAN without action mask", "No\nmask", VOAC),
    ("LAADAN without conservative critic", "No\nconservative", "#A45B9E"),
    ("LAADAN without expert KL", "No expert\nKL", "#7C6BB5"),
    ("LAADAN without smoothness proxy", "No\nsmoothness", "#C66B63"),
    ("LAADAN without Lagrangian cost control", "No\nLagrangian", "#668D52"),
]

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
    "xtick.major.width": 0.72,
    "ytick.major.width": 0.72,
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
        "expert_argmax_match", "mean_kl_to_expert",
        "mean_action_deviation_from_expert",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def method_values(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    rows = df[df["method"] == method].sort_values("seed")
    seeds = rows["seed"].astype(int).tolist()
    if seeds != SEEDS:
        raise ValueError(f"{method}: expected seeds {SEEDS}; found {seeds}")
    values = rows[metric].astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{method}/{metric}: non-finite values")
    return values


def padded_ylim(values: list[float], *, floor_zero: bool = False) -> tuple[float, float]:
    lo = float(min(values))
    hi = float(max(values))
    span = max(hi - lo, abs(hi) * 0.05, 1e-4)
    lo -= 0.16 * span
    hi += 0.20 * span
    if floor_zero:
        lo = max(0.0, lo)
    return lo, hi


def plot_paired(icu: pd.DataFrame, out_dir: Path, write_pdf: bool) -> list[Path]:
    specs = [
        ("survival_rate", "(a) Return", "Return"),
        ("expert_argmax_match", "(b) Expert alignment", "Expert argmax match"),
        ("mean_kl_to_expert", "(c) Distributional deviation", "KL to expert"),
        ("mean_action_deviation_from_expert", "(d) Action deviation", "Mean action deviation"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.94, hspace=0.38, wspace=0.30)

    for ax, (metric, title, ylabel) in zip(axes.ravel(), specs):
        post = method_values(icu, "Post-hoc Masked VOAC", metric)
        laadan = method_values(icu, "LAADAN-AC", metric)

        # pair identical training seeds
        for y0, y1 in zip(post, laadan):
            ax.plot([0, 1], [y0, y1], color="#B7B7B7", linewidth=0.9, zorder=1)

        ax.scatter(np.zeros(5), post, s=34, color=POSTHOC, edgecolor=INK, linewidth=0.6, zorder=3)
        ax.scatter(np.ones(5), laadan, s=34, color=LAADAN, edgecolor=INK, linewidth=0.6, zorder=3)

        post_mean, post_ci = ci95(post)
        laadan_mean, laadan_ci = ci95(laadan)
        ax.errorbar(0, post_mean, yerr=post_ci, fmt="D", markersize=4.8, color=POSTHOC,
                    markeredgecolor=INK, markeredgewidth=0.55, capsize=2.8, elinewidth=1.0, zorder=5)
        ax.errorbar(1, laadan_mean, yerr=laadan_ci, fmt="D", markersize=4.8, color=LAADAN,
                    markeredgecolor=INK, markeredgewidth=0.55, capsize=2.8, elinewidth=1.0, zorder=5)

        delta = laadan - post
        delta_mean, delta_ci = ci95(delta)
        ax.text(
            0.97, 0.96, rf"$\Delta$ = {delta_mean:+.4f} $\pm$ {delta_ci:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.20", facecolor=WHITE, edgecolor="#CFCFCF", linewidth=0.5),
        )

        lo, hi = padded_ylim([*(post - post_ci), *(post + post_ci), *(laadan - laadan_ci), *(laadan + laadan_ci)])
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.18, 1.18)
        ax.set_xticks([0, 1], ["Post-hoc mask", "LAADAN-AC"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=5)
        style_axis(ax)

    return save_figure(fig, out_dir, "fig2_training_vs_posthoc", write_pdf)


def plot_ablation(results_root: Path, out_dir: Path, sources: set[Path], write_pdf: bool) -> list[Path]:
    path = results_root / "icu_sepsis" / "aggregate" / "ablation_per_seed_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    sources.add(path)
    df = pd.read_csv(path)

    specs = [
        ("survival_rate", "(a) Return", "Return", 1.0),
        ("inadmissibility_rate", "(b) Selected-action inadmissibility", "Inadmissibility (%)", 100.0),
        ("expert_argmax_match", "(c) Expert alignment", "Expert argmax match", 1.0),
        ("mean_kl_to_expert", "(d) Distributional deviation", "KL to expert", 1.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.1))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.95, hspace=0.40, wspace=0.28)
    x = np.arange(len(ABLATIONS), dtype=float)

    for ax, (metric, title, ylabel, scale) in zip(axes.ravel(), specs):
        means, cis = [], []
        for method, _, _ in ABLATIONS:
            values = method_values(df, method, metric) * scale
            mean, half = ci95(values)
            means.append(mean)
            cis.append(half)

        ax.bar(
            x, means, width=0.62, color=[c for _, _, c in ABLATIONS], edgecolor=INK, linewidth=0.55,
            yerr=cis, error_kw={"ecolor": INK, "elinewidth": 0.85, "capsize": 2.4, "capthick": 0.8}, zorder=2,
        )

        for i, (method, _, color) in enumerate(ABLATIONS):
            values = method_values(df, method, metric) * scale
            jitter = np.linspace(-0.10, 0.10, 5)
            ax.scatter(np.full(5, i) + jitter, values, s=16, facecolors=WHITE,
                       edgecolors=color, linewidths=0.9, zorder=4)

        ax.set_xticks(x, ["Full", "Mask", "−Mask", "−Cons.", "−KL", "−Smooth", "−Lag."])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=5)
        style_axis(ax)
        if metric == "inadmissibility_rate":
            upper = max(max(means) + max(cis), 0.02)
            ax.set_ylim(0.0, upper * 1.22)

    return save_figure(fig, out_dir, "fig3_component_ablation", write_pdf)


def plot_cross_source(
    icu: pd.DataFrame,
    eicu: pd.DataFrame,
    out_dir: Path,
    write_pdf: bool,
) -> list[Path]:
    specs = [
        ("survival_rate", "(a) Return", "Return", 1.0, (0.0, 0.84)),
        ("inadmissibility_rate", "(b) Selected-action inadmissibility", "Inadmissibility (%)", 100.0, (0.0, 85.0)),
        ("expert_argmax_match", "(c) Expert alignment", "Expert argmax match", 1.0, (0.0, 1.06)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.65))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.84, wspace=0.34)
    x = np.arange(len(METHODS), dtype=float)
    width = 0.34

    for ax, (metric, title, ylabel, scale, ylim) in zip(axes, specs):
        for offset, df, color, label in [
            (-width / 2, icu, ICU, "ICU-Sepsis"),
            (width / 2, eicu, EICU, "eICU Demo"),
        ]:
            means, cis = [], []
            for method, _, _ in METHODS:
                values = method_values(df, method, metric) * scale
                mean, half = ci95(values)
                means.append(mean)
                cis.append(half)
            ax.bar(
                x + offset, means, width=width, color=color, edgecolor=INK, linewidth=0.5,
                yerr=cis, error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 2.0}, label=label, zorder=2,
            )

        short_labels = ["BC", "CQL", "VOAC", "Post", "LAADAN"]
        ax.set_xticks(x, short_labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", pad=5)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    return save_figure(fig, out_dir, "fig4_cross_source_portability", write_pdf)


def write_manifest(out_dir: Path, sources: set[Path], outputs: list[Path]) -> None:
    payload = {
        "git_head": git_head(),
        "sources": {str(path): sha256(path) for path in sorted(sources)},
        "outputs": {str(path): sha256(path) for path in outputs},
        "seeds": SEEDS,
        "interval": "95% t-CI across five random training seeds",
    }
    path = out_dir / "main_plot_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/main"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/camera_ready/main"))
    parser.add_argument("--pdf", action="store_true", help="also write vector PDF files")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    sources: set[Path] = set()

    icu = load_metrics(results_root, "icu_sepsis", sources)
    eicu = load_metrics(results_root, "eicu_demo", sources)

    outputs: list[Path] = []
    outputs += plot_paired(icu, out_dir, args.pdf)
    outputs += plot_ablation(results_root, out_dir, sources, args.pdf)
    outputs += plot_cross_source(icu, eicu, out_dir, args.pdf)
    write_manifest(out_dir, sources, outputs)


if __name__ == "__main__":
    main()
