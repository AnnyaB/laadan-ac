
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

COLORS = {
    "bc": "#C2DDF8",             
    "cql": "#5EA3EF",            
    "voac": "#D55745",          
    "posthoc": "#EF8733",       
    "laadan": "#0000F5",         
    "mask_only": "#EF8733",
    "no_mask": "#D55745",
    "no_conservative": "#C62F7C",
    "no_expert_kl": "#9352D5",  
    "no_smoothness": "#EC5F59",  
    "no_lagrangian": "#377E21", 
    "icu": "#20469B",          
    "eicu": "#72C580",          
    "gray": "#6B6B6B",
    "light_gray": "#D9D9D9",
}

METHODS = [
    ("Behavior Cloning", "BC", "bc"),
    ("CQL-regularised fitted-Q", "CQL", "cql"),
    ("Vanilla Offline Actor-Critic", "VOAC", "voac"),
    ("Post-hoc Masked VOAC", "Post-hoc\nmask", "posthoc"),
    ("LAADAN-AC", "LAADAN-\nAC", "laadan"),
]

ABLATIONS = [
    ("Full LAADAN-AC", "Full", "laadan"),
    ("Masking-only actor-critic", "Mask\nonly", "mask_only"),
    ("LAADAN without action mask", "No\nmask", "no_mask"),
    ("LAADAN without conservative critic", "No\nconservative", "no_conservative"),
    ("LAADAN without expert KL", "No expert\nKL", "no_expert_kl"),
    ("LAADAN without smoothness proxy", "No\nsmoothness", "no_smoothness"),
    ("LAADAN without Lagrangian cost control", "No\nLagrangian", "no_lagrangian"),
]

SEEDS = [42, 43, 44, 45, 46]


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.2,
        "axes.titlesize": 11.2,
        "axes.labelsize": 10.6,
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
    })


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.55, color="#BDBDBD", alpha=0.55)
        ax.set_axisbelow(True)


def panel_title(ax, letter: str, title: str):
    ax.set_title(f"({letter}) {title}", loc="left", pad=8, fontweight="normal")


def ensure_dirs(root: Path):
    for sub in ["main", "appendix", "supplementary"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def summary_stat(summary, method, metric):
    d = summary[method][metric]
    return float(d["mean"]), float(d["ci95_half"])


def values_for(rows, method, metric, scale=1.0):
    s = rows.loc[rows["method"] == method, ["seed", metric]].dropna().sort_values("seed")
    return s["seed"].to_numpy(int), s[metric].to_numpy(float) * scale


def draw_bar_seed_panel(ax, summary, rows, methods, metric, ylabel, scale=1.0, ylim=None):
    x = np.arange(len(methods), dtype=float)
    means, cis = [], []
    colors = []
    for method, _, key in methods:
        mean, ci = summary_stat(summary, method, metric)
        means.append(mean * scale)
        cis.append(ci * scale)
        colors.append(COLORS[key])

    ax.bar(
        x,
        means,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
        yerr=cis,
        error_kw={"ecolor": "black", "elinewidth": 0.85, "capsize": 2.8, "capthick": 0.85},
        zorder=2,
    )

    for i, (method, _, key) in enumerate(methods):
        _, vals = values_for(rows, method, metric, scale)
        offsets = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(
            i + offsets,
            vals,
            s=22,
            facecolor="white",
            edgecolor="black",
            linewidth=0.65,
            zorder=5,
        )

    ax.set_xticks(x, [m[1] for m in methods])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean_axis(ax)


def fig_main_icu(results_root: Path, out: Path):
    root = results_root / "icu_sepsis"
    summary = load_json(root / "aggregate/main_summary.json")
    rows = pd.read_csv(root / "aggregate/per_seed_metrics.csv")

    fig, axes = plt.subplots(1, 4, figsize=(13.7, 3.25), layout="constrained")
    specs = [
        ("survival_rate", "Survival / return", 1.0, None, "a", "Return"),
        ("inadmissibility_rate", "Inadmissibility (%)", 100.0, None, "b", "Selected-action inadmissibility"),
        ("expert_argmax_match", "Expert argmax match", 1.0, (0.0, 1.08), "c", "Expert alignment"),
        ("mean_kl_to_expert", "KL to expert", 1.0, None, "d", "Distributional deviation"),
    ]
    for ax, (metric, ylabel, scale, ylim, letter, title) in zip(axes, specs):
        draw_bar_seed_panel(ax, summary, rows, METHODS, metric, ylabel, scale, ylim)
        panel_title(ax, letter, title)
    fig.savefig(out / "main/fig2_icu_fixed_schedule_comparison.png", dpi=600)
    plt.close(fig)


def fig_paired(results_root: Path, out: Path):
    root = results_root / "icu_sepsis"
    rows = pd.read_csv(root / "aggregate/per_seed_metrics.csv")
    paired = load_json(root / "aggregate/paired_laadan_minus_posthoc_voac.json")

    specs = [
        ("survival_rate", "Survival / return", "a", "Return"),
        ("expert_argmax_match", "Expert argmax match", "b", "Expert alignment"),
        ("mean_kl_to_expert", "KL to expert", "c", "Distributional deviation"),
        ("mean_action_deviation_from_expert", "Mean action deviation", "d", "Action deviation"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.15), layout="constrained")

    for ax, (metric, ylabel, letter, title) in zip(axes, specs):
        _, post = values_for(rows, "Post-hoc Masked VOAC", metric)
        _, laad = values_for(rows, "LAADAN-AC", metric)
        for j, seed in enumerate(SEEDS):
            ax.plot([0, 1], [post[j], laad[j]], color="#B8B8B8", lw=1.25, zorder=1)
            ax.scatter(0, post[j], s=34, color=COLORS["posthoc"], edgecolor="black", linewidth=0.55, zorder=3)
            ax.scatter(1, laad[j], s=34, color=COLORS["laadan"], edgecolor="black", linewidth=0.55, zorder=3)
        ax.set_xticks([0, 1], ["Post-hoc\nmask", "LAADAN-\nAC"])
        ax.set_ylabel(ylabel)
        delta = paired["metrics"][metric]
        ax.text(
            0.5, 0.965,
            f"Δ = {float(delta['mean']):+.4f} ± {float(delta['ci95_half']):.4f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.8,
        )
        clean_axis(ax)
        panel_title(ax, letter, title)
    fig.savefig(out / "main/fig3_paired_training_vs_posthoc.png", dpi=600)
    plt.close(fig)


def fig_ablation(results_root: Path, out: Path):
    root = results_root / "icu_sepsis"
    summary = load_json(root / "aggregate/ablation_summary.json")
    rows = pd.read_csv(root / "aggregate/ablation_per_seed_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.3), layout="constrained")
    axes = axes.ravel()
    specs = [
        ("survival_rate", "Survival / return", 1.0, "a", "Return"),
        ("inadmissibility_rate", "Inadmissibility (%)", 100.0, "b", "Selected-action inadmissibility"),
        ("expert_argmax_match", "Expert argmax match", 1.0, "c", "Expert alignment"),
        ("mean_kl_to_expert", "KL to expert", 1.0, "d", "Distributional deviation"),
    ]

    x = np.arange(len(ABLATIONS))
    for ax, (metric, ylabel, scale, letter, title) in zip(axes, specs):
        means, cis, colors = [], [], []
        for method, _, key in ABLATIONS:
            mean, ci = summary_stat(summary, method, metric)
            means.append(mean * scale)
            cis.append(ci * scale)
            colors.append(COLORS[key])
        ax.bar(
            x, means, width=0.62, color=colors, edgecolor="black", linewidth=0.65,
            yerr=cis,
            error_kw={"ecolor": "black", "elinewidth": 0.85, "capsize": 2.6, "capthick": 0.85},
            zorder=2,
        )
        for i, (method, _, key) in enumerate(ABLATIONS):
            _, vals = values_for(rows, method, metric, scale)
            offsets = np.linspace(-0.12, 0.12, len(vals))
            ax.scatter(i + offsets, vals, s=20, facecolor="white", edgecolor="black", linewidth=0.6, zorder=5)
        ax.set_xticks(x, [m[1] for m in ABLATIONS])
        ax.set_ylabel(ylabel)
        clean_axis(ax)
        panel_title(ax, letter, title)

    fig.savefig(out / "main/fig4_full_component_ablation.png", dpi=600)
    plt.close(fig)


def fig_cross_source(results_root: Path, out: Path):
    icu_summary = load_json(results_root / "icu_sepsis/aggregate/main_summary.json")
    eicu_summary = load_json(results_root / "eicu_demo/aggregate/main_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.25), layout="constrained")
    specs = [
        ("survival_rate", "Survival / return", 1.0, "a", "Return"),
        ("inadmissibility_rate", "Inadmissibility (%)", 100.0, "b", "Selected-action inadmissibility"),
        ("expert_argmax_match", "Expert argmax match", 1.0, "c", "Expert alignment"),
    ]
    x = np.arange(len(METHODS))
    width = 0.34

    for ax, (metric, ylabel, scale, letter, title) in zip(axes, specs):
        for offset, (label, summary, color) in zip(
            [-width/2, width/2],
            [("ICU-Sepsis", icu_summary, COLORS["icu"]), ("eICU Demo", eicu_summary, COLORS["eicu"])],
        ):
            means, cis = [], []
            for method, _, _ in METHODS:
                mean, ci = summary_stat(summary, method, metric)
                means.append(mean * scale)
                cis.append(ci * scale)
            ax.bar(
                x + offset, means, width=width, color=color, edgecolor="black", linewidth=0.6,
                yerr=cis,
                error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2.2},
                label=label, zorder=2,
            )
        ax.set_xticks(x, [m[1] for m in METHODS])
        ax.set_ylabel(ylabel)
        if metric == "expert_argmax_match":
            ax.set_ylim(0, 1.08)
        clean_axis(ax)
        panel_title(ax, letter, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=2, frameon=False)
    fig.savefig(out / "appendix/figA1_cross_source_portability.png", dpi=600)
    plt.close(fig)


def fig_eicu(results_root: Path, out: Path):
    root = results_root / "eicu_demo"
    summary = load_json(root / "aggregate/main_summary.json")
    rows = pd.read_csv(root / "aggregate/per_seed_metrics.csv")

    fig, axes = plt.subplots(1, 4, figsize=(13.7, 3.25), layout="constrained")
    specs = [
        ("survival_rate", "Survival / return", 1.0, None, "a", "Return"),
        ("inadmissibility_rate", "Inadmissibility (%)", 100.0, None, "b", "Selected-action inadmissibility"),
        ("expert_argmax_match", "Expert argmax match", 1.0, (0, 1.08), "c", "Expert alignment"),
        ("mean_kl_to_expert", "KL to expert", 1.0, None, "d", "Distributional deviation"),
    ]
    for ax, (metric, ylabel, scale, ylim, letter, title) in zip(axes, specs):
        draw_bar_seed_panel(ax, summary, rows, METHODS, metric, ylabel, scale, ylim)
        panel_title(ax, letter, title)
    fig.savefig(out / "appendix/figA2_eicu_fixed_schedule_comparison.png", dpi=600)
    plt.close(fig)


def fig_policy_diagnostics(results_root: Path, out: Path):
    rows = pd.read_csv(results_root / "icu_sepsis/aggregate/per_seed_metrics.csv")
    post = rows.loc[rows["method"] == "Post-hoc Masked VOAC"].sort_values("seed")

    specs = [
        ("state_mask_intervention_rate", "Mask intervention (%)", "a", "Greedy action changed"),
        ("mean_inadmissible_probability_mass_before_mask", "Inadmissible mass (%)", "b", "Mean pre-mask probability mass"),
        ("max_inadmissible_probability_mass_before_mask", "Inadmissible mass (%)", "c", "Maximum pre-mask probability mass"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.05), layout="constrained")
    x = np.arange(len(post))
    for ax, (metric, ylabel, letter, title) in zip(axes, specs):
        vals = post[metric].to_numpy(float) * 100.0
        ax.plot(x, vals, color="#B6B6B6", linewidth=1.0, zorder=1)
        ax.scatter(x, vals, s=42, color=COLORS["voac"], edgecolor="black", linewidth=0.6, zorder=3)
        mean = vals.mean()
        ax.axhline(mean, color=COLORS["laadan"], linestyle="--", linewidth=1.15, zorder=2)
        ax.text(0.98, 0.96, f"mean = {mean:.2f}%", transform=ax.transAxes, ha="right", va="top", fontsize=8.7)
        ax.set_xticks(x, post["seed"].astype(int).astype(str))
        ax.set_xlabel("Training seed")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        clean_axis(ax)
        panel_title(ax, letter, title)

    fig.savefig(out / "appendix/figB1_policy_distribution_diagnostics.png", dpi=600)
    plt.close(fig)


def fig_safety(results_root: Path, out: Path):
    root = results_root / "icu_sepsis/safety_diagnostics"
    per_seed = pd.read_csv(root / "per_seed_diagnostics.csv").sort_values("seed")
    tables = root / "illustrative_seed/seed_42/tables"
    state = pd.read_csv(tables / "state_level_inadmissibility.csv")
    action = pd.read_csv(tables / "action_level_inadmissibility_counts.csv")
    q = pd.read_csv(tables / "representative_state_q_values.csv")
    traj = pd.read_csv(tables / "sample_trajectory_voac_vs_laadan.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), layout="constrained")


    ax = axes[0, 0]
    metrics = [
        ("voac_mean_state_inadmissibility", "VOAC", COLORS["voac"]),
        ("posthoc_mean_state_inadmissibility", "Post-hoc\nmask", COLORS["posthoc"]),
        ("laadan_mean_state_inadmissibility", "LAADAN-AC", COLORS["laadan"]),
    ]
    for i, (metric, label, color) in enumerate(metrics):
        vals = per_seed[metric].to_numpy(float) * 100.0
        offs = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(i + offs, vals, s=34, color=color, edgecolor="black", linewidth=0.55, zorder=3)
        ax.hlines(vals.mean(), i-0.18, i+0.18, color="black", lw=1.3, zorder=4)
    ax.set_xticks(range(3), [m[1] for m in metrics])
    ax.set_ylabel("States with inadmissible greedy action (%)")
    clean_axis(ax)
    panel_title(ax, "a", "State-level failure rate")

    ax = axes[0, 1]
    voac = action.loc[action["model"] == "Vanilla Offline Actor-Critic"].copy()
    voac = voac.loc[voac["inadmissible_selected_count"] > 0]
    ax.bar(voac["action"].astype(int), voac["inadmissible_selected_count"], color=COLORS["voac"], edgecolor="black", linewidth=0.55)
    ax.set_xlabel("Treatment-action bin")
    ax.set_ylabel("Inadmissible selections (states)")
    clean_axis(ax)
    panel_title(ax, "b", "Action-level VOAC failures (seed 42)")

    ax = axes[1, 0]
    state_id = int(q["state_id"].iloc[0])
    groups, positions, labels, colors = [], [], [], []
    pos = 1
    model_map = [
        ("Vanilla Offline Actor-Critic", "VOAC", COLORS["voac"]),
        ("Conservative Q-Learning", "CQL", COLORS["cql"]),
        ("LAADAN-AC", "LAADAN", COLORS["laadan"]),
    ]
    for model, short, color in model_map:
        sub = q.loc[(q["model"] == model) & (q["state_id"] == state_id)]
        for adm, suffix in [(1, "adm"), (0, "inad")]:
            vals = sub.loc[sub["admissible"] == adm, "q_value"].to_numpy(float)
            if len(vals):
                groups.append(vals); positions.append(pos); labels.append(f"{short}\n{suffix}"); colors.append(color); pos += 1
        pos += 0.45
    bp = ax.boxplot(groups, positions=positions, widths=0.56, patch_artist=True, showfliers=False, medianprops={"color":"black", "linewidth":1.0})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.72); patch.set_edgecolor("black"); patch.set_linewidth(0.6)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Q value")
    clean_axis(ax)
    panel_title(ax, "c", f"Q values at representative state {state_id}")

    ax = axes[1, 1]
    for model, label, color in [
        ("Vanilla Offline Actor-Critic", "VOAC", COLORS["voac"]),
        ("LAADAN-AC", "LAADAN-AC", COLORS["laadan"]),
    ]:
        s = traj.loc[traj["model"] == model].sort_values("step")
        ax.plot(s["step"], s["action"], marker="o", markersize=4.0, linewidth=1.7, color=color, label=label)
        if model == "Vanilla Offline Actor-Critic":
            bad = s.loc[s["admissible"] == 0]
            ax.scatter(bad["step"], bad["action"], marker="x", s=60, linewidth=1.4, color="black", label="VOAC inadmissible", zorder=5)
    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("Treatment-action bin")
    ax.legend(frameon=False, loc="best")
    clean_axis(ax)
    panel_title(ax, "d", "Matched-initial-state trajectory (seed 42)")

    fig.savefig(out / "appendix/figC1_safety_failure_diagnostics.png", dpi=600)
    plt.close(fig)


def make_animation(results_root: Path, out: Path):
    traj_path = results_root / "icu_sepsis/safety_diagnostics/illustrative_seed/seed_42/tables/sample_trajectory_voac_vs_laadan.csv"
    traj = pd.read_csv(traj_path)
    v = traj.loc[traj["model"] == "Vanilla Offline Actor-Critic"].sort_values("step")
    l = traj.loc[traj["model"] == "LAADAN-AC"].sort_values("step")
    max_step = int(max(v["step"].max(), l["step"].max()))

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    clean_axis(ax)
    ax.set_xlim(-0.25, max_step + 0.4)
    ax.set_ylim(-1, 25)
    ax.set_xlabel("Trajectory step")
    ax.set_ylabel("Treatment-action bin")
    ax.set_title("Matched-initial-state policy trajectory (illustrative seed 42)")

    line_v, = ax.plot([], [], color=COLORS["voac"], lw=2, marker="o", label="VOAC")
    line_l, = ax.plot([], [], color=COLORS["laadan"], lw=2, marker="o", label="LAADAN-AC")
    bad_scatter = ax.scatter([], [], marker="x", s=80, color="black", linewidth=1.5, label="VOAC inadmissible")
    ax.legend(frameon=False, loc="upper left")

    def update(frame):
        vv = v.loc[v["step"] <= frame]
        ll = l.loc[l["step"] <= frame]
        line_v.set_data(vv["step"], vv["action"])
        line_l.set_data(ll["step"], ll["action"])
        bad = vv.loc[vv["admissible"] == 0]
        offsets = np.column_stack([bad["step"].to_numpy(), bad["action"].to_numpy()]) if len(bad) else np.empty((0,2))
        bad_scatter.set_offsets(offsets)
        return line_v, line_l, bad_scatter

    anim = FuncAnimation(fig, update, frames=range(max_step + 1), interval=650, blit=False, repeat=True)
    anim.save(out / "supplementary/trajectory_animation.gif", writer=PillowWriter(fps=2), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/fixed_schedule_2026")
    parser.add_argument("--out-dir", default="figures/camera_ready")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out = Path(args.out_dir).resolve()
    required = [
        results_root / "icu_sepsis/aggregate/main_summary.json",
        results_root / "icu_sepsis/aggregate/ablation_summary.json",
        results_root / "eicu_demo/aggregate/main_summary.json",
        results_root / "icu_sepsis/safety_diagnostics/per_seed_diagnostics.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing required frozen result files:\n" + "\n".join(missing))

    configure_matplotlib()
    ensure_dirs(out)

    fig_main_icu(results_root, out)
    fig_paired(results_root, out)
    fig_ablation(results_root, out)
    fig_cross_source(results_root, out)
    fig_eicu(results_root, out)
    fig_policy_diagnostics(results_root, out)
    fig_safety(results_root, out)
    if args.animate:
        make_animation(results_root, out)

    outputs = sorted(p for p in out.rglob("*") if p.is_file())
    print("PASS: camera-ready figures generated from frozen results only.")
    for p in outputs:
        print(" ", p)


if __name__ == "__main__":
    main()
