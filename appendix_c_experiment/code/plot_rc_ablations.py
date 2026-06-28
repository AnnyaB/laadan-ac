
# plot_rc_ablations.py
#
# Creates summary figures for LAADAN-AC-RC ablation results.
#
# Expected input:
# <results-dir>/rc_ablation_summary.csv
# <results-dir>/final_ablation_models/<variant>/history.csv
#
# Output:
# <results-dir>/figures/

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


VARIANT_ORDER = [
    "full_rc",
    "no_lambda_updates",
    "no_barrier",
    "no_certified_nudging",
    "no_support_lambda",
    "no_cost_ucb",
    "lambda_only_no_mask",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clean_summary(df):
    df = df.copy()

    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str)
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
        df = df.sort_values("variant")

    numeric_cols = [
        "survival_rate",
        "inadmissibility_rate",
        "soft_inadmissibility_rate",
        "avg_return",
        "soft_avg_return",
        "expert_argmax_match",
        "mean_kl_to_expert",
        "policy_entropy",
        "soft_policy_entropy",
        "lambda_inad",
        "lambda_support",
        "lambda_smooth",
        "lambda_uncertainty",
        "barrier_weight",
        "support_weight",
        "rho_cost",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_value_labels(ax, values):
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue

        if abs(float(value)) < 1e-4 and float(value) != 0.0:
            label = f"{float(value):.1e}"
        else:
            label = f"{float(value):.4f}"

        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            xytext=(0, 3),
            textcoords="offset points",
        )


def save_bar(df, y_col, title, ylabel, output_path):
    plot_df = df.dropna(subset=[y_col]).copy()

    labels = plot_df["variant"].astype(str).tolist()
    values = plot_df[y_col].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Ablation variant")
    ax.tick_params(axis="x", rotation=30)

    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    add_value_labels(ax, values)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_scatter(df, output_path):
    plot_df = df.dropna(subset=["survival_rate", "inadmissibility_rate"]).copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        plot_df["inadmissibility_rate"].astype(float),
        plot_df["survival_rate"].astype(float),
    )

    for _, row in plot_df.iterrows():
        ax.annotate(
            str(row["variant"]),
            (float(row["inadmissibility_rate"]), float(row["survival_rate"])),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title("Return-safety comparison across LAADAN-AC-RC ablations")
    ax.set_xlabel("Greedy inadmissibility rate")
    ax.set_ylabel("Survival / return")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_history_curves(results_dir, metric_col, title, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_any = False

    for variant in VARIANT_ORDER:
        history_path = os.path.join(
            results_dir,
            "final_ablation_models",
            variant,
            "history.csv",
        )

        if not os.path.exists(history_path):
            continue

        hist = pd.read_csv(history_path)

        if "epoch" not in hist.columns:
            continue

        if metric_col not in hist.columns:
            continue

        hist[metric_col] = pd.to_numeric(hist[metric_col], errors="coerce")
        hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")

        hist = hist.dropna(subset=["epoch", metric_col])

        if hist.empty:
            continue

        ax.plot(hist["epoch"], hist[metric_col], label=variant)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return True


def main():
    parser = argparse.ArgumentParser(description="Plot LAADAN-AC-RC ablation figures.")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    summary_path = os.path.join(args.results_dir, "rc_ablation_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError("Missing summary file: " + summary_path)

    figures_dir = os.path.join(args.results_dir, "figures")
    ensure_dir(figures_dir)

    df = pd.read_csv(summary_path)
    df = clean_summary(df)

    save_bar(
        df,
        y_col="survival_rate",
        title="LAADAN-AC-RC ablation survival comparison",
        ylabel="Survival / return",
        output_path=os.path.join(figures_dir, "ablation_survival_comparison.png"),
    )

    save_bar(
        df,
        y_col="inadmissibility_rate",
        title="LAADAN-AC-RC greedy inadmissibility comparison",
        ylabel="Greedy inadmissibility rate",
        output_path=os.path.join(figures_dir, "ablation_greedy_inadmissibility_comparison.png"),
    )

    save_bar(
        df,
        y_col="soft_inadmissibility_rate",
        title="LAADAN-AC-RC soft-policy inadmissibility comparison",
        ylabel="Soft-policy inadmissibility rate",
        output_path=os.path.join(figures_dir, "ablation_soft_inadmissibility_comparison.png"),
    )

    if "lambda_inad" in df.columns:
        save_bar(
            df,
            y_col="lambda_inad",
            title="LAADAN-AC-RC final inadmissibility multiplier comparison",
            ylabel="Final lambda_inad",
            output_path=os.path.join(figures_dir, "ablation_lambda_inad_comparison.png"),
        )

    if "lambda_support" in df.columns:
        save_bar(
            df,
            y_col="lambda_support",
            title="LAADAN-AC-RC final support multiplier comparison",
            ylabel="Final lambda_support",
            output_path=os.path.join(figures_dir, "ablation_lambda_support_comparison.png"),
        )

    save_scatter(
        df,
        output_path=os.path.join(figures_dir, "ablation_return_safety_scatter.png"),
    )

    save_history_curves(
        args.results_dir,
        metric_col="survival_rate",
        title="Selected ablation checkpoint survival during training",
        ylabel="Survival / return",
        output_path=os.path.join(figures_dir, "selected_ablation_survival_curves.png"),
    )

    save_history_curves(
        args.results_dir,
        metric_col="inadmissibility_rate",
        title="Selected ablation checkpoint greedy inadmissibility during training",
        ylabel="Greedy inadmissibility rate",
        output_path=os.path.join(figures_dir, "selected_ablation_greedy_inadmissibility_curves.png"),
    )

    save_history_curves(
        args.results_dir,
        metric_col="soft_inadmissibility_rate",
        title="Selected ablation checkpoint soft-policy inadmissibility during training",
        ylabel="Soft-policy inadmissibility rate",
        output_path=os.path.join(figures_dir, "selected_ablation_soft_inadmissibility_curves.png"),
    )

    print("Saved ablation figures to:", figures_dir)


if __name__ == "__main__":
    main()
