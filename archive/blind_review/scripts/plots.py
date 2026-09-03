
import json


import os

import matplotlib.pyplot as plt

import numpy as np



PALETTE = {
    "Behavior Cloning": "#6baed6",
    "Conservative Q-Learning": "#08519c",
    "Vanilla Offline Actor-Critic": "#9ecae1",
    "LAADAN-AC": "#2171b5",
    "Random Policy": "#bdd7e7",
    "Expert Policy": "#4292c6",
    "Optimal Policy": "#08306b",
}


def ensure_dir(path):

    if not os.path.exists(path):

        os.makedirs(path)


def _get_color(name):

    return PALETTE.get(name, "#3182bd")


def _metric_label(metric_name):

    labels = {
        "survival_rate": "Survival rate",
        "inadmissibility_rate": "Inadmissible action rate",
        "expert_argmax_match": "Match with expert argmax",
    }

    return labels.get(metric_name, metric_name.replace("_", " ").title())


def _save_current(output_path):


    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    plt.close()


def plot_overall_performance(summary, output_path):

    model_names = list(summary.keys())

    metric_names = [
        "survival_rate",
        "inadmissibility_rate",
        "expert_argmax_match",
    ]

    # Creating x positions for the model groups.
    x = np.arange(len(model_names))

    # Width of each bar within a group.
    width = 0.22

    # Start a new figure with a report-friendly size.
    plt.figure(figsize=(11, 6), constrained_layout=True)

    # Looping over the selected metrics so each one gets its own bar offset.
    for metric_index, metric_name in enumerate(metric_names):
        # Collecting the mean metric value for each model.
        means = [summary[name][metric_name]["mean"] for name in model_names]

        # Collecting the 95% confidence-interval half-width for each model.
        cis = [summary[name][metric_name].get("ci95_half", 0.0) for name in model_names]

        # Computing the horizontal offset for this metric's bars inside each model group.
        offset = (metric_index - 1.0) * width

        # Drawing the bars with error bars.
        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=cis,
            capsize=4,
            label=_metric_label(metric_name),
            alpha=0.95,
        )

    # Setting the x-axis tick locations and model names.
    plt.xticks(x, model_names, rotation=12)

    # Labeling the y-axis generically because the three metrics all lie in the 0-1 range.
    plt.ylabel("Score")

    # Fixing the y-axis range so comparisons are visually stable and interpretable.
    plt.ylim(0.0, 1.05)

    # Adding a human-readable title.
    plt.title("Overall policy performance")

    # Adding light horizontal grid lines to improve readability.
    plt.grid(True, axis="y", alpha=0.25)

    # Placing the legend above the plot in a compact format.
    plt.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    # Saving and closing the figure.
    _save_current(output_path)


def plot_benchmark_context(reference_metrics, summary, output_path):

    """
    Comparing learned policies against Random, Expert, and Optimal references.

    """
    # Reading the names of the benchmark reference policies.
    reference_names = list(reference_metrics.keys())

    # Reading the names of the learned models.
    model_names = list(summary.keys())

    # Concatenating both groups so they appear in one chart.
    names = reference_names + model_names

    # Preparing lists for bar heights, error bars, and colors.
    values = []
    errors = []
    colors = []

    # Adding the benchmark reference values first.
    for name in reference_names:
        # Referencing metrics are exact single values, not multi-seed summaries.
        values.append(float(reference_metrics[name]["survival_rate"]))

        # No confidence interval is shown for these references.
        errors.append(0.0)

        # Using the predefined consistent color.
        colors.append(_get_color(name))

    # Adding the learned-model mean survival values next.
    for name in model_names:
        # Appending the mean survival across seeds.
        values.append(float(summary[name]["survival_rate"]["mean"]))

        # Appending the corresponding 95% confidence interval half-width.
        errors.append(float(summary[name]["survival_rate"].get("ci95_half", 0.0)))

        # Appending the model color.
        colors.append(_get_color(name))

    # Building the x positions for all bars.
    x = np.arange(len(names))

    # Starting a new figure.
    plt.figure(figsize=(11, 6), constrained_layout=True)

    # Drawing the comparison bar chart.
    plt.bar(x, values, yerr=errors, capsize=4, color=colors)

    # Placing labels on the x-axis.
    plt.xticks(x, names, rotation=15)

    # Labeling the y-axis with the metric being compared.
    plt.ylabel("Survival rate")

    # Making the upper y-limit slightly higher than the tallest bar.
    plt.ylim(0.0, max(1.0, float(np.max(values)) + 0.08))

    # Adding a clear title.
    plt.title("Learned policies compared with benchmark references")

    # Adding a light y-grid for readability.
    plt.grid(True, axis="y", alpha=0.25)

    # Saving and closing.
    _save_current(output_path)


def _extract_finite_curve(payload, metric_name):

    """
    Extracting a valid mean CI curve for one metric from an aggregated history.
    """

    # If the requested metric is not present at all, return None.
    if metric_name not in payload:
        return None

    # Reading epoch numbers as floats for plotting.
    epochs = np.asarray(payload.get("epoch", []), dtype=float)

    # Reading the mean values for the requested metric.
    mean_values = np.asarray(payload[metric_name].get("mean", []), dtype=float)

    # Reading the 95% CI half-width values for the requested metric.
    ci_values = np.asarray(payload[metric_name].get("ci95_half", []), dtype=float)

    # If either epochs or mean values are empty, there is nothing to plot.
    if epochs.size == 0 or mean_values.size == 0:
        return None

    # Finding which entries are finite numeric values.
    finite = np.isfinite(mean_values)

    # If no finite values exist, return None.
    if not np.any(finite):
        return None

    # Returning only the finite portions of the curve.
    return epochs[finite], mean_values[finite], ci_values[finite]


def plot_training_metric(aggregated_histories, metric_name, output_path, title, ylabel):

    """
    Plotting a clean mean 95% CI training curve using only valid evaluation epochs.

    This is used for:
    - survival during training
    - inadmissible action rate during training
    """
    # Starting a new figure.
    plt.figure(figsize=(10, 6), constrained_layout=True)

    # Tracking whether at least one model produced a valid curve.
    plotted_anything = False

    # Looping through each model family in the aggregated histories.
    for model_name, payload in aggregated_histories.items():
        # Extracting the valid finite portion of the requested metric.
        curve = _extract_finite_curve(payload, metric_name)

        # If the curve is missing, skip this model.
        if curve is None:
            continue

        # Unpacking epochs, means, and confidence intervals.
        epochs, mean_values, ci_values = curve

        # Getting the model's consistent line color.
        color = _get_color(model_name)

        # Plotting the mean curve.
        plt.plot(epochs, mean_values, linewidth=2.5, label=model_name, color=color)

        # Plotting the confidence interval as a shaded band.
        plt.fill_between(epochs, mean_values - ci_values, mean_values + ci_values, alpha=0.18, color=color)

        # Marking that something was successfully plotted.
        plotted_anything = True

    # If nothing valid was plotted, close the empty figure and stop.
    if not plotted_anything:
        plt.close()
        return

    # Labeling the x-axis.
    plt.xlabel("Epoch")

    # Labeling the y-axis according to the metric being plotted.
    plt.ylabel(ylabel)

    # Adding the provided title.
    plt.title(title)

    # Adding light grid lines.
    plt.grid(True, alpha=0.25)

    # Adding a simple legend.
    plt.legend(frameon=False, loc="best")

    # Saving and closing.
    _save_current(output_path)


def plot_policy_heatmaps(policy_snapshots, selected_state_ids, output_path):

    """
    Heatmap of action preferences across representative states.

    """
    # Reading the names of all policies to display.
    names = list(policy_snapshots.keys())

    # Counting how many rows the subplot grid needs.
    num_rows = len(names)

    # Building a vertical stack of heatmaps, one per policy.
    fig, axes = plt.subplots(num_rows, 1, figsize=(13, 2.2 * num_rows), squeeze=False, constrained_layout=True)

    # Placeholder for the most recent heatmap image so its colorbar can be shared.
    image = None

    # Looping through each policy.
    for row_index, name in enumerate(names):
        # Converting the policy array to NumPy.
        policy = np.asarray(policy_snapshots[name], dtype=float)

        # Keeping only the selected representative states.
        subset = policy[selected_state_ids, :]

        # Selecting the axis for this row.
        axis = axes[row_index, 0]

        # Drawing the heatmap.
        image = axis.imshow(subset, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)

        # Setting subplot title to the policy name.
        axis.set_title(name, fontsize=11)

        # Labeling the y-axis as states.
        axis.set_ylabel("State")

        # Putting one tick per selected state.
        axis.set_yticks(np.arange(len(selected_state_ids)))

        # Showing the actual selected state IDs as labels.
        axis.set_yticklabels(selected_state_ids, fontsize=8)

        # Putting one tick per action column.
        axis.set_xticks(np.arange(subset.shape[1]))

        # Showing action-bin numbers along the x-axis.
        axis.set_xticklabels(np.arange(subset.shape[1]), fontsize=7)

    # Labeling the x-axis only on the final subplot.
    axes[-1, 0].set_xlabel("Action bin")

    # Adding a shared colorbar for action probabilities.
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.96)

    # Labeling the colorbar.
    cbar.set_label("Action probability")

    # Adding an overall figure title.
    fig.suptitle("Action preferences across representative states", fontsize=13)

    # Saving the figure.
    fig.savefig(output_path, dpi=600, bbox_inches="tight")

    # Closing it afterward.
    plt.close(fig)


def plot_hyperparameter_study(study_payload, output_dir):

    """
    Drawing sensitivity plots for LAADAN-AC.

    """
    # If the study payload is empty, there is nothing to plot.
    if not study_payload:
        return

    # Making sure the appendix directory exists.
    ensure_dir(output_dir)

    # Looping over each studied hyperparameter.
    for study_name, entries in study_payload.items():
        # Collecting the x-axis labels, which are the tried parameter values.
        x_labels = list(entries.keys())

        # Creating numeric x positions for the plotted points.
        x = np.arange(len(x_labels))

        # Collecting the survival results for each tested value.
        survival = [entries[key]["survival_rate"] for key in x_labels]

        # Collecting the inadmissibility results for each tested value.
        inadmissibility = [entries[key]["inadmissibility_rate"] for key in x_labels]

        # Creating a new figure with one left axis and one right axis.
        fig, axis_left = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

        # Plotting survival on the left axis.
        axis_left.plot(x, survival, marker="o", linewidth=2.2, color="#08519c")

        axis_left.set_ylabel("Survival rate")

        axis_left.set_xlabel(study_name.replace("_", " ").title())

        axis_left.set_xticks(x)

        axis_left.set_xticklabels(x_labels)

        axis_left.grid(True, alpha=0.25)

        axis_right = axis_left.twinx()

        axis_right.plot(x, inadmissibility, marker="s", linewidth=2.0, color="#6baed6")

        axis_right.set_ylabel("Inadmissible action rate")

        plt.title("Sensitivity of LAADAN-AC to " + study_name.replace("_", " "))

        fig.savefig(os.path.join(output_dir, study_name + ".png"), dpi=600, bbox_inches="tight")

        plt.close(fig)


def build_all_plots(results_dir):


    summary_path = os.path.join(results_dir, "summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError("summary.json was not found. Run the experiments first.")

    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    figures_dir = os.path.join(results_dir, "figures")

    appendix_dir = os.path.join(figures_dir, "appendix")

    ensure_dir(figures_dir)
    ensure_dir(appendix_dir)

    plot_overall_performance(
        payload["aggregate_metrics"],
        os.path.join(figures_dir, "overall_policy_performance.png"),
    )

    plot_benchmark_context(
        payload.get("benchmark_reference_metrics", {}),
        payload["aggregate_metrics"],
        os.path.join(figures_dir, "benchmark_reference_comparison.png"),
    )

    plot_training_metric(
        payload["aggregated_histories"],
        "survival_rate",
        os.path.join(figures_dir, "survival_during_training.png"),
        title="Survival during training",
        ylabel="Survival rate",
    )

    plot_training_metric(
        payload["aggregated_histories"],
        "inadmissibility_rate",
        os.path.join(figures_dir, "inadmissible_actions_during_training.png"),
        title="Inadmissible action rate during training",
        ylabel="Inadmissible action rate",
    )

    if "policy_snapshots" in payload and "selected_state_ids" in payload:
        plot_policy_heatmaps(
            payload["policy_snapshots"],
            payload["selected_state_ids"],
            os.path.join(figures_dir, "action_preferences_heatmap.png"),
        )

    plot_hyperparameter_study(
        payload.get("laadan_hyperparameter_study", {}),
        appendix_dir,
    )

