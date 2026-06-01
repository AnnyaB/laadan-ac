

# pick_final_models.py

# This script scans the per-seed training results for each model family,
# picks the single best seed for each agent, copies that seed's key output files
# into a clean "final_models" folder, and writes a CSV summary of the selected
# checkpoints.

# Libraries used in this script:

# to write the final selection table as a CSV file.
import csv

# to read metrics.json files produced during training.
import json

# to create and manage file/folder paths.
import os

# shutil is used to copy the chosen final model files into the final_models
# directory while preserving metadata where possible.
import shutil

# Root results directory used throughout the project.
# All per-model training outputs are expected to live under this folder.
RESULTS_DIR = "results"

# AGENTS maps each folder name used in training to a human-readable model name.
# The keys must match the folder names created by the training pipeline.
AGENTS = {
    "bc": "Behavior Cloning",
    "cql": "Conservative Q-Learning",
    "voac": "Vanilla Offline Actor-Critic",
    "laadan_ac": "LAADAN-AC",
}

# Building the output folder path where the final chosen models will be stored.
# This creates a clean folder such as:
# results/final_models/
final_root = os.path.join(RESULTS_DIR, "final_models")

# Creating the final output folder if it does not already exist.
# exist_ok=True prevents an error if the folder is already there.
os.makedirs(final_root, exist_ok=True)

# rows will store one summary dictionary per selected model family.
# These dictionaries will later be written into final_model_selection.csv.
rows = []

# Looping through each agent family defined above.
# agent_dir is the folder name (e.g. "bc"), and agent_name is the readable name.
for agent_dir, agent_name in AGENTS.items():

    # Building the source directory for this agent family.
    # Example:
    # results/bc
    source_dir = os.path.join(RESULTS_DIR, agent_dir)

    # If that source directory does not exist, skip this agent family.
    # This makes the script more robust when some models have not been trained.
    if not os.path.isdir(source_dir):
        continue

    # best_seed will store the name of the currently best-performing seed folder,
    # for example "seed_42".
    best_seed = None

    # best_metrics will store the metrics dictionary corresponding to best_seed.
    best_metrics = None

    # Looking through every folder inside the source directory in sorted order.
    # Sorting makes the selection process deterministic when browsing folders.
    for folder in sorted(os.listdir(source_dir)):

        # Only consider folders that follow the expected seed naming scheme.
        # For example: seed_42, seed_43, ...
        if not folder.startswith("seed_"):
            continue

        # Building the path to the metrics file for this seed.
        metrics_path = os.path.join(source_dir, folder, "metrics.json")

        # Building the path to the model checkpoint for this seed.
        # This path is not used for selection itself here, but it is part of the
        # expected output structure and makes the code easier to understand.
        model_path = os.path.join(source_dir, folder, "model.pt")

        # Building the path to the history CSV for this seed.
        # Again, not used directly in the ranking rule, but this is one of the
        # files copied later if this seed wins.
        history_path = os.path.join(source_dir, folder, "history.csv")

        # If metrics.json does not exist for this seed, skip it.
        # The script chooses winners only from seeds that actually produced metrics.
        if not os.path.exists(metrics_path):
            continue

        # Opening the metrics file and load it into a Python dictionary.
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Reading the main selection metric: survival_rate.
        # If missing, use a very low fallback so the seed is never preferred.
        survival = float(metrics.get("survival_rate", -1e9))

        # Reading the tie-break metric: inadmissibility_rate.
        # If missing, use a very high fallback so the seed is never preferred.
        inad = float(metrics.get("inadmissibility_rate", 1e9))

        # If this is the first valid seed seen for the current agent family,
        # initialise best_seed and best_metrics with it.
        if best_metrics is None:
            best_seed = folder
            best_metrics = metrics
        else:
            # Extracting the current best survival value seen so far.
            best_survival = float(best_metrics.get("survival_rate", -1e9))

            # Extracting the current best inadmissibility value seen so far.
            best_inad = float(best_metrics.get("inadmissibility_rate", 1e9))

            # Prefer higher survival.
            # If survival is tied exactly, prefer lower inadmissibility.
            if (survival > best_survival) or (survival == best_survival and inad < best_inad):
                best_seed = folder
                best_metrics = metrics

    # If no valid seed was found for this agent family, skip it.
    if best_seed is None:
        continue

    # Building the full path to the chosen source seed directory.
    # Example:
    # results/bc/seed_42
    src_seed_dir = os.path.join(source_dir, best_seed)

    # Building the destination directory for the selected final model of this agent.
    # Example:
    # results/final_models/bc
    dst_agent_dir = os.path.join(final_root, agent_dir)

    # Creating the destination directory if needed.
    os.makedirs(dst_agent_dir, exist_ok=True)

    # Copying the core files from the chosen seed folder into the final model folder.   
    # These are the main files needed for testing and reproducibility.
    for filename in ["model.pt", "metrics.json", "history.csv"]:

        # Source path for the current file.
        src = os.path.join(src_seed_dir, filename)

        # Destination path for the current file.
        dst = os.path.join(dst_agent_dir, filename)

        # Copying the file only if it exists.
        # copy2 preserves file metadata better than a plain copy.
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Appending one row describing the selected best model for this agent family.
    # This row will later be written into the CSV selection table.
    rows.append({
        "agent_dir": agent_dir,
        "agent_name": agent_name,
        "best_seed": best_seed,
        "survival_rate": best_metrics.get("survival_rate", ""),
        "inadmissibility_rate": best_metrics.get("inadmissibility_rate", ""),
        "avg_return": best_metrics.get("avg_return", ""),
    })

# Building the full path to the final CSV selection summary.
csv_path = os.path.join(final_root, "final_model_selection.csv")

# Opening the CSV file for writing.
with open(csv_path, "w", newline="", encoding="utf-8") as f:

    # Creating a CSV DictWriter with a fixed column order.
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "agent_dir",
            "agent_name",
            "best_seed",
            "survival_rate",
            "inadmissibility_rate",
            "avg_return",
        ],
    )

    # Writing the CSV header row.
    writer.writeheader()

    # Writing all collected selection rows.
    writer.writerows(rows)

# Printing a short success message showing where the final copied models were saved.
print("Saved final selected models to:", final_root)

# Printing the path of the CSV selection table for quick reference.
print("Selection table:", csv_path)


