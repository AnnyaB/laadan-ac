import csv

import json

import os

import shutil

RESULTS_DIR = "results"

AGENTS = {
    "bc": "Behavior Cloning",
    "cql": "Conservative Q-Learning",
    "voac": "Vanilla Offline Actor-Critic",
    "laadan_ac": "LAADAN-AC",
}

final_root = os.path.join(RESULTS_DIR, "final_models")

os.makedirs(final_root, exist_ok=True)

rows = []

for agent_dir, agent_name in AGENTS.items():

    source_dir = os.path.join(RESULTS_DIR, agent_dir)

    if not os.path.isdir(source_dir):
        continue

    best_seed = None

    best_metrics = None

    for folder in sorted(os.listdir(source_dir)):

        if not folder.startswith("seed_"):
            continue

        metrics_path = os.path.join(source_dir, folder, "metrics.json")

        model_path = os.path.join(source_dir, folder, "model.pt")

        history_path = os.path.join(source_dir, folder, "history.csv")

        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        survival = float(metrics.get("survival_rate", -1e9))

        inad = float(metrics.get("inadmissibility_rate", 1e9))

        if best_metrics is None:
            best_seed = folder
            best_metrics = metrics
        else:

            best_survival = float(best_metrics.get("survival_rate", -1e9))

            best_inad = float(best_metrics.get("inadmissibility_rate", 1e9))

            if (survival > best_survival) or (survival == best_survival and inad < best_inad):
                best_seed = folder
                best_metrics = metrics

    if best_seed is None:
        continue

    src_seed_dir = os.path.join(source_dir, best_seed)

    dst_agent_dir = os.path.join(final_root, agent_dir)

    os.makedirs(dst_agent_dir, exist_ok=True)

    for filename in ["model.pt", "metrics.json", "history.csv"]:

        src = os.path.join(src_seed_dir, filename)

        dst = os.path.join(dst_agent_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)

    rows.append({
        "agent_dir": agent_dir,
        "agent_name": agent_name,
        "best_seed": best_seed,
        "survival_rate": best_metrics.get("survival_rate", ""),
        "inadmissibility_rate": best_metrics.get("inadmissibility_rate", ""),
        "avg_return": best_metrics.get("avg_return", ""),
    })

csv_path = os.path.join(final_root, "final_model_selection.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:

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

    writer.writeheader()

    writer.writerows(rows)

print("Saved final selected models to:", final_root)

print("Selection table:", csv_path)
