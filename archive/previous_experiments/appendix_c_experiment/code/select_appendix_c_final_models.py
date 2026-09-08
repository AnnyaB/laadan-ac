import os
import json
import shutil
import csv
import math

ROOT = "results_icu_sepsis_LAADAN_AC_PD"
OUT = os.path.join(ROOT, "final_models")

AGENTS = [
    ("bc", "Behavior Cloning", False),
    ("cql", "Conservative Q-Learning", False),
    ("voac", "Vanilla Offline Actor-Critic", False),
    ("laadan_ac", "LAADAN-AC", False),
    ("laadan_ac_pd_nomask", "LAADAN-AC-PD", True),
]

os.makedirs(OUT, exist_ok=True)

rows = []

# model selection
def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def better_standard(metrics, best_metrics):
    if best_metrics is None:
        return True
    return float(metrics.get("survival_rate", -math.inf)) > float(best_metrics.get("survival_rate", -math.inf))

def better_constrained(metrics, best_metrics, budget):
    if best_metrics is None:
        return True

    c = float(metrics.get("inadmissibility_rate", math.inf))
    b = float(best_metrics.get("inadmissibility_rate", math.inf))

    r = float(metrics.get("survival_rate", metrics.get("avg_return", -math.inf)))
    br = float(best_metrics.get("survival_rate", best_metrics.get("avg_return", -math.inf)))

    feasible = c <= budget + 1e-8
    best_feasible = b <= budget + 1e-8

    if feasible and not best_feasible:
        return True
    if feasible and best_feasible:
        return r > br
    if not feasible and not best_feasible:
        if c < b - 1e-8:
            return True
        if abs(c - b) <= 1e-8:
            return r > br

    return False

for folder, display_name, constrained in AGENTS:
    src_root = os.path.join(ROOT, folder)

    if not os.path.isdir(src_root):
        print("Skipping missing folder:", src_root)
        continue

    best_seed_dir = None
    best_metrics = None
    best_seed = None

    for seed_name in sorted(os.listdir(src_root)):
        seed_dir = os.path.join(src_root, seed_name)
        metrics_path = os.path.join(seed_dir, "metrics.json")

        if not os.path.isdir(seed_dir) or not os.path.exists(metrics_path):
            continue

        metrics = load_metrics(metrics_path)
        budget = float(metrics.get("cost_budget", 0.005))

        if constrained:
            is_better = better_constrained(metrics, best_metrics, budget)
        else:
            is_better = better_standard(metrics, best_metrics)

        if is_better:
            best_seed_dir = seed_dir
            best_metrics = metrics
            best_seed = seed_name

    if best_seed_dir is None:
        print("No valid seed found for:", folder)
        continue

    dst_dir = os.path.join(OUT, folder)
    os.makedirs(dst_dir, exist_ok=True)

    for filename in ["model.pt", "history.csv", "metrics.json"]:
        src_file = os.path.join(best_seed_dir, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dst_dir, filename))

    row = {
        "folder": folder,
        "name": display_name,
        "selected_seed": best_seed,
        "survival_rate": best_metrics.get("survival_rate"),
        "inadmissibility_rate": best_metrics.get("inadmissibility_rate"),
        "avg_return": best_metrics.get("avg_return"),
        "cost_budget": best_metrics.get("cost_budget", ""),
        "lagrange": best_metrics.get("lagrange", ""),
        "selection_rule": "constrained" if constrained else "highest_survival",
    }
    rows.append(row)

selection_path = os.path.join(OUT, "final_model_selection.csv")

with open(selection_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Saved final selected models to:", OUT)
print("Selection table:", selection_path)
