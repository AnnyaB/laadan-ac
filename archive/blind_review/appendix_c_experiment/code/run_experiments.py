# training icu-sepsis, eicu for appendix C experiments

# Libraries used in this file:

# argparse is used to read command-line arguments such as data path, device,
# result directory, and mode.
import argparse

# to deep-copy the default configuration and to save summaries.
import json

# for path construction and directory handling.
import os

# used here mainly for evenly spaced state selection for heatmaps.
import numpy as np

# used for device checks such as CUDA availability.
import torch

# Importing the benchmark environment/loader used by all models.
from benchmark import ICUSepsisOfflineBenchmark

# Importing the plotting entry function that converts summary.json into figures.
from plots import build_all_plots

# Importing all required training and aggregation helpers.
from trainers import (
    aggregate_histories,         # aggregate metric curves across seeds
    aggregate_seed_metrics,      # aggregate final per-seed metrics
    best_policy_from_runs,       # select the best seed's policy
    ensure_dir,                  # create folders if they do not exist
    summarize_history,           # make concise summaries from stored histories
    train_behavior_cloning,      # train BC baseline
    train_cql,                   # train CQL baseline
    train_voac,                  # train VOAC ablation
    train_laadan_ac,             # train proposed LAADAN-AC model
    train_laadan_ac_primal_dual, # train proposed LAADAN-AC primal dual
)


# Default configuration for the whole experiment.

DEFAULT_CONFIG = {
    
    # Finite evaluation horizon used by the benchmark dynamic-programming
    # evaluator and Monte Carlo simulator.
    "horizon": 50,

    # Whether to use exact one-hot state IDs or released cluster-centre features.
    # False means: use the benchmark's continuous state-centre representation.
    "use_one_hot_states": False, 
 

    # Random seeds used for multi-seed reporting.
    "seeds": [42, 43, 44, 45, 46],

    # Behaviour Cloning config

    "bc": {
        "epochs": 1000,          # total training epochs
        "lr": 1e-3,              # optimiser learning rate
        "dropout": 0.10,         # dropout used in the encoder
        "hidden_dim": 128,       # first hidden layer width
        "latent_dim": 128,       # latent representation width
        "eval_every": 10,        # evaluate every 10 epochs
        "entropy_bonus": 0.0,    # optional entropy encouragement, because entropy_bonus = 0.0, 
                                 # it simply learns to copy the expert action labels as closely as possible. 
                                 # It is not being encouraged to explore or stay soft.
    },


    # CQL baseline config

    "cql": {
        "epochs": 1000,
        "lr": 1e-3,
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "gamma": 1.0,            # future reward is fully counted within the finite horizon
        "cql_alpha": 0.5,        # strength of conservative Q penalty
        "tau": 0.02,             # target network moves 2% toward main network each update
    },

 
    # VOAC ablation config
   
    "voac": {
        "epochs": 1000,
        "actor_lr": 1e-3,        # actor optimiser learning rate
        "critic_lr": 1e-3,       # critics optimiser learning rate
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "gamma": 1.0,
        "tau": 0.02,             # target network update speed
        "entropy_coef": 0.02,    # encourages a softer/spread-out policy
    },

  
    # LAADAN-AC proposed model config

    "laadan_ac": {
        "epochs": 1000,
        "actor_lr": 5e-4,            # actor learning rate = 0.0005, smaller for stability
        "critic_lr": 1e-3,           # critic learning rate = 0.001
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "gamma": 1.0,                # future reward fully counted
        "tau": 0.01,                 # target network updates slowly, 1% per update
        "entropy_coef": 0.001,       # small entropy encouragement
        "conservative_alpha": 0.5,   # conservative critic penalty strength
        "expert_kl_weight": 0.005,   # how strongly policy stays near expert
        "smoothness_weight": 0.001,  # discourages rough action changes
        "cost_budget": 0.0,          # expected unsafe cost target is zero
        "lagrange_lr": 0.0002,       # update rate for Lagrange multiplier
        "lagrange_init": 0.0,        # Lagrange multiplier starts at zero
    },
    
        "laadan_ac_pd": {
        "epochs": 1000,
        "actor_lr": 5e-4,
        "critic_lr": 1e-3,
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "gamma": 1.0,
        "tau": 0.01,
        "entropy_coef": 0.001,

        # Robust critic settings.
        "num_cost_heads": 3,
        "rho_reward": 1.0,
        "rho_cost": 1.0,

        # Conservative critic pressure.
        "conservative_alpha": 0.5,

        # No hard softmax mask during training.
        # Safety is handled by risk-adjusted logits and certified nudging.
        "use_action_mask": False,

        # Multi-constraint budgets.
        "cost_budget": 0.0001,
        "support_budget": 0.25,
        "smooth_budget": 1.0,
        "uncertainty_budget": 0.25,

        # Multi-lambda initialisation.
        "lambda_inad_init": 0.0,
        "lambda_support_init": 0.0,
        "lambda_smooth_init": 0.0,
        "lambda_uncertainty_init": 0.0,

        # Multi-lambda learning rates.
        "lambda_inad_lr": 0.01,
        "lambda_support_lr": 0.001,
        "lambda_smooth_lr": 0.001,
        "lambda_uncertainty_lr": 0.001,

        "lagrange_max": 100.0,
        "lambda_update_every": 10,

        # Fixed robust safety shaping.
        "barrier_weight": 25.0,
        "support_weight": 1.0,
        "smoothness_weight": 0.001,
        "uncertainty_weight": 0.1,

        # Policy and certification controls.
        "policy_temperature": 1.0,
        "energy_scale": 1.0,
        "cost_threshold": 0.05,
        "uncertainty_threshold": 1.0,
        "expert_threshold": 1e-8,
    },
    
    # LAADAN-AC uses actor-critic learning but adds safety mechanisms: masking, conservative critics, 
    # expert regularisation, smoothness cost and a Lagrangian cost penalty.

    # Hyperparameter sensitivity study for LAADAN-AC.
    # Each key is varied one-at-a-time around the base configuration.
    "laadan_hyperparameter_study": {
        "conservative_alpha": [0.10, 0.25, 0.50],
        "expert_kl_weight": [0.002, 0.005, 0.010],
        "smoothness_weight": [0.0005, 0.001, 0.002],
    },
    
} 

def choose_device(requested):
    
    """
    Choosing CPU or CUDA based on the command-line request.

    """
    # If explicitly request CPU, return CPU immediately.
    if requested == "cpu":
        return "cpu"

    # If explicitly request CUDA, verify that CUDA exists.
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"

    # Otherwise auto-select CUDA if present, else fall back to CPU.
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(): 
    
    # parse_args() reads terminal options, so the same experiment script can run in debug mode, 
    # full mode, Kaggle, or locally without changing code.
    
    """
    Read command-line arguments.

    """
    
    # Creating the parser with a short description shown in --help.
    parser = argparse.ArgumentParser(description="Offline deep RL experiments on the ICU-Sepsis benchmark.")

    # Path to the extracted benchmark tables/files.
    parser.add_argument("--data-dir", required=True, help="Path to the extracted ICU-Sepsis CSV tables.")

    # Output folder for checkpoints, summaries, and figures.
    parser.add_argument("--results-dir", default="results", help="Directory for checkpoints, metrics, and figures.")

    # Device selection policy.
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Debug mode reduces work for quick end-to-end verification.
    parser.add_argument("--mode", choices=["debug", "main"], default="main")

    # Optional flag to skip the LAADAN hyperparameter study.
    parser.add_argument("--skip-study", action="store_true")

    # Optional flag to skip plot generation.
    parser.add_argument("--skip-plots", action="store_true")

    # Parsing and returning the command-line arguments.
    return parser.parse_args()


def config_from_mode(mode):
    
    """
    Building a copy of the default configuration and shrink it for debug mode.

    """
    
    # Deep-copy the nested configuration by converting to JSON and back.
    config = json.loads(json.dumps(DEFAULT_CONFIG))

    # If requested debug mode, reduce seeds and epochs.
    if mode == "debug":
        # Use only one seed to make the run faster.
        config["seeds"] = [42]

        # Reduce training length for each model.
        config["bc"]["epochs"] = 40
        config["cql"]["epochs"] = 40
        config["voac"]["epochs"] = 40
        config["laadan_ac"]["epochs"] = 40
        config["laadan_ac_pd"]["epochs"] = 40

        # Evaluating more frequently so the short run still produces curves.
        config["bc"]["eval_every"] = 5
        config["cql"]["eval_every"] = 5
        config["voac"]["eval_every"] = 5
        config["laadan_ac"]["eval_every"] = 5
        config["laadan_ac_pd"]["eval_every"] = 5

        # Update lambda at the same frequency during debug mode.
        config["laadan_ac_pd"]["lambda_update_every"] = 5

        # Keeping the debug sensitivity study tiny.
        config["laadan_hyperparameter_study"] = {
            "expert_kl_weight": [0.002, 0.005],
        }

    # Returning the final mode-specific config.
    return config


def save_json(path, payload):
    
    """
    Write JSON to disk.

    This helper avoids repeating the same file-writing code in multiple places.
    """
    # Open the output file for writing in UTF-8 text mode.
    with open(path, "w", encoding="utf-8") as handle:
        
        # Writing the JSON with indentation so it is readable for me later
        json.dump(payload, handle, indent=2)


def evaluate_best_policies(benchmark, grouped_runs):
    
    # After training is finished, simulate the already-trained best policy for 2000 sampled patient episodes.

    # Dictionary to store simulation results for each model family.
    simulation_metrics = {}

    # Looping over each model family and its list of seed runs.
    for model_name, run_list in grouped_runs.items():
        
      
        # Selecting the primary policy from the seed with the best survival.
        best_policy = best_policy_from_runs(run_list)

        # Simulating that best policy using the benchmark's Monte Carlo simulator.
        simulation_metrics[model_name] = benchmark.simulate_policy(best_policy, num_episodes=2000, seed=123)

    # Returning all simulation summaries.
    return simulation_metrics


def collect_best_histories(grouped_runs): # keeps the training history from the best seed of each model.
    
    """
    Keeping the best-seed history for each model for later inspection.

    """
    
    # Dictionary storing one selected history per model family.
    payload = {}

    # Looping over model families.
    for model_name, run_list in grouped_runs.items():
        # Starting by assuming the first seed is best.
        best_index = 0
        best_survival = run_list[0]["metrics"]["survival_rate"]

        # Comparing later seeds against the current best one.
        for index in range(1, len(run_list)):
            value = run_list[index]["metrics"]["survival_rate"]
            if value > best_survival:
                best_survival = value
                best_index = index

        # Saving the selected best history.
        payload[model_name] = run_list[best_index]["history"]

    # Returning the chosen histories.
    return payload


def choose_selected_state_ids(benchmark, max_states=12):
    
    """
    Choosing a small evenly spaced set of non-terminal states for heatmaps.

    """
    
    # Collecting all non-terminal state IDs.
    candidates = [state_id for state_id in range(benchmark.num_states) if benchmark.terminal_mask[state_id] == 0]

    # If there are already few enough states, return them all.
    if len(candidates) <= max_states:
        return candidates

    # Otherwise, choose evenly spaced indices over the candidate list.
    indices = np.linspace(0, len(candidates) - 1, num=max_states, dtype=int)

    # Returning the selected state IDs.
    return [candidates[index] for index in indices]


def build_policy_snapshots(benchmark, grouped_runs):
    
    """
    Building a small dictionary of policies for policy heatmaps.

    """
    # Returning a dictionary mapping each name to its policy array.
    
    return {
        # The benchmark expert policy is included for comparison.
        "Expert Policy": benchmark.expert_safe.tolist(),

        # Best primary policy from each learned model family.
        "Behavior Cloning": best_policy_from_runs(grouped_runs["Behavior Cloning"]).tolist(),
        "Conservative Q-Learning": best_policy_from_runs(grouped_runs["Conservative Q-Learning"]).tolist(),
        "Vanilla Offline Actor-Critic": best_policy_from_runs(grouped_runs["Vanilla Offline Actor-Critic"]).tolist(),
        "LAADAN-AC": best_policy_from_runs(grouped_runs["LAADAN-AC"]).tolist(),
        "LAADAN-AC-PD": best_policy_from_runs(grouped_runs["LAADAN-AC-PD"]).tolist(),
    }


def run_laadan_hyperparameter_study(benchmark, results_dir, study_cfg, base_cfg, seed):
    
    """
    Running a small sensitivity study for the proposed model.
    """
    
    # Dictionary storing study results grouped by parameter name.
    study_results = {}

    # Looping over each hyperparameter to be studied.
    for param_name, values in study_cfg.items():


        # Creating a nested dictionary for this parameter.
        study_results[param_name] = {}

        # Trying each candidate value one at a time.
        for value in values:
            # Starting from the base LAADAN configuration.
            config = dict(base_cfg)

            # Overriding just the parameter currently being studied.
            config[param_name] = value

            # Training one LAADAN run with the modified config.
            run = train_laadan_ac(benchmark, seed, os.path.join(results_dir, "studies"), config)

            # Storing the headline outcomes for this tested value.
            study_results[param_name][str(value)] = {
                "survival_rate": run["metrics"]["survival_rate"],
                "inadmissibility_rate": run["metrics"]["inadmissibility_rate"],
                "avg_return": run["metrics"]["avg_return"],
            }

    # Returning the full sensitivity-study payload.
    return study_results

def main():
    
    """
    Running the full multi-seed experiment pipeline.
    """

    # Reading command-line arguments.
    args = parse_args()
    
    """ args.data_dir
        args.results_dir
        args.device
        args.mode
        args.skip_study
        args.skip_plots
    """

    # Building the mode-specific configuration.
    config = config_from_mode(args.mode)

    # Resolving the actual device string, choose cpu or cuda.
    device = choose_device(args.device)

    # Making sure the results folder exists before writing anything.
    ensure_dir(args.results_dir)

    # Building the benchmark object used by all models.
    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=config["horizon"],
        device=device,
        use_one_hot_states=bool(config.get("use_one_hot_states", False)),
    )

    # Computing the benchmark reference policies' exact metrics for the expected return.
    benchmark_reference_metrics = benchmark.reference_metrics(gamma=1.0, horizon=config["horizon"])

    # Saving a small description of the loaded benchmark.
    benchmark.save_benchmark_description(os.path.join(args.results_dir, "benchmark_description.json"))

    # Saving the actual run configuration used for this experiment.
    save_json(os.path.join(args.results_dir, "run_config.json"), {"device": device, **config})

    # Lists that will hold one run dictionary per seed for each model family.
    bc_runs = []
    cql_runs = []
    voac_runs = []
    laadan_runs = []
    pd_runs = []

    # Training each model family for each requested seed.
    for seed in config["seeds"]:
        
        """ Loop over: 42, 43, 44, 45, 46. For each seed, train all models. """
    
        # Behaviour Cloning
        print("Running seed", seed, "for Behavior Cloning")
        bc_runs.append(
            train_behavior_cloning(
                benchmark,
                seed,
                args.results_dir,
                config["bc"],
            )
        )

        # Conservative Q-Learning
        print("Running seed", seed, "for Conservative Q-Learning")
        cql_runs.append(
            train_cql(
                benchmark,
                seed,
                args.results_dir,
                config["cql"],
            )
        )

        # VOAC ablation
        print("Running seed", seed, "for Vanilla Offline Actor-Critic")
        voac_runs.append(
            train_voac(
                benchmark,
                seed,
                args.results_dir,
                config["voac"],
            )
        )

        # LAADAN-AC proposed model
        print("Running seed", seed, "for LAADAN-AC")
        laadan_runs.append(
            train_laadan_ac(
                benchmark,
                seed,
                args.results_dir,
                config["laadan_ac"],
            )
        )

        # LAADAN-AC-PD primal-dual Lagrangian variant
        print("Running seed", seed, "for LAADAN-AC-PD")
        pd_runs.append(
            train_laadan_ac_primal_dual(
                benchmark,
                seed,
                args.results_dir,
                config["laadan_ac_pd"],
            )
        )
        
    # Grouping runs by model family name so later code is cleaner.
    grouped_runs = {
        "Behavior Cloning": bc_runs,
        "Conservative Q-Learning": cql_runs,
        "Vanilla Offline Actor-Critic": voac_runs,
        "LAADAN-AC": laadan_runs,
        "LAADAN-AC-PD": pd_runs,
    }

    # Aggregating final metrics across seeds for each model family.
    aggregate_metrics = {
        "Behavior Cloning": aggregate_seed_metrics(bc_runs),
        "Conservative Q-Learning": aggregate_seed_metrics(cql_runs),
        "Vanilla Offline Actor-Critic": aggregate_seed_metrics(voac_runs),
        "LAADAN-AC": aggregate_seed_metrics(laadan_runs),
        "LAADAN-AC-PD": aggregate_seed_metrics(pd_runs),
    }

    # Aggregating training histories across seeds for the main plotted metrics.
    aggregated_histories = {
        "Behavior Cloning": aggregate_histories(
            bc_runs,
            ["loss", "survival_rate", "inadmissibility_rate", "expert_argmax_match"],
        ),
        "Conservative Q-Learning": aggregate_histories(
            cql_runs,
            ["td_loss", "cql_loss", "survival_rate", "inadmissibility_rate"],
        ),
        "Vanilla Offline Actor-Critic": aggregate_histories(
            voac_runs,
            ["critic_loss", "actor_loss", "survival_rate", "inadmissibility_rate"],
        ),
        "LAADAN-AC": aggregate_histories(
            laadan_runs,
            ["critic_loss", "actor_loss", "lagrange", "survival_rate", "inadmissibility_rate"],
        ),
        "LAADAN-AC-PD": aggregate_histories(
            pd_runs,
            [
                "critic_loss",
                "actor_loss",
                "lagrange",
                "exact_soft_inadmissibility_rate",
                "survival_rate",
                "inadmissibility_rate",
                "soft_inadmissibility_rate",
            ],
        ),
    }
     
    # Keeping the best seed's raw history per model family.
    per_seed_histories = collect_best_histories(grouped_runs)

    # Building Monte Carlo simulation summaries from the best policies.
    simulation_metrics = evaluate_best_policies(benchmark, grouped_runs)

    # Selecting representative non-terminal states for compact heatmaps.
    selected_state_ids = choose_selected_state_ids(benchmark)

    # Building a dictionary of primary policies for the heatmaps.
    policy_snapshots = build_policy_snapshots(benchmark, grouped_runs)

    # Running or skipping the LAADAN sensitivity study depending on user choice.
    if args.skip_study:
        study_payload = {}
    else:
        study_payload = run_laadan_hyperparameter_study(
            benchmark,
            args.results_dir,
            config["laadan_hyperparameter_study"],
            config["laadan_ac"],
            seed=config["seeds"][0],
        )
        
    # Building short history summaries for quick inspection.
    concise_summaries = {
        "Behavior Cloning": summarize_history(per_seed_histories["Behavior Cloning"]),
        "Conservative Q-Learning": summarize_history(per_seed_histories["Conservative Q-Learning"]),
        "Vanilla Offline Actor-Critic": summarize_history(per_seed_histories["Vanilla Offline Actor-Critic"]),
        "LAADAN-AC": summarize_history(per_seed_histories["LAADAN-AC"]),
        "LAADAN-AC-PD": summarize_history(per_seed_histories["LAADAN-AC-PD"]),
    }

    # Constructing the full summary payload that downstream tools/plots will use.
    payload = {
        "benchmark_reference_metrics": benchmark_reference_metrics,
        "aggregate_metrics": aggregate_metrics,
        "simulation_metrics": simulation_metrics,
        "best_seed_histories": per_seed_histories,
        "aggregated_histories": aggregated_histories,
        "history_summaries": concise_summaries,
        "laadan_hyperparameter_study": study_payload,
        "policy_snapshots": policy_snapshots,
        "selected_state_ids": selected_state_ids,
    }

    # Saving the full summary to disk.
    save_json(os.path.join(args.results_dir, "summary.json"), payload)

    # Building plots unless the user explicitly asked to skip them.
    if not args.skip_plots:
        build_all_plots(args.results_dir)

    # Printing benchmark-reference results to the console for quick context.
    print("\nBenchmark references:")
    for reference_name, metrics in benchmark_reference_metrics.items():
        print(
            reference_name,
            "survival=",
            round(metrics["survival_rate"], 4),
            "inad=",
            round(metrics["inadmissibility_rate"], 4),
            "return=",
            round(metrics["avg_return"], 4),
        )

    # Printing final run-summary location.
    print("\nFinished. Summary written to", os.path.join(args.results_dir, "summary.json"))

    # Printing one concise headline line per learned model family.
    for model_name, metrics in aggregate_metrics.items():
        print(
            model_name,
            "survival=",
            round(metrics["survival_rate"]["mean"], 4),
            "inad=",
            round(metrics["inadmissibility_rate"]["mean"], 4),
            "return=",
            round(metrics["avg_return"]["mean"], 4),
        )


# Main entry point
if __name__ == "__main__":
    main()
