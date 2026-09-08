









import argparse


import json


import os


import numpy as np


import torch


from benchmark import ICUSepsisOfflineBenchmark


from plots import build_all_plots


from trainers import (
    aggregate_histories,
    aggregate_seed_metrics,
    best_policy_from_runs,
    ensure_dir,
    summarize_history,
    train_behavior_cloning,
    train_cql,
    train_voac,
    train_laadan_ac,
)




DEFAULT_CONFIG = {



    "horizon": 50,



    "use_one_hot_states": False,



    "seeds": [42, 43, 44, 45, 46],



    "bc": {
        "epochs": 1000,
        "lr": 1e-3,
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "entropy_bonus": 0.0,


    },




    "cql": {
        "epochs": 1000,
        "lr": 1e-3,
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "gamma": 1.0,
        "cql_alpha": 0.5,
        "tau": 0.02,
    },




    "voac": {
        "epochs": 1000,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "dropout": 0.10,
        "hidden_dim": 128,
        "latent_dim": 128,
        "eval_every": 10,
        "gamma": 1.0,
        "tau": 0.02,
        "entropy_coef": 0.02,
    },




    "laadan_ac": {
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
        "conservative_alpha": 0.5,
        "expert_kl_weight": 0.005,
        "smoothness_weight": 0.001,
        "cost_budget": 0.0,
        "lagrange_lr": 0.0002,
        "lagrange_init": 0.0,
    },






    "laadan_hyperparameter_study": {
        "conservative_alpha": [0.10, 0.25, 0.50],
        "expert_kl_weight": [0.002, 0.005, 0.010],
        "smoothness_weight": [0.0005, 0.001, 0.002],
    },

}


# experiments
# select device
def choose_device(requested):






    if requested == "cpu":
        return "cpu"


    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"


    return "cuda" if torch.cuda.is_available() else "cpu"


# arguments
def parse_args():










    parser = argparse.ArgumentParser(description="Offline deep RL experiments on the ICU-Sepsis benchmark.")


    parser.add_argument("--data-dir", required=True, help="Path to the extracted ICU-Sepsis CSV tables.")


    parser.add_argument("--results-dir", default="results", help="Directory for checkpoints, metrics, and figures.")


    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")


    parser.add_argument("--mode", choices=["debug", "main"], default="main")


    parser.add_argument("--skip-study", action="store_true")


    parser.add_argument("--skip-plots", action="store_true")


    return parser.parse_args()


# configuration
def config_from_mode(mode):







    config = json.loads(json.dumps(DEFAULT_CONFIG))


    if mode == "debug":

        config["seeds"] = [42]


        config["bc"]["epochs"] = 40
        config["cql"]["epochs"] = 40
        config["voac"]["epochs"] = 40
        config["laadan_ac"]["epochs"] = 40


        config["bc"]["eval_every"] = 5
        config["cql"]["eval_every"] = 5
        config["voac"]["eval_every"] = 5
        config["laadan_ac"]["eval_every"] = 5


        config["laadan_hyperparameter_study"] = {
            "expert_kl_weight": [0.002, 0.005],
        }


    return config


# save results
def save_json(path, payload):







    with open(path, "w", encoding="utf-8") as handle:


        json.dump(payload, handle, indent=2)


# policy evaluation
def evaluate_best_policies(benchmark, grouped_runs):




    simulation_metrics = {}


    for model_name, run_list in grouped_runs.items():



        best_policy = best_policy_from_runs(run_list)


        simulation_metrics[model_name] = benchmark.simulate_policy(best_policy, num_episodes=2000, seed=123)


    return simulation_metrics


# collect histories
def collect_best_histories(grouped_runs):







    payload = {}


    for model_name, run_list in grouped_runs.items():

        best_index = 0
        best_survival = run_list[0]["metrics"]["survival_rate"]


        for index in range(1, len(run_list)):
            value = run_list[index]["metrics"]["survival_rate"]
            if value > best_survival:
                best_survival = value
                best_index = index


        payload[model_name] = run_list[best_index]["history"]


    return payload


# select states
def choose_selected_state_ids(benchmark, max_states=12):







    candidates = [state_id for state_id in range(benchmark.num_states) if benchmark.terminal_mask[state_id] == 0]


    if len(candidates) <= max_states:
        return candidates


    indices = np.linspace(0, len(candidates) - 1, num=max_states, dtype=int)


    return [candidates[index] for index in indices]


# policy snapshots
def build_policy_snapshots(benchmark, grouped_runs):







    return {

        "Expert Policy": benchmark.expert_safe.tolist(),


        "Behavior Cloning": best_policy_from_runs(grouped_runs["Behavior Cloning"]).tolist(),
        "Conservative Q-Learning": best_policy_from_runs(grouped_runs["Conservative Q-Learning"]).tolist(),
        "Vanilla Offline Actor-Critic": best_policy_from_runs(grouped_runs["Vanilla Offline Actor-Critic"]).tolist(),
        "LAADAN-AC": best_policy_from_runs(grouped_runs["LAADAN-AC"]).tolist(),
    }


# hyperparameter study
def run_laadan_hyperparameter_study(benchmark, results_dir, study_cfg, base_cfg, seed):






    study_results = {}


    for param_name, values in study_cfg.items():



        study_results[param_name] = {}


        for value in values:

            config = dict(base_cfg)


            config[param_name] = value


            run = train_laadan_ac(benchmark, seed, os.path.join(results_dir, "studies"), config)


            study_results[param_name][str(value)] = {
                "survival_rate": run["metrics"]["survival_rate"],
                "inadmissibility_rate": run["metrics"]["inadmissibility_rate"],
                "avg_return": run["metrics"]["avg_return"],
            }


    return study_results


def main():






    args = parse_args()










    config = config_from_mode(args.mode)


    device = choose_device(args.device)


    ensure_dir(args.results_dir)


    benchmark = ICUSepsisOfflineBenchmark(
        args.data_dir,
        horizon=config["horizon"],
        device=device,
        use_one_hot_states=bool(config.get("use_one_hot_states", False)),
    )



    benchmark_reference_metrics = benchmark.reference_metrics(gamma=1.0, horizon=config["horizon"])


    benchmark.save_benchmark_description(os.path.join(args.results_dir, "benchmark_description.json"))


    save_json(os.path.join(args.results_dir, "run_config.json"), {"device": device, **config})


    bc_runs = []
    cql_runs = []
    voac_runs = []
    laadan_runs = []


    for seed in config["seeds"]:





        print("Running seed", seed, "for Behavior Cloning")
        bc_runs.append(train_behavior_cloning(benchmark, seed, args.results_dir, config["bc"]))




        print("Running seed", seed, "for Conservative Q-Learning")
        cql_runs.append(train_cql(benchmark, seed, args.results_dir, config["cql"]))




        print("Running seed", seed, "for Vanilla Offline Actor-Critic")
        voac_runs.append(train_voac(benchmark, seed, args.results_dir, config["voac"]))




        print("Running seed", seed, "for LAADAN-AC")
        laadan_runs.append(train_laadan_ac(benchmark, seed, args.results_dir, config["laadan_ac"]))




    grouped_runs = {
        "Behavior Cloning": bc_runs,
        "Conservative Q-Learning": cql_runs,
        "Vanilla Offline Actor-Critic": voac_runs,
        "LAADAN-AC": laadan_runs,
    }


    aggregate_metrics = {
        "Behavior Cloning": aggregate_seed_metrics(bc_runs),
        "Conservative Q-Learning": aggregate_seed_metrics(cql_runs),
        "Vanilla Offline Actor-Critic": aggregate_seed_metrics(voac_runs),
        "LAADAN-AC": aggregate_seed_metrics(laadan_runs),
    }



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
    }



    per_seed_histories = collect_best_histories(grouped_runs)


    simulation_metrics = evaluate_best_policies(benchmark, grouped_runs)


    selected_state_ids = choose_selected_state_ids(benchmark)


    policy_snapshots = build_policy_snapshots(benchmark, grouped_runs)


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



    concise_summaries = {
        "Behavior Cloning": summarize_history(per_seed_histories["Behavior Cloning"]),
        "Conservative Q-Learning": summarize_history(per_seed_histories["Conservative Q-Learning"]),
        "Vanilla Offline Actor-Critic": summarize_history(per_seed_histories["Vanilla Offline Actor-Critic"]),
        "LAADAN-AC": summarize_history(per_seed_histories["LAADAN-AC"]),
    }


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


    save_json(os.path.join(args.results_dir, "summary.json"), payload)


    if not args.skip_plots:
        build_all_plots(args.results_dir)


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


    print("\nFinished. Summary written to", os.path.join(args.results_dir, "summary.json"))


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



if __name__ == "__main__":
    main()
