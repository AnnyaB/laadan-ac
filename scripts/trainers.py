import csv


import json


import os


import time


import numpy as np



import torch



from models import BehaviorCloningNet, ConservativeQNet, OfflineActorCriticNet, soft_update



# training
def ensure_dir(path):


    if not os.path.exists(path):

        os.makedirs(path)




_T_CRIT_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def mean_ci95(values):



    values = np.asarray(values, dtype=float)


    values = values[np.isfinite(values)]


    n = int(values.size)



    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_half": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }


    mean = float(np.mean(values))



    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_half": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }



    std = float(np.std(values, ddof=1))



    t_crit = _T_CRIT_95.get(n, 1.96)


    ci95_half = float(t_crit * std / np.sqrt(n))



    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_half": ci95_half,
        "ci95_low": mean - ci95_half,
        "ci95_high": mean + ci95_half,
    }


def plain_softmax_from_logits(logits):


    return torch.softmax(logits, dim=1)


def plain_log_softmax_from_logits(logits):



    return torch.log_softmax(logits, dim=1)




# admissible policy
def masked_softmax_from_logits(logits, admissible_mask):

    large_negative = torch.full_like(logits, -1e9)



    masked = torch.where(admissible_mask > 0.5, logits, large_negative)


    return torch.softmax(masked, dim=1)


def masked_log_softmax_from_logits(logits, admissible_mask):


    large_negative = torch.full_like(logits, -1e9)


    masked = torch.where(admissible_mask > 0.5, logits, large_negative)


    return torch.log_softmax(masked, dim=1)


def greedy_policy_from_logits(logits):

    best = torch.argmax(logits, dim=1)


    policy = torch.zeros_like(logits)


    policy.scatter_(1, best.unsqueeze(1), 1.0)


    return policy


def masked_greedy_policy_from_logits(logits, admissible_mask):


    large_negative = torch.full_like(logits, -1e9)


    masked = torch.where(admissible_mask > 0.5, logits, large_negative)


    best = torch.argmax(masked, dim=1)


    policy = torch.zeros_like(logits)


    policy.scatter_(1, best.unsqueeze(1), 1.0)


    return policy


def weighted_policy_kl(logits, target_policy, admissible_mask=None):



    if admissible_mask is None:
        log_probs = plain_log_softmax_from_logits(logits)
    else:
        log_probs = masked_log_softmax_from_logits(logits, admissible_mask)




    target = target_policy / torch.clamp(torch.sum(target_policy, dim=1, keepdim=True), min=1e-8)



    kl = torch.sum(target * (torch.log(target + 1e-8) - log_probs), dim=1)


    return torch.mean(kl)


def cql_regularizer(q_values, expert_policy, admissible_mask=None):


    if admissible_mask is None:
        conservative_term = torch.logsumexp(q_values, dim=1)


    else:

        masked = torch.where(admissible_mask > 0.5, q_values, torch.full_like(q_values, -1e9))

        conservative_term = torch.logsumexp(masked, dim=1)



    data_term = torch.sum(expert_policy * q_values, dim=1)


    return torch.mean(conservative_term - data_term)


def masked_action_mse(pred, target, action_mask=None):

    squared = (pred - target) ** 2


    if action_mask is None:
        return torch.mean(squared)


    weights = action_mask.float()


    denom = torch.clamp(torch.sum(weights), min=1.0)


    return torch.sum(squared * weights) / denom


def save_history_csv(path, rows):


    if not rows:
        return


    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)


    with open(path, "w", newline="", encoding="utf-8") as handle:

        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")


        writer.writeheader()


        for row in rows:
            full_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(full_row)


def summarize_history(history):


    if not history:
        return {}


    all_keys = []
    seen = set()
    for row in history:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)


    summary = {}


    for key in all_keys:
        values = []


        for row in history:
            value = row.get(key, np.nan)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                values.append(float(value))


        if values:
            summary[key + "_last"] = values[-1]
            summary[key + "_best_min"] = float(np.min(values))
            summary[key + "_best_max"] = float(np.max(values))


    return summary


def epoch_to_fraction_of_best(history, metric_name, fraction=0.95):


    metric_pairs = []
    for row in history:
        value = row.get(metric_name, np.nan)
        if np.isfinite(value):
            metric_pairs.append((int(row["epoch"]), float(value)))


    if not metric_pairs:
        return None


    best_value = max(value for _, value in metric_pairs)


    threshold = fraction * best_value


    for epoch, value in metric_pairs:
        if value >= threshold:
            return int(epoch)


    return None


def save_model_run(seed_dir, model, history, metrics):


    ensure_dir(seed_dir)


    torch.save(model.state_dict(), os.path.join(seed_dir, "model.pt"))


    save_history_csv(os.path.join(seed_dir, "history.csv"), history)


    with open(os.path.join(seed_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


@torch.no_grad()
# evaluate policy set
def evaluate_policy_set(benchmark, policy, soft_policy=None):


    metrics = benchmark.exact_policy_evaluation(policy)



    if soft_policy is not None:
        soft_metrics = benchmark.exact_policy_evaluation(soft_policy)
        metrics["soft_survival_rate"] = soft_metrics["survival_rate"]
        metrics["soft_inadmissibility_rate"] = soft_metrics["inadmissibility_rate"]
        metrics["soft_avg_return"] = soft_metrics["avg_return"]
        metrics["soft_policy_entropy"] = soft_metrics["policy_entropy"]
        metrics["soft_mean_kl_to_expert"] = soft_metrics["mean_kl_to_expert"]



    return metrics


@torch.no_grad()
def policy_numpy(tensor_policy):





    return tensor_policy.detach().cpu().numpy()




# train bc
def train_behavior_cloning(benchmark, seed, results_dir, config):



    benchmark.set_seed(seed)


    device = benchmark.device


    model = BehaviorCloningNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)



    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))


    x = benchmark.state_features_t
    target = benchmark.expert_safe_t


    history = []
    final_metrics = None


    start_time = time.time()


    for epoch in range(1, epochs + 1):

        model.train()


        optimizer.zero_grad()


        logits = model(x)


        kl_loss = weighted_policy_kl(logits, target, admissible_mask=None)


        probs = plain_softmax_from_logits(logits)


        log_probs = plain_log_softmax_from_logits(logits)


        entropy = -torch.mean(torch.sum(probs * log_probs, dim=1))




        loss = kl_loss - float(config.get("entropy_bonus", 0.0)) * entropy


        loss.backward()


        optimizer.step()


        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "kl_loss": float(kl_loss.item()),
            "entropy": float(entropy.item()),
        }


        if epoch % eval_every == 0 or epoch == epochs:


            model.eval()

            with torch.no_grad():


                logits_eval = model(x)


                greedy_policy = greedy_policy_from_logits(logits_eval)


                soft_policy = plain_softmax_from_logits(logits_eval)





            metrics = evaluate_policy_set(
                benchmark,
                policy_numpy(greedy_policy),
                soft_policy=policy_numpy(soft_policy),
            )


            row.update(metrics)


            if epoch == epochs:
                final_metrics = metrics


        history.append(row)



    model.eval()
    with torch.no_grad():
        logits_final = model(x)
        final_policy = greedy_policy_from_logits(logits_final)
        final_soft_policy = plain_softmax_from_logits(logits_final)


    if final_metrics is None:
        final_metrics = evaluate_policy_set(
            benchmark,
            policy_numpy(final_policy),
            soft_policy=policy_numpy(final_soft_policy),
        )


    seed_dir = os.path.join(results_dir, "bc", "seed_" + str(seed))


    save_model_run(seed_dir, model, history, final_metrics)


    return {
        "name": "Behavior Cloning",
        "seed": seed,
        "model": model,
        "policy": policy_numpy(final_policy),
        "analysis_policy": policy_numpy(final_soft_policy),
        "history": history,
        "metrics": final_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }




# train cql
def train_cql(benchmark, seed, results_dir, config):


    benchmark.set_seed(seed)


    device = benchmark.device


    model = ConservativeQNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)


    target_model = ConservativeQNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
    ).to(device)


    target_model.load_state_dict(model.state_dict())


    target_model.eval()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


    gamma = float(config.get("gamma", 1.0))
    cql_alpha = float(config.get("cql_alpha", 1.0))
    tau = float(config.get("tau", 0.02))
    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))



    x = benchmark.state_features_t
    expert = benchmark.expert_safe_t
    reward_sa = benchmark.reward_sa_t
    transition = benchmark.transition_t
    terminal = benchmark.terminal_mask_t


    history = []
    final_metrics = None
    start_time = time.time()


    for epoch in range(1, epochs + 1):


        model.train()


        optimizer.zero_grad()


        q_values = model(x)


        with torch.no_grad():
            q_target_now = target_model(x)


            next_v = torch.max(q_target_now, dim=1).values * (1.0 - terminal)


            bellman_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)



        td_loss = torch.mean((q_values - bellman_target) ** 2)


        conservative_loss = cql_regularizer(q_values, expert, admissible_mask=None)


        loss = td_loss + cql_alpha * conservative_loss


        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)


        optimizer.step()


        soft_update(target_model, model, tau)


        target_model.eval()


        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(conservative_loss.item()),
        }


        if epoch % eval_every == 0 or epoch == epochs:

            model.eval()

            with torch.no_grad():

                q_now = model(x).detach().cpu().numpy()


                greedy_policy = benchmark.greedy_policy_from_q_unmasked(q_now)




            metrics = benchmark.exact_policy_evaluation(greedy_policy)


            row.update(metrics)


            if epoch == epochs:
                final_metrics = metrics


        history.append(row)


    model.eval()
    with torch.no_grad():
        q_final = model(x).detach().cpu().numpy()
        final_policy = benchmark.greedy_policy_from_q_unmasked(q_final)


    if final_metrics is None:
        final_metrics = benchmark.exact_policy_evaluation(final_policy)


    seed_dir = os.path.join(results_dir, "cql", "seed_" + str(seed))


    save_model_run(seed_dir, model, history, final_metrics)


    return {
        "name": "Conservative Q-Learning",
        "seed": seed,
        "model": model,
        "policy": final_policy,
        "analysis_policy": final_policy,
        "history": history,
        "metrics": final_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }


def _actor_critic_common_setup(benchmark, config, use_cost_head):


    device = benchmark.device


    model = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
        use_cost_head=use_cost_head,
    ).to(device)


    target_model = OfflineActorCriticNet(
        benchmark.feature_dim,
        benchmark.num_actions,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        dropout=float(config.get("dropout", 0.10)),
        use_cost_head=use_cost_head,
    ).to(device)



    target_model.load_state_dict(model.state_dict())


    target_model.eval()


    tensors = {
        "x": benchmark.state_features_t,
        "admissible": benchmark.admissible_mask_t,
        "expert": benchmark.expert_safe_t,
        "reward_sa": benchmark.reward_sa_t,
        "transition": benchmark.transition_t,
        "terminal": benchmark.terminal_mask_t,
        "immediate_cost": benchmark.immediate_cost_t,
        "smoothness_cost": benchmark.smoothness_cost_t,
    }


    return model, target_model, tensors




# train voac
def train_voac(benchmark, seed, results_dir, config):


    benchmark.set_seed(seed)


    model, target_model, tensors = _actor_critic_common_setup(benchmark, config, use_cost_head=False)


    actor_params = list(model.actor_encoder.parameters()) + list(model.actor_head.parameters())






    critic_params = (
        list(model.critic_encoder.parameters())
        + list(model.q1_head.parameters())
        + list(model.q2_head.parameters())
    )




    actor_optimizer = torch.optim.Adam(
        actor_params,
        lr=float(config.get("actor_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


    critic_optimizer = torch.optim.Adam(
        critic_params,
        lr=float(config.get("critic_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


    gamma = float(config.get("gamma", 1.0))
    tau = float(config.get("tau", 0.02))
    entropy_coef = float(config.get("entropy_coef", 0.02))
    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))


    x = tensors["x"]
    reward_sa = tensors["reward_sa"]
    transition = tensors["transition"]
    terminal = tensors["terminal"]


    history = []
    final_metrics = None
    start_time = time.time()


    for epoch in range(1, epochs + 1):

        model.train()




        with torch.no_grad():

            target_outputs = target_model(x)


            next_probs = plain_softmax_from_logits(target_outputs["logits"])
            next_log_probs = plain_log_softmax_from_logits(target_outputs["logits"])




            q1_next = target_outputs["q1"]
            q2_next = target_outputs["q2"]


            min_q_next = torch.min(q1_next, q2_next)




            next_v = torch.sum(
                next_probs * (min_q_next - entropy_coef * next_log_probs),
                dim=1,
            ) * (1.0 - terminal)


            q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)


        critic_optimizer.zero_grad()


        outputs = model(x)
        q1 = outputs["q1"]
        q2 = outputs["q2"]




        q1_loss = torch.mean((q1 - q_target) ** 2)
        q2_loss = torch.mean((q2 - q_target) ** 2)


        critic_loss = q1_loss + q2_loss




        critic_loss.backward()


        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)


        critic_optimizer.step()






        actor_optimizer.zero_grad()


        outputs = model(x)
        logits = outputs["logits"]
        probs = plain_softmax_from_logits(logits)
        log_probs = plain_log_softmax_from_logits(logits)



        q1 = outputs["q1"]
        q2 = outputs["q2"]


        min_q = torch.min(q1, q2)




        actor_loss = torch.mean(torch.sum(probs * (entropy_coef * log_probs - min_q), dim=1))




        actor_loss.backward()


        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)


        actor_optimizer.step()


        soft_update(target_model, model, tau)


        target_model.eval()


        row = {
            "epoch": epoch,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
        }


        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x)["logits"]
                greedy_policy = greedy_policy_from_logits(logits_eval)
                soft_policy = plain_softmax_from_logits(logits_eval)

            metrics = evaluate_policy_set(
                benchmark,
                policy_numpy(greedy_policy),
                soft_policy=policy_numpy(soft_policy),
            )

            row.update(metrics)


            if epoch == epochs:
                final_metrics = metrics


        history.append(row)


    model.eval()
    with torch.no_grad():
        logits_final = model(x)["logits"]
        final_policy = greedy_policy_from_logits(logits_final)
        final_soft_policy = plain_softmax_from_logits(logits_final)


    if final_metrics is None:
        final_metrics = evaluate_policy_set(
            benchmark,
            policy_numpy(final_policy),
            soft_policy=policy_numpy(final_soft_policy),
        )


    seed_dir = os.path.join(results_dir, "voac", "seed_" + str(seed))
    save_model_run(seed_dir, model, history, final_metrics)


    return {
        "name": "Vanilla Offline Actor-Critic",
        "seed": seed,
        "model": model,
        "policy": policy_numpy(final_policy),
        "analysis_policy": policy_numpy(final_soft_policy),
        "history": history,
        "metrics": final_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }




# train laadan-ac
def train_laadan_ac(benchmark, seed, results_dir, config):


    benchmark.set_seed(seed)


    model, target_model, tensors = _actor_critic_common_setup(benchmark, config, use_cost_head=True)


    actor_params = list(model.actor_encoder.parameters()) + list(model.actor_head.parameters())


    critic_params = (
        list(model.critic_encoder.parameters())
        + list(model.q1_head.parameters())
        + list(model.q2_head.parameters())
        + list(model.cost_head.parameters())
    )




    actor_optimizer = torch.optim.Adam(
        actor_params,
        lr=float(config.get("actor_lr", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


    critic_optimizer = torch.optim.Adam(
        critic_params,
        lr=float(config.get("critic_lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


    gamma = float(config.get("gamma", 1.0))
    tau = float(config.get("tau", 0.01))
    entropy_coef = float(config.get("entropy_coef", 0.001))
    conservative_alpha = float(config.get("conservative_alpha", 0.25))
    expert_kl_weight = float(config.get("expert_kl_weight", 0.005))
    smoothness_weight = float(config.get("smoothness_weight", 0.001))
    cost_budget = float(config.get("cost_budget", 0.0))
    lagrange_lr = float(config.get("lagrange_lr", 0.0002))
    lagrange_value = float(config.get("lagrange_init", 0.0))



    epochs = int(config.get("epochs", 300))
    eval_every = int(config.get("eval_every", 10))


    x = tensors["x"]
    admissible = tensors["admissible"]
    expert = tensors["expert"]
    reward_sa = tensors["reward_sa"]
    transition = tensors["transition"]
    terminal = tensors["terminal"]
    immediate_cost = tensors["immediate_cost"]
    smoothness_cost = tensors["smoothness_cost"]


    history = []
    final_metrics = None
    start_time = time.time()


    for epoch in range(1, epochs + 1):

        model.train()



        with torch.no_grad():
            target_outputs = target_model(x)


            next_probs = masked_softmax_from_logits(target_outputs["logits"], admissible)
            next_log_probs = masked_log_softmax_from_logits(target_outputs["logits"], admissible)


            q1_next = target_outputs["q1"]
            q2_next = target_outputs["q2"]


            cost_next = torch.relu(target_outputs["cost"])


            min_q_next = torch.min(q1_next, q2_next)


            next_v = torch.sum(
                next_probs * (min_q_next - entropy_coef * next_log_probs - lagrange_value * cost_next),
                dim=1,
            ) * (1.0 - terminal)


            next_c = torch.sum(next_probs * cost_next, dim=1) * (1.0 - terminal)


            q_target = reward_sa + gamma * torch.einsum("san,n->sa", transition, next_v)


            c_target = immediate_cost + gamma * torch.einsum("san,n->sa", transition, next_c)






        critic_optimizer.zero_grad()

        outputs = model(x)
        q1 = outputs["q1"]
        q2 = outputs["q2"]


        cost_pred = torch.relu(outputs["cost"])


        q1_loss = masked_action_mse(q1, q_target, action_mask=admissible)
        q2_loss = masked_action_mse(q2, q_target, action_mask=admissible)


        cost_loss = masked_action_mse(cost_pred, c_target, action_mask=admissible)


        cql_q1 = cql_regularizer(q1, expert, admissible_mask=admissible)
        cql_q2 = cql_regularizer(q2, expert, admissible_mask=admissible)


        critic_loss = q1_loss + q2_loss + cost_loss + conservative_alpha * (cql_q1 + cql_q2)


        critic_loss.backward()


        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=5.0)


        critic_optimizer.step()





        actor_optimizer.zero_grad()

        outputs = model(x)
        logits = outputs["logits"]




        probs = masked_softmax_from_logits(logits, admissible)
        log_probs = masked_log_softmax_from_logits(logits, admissible)



        q1 = outputs["q1"]
        q2 = outputs["q2"]
        cost_pred = torch.relu(outputs["cost"])


        min_q = torch.min(q1, q2)


        expected_cost = torch.sum(probs * cost_pred, dim=1)


        expected_q = torch.sum(probs * min_q, dim=1)


        entropy = -torch.sum(probs * log_probs, dim=1)


        expert_kl = torch.sum(expert * (torch.log(expert + 1e-8) - log_probs), dim=1)


        expected_smoothness = torch.sum(probs * smoothness_cost, dim=1)




        actor_loss = torch.mean(
            -expected_q
            - entropy_coef * entropy
            + lagrange_value * expected_cost
            + expert_kl_weight * expert_kl
            + smoothness_weight * expected_smoothness
        )


        actor_loss.backward()


        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=5.0)


        actor_optimizer.step()




        soft_update(target_model, model, tau)


        target_model.eval()


        mean_expected_cost = float(torch.mean(expected_cost).item())
        lagrange_value = max(0.0, lagrange_value + lagrange_lr * (mean_expected_cost - cost_budget))


        row = {
            "epoch": epoch,
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "cost_loss": float(cost_loss.item()),
            "expected_cost": mean_expected_cost,
            "lagrange": float(lagrange_value),
            "expert_kl": float(torch.mean(expert_kl).item()),
            "smoothness": float(torch.mean(expected_smoothness).item()),
        }


        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x)["logits"]
                greedy_policy = masked_greedy_policy_from_logits(logits_eval, admissible)
                soft_policy = masked_softmax_from_logits(logits_eval, admissible)

            metrics = evaluate_policy_set(
                benchmark,
                policy_numpy(greedy_policy),
                soft_policy=policy_numpy(soft_policy),
            )

            row.update(metrics)


            if epoch == epochs:
                final_metrics = metrics


        history.append(row)


    model.eval()
    with torch.no_grad():
        logits_final = model(x)["logits"]
        final_policy = masked_greedy_policy_from_logits(logits_final, admissible)
        final_soft_policy = masked_softmax_from_logits(logits_final, admissible)


    if final_metrics is None:
        final_metrics = evaluate_policy_set(
            benchmark,
            policy_numpy(final_policy),
            soft_policy=policy_numpy(final_soft_policy),
        )


    seed_dir = os.path.join(results_dir, "laadan_ac", "seed_" + str(seed))
    save_model_run(seed_dir, model, history, final_metrics)


    return {
        "name": "LAADAN-AC",
        "seed": seed,
        "model": model,
        "policy": policy_numpy(final_policy),
        "analysis_policy": policy_numpy(final_soft_policy),
        "history": history,
        "metrics": final_metrics,
        "train_time_seconds": float(time.time() - start_time),
        "convergence_epoch_95": epoch_to_fraction_of_best(history, "survival_rate", 0.95),
    }


# aggregate seeds
def aggregate_seed_metrics(run_list):


    if not run_list:
        return {}


    metric_names = list(run_list[0]["metrics"].keys())


    summary = {}


    for metric_name in metric_names:
        values = np.asarray([run["metrics"][metric_name] for run in run_list], dtype=float)
        stats = mean_ci95(values)
        stats["min"] = float(np.min(values))
        stats["max"] = float(np.max(values))
        summary[metric_name] = stats


    times = np.asarray([run["train_time_seconds"] for run in run_list], dtype=float)
    time_stats = mean_ci95(times)
    time_stats["min"] = float(np.min(times))
    time_stats["max"] = float(np.max(times))
    summary["train_time_seconds"] = time_stats


    convergence_values = [
        run["convergence_epoch_95"]
        for run in run_list
        if run["convergence_epoch_95"] is not None
    ]
    if convergence_values:
        convergence_values = np.asarray(convergence_values, dtype=float)
        conv_stats = mean_ci95(convergence_values)
        conv_stats["min"] = float(np.min(convergence_values))
        conv_stats["max"] = float(np.max(convergence_values))
        summary["convergence_epoch_95"] = conv_stats


    return summary


# aggregate histories
def aggregate_histories(run_list, metric_names):


    if not run_list:
        return {}


    result = {}


    epochs = np.asarray([row["epoch"] for row in run_list[0]["history"]], dtype=int)
    result["epoch"] = epochs.tolist()


    for metric_name in metric_names:
        stacked = []


        for run in run_list:
            values = []
            for row in run["history"]:
                value = row.get(metric_name, np.nan)
                values.append(float(value) if np.isfinite(value) else np.nan)
            stacked.append(values)


        stacked = np.asarray(stacked, dtype=float)


        if stacked.size == 0:
            result[metric_name] = {
                "mean": [],
                "std": [],
                "ci95_half": [],
                "ci95_low": [],
                "ci95_high": [],
                "n": [],
            }
            continue


        mean_values = []
        std_values = []
        ci95_half_values = []
        ci95_low_values = []
        ci95_high_values = []
        n_values = []


        for column_index in range(stacked.shape[1]):
            stats = mean_ci95(stacked[:, column_index])
            mean_values.append(stats["mean"])
            std_values.append(stats["std"])
            ci95_half_values.append(stats["ci95_half"])
            ci95_low_values.append(stats["ci95_low"])
            ci95_high_values.append(stats["ci95_high"])
            n_values.append(stats["n"])


        result[metric_name] = {
            "mean": mean_values,
            "std": std_values,
            "ci95_half": ci95_half_values,
            "ci95_low": ci95_low_values,
            "ci95_high": ci95_high_values,
            "n": n_values,
        }


    return result


def best_policy_from_runs(run_list, use_analysis_policy=False):


    if not run_list:
        return None


    best_index = 0
    best_value = run_list[0]["metrics"]["survival_rate"]


    for index in range(1, len(run_list)):
        value = run_list[index]["metrics"]["survival_rate"]
        if value > best_value:
            best_value = value
            best_index = index


    if use_analysis_policy:
        return run_list[best_index].get("analysis_policy", run_list[best_index]["policy"])
    return run_list[best_index]["policy"]
