


import json

import os

import random

import numpy as np

import torch





# benchmark
# evaluation
class ICUSepsisOfflineBenchmark:


    def __init__(self, data_dir, horizon=20, device="cpu", use_one_hot_states=False):

        self.data_dir = data_dir

        self.horizon = int(horizon)

        self.device = device

        self.use_one_hot_states = bool(use_one_hot_states)



        self.transition = None



        self.reward_by_next_state = None


        self.initial_state_dist = None


        self.expert_policy = None



        self.admissible_mask = None


        self.state_features = None


        self.num_states = 0


        self.num_actions = 0


        self.feature_dim = 0




        self.death_state = 713
        self.survival_state = 714

        self.terminal_mask = None

        self._load_all()



        self._fix_terminal_indices_if_needed()


        self._build_terminal_mask()






        self.reward_sa = self.transition @ self.reward_by_next_state






        self.immediate_cost = 1.0 - self.admissible_mask.astype(np.float32)


        self.expert_safe = self._safe_expert_policy()


        self.expert_mean_action = np.sum(
            self.expert_safe * np.arange(self.num_actions, dtype=np.float32)[None, :],
            axis=1,
        )

        self.smoothness_cost = self._build_smoothness_cost()


        self.transition_t = torch.as_tensor(self.transition, dtype=torch.float32, device=self.device)


        self.reward_sa_t = torch.as_tensor(self.reward_sa, dtype=torch.float32, device=self.device)


        self.initial_state_dist_t = torch.as_tensor(self.initial_state_dist, dtype=torch.float32, device=self.device)


        self.admissible_mask_t = torch.as_tensor(
            self.admissible_mask.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )


        self.terminal_mask_t = torch.as_tensor(
            self.terminal_mask.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )


        self.state_features_t = torch.as_tensor(self.state_features, dtype=torch.float32, device=self.device)


        self.expert_policy_t = torch.as_tensor(self.expert_policy, dtype=torch.float32, device=self.device)


        self.expert_safe_t = torch.as_tensor(self.expert_safe, dtype=torch.float32, device=self.device)


        self.immediate_cost_t = torch.as_tensor(self.immediate_cost, dtype=torch.float32, device=self.device)


        self.smoothness_cost_t = torch.as_tensor(self.smoothness_cost, dtype=torch.float32, device=self.device)

    def _resolve_path(self, filename):

        direct = os.path.join(self.data_dir, filename)
        if os.path.exists(direct):
            return direct

        extra = os.path.join(self.data_dir, "extras", filename)
        if os.path.exists(extra):
            return extra

        return None

    def _load_all(self):

        transition_path = self._resolve_path("transitionFunction.csv")
        reward_path = self._resolve_path("rewardFunction.csv")
        initial_path = self._resolve_path("initialStateDistribution.csv")
        expert_path = self._resolve_path("expertPolicy.csv")

        admissible_path = self._resolve_path("admissibleActions.txt")

        centre_path = self._resolve_path("stateClusterCenters.csv")

        if transition_path is None or reward_path is None or initial_path is None or expert_path is None:
            raise FileNotFoundError(
                "Missing one or more required benchmark files: "
                "transitionFunction.csv, rewardFunction.csv, "
                "initialStateDistribution.csv, expertPolicy.csv"
            )

        transition_raw = np.loadtxt(transition_path, delimiter=",")

        reward_raw = np.asarray(np.loadtxt(reward_path, delimiter=","), dtype=np.float32).reshape(-1)

        initial_raw = np.asarray(np.loadtxt(initial_path, delimiter=","), dtype=np.float32).reshape(-1)

        expert_raw = np.asarray(np.loadtxt(expert_path, delimiter=","), dtype=np.float32)

        self.num_states = int(reward_raw.shape[0])

        self.num_actions = int(expert_raw.shape[1])

        self.transition = transition_raw.reshape(
            self.num_states,
            self.num_actions,
            self.num_states,
        ).astype(np.float32)


        self.reward_by_next_state = reward_raw.astype(np.float32)

        self.initial_state_dist = self._normalize_1d(initial_raw)

        self.expert_policy = self._row_normalize(expert_raw)

        self.transition[self.transition < 0.0] = 0.0

        row_sums = self.transition.sum(axis=2, keepdims=True)

        zero_rows = row_sums.squeeze(-1) <= 0.0

        safe_row_sums = row_sums.copy()
        safe_row_sums[safe_row_sums <= 0.0] = 1.0

        self.transition = self.transition / safe_row_sums

        for state_index in range(self.num_states):
            for action_index in range(self.num_actions):
                if zero_rows[state_index, action_index]:
                    self.transition[state_index, action_index] = 0.0
                    self.transition[state_index, action_index, state_index] = 1.0

        self.admissible_mask = np.zeros((self.num_states, self.num_actions), dtype=np.int8)


        if admissible_path is None:
            self.admissible_mask[self.expert_policy > 0.0] = 1

            for state_index in range(self.num_states):
                if np.sum(self.admissible_mask[state_index]) == 0:
                    best_action = int(np.argmax(self.expert_policy[state_index]))
                    self.admissible_mask[state_index, best_action] = 1

        else:

            with open(admissible_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]

            for state_index, line in enumerate(lines[1:]):
                if state_index >= self.num_states:
                    break

                for token in line.split():
                    action_index = int(token)
                    if 0 <= action_index < self.num_actions:
                        self.admissible_mask[state_index, action_index] = 1

            for state_index in range(self.num_states):
                if np.sum(self.admissible_mask[state_index]) == 0:
                    best_action = int(np.argmax(self.expert_policy[state_index]))
                    self.admissible_mask[state_index, best_action] = 1

        if self.use_one_hot_states:
            self.state_features = self._build_one_hot_features()
            self.feature_dim = int(self.state_features.shape[1])

        else:
            if centre_path is None:
                raise FileNotFoundError(
                    "stateClusterCenters.csv was not found. "
                    "The offline neural version needs the released state-centre features."
                )

            centres = np.loadtxt(centre_path, delimiter=",")
            centres = np.asarray(centres, dtype=np.float32)

            if centres.shape[0] != self.num_states:
                raise ValueError(
                    "stateClusterCenters.csv does not match the number of states in the benchmark."
                )

            self.state_features = self._standardize_features(centres)

            self.feature_dim = int(self.state_features.shape[1])

    def _normalize_1d(self, values):

        values = np.asarray(values, dtype=np.float32)

        values = np.maximum(values, 0.0)

        total = float(np.sum(values))

        if total <= 0.0:

            values[:] = 1.0 / float(max(1, values.shape[0]))
        else:

            values /= total

        return values

    def _row_normalize(self, values):

        values = np.asarray(values, dtype=np.float32)

        values = np.maximum(values, 0.0)

        row_sums = np.sum(values, axis=1, keepdims=True)

        zero_rows = row_sums.squeeze(-1) <= 0.0

        safe = row_sums.copy()
        safe[safe <= 0.0] = 1.0

        values = values / safe


        for row_index in range(values.shape[0]):
            if zero_rows[row_index]:
                values[row_index] = 1.0 / float(values.shape[1])

        return values.astype(np.float32)

    def _standardize_features(self, values):



        mean = np.mean(values, axis=0, keepdims=True)


        std = np.std(values, axis=0, keepdims=True)


        std[std < 1e-6] = 1.0


        return ((values - mean) / std).astype(np.float32)

    def _build_one_hot_features(self):


        return np.eye(self.num_states, dtype=np.float32)

    def _fix_terminal_indices_if_needed(self):

        if self.death_state < self.num_states and self.survival_state < self.num_states:
            return

        import json as _json
        import os as _os

        metadata_candidates = [
            _os.path.join(self.data_dir, "extras", "eicu_demo_mdp_description.json"),
            _os.path.join(self.data_dir, "extras", "benchmark_description.json"),
            _os.path.join(self.data_dir, "benchmark_description.json"),
        ]

        for metadata_path in metadata_candidates:
            if not _os.path.exists(metadata_path):
                continue

            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = _json.load(handle)

            if "death_state" in metadata and "survival_state" in metadata:
                self.death_state = int(metadata["death_state"])
                self.survival_state = int(metadata["survival_state"])
                return

        self.survival_state = int(np.argmax(self.reward_by_next_state))

        candidate_death = self.num_states - 2
        if candidate_death == self.survival_state:
            candidate_death = self.num_states - 1

        self.death_state = int(candidate_death)

    def _build_terminal_mask(self):

        self.terminal_mask = np.zeros(self.num_states, dtype=np.int8)
        self.terminal_mask[self.death_state] = 1
        self.terminal_mask[self.survival_state] = 1

    def _safe_expert_policy(self):


        safe = self.expert_policy * self.admissible_mask.astype(np.float32)
        return self._row_normalize(safe)

    def _build_smoothness_cost(self):


        action_ids = np.arange(self.num_actions, dtype=np.float32)[None, :]

        target = self.expert_mean_action[:, None]

        cost = np.abs(action_ids - target) / float(max(1, self.num_actions - 1))

        return cost.astype(np.float32)

    def set_seed(self, seed):

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def masked_logits(self, logits, admissible_mask_t=None):

        if admissible_mask_t is None:
            admissible_mask_t = self.admissible_mask_t

        big_negative = torch.full_like(logits, -1e9)

        return torch.where(admissible_mask_t > 0.5, logits, big_negative)

    def policy_from_logits(self, logits, admissible_mask_t=None):

        masked = self.masked_logits(logits, admissible_mask_t)


        return torch.softmax(masked, dim=1)

    def greedy_policy_from_q(self, q_values):

        q_values = np.asarray(q_values, dtype=np.float32).copy()


        q_values[self.admissible_mask == 0] = -1e9


        policy = np.zeros((self.num_states, self.num_actions), dtype=np.float32)


        best_actions = np.argmax(q_values, axis=1)


        policy[np.arange(self.num_states), best_actions] = 1.0

        return policy

    def greedy_policy_from_q_unmasked(self, q_values):


        q_values = np.asarray(q_values, dtype=np.float32)


        policy = np.zeros((self.num_states, self.num_actions), dtype=np.float32)


        best_actions = np.argmax(q_values, axis=1)


        policy[np.arange(self.num_states), best_actions] = 1.0

        return policy

    def random_admissible_policy(self):

        policy = self.admissible_mask.astype(np.float32)

        row_sums = np.sum(policy, axis=1, keepdims=True)

        row_sums[row_sums <= 0.0] = 1.0

        return policy / row_sums

    def optimal_policy(self, gamma=1.0, horizon=None):

        if horizon is None:
            horizon = self.horizon

        horizon = int(horizon)

        value = np.zeros(self.num_states, dtype=np.float32)

        q_values = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        mask = 1.0 - self.terminal_mask.astype(np.float32)


        for _ in range(horizon):

            q_values = self.reward_sa + gamma * np.einsum("san,n->sa", self.transition, value)

            q_values[self.admissible_mask == 0] = -1e9

            value = mask * np.max(q_values, axis=1)

        policy = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        best_actions = np.argmax(q_values, axis=1)
        policy[np.arange(self.num_states), best_actions] = 1.0

        return policy

    def reference_policies(self, gamma=1.0, horizon=None):

        return {
            "Random Policy": self.random_admissible_policy(),
            "Expert Policy": self.expert_safe.copy(),
            "Optimal Policy": self.optimal_policy(gamma=gamma, horizon=horizon),
        }

    def reference_metrics(self, gamma=1.0, horizon=None):

        payload = {}


        for name, policy in self.reference_policies(gamma=gamma, horizon=horizon).items():
            payload[name] = self.exact_policy_evaluation(policy, horizon=horizon)

        return payload

    def exact_policy_evaluation(self, policy, horizon=None):

        if horizon is None:
            horizon = self.horizon

        horizon = int(horizon)

        policy = np.asarray(policy, dtype=np.float32)

        p_pi = np.einsum("sa,san->sn", policy, self.transition)

        immediate_reward = np.sum(policy * self.reward_sa, axis=1)


        immediate_survival = p_pi[:, self.survival_state]


        immediate_cost = np.sum(policy * self.immediate_cost, axis=1)


        immediate_length = 1.0 - self.terminal_mask.astype(np.float32)


        value = np.zeros(self.num_states, dtype=np.float32)
        survival = np.zeros(self.num_states, dtype=np.float32)
        length = np.zeros(self.num_states, dtype=np.float32)
        cost = np.zeros(self.num_states, dtype=np.float32)


        mask = 1.0 - self.terminal_mask.astype(np.float32)


        for _ in range(horizon):
            value = mask * (immediate_reward + p_pi @ value)
            survival = mask * (immediate_survival + p_pi @ survival)
            length = mask * (immediate_length + p_pi @ length)
            cost = mask * (immediate_cost + p_pi @ cost)


        avg_return = float(self.initial_state_dist @ value)
        survival_rate = float(self.initial_state_dist @ survival)
        avg_length = float(self.initial_state_dist @ length)
        expected_cost = float(self.initial_state_dist @ cost)


        inadmissibility_rate = 0.0 if avg_length <= 1e-12 else expected_cost / avg_length


        mortality_rate = 1.0 - survival_rate


        state_weights = self.initial_state_dist / max(1e-12, float(np.sum(self.initial_state_dist)))


        expert_argmax = np.argmax(self.expert_safe, axis=1)
        model_argmax = np.argmax(policy, axis=1)
        argmax_match = float(np.sum(state_weights * (expert_argmax == model_argmax).astype(np.float32)))


        kl = np.sum(
            state_weights[:, None]
            * self.expert_safe
            * (np.log(self.expert_safe + 1e-8) - np.log(policy + 1e-8)),
            axis=1,
        )
        mean_kl_to_expert = float(np.sum(kl))


        entropy = -np.sum(state_weights[:, None] * policy * np.log(policy + 1e-8))


        expected_action = np.sum(policy * np.arange(self.num_actions, dtype=np.float32)[None, :], axis=1)


        action_deviation_from_expert = float(
            np.sum(state_weights * np.abs(expected_action - self.expert_mean_action))
        )

        return {
            "avg_return": avg_return,
            "survival_rate": survival_rate,
            "mortality_rate": mortality_rate,
            "avg_length": avg_length,
            "inadmissibility_rate": float(inadmissibility_rate),
            "expert_argmax_match": argmax_match,
            "mean_kl_to_expert": mean_kl_to_expert,
            "policy_entropy": float(entropy),
            "mean_action_deviation_from_expert": action_deviation_from_expert,
        }

    def simulate_policy(self, policy, num_episodes=2000, seed=0):

        rng = np.random.RandomState(seed)

        policy = np.asarray(policy, dtype=np.float32)

        action_hist = np.zeros(self.num_actions, dtype=np.int64)

        jump_values = []

        episodic_returns = []
        episodic_survival = []
        episodic_cost = []
        episodic_length = []

        for _ in range(int(num_episodes)):

            state = int(rng.choice(self.num_states, p=self.initial_state_dist))


            previous_action = None


            total_reward = 0.0
            total_cost = 0.0
            length = 0


            for _step in range(self.horizon):

                action = int(rng.choice(self.num_actions, p=policy[state]))

                action_hist[action] += 1

                if previous_action is not None:
                    jump_values.append(abs(action - previous_action))

                previous_action = action

                next_state = int(rng.choice(self.num_states, p=self.transition[state, action]))

                reward = float(self.reward_by_next_state[next_state])

                cost = float(self.immediate_cost[state, action])

                total_reward += reward
                total_cost += cost
                length += 1

                state = next_state

                if self.terminal_mask[state] == 1:
                    break

            episodic_returns.append(total_reward)

            episodic_survival.append(1.0 if state == self.survival_state else 0.0)

            episodic_cost.append(0.0 if length == 0 else total_cost / float(length))

            episodic_length.append(float(length))

        action_hist = action_hist.astype(np.float64)
        if np.sum(action_hist) > 0.0:
            action_hist = action_hist / np.sum(action_hist)

        return {
            "return_mean": float(np.mean(episodic_returns)),
            "survival_mean": float(np.mean(episodic_survival)),
            "inadmissibility_mean": float(np.mean(episodic_cost)),
            "length_mean": float(np.mean(episodic_length)),
            "mean_action_jump": float(np.mean(jump_values)) if jump_values else 0.0,
            "action_histogram": action_hist.tolist(),
        }

    def save_benchmark_description(self, path):

        payload = {
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "feature_dim": self.feature_dim,
            "death_state": self.death_state,
            "survival_state": self.survival_state,
            "horizon": self.horizon,
            "use_one_hot_states": self.use_one_hot_states,
            "state_representation": (
                "one_hot_state_id" if self.use_one_hot_states else "standardized_cluster_centres"
            ),
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
