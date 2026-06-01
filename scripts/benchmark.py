
# benchmark.py

# Offline loader and exact evaluator for the ICU-Sepsis benchmark.

# These are the libraries used
# for saving benchmark descriptions as JSON.
import json

# for filesystem path handling.
import os

# for Python-level random seed control.
import random

# for table loading, vectorised MDP calculations, and simulation.
import numpy as np

# PyTorch is used to create tensor copies of the benchmark data for training.
import torch


class ICUSepsisOfflineBenchmark:
    
    """
    Offline benchmark loader for ICU-Sepsis.

    Conceptually, the released ICU-Sepsis benchmark is a tabular Markov Decision
    Process (MDP). However, the release also provides continuous state-cluster
    centre vectors, which allows this project to use neural networks on top
    of the fixed benchmark dynamics.

    This class therefore supports two possible state representations:
    1. one-hot state identity vectors
    2. released continuous state-centre features

    The class is responsible for:
    - loading transition, reward, initial-state, expert-policy, and admissibility
      data from the benchmark files
    - storing them in NumPy form for exact evaluation
    - storing them again in PyTorch tensor form for model training
    - providing exact finite-horizon evaluation of learned policies
    - providing reference policies such as random, expert, and optimal
    """

    def __init__(self, data_dir, horizon=20, device="cpu", use_one_hot_states=False):
        
        """
        Building the benchmark object.

        Parameters
        ----------
        data_dir : str
            Folder containing the ICU-Sepsis benchmark files.
        horizon : int, default=20
            Finite evaluation horizon used for exact dynamic-programming-style
            policy evaluation and Monte Carlo rollouts.
        device : str, default="cpu"
            Device name used when creating PyTorch tensors, for instance "cpu" or "cuda".
        use_one_hot_states : bool, default=False
            If True, represent each discrete state as a one-hot vector.
            If False, use the released continuous state-cluster centre features.
        """

        # Storing the path to the benchmark data folder.
        self.data_dir = data_dir

        # Storing the evaluation horizon as an integer.
        self.horizon = int(horizon)

        # Storing the requested tensor device.
        self.device = device

        # Storing whether state representation should be one-hot or continuous.
        self.use_one_hot_states = bool(use_one_hot_states)

        # Transition tensor p(s, a, s').
        # Will later have shape [num_states, num_actions, num_states].
        self.transition = None

        # Reward assigned by next state.
        # In this benchmark, reward depends only on the state transitioned into.
        self.reward_by_next_state = None

        # Initial-state distribution d0(s).
        self.initial_state_dist = None

        # Released expert policy from the benchmark.
        self.expert_policy = None

        # Binary admissibility mask indicating which actions are considered
        # supported / admissible in each state.
        self.admissible_mask = None

        # State feature matrix used as neural-network input.
        self.state_features = None

        # Number of states in the benchmark MDP.
        self.num_states = 0

        # Number of discrete actions in the benchmark MDP.
        self.num_actions = 0

        # Dimensionality of the chosen state feature representation.
        self.feature_dim = 0

        # Default terminal-state indices used by the released ICU-Sepsis files.
        # These are kept as defaults, but checked later in case a future release
        # changes the exact terminal-state numbering.
        self.death_state = 713
        self.survival_state = 714

        # Binary mask indicating whether each state is terminal.
        self.terminal_mask = None

        # Loading all benchmark files from disk and preparing NumPy arrays.
        self._load_all()

        # If terminal indices ever change in another benchmark release, recover
        # them from the reward structure.
        self._fix_terminal_indices_if_needed()

        # Building terminal-state mask after terminal indices are known.
        self._build_terminal_mask()

        # Precomputing expected immediate reward for each (state, action) pair.
        #
        # transition has shape [S, A, S]
        # reward_by_next_state has shape [S]
        #
        # The matrix multiplication below produces reward_sa[s, a] =
        # sum_{s'} p(s, a, s') * r(s').
        self.reward_sa = self.transition @ self.reward_by_next_state

        # Define an immediate cost table:
        # admissible actions -> cost 0
        # inadmissible actions -> cost 1
        
        # This is used later for safety-related analysis and constraints.
        self.immediate_cost = 1.0 - self.admissible_mask.astype(np.float32)

        # Filter the expert policy through admissibility, then renormalise it.
        # This gives an expert policy that only places mass on admissible actions.
        self.expert_safe = self._safe_expert_policy()

        # Computing the expert-supported mean action index per state.
        # This is used to build a simple smoothness-style action penalty later.
        self.expert_mean_action = np.sum(
            self.expert_safe * np.arange(self.num_actions, dtype=np.float32)[None, :],
            axis=1,
        )

        # Building a benchmark-level smoothness proxy cost.
        # This is not a true trajectory-level dose-jump penalty, but a static
        # penalty based on how far an action is from the expert-supported mean.
        self.smoothness_cost = self._build_smoothness_cost()


        # Create PyTorch tensor copies of the benchmark data.
        # These are useful because model training code expects tensors on the
        # chosen device (CPU or GPU).


        # Transition tensor in PyTorch form.
        self.transition_t = torch.as_tensor(self.transition, dtype=torch.float32, device=self.device)

        # Expected immediate reward table in PyTorch form.
        self.reward_sa_t = torch.as_tensor(self.reward_sa, dtype=torch.float32, device=self.device)

        # Initial-state distribution in PyTorch form.
        self.initial_state_dist_t = torch.as_tensor(self.initial_state_dist, dtype=torch.float32, device=self.device)

        # Admissibility mask in PyTorch form.
        self.admissible_mask_t = torch.as_tensor(
            self.admissible_mask.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        # Terminal-state mask in PyTorch form.
        self.terminal_mask_t = torch.as_tensor(
            self.terminal_mask.astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        # State features in PyTorch form.
        self.state_features_t = torch.as_tensor(self.state_features, dtype=torch.float32, device=self.device)

        # Original expert policy in PyTorch form.
        self.expert_policy_t = torch.as_tensor(self.expert_policy, dtype=torch.float32, device=self.device)

        # Admissibility-filtered expert policy in PyTorch form.
        self.expert_safe_t = torch.as_tensor(self.expert_safe, dtype=torch.float32, device=self.device)

        # Immediate inadmissibility cost table in PyTorch form.
        self.immediate_cost_t = torch.as_tensor(self.immediate_cost, dtype=torch.float32, device=self.device)

        # Smoothness proxy cost table in PyTorch form.
        self.smoothness_cost_t = torch.as_tensor(self.smoothness_cost, dtype=torch.float32, device=self.device)

    def _resolve_path(self, filename):
        
        """
        Find a benchmark file either in the root folder or in an extras folder.

        Some ICU-Sepsis releases store files directly in data_dir, while others
        may store some files in data_dir/extras. This helper checks both places.

        Parameters
        ----------
        filename : str
            Name of the file to search for.

        Returns
        -------
        str or None
            Full path if found, otherwise None.
        """

        # Checking if the file exists directly inside the benchmark folder.
        direct = os.path.join(self.data_dir, filename)
        if os.path.exists(direct):
            return direct

        # If not found directly, check inside an extras subfolder.
        extra = os.path.join(self.data_dir, "extras", filename)
        if os.path.exists(extra):
            return extra

        # If neither exists, return None so the caller can decide what to do.
        return None

    def _load_all(self):
        
        """
        Loading all required CSV and text files from the benchmark release.

        This method is responsible for:
        - resolving benchmark file paths
        - loading transition/reward/initial/expert tables
        - cleaning numeric issues in the transition tensor
        - building the admissibility mask
        - building the chosen state-feature representation
        """

        # Resolving paths to the required benchmark files.
        transition_path = self._resolve_path("transitionFunction.csv")
        reward_path = self._resolve_path("rewardFunction.csv")
        initial_path = self._resolve_path("initialStateDistribution.csv")
        expert_path = self._resolve_path("expertPolicy.csv")

        # Optional path: admissible action list by state.
        admissible_path = self._resolve_path("admissibleActions.txt")

        # Optional path: released continuous state-cluster centre features.
        centre_path = self._resolve_path("stateClusterCenters.csv")

        # These four files are essential for the benchmark to function.
        if transition_path is None or reward_path is None or initial_path is None or expert_path is None:
            raise FileNotFoundError(
                "Missing one or more required benchmark files: "
                "transitionFunction.csv, rewardFunction.csv, "
                "initialStateDistribution.csv, expertPolicy.csv"
            )

        # Loading the flattened transition tensor from CSV.
        transition_raw = np.loadtxt(transition_path, delimiter=",")

        # Loading reward vector and flattening it to shape [S].
        reward_raw = np.asarray(np.loadtxt(reward_path, delimiter=","), dtype=np.float32).reshape(-1)

        # Loading initial-state distribution and flattening it to shape [S].
        initial_raw = np.asarray(np.loadtxt(initial_path, delimiter=","), dtype=np.float32).reshape(-1)

        # Loading expert policy matrix of shape [S, A].
        expert_raw = np.asarray(np.loadtxt(expert_path, delimiter=","), dtype=np.float32)

        # Infer number of states from reward vector length.
        self.num_states = int(reward_raw.shape[0])

        # Infer number of actions from expert-policy width.
        self.num_actions = int(expert_raw.shape[1])

        # Reshaping flattened transition table into [S, A, S].
        self.transition = transition_raw.reshape(
            self.num_states,
            self.num_actions,
            self.num_states,
        ).astype(np.float32)

        # Storing reward-by-next-state vector.
        self.reward_by_next_state = reward_raw.astype(np.float32)

        # Normalising initial-state distribution so it sums to 1.
        self.initial_state_dist = self._normalize_1d(initial_raw)

        # Row-normalising expert policy so each state row sums to 1.
        self.expert_policy = self._row_normalize(expert_raw)


        # Cleaning any numeric issues in the transition tensor.
      

        # Negative probabilities are invalid, so clip them to zero.
        self.transition[self.transition < 0.0] = 0.0

        # Computing per-(state, action) row sums over next-state dimension.
        row_sums = self.transition.sum(axis=2, keepdims=True)

        # Identifying any rows whose probability mass is zero.
        zero_rows = row_sums.squeeze(-1) <= 0.0

        # Building a safe copy for division, replacing zeros with ones.
        safe_row_sums = row_sums.copy()
        safe_row_sums[safe_row_sums <= 0.0] = 1.0

        # Renormalising transition rows so valid rows sum to 1.
        self.transition = self.transition / safe_row_sums

        # For any completely empty row, fall back to a deterministic self-loop.
        # This prevents invalid distributions and keeps the tensor usable.
        for state_index in range(self.num_states):
            for action_index in range(self.num_actions):
                if zero_rows[state_index, action_index]:
                    self.transition[state_index, action_index] = 0.0
                    self.transition[state_index, action_index, state_index] = 1.0


        # Building the admissibility mask.
        # Shape: [num_states, num_actions]
        # Value 1 = admissible / supported
        # Value 0 = inadmissible / unsupported


        self.admissible_mask = np.zeros((self.num_states, self.num_actions), dtype=np.int8)

        # If no admissibility file exists, infer support from non-zero expert mass.
        if admissible_path is None:
            self.admissible_mask[self.expert_policy > 0.0] = 1

            # If a row still has no admissible action, force the expert argmax to
            # be admissible so every state has at least one available action.
            for state_index in range(self.num_states):
                if np.sum(self.admissible_mask[state_index]) == 0:
                    best_action = int(np.argmax(self.expert_policy[state_index]))
                    self.admissible_mask[state_index, best_action] = 1

        else:
            # If admissibility file exists, parse each state's admissible actions.
            with open(admissible_path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle.readlines() if line.strip()]

            # Skipping the first line, which is typically a header in this file format.
            for state_index, line in enumerate(lines[1:]):
                if state_index >= self.num_states:
                    break

                # Each token is an action index admissible in the current state.
                for token in line.split():
                    action_index = int(token)
                    if 0 <= action_index < self.num_actions:
                        self.admissible_mask[state_index, action_index] = 1

            # Again ensuring every state has at least one admissible action.
            for state_index in range(self.num_states):
                if np.sum(self.admissible_mask[state_index]) == 0:
                    best_action = int(np.argmax(self.expert_policy[state_index]))
                    self.admissible_mask[state_index, best_action] = 1


        # Building the chosen state representation.

        # Option 1: exact one-hot state identity.
        if self.use_one_hot_states:
            self.state_features = self._build_one_hot_features()
            self.feature_dim = int(self.state_features.shape[1])

        # Option 2: released continuous state-cluster centre vectors.
        else:
            if centre_path is None:
                raise FileNotFoundError(
                    "stateClusterCenters.csv was not found. "
                    "The offline neural version needs the released state-centre features."
                )

            # Loading state-cluster centre table.
            centres = np.loadtxt(centre_path, delimiter=",")
            centres = np.asarray(centres, dtype=np.float32)

            # Ensuring number of feature rows matches number of benchmark states.
            if centres.shape[0] != self.num_states:
                raise ValueError(
                    "stateClusterCenters.csv does not match the number of states in the benchmark."
                )

            # Standardising continuous features before using them in neural models.
            self.state_features = self._standardize_features(centres)

            # Storing feature dimension.
            self.feature_dim = int(self.state_features.shape[1])

    def _normalize_1d(self, values):
        
        """
        Normalizing a non-negative 1D vector so that it sums to 1.

        This is used for the initial-state distribution.
        """

        # Converting input to float32 NumPy array.
        values = np.asarray(values, dtype=np.float32)

        # Clipping any negative numeric noise to zero.
        values = np.maximum(values, 0.0)

        # Computing total mass.
        total = float(np.sum(values))

        # If the vector is empty / invalid, fall back to a uniform distribution.
        if total <= 0.0:
            values[:] = 1.0 / float(max(1, values.shape[0]))
        else:
            # Otherwise normalise normally.
            values /= total

        return values

    def _row_normalize(self, values):
        
        """
        Normalizing each row of a 2D array so that row sums equal 1.

        This is used for expert-policy tables and admissibility-filtered policies.
        """

        # Converting input to float32 NumPy array.
        values = np.asarray(values, dtype=np.float32)

        # Removing negative numeric noise.
        values = np.maximum(values, 0.0)

        # Computing row sums.
        row_sums = np.sum(values, axis=1, keepdims=True)

        # Identifying rows that sum to zero.
        zero_rows = row_sums.squeeze(-1) <= 0.0

        # Safe copying for division.
        safe = row_sums.copy()
        safe[safe <= 0.0] = 1.0

        # Dividing row-wise.
        values = values / safe

        # Replacing any zero row with a uniform distribution.
        for row_index in range(values.shape[0]):
            if zero_rows[row_index]:
                values[row_index] = 1.0 / float(values.shape[1])

        return values.astype(np.float32)

    def _standardize_features(self, values):
        
        """
        Standardizing continuous feature columns to zero mean and unit variance.

        This is standard preprocessing for neural-network inputs.
        """

        # Computing feature-wise mean.
        mean = np.mean(values, axis=0, keepdims=True)

        # Computing feature-wise standard deviation.
        std = np.std(values, axis=0, keepdims=True)

        # Preventing division by very small values.
        std[std < 1e-6] = 1.0

        # Returning standardised features.
        return ((values - mean) / std).astype(np.float32)

    def _build_one_hot_features(self):
        
        """
        Building exact one-hot state identity features.

        Each state becomes a vector of length num_states with a single 1.
        """
        return np.eye(self.num_states, dtype=np.float32)

    def _fix_terminal_indices_if_needed(self):
        
        """
        Recover terminal-state indices if the dataset is not the original ICU-Sepsis release.

        The original ICU-Sepsis benchmark uses death_state=713 and survival_state=714.
        Smaller portability MDPs, such as the eICU demo MDP, store terminal indices
        in a metadata file. Metadata is used first because reward-based inference is
        ambiguous when intermediate states and death both have reward 0.
        """
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
        
        """
        Building a binary terminal-state mask.

        terminal_mask[s] = 1 if state s is terminal, else 0.
        """
        self.terminal_mask = np.zeros(self.num_states, dtype=np.int8)
        self.terminal_mask[self.death_state] = 1
        self.terminal_mask[self.survival_state] = 1

    def _safe_expert_policy(self):
        
        """
        Filtering the expert policy through the admissibility mask and renormalising it.

        This ensures expert comparisons only use supported benchmark actions.
        """
        safe = self.expert_policy * self.admissible_mask.astype(np.float32)
        return self._row_normalize(safe)

    def _build_smoothness_cost(self):
        
        """
        Building a simple state-action smoothness proxy.

        This is not a full trajectory-based dose-jump penalty. Instead it assigns
        higher penalty to actions that are farther from the expert-supported mean
        action in that state.
        """

        # Action IDs laid out as one row: [0, 1, ..., A-1].
        action_ids = np.arange(self.num_actions, dtype=np.float32)[None, :]

        # Expert mean action per state as a column vector.
        target = self.expert_mean_action[:, None]

        # Absolute distance from expert mean action, scaled to roughly [0, 1].
        cost = np.abs(action_ids - target) / float(max(1, self.num_actions - 1))

        return cost.astype(np.float32)

    def set_seed(self, seed):
        
        """
        Setting Python, NumPy, and PyTorch seeds for reproducibility.

        This is called before training runs so that results are repeatable.
        """

        # Python random seed.
        random.seed(seed)

        # NumPy random seed.
        np.random.seed(seed)

        # PyTorch CPU seed.
        torch.manual_seed(seed)

        # PyTorch CUDA seed for all GPUs if CUDA exists.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Make cuDNN more deterministic where supported.
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def masked_logits(self, logits, admissible_mask_t=None):
        
        """
        Applying a large negative value to inadmissible actions before softmax.

        This effectively removes inadmissible actions from the policy
        distribution because exp(-1e9) is numerically negligible.
        """

        # If caller did not provide a mask, use the benchmark's own mask.
        if admissible_mask_t is None:
            admissible_mask_t = self.admissible_mask_t

        # Large negative constant for blocked actions.
        big_negative = torch.full_like(logits, -1e9)

        # Keeping admissible logits, replace inadmissible ones with huge negative.
        return torch.where(admissible_mask_t > 0.5, logits, big_negative)

    def policy_from_logits(self, logits, admissible_mask_t=None):
        
        """
        Converting logits into a masked probability distribution over actions.
        """

        # First hide inadmissible actions.
        masked = self.masked_logits(logits, admissible_mask_t)

        # Then applying softmax to obtain a valid policy distribution.
        return torch.softmax(masked, dim=1)

    def greedy_policy_from_q(self, q_values):
        
        """
        Building a deterministic one-hot policy from Q-values with admissibility masking.

        The selected action is the highest-valued admissible action in each state.
        """

        # Copying incoming values so the original array is not modified in place.
        q_values = np.asarray(q_values, dtype=np.float32).copy()

        # Forcing inadmissible actions to a huge negative score so they cannot win.
        q_values[self.admissible_mask == 0] = -1e9

        # Initialising one-hot policy matrix.
        policy = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        # Best action per state.
        best_actions = np.argmax(q_values, axis=1)

        # Putting probability 1.0 on the chosen action in each state.
        policy[np.arange(self.num_states), best_actions] = 1.0

        return policy

    def greedy_policy_from_q_unmasked(self, q_values):
        
        """
        Building a deterministic one-hot policy from raw Q-values without masking.

        This is used for unguided baselines such as CQL in the current setup.
        """

        # Converting to NumPy array.
        q_values = np.asarray(q_values, dtype=np.float32)

        # Initialising one-hot policy matrix.
        policy = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        # Choosing highest-valued action in each state.
        best_actions = np.argmax(q_values, axis=1)

        # Putting probability 1.0 on the chosen action.
        policy[np.arange(self.num_states), best_actions] = 1.0

        return policy

    def random_admissible_policy(self):
        
        """
        Building a uniform random policy over admissible actions only.

        This is a more benchmark-faithful random baseline than uniform over all
        25 actions, because unsupported actions are not treated as equally valid.
        """

        # Starting with the binary admissibility mask.
        policy = self.admissible_mask.astype(np.float32)

        # Counting admissible actions in each state.
        row_sums = np.sum(policy, axis=1, keepdims=True)

        # Protecting against zero division.
        row_sums[row_sums <= 0.0] = 1.0

        # Converting mask into uniform probabilities over admissible actions.
        return policy / row_sums

    def optimal_policy(self, gamma=1.0, horizon=None):
        
        """
        Computing finite-horizon admissibility-aware optimal policy by backward DP.

        This is not a learned model. It is a benchmark reference upper bound for
        the released MDP under the chosen horizon.
        """

        # Using the object's default horizon unless caller overrides it.
        if horizon is None:
            horizon = self.horizon

        # Ensuring integer horizon.
        horizon = int(horizon)

        # Initialising value function V(s) = 0.
        value = np.zeros(self.num_states, dtype=np.float32)

        # Storaging for Q(s, a).
        q_values = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

        # Non-terminal mask: 1 for non-terminal states, 0 for terminal ones.
        mask = 1.0 - self.terminal_mask.astype(np.float32)

        # Finite-horizon backward updates.
        for _ in range(horizon):
            # One-step Bellman backup.
            q_values = self.reward_sa + gamma * np.einsum("san,n->sa", self.transition, value)

            # Inadmissible actions are blocked.
            q_values[self.admissible_mask == 0] = -1e9

            # Greedy value update on non-terminal states only.
            value = mask * np.max(q_values, axis=1)

        # Converting final Q-values into deterministic greedy policy.
        policy = np.zeros((self.num_states, self.num_actions), dtype=np.float32)
        best_actions = np.argmax(q_values, axis=1)
        policy[np.arange(self.num_states), best_actions] = 1.0

        return policy

    def reference_policies(self, gamma=1.0, horizon=None):
        
        """
        Returning the benchmark reference policies used for comparison.

        These provide context for the learned models.
        """
        return {
            "Random Policy": self.random_admissible_policy(),
            "Expert Policy": self.expert_safe.copy(),
            "Optimal Policy": self.optimal_policy(gamma=gamma, horizon=horizon),
        }

    def reference_metrics(self, gamma=1.0, horizon=None):
        
        """
        Evaluating all benchmark reference policies under the same exact evaluator.
        """

        # Outputting dictionary keyed by policy name.
        payload = {}

        # Evaluating each reference policy exactly.
        for name, policy in self.reference_policies(gamma=gamma, horizon=horizon).items():
            payload[name] = self.exact_policy_evaluation(policy, horizon=horizon)

        return payload

    def exact_policy_evaluation(self, policy, horizon=None):
        
        """
        Exact finite-horizon policy evaluation under the released benchmark model.

        This is one of the most important methods in the whole file because it
        provides deterministic, reproducible evaluation of a policy on the known
        MDP rather than relying only on Monte Carlo sampling.

        Returned metrics include:
        - expected return
        - survival rate
        - mortality rate
        - expected episode length
        - inadmissibility rate
        - alignment with expert argmax
        - KL divergence to expert policy
        - policy entropy
        - expected action deviation from expert mean action
        """

        # Using default horizon unless caller overrides.
        if horizon is None:
            horizon = self.horizon

        # Ensuring integer horizon.
        horizon = int(horizon)

        # Converting incoming policy to NumPy float32 array.
        policy = np.asarray(policy, dtype=np.float32)

        # Computing transition matrix induced by the policy:
        # p_pi[s, s'] = sum_a pi(a|s) p(s, a, s')
        p_pi = np.einsum("sa,san->sn", policy, self.transition)

        # Expected one-step reward under the policy in each state.
        immediate_reward = np.sum(policy * self.reward_sa, axis=1)

        # One-step probability of transitioning directly into survival state.
        immediate_survival = p_pi[:, self.survival_state]

        # Expected one-step inadmissibility cost under the policy.
        immediate_cost = np.sum(policy * self.immediate_cost, axis=1)

        # Non-terminal length contribution: 1 step for non-terminal states, 0 for terminal.
        immediate_length = 1.0 - self.terminal_mask.astype(np.float32)

        # Initialising finite-horizon value-like vectors.
        value = np.zeros(self.num_states, dtype=np.float32)
        survival = np.zeros(self.num_states, dtype=np.float32)
        length = np.zeros(self.num_states, dtype=np.float32)
        cost = np.zeros(self.num_states, dtype=np.float32)

        # Non-terminal mask again, used to stop backup through terminal states.
        mask = 1.0 - self.terminal_mask.astype(np.float32)

        # Finite-horizon backward recursion.
        for _ in range(horizon):
            value = mask * (immediate_reward + p_pi @ value)
            survival = mask * (immediate_survival + p_pi @ survival)
            length = mask * (immediate_length + p_pi @ length)
            cost = mask * (immediate_cost + p_pi @ cost)

        # Aggregating state-wise values under initial-state distribution.
        avg_return = float(self.initial_state_dist @ value)
        survival_rate = float(self.initial_state_dist @ survival)
        avg_length = float(self.initial_state_dist @ length)
        expected_cost = float(self.initial_state_dist @ cost)

        # Defining inadmissibility rate as expected cost per expected step.
        inadmissibility_rate = 0.0 if avg_length <= 1e-12 else expected_cost / avg_length

        # Mortality is 1 minus survival.
        mortality_rate = 1.0 - survival_rate

        # Normalised state weights for policy-analysis metrics.
        state_weights = self.initial_state_dist / max(1e-12, float(np.sum(self.initial_state_dist)))

        # Comparing greedy action of model vs greedy action of safe expert.
        expert_argmax = np.argmax(self.expert_safe, axis=1)
        model_argmax = np.argmax(policy, axis=1)
        argmax_match = float(np.sum(state_weights * (expert_argmax == model_argmax).astype(np.float32)))

        # Computing weighted KL(expert_safe || policy) per state, then average.
        kl = np.sum(
            state_weights[:, None]
            * self.expert_safe
            * (np.log(self.expert_safe + 1e-8) - np.log(policy + 1e-8)),
            axis=1,
        )
        mean_kl_to_expert = float(np.sum(kl))

        # Computing weighted policy entropy.
        entropy = -np.sum(state_weights[:, None] * policy * np.log(policy + 1e-8))

        # Computing expected action index in each state.
        expected_action = np.sum(policy * np.arange(self.num_actions, dtype=np.float32)[None, :], axis=1)

        # Comparing expected action index against expert-supported mean action.
        action_deviation_from_expert = float(
            np.sum(state_weights * np.abs(expected_action - self.expert_mean_action))
        )

        # Returning all exact metrics in a dictionary.
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
        
        """
        Simulate trajectories under a policy for qualitative analysis.

        Unlike exact_policy_evaluation(), this method samples episodes and is
        useful for:
        - action histograms
        - average jump size between actions
        - qualitative trajectory-level summaries
        """

        # Creating local NumPy RNG for reproducible Monte Carlo simulation.
        rng = np.random.RandomState(seed)

        # Converting policy to NumPy float32 array.
        policy = np.asarray(policy, dtype=np.float32)

        # Action usage counts across all simulated trajectories.
        action_hist = np.zeros(self.num_actions, dtype=np.int64)

        # Stores absolute changes between consecutive actions.
        jump_values = []

        # Episode-level outputs.
        episodic_returns = []
        episodic_survival = []
        episodic_cost = []
        episodic_length = []

        # Simulate many episodes.
        for _ in range(int(num_episodes)):
            # Sample initial state from released initial-state distribution.
            state = int(rng.choice(self.num_states, p=self.initial_state_dist))

            # No previous action at the start of the episode.
            previous_action = None

            # Episode accumulators.
            total_reward = 0.0
            total_cost = 0.0
            length = 0

            # Roll out episode up to finite horizon.
            for _step in range(self.horizon):
                # Sample action from current state's policy row.
                action = int(rng.choice(self.num_actions, p=policy[state]))

                # Count action usage.
                action_hist[action] += 1

                # If this is not the first action, record jump size.
                if previous_action is not None:
                    jump_values.append(abs(action - previous_action))

                # Update previous action tracker.
                previous_action = action

                # Sample next state from transition distribution p(s, a, s').
                next_state = int(rng.choice(self.num_states, p=self.transition[state, action]))

                # Reward in this benchmark depends only on next state.
                reward = float(self.reward_by_next_state[next_state])

                # Cost depends on whether chosen action was inadmissible.
                cost = float(self.immediate_cost[state, action])

                # Accumulate episode totals.
                total_reward += reward
                total_cost += cost
                length += 1

                # Move to next state.
                state = next_state

                # Stop episode if terminal state reached.
                if self.terminal_mask[state] == 1:
                    break

            # Storing episode return.
            episodic_returns.append(total_reward)

            # Survival indicator: 1 only if final state is survival terminal state.
            episodic_survival.append(1.0 if state == self.survival_state else 0.0)

            # Per-step inadmissibility cost for this episode.
            episodic_cost.append(0.0 if length == 0 else total_cost / float(length))

            # Storing episode length.
            episodic_length.append(float(length))

        # Converting action counts to relative frequencies.
        action_hist = action_hist.astype(np.float64)
        if np.sum(action_hist) > 0.0:
            action_hist = action_hist / np.sum(action_hist)

        # Returning qualitative Monte Carlo metrics.
        return {
            "return_mean": float(np.mean(episodic_returns)),
            "survival_mean": float(np.mean(episodic_survival)),
            "inadmissibility_mean": float(np.mean(episodic_cost)),
            "length_mean": float(np.mean(episodic_length)),
            "mean_action_jump": float(np.mean(jump_values)) if jump_values else 0.0,
            "action_histogram": action_hist.tolist(),
        }

    def save_benchmark_description(self, path):
        
        """
        Saving a short JSON file describing the loaded benchmark configuration.

        This improves reproducibility by recording the benchmark settings that
        were actually used in an experiment run.
        """

        # Building description payload.
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

        # Saving as readable JSON for reference
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)