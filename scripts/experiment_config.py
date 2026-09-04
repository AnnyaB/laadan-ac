#!/usr/bin/env python3
"""Fixed experiment configurations used by the camera-ready schedule."""

from __future__ import annotations

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
