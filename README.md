# LAADAN-AC

### Beyond Survival in Admissible Offline Treatment-Policy Learning

[Riya Basak](https://github.com/AnnyaB), [Manal Helal](https://github.com/mhelal)

**LAADAN-AC** is a research codebase for admissibility-aware offline reinforcement learning in sepsis treatment-policy benchmarks. The project studies a central failure mode in offline treatment-policy learning: a policy may obtain high estimated survival/return while selecting actions that are weakly supported or inadmissible under the benchmark action mask.

We introduce **Lagrangian Admissibility-Aware Deep Action-Nudging Actor-Critic (LAADAN-AC)**, an offline actor-critic framework that combines hard admissibility masking, twin reward critics, conservative critic regularisation, expert-policy regularisation, a state-action smoothness proxy and Lagrangian cost control. The method is evaluated on the **ICU-Sepsis** benchmark and on a constructed **eICU-CRD Demo** Markov decision process used as a cross-source portability check.

<p align="center">
   <b>[ <a href="https://github.com/AnnyaB/laadan-ac">Code</a> | Results included in this repository ]</b>
</p>

<br>

<p align="center">
  <img src="assets/architecture_diagram.png" width="80%">
</p>

## Abstract

Offline reinforcement learning offers a way to study sepsis treatment policies without online patient experimentation, but high estimated survival can be misleading when a policy selects poorly supported or inadmissible actions. LAADAN-AC learns within a benchmark-defined admissible action interface by combining hard action masking, twin reward critics, conservative critic regularisation, expert-policy regularisation, a state-action smoothness proxy and Lagrangian cost control. We evaluate the framework on the ICU-Sepsis benchmark using five random seeds and exact finite-horizon Markov decision process evaluation, then test portability on a constructed eICU Collaborative Research Database Demo Markov decision process. On ICU-Sepsis, LAADAN-AC achieves competitive survival/return with zero selected-action inadmissibility and the strongest expert alignment among the main methods. A relaxed LAADAN-AC variant raises selected-checkpoint return on both ICU-Sepsis and eICU-CRD Demo while retaining zero inadmissibility. Ablations, a no-mask Lagrangian frontier and safety-failure diagnostics show that hard masking supplies the direct admissibility guarantee, while conservative and expert-guided regularisation shape the policy learned inside the admissible set.

## Main findings

| Experiment | Finding |
|---|---|
| ICU-Sepsis main comparison | LAADAN-AC achieves zero selected-action inadmissibility and the strongest expert argmax agreement among the main methods |
| VOAC comparison | Vanilla Offline Actor-Critic reaches higher return but selects inadmissible actions at a high rate |
| Relaxed LAADAN-AC | Lower soft regularisation improves selected-checkpoint return on ICU-Sepsis and eICU-CRD Demo while retaining zero inadmissibility |
| eICU-CRD Demo portability check | The same pipeline can be reused under a shifted MDP when transition dynamics, expert policy and admissibility masks are available |
| Component ablation | Hard masking provides the direct admissibility guarantee; conservative and expert-guided regularisation shape the policy within the admissible set |
| No-mask Lagrangian frontier | Lagrangian cost control alone does not replace masked admissible action selection in this benchmark |
| Safety-failure analysis | VOAC's return advantage is associated with unsupported action selection at state, action, value and trajectory levels |

<p align="center">
  <img src="assets/cross_domain_portability.png" width="80%">
</p>

## Repository structure

```text
laadan-ac/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── assets/
│   ├── architecture_diagram.png
│   ├── cross_domain_portability.png
│   ├── component_ablation.png
│   └── safety_failure_analysis.png
├── data/
│   ├── README.md
│   ├── icu_sepsis/
│   │   ├── expertPolicy.csv
│   │   ├── initialStateDistribution.csv
│   │   ├── rewardFunction.csv
│   │   ├── transitionFunction.csv
│   │   └── extras/
│   │       ├── admissibleActions.txt
│   │       ├── sofaScores.csv
│   │       └── stateClusterCenters.csv
│   └── eicu_demo_mdp/
│       ├── admissibleActions.txt
│       ├── expertPolicy.csv
│       ├── initialStateDistribution.csv
│       ├── rewardFunction.csv
│       ├── transitionFunction.csv
│       └── extras/
│           ├── action_names.json
│           ├── build_summary.csv
│           ├── eicu_demo_mdp_description.json
│           └── stateClusterCenters.csv
├── scripts/
│   ├── benchmark.py
│   ├── build_eicu_demo_mdp.py
│   ├── models.py
│   ├── trainers.py
│   ├── run_experiments.py
│   ├── pick_final_models.py
│   ├── test_final_models.py
│   ├── lagrangian_frontier.py
│   ├── safety_failure_analysis.py
│   └── plots.py
└── results/
    ├── icu_sepsis_main/
    ├── icu_sepsis_relaxed/
    ├── eicu_demo_main/
    ├── eicu_demo_relaxed/
    ├── lagrangian_frontier/
    └── safety_failure_analysis/
```

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Check that the main scripts compile:

```bash
python -m py_compile scripts/*.py
```

## Data

This repository contains the two processed Markov decision process folders used in the released experiments.

### ICU-Sepsis benchmark

The main experiment uses the released ICU-Sepsis benchmark Markov decision process. The expected files are:

```text
data/icu_sepsis/
├── expertPolicy.csv
├── initialStateDistribution.csv
├── rewardFunction.csv
├── transitionFunction.csv
└── extras/
    ├── admissibleActions.txt
    ├── sofaScores.csv
    └── stateClusterCenters.csv
```

### eICU-CRD Demo MDP

The cross-source portability check uses the constructed eICU-CRD Demo Markov decision process included in this repository. The expected files are:

```text
data/eicu_demo_mdp/
├── admissibleActions.txt
├── expertPolicy.csv
├── initialStateDistribution.csv
├── rewardFunction.csv
├── transitionFunction.csv
└── extras/
    ├── action_names.json
    ├── build_summary.csv
    ├── eicu_demo_mdp_description.json
    └── stateClusterCenters.csv
```

The released experiments use the processed `data/eicu_demo_mdp/` files above.

## Using the code

The repository is organised around the final experiments and outputs used in the manuscript. The `results/` folder contains training histories, metric files, selected checkpoints and diagnostic outputs for:

```text
results/icu_sepsis_main/
results/icu_sepsis_relaxed/
results/eicu_demo_main/
results/eicu_demo_relaxed/
results/lagrangian_frontier/
results/safety_failure_analysis/
```

The main scripts are:

| Script | Purpose |
|---|---|
| `scripts/benchmark.py` | Loads tabular Markov decision process files and performs exact finite-horizon evaluation |
| `scripts/models.py` | Defines Behaviour Cloning, Conservative Q and actor-critic networks |
| `scripts/trainers.py` | Implements Behaviour Cloning, CQL-regularised fitted-Q, Vanilla Offline Actor-Critic and LAADAN-AC training |
| `scripts/run_experiments.py` | Runs the main training and evaluation pipeline |
| `scripts/pick_final_models.py` | Selects final checkpoints from completed runs |
| `scripts/test_final_models.py` | Reloads selected checkpoints and recomputes final metrics |
| `scripts/lagrangian_frontier.py` | Runs component ablations and no-mask Lagrangian frontier experiments |
| `scripts/safety_failure_analysis.py` | Generates state-, action-, value- and trajectory-level safety diagnostics |
| `scripts/build_eicu_demo_mdp.py` | Documents the construction of the eICU-CRD Demo Markov decision process |
| `scripts/plots.py` | Shared plotting utilities used by experiment scripts |

### Example usage

Run the main ICU-Sepsis experiment:

```bash
DATA_DIR="data/icu_sepsis"

python -u scripts/run_experiments.py \
  --data-dir "$DATA_DIR" \
  --results-dir results/icu_sepsis_main \
  --device auto
```

Select and verify the final ICU-Sepsis checkpoints:

```bash
python -u scripts/pick_final_models.py \
  --results-dir results/icu_sepsis_main

python -u scripts/test_final_models.py \
  --data-dir "$DATA_DIR" \
  --final-models-dir results/icu_sepsis_main/final_models \
  --config-path results/icu_sepsis_main/run_config.json \
  --device auto \
  --horizon 50 \
  --output-json results/icu_sepsis_main/final_models/test_summary.json
```

Run the same pipeline on the processed eICU-CRD Demo MDP:

```bash
DATA_DIR="data/eicu_demo_mdp"

python -u scripts/run_experiments.py \
  --data-dir "$DATA_DIR" \
  --results-dir results/eicu_demo_main \
  --device auto

python -u scripts/pick_final_models.py \
  --results-dir results/eicu_demo_main

python -u scripts/test_final_models.py \
  --data-dir "$DATA_DIR" \
  --final-models-dir results/eicu_demo_main/final_models \
  --config-path results/eicu_demo_main/run_config.json \
  --device auto \
  --horizon 50 \
  --output-json results/eicu_demo_main/final_models/test_summary.json
```

Run the component-ablation and no-mask Lagrangian frontier analysis:

```bash
python -u scripts/lagrangian_frontier.py \
  --data-dir data/icu_sepsis \
  --results-dir results/lagrangian_frontier \
  --device auto
```

Run the safety-failure diagnostics from the selected ICU-Sepsis checkpoints:

```bash
python -u scripts/safety_failure_analysis.py \
  --data-dir data/icu_sepsis \
  --final-models-dir results/icu_sepsis_main/final_models \
  --config-path results/icu_sepsis_main/run_config.json \
  --output-dir results/safety_failure_analysis \
  --device auto
```

Use `--help` to inspect the available command-line options for each script:

```bash
python scripts/run_experiments.py --help
python scripts/pick_final_models.py --help
python scripts/test_final_models.py --help
python scripts/lagrangian_frontier.py --help
python scripts/safety_failure_analysis.py --help
```

## Saved checkpoints

The released `results/` folders include `.pt` checkpoint files for the trained models. These checkpoint files are stored with Git LFS. Before cloning the repository, install Git LFS:

```bash
git lfs install
git clone https://github.com/AnnyaB/laadan-ac.git
cd laadan-ac
git lfs pull
```

Including checkpoints allows the final metrics and safety-failure diagnostics to be reloaded directly from the repository.

## Figures included in this repository

The README uses four summary figures stored in `assets/`:

```text
assets/architecture_diagram.png
assets/cross_domain_portability.png
assets/component_ablation.png
assets/safety_failure_analysis.png
```

These figures summarise the architecture, cross-domain portability check, component ablation and safety-failure analysis. The reproducibility record is stored in the corresponding data, script and result folders.

<p align="center">
  <img src="assets/component_ablation.png" width="80%">
</p>

<p align="center">
  <img src="assets/safety_failure_analysis.png" width="80%">
</p>

## Important note

This repository is a benchmark research implementation. It is **not** a clinical decision-support system and must not be used to guide patient treatment. The experiments evaluate return, admissibility and expert alignment under fixed benchmark Markov decision processes.

## Citation

If you find this code useful, please cite it as:

```bibtex
@misc{basak2026laadanac,
  title        = {LAADAN-AC: Beyond Survival in Admissible Offline Treatment-Policy Learning},
  author       = {Basak, Riya and Helal, Manal},
  year         = {2026},
  url          = {https://github.com/AnnyaB/laadan-ac},
  note         = {Research code repository}
}
```

## Contact and contributions

For questions, reproducibility issues or suggested improvements, please open a GitHub issue.
