# Appendix C Experiment: No-Mask PD/RC Diagnostic

This folder contains the additional Appendix C diagnostic experiment for LAADAN-AC.

The experiment tests whether adaptive Lagrangian cost control remains informative when the original hard admissibility mask is removed from actor optimisation. It separates three effects: adaptive primal-dual cost pressure, risk-adjusted action scoring, and deterministic cost-checked action selection.

This experiment is not intended to replace the main masked LAADAN-AC result. It is an auxiliary diagnostic supporting the manuscript interpretation that reliable selected-action admissibility requires an explicit feasible-action mechanism, either masking or cost-checked action selection.

## Structure

```text
appendix_c_experiment/
├── README.md
├── code/
│   ├── models.py
│   ├── trainers.py
│   ├── run_experiments.py
│   ├── pick_final_models.py
│   ├── test_final_models.py
│   ├── run_rc_ablations.py
│   ├── test_rc_ablation_models.py
│   ├── plot_rc_ablations.py
│   └── select_appendix_c_final_models.py
└── results/
    ├── icu_sepsis_LAADAN_AC_PD/
    ├── eicu_LAADAN_AC_PD/
    ├── icu_rc_ablations/
    └── eicu_rc_ablations/
```

## Notes

- `code/` is reserved for the Appendix C no-mask PD/RC diagnostic scripts. Scripts can be added in a follow-up commit.
- `results/` contains the corresponding saved outputs.
- This diagnostic is for benchmark research and reproducibility only.
- It is not a clinical decision-support system and must not be used to guide patient treatment.
