#!/usr/bin/env bash
set -euo pipefail

# Final reviewer-response experiment/reproducibility pipeline.
# Run from the repository root on the fixed-schedule working branch.
# This script never checks out or modifies main.

ROOT="results/fixed_schedule_2026"
ICU_ROOT="$ROOT/icu_sepsis"
EICU_ROOT="$ROOT/eicu_demo"

printf '\n[1/10] Running deterministic/unit tests...\n'
python -m pytest -q tests

printf '\n[2/10] Rebuilding existing ICU aggregates + recursive data manifest without retraining...\n'
python scripts/rebuild_fixed_schedule_aggregates.py \
  --root "$ICU_ROOT" \
  --data-dir data/icu_sepsis

printf '\n[3/10] Completing the four missing ICU leave-one-out ablations only...\n'
python scripts/run_fixed_schedule_ablations.py \
  --data-dir data/icu_sepsis \
  --output-root "$ICU_ROOT" \
  --device auto

printf '\n[4/10] Running the eICU fixed-schedule main suite (no eICU ablations)...\n'
if [[ -d "$EICU_ROOT/main_fixed_schedule" ]]; then
  echo "STOP: $EICU_ROOT/main_fixed_schedule already exists."
  echo "Refusing to overwrite an existing eICU fixed-schedule run automatically."
  echo "Inspect/archive it first, then rerun this command intentionally if needed."
  exit 2
fi
python scripts/run_fixed_schedule.py \
  --datasets eicu \
  --eicu-data data/eicu_demo_mdp \
  --output "$ROOT" \
  --eicu-ablations none \
  --device auto

printf '\n[5/10] Rebuilding eICU aggregates + recursive data manifest from saved metrics...\n'
python scripts/rebuild_fixed_schedule_aggregates.py \
  --root "$EICU_ROOT" \
  --data-dir data/eicu_demo_mdp

printf '\n[6/10] Validating numeric/protocol integrity before diagnostics/plotting...\n'
python scripts/validate_fixed_schedule_results.py \
  --root "$ICU_ROOT" \
  --data-dir data/icu_sepsis \
  --ablations full \
  --skip-figures
python scripts/validate_fixed_schedule_results.py \
  --root "$EICU_ROOT" \
  --data-dir data/eicu_demo_mdp \
  --ablations none \
  --skip-figures

printf '\n[7/10] Regenerating fixed-checkpoint safety-failure diagnostics...\n'
python scripts/run_fixed_schedule_safety_diagnostics.py \
  --data-dir data/icu_sepsis \
  --results-root "$ICU_ROOT" \
  --output-dir "$ICU_ROOT/safety_diagnostics" \
  --illustrative-seed 42 \
  --num-trajectories 2000 \
  --device auto

printf '\n[8/10] Generating final 600-dpi PNG figures from frozen result files...\n'
python scripts/plot_fixed_schedule_release.py \
  --root "$ROOT" \
  --out-dir "$ICU_ROOT/figures"

printf '\n[9/10] Final ICU/eICU validation after figure generation...\n'
python scripts/validate_fixed_schedule_results.py \
  --root "$ICU_ROOT" \
  --data-dir data/icu_sepsis \
  --ablations full
python scripts/validate_fixed_schedule_results.py \
  --root "$EICU_ROOT" \
  --data-dir data/eicu_demo_mdp \
  --ablations none \
  --skip-figures

printf '\n[10/10] Freezing final numbers, protocol metadata, and SHA-256 file manifest...\n'
python scripts/freeze_fixed_schedule_release.py \
  --root "$ROOT" \
  --output "$ROOT/release_manifest.json"

printf '\nPASS: final fixed-schedule experiment suite completed.\n'
printf 'Next: inspect generated figures and git diff; do not merge to main yet.\n'
