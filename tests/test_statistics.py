from __future__ import annotations

import math

import numpy as np

from trainers import mean_ci95


def test_mean_ci95_uses_sample_standard_deviation_and_t_interval():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    summary = mean_ci95(values)

    expected_mean = 3.0
    expected_std = float(np.std(values, ddof=1))
    expected_half = 2.776 * expected_std / math.sqrt(5)

    assert summary["n"] == 5
    assert math.isclose(summary["mean"], expected_mean, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["std"], expected_std, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["ci95_half"], expected_half, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["ci95_low"], expected_mean - expected_half, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["ci95_high"], expected_mean + expected_half, rel_tol=0.0, abs_tol=1e-12)


def test_single_run_ci_is_not_interpreted_as_population_uncertainty():
    summary = mean_ci95([0.5])
    assert summary["n"] == 1
    assert summary["std"] == 0.0
    assert summary["ci95_half"] == 0.0
