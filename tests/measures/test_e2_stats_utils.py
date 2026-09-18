"""Tests for experiments/E2_online_4zones/stats_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

su = load_experiment_module("E2_online_4zones", "stats_utils.py", "_e2_stats_utils_for_stats_test")


def test_mean_ci95_contains_the_true_mean_most_of_the_time():
    rng = np.random.default_rng(0)
    true_mean = 5.0
    n_trials = 200
    n_covering = 0
    for _ in range(n_trials):
        sample = rng.normal(loc=true_mean, scale=2.0, size=20)
        mean, lo, hi = su.mean_ci95(sample)
        n_covering += int(lo <= true_mean <= hi)
    # a 95% CI should cover the true mean close to 95% of the time
    assert 0.85 < n_covering / n_trials <= 1.0


def test_mean_ci95_single_value_returns_nan_bounds():
    mean, lo, hi = su.mean_ci95(np.array([3.0]))
    assert mean == 3.0
    assert np.isnan(lo) and np.isnan(hi)


def test_paired_comparison_all_zero_diffs_is_well_defined():
    result = su.paired_comparison(np.zeros(20))
    assert result["mean"] == 0.0
    assert result["cohens_d"] == 0.0
    assert np.isnan(result["wilcoxon_p"])


def test_paired_comparison_detects_a_real_systematic_difference():
    rng = np.random.default_rng(1)
    diffs = rng.normal(loc=2.0, scale=0.5, size=20)  # clearly nonzero, low variance
    result = su.paired_comparison(diffs)
    assert result["wilcoxon_p"] < 0.01
    assert result["cohens_d"] > 1.0  # large effect
    assert result["ci_low"] > 0  # CI excludes 0


def test_paired_comparison_no_effect_gives_high_p_value():
    rng = np.random.default_rng(2)
    diffs = rng.normal(loc=0.0, scale=1.0, size=20)
    result = su.paired_comparison(diffs)
    assert result["ci_low"] < 0 < result["ci_high"]  # CI should straddle 0
