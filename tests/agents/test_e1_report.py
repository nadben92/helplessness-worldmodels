"""Tests for experiments/E1_offline_4zones/report.py: predicted_b()
against fresh analytic values, fit_inverse_sqrt_n()/fit_power_law_slope()
(fitted 1/sqrt(n) curve across all budgets, not a pointwise plateau read
-- an SD estimated from a handful of seeds is too noisy to read
pointwise), and the headline/inflated table split between the true-H0
(controllable) and false-H0 (uncontrollable) zones.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from report import (  # noqa: E402
    across_seed_sd,
    detectable_b_table_inflated,
    fit_inverse_sqrt_n,
    fit_power_law_slope,
    fit_power_law_slope_ci,
    headline_detectable_b_table,
    out_of_tolerance_rate,
    predicted_b,
    recommended_n,
)

from helplessness.envs.dynamics.controllable import ControllableNoisy


def test_predicted_b_deterministic_zone_is_minus_log2_5():
    assert predicted_b("uncontrollable_deterministic") == pytest.approx(-np.log2(5))


def test_predicted_b_noisy_zone_matches_fresh_analytic_controllability():
    expected = -ControllableNoisy(p=0.125).analytic_controllability(5)
    assert predicted_b("uncontrollable_noisy") == pytest.approx(expected)


def test_predicted_b_true_h0_zones_are_zero():
    assert predicted_b("controllable_noisy") == 0.0
    assert predicted_b("controllable_deterministic") == 0.0


def test_predicted_b_rejects_unknown_zone():
    with pytest.raises(ValueError):
        predicted_b("nowhere")


# ---------------------------------------------------------------------
# fit_inverse_sqrt_n / fit_power_law_slope
# ---------------------------------------------------------------------


def test_fit_inverse_sqrt_n_recovers_a_known_amplitude_exactly():
    A_true = 0.3
    ns = [10, 30, 100, 300, 1000]
    values = [A_true / np.sqrt(n) for n in ns]
    assert fit_inverse_sqrt_n(ns, values) == pytest.approx(A_true, rel=1e-9)


def test_fit_inverse_sqrt_n_is_robust_to_per_point_noise():
    """The whole point of fitting across all budgets: a single noisy
    point must not swing the fitted amplitude as much as reading that
    point alone would."""
    A_true = 0.3
    ns = [10, 30, 100, 300, 1000]
    values = [A_true / np.sqrt(n) for n in ns]
    values[2] *= 3.0  # one point badly corrupted (e.g. an unlucky seed set)
    A_fit = fit_inverse_sqrt_n(ns, values)
    assert abs(A_fit - A_true) < abs(values[2] * np.sqrt(ns[2]) - A_true)


def test_fit_power_law_slope_recovers_known_exponent():
    ns = [10, 30, 100, 300, 1000]
    values = [0.5 * n**-0.5 for n in ns]
    slope, _ = fit_power_law_slope(ns, values)
    assert slope == pytest.approx(-0.5, abs=1e-6)

    values_inv_n = [0.5 * n**-1.0 for n in ns]
    slope2, _ = fit_power_law_slope(ns, values_inv_n)
    assert slope2 == pytest.approx(-1.0, abs=1e-6)


def test_fit_power_law_slope_nan_with_fewer_than_two_positive_points():
    slope, intercept = fit_power_law_slope([10, 30], [0.0, 0.0])
    assert slope != slope  # NaN


# ---------------------------------------------------------------------
# fit_power_law_slope_ci
# ---------------------------------------------------------------------


def test_fit_power_law_slope_ci_contains_true_slope_for_noiseless_data():
    ns = [10, 30, 100, 300, 1000]
    values = [0.5 * n**-0.5 for n in ns]
    slope, ci_low, ci_high = fit_power_law_slope_ci(ns, values)
    assert slope == pytest.approx(-0.5, abs=1e-6)
    # noiseless data -> near-zero residual -> a very tight (possibly
    # degenerate) CI; the true slope must still be inside or on it.
    assert ci_low - 1e-6 <= -0.5 <= ci_high + 1e-6


def test_fit_power_law_slope_ci_excludes_a_genuinely_different_slope():
    """The CI must be tight enough to EXCLUDE a slope that is not -0.5
    when the underlying data is actually -1/n, not -1/sqrt(n) -- this is
    the exact scenario the pooled-bootstrap bug produced, and the whole
    point of adding a CI is to catch that kind of discrepancy."""
    ns = [10, 30, 100, 300, 1000]
    values = [0.5 * n**-1.0 for n in ns]  # genuinely -1 slope, noiseless
    slope, ci_low, ci_high = fit_power_law_slope_ci(ns, values)
    assert slope == pytest.approx(-1.0, abs=1e-6)
    assert not (ci_low <= -0.5 <= ci_high)


def test_fit_power_law_slope_ci_widens_with_noisier_data():
    ns = [10, 30, 100, 300, 1000]
    clean = [0.5 * n**-0.5 for n in ns]
    rng = np.random.default_rng(0)
    noisy = [v * np.exp(rng.normal(0, 0.3)) for v in clean]
    _, ci_low_clean, ci_high_clean = fit_power_law_slope_ci(ns, clean)
    _, ci_low_noisy, ci_high_noisy = fit_power_law_slope_ci(ns, noisy)
    assert (ci_high_noisy - ci_low_noisy) > (ci_high_clean - ci_low_clean)


def test_fit_power_law_slope_ci_nan_with_fewer_than_three_points():
    slope, ci_low, ci_high = fit_power_law_slope_ci([10, 30], [0.1, 0.05])
    assert slope != slope  # NaN: need >=3 points for a t-distribution CI (dof = n-2 >= 1)


# ---------------------------------------------------------------------
# out_of_tolerance_rate
# ---------------------------------------------------------------------


def test_out_of_tolerance_rate_counts_directly_not_via_the_fit():
    pred = predicted_b("uncontrollable_noisy")
    # 3 of 5 seeds land outside tolerance, by construction
    values = [pred, pred + 0.001, pred + 0.05, pred - 0.05, pred + 0.031]
    rows = _rows_with_sd("uncontrollable_noisy", 30, values)
    rate, n_over, n_total = out_of_tolerance_rate(rows, "uncontrollable_noisy", 30)
    assert n_total == 5
    assert n_over == 3
    assert rate == pytest.approx(0.6)


def test_out_of_tolerance_rate_nan_with_no_data():
    rate, n_over, n_total = out_of_tolerance_rate([], "uncontrollable_noisy", 30)
    assert rate != rate  # NaN
    assert n_total == 0


# ---------------------------------------------------------------------
# across_seed_sd / recommended_n
# ---------------------------------------------------------------------


def _rows_with_sd(zone, n, values):
    return [{"zone": zone, "n_budget": n, "env": "exp", "region": "all", "b": v} for v in values]


def test_across_seed_sd_matches_numpy_std_ddof1():
    rows = _rows_with_sd("uncontrollable_noisy", 100, [1.0, 1.1, 0.9, 1.05, 0.95])
    expected = np.std([1.0, 1.1, 0.9, 1.05, 0.95], ddof=1)
    assert across_seed_sd(rows, "uncontrollable_noisy", 100) == pytest.approx(expected)


def test_recommended_n_uses_a_safety_margin_not_the_raw_crossing():
    """The raw (1-sigma) crossing point is where the TYPICAL run's error
    equals tolerance -- roughly half the runs exceed it there, which is
    not a reliable tolerance budget. recommended_n must apply the
    2-sigma safety margin by default: with a genuine 1/sqrt(n) sd law
    A=0.15, the RAW crossing is n=(0.15/0.03)^2=25 (would round up to the
    tested budget 30 under the old, corrected criterion), but the
    2-sigma crossing is n=(2*0.15/0.03)^2=100 -- exactly a tested budget,
    chosen so this test pins the new behaviour precisely rather than via
    a "next budget up" rounding coincidence."""
    pred_det = predicted_b("uncontrollable_deterministic")
    pred_noisy = predicted_b("uncontrollable_noisy")

    rows = []
    for n in (10, 30, 100, 300, 1000):
        rows += _rows_with_sd("uncontrollable_deterministic", n, [pred_det] * 5)

    A_true = 0.15
    sd_targets = {n: A_true / np.sqrt(n) for n in (10, 30, 100, 300, 1000)}
    rng = np.random.default_rng(0)
    for n, sd in sd_targets.items():
        offsets = rng.normal(0, sd, size=5)
        offsets -= offsets.mean()
        rows += _rows_with_sd("uncontrollable_noisy", n, (pred_noisy + offsets).tolist())

    text = recommended_n(rows, perm_rows=[])
    assert "RECOMMENDED: n=100" in text
    assert "2-sigma" in text
    assert "3-sigma" in text  # the more conservative alternative is also reported, not just the primary pick


def test_recommended_n_does_not_crash_with_single_row_per_budget():
    """A single row per (zone, budget) can't produce a sample sd
    (ddof=1) -- across_seed_sd returns NaN, and fit_inverse_sqrt_n must
    drop it gracefully (it only uses positive values) rather than crash."""
    rows = [
        {"zone": "uncontrollable_deterministic", "n_budget": n, "env": "exp", "region": "all", "b": -np.log2(5)}
        for n in (10, 30, 100, 300, 1000)
    ] + [
        {"zone": "uncontrollable_noisy", "n_budget": n, "env": "exp", "region": "all", "b": predicted_b("uncontrollable_noisy")}
        for n in (10, 30, 100, 300, 1000)
    ]
    text = recommended_n(rows, perm_rows=[])  # must not raise
    assert "RECOMMENDED" in text


# ---------------------------------------------------------------------
# headline vs inflated table split
# ---------------------------------------------------------------------


def _perm_row(zone, n, p95, method="bootstrap", quantity="b"):
    return {"zone": zone, "quantity": quantity, "method": method, "n_budget": n, "p95": p95, "observed": 0.0, "detectable": False}


def test_headline_table_uses_controllable_zones_and_bootstrap_only():
    perm_rows = [_perm_row("controllable_noisy", n, 0.5 / np.sqrt(n), method="bootstrap") for n in (10, 30, 100, 300, 1000)]
    perm_rows += [_perm_row("controllable_noisy", n, 0.5, method="shuffle") for n in (10, 30, 100, 300, 1000)]  # must be ignored
    perm_rows += [_perm_row("controllable_deterministic", n, 0.02, method="bootstrap") for n in (10, 30, 100, 300, 1000)]
    table = headline_detectable_b_table(perm_rows)
    assert "controllable_noisy" in table
    assert "controllable_deterministic" in table
    assert "0.50000" not in table  # the shuffle value must not leak into the headline
    assert "DEGENERATE" in table  # controllable_deterministic flagged, not silently slope-fit
    assert "log-log slope" in table  # controllable_noisy DOES get a fitted slope


def test_inflated_table_labels_uncontrollable_zones():
    perm_rows = [_perm_row("uncontrollable_deterministic", n, 0.31, method="bootstrap") for n in (10, 30, 100, 300, 1000)]
    perm_rows += [_perm_row("uncontrollable_noisy", n, 0.15, method="bootstrap") for n in (10, 30, 100, 300, 1000)]
    table = detectable_b_table_inflated(perm_rows)
    assert "uncontrollable_deterministic" in table
    assert "WRONG number" in table
