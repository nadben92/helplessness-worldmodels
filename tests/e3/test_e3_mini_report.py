"""Verdict rules of the mini report (docs/e3/DECISIONS.md, 2026-09-24):
3-standard-error bounds, "non concluant" for orderings within noise, and
the mechanics gate that turns downstream indicators into "non
évaluable" when template-match accuracy is below 0.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_mini_report_run_mini")


def test_ordering_within_three_combined_se_is_non_concluant_never_ok():
    # gap 0.05, combined se sqrt(0.02^2 + 0.02^2) = 0.0283 -> 3 x = 0.085 > gap
    assert run_mini.ordering_verdict(0.10, 0.02, 0.15, 0.02) == "non concluant"


def test_ordering_beyond_three_combined_se_in_the_expected_direction_is_ok():
    assert run_mini.ordering_verdict(0.10, 0.01, 0.50, 0.01) == "OK"


def test_ordering_beyond_three_combined_se_in_the_wrong_direction_is_fail():
    assert run_mini.ordering_verdict(0.50, 0.01, 0.10, 0.01) == "FAIL"


def test_ordering_exactly_at_the_margin_is_non_concluant():
    # gap == 3 * combined se exactly (0.3 and 0.1 exact in binary after the arithmetic below)
    se = 0.1 / np.sqrt(2.0)
    assert run_mini.ordering_verdict(0.0, se, 3 * 0.1, se) == "non concluant"


def test_bounds_verdict_tolerates_three_se_but_not_more():
    ceiling = np.log2(5)
    se = np.array([0.01, 0.01, 0.01])
    assert run_mini.bounds_verdict(np.array([-0.029, 0.5, ceiling + 0.029]), se, ceiling)["ok"]
    result = run_mini.bounds_verdict(np.array([-0.031, 0.5, ceiling + 0.031]), se, ceiling)
    assert not result["ok"]
    assert result["n_violations"] == 2


def test_zone_mean_mc_se_combines_per_state_se_in_quadrature():
    se = np.array([0.03, 0.04])
    assert run_mini.zone_mean_mc_se(se) == pytest.approx(np.sqrt(0.03**2 + 0.04**2) / 2)


def test_mechanics_gate_threshold():
    assert not run_mini.mechanics_ok(0.49)
    assert run_mini.mechanics_ok(0.5)
