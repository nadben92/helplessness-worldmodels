"""Unit tests for the building blocks used by
experiments/E1_offline_4zones/calibration.py, kept separate from the full
sweep so the sweep script itself stays thin orchestration (see the plan:
"the sweep script itself stays thin orchestration"). Also covers the
reward/terminal array builder and the reflection-symmetry invariant on a
small hand-built example, independent of the real 4-zone layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from calibration import (  # noqa: E402
    distance_to_zone,
    gradient_curve,
    joint_confounding_regression,
    reachable_states_and_hops,
    relative_contrast,
)
from env import reward_and_terminal  # noqa: E402

from helplessness.planning.value_iteration import value_iteration


# ---------------------------------------------------------------------
# distance_to_zone
# ---------------------------------------------------------------------


def test_distance_to_zone_is_zero_inside_the_zone():
    grid_shape = (4, 4)
    mask = np.zeros(grid_shape, dtype=bool)
    mask[1, 1] = True
    dist = distance_to_zone(mask, grid_shape)
    assert dist[1 * 4 + 1] == 0


def test_distance_to_zone_wraps_on_the_torus():
    grid_shape = (4, 4)
    mask = np.zeros(grid_shape, dtype=bool)
    mask[0, 0] = True
    dist = distance_to_zone(mask, grid_shape)
    # (3,3) is distance 1+1=2 the "short way" around the torus, not 3+3=6
    assert dist[3 * 4 + 3] == 2


# ---------------------------------------------------------------------
# reachable_states_and_hops
# ---------------------------------------------------------------------


def test_reachable_states_and_hops_on_a_simple_chain():
    # 3 states, 1 action, deterministic chain 0 -> 1 -> 2 (goal)
    n_states, n_actions = 3, 1
    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 1] = 1.0
    P[1, 0, 2] = 1.0
    P[2, 0, 2] = 1.0
    policy = np.zeros(n_states, dtype=int)
    reachable, hops = reachable_states_and_hops(P, policy, goal_s=2)
    assert reachable.all()
    assert hops.tolist() == [2, 1, 0]


def test_reachable_states_and_hops_flags_unreachable_states():
    n_states, n_actions = 3, 1
    P = np.zeros((n_states, n_actions, n_states))
    P[0, 0, 0] = 1.0  # self-loop, never reaches goal
    P[1, 0, 2] = 1.0
    P[2, 0, 2] = 1.0
    policy = np.zeros(n_states, dtype=int)
    reachable, hops = reachable_states_and_hops(P, policy, goal_s=2)
    assert reachable.tolist() == [False, True, True]
    assert hops[0] == -1


# ---------------------------------------------------------------------
# gradient_curve / relative_contrast
# ---------------------------------------------------------------------


def test_gradient_curve_groups_by_distance_and_reports_counts():
    V = np.array([10.0, 20.0, 30.0, 40.0])
    distances = np.array([0, 0, 1, 2])
    eligible = np.array([True, True, True, True])
    curve = gradient_curve(V, distances, eligible)
    assert curve[0] == {"mean": 15.0, "n": 2}
    assert curve[1] == {"mean": 30.0, "n": 1}
    assert curve[2] == {"mean": 40.0, "n": 1}


def test_gradient_curve_respects_eligibility_mask():
    V = np.array([10.0, 20.0, 30.0])
    distances = np.array([0, 1, 2])
    eligible = np.array([True, False, True])
    curve = gradient_curve(V, distances, eligible)
    assert set(curve) == {0, 2}


def test_relative_contrast_matches_the_spec_formula():
    # (V_far - V_near) / |V_far|
    curve = {0: {"mean": -10.0, "n": 5}, 3: {"mean": -2.0, "n": 5}}
    contrast = relative_contrast(curve)
    assert contrast == pytest.approx((-2.0 - -10.0) / abs(-2.0))


def test_relative_contrast_none_with_fewer_than_two_bins():
    assert relative_contrast({0: {"mean": 1.0, "n": 1}}) is None


# ---------------------------------------------------------------------
# joint_confounding_regression
# ---------------------------------------------------------------------


def test_joint_confounding_regression_recovers_exact_linear_relationship():
    # V = 2 + 3*d_a - 1*d_b exactly -> R^2 should be 1.0, coefficients exact
    n = 20
    rng = np.random.default_rng(0)
    d_a = rng.integers(0, 5, size=n).astype(float)
    d_b = rng.integers(0, 5, size=n).astype(float)
    V = 2.0 + 3.0 * d_a - 1.0 * d_b
    eligible = np.ones(n, dtype=bool)
    result = joint_confounding_regression(V, d_a, d_b, eligible)
    assert result["intercept"] == pytest.approx(2.0, abs=1e-8)
    assert result["coef_d_a"] == pytest.approx(3.0, abs=1e-8)
    assert result["coef_d_b"] == pytest.approx(-1.0, abs=1e-8)
    assert result["r2"] == pytest.approx(1.0, abs=1e-8)


def test_joint_confounding_regression_low_r2_when_v_is_noise():
    n = 50
    rng = np.random.default_rng(1)
    d_a = rng.integers(0, 5, size=n).astype(float)
    d_b = rng.integers(0, 5, size=n).astype(float)
    V = rng.normal(size=n) * 1e-9  # negligible signal
    eligible = np.ones(n, dtype=bool)
    result = joint_confounding_regression(V, d_a, d_b, eligible)
    assert result["r2"] < 0.5


# ---------------------------------------------------------------------
# reward_and_terminal + a small reflection-symmetry sanity check,
# independent of the real 4-zone layout
# ---------------------------------------------------------------------


def test_reward_and_terminal_is_minus_one_step_plus_goal_reward():
    cfg = {"grid_shape": [3, 3], "goal": [0, 0]}
    R, terminal = reward_and_terminal(cfg, goal_reward=100.0)
    assert R.shape == (9,) and terminal.shape == (9,)
    assert terminal[0] and terminal.sum() == 1
    assert R[0] == 100.0
    assert np.all(R[1:] == -1.0)


def test_value_iteration_symmetric_ring_has_symmetric_values():
    """Small sanity check of the reflection-symmetry reasoning used in
    calibration.py's control-environment assertion, on a toy ring (not the
    real 4-zone layout) so the invariant is checked in isolation."""
    n_states, n_actions = 6, 2  # ring of 6 states, "clockwise"/"counterclockwise"
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        P[s, 0, (s + 1) % n_states] = 1.0
        P[s, 1, (s - 1) % n_states] = 1.0
    R = np.full(n_states, -1.0)
    R[0] = 10.0
    terminal = np.zeros(n_states, dtype=bool)
    terminal[0] = True
    V, _ = value_iteration(P, R, terminal, gamma=0.9)
    # reflection s -> (-s) mod 6 fixes state 0 and is a symmetry of this ring
    for s in range(n_states):
        assert V[s] == pytest.approx(V[(-s) % n_states], abs=1e-8)
