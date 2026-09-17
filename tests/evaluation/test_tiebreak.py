"""Tests for evaluation.tiebreak: argmax_random_tie (sampled tie-break,
used for E2's collection policy) and uniform_tie_policy_probs (the exact
mixture over ties, used for E2's Delta evaluation policy -- see
docs/E2_online_4zones.md and planning.value_iteration.policy_evaluation_stochastic).
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.evaluation.tiebreak import argmax_random_tie, uniform_tie_policy_probs


def test_uniform_tie_policy_probs_rows_sum_to_one():
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(20, 5))
    probs = uniform_tie_policy_probs(Q)
    assert probs.sum(axis=1) == pytest.approx(1.0)


def test_uniform_tie_policy_probs_is_one_hot_for_a_unique_max():
    Q = np.array([[1.0, 5.0, 2.0, 0.0, -1.0]])
    probs = uniform_tie_policy_probs(Q)
    expected = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    assert probs == pytest.approx(expected)


def test_uniform_tie_policy_probs_splits_uniformly_over_exact_ties():
    Q = np.array([[3.0, 3.0, 1.0, 3.0, 0.0]])
    probs = uniform_tie_policy_probs(Q)
    expected = np.array([[1 / 3, 1 / 3, 0.0, 1 / 3, 0.0]])
    assert probs == pytest.approx(expected)


def test_uniform_tie_policy_probs_all_actions_tied_under_a_flat_prior():
    """The motivating case: under a Dirichlet prior alone (no data yet),
    every action at a state has the exact same Q -- must be exactly
    uniform, not collapsed onto action 0 by np.argmax."""
    Q = np.full((1, 5), 7.0)
    probs = uniform_tie_policy_probs(Q)
    assert probs == pytest.approx(np.full((1, 5), 0.2))


def test_uniform_tie_policy_probs_respects_tolerance_not_exact_equality():
    """Under real (converged, not exactly-equal) floating point Q values,
    near-ties within tol must still be pooled -- exact equality is too
    strict for Q values produced by an iterative fixed-point solver.
    Margins are set an order of magnitude inside/outside tol so the
    comparison is robust to float64 rounding of the boundary itself."""
    Q = np.array([[1.0, 1.0 + 1e-10, 1.0 - 1e-10, 0.0, -1.0]])
    probs = uniform_tie_policy_probs(Q, tol=1e-9)
    assert probs[0, 0] == pytest.approx(1 / 3)
    assert probs[0, 1] == pytest.approx(1 / 3)
    assert probs[0, 2] == pytest.approx(1 / 3)
    assert probs[0, 3] == 0.0
    assert probs[0, 4] == 0.0


def test_uniform_tie_policy_probs_matches_argmax_random_tie_support():
    """uniform_tie_policy_probs and argmax_random_tie must agree on WHICH
    actions are tied (same tol), even though one returns a mixture and the
    other a sampled draw -- regression pin against divergent tie sets."""
    rng = np.random.default_rng(2)
    Q = rng.integers(0, 3, size=(50, 4)).astype(np.float64)  # many exact ties by construction
    probs = uniform_tie_policy_probs(Q, tol=1e-9)
    tie_break_rng = np.random.default_rng(3)
    n_draws = 4000
    empirical = np.zeros(Q.shape)
    for _ in range(n_draws):
        policy = argmax_random_tie(Q, tie_break_rng)
        empirical[np.arange(Q.shape[0]), policy] += 1
    empirical /= n_draws
    # only compare where uniform_tie_policy_probs says an action is reachable
    reachable = probs > 0
    assert np.all(empirical[reachable] > 0), "argmax_random_tie never sampled an action uniform_tie_policy_probs says is tied"
    assert np.allclose(empirical[~reachable], 0.0), "argmax_random_tie sampled an action uniform_tie_policy_probs says is NOT tied"


def test_argmax_random_tie_uses_tolerance_not_exact_equality():
    """Same near-tie robustness check as above, for the sampled tie-break."""
    Q = np.array([[1.0, 1.0 + 5e-10, 0.0]])
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    for _ in range(2000):
        a = argmax_random_tie(Q, rng)[0]
        counts[a] += 1
    assert counts[2] == 0  # action 2 is not tied, never picked
    assert counts[0] > 0 and counts[1] > 0  # both near-tied actions get picked
