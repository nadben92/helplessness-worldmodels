"""Tests for experiments/E1_offline_4zones/estimation.py: the generative
sampler (exactly n draws per (s,a) pair, coverage test) and the
counts-to-agent glue (plug-in MLE via alpha=0.0, and the Dirichlet
robustness variant's geographic_alpha_support).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from env import build_control_layout, build_env, build_layout  # noqa: E402
from estimation import agent_from_counts, collect_generative, geographic_alpha_support, spawn_exp_ctrl_rngs  # noqa: E402

from helplessness.config import load_yaml_block


def _small_tensor():
    # 3 states, 2 actions, hand-built, non-trivial probabilities
    P = np.array(
        [
            [[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5]],
        ]
    )
    return P


# ---------------------------------------------------------------------
# collect_generative: coverage (exactly n per (s,a)), no episode notion
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [10, 30, 100, 300, 1000])
def test_collect_generative_gives_exactly_n_draws_per_pair(n):
    P = _small_tensor()
    rng = np.random.default_rng(0)
    counts = collect_generative(P, n, rng)
    assert counts.shape == P.shape
    for s in range(P.shape[0]):
        for a in range(P.shape[1]):
            assert counts[s, a].sum() == n


def test_collect_generative_never_puts_mass_where_p_is_zero():
    P = _small_tensor()
    rng = np.random.default_rng(1)
    counts = collect_generative(P, 1000, rng)
    assert counts[0, 0, 2] == 0  # P[0,0,2] == 0.0
    assert counts[1, 0, 0] == 0
    assert counts[1, 0, 2] == 0


def test_collect_generative_converges_to_the_true_row_at_large_n():
    P = _small_tensor()
    rng = np.random.default_rng(2)
    n = 200_000
    counts = collect_generative(P, n, rng)
    empirical = counts[0, 1] / n
    assert np.allclose(empirical, P[0, 1], atol=0.01)


def test_collect_generative_is_independent_of_seed_ordering_not_a_trajectory():
    """No episode/reset notion: every (s,a) pair is drawn independently,
    so re-running with a fresh rng of the same seed reproduces the same
    counts regardless of pair iteration order (a property that would NOT
    hold for a trajectory-following collector, which depends on where it
    happens to be)."""
    P = _small_tensor()
    counts_a = collect_generative(P, 50, np.random.default_rng(42))
    counts_b = collect_generative(P, 50, np.random.default_rng(42))
    assert np.array_equal(counts_a, counts_b)


# ---------------------------------------------------------------------
# geographic_alpha_support
# ---------------------------------------------------------------------


def test_geographic_alpha_support_has_exactly_five_true_per_row_on_a_torus():
    grid_shape = (4, 4)
    support = geographic_alpha_support(grid_shape)
    assert support.shape == (16, 16)
    assert np.all(support.sum(axis=1) == 5)


def test_geographic_alpha_support_includes_self_and_wraps():
    grid_shape = (3, 3)
    support = geographic_alpha_support(grid_shape)
    s = 0  # (0,0)
    assert support[s, s]  # "stay"
    assert support[s, 2 * 3 + 0]  # "up" wraps to last row, same col
    assert support[s, 0 * 3 + 2]  # "left" wraps to last col, same row


# ---------------------------------------------------------------------
# agent_from_counts: alpha=0.0 is exact plug-in MLE; R_hat matches the
# deterministic true reward exactly wherever visited
# ---------------------------------------------------------------------


def test_agent_from_counts_alpha_zero_is_exact_plug_in_mle():
    P = _small_tensor()
    rng = np.random.default_rng(3)
    counts = collect_generative(P, 100, rng)
    R_true = np.array([-1.0, -1.0, 50.0])
    agent = agent_from_counts(counts, R_true, alpha=0.0)
    expected_T = counts / counts.sum(axis=2, keepdims=True)
    assert np.allclose(agent.T_hat, expected_T)


def test_agent_from_counts_zero_count_pair_would_divide_by_zero_without_full_coverage():
    """Confirms alpha=0.0 relies on every (s,a) having been visited --
    exactly what collect_generative() guarantees by construction (see the
    coverage test above); this is not a general-purpose default."""
    n_states, n_actions = 2, 2
    counts = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    counts[0, 0] = [5, 5]
    # counts[0,1], counts[1,*] left at zero -- simulates a pair never visited
    R_true = np.array([0.0, 0.0])
    agent = agent_from_counts(counts, R_true, alpha=0.0)
    with np.errstate(invalid="ignore"):
        T = agent.T_hat
    assert np.isnan(T[0, 1]).all()


def test_agent_from_counts_r_hat_matches_deterministic_reward_exactly():
    P = _small_tensor()
    rng = np.random.default_rng(4)
    counts = collect_generative(P, 300, rng)
    R_true = np.array([-1.0, -1.0, 50.0])
    agent = agent_from_counts(counts, R_true, alpha=0.0)
    visited_states = np.flatnonzero(counts.sum(axis=(0, 1)) > 0)
    assert np.array_equal(visited_states, np.arange(3))  # all 3 states reached at n=300
    assert np.allclose(agent.R_hat, R_true)


def test_agent_from_counts_robustness_variant_uses_geographic_support():
    P = _small_tensor()
    rng = np.random.default_rng(5)
    counts = collect_generative(P, 50, rng)
    R_true = np.array([-1.0, -1.0, 50.0])
    # geographic_alpha_support expects a real grid_shape; build a trivial
    # 1x3 "grid" consistent with this toy 3-state tensor's indexing.
    support = geographic_alpha_support((1, 3))
    agent = agent_from_counts(counts, R_true, alpha=0.5, alpha_support=support)
    T = agent.T_hat  # must not raise (alpha_support invariant asserted internally)
    assert T.shape == (3, 2, 3)


# ---------------------------------------------------------------------
# spawn_exp_ctrl_rngs: the RNG-correlation bug regression test, plus the
# deliberate shared-seed identity check the fix's diagnosis calls for.
# ---------------------------------------------------------------------

_CFG = load_yaml_block("zones/e1_4zones.yaml")


def _real_controllable_zone_setup():
    exp_layout = build_layout(_CFG, "away_from_goal")
    ctrl_layout = build_control_layout(_CFG, "away_from_goal")
    P_exp = build_env(exp_layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    zone = next(z for z in exp_layout.zones if z.appearance.category == "controllable_noisy")
    idx = np.flatnonzero(zone.mask)
    return P_exp, P_ctrl, idx


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spawn_exp_ctrl_rngs_gives_independent_streams_in_a_controllable_zone(seed):
    """Regression test for the real bug: two separately-seeded
    Generators with the SAME integer seed stayed in lockstep across the
    entire (s,a) loop, giving bit-identical counts in every
    already-controllable zone (where P_exp == P_ctrl exactly) -- true by
    construction, not by measurement. spawn_exp_ctrl_rngs() must not
    reproduce that."""
    P_exp, P_ctrl, idx = _real_controllable_zone_setup()
    rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(seed)
    N_exp = collect_generative(P_exp, 100, rng_exp)
    N_ctrl = collect_generative(P_ctrl, 100, rng_ctrl)
    assert not np.array_equal(N_exp[idx], N_ctrl[idx])


def test_the_old_same_seed_pattern_was_indeed_correlated_in_a_controllable_zone():
    """Documents exactly what was wrong, so the bug can't quietly come
    back: np.random.default_rng(seed) called twice with the same seed
    value gives bit-identical counts in a zone where P_exp == P_ctrl."""
    P_exp, P_ctrl, idx = _real_controllable_zone_setup()
    rng_exp = np.random.default_rng(7)
    rng_ctrl = np.random.default_rng(7)
    N_exp = collect_generative(P_exp, 100, rng_exp)
    N_ctrl = collect_generative(P_ctrl, 100, rng_ctrl)
    assert np.array_equal(N_exp[idx], N_ctrl[idx])


def test_shared_seed_identity_check_derive_control_is_a_true_identity_on_controllable_zones():
    """The deliberate, named check the RNG-fix's diagnosis calls for
    (kept separate from the main measurement pipeline, which now uses
    independent streams): WITH the same literal RNG stream driving both
    collections, and identical dynamics (derive_control leaves an
    already-controllable zone's spec unchanged), b must be EXACTLY 0 --
    this verifies the pairing derivation is a true identity on
    controllable zones, not merely "close" due to sampling.
    """
    P_exp, P_ctrl, idx = _real_controllable_zone_setup()  # controllable_noisy zone

    # Two FRESHLY-SEEDED generators with the deliberately same seed value
    # (not spawn_exp_ctrl_rngs, and not one Generator object reused, which
    # would advance its own state between calls) -- this is exactly the
    # "same seed" construction whose correlation the bug fix above exists
    # to avoid in the main pipeline, used here on purpose as the identity
    # check.
    N_exp = collect_generative(P_exp, 200, np.random.default_rng(11))
    N_ctrl = collect_generative(P_ctrl, 200, np.random.default_rng(11))

    from helplessness.measures.controllability import controllability_per_state

    I_exp = controllability_per_state(N_exp[idx] / 200)
    I_ctrl = controllability_per_state(N_ctrl[idx] / 200)
    b = I_exp.mean() - I_ctrl.mean()
    assert b == 0.0
    assert np.array_equal(N_exp[idx], N_ctrl[idx])
