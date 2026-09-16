"""Tests for measures.controllability: the regression pin below is the
guard CLAUDE.md's "single implementation" contract needs before
controllability_per_state()/conditional_entropy_per_state() are extracted
-- controllability_from_tensor() is a frozen-E0 dependency, and a pure loop
-> vectorized-array refactor is exactly the kind of change floating-point
reassociation can shift by a bit or two. This file is written and must pass
against the *current* implementation before that refactor lands, then must
still pass, byte-for-byte, afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures.controllability import (
    conditional_entropy_from_tensor,
    conditional_entropy_per_state,
    controllability_from_tensor,
    controllability_per_state,
)

# A small, hand-built, non-trivial reference tensor -- 3 states, 2 actions.
# Not derived from any zone/dynamics class, so it exercises the measure in
# isolation from everything else that changed in this delivery.
_P = np.array(
    [
        [[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
)


def test_controllability_from_tensor_regression_one_terminal():
    terminal = np.array([False, False, True])
    assert controllability_from_tensor(_P, terminal) == 0.6580670999158466


def test_controllability_from_tensor_regression_all_terminal():
    terminal = np.array([True, True, True])
    assert controllability_from_tensor(_P, terminal) == 0.0


def test_controllability_from_tensor_regression_none_terminal():
    terminal = np.array([False, False, False])
    assert controllability_from_tensor(_P, terminal) == 0.4387113999438977


def test_controllability_from_tensor_regression_custom_prior():
    terminal = np.array([False, False, False])
    prior = np.array([0.7, 0.3])
    assert controllability_from_tensor(_P, terminal, action_prior=prior) == 0.3982415588007047


# ---------------------------------------------------------------------
# controllability_per_state: the new primitive, and the contract that
# controllability_from_tensor is exactly its terminal-masked mean.
# ---------------------------------------------------------------------


def test_controllability_from_tensor_is_the_masked_mean_of_per_state():
    terminal = np.array([False, True, False])
    per_state = controllability_per_state(_P)
    expected = per_state[~terminal].mean()
    assert controllability_from_tensor(_P, terminal) == pytest.approx(expected, abs=1e-12)


def test_controllability_per_state_shape_and_all_terminal_scalar_matches():
    per_state = controllability_per_state(_P)
    assert per_state.shape == (3,)
    # scalar wrapper with everything terminal must still be the documented 0.0,
    # not nan from an empty mean.
    terminal = np.array([True, True, True])
    assert controllability_from_tensor(_P, terminal) == 0.0


def test_controllability_per_state_respects_custom_prior():
    prior = np.array([0.7, 0.3])
    per_state = controllability_per_state(_P, action_prior=prior)
    terminal = np.array([False, False, False])
    assert controllability_from_tensor(_P, terminal, action_prior=prior) == pytest.approx(
        per_state.mean(), abs=1e-12
    )


# ---------------------------------------------------------------------
# conditional_entropy_{per_state,from_tensor}: H(s'|s,a), same structure.
# ---------------------------------------------------------------------


def test_conditional_entropy_is_zero_for_deterministic_rows():
    # state 1: both actions are point masses -> H(s'|s,a) = 0 regardless of prior.
    per_state = conditional_entropy_per_state(_P)
    assert per_state[1] == pytest.approx(0.0, abs=1e-12)


def test_conditional_entropy_from_tensor_is_the_masked_mean_of_per_state():
    terminal = np.array([False, True, False])
    per_state = conditional_entropy_per_state(_P)
    expected = per_state[~terminal].mean()
    assert conditional_entropy_from_tensor(_P, terminal) == pytest.approx(expected, abs=1e-12)


def test_conditional_entropy_matches_hand_computed_value_for_state_0():
    # H(s'|s=0,a) = prior(0)*H(0.5,0.5,0) + prior(1)*H(0.2,0.3,0.5), uniform prior.
    def h(p):
        p = np.array([x for x in p if x > 0])
        return float(-(p * np.log2(p)).sum())

    expected = 0.5 * h([0.5, 0.5]) + 0.5 * h([0.2, 0.3, 0.5])
    per_state = conditional_entropy_per_state(_P)
    assert per_state[0] == pytest.approx(expected, abs=1e-12)
