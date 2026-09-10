"""Consistency checks for helplessness.interventions.controllability_dial --
the closed-form identity must be verified, never merely assumed (cf. module
docstring)."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.interventions.controllability_dial import (
    clip_diagnostics,
    controllability_dial,
    marginalise,
    p_effective,
    structure_injection,
)
from helplessness.measures.controllability import controllability_from_tensor


@pytest.fixture
def T0():
    return GridWorld(p=0.0, seed=0).transition_tensor()


def test_dial_lam_zero_returns_unchanged(T0):
    T_lam, clipped = controllability_dial(T0, 0.0)
    assert np.array_equal(T_lam, T0)
    assert clipped == 0.0


def test_dial_lam_one_all_actions_identical(T0):
    T_lam, clipped = controllability_dial(T0, 1.0)
    for a in range(1, T_lam.shape[1]):
        assert np.allclose(T_lam[:, a, :], T_lam[:, 0, :])
    assert clipped == 0.0


def test_perceived_controllability_at_lam_one_is_zero(T0):
    terminal = GridWorld(p=0.0, seed=0).terminal_mask()
    T_lam, _ = controllability_dial(T0, 1.0)
    assert controllability_from_tensor(T_lam, terminal) == pytest.approx(0.0, abs=1e-9)


def test_marginal_invariant_across_p():
    env = GridWorld(p=0.0, seed=0)
    T_det = env.transition_tensor(0.0)
    M_ref = marginalise(T_det)
    for p in (0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0):
        Tp = env.transition_tensor(p)
        assert np.allclose(marginalise(Tp), M_ref, atol=1e-12)


@pytest.mark.parametrize("lam", [0.0, 0.15, 0.3, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0])
def test_no_clipping_and_valid_simplex_for_lam_in_0_1(T0, lam):
    T_lam, clipped = controllability_dial(T0, lam)
    assert clipped == 0.0
    assert np.all(T_lam >= 0.0)
    assert np.allclose(T_lam.sum(axis=2), 1.0)


@pytest.mark.parametrize("lam", [0.0, 0.15, 0.3, 0.45, 0.5, 0.6, 0.7, 0.85, 1.0])
def test_dial_on_ground_truth_matches_transition_tensor_at_p_equals_lam(lam):
    env = GridWorld(p=0.0, seed=0)
    T0 = env.transition_tensor(0.0)
    T_lam, _ = controllability_dial(T0, lam)
    T_p = env.transition_tensor(lam)
    assert np.allclose(T_lam, T_p, atol=1e-12)


def test_negative_lam_reports_nonzero_clipping_when_it_occurs():
    env = GridWorld(p=0.0, seed=0)
    T0 = env.transition_tensor(0.0)
    T_lam, clipped = controllability_dial(T0, -0.5)
    assert clipped > 0.0
    assert np.all(T_lam >= 0.0)
    assert np.allclose(T_lam.sum(axis=2), 1.0)


def test_p_effective_formula():
    assert p_effective(0.0, 0.7) == pytest.approx(0.7)
    assert p_effective(0.7, 0.0) == pytest.approx(0.7)
    assert p_effective(0.7, 1.0) == pytest.approx(1.0)
    assert p_effective(0.5, -0.5) == pytest.approx(0.25)


# ----------------------------------------------------------------------
# structure_injection
# ----------------------------------------------------------------------
@pytest.fixture
def T_hat_like(T0):
    # a plausible "learned" belief: T0 slightly blurred towards UNIFORM
    # (not towards its own marginal M -- blending T0 with M0 would leave the
    # result's marginal equal to M0 by construction, a degenerate case that
    # would invalidate the "does not preserve the marginal" test below).
    n_states = T0.shape[0]
    uniform = np.full_like(T0, 1.0 / n_states)
    return 0.85 * T0 + 0.15 * uniform


def test_structure_injection_mu_zero_returns_T_hat_unchanged(T_hat_like, T0):
    T_mu = structure_injection(T_hat_like, T0, 0.0)
    assert np.array_equal(T_mu, T_hat_like)


def test_structure_injection_mu_one_returns_T_det_exactly(T_hat_like, T0):
    T_mu = structure_injection(T_hat_like, T0, 1.0)
    assert np.array_equal(T_mu, T0)


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
def test_structure_injection_valid_simplex_no_clip_needed(T_hat_like, T0, mu):
    T_mu = structure_injection(T_hat_like, T0, mu)
    assert np.all(T_mu >= 0.0)
    assert np.allclose(T_mu.sum(axis=2), 1.0)


def test_structure_injection_does_not_preserve_marginal(T_hat_like, T0):
    # unlike controllability_dial, nothing guarantees mean_a T_mu equals the
    # marginal of T_hat -- checks that this is NOT the case in general, to
    # avoid conflating the two interventions.
    M_hat = marginalise(T_hat_like)
    T_mu = structure_injection(T_hat_like, T0, 0.5)
    assert not np.allclose(marginalise(T_mu), M_hat)


# ----------------------------------------------------------------------
# clip_diagnostics
# ----------------------------------------------------------------------
def test_clip_diagnostics_mass_grows_with_lam_magnitude_but_fraction_does_not(T0):
    small = clip_diagnostics(T0, -0.01)
    large = clip_diagnostics(T0, -0.9)
    assert small["clipped_fraction"] == pytest.approx(large["clipped_fraction"])
    assert small["clipped_mass"] < large["clipped_mass"]


def test_clip_diagnostics_zero_for_lam_in_0_1(T0):
    d = clip_diagnostics(T0, 0.3)
    assert d["clipped_fraction"] == 0.0
    assert d["clipped_mass"] == pytest.approx(0.0, abs=1e-12)
