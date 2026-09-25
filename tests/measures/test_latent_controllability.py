"""Analytical, numpy-only tests for measures.latent_controllability
(spec section 9, phase-3 point 3a): hand-built factorised distributions
with known closed-form answers, checked through the ACTUAL MC codepath
(not a separate hand-rolled estimator), plus the se ~ 1/sqrt(M) scaling
and the unimix-floor cross-check against an existing analytic formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures.latent_controllability import (
    conditional_entropy_z_mean,
    latent_controllability,
    marginal_entropy_z_mc,
)


def _delta(n_vars: int, n_classes: int, class_idx: int) -> np.ndarray:
    """A single product-distribution: every variable is a pure delta on
    class_idx -- H(z'|s,a) = 0 exactly for this action."""
    p = np.zeros((n_vars, n_classes))
    p[:, class_idx] = 1.0
    return p


def test_five_identical_action_distributions_give_i_zero():
    """Every action induces the SAME (non-trivial, stochastic) z
    distribution -> the mixture equals that distribution itself -> I_z
    == 0 exactly (up to MC noise, which is non-zero here since the
    per-action distribution is genuinely stochastic)."""
    rng = np.random.default_rng(0)
    n_vars, n_classes = 2, 4
    single = np.random.default_rng(1).dirichlet(np.ones(n_classes), size=n_vars)
    prior = np.stack([single] * 5, axis=0)  # (5, n_vars, n_classes), all identical

    result = latent_controllability(prior, M=20_000, rng=rng)
    assert result["I_z"] == pytest.approx(0.0, abs=6 * result["se"] + 1e-6)


def test_five_disjoint_one_hot_actions_give_i_log2_5_exactly():
    """Each action is a pure delta on a DISTINCT class (n_vars=1,
    n_classes=5): H(z'|s,a) = 0 for every a, and the mixture is a
    uniform distribution over 5 disjoint outcomes -> H(z'|s) = log2(5)
    exactly. Every MC sample of z is deterministic given its action, so
    log m(z) = log2(1/5) for EVERY sample -- zero MC variance, giving an
    EXACT (not just tolerance-bounded) match."""
    rng = np.random.default_rng(2)
    n_vars, n_classes = 1, 5
    prior = np.stack([_delta(n_vars, n_classes, a) for a in range(5)], axis=0)

    result = latent_controllability(prior, M=2000, rng=rng)
    assert result["se"] == pytest.approx(0.0, abs=1e-9)
    assert result["I_z"] == pytest.approx(np.log2(5), abs=1e-9)
    assert result["H_conditional_mean"] == pytest.approx(0.0, abs=1e-9)


def test_hand_computed_intermediate_case():
    """n_vars=1, n_classes=2: 3 actions delta on class 0, 2 actions
    delta on class 1. Mixture: m(0)=3/5, m(1)=2/5. H(m) = -(0.6*log2(0.6)
    + 0.4*log2(0.4)) ~= 0.970951 bits, computed by hand here and
    independently by numpy -- H(z'|s,a)=0 for every action (all deltas),
    so I_z == H(m) exactly (again zero MC variance: every sample of z is
    one of exactly 2 deterministic outcomes)."""
    rng = np.random.default_rng(3)
    n_vars, n_classes = 1, 2
    prior = np.stack(
        [_delta(n_vars, n_classes, 0)] * 3 + [_delta(n_vars, n_classes, 1)] * 2, axis=0
    )

    expected_H = -(0.6 * np.log2(0.6) + 0.4 * np.log2(0.4))
    assert expected_H == pytest.approx(0.970951, abs=1e-5)  # the well-known binary-entropy(0.6) value

    # NOT zero MC variance here (unlike the 5-disjoint-outcome case
    # above): z only takes 2 distinct values (0 or 1), and m(z) differs
    # between them (0.6 vs 0.4), so log_m(Z) genuinely varies across
    # samples even though the ESTIMATOR is unbiased -- check within a
    # generous multiple of the reported se instead of exact equality.
    result = latent_controllability(prior, M=20_000, rng=rng)
    assert result["I_z"] == pytest.approx(expected_H, abs=6 * result["se"] + 1e-6)


def test_mc_standard_error_scales_as_one_over_sqrt_m():
    """Uses a genuinely stochastic (non-delta) case so there is real MC
    variance to measure. Compares the EMPIRICAL spread of repeated
    point estimates at M and at 4M -- should shrink by ~2x (form test,
    not an exact-value test, generous tolerance given the small number
    of repeats)."""
    n_vars, n_classes = 2, 4
    base_rng = np.random.default_rng(4)
    prior = base_rng.dirichlet(np.ones(n_classes), size=(5, n_vars))

    def repeated_H_hat(M: int, n_repeats: int, seed: int) -> np.ndarray:
        estimates = []
        for i in range(n_repeats):
            r = np.random.default_rng(seed + i)
            marg = marginal_entropy_z_mc(prior, M=M, rng=r)
            estimates.append(marg["H_hat"])
        return np.array(estimates)

    n_repeats = 300
    small_M = 200
    large_M = 800  # 4x

    std_small = repeated_H_hat(small_M, n_repeats, seed=1000).std()
    std_large = repeated_H_hat(large_M, n_repeats, seed=2000).std()

    ratio = std_small / std_large
    expected_ratio = np.sqrt(large_M / small_M)  # = 2.0
    assert ratio == pytest.approx(expected_ratio, rel=0.5)  # generous: form test, not exact


def test_reported_se_is_consistent_with_the_empirical_spread_of_repeated_estimates():
    n_vars, n_classes = 2, 3
    prior = np.random.default_rng(5).dirichlet(np.ones(n_classes), size=(5, n_vars))
    M = 500

    reported_ses = []
    point_estimates = []
    for i in range(200):
        r = np.random.default_rng(3000 + i)
        marg = marginal_entropy_z_mc(prior, M=M, rng=r)
        reported_ses.append(marg["se"])
        point_estimates.append(marg["H_hat"])

    empirical_se = np.std(point_estimates)
    mean_reported_se = np.mean(reported_ses)
    assert mean_reported_se == pytest.approx(empirical_se, rel=0.3)


def test_unimix_floor_matches_a_hand_computed_value():
    """What is compared: n_classes = n_actions = 5, n_vars = 1. Before
    unimix, action a's z-distribution is a pure delta on class a
    (_delta(), same construction as the disjoint-one-hot test above).
    apply_unimix(ratio=r) replaces it with p_a(z=a) = (1-r) + r/5 = q,
    p_a(z=b) = r/5 for the 4 other classes b != a -- spec 3.1's 1%
    floor, generalised here to r=0.1 for a visibly non-trivial effect.

    By-hand derivation of the expected I_z (r = 0.1, n = 5), computed
    independently of BOTH measures.latent_controllability (the code
    under test) AND envs.dynamics.controllable.ControllableNoisy
    (structurally the same mixture, but a SEPARATE closed-form
    derivation elsewhere in the codebase -- not reused here, so this
    test cannot pass merely because two implementations share a bug):

      q = (1 - r) + r/n = 0.9 + 0.02 = 0.92

      H(z'|s,a) is exact and identical for every action (by the
      construction's symmetry): the entropy of a single categorical
      with one outcome at probability q and the other (n-1) at r/n each:
        H(z'|s,a) = -[ q*log2(q) + (n-1)*(r/n)*log2(r/n) ]

      The MARGINAL m(z) needs no Monte Carlo to get exactly: the
      construction is symmetric under jointly permuting (action, class)
      labels (action a's distribution is the SAME shape centred on
      class a, for every a), so every one of the n classes receives
      EXACTLY the same total mixture mass under the uniform-action
      average -- m(z) = 1/n for every z, i.e. h(z') is the UNIFORM
      distribution over n outcomes, exactly:
        H(z'|s) = log2(n)

      So, exactly (no MC needed for this construction, though the test
      below still runs the REAL MC-based implementation and checks it
      recovers this value):
        I_z = H(z'|s) - H(z'|s,a)
            = log2(n) + q*log2(q) + (n-1)*(r/n)*log2(r/n)
            = log2(5) + 0.92*log2(0.92) + 4*0.02*log2(0.02)

      Plugging in (verified by hand with a calculator, log2(x) = ln(x)/ln(2)):
        log2(5)          =  2.321928094887362
        0.92*log2(0.92)  = -0.110670695020295   (log2(0.92) = -0.120294...)
        4*0.02*log2(0.02) = -0.451508495181978   (log2(0.02) = -5.643856...)
        I_z = 2.321928094887362 - 0.110670695020295 - 0.451508495181978
            = 1.759748904685089
    """
    import torch

    from helplessness.agents.worldmodel.rssm import apply_unimix

    n_actions = 5
    n_vars, n_classes = 1, n_actions
    ratio = 0.1
    delta_prior = np.stack([_delta(n_vars, n_classes, a) for a in range(n_actions)], axis=0)

    # apply_unimix is torch (the RSSM's own function, deliberately
    # reused here rather than re-derived) -- convert at the boundary,
    # latent_controllability itself stays pure numpy.
    mixed = apply_unimix(torch.as_tensor(delta_prior), ratio).numpy()

    rng = np.random.default_rng(6)
    result = latent_controllability(mixed, M=200_000, rng=rng, bounds_tol_se_multiplier=8.0)

    HAND_COMPUTED_I_Z = 1.759748904685089  # see derivation above -- a literal, not a formula
    assert result["I_z"] == pytest.approx(HAND_COMPUTED_I_Z, abs=8 * result["se"] + 1e-3)


def test_conditional_entropy_z_mean_matches_hand_computation_for_a_simple_case():
    n_vars, n_classes = 1, 2
    # action 0: certain class 0 (H=0); action 1: uniform (H=1 bit)
    prior = np.stack([_delta(n_vars, n_classes, 0), np.full((n_vars, n_classes), 0.5)], axis=0)
    H_mean = conditional_entropy_z_mean(prior)
    assert H_mean == pytest.approx(0.5, abs=1e-9)  # (0 + 1) / 2


# --- I(z; s), posterior-collapse diagnostic (docs/e3/DECISIONS.md, 2026-09-24) ---


def test_latent_state_information_is_log2_n_states_for_disjoint_state_codes():
    """8 states, each posterior a pure delta on its own class (n_vars=1,
    n_classes=8): z identifies the state exactly -> I(z;s) = log2(8) = 3
    bits exactly (zero MC variance: every sample's mixture density is 1/8)."""
    from helplessness.measures.latent_controllability import latent_state_information

    n_states = 8
    posteriors = np.stack([_delta(1, n_states, s) for s in range(n_states)], axis=0)
    result = latent_state_information(posteriors, M=2000, rng=np.random.default_rng(0))
    assert result["I_zs"] == pytest.approx(3.0, abs=1e-9)


def test_latent_state_information_is_zero_under_collapse():
    """Every state gets the SAME posterior (the collapse signature): the
    posterior carries no information about the state -> I(z;s) = 0."""
    from helplessness.measures.latent_controllability import latent_state_information

    single = np.random.default_rng(1).dirichlet(np.ones(4), size=3)  # (n_vars=3, n_classes=4)
    posteriors = np.stack([single] * 144, axis=0)
    result = latent_state_information(posteriors, M=20_000, rng=np.random.default_rng(2))
    assert result["I_zs"] == pytest.approx(0.0, abs=6 * result["se"] + 1e-6)


def test_latent_controllability_and_latent_state_information_share_one_core():
    """Same component array -> same number: one implementation of the
    mixture mutual information, whatever the components stand for."""
    from helplessness.measures.latent_controllability import latent_state_information

    probs = np.random.default_rng(3).dirichlet(np.ones(4), size=(5, 2))
    a = latent_controllability(probs, M=5000, rng=np.random.default_rng(4))["I_z"]
    b = latent_state_information(probs, M=5000, rng=np.random.default_rng(4))["I_zs"]
    assert a == b
