"""Tests for experiments/E1_offline_4zones/permutation_null.py: the
degenerate-zero sanity check the spec calls out explicitly (a bug if it
produces anything else), plus basic correctness of the Ihat and b nulls
and the detectable-effect percentile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from env import build_control_layout, build_env, build_layout  # noqa: E402
from estimation import collect_generative  # noqa: E402
from permutation_null import (  # noqa: E402
    _resample_rows_per_action_batch,
    bootstrap_null_b_zone,
    bootstrap_null_i_zone,
    detectable_effect,
    per_state_i_under_bootstrap_batch,
    permutation_null_b_zone,
    permutation_null_i_zone,
)

from helplessness.config import load_yaml_block

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"


def _zone_indices(layout, category):
    zone = next(z for z in layout.zones if z.appearance.category == category)
    return np.flatnonzero(zone.mask), zone


def test_degenerate_zero_null_in_the_uncontrollable_deterministic_zone():
    layout = build_layout(CFG, DIRECTION)
    P = build_env(layout).transition_tensor()
    idx, zone = _zone_indices(layout, "uncontrollable_deterministic")

    rng_collect = np.random.default_rng(0)
    N = collect_generative(P, 50, rng_collect)

    rng_perm = np.random.default_rng(1)
    null = permutation_null_i_zone(N, idx, n_permutations=50, rng=rng_perm)
    assert np.allclose(null, 0.0, atol=1e-9), "a deterministic zone's null must be exactly 0 under every permutation"


def test_noisy_zone_null_has_real_spread_and_is_nonnegative():
    layout = build_layout(CFG, DIRECTION)
    P = build_env(layout).transition_tensor()
    idx, zone = _zone_indices(layout, "uncontrollable_noisy")

    rng_collect = np.random.default_rng(2)
    N = collect_generative(P, 100, rng_collect)

    rng_perm = np.random.default_rng(3)
    null = permutation_null_i_zone(N, idx, n_permutations=100, rng=rng_perm)
    assert np.all(null >= -1e-9)
    assert null.std() > 0.0
    # the null should be small relative to a real, non-zero controllable
    # gap -- a loose sanity bound, not a tight tolerance
    assert null.mean() < 0.5


def test_detectable_effect_is_a_high_percentile_of_abs_null():
    null = np.array([-0.5, 0.1, 0.2, -0.3, 0.05, 0.9])
    d = detectable_effect(null, percentile=50.0)
    assert d == pytest.approx(np.percentile(np.abs(null), 50.0))


def test_b_null_uses_both_environments_and_is_not_degenerate_even_for_the_deterministic_pair():
    """The exp side (uncontrollable-det) is degenerate at 0, but its
    paired control (ControllableDeterministic, genuinely
    action-dependent) is not -- so b's null must show real spread, unlike
    Ihat's null on the exp side alone."""
    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "uncontrollable_deterministic")

    N_exp = collect_generative(P_exp, 50, np.random.default_rng(4))
    N_ctrl = collect_generative(P_ctrl, 50, np.random.default_rng(4))

    null_b = permutation_null_b_zone(N_exp, N_ctrl, idx, n_permutations=30, rng=np.random.default_rng(5))
    assert null_b.std() > 0.0


def test_permutation_null_i_zone_degenerate_at_full_budget_500_permutations():
    """Integration-style check at the spec's real n_permutations=500 and a
    representative budget, still fast (uncontrollable-deterministic
    degenerates trivially regardless of sample count)."""
    layout = build_layout(CFG, DIRECTION)
    P = build_env(layout).transition_tensor()
    idx, zone = _zone_indices(layout, "uncontrollable_deterministic")
    N = collect_generative(P, 300, np.random.default_rng(6))
    null = permutation_null_i_zone(N, idx, n_permutations=500, rng=np.random.default_rng(7))
    assert np.allclose(null, 0.0, atol=1e-9)


# ---------------------------------------------------------------------
# bootstrap null: variance-restoration relative to the shuffle null
# ---------------------------------------------------------------------


def test_bootstrap_null_i_zone_also_degenerate_zero_in_the_deterministic_zone():
    """Same sanity check as the shuffle null: resampling with replacement
    from a pooled distribution that is itself a single point mass changes
    nothing."""
    layout = build_layout(CFG, DIRECTION)
    P = build_env(layout).transition_tensor()
    idx, zone = _zone_indices(layout, "uncontrollable_deterministic")
    N = collect_generative(P, 100, np.random.default_rng(8))
    null = bootstrap_null_i_zone(N, idx, n_permutations=50, rng=np.random.default_rng(9))
    assert np.allclose(null, 0.0, atol=1e-9)


def test_bootstrap_null_b_zone_degenerate_at_minus_log2_5_for_the_uncontrollable_deterministic_pair():
    """Per-action resampling (see permutation_null.py's module docstring
    on the pooled-vs-per-action bootstrap redesign): resampling with
    replacement from a POINT MASS reproduces that exact point mass every
    time, deterministically -- true regardless of whether the point mass
    is the SAME for every action (uncontrollable_deterministic, exp side:
    Ihat_boot = 0 exactly) or DIFFERENT per action
    (ControllableDeterministic, the paired control: Ihat_boot = log2(5)
    exactly). Both sides are individually degenerate, so b's bootstrap
    null is degenerate too, at exactly -log2(5) -- not spread, and not 0."""
    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "uncontrollable_deterministic")

    N_exp = collect_generative(P_exp, 50, np.random.default_rng(10))
    N_ctrl = collect_generative(P_ctrl, 50, np.random.default_rng(11))
    null_b = bootstrap_null_b_zone(N_exp, N_ctrl, idx, n_permutations=30, rng=np.random.default_rng(12))
    assert np.allclose(null_b, -np.log2(5), atol=1e-9)


def test_bootstrap_null_b_zone_scales_slower_than_1_over_n_in_a_live_controllable_zone():
    """Regression test for the pooled-bootstrap bug: pooling actions
    before resampling nulled out the real action-dependence in
    controllable_noisy, making its detectability threshold scale like an
    independence-test statistic (~1/n, a 100x drop over a 100x budget
    increase) instead of ordinary estimator sampling noise (~1/sqrt(n), a
    ~10x drop). With per-action resampling the ratio between n=10 and
    n=1000's p95 must be well under the ~100x a 1/n law would predict."""
    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "controllable_noisy")

    p95s = {}
    for n in (10, 1000):
        N_exp = collect_generative(P_exp, n, np.random.default_rng(20))
        N_ctrl = collect_generative(P_ctrl, n, np.random.default_rng(21))
        null_b = bootstrap_null_b_zone(N_exp, N_ctrl, idx, n_permutations=200, rng=np.random.default_rng(22))
        p95s[n] = detectable_effect(null_b)
    ratio = p95s[10] / p95s[1000]
    assert ratio < 30  # well under a 1/n law's ~100x; consistent with ~1/sqrt(n)'s ~10x


def test_bootstrap_null_b_zone_degenerate_at_zero_for_the_controllable_deterministic_zone():
    """Both sides are ControllableDeterministic (derive_control is a
    no-op) -- each action's own point mass resamples to itself exactly on
    both sides, so Ihat_boot = log2(5) exactly on each side and b's
    bootstrap null is degenerate at exactly 0."""
    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "controllable_deterministic")

    N_exp = collect_generative(P_exp, 50, np.random.default_rng(13))
    N_ctrl = collect_generative(P_ctrl, 50, np.random.default_rng(14))
    null_b = bootstrap_null_b_zone(N_exp, N_ctrl, idx, n_permutations=30, rng=np.random.default_rng(15))
    assert np.allclose(null_b, 0.0, atol=1e-9)


# ---------------------------------------------------------------------
# rng.choice -> rng.multinomial equivalence: the performance fix changes
# HOW the RNG stream is consumed, so exact sample equality can't hold and
# isn't what's being checked -- what must hold is that the two
# implementations draw from the SAME distribution. A performance
# optimisation that silently changes a distribution is the worst possible
# bug at this stage of the project, per the explicit instruction that
# prompted this test.
# ---------------------------------------------------------------------


def _reference_choice_based_resample(N_s, n_per_action, n_actions, n_states, n_permutations, rng):
    """The PRE-optimisation implementation, reproduced here only for this
    equivalence test (not used anywhere in the pipeline any more) --
    draws n_per_action individual categorical samples per permutation via
    rng.choice(..., p=...), exactly as permutation_null.py did before the
    rng.multinomial rewrite."""
    counts = np.zeros((n_permutations, n_actions, n_states), dtype=np.int64)
    for a in range(n_actions):
        p_a = N_s[a] / N_s[a].sum()
        draws = rng.choice(n_states, size=(n_permutations, n_per_action), p=p_a)
        for state in range(n_states):
            counts[:, a, state] = (draws == state).sum(axis=1)
    return counts


def test_multinomial_optimization_matches_choice_based_reference_distribution():
    """Fixed-seed comparison of the two implementations' OUTPUT
    DISTRIBUTIONS on a toy case -- not sample-for-sample equality, which
    cannot hold (rng.multinomial and rng.choice(p=...) consume the
    underlying bit stream differently for the same seed). For each
    action, aggregates many permutations' count vectors from each
    implementation into one large per-state total and runs a chi-squared
    test of homogeneity (scipy.stats.chi2_contingency) between the two
    aggregates: a large p-value means the two samples are consistent with
    being drawn from the same underlying distribution.
    """
    n_actions, n_states = 3, 4
    n_per_action = 200
    # Distinct, non-uniform per-action distributions (rows sum to n_per_action)
    N_s = np.array(
        [
            [40, 80, 30, 50],
            [10, 10, 10, 170],
            [60, 60, 60, 20],
        ]
    )
    n_permutations = 4000

    old_counts = _reference_choice_based_resample(N_s, n_per_action, n_actions, n_states, n_permutations, np.random.default_rng(42))
    new_probs = _resample_rows_per_action_batch(N_s, n_per_action, n_actions, n_states, n_permutations, np.random.default_rng(42))
    new_counts = np.round(new_probs * n_per_action).astype(np.int64)

    for a in range(n_actions):
        old_aggregate = old_counts[:, a, :].sum(axis=0)  # (n_states,) total over all permutations
        new_aggregate = new_counts[:, a, :].sum(axis=0)
        assert old_aggregate.sum() == new_aggregate.sum() == n_permutations * n_per_action
        contingency = np.vstack([old_aggregate, new_aggregate])
        _, p_value, _, _ = chi2_contingency(contingency)
        assert p_value > 0.01, f"action {a}: distributions differ (p={p_value}) -- old={old_aggregate}, new={new_aggregate}"


# ---------------------------------------------------------------------
# The detectability scaling exponent, as an ASSERTION, not an observation
# (see report.py::fit_power_law_slope): now that the pooled-vs-per-action
# bootstrap root cause is identified and fixed, a future regression of the
# same kind (accidentally nulling out real action-dependence again) must
# fail a test, not just look wrong on a plot.
# ---------------------------------------------------------------------


def test_controllable_noisy_detectability_slope_is_near_minus_half():
    """controllable_noisy's bootstrap b-null p95, fit in log-log space
    across 5 budgets (3 seeds averaged per budget to damp per-seed
    noise): must land near -0.5 (ordinary estimator sampling noise around
    a non-zero true value), not near -1 (the pooled-bootstrap bug's
    independence-test-like scaling)."""
    from report import fit_power_law_slope

    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "controllable_noisy")

    from estimation import spawn_exp_ctrl_rngs

    budgets = (10, 30, 100, 300, 1000)
    p95_by_n = {}
    for n in budgets:
        p95s = []
        for seed in (0, 1, 2):
            rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(seed)
            N_exp = collect_generative(P_exp, n, rng_exp)
            N_ctrl = collect_generative(P_ctrl, n, rng_ctrl)
            null_b = bootstrap_null_b_zone(N_exp, N_ctrl, idx, 500, np.random.default_rng(seed * 7 + 1))
            p95s.append(detectable_effect(null_b))
        p95_by_n[n] = np.mean(p95s)

    slope, _ = fit_power_law_slope(list(budgets), [p95_by_n[n] for n in budgets])
    assert -0.75 <= slope <= -0.25, f"slope={slope}, expected near -0.5 (got p95_by_n={p95_by_n})"


def test_controllable_deterministic_detectability_null_stays_degenerate_across_budgets():
    """The companion assertion: controllable_deterministic's bootstrap
    b-null must be exactly 0 at every budget, not just one -- zero
    estimation noise for a deterministic transition is a structural
    property, not a coincidence at a single n."""
    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "controllable_deterministic")

    from estimation import spawn_exp_ctrl_rngs

    for n in (10, 100, 1000):
        rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(0)
        N_exp = collect_generative(P_exp, n, rng_exp)
        N_ctrl = collect_generative(P_ctrl, n, rng_ctrl)
        null_b = bootstrap_null_b_zone(N_exp, N_ctrl, idx, 200, np.random.default_rng(1))
        assert np.allclose(null_b, 0.0, atol=1e-9), f"n={n}: null not degenerate"


def test_p95_estimate_has_no_permutation_count_bias():
    """Regression guard for the investigation into why
    controllable_noisy's detectability slope came out shallower than
    -0.5: one candidate cause was that the p95 estimate itself, from only
    500 bootstrap draws, might carry a bias that doesn't shrink with n
    and so flattens the fitted slope. Checked (and ruled out) by
    comparing p95 at 500 vs a much larger permutation count on the same
    fixed dataset -- the MEAN must not move by more than sampling noise
    allows; only the spread should shrink with more permutations. Lighter
    replicate counts than the original investigation (which used 10/5)
    for test-suite speed, but the same comparison."""
    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "controllable_noisy")

    from estimation import spawn_exp_ctrl_rngs

    rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(0)
    N_exp = collect_generative(P_exp, 1000, rng_exp)
    N_ctrl = collect_generative(P_ctrl, 1000, rng_ctrl)

    p95s_small = [detectable_effect(bootstrap_null_b_zone(N_exp, N_ctrl, idx, 500, np.random.default_rng(1000 + r))) for r in range(4)]
    p95s_large = [detectable_effect(bootstrap_null_b_zone(N_exp, N_ctrl, idx, 3000, np.random.default_rng(2000 + r))) for r in range(3)]

    mean_small, sd_small = np.mean(p95s_small), np.std(p95s_small, ddof=1)
    mean_large = np.mean(p95s_large)
    # the large-count mean must land within a few small-count standard
    # errors of the small-count mean -- a systematic bias would show up
    # as a shift much larger than sampling noise explains
    se_small = sd_small / np.sqrt(len(p95s_small))
    assert abs(mean_large - mean_small) < 5 * se_small, (
        f"mean shifted from {mean_small} (n_perm=500) to {mean_large} (n_perm=3000) by more than "
        f"5 standard errors ({5 * se_small}) -- possible permutation-count bias"
    )


def test_per_state_slopes_are_not_shallower_than_the_zone_aggregate():
    """Regression guard for the third candidate cause investigated for
    controllable_noisy's shallower-than-(-0.5) detectability slope:
    aggregation of heterogeneous per-state rates. If individual states
    scaled at -0.5 with different amplitudes, averaging them could NOT
    itself produce a shallower exponent -- a linear combination of
    same-exponent power laws is exactly that exponent, amplitude-
    averaged. So if aggregation were the cause, the zone-aggregate slope
    would be shallower than the MEAN of individual per-state slopes.
    Checked directly (2 seeds, lighter than the original 5-seed
    investigation, for test-suite speed): it is not -- the aggregate and
    the per-state mean must land within a small tolerance of each other."""
    from report import fit_power_law_slope_ci

    layout = build_layout(CFG, DIRECTION)
    ctrl_layout = build_control_layout(CFG, DIRECTION)
    P_exp = build_env(layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()
    idx, zone = _zone_indices(layout, "controllable_noisy")

    from estimation import spawn_exp_ctrl_rngs

    budgets = (10, 30, 100, 300, 1000)
    seeds = (0, 1)
    per_state_p95_by_n = {}
    for n in budgets:
        accum = []
        for seed in seeds:
            rng_exp, rng_ctrl = spawn_exp_ctrl_rngs(seed)
            N_exp = collect_generative(P_exp, n, rng_exp)
            N_ctrl = collect_generative(P_ctrl, n, rng_ctrl)
            exp_per_state = per_state_i_under_bootstrap_batch(N_exp, idx, 300, np.random.default_rng(seed * 11 + 1))
            ctrl_per_state = per_state_i_under_bootstrap_batch(N_ctrl, idx, 300, np.random.default_rng(seed * 11 + 2))
            b_per_state = exp_per_state - ctrl_per_state  # (zone_size, 300)
            accum.append(np.percentile(np.abs(b_per_state), 95, axis=1))
        per_state_p95_by_n[n] = np.mean(accum, axis=0)  # (zone_size,)

    zone_size = idx.size
    per_state_slopes = []
    for i in range(zone_size):
        vals = [per_state_p95_by_n[n][i] for n in budgets]
        slope, _, _ = fit_power_law_slope_ci(list(budgets), vals)
        per_state_slopes.append(slope)
    per_state_mean_slope = float(np.mean(per_state_slopes))

    agg_p95_by_n = [per_state_p95_by_n[n].mean() for n in budgets]
    agg_slope, _, _ = fit_power_law_slope_ci(list(budgets), agg_p95_by_n)

    # the aggregate must not be meaningfully shallower than the per-state
    # mean -- a wide but real tolerance (0.1 on the slope), since this is
    # only 2 seeds/300 perms and individual per-state slope estimates are
    # noisy on their own
    assert agg_slope <= per_state_mean_slope + 0.1, (
        f"zone-aggregate slope ({agg_slope}) is shallower than the per-state mean slope "
        f"({per_state_mean_slope}) by more than the aggregation-artifact hypothesis would predict is safe"
    )
