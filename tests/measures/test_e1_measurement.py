"""Tests for experiments/E1_offline_4zones/measurement.py, on the real E1
layout at the frozen config (gamma=0.9, R=50, away_from_goal): convergence
of Ihat -> I* per zone type, exactness at n=1 in deterministic zones, and
Delta >= 0 -- the properties the spec's tolerances section requires.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from env import build_control_layout, build_env, build_layout, reward_and_terminal  # noqa: E402
from estimation import agent_from_counts, collect_generative  # noqa: E402
from measurement import aggregate_zone_region, compute_state_and_pair_arrays  # noqa: E402

GAMMA = 0.9
GOAL_REWARD = 50.0
DIRECTION = "away_from_goal"


def _cfg():
    from helplessness.config import load_yaml_block

    return load_yaml_block("zones/e1_4zones.yaml")


CFG = _cfg()
GRID_SHAPE = tuple(CFG["grid_shape"])


def _exp_setup():
    layout = build_layout(CFG, DIRECTION)
    P = build_env(layout).transition_tensor()
    R, terminal = reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
    return layout, P, R, terminal


@pytest.mark.parametrize(
    "zone_category",
    ["uncontrollable_deterministic", "controllable_noisy", "controllable_deterministic", "uncontrollable_noisy"],
)
def test_i_hat_converges_to_i_star_as_n_grows(zone_category):
    layout, P, R, terminal = _exp_setup()
    zone = next(z for z in layout.zones if z.appearance.category == zone_category)
    idx = np.flatnonzero(zone.mask)

    I_star_true = float(compute_arrays_i_star_only(P)[idx].mean())

    errors = []
    for n in (10, 100, 1000):
        rng = np.random.default_rng(0)
        counts = collect_generative(P, n, rng)
        agent = agent_from_counts(counts, R, alpha=0.0)
        arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, agent.T_hat, agent.R_hat)
        errors.append(abs(arrays["I_hat"][idx].mean() - I_star_true))
    if "deterministic" in zone_category:
        # exact at every n (see the dedicated exactness test below) -- error is
        # identically 0, so "shrinks" doesn't apply; nothing more to check here.
        assert all(e == pytest.approx(0.0, abs=1e-9) for e in errors)
    else:
        # error should shrink (not necessarily monotonically at small n, but n=1000 must clearly beat n=10)
        assert errors[-1] < errors[0]
        assert errors[-1] < 0.05  # loose bound; the spec's tight 0.03-bit tolerance is checked at n=1000 specifically below


def compute_arrays_i_star_only(P):
    from helplessness.measures.controllability import controllability_per_state

    return controllability_per_state(P)


@pytest.mark.parametrize("zone_category", ["uncontrollable_deterministic", "controllable_deterministic"])
def test_i_hat_exact_at_n_equals_one_in_deterministic_zones(zone_category):
    layout, P, R, terminal = _exp_setup()
    zone = next(z for z in layout.zones if z.appearance.category == zone_category)
    idx = np.flatnonzero(zone.mask)
    I_star = compute_arrays_i_star_only(P)

    rng = np.random.default_rng(1)
    counts = collect_generative(P, 1, rng)
    agent = agent_from_counts(counts, R, alpha=0.0)
    I_hat = controllability_per_state_local(agent.T_hat)
    assert np.allclose(I_hat[idx], I_star[idx], atol=1e-9)


def controllability_per_state_local(T_hat):
    from helplessness.measures.controllability import controllability_per_state

    return controllability_per_state(T_hat)


def test_i_hat_within_tolerance_at_n_1000_for_noisy_zones():
    layout, P, R, terminal = _exp_setup()
    I_star = compute_arrays_i_star_only(P)
    rng = np.random.default_rng(2)
    counts = collect_generative(P, 1000, rng)
    agent = agent_from_counts(counts, R, alpha=0.0)
    I_hat = controllability_per_state_local(agent.T_hat)
    for category in ("controllable_noisy", "uncontrollable_noisy"):
        zone = next(z for z in layout.zones if z.appearance.category == category)
        idx = np.flatnonzero(zone.mask)
        assert abs(I_hat[idx].mean() - I_star[idx].mean()) < 0.03


def test_delta_positive_control_uniform_random_t_hat():
    """A test that only ever proves Delta >= 0 (already covered above)
    can't distinguish a working measurement from a broken one that always
    returns ~0 -- this is the other half: a deliberately WRONG T_hat
    (uniform over every successor state, discarding the kernel entirely)
    must produce a substantially positive Delta, not silently pass through
    near-zero. Checked against the actual computed magnitude (~27 mean,
    ~60 max at this R/gamma), not a guessed threshold."""
    layout, P, R, terminal = _exp_setup()
    n_states, n_actions, _ = P.shape
    T_bad = np.full_like(P, 1.0 / n_states)
    arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, T_bad, R)
    assert arrays["delta"].mean() > 5.0
    assert arrays["delta"].max() > 10.0
    assert np.all(arrays["delta"] >= -1e-6)  # still nonnegative, per the fundamental MDP fact


def test_delta_positive_control_permuted_action_axis():
    """A subtler corruption: the marginal transition distributions are all
    real (each row IS some action's true row), but attached to the wrong
    action label -- this specifically stresses whether Delta detects a
    wrong POLICY from action mislabeling, not just a wrong kernel."""
    layout, P, R, terminal = _exp_setup()
    n_actions = P.shape[1]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_actions)
    while np.array_equal(perm, np.arange(n_actions)):
        perm = rng.permutation(n_actions)
    T_bad = P[:, perm, :]
    arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, T_bad, R)
    assert arrays["delta"].mean() > 5.0
    assert np.all(arrays["delta"] >= -1e-6)


@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_delta_nonnegative_on_the_real_layout(n, seed):
    layout, P, R, terminal = _exp_setup()
    rng = np.random.default_rng(seed)
    counts = collect_generative(P, n, rng)
    agent = agent_from_counts(counts, R, alpha=0.0)
    arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, agent.T_hat, agent.R_hat)
    assert np.all(arrays["delta"] >= -1e-6)


def test_delta_small_relative_to_v_star_on_interior_at_n_1000():
    layout, P, R, terminal = _exp_setup()
    rng = np.random.default_rng(3)
    counts = collect_generative(P, 1000, rng)
    agent = agent_from_counts(counts, R, alpha=0.0)
    arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, agent.T_hat, agent.R_hat)
    from helplessness.planning.value_iteration import value_iteration

    V_star, _ = value_iteration(P, R, terminal, GAMMA)
    for zone in layout.zones:
        if zone.appearance.category == "corridor":
            continue
        row = aggregate_zone_region(arrays, zone, GRID_SHAPE, "interior")
        idx_v = V_star[np.flatnonzero(zone.mask)]
        v_star_mean = np.abs(idx_v).mean()
        if v_star_mean > 0:
            assert row["delta"] / v_star_mean < 0.01, f"{zone.appearance.category}: {row['delta']} / {v_star_mean}"


# ---------------------------------------------------------------------
# aggregate_zone_region: NaN handling for I_hat/I_star/l1 outside "all"
# ---------------------------------------------------------------------


def test_aggregate_zone_region_nans_i_hat_outside_all():
    layout, P, R, terminal = _exp_setup()
    rng = np.random.default_rng(4)
    counts = collect_generative(P, 100, rng)
    agent = agent_from_counts(counts, R, alpha=0.0)
    arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, agent.T_hat, agent.R_hat)
    zone = next(z for z in layout.zones if z.appearance.category == "uncontrollable_deterministic")

    row_all = aggregate_zone_region(arrays, zone, GRID_SHAPE, "all")
    row_interior = aggregate_zone_region(arrays, zone, GRID_SHAPE, "interior")
    row_boundary = aggregate_zone_region(arrays, zone, GRID_SHAPE, "boundary")

    assert not np.isnan(row_all["I_hat"])
    assert np.isnan(row_interior["I_hat"])
    assert np.isnan(row_boundary["I_hat"])
    assert row_interior["n"] == 9
    assert row_boundary["n"] == 25 - 9
    assert row_all["n"] == 25


def test_aggregate_zone_region_rejects_unknown_region():
    layout, P, R, terminal = _exp_setup()
    rng = np.random.default_rng(5)
    counts = collect_generative(P, 10, rng)
    agent = agent_from_counts(counts, R, alpha=0.0)
    arrays = compute_state_and_pair_arrays(P, R, terminal, GAMMA, agent.T_hat, agent.R_hat)
    zone = layout.zones[0]
    with pytest.raises(ValueError, match="region"):
        aggregate_zone_region(arrays, zone, GRID_SHAPE, "nowhere")
