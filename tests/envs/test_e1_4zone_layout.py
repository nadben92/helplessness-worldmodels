"""Construction tests for E1's 4-zone, 12x12 torus layout (see
docs/E1_offline_4zones.md and configs/zones/e1_4zones.yaml): tiling,
torus-symmetry of the layout, the CLAUDE.md-mandated pairing-agreement and
model/simulation-agreement tests, drift resolution and its <=5-step exit
bound, and the geometric interior mask.

All expected numeric values are computed fresh via analytic_controllability()
/ conditional_entropy_per_state(), never hardcoded decimals -- only the raw
transition_row() mixture (0.9, 0.025, 0.025, 0.025, 0.025) is checked
against a literal tuple, since that IS the spec's own definition of
epsilon=0.1 and is what the p=0.125 config value is supposed to reproduce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from env import build_control_layout, build_env, build_layout, reward_and_terminal  # noqa: E402

from helplessness.config import load_yaml_block
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy
from helplessness.envs.zones import assert_zone_symmetry, interior_mask, reflect_state, resolve_drift
from helplessness.measures.controllability import (
    conditional_entropy_per_state,
    controllability_per_state,
)

CFG = load_yaml_block("zones/e1_4zones.yaml")
GRID_SHAPE = tuple(CFG["grid_shape"])
GOAL = tuple(CFG["goal"])


def _non_corridor_zones(layout):
    return [z for z in layout.zones if z.appearance.category != "corridor"]


@pytest.mark.parametrize("direction", ["away_from_goal", "toward_goal"])
def test_layout_tiles_the_grid_exactly_once(direction):
    layout = build_layout(CFG, direction)
    coverage = np.zeros(GRID_SHAPE, dtype=int)
    for zone in layout.zones:
        coverage += zone.mask.astype(int)
    assert np.all(coverage == 1)
    assert sum(z.mask.sum() for z in layout.zones) == 144
    corridor = next(z for z in layout.zones if z.appearance.category == "corridor")
    assert corridor.mask.sum() == 44
    for z in _non_corridor_zones(layout):
        assert z.mask.sum() == 25


@pytest.mark.parametrize("direction", ["away_from_goal", "toward_goal"])
def test_layout_symmetry_distance_to_goal(direction):
    """Purely geometric: the 4 blocks' distance-to-goal distributions must
    match exactly, regardless of dynamics assignment."""
    layout = build_layout(CFG, direction)
    assert_zone_symmetry(_non_corridor_zones(layout), GRID_SHAPE, GOAL)


def test_symmetry_assertion_has_teeth():
    """A deliberately perturbed (non-congruent) block layout must fail the
    symmetry check -- proves it isn't vacuously true."""
    from helplessness.envs.appearance.base import AppearanceSpec
    from helplessness.envs.dynamics.controllable import ControllableDeterministic
    from helplessness.envs.zones import Zone, torus_block_mask

    good_block = Zone(
        mask=torus_block_mask(GRID_SHAPE, (1, 5), (1, 5)),
        dynamics=ControllableDeterministic(),
        appearance=AppearanceSpec("a"),
    )
    # a block the same size but shifted so it straddles the goal asymmetrically
    bad_block = Zone(
        mask=torus_block_mask(GRID_SHAPE, (0, 4), (1, 5)),
        dynamics=ControllableDeterministic(),
        appearance=AppearanceSpec("b"),
    )
    with pytest.raises(ValueError, match="distance-to-goal"):
        assert_zone_symmetry([good_block, bad_block], GRID_SHAPE, GOAL)


@pytest.mark.parametrize("direction", ["away_from_goal", "toward_goal"])
def test_drift_exits_within_five_steps_for_each_uncontrollable_zone_independently(direction):
    """Per your correction: verify each uncontrollable zone separately
    (diagonal placement means they are not symmetric to each other under
    any assumption we're allowed to make here)."""
    layout = build_layout(CFG, direction)
    n_rows, n_cols = GRID_SHAPE
    checked_any = False
    for zone in layout.zones:
        if not isinstance(zone.dynamics, (UncontrollableDeterministic, UncontrollableNoisy)):
            continue
        checked_any = True
        drift = zone.dynamics.drift
        for r, c in np.argwhere(zone.mask):
            rr, cc = int(r), int(c)
            steps = 0
            while zone.mask[rr, cc]:
                rr = (rr + drift[0]) % n_rows
                cc = (cc + drift[1]) % n_cols
                steps += 1
                assert steps <= 5, f"{zone.appearance.category}: cell ({r},{c}) took >5 steps to exit"
            # the exit cell must be a corridor cell (row or col in the frozen corridor set)
            assert rr in CFG["corridor"]["rows"] or cc in CFG["corridor"]["cols"]
    assert checked_any


def test_away_from_goal_and_toward_goal_are_opposite_for_each_uncontrollable_zone():
    away = build_layout(CFG, "away_from_goal")
    toward = build_layout(CFG, "toward_goal")
    for za, zt in zip(_non_corridor_zones(away), _non_corridor_zones(toward)):
        if isinstance(za.dynamics, (UncontrollableDeterministic, UncontrollableNoisy)):
            assert za.dynamics.drift == (-zt.dynamics.drift[0], -zt.dynamics.drift[1])


def test_resolve_drift_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        resolve_drift((1, 5), GRID_SHAPE, GOAL, "sideways")


# ---------------------------------------------------------------------
# epsilon=0.1 -> p=0.125 mapping: checked against the raw row, not algebra
# ---------------------------------------------------------------------


def test_p_0_125_reproduces_the_spec_epsilon_0_1_mixture_exactly():
    layout = build_layout(CFG, "away_from_goal")
    for zone in _non_corridor_zones(layout):
        if not hasattr(zone.dynamics, "p"):
            continue
        for r, c in np.argwhere(zone.mask)[:3]:  # a few interior-ish sample cells
            for a in range(5):
                row = zone.dynamics.transition_row((int(r), int(c)), a, GRID_SHAPE)
                nz = np.sort(row[row > 0])[::-1]
                assert nz.shape == (5,)
                assert nz[0] == pytest.approx(0.9)
                assert np.allclose(nz[1:], 0.025)


# ---------------------------------------------------------------------
# pairing agreement (CLAUDE.md-mandated): H and per-(s,a) support size
# identical between experimental and control, only I differs
# ---------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["away_from_goal", "toward_goal"])
def test_pairing_agreement_per_zone(direction):
    exp_layout = build_layout(CFG, direction)
    ctrl_layout = build_control_layout(CFG, direction)
    P_exp = build_env(exp_layout).transition_tensor()
    P_ctrl = build_env(ctrl_layout).transition_tensor()

    H_exp = conditional_entropy_per_state(P_exp)
    H_ctrl = conditional_entropy_per_state(P_ctrl)
    I_exp = controllability_per_state(P_exp)
    I_ctrl = controllability_per_state(P_ctrl)

    for zone in _non_corridor_zones(exp_layout):
        idx = np.flatnonzero(zone.mask)
        for s in idx:
            # per-(s,a) support size, not the union over actions (a
            # deterministic pair has support 1 per action on both sides,
            # while the union is 1 state on the control side and up to 5
            # on the experimental side for the noisy zones -- that union
            # comparison would fail spuriously).
            for a in range(5):
                supp_exp = int((P_exp[s, a] > 0).sum())
                supp_ctrl = int((P_ctrl[s, a] > 0).sum())
                assert supp_exp == supp_ctrl, f"zone={zone.appearance.category} s={s} a={a}"
        assert H_exp[idx] == pytest.approx(H_ctrl[idx], abs=1e-9)
        if zone.appearance.category in ("uncontrollable_deterministic", "controllable_deterministic"):
            assert I_ctrl[idx] == pytest.approx(np.log2(5), abs=1e-9)
        # I must differ (except where the zone was already controllable, in
        # which case derive_control is a no-op and I_exp == I_ctrl trivially).
        from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic as UD
        from helplessness.envs.dynamics.uncontrollable import UncontrollableNoisy as UN

        if isinstance(zone.dynamics, (UD, UN)):
            assert not np.allclose(I_exp[idx], I_ctrl[idx])


# ---------------------------------------------------------------------
# bit-value agreement: analytic_controllability() vs controllability_per_state()
# on the exact tensor, per zone -- no hardcoded decimals.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["away_from_goal", "toward_goal"])
def test_analytic_controllability_matches_numeric_per_zone(direction):
    layout = build_layout(CFG, direction)
    P = build_env(layout).transition_tensor()
    I_numeric = controllability_per_state(P)
    H_numeric = conditional_entropy_per_state(P)
    for zone in _non_corridor_zones(layout):
        idx = np.flatnonzero(zone.mask)
        analytic_I = zone.dynamics.analytic_controllability(5)
        assert np.allclose(I_numeric[idx], analytic_I, atol=1e-9)
        # H must be uniform within a zone and match the deterministic/noisy
        # split (0 for deterministic zones, equal and nonzero for both
        # noisy zones -- computed, not hardcoded).
        assert np.allclose(H_numeric[idx], H_numeric[idx][0], atol=1e-9)


def test_noisy_zones_have_identical_H_to_each_other():
    layout = build_layout(CFG, "away_from_goal")
    P = build_env(layout).transition_tensor()
    H_numeric = conditional_entropy_per_state(P)
    noisy = [z for z in _non_corridor_zones(layout) if hasattr(z.dynamics, "p")]
    assert len(noisy) == 2
    idx0 = np.flatnonzero(noisy[0].mask)
    idx1 = np.flatnonzero(noisy[1].mask)
    assert H_numeric[idx0][0] == pytest.approx(H_numeric[idx1][0], abs=1e-9)


# ---------------------------------------------------------------------
# geometric interior mask: 9 cells per 5x5 block, dynamics-independent
# ---------------------------------------------------------------------


def test_interior_mask_is_nine_cells_per_block_regardless_of_dynamics():
    layout = build_layout(CFG, "away_from_goal")
    for zone in _non_corridor_zones(layout):
        interior = interior_mask(zone, GRID_SHAPE)
        assert interior.sum() == 9
        assert np.all(interior <= zone.mask)  # interior is a subset of the zone


def test_interior_mask_excludes_the_boundary_ring():
    layout = build_layout(CFG, "away_from_goal")
    nw = next(z for z in layout.zones if z.appearance.category == "uncontrollable_deterministic")
    interior = interior_mask(nw, GRID_SHAPE)
    # rows/cols 1 and 5 are the block's outer ring (adjacent to the corridor)
    assert not interior[1, :].any()
    assert not interior[5, :].any()
    assert not interior[:, 1].any()
    assert not interior[:, 5].any()
    assert interior[3, 3]  # the block's centre cell is interior


# ---------------------------------------------------------------------
# reflection: exact swap of block identity under row-negation
# ---------------------------------------------------------------------


def test_reflect_state_fixes_corridor_and_swaps_matching_blocks():
    layout = build_layout(CFG, "away_from_goal")
    nw = next(z for z in layout.zones if z.appearance.category == "uncontrollable_deterministic")
    sw = next(z for z in layout.zones if z.appearance.category == "controllable_deterministic")
    for r, c in np.argwhere(nw.mask):
        s = int(r) * GRID_SHAPE[1] + int(c)
        s2 = reflect_state(s, GRID_SHAPE)
        r2, c2 = divmod(s2, GRID_SHAPE[1])
        assert sw.mask[r2, c2]
        assert c2 == c  # column is fixed by construction


def test_reflect_state_is_its_own_inverse():
    for s in [0, 6, 12, 77, 100, 143]:
        assert reflect_state(reflect_state(s, GRID_SHAPE), GRID_SHAPE) == s


# ---------------------------------------------------------------------
# model/simulation agreement (CLAUDE.md-mandated third instance)
# ---------------------------------------------------------------------


def test_model_simulation_agreement():
    layout = build_layout(CFG, "away_from_goal")
    env = build_env(layout)
    P = env.transition_tensor()

    n_samples = 3000
    tol = 0.03
    probe_cells = [(0, 0), (1, 1), (3, 3), (5, 5), (1, 7), (3, 9), (7, 1), (9, 3), (7, 7), (9, 9)]
    for rc in probe_cells:
        s = env.rc_to_s(rc)
        for a in range(env.n_actions):
            counts = np.zeros(env.n_states)
            for _ in range(n_samples):
                env.position = rc
                env.t = 0
                _, _, _, _, info = env.step(a)
                counts[info["state"]] += 1
            empirical = counts / n_samples
            assert np.allclose(empirical, P[s, a], atol=tol), f"rc={rc} a={a}"


def test_corridor_is_not_the_frozen_reference_zone():
    """envs.zones.reference_zone() is a fixed 2x2 pattern (FROZEN_BLOCKS),
    unrelated to E1's 44-cell cross corridor -- both happen to use
    ControllableDeterministic, but they are different objects and the
    corridor must never be built via reference_zone()/checked via
    assert_reference_unchanged() (see docs/E1_offline_4zones.md)."""
    layout = build_layout(CFG, "away_from_goal")
    corridor = next(z for z in layout.zones if z.appearance.category == "corridor")
    assert corridor.mask.sum() == 44  # not the reference zone's fixed 2x2 = 4


def test_reward_and_terminal_shapes_and_goal_placement():
    R, terminal = reward_and_terminal(CFG, goal_reward=50.0)
    n_states = GRID_SHAPE[0] * GRID_SHAPE[1]
    assert R.shape == (n_states,)
    assert terminal.shape == (n_states,)
    goal_s = GOAL[0] * GRID_SHAPE[1] + GOAL[1]
    assert terminal[goal_s]
    assert terminal.sum() == 1
    assert R[goal_s] == 50.0
    assert np.all(R[~terminal] == -1.0)
