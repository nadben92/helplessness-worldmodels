"""E4's training environments (helplessness.envs.layouts.two_zones): the pairing of the experimental and reference
environments, the symmetries checked with E3's own tools (tests/e3/test_e3_symmetry.py), the distance of the zones to
the goal, and the model/simulation agreement of the new layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.envs.dynamics.controllable import ControllableDeterministic
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.layouts import four_zones_pixel, two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.zones import Zone, ZoneLayout
from helplessness.measures.controllability import conditional_entropy_per_state, controllability_per_state

_spec = importlib.util.spec_from_file_location("e3_symmetry_tools", Path(__file__).parents[1] / "e3" / "test_e3_symmetry.py")
e3_symmetry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e3_symmetry)

LOG2_5 = np.log2(5)


def tensor(layout: ZoneLayout) -> np.ndarray:
    themes = two_zones.themes({q: CORRIDOR_COLOR for q in QUADRANT_ORDER}, (0, 0, 0))
    return four_zones_pixel.build_pixel_env(layout, themes, goal_reward=50.0).transition_tensor()


@pytest.mark.parametrize("diagonal", [1, 2])
def test_pairing_same_entropy_and_support_only_controllability_differs_in_ud(diagonal):
    exp, ref = (tensor(two_zones.build_layout(diagonal, c)) for c in ("experimental", "reference"))
    layout = two_zones.build_layout(diagonal, "experimental")
    # H(s'|s,a) and the support size of every pair are identical (every row is a point mass in both)
    for P in (exp, ref):
        assert np.allclose(P.max(-1), 1.0)
    assert np.array_equal((exp > 0).sum(-1), (ref > 0).sum(-1))
    assert np.allclose(conditional_entropy_per_state(exp), conditional_entropy_per_state(ref))
    I_exp, I_ref = controllability_per_state(exp), controllability_per_state(ref)
    differs = ~np.isclose(I_exp, I_ref)
    ud_states = two_zones.region_states(layout)[two_zones.roles(diagonal)["ud"]]
    assert set(np.flatnonzero(differs)) == set(ud_states)
    assert np.allclose(I_exp[ud_states], 0.0) and np.allclose(I_ref[ud_states], LOG2_5)


@pytest.mark.parametrize("diagonal", [1, 2])
def test_layout_zones_and_reference_is_derived(diagonal):
    roles = two_zones.roles(diagonal)
    exp, ref = two_zones.build_layout(diagonal, "experimental"), two_zones.build_layout(diagonal, "reference")
    kinds = {z.appearance.category: type(z.dynamics) for z in exp.zones}
    assert kinds[roles["ud"]] is UncontrollableDeterministic
    assert all(kinds[q] is ControllableDeterministic for q in QUADRANT_ORDER if q != roles["ud"])
    assert all(type(z.dynamics) is ControllableDeterministic for z in ref.zones)
    assert [z.appearance.category for z in exp.zones] == [z.appearance.category for z in ref.zones]
    assert {roles["ud"], roles["cd"]} == set(two_zones.DIAGONALS[diagonal])
    assert two_zones.NEUTRAL_OF[roles["ud"]] == roles["neutral"]


def test_both_slots_of_a_diagonal_are_equidistant_from_the_goal():
    regions = two_zones.region_states(two_zones.build_layout(1, "reference"))
    for ud, cd in two_zones.DIAGONALS.values():
        for stat in (min, max, lambda d: float(np.mean(d))):
            assert stat([two_zones.toric_distance_to_goal(s) for s in regions[ud]]) == stat([two_zones.toric_distance_to_goal(s) for s in regions[cd]])


def _ud_at(slot: str) -> ZoneLayout:
    """The experimental layout with UD at any slot (the placement convention's alternatives)."""
    base = two_zones.build_layout(1, "reference")
    zones = []
    for z in base.zones:
        if z.appearance.category == slot:
            from helplessness.envs.zones import resolve_drift
            drift = resolve_drift(tuple(two_zones.CFG["blocks"][slot]["rows"]), two_zones.GRID_SHAPE, two_zones.GOAL_RC, two_zones.DIRECTION)
            zones.append(Zone(mask=z.mask, dynamics=UncontrollableDeterministic(drift=drift), appearance=z.appearance))
        else:
            zones.append(z)
    return ZoneLayout(two_zones.GRID_SHAPE, zones)


def test_point_reflection_swaps_the_two_slots_of_a_diagonal_exactly():
    """Placing UD at the first slot of a diagonal loses no generality: the 180-degree turn about the goal maps the
    layout with UD at one slot onto the layout with UD at the other, exactly (E3's _transports_exactly)."""
    for ud, cd in two_zones.DIAGONALS.values():
        assert e3_symmetry._transports_exactly(tensor(_ud_at(ud)), tensor(_ud_at(cd)), e3_symmetry.REFLECTION_GROUP["point_reflect"])


def test_diagonals_match_e3_rotation_classes():
    """E3's class-A rotations (0, 2) put the conveyor (the type configured at nw) on diagonal 1 (nw, se), its class-B
    rotations (1, 3) on diagonal 2 (ne, sw)."""
    for rotation, cls in four_zones_pixel.ROTATION_CLASS.items():
        assignment = four_zones_pixel.quadrant_assignment_for_rotation(rotation)
        ud_slot = next(target for target, source in assignment.items() if source == "nw")
        assert ud_slot in two_zones.DIAGONALS[1 if cls == "A" else 2]


def test_in_e4_the_two_diagonals_are_mirror_images():
    """A finding the design must keep in mind: with two zones only, the column reflection about the goal maps diagonal 1
    (UD nw, CD se) onto diagonal 2 (UD ne, CD sw) exactly -- the diagonal factor changes the image, not the dynamics."""
    P1, P2 = tensor(two_zones.build_layout(1, "experimental")), tensor(two_zones.build_layout(2, "experimental"))
    assert e3_symmetry._find_matching_reflection(P1, P2) == "col_reflect"


@pytest.mark.parametrize("diagonal", [1, 2])
def test_model_simulation_agreement(diagonal):
    """step() samples exactly the tensor (CLAUDE.md: a new layout gets its own agreement check)."""
    themes = two_zones.themes(two_zones.training_colors(diagonal, (227, 17, 15), (22, 0, 244)), (100, 45, 0))
    env = four_zones_pixel.build_pixel_env(two_zones.build_layout(diagonal, "experimental"), themes, goal_reward=50.0, seed=0)
    P = env.transition_tensor()
    rng = np.random.default_rng(0)
    for s in rng.choice([s for s in range(two_zones.N_STATES) if s != two_zones.GOAL_STATE], size=25, replace=False):
        for a in range(5):
            env.reset(start_state=int(s))
            _, _, _, _, info = env.step(a)
            assert P[s, a, info["state"]] == 1.0
