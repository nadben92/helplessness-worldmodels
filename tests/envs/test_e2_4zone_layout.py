"""Tests for E2's quadrant-permutable layout (experiments/E2_online_4zones/env.py):
identity reproduces E1 exactly, drift is recomputed per target quadrant
(never copied from the source config), and the layout's V* is invariant
(as a per-zone-type multiset) under every counterbalancing rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module


def _import_e1_env():
    return load_experiment_module("E1_offline_4zones", "env.py", "_e1_env_for_e2_layout_test")


def _import_e2_env():
    return load_experiment_module("E2_online_4zones", "env.py", "_e2_env_for_e2_layout_test")


from helplessness.config import load_yaml_block  # noqa: E402
from helplessness.envs.zones import zone_spec_hash  # noqa: E402
from helplessness.planning.value_iteration import value_iteration  # noqa: E402

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"


def test_identity_quadrant_assignment_matches_rotation_zero():
    e2_env = _import_e2_env()
    assert e2_env.quadrant_assignment_for_rotation(0) == e2_env.identity_quadrant_assignment()


def test_e2_identity_layout_matches_e1_layout_bit_for_bit():
    e1_env = _import_e1_env()
    e2_env = _import_e2_env()

    e1_layout = e1_env.build_layout(CFG, DIRECTION)
    e2_layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=None)

    assert e1_layout.grid_shape == e2_layout.grid_shape
    e1_hashes = sorted(zone_spec_hash(z) for z in e1_layout.zones)
    e2_hashes = sorted(zone_spec_hash(z) for z in e2_layout.zones)
    assert e1_hashes == e2_hashes


def test_e2_identity_control_layout_matches_e1_control_layout_bit_for_bit():
    e1_env = _import_e1_env()
    e2_env = _import_e2_env()

    e1_ctrl = e1_env.build_control_layout(CFG, DIRECTION)
    e2_ctrl = e2_env.build_control_layout(CFG, DIRECTION, quadrant_assignment=None)

    e1_hashes = sorted(zone_spec_hash(z) for z in e1_ctrl.zones)
    e2_hashes = sorted(zone_spec_hash(z) for z in e2_ctrl.zones)
    assert e1_hashes == e2_hashes


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_rotation_is_a_valid_permutation_of_the_four_types(rotation):
    e2_env = _import_e2_env()
    assignment = e2_env.quadrant_assignment_for_rotation(rotation)
    assert set(assignment.keys()) == set(e2_env.QUADRANT_ORDER)
    assert sorted(assignment.values()) == sorted(e2_env.QUADRANT_ORDER)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_drift_points_away_from_goal_for_every_uncontrollable_zone_at_every_rotation(rotation):
    """The direction-sensitive invariant the counterbalancing review
    required: drift must be recomputed for the TARGET quadrant, never
    copied from the source config's own rows -- checked here by
    confirming the built dynamics' own drift vector always points away
    from the goal from wherever that block actually sits, not from
    wherever the type used to sit before permutation."""
    from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy
    from helplessness.envs.zones import resolve_drift

    e2_env = _import_e2_env()
    grid_shape = tuple(CFG["grid_shape"])
    goal_rc = tuple(CFG["goal"])
    assignment = e2_env.quadrant_assignment_for_rotation(rotation)
    layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)

    for target_quadrant in e2_env.QUADRANT_ORDER:
        target_cfg = CFG["blocks"][target_quadrant]
        source_key = assignment[target_quadrant]
        source_type = CFG["blocks"][source_key]["dynamics_type"]
        if source_type not in ("UncontrollableDeterministic", "UncontrollableNoisy"):
            continue

        expected_drift = resolve_drift(tuple(target_cfg["rows"]), grid_shape, goal_rc, DIRECTION)
        # find the zone actually covering this target quadrant's cells
        r0, c0 = target_cfg["rows"][0], target_cfg["cols"][0]
        zone = layout.zone_at((r0, c0))
        assert isinstance(zone.dynamics, (UncontrollableDeterministic, UncontrollableNoisy))
        assert zone.dynamics.drift == expected_drift, (
            f"rotation={rotation} target={target_quadrant} source={source_key}: "
            f"drift {zone.dynamics.drift} != expected {expected_drift} for this quadrant's own rows"
        )


@pytest.mark.parametrize("rotation", [1, 2, 3])
def test_drift_is_not_silently_copied_from_the_source_quadrant(rotation):
    """Direct negative check: whenever a permutation moves an
    uncontrollable type to a quadrant in a DIFFERENT row band (nw/ne share
    rows [1,5], se/sw share rows [7,11] -- only a north<->south move
    actually changes the row range drift depends on), its drift must
    differ from what it would have been at its ORIGINAL (source) quadrant
    -- catches the specific bug of forgetting to recompute drift. A move
    within the same row band (e.g. nw -> ne) is skipped: drift is row-axis
    only (envs.zones.resolve_drift), so identical row ranges legitimately
    produce identical drift regardless of source -- that is not evidence
    of copying, just shared geometry."""
    from helplessness.envs.zones import resolve_drift

    e2_env = _import_e2_env()
    grid_shape = tuple(CFG["grid_shape"])
    goal_rc = tuple(CFG["goal"])
    assignment = e2_env.quadrant_assignment_for_rotation(rotation)
    layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)

    checked_a_moved_uncontrollable_zone = False
    for target_quadrant in e2_env.QUADRANT_ORDER:
        source_key = assignment[target_quadrant]
        if source_key == target_quadrant:
            continue
        source_type = CFG["blocks"][source_key]["dynamics_type"]
        if source_type not in ("UncontrollableDeterministic", "UncontrollableNoisy"):
            continue

        target_cfg = CFG["blocks"][target_quadrant]
        source_cfg = CFG["blocks"][source_key]
        if tuple(target_cfg["rows"]) == tuple(source_cfg["rows"]):
            continue  # same row band -- drift is legitimately identical, not copied

        drift_at_source_rows = resolve_drift(tuple(source_cfg["rows"]), grid_shape, goal_rc, DIRECTION)

        r0, c0 = target_cfg["rows"][0], target_cfg["cols"][0]
        zone = layout.zone_at((r0, c0))
        assert zone.dynamics.drift != drift_at_source_rows, (
            f"rotation={rotation}: drift at target={target_quadrant} equals what it would have "
            f"been at its ORIGINAL quadrant {source_key} -- looks like drift was copied, not recomputed"
        )
        checked_a_moved_uncontrollable_zone = True
    if not checked_a_moved_uncontrollable_zone:
        # rotation=3 on this specific 4-quadrant cycle happens to keep
        # both uncontrollable types within their own row band (a genuine
        # property of this permutation, not a test gap) -- rotations 1
        # and 2 already exercise the cross-band case, so skip rather than
        # fail here.
        pytest.skip(f"rotation={rotation} moves no uncontrollable zone across row bands on this cycle")


@pytest.mark.parametrize("category", [
    "uncontrollable_deterministic", "controllable_noisy", "controllable_deterministic", "uncontrollable_noisy",
])
def test_v_star_multiset_per_zone_type_is_identical_across_rotations(category):
    """Torus-symmetry invariant: the SET of optimal values V*(s) for the
    states belonging to a given dynamics TYPE must be identical (as a
    multiset, order irrelevant) no matter which quadrant that type is
    placed in -- the whole point of the corridor's symmetric geometry."""
    e2_env = _import_e2_env()
    gamma = 0.9
    goal_reward = 50.0

    multisets = []
    for rotation in range(4):
        assignment = e2_env.quadrant_assignment_for_rotation(rotation)
        layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)
        env = e2_env.build_env(layout, CFG, goal_reward=goal_reward, max_steps=1)
        P = env.transition_tensor()
        R, terminal = e2_env.reward_and_terminal(CFG, goal_reward=goal_reward)
        V_star, _ = value_iteration(P, R, terminal, gamma=gamma)

        zone = next(z for z in layout.zones if z.appearance.category == category)
        values = np.sort(V_star[zone.mask.ravel()])
        multisets.append(values)

    for rotation in range(1, 4):
        assert multisets[rotation] == pytest.approx(multisets[0], abs=1e-6), (
            f"category={category}: V* multiset at rotation {rotation} differs from rotation 0"
        )
