"""Tests for Zone/ZoneLayout (arbitrary-shape masks, strict tiling
validation) and the reference zone (constancy check)."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.zones import (
    Zone,
    ZoneLayout,
    assert_reference_unchanged,
    geographic_support_mask,
    reference_zone,
    zone_spec_hash,
)

GRID_SHAPE = (4, 4)


def _quadrant_masks(grid_shape):
    n_rows, n_cols = grid_shape
    masks = []
    for r0 in (0, n_rows // 2):
        for c0 in (0, n_cols // 2):
            m = np.zeros(grid_shape, dtype=bool)
            m[r0 : r0 + n_rows // 2, c0 : c0 + n_cols // 2] = True
            masks.append(m)
    return masks


def test_zone_layout_accepts_a_valid_tiling():
    masks = _quadrant_masks(GRID_SHAPE)
    zones = [
        Zone(mask=masks[i], dynamics=ControllableDeterministic(), appearance=AppearanceSpec(category=f"z{i}"))
        for i in range(4)
    ]
    layout = ZoneLayout(GRID_SHAPE, zones)
    assert layout.grid_shape == GRID_SHAPE


def test_zone_layout_rejects_a_hole():
    masks = _quadrant_masks(GRID_SHAPE)
    masks[0][0, 0] = False  # punch a hole, covered by nobody
    zones = [
        Zone(mask=masks[i], dynamics=ControllableDeterministic(), appearance=AppearanceSpec(category=f"z{i}"))
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="do not tile"):
        ZoneLayout(GRID_SHAPE, zones)


def test_zone_layout_rejects_an_overlap():
    masks = _quadrant_masks(GRID_SHAPE)
    masks[1][0, 0] = True  # now covered by both zone 0 and zone 1
    zones = [
        Zone(mask=masks[i], dynamics=ControllableDeterministic(), appearance=AppearanceSpec(category=f"z{i}"))
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="do not tile"):
        ZoneLayout(GRID_SHAPE, zones)


def test_zone_layout_rejects_mismatched_mask_shape():
    zone = Zone(mask=np.ones((3, 3), dtype=bool), dynamics=ControllableDeterministic(), appearance=AppearanceSpec("a"))
    with pytest.raises(ValueError, match="grid_shape"):
        ZoneLayout(GRID_SHAPE, [zone])


def test_zone_layout_arbitrary_shape_not_just_quadrants():
    # an L-shaped zone plus its complement -- no quadrant assumption anywhere
    n_rows, n_cols = GRID_SHAPE
    l_shape = np.zeros(GRID_SHAPE, dtype=bool)
    l_shape[0, :] = True
    l_shape[:, 0] = True
    complement = ~l_shape
    zones = [
        Zone(mask=l_shape, dynamics=ControllableDeterministic(), appearance=AppearanceSpec("L")),
        Zone(mask=complement, dynamics=ControllableNoisy(p=0.5), appearance=AppearanceSpec("rest")),
    ]
    layout = ZoneLayout(GRID_SHAPE, zones)
    assert layout.zone_at((0, 2)).appearance.category == "L"
    assert layout.zone_at((2, 2)).appearance.category == "rest"


def test_transition_row_routes_through_the_right_zone():
    n_rows, n_cols = GRID_SHAPE
    top = np.zeros(GRID_SHAPE, dtype=bool)
    top[: n_rows // 2, :] = True
    bottom = ~top
    zones = [
        Zone(mask=top, dynamics=UncontrollableDeterministic(drift=(1, 0)), appearance=AppearanceSpec("top")),
        Zone(mask=bottom, dynamics=ControllableDeterministic(), appearance=AppearanceSpec("bottom")),
    ]
    layout = ZoneLayout(GRID_SHAPE, zones)
    # from the top zone, action is ignored (conveyor drift (1,0))
    row_top = layout.transition_row((0, 0), a=2)  # a=down, irrelevant
    assert row_top[1 * n_cols + 0] == pytest.approx(1.0)
    # from the bottom zone, the chosen action determines the outcome
    row_bottom = layout.transition_row((n_rows - 1, 0), a=1)  # a=right
    assert row_bottom[(n_rows - 1) * n_cols + 1] == pytest.approx(1.0)


def test_reference_zone_is_deterministic_given_the_same_anchor():
    z1 = reference_zone((8, 8), anchor=(2, 2))
    z2 = reference_zone((8, 8), anchor=(2, 2))
    assert zone_spec_hash(z1) == zone_spec_hash(z2)


def test_reference_zone_hash_depends_on_grid_shape_and_anchor():
    z1 = reference_zone((8, 8), anchor=(2, 2))
    z2 = reference_zone((8, 8), anchor=(3, 2))
    assert zone_spec_hash(z1) != zone_spec_hash(z2)


def test_assert_reference_unchanged_passes_for_the_real_thing():
    zone = reference_zone((10, 10), anchor=(1, 1))
    assert_reference_unchanged(zone, anchor=(1, 1))  # must not raise


def test_assert_reference_unchanged_catches_dynamics_drift():
    zone = reference_zone((10, 10), anchor=(1, 1))
    zone.dynamics = ControllableNoisy(p=0.1)  # someone "tweaked" it
    with pytest.raises(ValueError, match="diverged"):
        assert_reference_unchanged(zone, anchor=(1, 1))


def test_assert_reference_unchanged_catches_appearance_drift():
    zone = reference_zone((10, 10), anchor=(1, 1))
    zone.appearance = AppearanceSpec(category="not_reference")
    with pytest.raises(ValueError, match="diverged"):
        assert_reference_unchanged(zone, anchor=(1, 1))


# ---------------------------------------------------------------------
# geographic_support_mask (promoted from E1's estimation.py::geographic_alpha_support,
# see experiments/E1_offline_4zones/estimation.py -- E1 re-exports this
# function unchanged, tested there for byte-identical behaviour)
# ---------------------------------------------------------------------


def test_geographic_support_mask_has_exactly_five_true_per_row_on_a_torus():
    support = geographic_support_mask((6, 6))
    assert np.all(support.sum(axis=1) == 5)


def test_geographic_support_mask_includes_self_and_the_four_neighbours():
    grid_shape = (6, 6)
    support = geographic_support_mask(grid_shape)
    s = 2 * grid_shape[1] + 2  # state (2, 2)
    expected = {
        2 * grid_shape[1] + 2,  # self (stay)
        1 * grid_shape[1] + 2,  # up
        3 * grid_shape[1] + 2,  # down
        2 * grid_shape[1] + 1,  # left
        2 * grid_shape[1] + 3,  # right
    }
    assert set(np.flatnonzero(support[s])) == expected


def test_geographic_support_mask_wraps_on_the_torus():
    grid_shape = (5, 5)
    support = geographic_support_mask(grid_shape)
    s = 0  # state (0, 0), corner -- neighbours wrap
    expected = {
        0,  # self
        4 * grid_shape[1] + 0,  # up wraps to last row
        1 * grid_shape[1] + 0,  # down
        0 * grid_shape[1] + 4,  # left wraps to last col
        0 * grid_shape[1] + 1,  # right
    }
    assert set(np.flatnonzero(support[s])) == expected


def test_geographic_support_mask_matches_e1s_reexport():
    """experiments/E1_offline_4zones/estimation.py::geographic_alpha_support
    must remain a byte-identical re-export of this function -- E1 is frozen,
    see CLAUDE.md."""
    import sys
    from pathlib import Path

    e1_dir = str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones")
    sys.path.insert(0, e1_dir)
    try:
        from estimation import geographic_alpha_support
    finally:
        sys.path.remove(e1_dir)

    grid_shape = (12, 12)
    assert np.array_equal(geographic_alpha_support(grid_shape), geographic_support_mask(grid_shape))
