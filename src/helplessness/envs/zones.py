"""Zone: a mask on the grid, plus a dynamics spec and an appearance spec --
independent of each other, and of every other zone's. ZoneLayout validates
that a set of zones tiles the grid exactly once (see CLAUDE.md: refuses to
construct on a hole or an overlap, never truncates silently).

Also home to the reference zone: the one zone that must be byte-identical
(dynamics, appearance category, geometric pattern) across every
environment configuration that uses it -- reference_zone() constructs it
from the frozen definition, assert_reference_unchanged() catches a
divergence at load time rather than six weeks later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.dynamics.base import DynamicsSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic


@dataclass
class Zone:
    mask: np.ndarray  # (n_rows, n_cols) bool
    dynamics: DynamicsSpec
    appearance: AppearanceSpec


class ZoneLayout:
    """A set of Zone covering a grid_shape exactly once. Refuses to
    construct otherwise -- see module docstring."""

    def __init__(self, grid_shape: tuple[int, int], zones: list[Zone]):
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.zones = list(zones)
        self._validate()

    def _validate(self) -> None:
        for zone in self.zones:
            if zone.mask.shape != self.grid_shape:
                raise ValueError(
                    f"zone (appearance={zone.appearance.category!r}) has mask shape "
                    f"{zone.mask.shape} != grid_shape {self.grid_shape}"
                )
        coverage = np.zeros(self.grid_shape, dtype=int)
        for zone in self.zones:
            coverage += zone.mask.astype(int)
        if not np.all(coverage == 1):
            bad = [tuple(int(x) for x in rc) for rc in np.argwhere(coverage != 1)]
            raise ValueError(
                f"zones do not tile the grid exactly once at cells {bad} "
                "(0 = hole, >=2 = overlap) -- refusing to silently truncate or merge"
            )

    def zone_at(self, rc: tuple[int, int]) -> Zone:
        for zone in self.zones:
            if zone.mask[rc]:
                return zone
        raise ValueError(f"no zone covers {rc} -- unreachable after _validate()")

    def transition_row(self, rc: tuple[int, int], a: int) -> np.ndarray:
        return self.zone_at(rc).dynamics.transition_row(rc, a, self.grid_shape)


def zone_spec_hash(zone: Zone) -> str:
    """Stable hash of (dynamics class + params, appearance category, mask)
    -- order-independent of dict iteration, insensitive to numpy dtype
    quirks (everything funnels through plain Python types before hashing).
    """
    dynamics_params = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in vars(zone.dynamics).items()
    }
    payload = {
        "dynamics_class": type(zone.dynamics).__name__,
        "dynamics_params": dynamics_params,
        "appearance_category": zone.appearance.category,
        "mask": zone.mask.astype(int).tolist(),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------
# the reference zone
# ---------------------------------------------------------------------
_REFERENCE_DYNAMICS_FACTORY = ControllableDeterministic
_REFERENCE_APPEARANCE_CATEGORY = "reference"
_REFERENCE_PATTERN_SHAPE = (2, 2)  # fixed geometry; only the anchor may vary per config


def reference_zone(grid_shape: tuple[int, int], anchor: tuple[int, int]) -> Zone:
    """The reference zone, anchored at `anchor` (top-left corner, wrapped
    on the torus) on a grid of `grid_shape`. Dynamics, appearance category,
    and the 2x2 pattern shape never change; only the anchor position is a
    per-config parameter, since grid size itself varies across experiments
    (see CLAUDE.md) and an absolute position can't be frozen along with it.
    """
    mask = np.zeros(grid_shape, dtype=bool)
    r0, c0 = anchor
    for dr in range(_REFERENCE_PATTERN_SHAPE[0]):
        for dc in range(_REFERENCE_PATTERN_SHAPE[1]):
            mask[(r0 + dr) % grid_shape[0], (c0 + dc) % grid_shape[1]] = True
    return Zone(
        mask=mask,
        dynamics=_REFERENCE_DYNAMICS_FACTORY(),
        appearance=AppearanceSpec(category=_REFERENCE_APPEARANCE_CATEGORY),
    )


def assert_reference_unchanged(zone: Zone, anchor: tuple[int, int]) -> None:
    """Raises if `zone` does not match the frozen reference definition at
    `anchor` -- call this on any zone a config claims is "the reference
    zone" before using it in an experiment, so a silent divergence
    (someone tweaked its dynamics for one config and forgot the others) is
    caught at load time, not discovered by a confused result six weeks
    later."""
    expected = reference_zone(zone.mask.shape, anchor)
    if zone_spec_hash(zone) != zone_spec_hash(expected):
        raise ValueError(
            "zone does not match the frozen reference definition (envs.zones.reference_zone) "
            "-- its dynamics, appearance category, or mask pattern has diverged"
        )


# ---------------------------------------------------------------------
# rectangular-block layouts on a torus (E1's 4-zone geometry, and any
# future zone/dynamics/appearance family that reuses the same shape --
# not specific to 4 zones or to any particular grid size)
# ---------------------------------------------------------------------


def torus_block_mask(
    grid_shape: tuple[int, int], row_range: tuple[int, int], col_range: tuple[int, int]
) -> np.ndarray:
    """Bool mask, True on the closed rectangle
    rows in [row_range[0], row_range[1]] x cols in [col_range[0], col_range[1]]
    (inclusive both ends, matching how the ranges read in a spec: "rows
    1-5"). No torus wrapping is applied to the range itself -- callers pass
    ranges that already lie within grid_shape."""
    mask = np.zeros(grid_shape, dtype=bool)
    r0, r1 = row_range
    c0, c1 = col_range
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def cross_corridor_mask(
    grid_shape: tuple[int, int], corridor_rows: tuple[int, ...], corridor_cols: tuple[int, ...]
) -> np.ndarray:
    """Bool mask, True where row in corridor_rows or col in corridor_cols --
    the neutral cross corridor separating a torus into rectangular blocks."""
    mask = np.zeros(grid_shape, dtype=bool)
    for r in corridor_rows:
        mask[r, :] = True
    for c in corridor_cols:
        mask[:, c] = True
    return mask


def _toric_axis_distance(x: int, y: int, n: int) -> int:
    d = abs(int(x) - int(y))
    return min(d, n - d)


def resolve_drift(
    row_range: tuple[int, int], grid_shape: tuple[int, int], goal_rc: tuple[int, int], mode: str
) -> tuple[int, int]:
    """Resolves `mode` ("away_from_goal" | "toward_goal") to a concrete
    row-axis drift vector for a block spanning `row_range`, using TORIC
    row-distance to the goal's row (min(r, n_rows - r)), not a naive index
    comparison -- for a block on the far side of the torus from the goal,
    increasing the row index can decrease the toric distance (see the E1
    plan: this matters for the SE block, rows 7-11, where "down" is
    toward_goal even though row indices are increasing away from row 0).

    Drift is always along the row axis only (a fixed, documented
    convention -- the two axes are related by a 90-degree torus rotation
    and neither is closer by construction, so the choice is arbitrary but
    must be single and consistent). Column component of the returned drift
    is always 0.
    """
    if mode not in ("away_from_goal", "toward_goal"):
        raise ValueError(f"mode must be 'away_from_goal' or 'toward_goal', got {mode!r}")
    n_rows = grid_shape[0]
    goal_row = goal_rc[0]
    r0, r1 = row_range
    # Probe with a one-step move in each row direction from the block's
    # near edge (r0) and see which one decreases toric distance to the
    # goal row -- that is "toward_goal" by definition, regardless of which
    # side of the torus the block sits on.
    dist_here = _toric_axis_distance(r0, goal_row, n_rows)
    dist_up = _toric_axis_distance((r0 - 1) % n_rows, goal_row, n_rows)
    dist_down = _toric_axis_distance((r0 + 1) % n_rows, goal_row, n_rows)
    if dist_up < dist_here and dist_down >= dist_here:
        toward = (-1, 0)
    elif dist_down < dist_here and dist_up >= dist_here:
        toward = (1, 0)
    else:
        raise ValueError(
            f"could not resolve a unique toward-goal row direction for row_range={row_range}, "
            f"goal_row={goal_row}, n_rows={n_rows} (dist_here={dist_here}, dist_up={dist_up}, "
            f"dist_down={dist_down}) -- block does not have a well-defined near/far side"
        )
    if mode == "toward_goal":
        return toward
    return (-toward[0], -toward[1])


def assert_zone_symmetry(
    blocks: list[Zone], grid_shape: tuple[int, int], goal_rc: tuple[int, int]
) -> None:
    """Raises unless every block's cells have the exact same (sorted)
    distribution of toric-Manhattan distance to the goal -- purely
    geometric (masks only), holds regardless of which dynamics type sits
    in which block. If this fails, the layout itself is wrong (masks are
    not congruent under the torus symmetry the corridor is supposed to
    guarantee), not a numerical-tolerance issue -- exact equality, not
    means (per the E1 spec)."""
    n_rows, n_cols = grid_shape
    goal_r, goal_c = goal_rc
    reference_dist = None
    reference_idx = None
    for idx, block in enumerate(blocks):
        cells = np.argwhere(block.mask)
        dists = np.sort(
            [
                _toric_axis_distance(r, goal_r, n_rows) + _toric_axis_distance(c, goal_c, n_cols)
                for r, c in cells
            ]
        )
        if reference_dist is None:
            reference_dist = dists
            reference_idx = idx
        elif not np.array_equal(dists, reference_dist):
            raise ValueError(
                f"block {idx} (appearance={block.appearance.category!r}) has a distance-to-goal "
                f"distribution that differs from block {reference_idx} "
                f"(appearance={blocks[reference_idx].appearance.category!r}) -- the layout is not "
                "torus-symmetric as required, this is a bug in the layout, not a tolerance issue"
            )


def reflect_rc(rc: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    """Row-negation reflection r -> (-r) mod n_rows, columns fixed. On a
    layout whose corridor rows include 0 and n_rows/2, this reflection
    fixes both corridor rows and swaps blocks on opposite sides of the
    row-corridor -- used to assert an exact value-function symmetry (see
    experiments/E1_offline_4zones/calibration.py), stronger than comparing
    means across blocks."""
    r, c = rc
    n_rows = grid_shape[0]
    return ((-int(r)) % n_rows, int(c))


def reflect_state(s: int, grid_shape: tuple[int, int]) -> int:
    n_rows, n_cols = grid_shape
    r, c = divmod(int(s), n_cols)
    r2, c2 = reflect_rc((r, c), grid_shape)
    return r2 * n_cols + c2


def interior_mask(zone: Zone, grid_shape: tuple[int, int]) -> np.ndarray:
    """Bool mask over the full grid (True only within `zone.mask`): a cell
    is interior iff all 4 cardinal grid-neighbours (up/down/left/right,
    torus-wrapped -- not "stay") also lie in the zone's mask.

    Deliberately purely geometric, not derived from the zone's own
    dynamics/transition_row(): a successor-support definition would make
    the interior's size depend on dynamics (e.g. a single-cell-per-step
    conveyor's interior would balloon to "two steps from the edge", hiding
    that those cells' value is still set mostly by what's outside the
    zone), which breaks equal precision-per-zone across zone types. The
    geometric definition gives the same interior size for every block of
    the same shape, regardless of its dynamics.
    """
    n_rows, n_cols = grid_shape
    mask = zone.mask
    interior = np.zeros(grid_shape, dtype=bool)
    for r, c in np.argwhere(mask):
        neighbours = [
            ((r - 1) % n_rows, c),
            ((r + 1) % n_rows, c),
            (r, (c - 1) % n_cols),
            (r, (c + 1) % n_cols),
        ]
        if all(mask[nr, nc] for nr, nc in neighbours):
            interior[r, c] = True
    return interior
