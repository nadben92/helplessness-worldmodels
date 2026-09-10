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
