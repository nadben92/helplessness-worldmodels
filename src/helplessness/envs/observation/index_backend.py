"""Tabular observation: the flat state index, same row-major convention
used everywhere else (rc_to_s: s = r * n_cols + c)."""

from __future__ import annotations

from helplessness.envs.observation.base import ObservationBackend
from helplessness.envs.zones import ZoneLayout


class IndexObservationBackend(ObservationBackend):
    def observe(self, position: tuple[int, int], zone_layout: ZoneLayout, theme_seed: int) -> int:
        r, c = position
        return r * zone_layout.grid_shape[1] + c
