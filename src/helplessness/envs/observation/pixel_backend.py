"""Pixel observation: every cell coloured by its zone's appearance category
under the episode's theme, agent position overlaid as a marker colour.

One array per call, not cached: theme_seed varies per episode and the
agent's position varies per step, so there is nothing safe to memoise
across calls without also invalidating on both of those -- simpler and
correct to just rebuild the (small) grid each time.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.appearance.themes import render
from helplessness.envs.observation.base import ObservationBackend
from helplessness.envs.zones import ZoneLayout


class PixelObservationBackend(ObservationBackend):
    def __init__(self, agent_marker_color: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.agent_marker_color = agent_marker_color

    def observe(self, position: tuple[int, int], zone_layout: ZoneLayout, theme_seed: int) -> np.ndarray:
        n_rows, n_cols = zone_layout.grid_shape
        img = np.zeros((n_rows, n_cols, 3), dtype=np.float64)

        color_by_category: dict[str, tuple[float, float, float]] = {}
        for zone in zone_layout.zones:
            category = zone.appearance.category
            if category not in color_by_category:
                color_by_category[category] = render(category, theme_seed)
            img[zone.mask] = color_by_category[category]

        img[position] = self.agent_marker_color
        return img
