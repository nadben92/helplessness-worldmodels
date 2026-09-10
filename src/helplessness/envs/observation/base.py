"""ObservationBackend: how a position + zone layout become an observation.

The interchangeable half of "same dynamics, two backends" -- TorusZonedEnv
computes its dynamics entirely on the abstract (row, col) position; only
the observation returned to the caller depends on which backend is
plugged in. Swapping tabular <-> pixel is a constructor argument, never a
different environment class.
"""

from __future__ import annotations

import abc
from typing import Any

from helplessness.envs.zones import ZoneLayout


class ObservationBackend(abc.ABC):
    @abc.abstractmethod
    def observe(self, position: tuple[int, int], zone_layout: ZoneLayout, theme_seed: int) -> Any:
        """The observation for `position` on `zone_layout`, rendered under
        `theme_seed` -- theme_seed is passed on every call (not cached
        internally) because it is drawn per-episode, not fixed per env
        instance (see envs/appearance/themes.py)."""
