"""E3's theme sampling: distinct RGB colours per zone appearance category,
resampled under a minimum pairwise CIELAB distance (spec section 2.5).

Deliberately a NEW module, not a patch to envs.appearance.themes: that
module's hash+rotation scheme (still used elsewhere) guarantees equal
ANGULAR spacing on a fixed hue wheel with no CIELAB guarantee and no
per-pair control. This module adds both, for E3 onward, via a rejection
sampler over CIELAB, and carries the `pair_distance` knob spec 2.5
reserves for E4/E5 (unused here -- raises if given, so a future caller
cannot silently rely on unimplemented behaviour).

corridor and goal colours are fixed constants, never sampled (spec 2.4:
"couloir gris neutre constant; but dans une couleur constante dédiée").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from helplessness.envs.appearance.cielab import lab_distance, srgb_to_lab

CORRIDOR_COLOR: tuple[int, int, int] = (128, 128, 128)  # neutral grey, constant
GOAL_COLOR: tuple[int, int, int] = (255, 215, 0)  # dedicated constant colour (gold)

_MAX_RESAMPLE_ATTEMPTS = 10_000


@dataclass(frozen=True)
class Themes:
    """Fixed-per-run theme assignment: category -> RGB (uint8-range int
    triples), plus the constant corridor/goal colours and the chosen
    agent-marker colour. Immutable -- a run's Themes object is built once
    and never redrawn per episode (spec 2.5's "fixité" requirement is
    enforced by construction: nothing in this module offers a way to
    mutate one, and AllocentricPixelBackend refuses a per-call theme_seed
    override, see envs/observation/allocentric_pixel_backend.py)."""

    zone_colors: dict[str, tuple[int, int, int]]
    agent_marker_color: tuple[int, int, int]
    goal_state: int
    corridor_color: tuple[int, int, int] = CORRIDOR_COLOR
    goal_color: tuple[int, int, int] = GOAL_COLOR

    def color_for_category(self, category: str) -> tuple[int, int, int]:
        if category == "corridor":
            return self.corridor_color
        return self.zone_colors[category]


def _random_rgb(rng: np.random.Generator) -> tuple[int, int, int]:
    return tuple(int(x) for x in rng.integers(0, 256, size=3))


def _lab(rgb: tuple[int, int, int]) -> np.ndarray:
    return srgb_to_lab(np.array(rgb, dtype=np.float64) / 255.0)


def sample_themes(
    categories: Iterable[str],
    rng: np.random.Generator,
    min_dist: float,
    metric: str = "cielab",
    pair_distance: float | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Rejection-samples one RGB colour per category such that every pair
    of categories is >= min_dist apart in CIELAB space (deltaE76).

    pair_distance: reserved for E4/E5 (a controlled-similarity pair of
    categories at a SPECIFIC distance rather than merely >= min_dist for
    everyone) -- not implemented, not used by E3; raises rather than
    silently ignoring the argument, per spec 2.5's own phrasing ("il
    n'est pas utilise en E3, mais servira en E4/E5").
    """
    if metric != "cielab":
        raise ValueError(f"unsupported metric {metric!r}; only 'cielab' is implemented")
    if pair_distance is not None:
        raise NotImplementedError(
            "pair_distance is reserved for E4/E5 and not implemented here -- "
            "E3 never passes it (spec section 2.5)"
        )

    categories = list(categories)
    for _ in range(_MAX_RESAMPLE_ATTEMPTS):
        colors = {c: _random_rgb(rng) for c in categories}
        labs = {c: _lab(rgb) for c, rgb in colors.items()}
        if _all_pairs_at_least(labs.values(), min_dist):
            return colors
    raise RuntimeError(
        f"could not find {len(categories)} colours pairwise >= min_dist={min_dist} "
        f"apart in CIELAB after {_MAX_RESAMPLE_ATTEMPTS} attempts -- min_dist is "
        "likely too large for 3-byte RGB space"
    )


def _all_pairs_at_least(labs: Iterable[np.ndarray], min_dist: float) -> bool:
    labs = list(labs)
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            if lab_distance(labs[i], labs[j]) < min_dist:
                return False
    return True


def choose_agent_marker_color(
    existing_colors: Iterable[tuple[int, int, int]],
    rng: np.random.Generator,
    min_dist: float,
) -> tuple[int, int, int]:
    """A colour >= min_dist (CIELAB) from every colour in
    existing_colors -- spec 2.4's requirement that the agent marker be
    "hors palette" of every zone theme, the corridor, and the goal."""
    existing_labs = [_lab(c) for c in existing_colors]
    for _ in range(_MAX_RESAMPLE_ATTEMPTS):
        candidate = _random_rgb(rng)
        cand_lab = _lab(candidate)
        if all(lab_distance(cand_lab, lab) >= min_dist for lab in existing_labs):
            return candidate
    raise RuntimeError(
        f"could not find an agent-marker colour >= min_dist={min_dist} from "
        f"{len(existing_labs)} existing colours after {_MAX_RESAMPLE_ATTEMPTS} attempts"
    )


def build_themes(
    zone_categories: Iterable[str],
    goal_state: int,
    rng: np.random.Generator,
    min_dist: float,
    marker_min_dist: float | None = None,
) -> Themes:
    """One call builds a run's whole fixed theme: zone colours (sampled,
    pairwise >= min_dist apart), plus an agent-marker colour that is also
    >= marker_min_dist (default: min_dist) from every zone colour AND
    from the constant corridor/goal colours -- spec 2.4's full distance
    requirement in one place."""
    zone_colors = sample_themes(zone_categories, rng, min_dist)
    marker_min_dist = min_dist if marker_min_dist is None else marker_min_dist
    existing = list(zone_colors.values()) + [CORRIDOR_COLOR, GOAL_COLOR]
    marker_color = choose_agent_marker_color(existing, rng, marker_min_dist)
    return Themes(zone_colors=zone_colors, agent_marker_color=marker_color, goal_state=int(goal_state))
