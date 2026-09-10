"""Theme space: a deterministic (category, theme_seed) -> colour function.

Design: every appearance category gets a fixed base angle on a colour
wheel, derived once from a hash of its name (stable across the whole
project, not re-randomised per run). A theme is a single rotation applied
to every category's angle at once -- a rigid transform of the circle, so
the pairwise angular distance between any two categories is EXACTLY the
same in every theme. That is the "preserve relative distances between zone
types across themes" requirement, satisfied structurally rather than
approximately: rotating a circle can never change the distances between
points on it.

Test themes are a reserved, disjoint range of the seed space
(>= TEST_THEME_SEED_START), never reachable by sample_train_theme -- this
is a partition of the integers, not a convention someone could
accidentally violate by picking an unlucky training seed.
"""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np

TEST_THEME_SEED_START = 900_000_000


def _category_base_angle(category: str) -> float:
    """Stable per-category angle in [0, 2*pi), derived from a hash of the
    category name -- fixed forever for a given category string, the same
    across every process and every run."""
    digest = hashlib.sha256(category.encode("utf-8")).hexdigest()
    return (int(digest, 16) % 360) / 360.0 * 2 * np.pi


def _theme_rotation(theme_seed: int) -> float:
    rng = np.random.default_rng(theme_seed)
    return float(rng.uniform(0.0, 2 * np.pi))


def render(category: str, theme_seed: int) -> tuple[float, float, float]:
    """Deterministic RGB triple in [0, 1]^3 for this (category, theme).
    Saturation/value are fixed (readable, not washed out); only hue varies
    by category and by theme."""
    angle = (_category_base_angle(category) + _theme_rotation(theme_seed)) % (2 * np.pi)
    hue = angle / (2 * np.pi)
    return colorsys.hsv_to_rgb(hue, 0.65, 0.85)


def is_test_theme(theme_seed: int) -> bool:
    return theme_seed >= TEST_THEME_SEED_START


def sample_train_theme(rng: np.random.Generator) -> int:
    """Never returns a value >= TEST_THEME_SEED_START -- the training pool
    and the test pool are disjoint ranges, not an honour system."""
    return int(rng.integers(0, TEST_THEME_SEED_START))


def sample_test_theme(rng: np.random.Generator) -> int:
    return int(rng.integers(TEST_THEME_SEED_START, 2**31 - 1))
