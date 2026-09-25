from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.appearance.cielab import lab_distance, srgb_to_lab
from helplessness.envs.appearance.theme_sampling import (
    CORRIDOR_COLOR,
    GOAL_COLOR,
    Themes,
    build_themes,
    choose_agent_marker_color,
    sample_themes,
)

CATEGORIES = (
    "uncontrollable_deterministic",
    "controllable_noisy",
    "controllable_deterministic",
    "uncontrollable_noisy",
)


def _lab(rgb):
    return srgb_to_lab(np.array(rgb, dtype=np.float64) / 255.0)


def test_sample_themes_returns_one_colour_per_category():
    rng = np.random.default_rng(0)
    colors = sample_themes(CATEGORIES, rng, min_dist=10.0)
    assert set(colors) == set(CATEGORIES)
    for rgb in colors.values():
        assert len(rgb) == 3
        assert all(0 <= v <= 255 for v in rgb)


def test_sample_themes_respects_min_dist():
    rng = np.random.default_rng(1)
    min_dist = 25.0
    colors = sample_themes(CATEGORIES, rng, min_dist=min_dist)
    labs = [_lab(c) for c in colors.values()]
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            assert lab_distance(labs[i], labs[j]) >= min_dist


def test_sample_themes_rejects_unsupported_metric():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_themes(CATEGORIES, rng, min_dist=10.0, metric="euclidean_rgb")


def test_sample_themes_rejects_pair_distance_not_implemented():
    rng = np.random.default_rng(0)
    with pytest.raises(NotImplementedError):
        sample_themes(CATEGORIES, rng, min_dist=10.0, pair_distance=5.0)


def test_sample_themes_raises_when_min_dist_is_infeasible():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        sample_themes(CATEGORIES, rng, min_dist=1000.0)  # unreachable in Lab space


def test_choose_agent_marker_color_respects_min_dist_from_existing():
    rng = np.random.default_rng(2)
    existing = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    min_dist = 20.0
    marker = choose_agent_marker_color(existing, rng, min_dist)
    marker_lab = _lab(marker)
    for c in existing:
        assert lab_distance(marker_lab, _lab(c)) >= min_dist


def test_build_themes_marker_is_far_from_corridor_and_goal_too():
    rng = np.random.default_rng(3)
    themes = build_themes(CATEGORIES, goal_state=0, rng=rng, min_dist=20.0)
    marker_lab = _lab(themes.agent_marker_color)
    assert lab_distance(marker_lab, _lab(CORRIDOR_COLOR)) >= 20.0
    assert lab_distance(marker_lab, _lab(GOAL_COLOR)) >= 20.0
    for rgb in themes.zone_colors.values():
        assert lab_distance(marker_lab, _lab(rgb)) >= 20.0


def test_build_themes_is_deterministic_given_the_same_rng_state():
    themes_a = build_themes(CATEGORIES, goal_state=5, rng=np.random.default_rng(42), min_dist=15.0)
    themes_b = build_themes(CATEGORIES, goal_state=5, rng=np.random.default_rng(42), min_dist=15.0)
    assert themes_a == themes_b


def test_themes_color_for_category_resolves_corridor_and_zone_categories():
    themes = Themes(
        zone_colors={"controllable_deterministic": (10, 20, 30)},
        agent_marker_color=(1, 2, 3),
        goal_state=0,
    )
    assert themes.color_for_category("corridor") == CORRIDOR_COLOR
    assert themes.color_for_category("controllable_deterministic") == (10, 20, 30)


def test_themes_is_frozen():
    themes = Themes(zone_colors={}, agent_marker_color=(0, 0, 0), goal_state=0)
    with pytest.raises(Exception):
        themes.goal_state = 1  # frozen dataclass -- must refuse mutation
