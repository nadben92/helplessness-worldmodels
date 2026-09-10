"""Tests for the theme system: determinism, relative-distance preservation
across themes, and the train/test partition."""

from __future__ import annotations

import colorsys

import numpy as np
import pytest

from helplessness.envs.appearance.themes import (
    TEST_THEME_SEED_START,
    is_test_theme,
    render,
    sample_test_theme,
    sample_train_theme,
)


def test_render_is_deterministic():
    assert render("A", 42) == render("A", 42)


def test_different_categories_render_differently_within_a_theme():
    assert render("A", 0) != render("B", 0)


def _hue(rgb):
    r, g, b = rgb
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return h


def _angular_distance(h1, h2):
    d = abs(h1 - h2) % 1.0
    return min(d, 1.0 - d)


def test_relative_distances_between_categories_preserved_across_themes():
    categories = ["A", "B", "C", "reference"]
    themes = [0, 1, 12345, 999_999]

    pairwise_distances_by_theme = []
    for theme_seed in themes:
        hues = [_hue(render(cat, theme_seed)) for cat in categories]
        distances = [
            _angular_distance(hues[i], hues[j])
            for i in range(len(categories))
            for j in range(i + 1, len(categories))
        ]
        pairwise_distances_by_theme.append(distances)

    reference = pairwise_distances_by_theme[0]
    for distances in pairwise_distances_by_theme[1:]:
        assert np.allclose(distances, reference, atol=1e-9)


def test_train_theme_pool_never_reaches_test_range():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        theme = sample_train_theme(rng)
        assert not is_test_theme(theme)
        assert theme < TEST_THEME_SEED_START


def test_test_theme_pool_always_in_test_range():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        theme = sample_test_theme(rng)
        assert is_test_theme(theme)
