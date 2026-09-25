"""Tests for envs.observation.allocentric_pixel_backend. Uses a small
synthetic 4x4 four-quadrant ZoneLayout (not E1/E2's real geometry -- that
reuse is tested in tests/e3/) so these tests stay about rendering only.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.appearance.theme_sampling import Themes, build_themes
from helplessness.envs.dynamics.controllable import ControllableDeterministic
from helplessness.envs.observation.allocentric_pixel_backend import (
    AllocentricPixelBackend,
    render,
    render_background,
    render_counterfactual,
)
from helplessness.envs.zones import Zone, ZoneLayout, torus_block_mask

GRID_SHAPE = (4, 4)
CATEGORIES = ("north_west", "north_east", "south_west", "south_east")


def _layout() -> ZoneLayout:
    zones = [
        Zone(mask=torus_block_mask(GRID_SHAPE, (0, 1), (0, 1)), dynamics=ControllableDeterministic(),
             appearance=_appearance("north_west")),
        Zone(mask=torus_block_mask(GRID_SHAPE, (0, 1), (2, 3)), dynamics=ControllableDeterministic(),
             appearance=_appearance("north_east")),
        Zone(mask=torus_block_mask(GRID_SHAPE, (2, 3), (0, 1)), dynamics=ControllableDeterministic(),
             appearance=_appearance("south_west")),
        Zone(mask=torus_block_mask(GRID_SHAPE, (2, 3), (2, 3)), dynamics=ControllableDeterministic(),
             appearance=_appearance("south_east")),
    ]
    return ZoneLayout(GRID_SHAPE, zones)


def _appearance(category: str):
    from helplessness.envs.appearance.base import AppearanceSpec

    return AppearanceSpec(category=category)


def _themes(goal_state: int = 0) -> Themes:
    rng = np.random.default_rng(0)
    return build_themes(CATEGORIES, goal_state=goal_state, rng=rng, min_dist=15.0)


def test_render_shape_and_dtype():
    layout = _layout()
    themes = _themes()
    img = render(5, layout, themes, k=6)
    assert img.shape == (4 * 6, 4 * 6, 3)
    assert img.dtype == np.uint8


@pytest.mark.parametrize("k", [2, 6, 10])
def test_render_respects_k_pixels_per_cell(k):
    layout = _layout()
    themes = _themes()
    img = render(0, layout, themes, k=k)
    assert img.shape == (4 * k, 4 * k, 3)


def test_render_is_injective_over_all_states():
    """T1: no two of the grid's states render to the same image."""
    layout = _layout()
    themes = _themes()
    n_states = GRID_SHAPE[0] * GRID_SHAPE[1]
    images = [render(s, layout, themes, k=6) for s in range(n_states)]
    for i in range(n_states):
        for j in range(i + 1, n_states):
            assert not np.array_equal(images[i], images[j]), f"states {i} and {j} rendered identically"


def test_render_background_has_no_agent_marker():
    layout = _layout()
    themes = _themes()
    bg = render_background(layout, themes, k=6)
    full = render(5, layout, themes, k=6)
    assert not np.array_equal(bg, full)
    # every non-agent-cell pixel must match between the two
    assert np.array_equal(bg[:6, :6], full[:6, :6])  # state 0's cell untouched by an agent at state 5


def test_render_background_is_the_same_regardless_of_which_state_would_have_been_rendered():
    layout = _layout()
    themes = _themes()
    bg1 = render_background(layout, themes, k=6)
    bg2 = render_background(layout, themes, k=6)
    assert np.array_equal(bg1, bg2)


def test_goal_cell_gets_the_dedicated_goal_color_not_its_zone_color():
    layout = _layout()
    themes = _themes(goal_state=0)
    bg = render_background(layout, themes, k=6)
    goal_pixel = tuple(int(v) for v in bg[0, 0])
    assert goal_pixel == themes.goal_color
    assert goal_pixel != themes.zone_colors["north_west"]


def test_render_counterfactual_matches_render_with_no_overrides():
    layout = _layout()
    themes = _themes()
    assert np.array_equal(render(5, layout, themes, k=6), render_counterfactual(5, layout, themes, k=6))


def test_render_counterfactual_cell_override_changes_only_the_agents_own_cell():
    layout = _layout()
    themes = _themes()
    baseline = render(5, layout, themes, k=6)  # state 5 = row1,col1, in north_west
    cf = render_counterfactual(5, layout, themes, k=6, cell_category_override="south_east")
    # the agent's own cell background differs
    assert not np.array_equal(baseline[6:12, 6:12], cf[6:12, 6:12])
    # a DIFFERENT cell of the same zone (state 4 = row1,col0) is untouched
    assert np.array_equal(baseline[6:12, 0:6], cf[6:12, 0:6])


def test_render_counterfactual_global_swap_changes_every_cell_of_both_categories():
    layout = _layout()
    themes = _themes()
    baseline_bg = render_background(layout, themes, k=6)
    cf_bg = render_background(layout, themes, k=6)  # placeholder, unused directly
    from helplessness.envs.observation.allocentric_pixel_backend import _base_image

    swapped = _base_image(layout, themes, k=6, grid_lines=False,
                           category_overrides={"north_west": "south_east", "south_east": "north_west"})
    # cell (1,1) (state 5, north_west, NOT the goal cell at (0,0)) now
    # shows south_east's colour
    assert np.array_equal(swapped[6:12, 6:12][0, 0], np.array(themes.zone_colors["south_east"], dtype=np.uint8))
    # north_east region (untouched category) is unchanged
    assert np.array_equal(swapped[0:6, 12:18], baseline_bg[0:6, 12:18])


def test_render_counterfactual_never_mutates_the_themes_object():
    layout = _layout()
    themes = _themes()
    before = (dict(themes.zone_colors), themes.agent_marker_color, themes.goal_state)
    render_counterfactual(5, layout, themes, k=6, cell_category_override="south_east",
                           global_category_swap=("north_west", "south_east"))
    after = (dict(themes.zone_colors), themes.agent_marker_color, themes.goal_state)
    assert before == after


class TestAllocentricPixelBackend:
    def test_observe_with_theme_seed_none_matches_render(self):
        layout = _layout()
        themes = _themes()
        backend = AllocentricPixelBackend(themes=themes, k=6)
        obs = backend.observe((1, 1), layout, None)
        assert np.array_equal(obs, render(5, layout, themes, k=6))

    @pytest.mark.parametrize("bad_theme_seed", [0, 1, 42, -1])
    def test_observe_raises_on_any_non_none_theme_seed(self, bad_theme_seed):
        layout = _layout()
        themes = _themes()
        backend = AllocentricPixelBackend(themes=themes, k=6)
        with pytest.raises(ValueError, match="section 2.5"):
            backend.observe((0, 0), layout, bad_theme_seed)
