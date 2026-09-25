"""Allocentric k-pixel-per-cell renderer for E3 (spec section 2.4).

A NEW file, not a patch to observation/pixel_backend.py::PixelObservationBackend:
that backend is a 1-pixel-per-cell renderer with no k-pixel upscaling, no
grid lines, no CIELAB-constrained agent marker, and no background/
counterfactual split -- insufficient for E3's needs and left untouched.

render()/render_background()/render_counterfactual() are pure functions
(state index in, uint8 image out) -- AllocentricPixelBackend is the thin
ObservationBackend-conforming adapter TorusZonedEnv actually calls.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.appearance.theme_sampling import Themes
from helplessness.envs.observation.base import ObservationBackend
from helplessness.envs.zones import ZoneLayout

GRID_LINE_COLOR: tuple[int, int, int] = (0, 0, 0)


def _s_to_rc(s: int, n_cols: int) -> tuple[int, int]:
    return divmod(int(s), n_cols)


def _category_at(layout: ZoneLayout, r: int, c: int) -> str:
    return layout.zone_at((r, c)).appearance.category


def _cell_color(
    r: int,
    c: int,
    layout: ZoneLayout,
    themes: Themes,
    category_overrides: dict[str, str] | None,
) -> tuple[int, int, int]:
    n_cols = layout.grid_shape[1]
    s = r * n_cols + c
    if s == themes.goal_state:
        return themes.goal_color
    category = _category_at(layout, r, c)
    if category_overrides is not None and category in category_overrides:
        category = category_overrides[category]
    return themes.color_for_category(category)


def _base_image(
    layout: ZoneLayout,
    themes: Themes,
    k: int,
    grid_lines: bool,
    category_overrides: dict[str, str] | None = None,
) -> np.ndarray:
    n_rows, n_cols = layout.grid_shape
    img = np.zeros((n_rows * k, n_cols * k, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            color = _cell_color(r, c, layout, themes, category_overrides)
            img[r * k : (r + 1) * k, c * k : (c + 1) * k] = color
    if grid_lines:
        img[::k, :] = GRID_LINE_COLOR
        img[:, ::k] = GRID_LINE_COLOR
    return img


def _draw_agent_marker(img: np.ndarray, r: int, c: int, k: int, color: tuple[int, int, int]) -> None:
    """Filled disk of diameter ~= k/2, centred in cell (r, c)."""
    radius = k / 4.0
    cy, cx = r * k + k / 2.0, c * k + k / 2.0
    rows = np.arange(r * k, (r + 1) * k)
    cols = np.arange(c * k, (c + 1) * k)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    patch = img[r * k : (r + 1) * k, c * k : (c + 1) * k]
    patch[mask] = color


def render(s: int, layout: ZoneLayout, themes: Themes, k: int = 6, grid_lines: bool = False) -> np.ndarray:
    """uint8 [n_rows*k, n_cols*k, 3] -- the full allocentric view with the
    agent marker drawn at state s. Never used with a per-episode theme
    redraw (see AllocentricPixelBackend below)."""
    n_cols = layout.grid_shape[1]
    img = _base_image(layout, themes, k, grid_lines)
    r, c = _s_to_rc(s, n_cols)
    _draw_agent_marker(img, r, c, k, themes.agent_marker_color)
    return img


def render_background(layout: ZoneLayout, themes: Themes, k: int = 6, grid_lines: bool = False) -> np.ndarray:
    """Same as render(), without the agent marker -- used to compute the
    reconstruction-loss mask (spec 3.2) from the image alone, no state
    label involved."""
    return _base_image(layout, themes, k, grid_lines)


def render_counterfactual(
    s: int,
    layout: ZoneLayout,
    themes: Themes,
    k: int = 6,
    grid_lines: bool = False,
    *,
    cell_category_override: str | None = None,
    global_category_swap: tuple[str, str] | None = None,
) -> np.ndarray:
    """Diagnostics-only (spec section 8) -- NEVER used for training.

    cell_category_override: renders the single cell AT STATE s under a
    foreign category's colour (everything else normal) -- D1-local: "pour
    s dans la zone L, rendre l'image ou la case de l'agent prend le theme
    d'une autre zone L'."

    global_category_swap=(catA, catB): swaps the rendered colour used for
    EVERY cell of catA and catB across the whole grid -- D1-global:
    "echanger les themes des zones L et L' sur toute la carte."

    The two are independent and may be combined (no validation forbidding
    it -- this is a rendering primitive; which diagnostic actually needs
    which combination is decided in Phase 3, not here).
    """
    overrides = None
    if global_category_swap is not None:
        cat_a, cat_b = global_category_swap
        overrides = {cat_a: cat_b, cat_b: cat_a}
    img = _base_image(layout, themes, k, grid_lines, category_overrides=overrides)

    n_cols = layout.grid_shape[1]
    r, c = _s_to_rc(s, n_cols)
    if cell_category_override is not None:
        img[r * k : (r + 1) * k, c * k : (c + 1) * k] = themes.color_for_category(cell_category_override)
    _draw_agent_marker(img, r, c, k, themes.agent_marker_color)
    return img


class AllocentricPixelBackend(ObservationBackend):
    """ObservationBackend adapter around render(). Holds a fixed Themes
    object for the whole run -- theme_seed is never used to derive a
    theme here (spec 2.5: themes are fixed per run, never redrawn per
    episode). Rather than silently ignoring a caller-supplied
    theme_seed, observe() raises if it is anything other than None, so a
    TorusZonedEnv accidentally constructed/reset with a non-None
    theme_seed fails loudly instead of silently doing nothing with it.
    """

    def __init__(self, themes: Themes, k: int = 6, grid_lines: bool = False):
        self.themes = themes
        self.k = k
        self.grid_lines = grid_lines

    def observe(self, position: tuple[int, int], zone_layout: ZoneLayout, theme_seed) -> np.ndarray:
        if theme_seed is not None:
            raise ValueError(
                f"AllocentricPixelBackend.observe() got theme_seed={theme_seed!r}, expected "
                "None: themes are fixed for the whole run and are never redrawn per "
                "episode (spec section 2.5). Construct the owning TorusZonedEnv with "
                "theme_seed=None and never pass a theme_seed to reset()."
            )
        n_cols = zone_layout.grid_shape[1]
        s = position[0] * n_cols + position[1]
        return render(s, zone_layout, self.themes, k=self.k, grid_lines=self.grid_lines)
