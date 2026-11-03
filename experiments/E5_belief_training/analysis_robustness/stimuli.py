"""Step 4's images (README.md, PREDICTIONS.md): the geometry, the goal and the agent's states of the probe world unchanged,
only colours changed. Colours are chosen in CIELAB (deltaE76, envs.appearance.theme_axes.lab_distances, the metric of the
training's theme sampling), away from every colour the models saw: T_inc, T_ctrl, the corridor's grey, the goal's gold
and the agent's brown (FORBIDDEN).

  far_colors       farthest-point sampling over an RGB grid (step 5): the colour farthest from FORBIDDEN, then the one
                   farthest from FORBIDDEN and the first, then from all three. The first is the colour farthest from
                   every training colour and from the agent.
  4a uniform       the four quadrants and the corridor painted one far colour (3 images).
  4b novel         each quadrant and the corridor its own new colour: theme_sampling.sample_themes (pairwise
                   deltaE >= MIN_DE, the training's cielab_min_dist), redrawn until every colour is also >= MIN_DE from
                   FORBIDDEN; 5 draws from SeedSequence([STIMULI_SEED, draw]).
  4c agent         the canonical training image with the agent drawn in each far colour (3 images); far from every
                   background colour by construction.

images_for(): the state images of a probe world under given themes, as pipeline.two_zones.build_world renders them
(the reference layout, the run's render config); probe_images() checks it against two_zones_setup.probe_world.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from helplessness.envs.appearance.theme_axes import lab_distances
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR, GOAL_COLOR, sample_themes
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.observation.allocentric_pixel_backend import render

MIN_DE = 30.0           # render.cielab_min_dist of the runs' config
STIMULI_SEED = 4004     # 4b's draws
N_FAR = 3
N_NOVEL = 5
IMAGE_BASE = 4000       # step 4's image indices (measurement seeds), clear of E4's (0-92), the swap set's (2000+)


def forbidden(conf: dict) -> np.ndarray:
    return np.array([conf["t_inc"], conf["t_ctrl"], CORRIDOR_COLOR, GOAL_COLOR, conf["agent_color"]], dtype=np.int64)


def far_colors(conf: dict, n: int = N_FAR, step: int = 5) -> list[tuple[int, int, int]]:
    grid = np.stack(np.meshgrid(*(np.arange(0, 256, step),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    chosen: list[tuple[int, int, int]] = []
    for _ in range(n):
        ref = np.concatenate([forbidden(conf), np.array(chosen, dtype=np.int64).reshape(-1, 3)])
        d = lab_distances(grid, ref).min(axis=1)
        chosen.append(tuple(int(v) for v in grid[int(d.argmax())]))
    return chosen


def novel_backgrounds(conf: dict, n: int = N_NOVEL) -> list[dict]:
    """n draws: {"nw", "ne", "se", "sw", "corridor"} -> RGB."""
    out = []
    for draw in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([STIMULI_SEED, draw]))
        while True:
            colors = sample_themes([*QUADRANT_ORDER, "corridor"], rng, MIN_DE)
            if lab_distances(np.array(list(colors.values())), forbidden(conf)).min() >= MIN_DE:
                out.append(colors)
                break
    return out


def step4_images(conf: dict) -> list[dict]:
    """Step 4's images: {"index", "family" (4a/4b/4c), "name", "slot_colors", "corridor_color", "agent_color", "min_de"}
    (min_de: the smallest deltaE between a colour of the image and FORBIDDEN, the agent's colour left out for 4c)."""
    canon = conf["training_colors"]
    images = []
    for c in far_colors(conf):
        images.append({"family": "4a", "name": f"uniform {c}", "slot_colors": {q: c for q in QUADRANT_ORDER}, "corridor_color": c,
                       "agent_color": conf["agent_color"]})
    for i, colors in enumerate(novel_backgrounds(conf)):
        images.append({"family": "4b", "name": f"novel draw {i}", "slot_colors": {q: colors[q] for q in QUADRANT_ORDER},
                       "corridor_color": colors["corridor"], "agent_color": conf["agent_color"]})
    for c in far_colors(conf):
        images.append({"family": "4c", "name": f"agent {c}", "slot_colors": dict(canon), "corridor_color": tuple(CORRIDOR_COLOR),
                       "agent_color": c})
    for i, im in enumerate(images):
        im["index"] = IMAGE_BASE + i
        cols = np.array([*im["slot_colors"].values(), im["corridor_color"]] if im["family"] != "4c" else [im["agent_color"]])
        ref = forbidden(conf) if im["family"] != "4c" else np.array([*canon.values(), CORRIDOR_COLOR, GOAL_COLOR, conf["agent_color"]])
        im["min_de"] = float(lab_distances(cols, ref).min())
    return images


def images_for(cfg: dict, conf: dict, slot_colors: dict, corridor_color=CORRIDOR_COLOR, agent_color=None) -> np.ndarray:
    """(n_states, H, W, 3) uint8: every state's image of the probe world (reference layout) with these colours."""
    layout = two_zones.build_layout(conf["diagonal"], "reference")
    themes = dataclasses.replace(two_zones.themes(slot_colors, agent_color or conf["agent_color"]),
                                 corridor_color=tuple(int(v) for v in corridor_color))
    k, grid_lines = cfg["render"]["k"], cfg["render"]["grid_lines"]
    return np.stack([render(s, layout, themes, k=k, grid_lines=grid_lines) for s in range(two_zones.N_STATES)])


if __name__ == "__main__":
    import sys

    import common

    cfg, conf, _ = common.setup(sys.argv[1])
    for im in step4_images(conf):
        print(im["index"], im["family"], im["name"], {q: im["slot_colors"][q] for q in QUADRANT_ORDER}, "corridor", im["corridor_color"],
              "agent", im["agent_color"], f"min dE {im['min_de']:.1f}")
