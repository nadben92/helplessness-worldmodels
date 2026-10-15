"""The world of a two-zone environment (E4 on, docs/E4_generalization_gradient.md): layout, themes, pixel environment,
background and pre-rendered state images, as pipeline.world.World -- the object the closed-loop run
(pipeline.agent_run.run_one with hooks) and the probes take. Built from explicit colours (no theme stream: E4's
themes are fixed by configs/e4/themes.yaml), a diagonal and a condition; the episode parameters and the dynamics
seed are the run's.
"""

from __future__ import annotations

import numpy as np
import torch

from helplessness.envs.layouts import four_zones_pixel, two_zones
from helplessness.envs.observation.allocentric_pixel_backend import render, render_background
from helplessness.pipeline.world import World
from helplessness.streams import worldmodel as wm_streams


def build_world(slot_colors: dict, agent_color, diagonal: int, condition: str, render_cfg: dict, episode: dict,
                seed: int, env_condition: str | None = None) -> World:
    """slot_colors: slot -> RGB of the image; condition: the dynamics ("experimental" = UD at the diagonal's first slot,
    "reference" = everything controllable-deterministic). The dynamics seed is drawn from the env_dynamics stream of
    (seed, env_condition or condition), as E3's build_world. Tensors on the CPU."""
    layout = two_zones.build_layout(diagonal, condition)
    themes = two_zones.themes(slot_colors, agent_color)
    k, grid_lines = render_cfg["k"], render_cfg["grid_lines"]
    env_seed = wm_streams.env_seed(wm_streams.spawn_streams(seed, env_condition or condition))
    env = four_zones_pixel.build_pixel_env(layout, themes, goal_reward=episode["goal_reward"], step_reward=episode["step_reward"],
                                           max_steps=episode["max_steps"], seed=env_seed, k=k, grid_lines=grid_lines)
    background = render_background(layout, themes, k=k, grid_lines=grid_lines)
    background_t = torch.as_tensor(background, dtype=torch.float32).permute(2, 0, 1) / 255.0
    templates_np = np.stack([render(s, layout, themes, k=k, grid_lines=grid_lines) for s in range(two_zones.N_STATES)])
    templates_t = torch.as_tensor(templates_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return World(seed, condition, layout, themes, env, env_seed, background_t, templates_t, templates_np)
