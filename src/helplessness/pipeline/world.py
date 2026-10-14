"""E3's run pipeline: the world of a (seed, condition) -- layout, themes, pixel environment, pre-rendered
templates, the episode parameters, zone bookkeeping, data collection with counts, and the tabular twin's T_hat.
Moved verbatim from experiments/E3_worldmodel_4zones/run_mini.py on 2026-10-14 (run_mini.py re-exports every
name); the module-level names those functions read are imports here (E3's environment from
helplessness.envs.layouts.four_zones_pixel, its streams from helplessness.streams.worldmodel, REPO_ROOT from
helplessness.config).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re

import numpy as np
import torch
import yaml

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.agents.worldmodel.loop import draw_start_state, non_terminal_states
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.config import REPO_ROOT
from helplessness.envs.appearance.theme_sampling import build_themes, Themes
from helplessness.envs.layouts import four_zones_pixel as e3_env
from helplessness.envs.observation.allocentric_pixel_backend import render, render_background
from helplessness.envs.zones import geographic_support_mask
from helplessness.streams import worldmodel as e3_rng


# Episode parameters (reward, step cost, horizon, gamma) come from the experiment's own
# config, checked against E2's values by tests/e3/test_e3_protocol_conformance.py.
EXPERIMENT_CONFIG_PATH = REPO_ROOT / "experiments" / "E3_worldmodel_4zones" / "config.yaml"
EXPERIMENT_ID = "E3_worldmodel_4zones"
ZONE_PRESET = "4zones"
GRID_SHAPE = (12, 12)
N_STATES, N_ACTIONS = 144, 5
PAIR_ROLE = {"experimental": "experimental", "reference": "control"}  # runs.make_run_name's vocabulary


NOISY_ZONES = ("controllable_noisy", "uncontrollable_noisy")


def zone_of_state(layout, s: int) -> str:
    r, c = divmod(s, layout.grid_shape[1])
    return layout.zone_at((r, c)).appearance.category


def zone_states(layout, category: str) -> list[int]:
    return [s for s in range(N_STATES) if zone_of_state(layout, s) == category]


def _dynamics_kind(spec) -> str:
    """ControllableDeterministic -> "controllable_deterministic", etc."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(spec).__name__).lower()


def zone_kinds(layout) -> dict[str, str]:
    """Appearance category -> dynamics type of that zone in this layout
    (identity in the experimental environment; in the reference, the
    uncontrollable types become their controllable counterparts)."""
    kinds = {}
    for zone in layout.zones:
        category = zone.appearance.category
        if category in e3_env.ZONE_CATEGORIES:
            kinds[category] = _dynamics_kind(zone.dynamics)
    return kinds


def load_episode_config() -> dict:
    """experiments/E3_worldmodel_4zones/config.yaml's episode block:
    goal_reward, step_reward, max_steps (H), gamma."""
    with open(EXPERIMENT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["episode"]


def run_identity(seed: int, themes, condition: str = "experimental") -> dict:
    """What defines a run's world, for meta.yaml (spec section 10): the
    condition, the rotation and quadrant assignment (the permutation), the
    block class A/B and the theme colours."""
    rotation = e3_env.rotation_for_seed_index(seed)
    return {
        "condition": condition,
        "rotation": int(rotation),
        "quadrant_assignment": {str(k): str(v) for k, v in e3_env.quadrant_assignment_for_rotation(rotation).items()},
        "block_class": e3_env.block_class_for_seed(seed),
        "themes": themes_to_dict(themes),
    }


def themes_to_dict(themes) -> dict:
    return {
        "zone_colors": {cat: list(map(int, rgb)) for cat, rgb in themes.zone_colors.items()},
        "agent_marker_color": list(map(int, themes.agent_marker_color)),
        "corridor_color": list(map(int, themes.corridor_color)),
        "goal_color": list(map(int, themes.goal_color)),
        "goal_state": int(themes.goal_state),
    }


def themes_from_dict(d: dict) -> Themes:
    return Themes(
        zone_colors={cat: tuple(int(v) for v in rgb) for cat, rgb in d["zone_colors"].items()},
        agent_marker_color=tuple(int(v) for v in d["agent_marker_color"]), goal_state=int(d["goal_state"]),
        corridor_color=tuple(int(v) for v in d["corridor_color"]), goal_color=tuple(int(v) for v in d["goal_color"]),
    )


@dataclasses.dataclass
class World:
    """Everything fixed for one (seed, condition): layout, themes,
    interactive environment, background and the pre-rendered images."""

    seed: int
    condition: str
    layout: object
    themes: Themes
    env: object
    env_seed: int
    background_t: torch.Tensor
    templates_t: torch.Tensor
    templates_np: np.ndarray

    @property
    def zone_kinds(self) -> dict[str, str]:
        return zone_kinds(self.layout)

    def templates_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.templates_np).tobytes()).hexdigest()


def build_world(cfg: dict, seed: int, condition: str = "experimental", themes=None) -> World:
    """Layout, themes, environment, background and the 144 templates of a
    (seed, condition). The reference environment (spec 2.3) is the
    experimental layout with derive_control applied per zone
    (env.build_control_layout): same permutation, same themes -- always
    the EXPERIMENTAL seed's theme stream -- same goal. The dynamics seed
    comes from the condition's own env_dynamics stream (rng_streams.py);
    the episode parameters from experiments/E3_worldmodel_4zones/config.yaml.
    Tensors are returned on the CPU; the caller moves them. themes (a
    Themes object) replaces the seed's theme draw -- for tests that must
    not depend on the theme stream."""
    if condition not in e3_rng.CONDITIONS:
        raise ValueError(f"condition must be one of {e3_rng.CONDITIONS}, got {condition!r}")
    rotation = e3_env.rotation_for_seed_index(seed)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    layout = e3_env.build_layout(assignment) if condition == "experimental" else e3_env.build_control_layout(assignment)
    k = cfg["render"]["k"]
    if themes is None:
        themes = build_themes(
            e3_env.ZONE_CATEGORIES, goal_state=e3_env.goal_state(), rng=e3_rng.spawn_streams(seed)["theme"],
            min_dist=cfg["render"]["cielab_min_dist"], marker_min_dist=cfg["render"]["cielab_marker_min_dist"],
        )
    episode = load_episode_config()
    env_seed = e3_rng.env_seed(e3_rng.spawn_streams(seed, condition))
    env = e3_env.build_pixel_env(
        layout, themes, goal_reward=episode["goal_reward"], step_reward=episode["step_reward"], max_steps=episode["max_steps"],
        seed=env_seed, k=k, grid_lines=cfg["render"]["grid_lines"],
    )
    background = render_background(layout, themes, k=k, grid_lines=cfg["render"]["grid_lines"])
    background_t = torch.as_tensor(background, dtype=torch.float32).permute(2, 0, 1) / 255.0
    templates_np = np.stack([render(s, layout, themes, k=k, grid_lines=cfg["render"]["grid_lines"]) for s in range(N_STATES)])
    templates_t = torch.as_tensor(templates_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return World(seed, condition, layout, themes, env, env_seed, background_t, templates_t, templates_np)


def build_run(cfg: dict, seed: int, themes=None, condition: str = "experimental"):
    """(layout, themes, env, background_t, templates_t) of build_world."""
    w = build_world(cfg, seed, condition, themes)
    return w.layout, w.themes, w.env, w.background_t, w.templates_t


def collect_and_count(
    env, replay: ReplayBuffer, n_steps: int, rng: np.random.Generator, n_states: int, n_actions: int,
    start_rng: np.random.Generator, starts_out: list | None = None, next_states_out: list | None = None,
) -> np.ndarray:
    """Local to run_mini.py (not a modification of Phase 2's
    collect_fixed_policy_data): fills replay AND accumulates
    N[s, a, s'] counts in the same pass, so the report's tabular-twin
    comparison uses the EXACT SAME transitions the world model trained
    on, without a second collection pass. Every episode starts in a state
    drawn uniformly among the non-terminal states from start_rng (E2's
    protocol, docs/e3/DECISIONS.md 2026-09-25); starts_out, if given,
    receives every start state in order (including one for a final,
    unfinished episode); next_states_out, the true s' of every stored step
    (the replay's next entry is a reset state at an episode's end)."""
    N = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    start_states = non_terminal_states(env)

    def reset():
        obs0, info0 = env.reset(start_state=draw_start_state(start_states, start_rng))
        if starts_out is not None:
            starts_out.append(int(info0["state"]))
        return obs0, info0

    obs, info = reset()
    s = info["state"]
    is_first = True
    for _ in range(n_steps):
        a = int(rng.integers(n_actions))
        obs_next, r, terminated, truncated, info = env.step(a)
        s_next = info["state"]
        replay.add(x=obs, a=a, r=r, terminal=terminated, truncated=truncated, is_first=is_first, s=s)
        N[s, a, s_next] += 1
        if next_states_out is not None:
            next_states_out.append(int(s_next))
        is_first = terminated or truncated
        if is_first:
            obs_next, info = reset()
        s = info["state"]
        obs = obs_next
    return N


def build_tabular_twin(N: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
    """T_hat_B from the SAME counts the world model trained on -- reuses
    helplessness.agents.tabular.TabularModelBasedAgent directly (core
    library, alpha=1 matching E2's own reference condition, restricted
    to the geographic support E2 also uses), never a reimplementation."""
    support = geographic_support_mask(GRID_SHAPE)
    agent = TabularModelBasedAgent(n_states, n_actions, alpha=1.0, alpha_support=support)
    agent.N = N
    return agent.T_hat
