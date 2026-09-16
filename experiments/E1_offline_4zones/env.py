"""Builds the E1 4-zone ZoneLayout (and its derived control pair) from the
resolved config. Orchestration only, per CLAUDE.md's "Adding an
experiment": the reusable geometry (block/corridor masks, drift
resolution, symmetry assertion, reflection, interior mask) lives in
helplessness.envs.zones so docs/E2_online_4zones.md's experiment can reuse
this same layout unchanged.

Direction ("away_from_goal" | "toward_goal") is never baked into
configs/zones/e1_4zones.yaml: it is Etape A's swept parameter, so it is
always passed explicitly here rather than defaulted.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.pairing import derive_control
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy
from helplessness.envs.observation.index_backend import IndexObservationBackend
from helplessness.envs.rewards.base import ZeroReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout, cross_corridor_mask, resolve_drift, torus_block_mask

_DRIFT_TYPES = ("UncontrollableDeterministic", "UncontrollableNoisy")


def _build_block_dynamics(block_cfg: dict, grid_shape: tuple[int, int], goal_rc: tuple[int, int], direction: str):
    dtype = block_cfg["dynamics_type"]
    row_range = tuple(block_cfg["rows"])
    if dtype == "ControllableDeterministic":
        return ControllableDeterministic()
    if dtype == "ControllableNoisy":
        return ControllableNoisy(p=block_cfg["p"])
    if dtype == "UncontrollableDeterministic":
        drift = resolve_drift(row_range, grid_shape, goal_rc, direction)
        return UncontrollableDeterministic(drift=drift)
    if dtype == "UncontrollableNoisy":
        drift = resolve_drift(row_range, grid_shape, goal_rc, direction)
        return UncontrollableNoisy(drift=drift, p=block_cfg["p"])
    raise ValueError(f"unknown dynamics_type {dtype!r} in E1 zone config")


def build_layout(cfg: dict, direction: str) -> ZoneLayout:
    """The experimental layout: corridor (controllable-deterministic) plus
    the 4 blocks, each built from cfg["blocks"] with drift (for the two
    uncontrollable blocks) resolved via `direction`."""
    grid_shape = tuple(cfg["grid_shape"])
    goal_rc = tuple(cfg["goal"])

    corridor_cfg = cfg["corridor"]
    corridor_mask = cross_corridor_mask(grid_shape, tuple(corridor_cfg["rows"]), tuple(corridor_cfg["cols"]))
    zones = [
        Zone(
            mask=corridor_mask,
            dynamics=ControllableDeterministic(),
            appearance=AppearanceSpec(category=corridor_cfg["appearance"]),
        )
    ]

    for block_cfg in cfg["blocks"].values():
        mask = torus_block_mask(grid_shape, tuple(block_cfg["rows"]), tuple(block_cfg["cols"]))
        dynamics = _build_block_dynamics(block_cfg, grid_shape, goal_rc, direction)
        zones.append(Zone(mask=mask, dynamics=dynamics, appearance=AppearanceSpec(category=block_cfg["appearance"])))

    return ZoneLayout(grid_shape, zones)


def build_control_layout(cfg: dict, direction: str) -> ZoneLayout:
    """The paired control: build_layout()'s zones with derive_control()
    applied per zone (CLAUDE.md: the control is always derived, never a
    second hand-written environment)."""
    experimental = build_layout(cfg, direction)
    control_zones = [
        Zone(mask=zone.mask, dynamics=derive_control(zone.dynamics), appearance=zone.appearance)
        for zone in experimental.zones
    ]
    return ZoneLayout(experimental.grid_shape, control_zones)


def build_env(zone_layout: ZoneLayout) -> TorusZonedEnv:
    """A TorusZonedEnv used only to reuse the existing, tested
    transition_tensor() -- E1 never calls step()/reset() in a rollout loop
    (Etape A is pure DP under the true kernel, Etape B samples directly
    from zone_layout.transition_row() per (s,a) pair), so reward_fn here is
    an unused placeholder; planning-time R/terminal arrays are built
    independently by reward_and_terminal() below."""
    return TorusZonedEnv(
        zone_layout=zone_layout,
        observation_backend=IndexObservationBackend(),
        reward_fn=ZeroReward(),
    )


def reward_and_terminal(cfg: dict, goal_reward: float, step_reward: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    """Plain numpy R (step_reward everywhere, goal_reward on arrival at the
    goal state) and terminal (True only at the goal) arrays for planning --
    built directly rather than through a RewardFn/TerminationFn, since
    nothing in E1 calls env.step(). See docs/E1_offline_4zones.md for why
    TorusZonedEnv itself is not extended for this."""
    grid_shape = tuple(cfg["grid_shape"])
    n_states = grid_shape[0] * grid_shape[1]
    goal_r, goal_c = cfg["goal"]
    goal_s = int(goal_r) * grid_shape[1] + int(goal_c)

    R = np.full(n_states, step_reward, dtype=np.float64)
    R[goal_s] = goal_reward
    terminal = np.zeros(n_states, dtype=bool)
    terminal[goal_s] = True
    return R, terminal
