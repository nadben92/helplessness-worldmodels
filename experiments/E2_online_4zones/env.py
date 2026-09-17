"""Builds E2's 4-zone ZoneLayout (and its derived control pair) from the
SAME frozen geometry E1 uses (configs/zones/e1_4zones.yaml -- grid shape,
goal, corridor, and each quadrant's fixed (rows, cols)), with one addition:
an optional quadrant_assignment that permutes WHICH dynamics type sits in
WHICH geometric quadrant, for counterbalancing across seeds (see
docs/E2_online_4zones.md).

Deliberately not a modification of experiments/E1_offline_4zones/env.py:
E1 is frozen (see CLAUDE.md and the project's own working rules for this
task -- extensions must be explicit, tested, and never silently folded
into E1's own files). This module reuses the same shared primitives E1's
env.py reuses (helplessness.envs.zones: cross_corridor_mask,
torus_block_mask, resolve_drift) directly, rather than importing across
experiment directories (CLAUDE.md: experiments/ orchestrate, they don't
depend on each other).

Counterbalancing convention: a quadrant's (rows, cols) never move -- they
are fixed geometry, exactly as in configs/zones/e1_4zones.yaml. What moves
is which TYPE (dynamics_type + params + appearance) is assigned to that
geometric slot. quadrant_assignment maps {target_quadrant: source_type_key},
e.g. {"nw": "ne", "ne": "se", "se": "sw", "sw": "nw"} means "the type
originally configured for 'ne' is placed at the 'nw' quadrant". Drift
(for the two uncontrollable types) is ALWAYS recomputed from the TARGET
quadrant's own row_range, never copied from the source config's rows --
see build_layout()'s docstring.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.pairing import derive_control
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy
from helplessness.envs.observation.index_backend import IndexObservationBackend
from helplessness.envs.rewards.base import GoalReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout, cross_corridor_mask, resolve_drift, torus_block_mask

# Clockwise order around the 4 quadrants -- the cycle counterbalancing
# rotates over. Fixed and arbitrary (any consistent cyclic order works),
# documented once here rather than re-derived per caller.
QUADRANT_ORDER = ("nw", "ne", "se", "sw")


def identity_quadrant_assignment() -> dict[str, str]:
    """{target: target} for every quadrant -- reproduces E1's own literal
    assignment (nw=UncontrollableDeterministic, ne=ControllableNoisy,
    sw=ControllableDeterministic, se=UncontrollableNoisy) bit-for-bit."""
    return {q: q for q in QUADRANT_ORDER}


def quadrant_assignment_for_rotation(rotation: int) -> dict[str, str]:
    """One of the 4 cyclic rotations of QUADRANT_ORDER: target quadrant
    QUADRANT_ORDER[i] is assigned the type originally configured for
    QUADRANT_ORDER[(i + rotation) % 4]. rotation=0 is the identity
    (matches identity_quadrant_assignment() exactly)."""
    if rotation not in (0, 1, 2, 3):
        raise ValueError(f"rotation must be in 0..3, got {rotation}")
    n = len(QUADRANT_ORDER)
    return {
        QUADRANT_ORDER[i]: QUADRANT_ORDER[(i + rotation) % n]
        for i in range(n)
    }


def rotation_for_seed_index(seed_index: int, seeds_per_rotation: int = 5) -> int:
    """Deterministic rotation from a seed's POSITION in the seed list
    (never a random draw, per the counterbalancing spec): seed_index 0..4
    -> rotation 0, 5..9 -> rotation 1, 10..14 -> rotation 2, 15..19 ->
    rotation 3, for the default seeds_per_rotation=5 (a Latin square of
    4 rotations x 5 seeds each, 20 seeds total)."""
    return (seed_index // seeds_per_rotation) % 4


def _build_block_dynamics(block_cfg: dict, target_row_range: tuple[int, int], grid_shape: tuple[int, int], goal_rc: tuple[int, int], direction: str):
    """Builds one block's dynamics from `block_cfg` (the TYPE's own
    parameters: dynamics_type, p if noisy), but resolves drift from
    `target_row_range` -- the geometric quadrant this type is actually
    being placed at, which may differ from block_cfg's own "rows" entry
    once quadrant_assignment permutes things. Never resolves drift from
    block_cfg["rows"] directly: that would silently keep the ORIGINAL
    quadrant's direction after a permutation, exactly the bug the
    counterbalancing review flagged."""
    dtype = block_cfg["dynamics_type"]
    if dtype == "ControllableDeterministic":
        return ControllableDeterministic()
    if dtype == "ControllableNoisy":
        return ControllableNoisy(p=block_cfg["p"])
    if dtype == "UncontrollableDeterministic":
        drift = resolve_drift(target_row_range, grid_shape, goal_rc, direction)
        return UncontrollableDeterministic(drift=drift)
    if dtype == "UncontrollableNoisy":
        drift = resolve_drift(target_row_range, grid_shape, goal_rc, direction)
        return UncontrollableNoisy(drift=drift, p=block_cfg["p"])
    raise ValueError(f"unknown dynamics_type {dtype!r} in E2 zone config")


def build_layout(cfg: dict, direction: str, quadrant_assignment: dict[str, str] | None = None) -> ZoneLayout:
    """The experimental layout: corridor (controllable-deterministic,
    never permuted) plus the 4 blocks. quadrant_assignment (default:
    identity, reproducing E1's literal cfg["blocks"] assignment exactly)
    maps {target_quadrant: source_type_key} -- each target quadrant keeps
    its own fixed (rows, cols) from cfg["blocks"][target_quadrant], but
    takes its dynamics_type/params/appearance from
    cfg["blocks"][source_type_key]. Drift is always recomputed for the
    target quadrant's own rows (see _build_block_dynamics)."""
    if quadrant_assignment is None:
        quadrant_assignment = identity_quadrant_assignment()

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

    for target_quadrant in QUADRANT_ORDER:
        target_cfg = cfg["blocks"][target_quadrant]
        source_key = quadrant_assignment[target_quadrant]
        source_cfg = cfg["blocks"][source_key]

        mask = torus_block_mask(grid_shape, tuple(target_cfg["rows"]), tuple(target_cfg["cols"]))
        dynamics = _build_block_dynamics(source_cfg, tuple(target_cfg["rows"]), grid_shape, goal_rc, direction)
        zones.append(
            Zone(mask=mask, dynamics=dynamics, appearance=AppearanceSpec(category=source_cfg["appearance"]))
        )

    return ZoneLayout(grid_shape, zones)


def build_control_layout(cfg: dict, direction: str, quadrant_assignment: dict[str, str] | None = None) -> ZoneLayout:
    """The paired control: build_layout()'s zones with derive_control()
    applied per zone (CLAUDE.md: the control is always derived, never a
    second hand-written environment) -- same convention as E1's."""
    experimental = build_layout(cfg, direction, quadrant_assignment)
    control_zones = [
        Zone(mask=zone.mask, dynamics=derive_control(zone.dynamics), appearance=zone.appearance)
        for zone in experimental.zones
    ]
    return ZoneLayout(experimental.grid_shape, control_zones)


def goal_state(cfg: dict) -> int:
    grid_shape = tuple(cfg["grid_shape"])
    goal_r, goal_c = cfg["goal"]
    return int(goal_r) * grid_shape[1] + int(goal_c)


def reward_and_terminal(cfg: dict, goal_reward: float, step_reward: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    """Same convention as E1's env.py::reward_and_terminal(): step_reward
    everywhere, goal_reward (replacing, not added to, step_reward) on
    arrival at the goal, which is the only terminal state."""
    grid_shape = tuple(cfg["grid_shape"])
    n_states = grid_shape[0] * grid_shape[1]
    goal_s = goal_state(cfg)

    R = np.full(n_states, step_reward, dtype=np.float64)
    R[goal_s] = goal_reward
    terminal = np.zeros(n_states, dtype=bool)
    terminal[goal_s] = True
    return R, terminal


def build_env(
    zone_layout: ZoneLayout,
    cfg: dict,
    goal_reward: float,
    step_reward: float = -1.0,
    start_position: tuple[int, int] = (0, 0),
    max_steps: int = 36,
    seed: int | None = None,
) -> TorusZonedEnv:
    """A real, interactive TorusZonedEnv (unlike E1's build_env(), which
    only ever uses .transition_tensor() -- E2's online agent calls
    step()/reset() for real). reward_fn and terminal_states reproduce
    reward_and_terminal()'s convention exactly through
    rewards.base.GoalReward + TorusZonedEnv's terminal_states constructor
    argument (see docs/E2_online_4zones.md)."""
    goal_s = goal_state(cfg)
    reward_fn = GoalReward(goal_state=goal_s, step_reward=step_reward, goal_reward=goal_reward)
    return TorusZonedEnv(
        zone_layout=zone_layout,
        observation_backend=IndexObservationBackend(),
        reward_fn=reward_fn,
        start_position=start_position,
        max_steps=max_steps,
        seed=seed,
        terminal_states=[goal_s],
    )
