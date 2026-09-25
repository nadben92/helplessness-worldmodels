"""E3's environment builder.

Reuses E1/E2's frozen zone/dynamics/permutation logic UNCHANGED: imports
experiments/E2_online_4zones/env.py via
helplessness.experiment_loader.load_experiment_module (see
docs/e3/DECISIONS.md, points 4 and 5) rather than re-implementing
counterbalancing, drift resolution, or the paired-control derivation.
The only thing E3 adds is the pixel observation backend E2 never needed
(E2 was tabular): AllocentricPixelBackend + a fixed, per-run Themes
object, wired onto the SAME ZoneLayout/TorusZonedEnv machinery.

Per docs/e3/DECISIONS.md point 5: E3's seed `k` reuses E2's own
rotation_for_seed_index/quadrant_assignment_for_rotation output
bit-for-bit, so E3's seed k is directly comparable to E2's seed k.
"""

from __future__ import annotations

import numpy as np

from helplessness.config import load_yaml_block
from helplessness.envs.appearance.theme_sampling import Themes, build_themes
from helplessness.envs.observation.allocentric_pixel_backend import AllocentricPixelBackend
from helplessness.envs.rewards.base import GoalReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import ZoneLayout
from helplessness.experiment_loader import load_experiment_module

_e2_env = load_experiment_module("E2_online_4zones", "env.py", "e3_e2_env")

CFG: dict = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE: tuple[int, int] = tuple(CFG["grid_shape"])
SEEDS_PER_ROTATION = 5

QUADRANT_ORDER = _e2_env.QUADRANT_ORDER
ZONE_CATEGORIES = tuple(CFG["blocks"][q]["appearance"] for q in QUADRANT_ORDER)

# Block-factor classification of E2's 4 counterbalancing rotations (see
# docs/e3/DECISIONS.md, Phase 1 T2 reformulation, and
# tests/e3/test_e3_symmetry.py): E2's rotations 0..3 are the cyclic group
# C4 (successive 90-degree turns of WHICH TYPE occupies which quadrant),
# not the goal-centred axial-reflection group {identity, row-reflect,
# col-reflect, point-reflect} that actually transports P exactly for
# every zone type (drift included). Rotations 0 and 2 coincide with two
# elements of that reflection group (identity, point-reflect/180-degree
# turn) and are therefore related to each other by an exact symmetry --
# "class A". Rotations 1 and 3 (the two genuine quarter turns) are
# likewise related to EACH OTHER by point-reflect, but NOT to class A by
# any element of the reflection group -- "class B". Verified exhaustively
# in tests/e3/test_e3_symmetry.py; ROTATION_CLASS is this experiment's
# declared block factor, not a hypothesis.
ROTATION_CLASS: dict[int, str] = {0: "A", 1: "B", 2: "A", 3: "B"}


def rotation_for_seed_index(seed_index: int, seeds_per_rotation: int = SEEDS_PER_ROTATION) -> int:
    """Reused unchanged from E2 -- see module docstring."""
    return _e2_env.rotation_for_seed_index(seed_index, seeds_per_rotation)


def quadrant_assignment_for_rotation(rotation: int) -> dict[str, str]:
    """Reused unchanged from E2 -- see module docstring."""
    return _e2_env.quadrant_assignment_for_rotation(rotation)


def block_class_for_seed(seed_index: int, seeds_per_rotation: int = SEEDS_PER_ROTATION) -> str:
    """"A" or "B" -- the symmetry-class block factor for this seed (see
    ROTATION_CLASS above). Recorded in every run's metadata and reported
    as an explicit column in aggregated tables from Phase 4 onward
    (spec T2 reformulation, decision 2026-09-23)."""
    return ROTATION_CLASS[rotation_for_seed_index(seed_index, seeds_per_rotation)]


def build_layout(quadrant_assignment: dict[str, str] | None = None) -> ZoneLayout:
    return _e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=quadrant_assignment)


def build_control_layout(quadrant_assignment: dict[str, str] | None = None) -> ZoneLayout:
    return _e2_env.build_control_layout(CFG, DIRECTION, quadrant_assignment=quadrant_assignment)


def goal_state() -> int:
    return _e2_env.goal_state(CFG)


def reward_and_terminal(goal_reward: float, step_reward: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    return _e2_env.reward_and_terminal(CFG, goal_reward=goal_reward, step_reward=step_reward)


def build_themes_for_run(rng: np.random.Generator, min_dist: float, marker_min_dist: float | None = None) -> Themes:
    """One fixed Themes object for a whole run -- shared between the
    experimental and reference environments for a given seed (spec 2.3),
    since both use the SAME ZONE_CATEGORIES and the SAME goal_state()."""
    return build_themes(ZONE_CATEGORIES, goal_state=goal_state(), rng=rng, min_dist=min_dist, marker_min_dist=marker_min_dist)


def build_pixel_env(
    layout: ZoneLayout,
    themes: Themes,
    goal_reward: float,
    step_reward: float = -1.0,
    start_position: tuple[int, int] = (0, 0),
    max_steps: int = 36,
    seed: int | None = None,
    k: int = 6,
    grid_lines: bool = False,
) -> TorusZonedEnv:
    """A real, interactive pixel-observation TorusZonedEnv. Mirrors
    E2's own build_env() convention for reward/terminal (GoalReward +
    terminal_states=[goal_s]), but with AllocentricPixelBackend instead
    of E2's IndexObservationBackend -- TorusZonedEnv itself needs no
    change to support this (observation backend is a constructor
    argument, see envs/torus_env.py's own docstring).

    theme_seed=None always: this environment's themes are fixed for the
    whole run (see AllocentricPixelBackend), never re-derived per
    episode -- callers must never override theme_seed at reset().
    """
    goal_s = goal_state()
    reward_fn = GoalReward(goal_state=goal_s, step_reward=step_reward, goal_reward=goal_reward)
    backend = AllocentricPixelBackend(themes=themes, k=k, grid_lines=grid_lines)
    return TorusZonedEnv(
        zone_layout=layout,
        observation_backend=backend,
        reward_fn=reward_fn,
        start_position=start_position,
        max_steps=max_steps,
        seed=seed,
        theme_seed=None,
        terminal_states=[goal_s],
    )
