"""Note on ACTIONS/N_ACTIONS: re-exported below are the legacy GridWorld's
(4 actions, no "stay"). The new zone/torus family has its own, different
ACTIONS/N_ACTIONS (5 actions) in helplessness.envs.dynamics -- imported
from there explicitly (`from helplessness.envs.dynamics import ACTIONS`),
never re-exported here under the same bare name, to avoid exactly the kind
of silent ambiguity CLAUDE.md warns against for the controllability
measure: two things sharing a name must never quietly diverge."""

from helplessness.envs.base import Env
from helplessness.envs.legacy.gridworld import (
    ACTION_NAMES,
    ACTIONS,
    ALLOWED_CHARS,
    DEFAULT_LAYOUT,
    N_ACTIONS,
    GridWorld,
)
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout

__all__ = [
    "Env",
    "GridWorld",
    "ACTIONS",
    "ACTION_NAMES",
    "N_ACTIONS",
    "DEFAULT_LAYOUT",
    "ALLOWED_CHARS",
    "TorusZonedEnv",
    "Zone",
    "ZoneLayout",
]
