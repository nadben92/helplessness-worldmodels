"""RewardFn: reserved extension point, structure not decided yet.

Kept to the same minimal shape the rest of the codebase already uses
(reward on arrival at s', see legacy GridWorld's reward_vector()) purely
so TorusZonedEnv has something concrete to run and test against --
ZeroReward is a placeholder, not a design decision. Whoever designs the
actual reward structure for E1+ should feel free to replace this
interface's shape entirely; nothing else in envs/ depends on reward
beyond "TorusZonedEnv needs *a* RewardFn to call".
"""

from __future__ import annotations

import abc

import numpy as np


class RewardFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, s: int) -> float:
        """Reward received on arrival at state s (flat index, row-major)."""


class ZeroReward(RewardFn):
    """Placeholder: no reward signal at all. Lets TorusZonedEnv be built
    and its dynamics tested before a real reward structure exists."""

    def __call__(self, s: int) -> float:
        return 0.0


class GoalReward(RewardFn):
    """step_reward on arrival at any state, except goal_state which gets
    goal_reward instead (not step_reward + goal_reward -- the goal simply
    replaces the per-step cost, exactly like
    experiments/E1_offline_4zones/env.py::reward_and_terminal()'s R array:
    `R = full(n_states, step_reward); R[goal_s] = goal_reward`).

    Built so that TorusZonedEnv.step()/.reward_vector() can reproduce that
    exact convention when an experiment (E2 onward) needs real step()/
    reset() interaction rather than E1's DP-only, never-call-step() usage
    -- see torus_env.py's module docstring on the terminal_states
    constructor argument this pairs with.
    """

    def __init__(self, goal_state: int, step_reward: float, goal_reward: float):
        self.goal_state = int(goal_state)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)

    def __call__(self, s: int) -> float:
        return self.goal_reward if int(s) == self.goal_state else self.step_reward


def reward_vector(reward_fn: RewardFn, n_states: int) -> np.ndarray:
    return np.array([reward_fn(s) for s in range(n_states)], dtype=np.float64)
