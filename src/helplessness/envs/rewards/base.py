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


def reward_vector(reward_fn: RewardFn, n_states: int) -> np.ndarray:
    return np.array([reward_fn(s) for s in range(n_states)], dtype=np.float64)
