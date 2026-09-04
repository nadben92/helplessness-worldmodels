"""Minimal common interface for environments.

Kept thin on purpose: a second environment (roadmap step 2) must be able to
plug into every downstream call site (agents, metrics, interventions,
evaluation) without those call sites branching on which environment they got.
Gymnasium-style reset/step plus the four array-producing methods that the
rest of the codebase actually depends on.
"""

from __future__ import annotations

import abc

import numpy as np


class Env(abc.ABC):
    """Common interface. Subclasses provide the concrete dynamics."""

    @abc.abstractmethod
    def reset(self, seed: int | None = None, **kwargs):
        """Returns (obs, info). See Gymnasium's reset() convention."""

    @abc.abstractmethod
    def step(self, action: int):
        """Returns (obs, reward, terminated, truncated, info)."""

    @abc.abstractmethod
    def transition_tensor(self, *args, **kwargs) -> np.ndarray:
        """Exact P[s, a, s']. Ground truth -- must stay consistent with step()."""

    @abc.abstractmethod
    def reward_vector(self) -> np.ndarray:
        """R[s'] : reward received upon arriving at s'."""

    @abc.abstractmethod
    def terminal_mask(self) -> np.ndarray:
        """Boolean mask of genuinely absorbing states (never a truncation)."""

    @abc.abstractmethod
    def controllability(self, *args, **kwargs) -> float:
        """Average I(a ; s' | s) in bits over non-terminal states."""
