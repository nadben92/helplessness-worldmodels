"""Uncontrollable dynamics: the chosen action has no influence on the
outcome at all -- I(a;s'|s) = 0 exactly, by construction, for both types.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.dynamics.base import ACTIONS, N_ACTIONS, DynamicsSpec, _rc_to_s, _wrap


class UncontrollableDeterministic(DynamicsSpec):
    """"Conveyor belt": a fixed displacement applied every step,
    independent of the action -- the action is read and ignored."""

    def __init__(self, drift: tuple[int, int]):
        self.drift = (int(drift[0]), int(drift[1]))

    def transition_row(self, rc, a, grid_shape) -> np.ndarray:
        n_states = grid_shape[0] * grid_shape[1]
        row = np.zeros(n_states)
        dr, dc = self.drift
        s_next = _rc_to_s(_wrap((rc[0] + dr, rc[1] + dc), grid_shape), grid_shape)
        row[s_next] = 1.0
        return row

    def analytic_controllability(self, n_actions: int = N_ACTIONS) -> float:
        return 0.0


class UncontrollableNoisy(DynamicsSpec):
    """The executed displacement is drawn from move_probs, independent of
    the chosen action -- move_probs defaults to uniform over the 5
    directions (equivalent in outcome, though not in the agent's own
    belief about why, to ControllableNoisy(p=1))."""

    def __init__(self, move_probs: np.ndarray | None = None):
        if move_probs is None:
            move_probs = np.full(len(ACTIONS), 1.0 / len(ACTIONS))
        move_probs = np.asarray(move_probs, dtype=np.float64)
        if move_probs.shape != (len(ACTIONS),):
            raise ValueError(f"move_probs must have shape ({len(ACTIONS)},), got {move_probs.shape}")
        if not np.isclose(move_probs.sum(), 1.0):
            raise ValueError(f"move_probs must sum to 1, got {move_probs.sum()}")
        self.move_probs = move_probs

    def transition_row(self, rc, a, grid_shape) -> np.ndarray:
        n_states = grid_shape[0] * grid_shape[1]
        row = np.zeros(n_states)
        for direction, prob in enumerate(self.move_probs):
            if prob > 0.0:
                dr, dc = ACTIONS[direction]
                s_next = _rc_to_s(_wrap((rc[0] + dr, rc[1] + dc), grid_shape), grid_shape)
                row[s_next] += prob
        return row

    def analytic_controllability(self, n_actions: int = N_ACTIONS) -> float:
        return 0.0
