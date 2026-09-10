"""Uncontrollable dynamics: the chosen action has no influence on the
outcome at all -- I(a;s'|s) = 0 exactly, by construction, for both types.

UncontrollableNoisy shares its mixture structure with ControllableNoisy on
purpose: "with probability (1-p), the clean component; with probability p,
a uniform random direction" -- the only difference is what plays the role
of the clean component (a fixed drift here, the chosen action there). That
shared shape is what lets pairing.derive_control map
UncontrollableNoisy(drift, p) -> ControllableNoisy(p) and genuinely
preserve p, rather than collapsing every uncontrollable zone to the same
"how much randomness" bucket regardless of how noisy it actually was.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.dynamics.base import ACTIONS, N_ACTIONS, DynamicsSpec, _rc_to_s, _wrap


class UncontrollableDeterministic(DynamicsSpec):
    """"Conveyor belt": a fixed displacement applied every step,
    independent of the action -- the action is read and ignored.
    Equivalent to UncontrollableNoisy(drift, p=0) (tested directly), but
    kept as its own type since it needs no noise parameter at all."""

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
    """With probability (1-p), the fixed `drift` displacement applies
    (like UncontrollableDeterministic); with probability p, a uniform
    random direction applies instead, independent of the action either
    way. p=0 is exactly UncontrollableDeterministic(drift); p=1 is a pure
    random walk regardless of drift.

    The action never appears in transition_row() at all -- I(a;s'|s) = 0
    exactly, for every p, not just approximately for large p.
    """

    def __init__(self, drift: tuple[int, int], p: float):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.drift = (int(drift[0]), int(drift[1]))
        self.p = float(p)

    def transition_row(self, rc, a, grid_shape) -> np.ndarray:
        n_states = grid_shape[0] * grid_shape[1]
        n_actions = len(ACTIONS)
        row = np.zeros(n_states)

        dr, dc = self.drift
        drift_s = _rc_to_s(_wrap((rc[0] + dr, rc[1] + dc), grid_shape), grid_shape)
        row[drift_s] += 1.0 - self.p

        for direction in range(n_actions):
            dr2, dc2 = ACTIONS[direction]
            s2 = _rc_to_s(_wrap((rc[0] + dr2, rc[1] + dc2), grid_shape), grid_shape)
            row[s2] += self.p / n_actions

        return row

    def analytic_controllability(self, n_actions: int = N_ACTIONS) -> float:
        return 0.0
