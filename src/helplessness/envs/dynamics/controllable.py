"""Controllable dynamics: the chosen action determines the outcome, exactly
(deterministic) or with some probability of being overridden (noisy).
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.dynamics.base import ACTIONS, N_ACTIONS, DynamicsSpec, _rc_to_s, _wrap


class ControllableDeterministic(DynamicsSpec):
    """The intended action always executes. Ceiling of the family: on a
    torus (no walls), every action leads to a distinct cell, so
    I(a;s'|s) = log2(n_actions) exactly."""

    def transition_row(self, rc, a, grid_shape) -> np.ndarray:
        n_states = grid_shape[0] * grid_shape[1]
        row = np.zeros(n_states)
        dr, dc = ACTIONS[a]
        s_next = _rc_to_s(_wrap((rc[0] + dr, rc[1] + dc), grid_shape), grid_shape)
        row[s_next] = 1.0
        return row

    def analytic_controllability(self, n_actions: int = N_ACTIONS) -> float:
        return float(np.log2(n_actions))


class ControllableNoisy(DynamicsSpec):
    """With probability p, the executed action is replaced by a uniform
    draw over all n_actions, independently of the one chosen -- the same
    stochasticity model as the legacy GridWorld's p, generalised to a
    torus. p=0 is ControllableDeterministic; p=1 is fully uncontrolled
    (equivalent in outcome, not in name, to UncontrollableNoisy with a
    uniform move distribution).
    """

    def __init__(self, p: float):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def transition_row(self, rc, a, grid_shape) -> np.ndarray:
        n_states = grid_shape[0] * grid_shape[1]
        n_actions = len(ACTIONS)
        row = np.zeros(n_states)
        for a_eff in range(n_actions):
            prob = self.p / n_actions + (1.0 - self.p) * (a_eff == a)
            if prob > 0.0:
                dr, dc = ACTIONS[a_eff]
                s_next = _rc_to_s(_wrap((rc[0] + dr, rc[1] + dc), grid_shape), grid_shape)
                row[s_next] += prob
        return row

    def analytic_controllability(self, n_actions: int = N_ACTIONS) -> float:
        """Closed form for a symmetric channel: each action leads to its
        own target with probability q = (1-p) + p/n and to each of the
        other (n-1) targets with probability p/n (all n targets distinct
        on a torus, no walls to collide two actions into one cell).
        I = log2(n) + q*log2(q) + (n-1)*(p/n)*log2(p/n), with the
        convention x*log2(x) = 0 at x=0 (p=0 -> q=1 -> I=log2(n); p=1 ->
        q=1/n -> I=0)."""
        n = n_actions
        p = self.p
        q = (1.0 - p) + p / n

        def xlog2x(x: float) -> float:
            return 0.0 if x <= 0.0 else x * np.log2(x)

        return float(np.log2(n) + xlog2x(q) + (n - 1) * xlog2x(p / n))
