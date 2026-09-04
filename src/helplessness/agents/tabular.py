"""Tabular model-based agent: counts transitions, derives T_hat/R_hat, plans.

Pure container + estimators, with no learning loop: observe() is called by
the caller (agents/collect.py or an online loop later), the agent never
touches the environment directly.
"""

from __future__ import annotations

import numpy as np

from helplessness.planning.value_iteration import GAMMA, value_iteration


class TabularModelBasedAgent:
    """Tabular T_hat/R_hat with a Dirichlet prior, planning by value iteration."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha

        self.N = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
        self.R_sum = np.zeros(n_states, dtype=np.float64)
        self.R_count = np.zeros(n_states, dtype=np.int64)

        self.V: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.policy: np.ndarray | None = None

    def observe(self, s: int, a: int, r: float, s_next: int) -> None:
        """Records a raw transition. Indifferent to terminated/truncated:
        it is up to the caller to handle episode resets."""
        self.N[s, a, s_next] += 1
        self.R_sum[s_next] += r
        self.R_count[s_next] += 1

    @property
    def T_hat(self) -> np.ndarray:
        """(S, A, S), normalised rows. Dirichlet(alpha) prior: a (s,a) pair
        never visited yields a uniform distribution over states rather than
        a division by zero."""
        counts = self.N + self.alpha
        return counts / counts.sum(axis=2, keepdims=True)

    @property
    def R_hat(self) -> np.ndarray:
        """(S,), reward on arrival. Falls back to 0.0 wherever the state was
        never visited."""
        R = np.zeros(self.n_states, dtype=np.float64)
        visited = self.R_count > 0
        R[visited] = self.R_sum[visited] / self.R_count[visited]
        return R

    @property
    def visited_mask(self) -> np.ndarray:
        """(S, A) boolean: (s,a) pairs observed at least once."""
        return self.N.sum(axis=2) > 0

    def plan(self, terminal: np.ndarray, gamma: float = GAMMA) -> None:
        """Value iteration on (T_hat, R_hat). Stores V, Q, policy."""
        P = self.T_hat
        R = self.R_hat
        self.V, self.policy = value_iteration(P, R, terminal, gamma=gamma)
        self.Q = np.einsum("sap,p->sa", P, R + gamma * self.V * (~terminal))

    def save(self, path: str) -> None:
        """Saves the raw counts (N, R_sum, R_count) and alpha -- enough to
        reconstruct T_hat/R_hat identically. V/Q/policy are not saved: they
        are derived quantities, recomputable by plan() without recollecting
        any data (not a retraining, just a replanning)."""
        np.savez(
            path,
            N=self.N,
            R_sum=self.R_sum,
            R_count=self.R_count,
            alpha=np.array(self.alpha),
            n_states=np.array(self.n_states),
            n_actions=np.array(self.n_actions),
        )

    @classmethod
    def load(cls, path: str) -> "TabularModelBasedAgent":
        """Reloads an agent from the counts saved by save(). Does not
        recompute V/Q/policy: call plan(terminal) after loading."""
        data = np.load(path)
        agent = cls(int(data["n_states"]), int(data["n_actions"]), alpha=float(data["alpha"]))
        agent.N = data["N"]
        agent.R_sum = data["R_sum"]
        agent.R_count = data["R_count"]
        return agent
