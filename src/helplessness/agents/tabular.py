"""Tabular model-based agent: counts transitions, derives T_hat/R_hat, plans.

Pure container + estimators, with no learning loop: observe() is called by
the caller (agents/collect.py or an online loop later), the agent never
touches the environment directly.
"""

from __future__ import annotations

import numpy as np

from helplessness.agents.base import Agent
from helplessness.planning.value_iteration import GAMMA, value_iteration


class TabularModelBasedAgent(Agent):
    """Tabular T_hat/R_hat with a Dirichlet prior, planning by value iteration.

    alpha_support: optional (n_states, n_states) boolean mask restricting
    where the Dirichlet prior mass is spread from each state -- alpha_support[s]
    is the set of successor states an unvisited (s, a) pair is allowed to put
    prior mass on. None (default) reproduces the original behaviour exactly:
    alpha spread uniformly over ALL states, which is what every offline
    result was produced with and must keep producing bit-for-bit. Used by
    the online experiment (see online/agent.py), where a prior uniform over
    all states would assert that an untried action can teleport anywhere on
    the grid -- including into a trap -- which biases planning against
    exploration before any data has been collected.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        alpha_support: np.ndarray | None = None,
        r_default: float = 0.0,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.r_default = r_default

        if alpha_support is not None:
            alpha_support = np.asarray(alpha_support, dtype=bool)
            if alpha_support.shape != (n_states, n_states):
                raise ValueError(
                    f"alpha_support must have shape ({n_states}, {n_states}), got {alpha_support.shape}"
                )
            if not alpha_support.any(axis=1).all():
                raise ValueError("alpha_support has a state with empty support (every row needs >=1 True)")
        self.alpha_support = alpha_support

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
        never visited yields a distribution over states equal to the prior
        alone, rather than a division by zero -- uniform over all states by
        default, or restricted to alpha_support[s] if one was given."""
        if self.alpha_support is None:
            counts = self.N + self.alpha
            return counts / counts.sum(axis=2, keepdims=True)

        prior = self.alpha_support[:, None, :] * self.alpha  # (S, 1, S) broadcasts over A
        counts = self.N + prior
        T = counts / counts.sum(axis=2, keepdims=True)

        # explicit support restriction, asserted every call: cheap at this
        # problem size, and this is precisely the invariant that must never
        # silently break (see class docstring) -- an unvisited (s, a) pair
        # must carry zero belief mass outside alpha_support[s].
        unvisited = self.N.sum(axis=2) == 0  # (S, A)
        support_per_sa = np.broadcast_to(self.alpha_support[:, None, :], T.shape)
        mass_outside_support = np.where(support_per_sa, 0.0, T).sum(axis=2)  # (S, A)
        assert np.allclose(mass_outside_support[unvisited], 0.0), (
            "alpha_support violated: an unvisited (s, a) pair has belief mass "
            "outside its declared support"
        )
        return T

    @property
    def R_hat(self) -> np.ndarray:
        """(S,), reward on arrival. Falls back to r_default wherever the
        state was never visited (0.0 by default)."""
        R = np.full(self.n_states, self.r_default, dtype=np.float64)
        visited = self.R_count > 0
        R[visited] = self.R_sum[visited] / self.R_count[visited]
        return R

    @property
    def visited_mask(self) -> np.ndarray:
        """(S, A) boolean: (s,a) pairs observed at least once."""
        return self.N.sum(axis=2) > 0

    def transition_distribution(self, s: int, a: int) -> np.ndarray:
        """Agent.transition_distribution: a single row of T_hat."""
        return self.T_hat[s, a]

    def plan(self, terminal: np.ndarray, gamma: float = GAMMA) -> None:
        """Value iteration on (T_hat, R_hat). Stores V, Q, policy."""
        P = self.T_hat
        R = self.R_hat
        self.V, self.policy = value_iteration(P, R, terminal, gamma=gamma)
        self.Q = np.einsum("sap,p->sa", P, R + gamma * self.V * (~terminal))

    def save(self, path: str) -> None:
        """Saves the raw counts (N, R_sum, R_count), alpha, r_default and
        alpha_support -- enough to reconstruct T_hat/R_hat identically. V/Q/
        policy are not saved: they are derived quantities, recomputable by
        plan() without recollecting any data (not a retraining, just a
        replanning). alpha_support is omitted when None, so files written by
        the offline pipeline (alpha_support=None, r_default=0.0 always) stay
        byte-for-byte the same as before this class was extended."""
        kwargs = dict(
            N=self.N,
            R_sum=self.R_sum,
            R_count=self.R_count,
            alpha=np.array(self.alpha),
            n_states=np.array(self.n_states),
            n_actions=np.array(self.n_actions),
        )
        if self.r_default != 0.0:
            kwargs["r_default"] = np.array(self.r_default)
        if self.alpha_support is not None:
            kwargs["alpha_support"] = self.alpha_support
        np.savez(path, **kwargs)

    @classmethod
    def load(cls, path: str) -> "TabularModelBasedAgent":
        """Reloads an agent from the counts saved by save(). Does not
        recompute V/Q/policy: call plan(terminal) after loading. Tolerant of
        files saved before alpha_support/r_default existed (falls back to
        their defaults, matching the original offline behaviour)."""
        data = np.load(path)
        alpha_support = data["alpha_support"] if "alpha_support" in data.files else None
        r_default = float(data["r_default"]) if "r_default" in data.files else 0.0
        agent = cls(
            int(data["n_states"]),
            int(data["n_actions"]),
            alpha=float(data["alpha"]),
            alpha_support=alpha_support,
            r_default=r_default,
        )
        agent.N = data["N"]
        agent.R_sum = data["R_sum"]
        agent.R_count = data["R_count"]
        return agent
