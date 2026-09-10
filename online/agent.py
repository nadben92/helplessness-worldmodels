"""Online model-based agent: model construction plus a periodically-replanned
policy. Not Dyna-Q -- no sampled backups on imagined transitions, only a
complete value iteration on the current (T_hat, R_hat) every replan (see
online/loop.py for the main loop, and CLAUDE.md for why full VI is affordable
and preferable here).

Wraps helplessness.agents.tabular.TabularModelBasedAgent (imported, not
copy-pasted) for the counts/T_hat/R_hat estimator, using the alpha_support
argument added to that class for exactly this experiment (see its docstring):
the online agent's Dirichlet prior must be spread only over geographically
reachable successors, never uniformly over the whole grid, otherwise an
untried action looks like it can teleport into a trap and value iteration
kills exploration before any data exists.

terminal[s] is learned from experience (starts all False, set True the first
time a transition into s is observed to terminate the episode) -- never read
from env.terminal_mask(). The agent must discover which cells are absorbing
the same way it discovers everything else about the world.
"""

from __future__ import annotations

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.evaluation.tiebreak import argmax_random_tie
from helplessness.measures.belief import perceived_controllability
from helplessness.planning.value_iteration import GAMMA, value_iteration


def geographic_support(env) -> np.ndarray:
    """(n_states, n_states) boolean mask: support[s, s'] is True iff s' is
    "geographically reachable" from s -- one of the 4 neighbours, or s
    itself (self-loop from a wall/border bump, or the "stay" action).

    Derived from env.transition_tensor(p=1.0): at p=1 the executed action is
    drawn uniformly over all n_actions regardless of what was chosen, so
    every action row is identical and its support is exactly the set of
    cells reachable by attempting any single action from s -- precisely "the
    4 neighbours plus self", with no assumption baked in here beyond the
    Env interface (transition_tensor), so this works for any grid layout
    without hardcoding wall positions.

    This queries only the p=1 structural row, never reward or terminal
    information -- a prior about the shape of the state space, not about
    where the goal or the traps are.
    """
    P1 = env.transition_tensor(p=1.0)  # (S, A, S)
    row0 = P1[:, 0, :] > 0
    for a in range(1, P1.shape[1]):
        assert np.array_equal(row0, P1[:, a, :] > 0), (
            "geographic_support assumes all action rows are identical at p=1 "
            "(the defining property of GridWorld's stochasticity model)"
        )
    return row0


class OnlineTabularAgent:
    """Owns the model estimator (via TabularModelBasedAgent) and the
    periodically-replanned greedy policy.

    use_eta_update: optional alternative to Dirichlet counts -- a constant
    step-size model update T[s,a,:] += eta * (onehot(s') - T[s,a,:]), seeded
    at the same restricted-support prior as the count estimator. Implemented
    for completeness (see CLAUDE.md online task, step 2) but OFF by default
    and not swept in this phase -- counts are the estimator actually used.
    """

    def __init__(
        self,
        env,
        alpha: float = 0.1,
        r_default: float = 0.0,
        gamma: float = GAMMA,
        use_eta_update: bool = False,
        eta: float = 0.1,
    ):
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.gamma = gamma
        self.support = geographic_support(env)
        self.terminal = np.zeros(self.n_states, dtype=bool)  # learned online

        self._model = TabularModelBasedAgent(
            self.n_states, self.n_actions, alpha=alpha,
            alpha_support=self.support, r_default=r_default,
        )

        self.use_eta_update = use_eta_update
        self.eta = eta
        self._T_eta = self._model.T_hat.copy() if use_eta_update else None

        self.V: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.policy: np.ndarray | None = None

    def observe(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> None:
        self._model.observe(s, a, r, s_next)
        if terminated:
            self.terminal[s_next] = True
        if self.use_eta_update:
            onehot = np.zeros(self.n_states)
            onehot[s_next] = 1.0
            self._T_eta[s, a] += self.eta * (onehot - self._T_eta[s, a])

    @property
    def N(self) -> np.ndarray:
        return self._model.N

    @property
    def T_hat(self) -> np.ndarray:
        return self._T_eta if self.use_eta_update else self._model.T_hat

    @property
    def R_hat(self) -> np.ndarray:
        return self._model.R_hat

    @property
    def visited_mask(self) -> np.ndarray:
        return self._model.visited_mask

    def plan(self, tiebreak_rng: np.random.Generator) -> None:
        """Full value iteration on the current belief. The policy is
        extracted with a random tie-break (never np.argmax's lowest-index
        default), so that states with equivalent actions don't produce a
        deterministic bias that would contaminate cycle-rate metrics."""
        T, R = self.T_hat, self.R_hat
        self.V, _ = value_iteration(T, R, self.terminal, gamma=self.gamma)
        self.Q = np.einsum("sap,p->sa", T, R + self.gamma * self.V * (~self.terminal))
        self.policy = argmax_random_tie(self.Q, tiebreak_rng)

    def rescale_counts(self, n_cap: float, tol: float = 0.35) -> None:
        """Globally rescales N so that N.sum() == n_cap, as a single
        multiplicative factor applied to the whole array -- never per (s, a)
        pair, which would erase the visitation imbalance that is precisely
        the trace of the behavioural spiral this experiment wants to keep
        (see CLAUDE.md online task, step 3). Counts become non-integer
        Dirichlet pseudo-counts; expected -- alpha itself is already
        fractional.

        Only the model's confidence should change, not its shape: asserted
        via perceived_controllability (I(a;s'|s) under T_hat), which must
        stay within tol bits of its pre-rescale value. This is an empirical
        approximation, not an algebraic identity: alpha (fixed absolute
        strength) becomes relatively more or less influential per (s, a)
        pair depending how far below n_cap its own count was, so some shift
        is expected. tol=0.35 was calibrated by measuring the actual shift
        on real 50k-transition Phase-1 snapshots (n_cap=5000) across the
        full p_train range: ~0.03 bits at p_train in {0.85, 0.95, 1.0}, but
        ~0.29 bits at p_train=0.0 -- a p=0 agent solves the task efficiently
        and therefore samples many (s, a) pairs only a handful of times, so
        a 10x downscale pushes those specific pairs' counts down near
        alpha's own fixed strength. Flagged here rather than silently
        tightened: "unchanged within tolerance" is looser for p_train=0
        than for the high-p conditions, which the full sweep's belief-metric
        comparisons should keep in mind. At n_cap == the collected budget
        (e.g. the 5k Phase-1 condition), the factor is 1.0 and the shift is
        exactly zero.
        """
        before = perceived_controllability(self.T_hat, self.terminal)
        total = self._model.N.sum()
        if total > 0:
            self._model.N = self._model.N * (n_cap / total)
        after = perceived_controllability(self.T_hat, self.terminal)
        assert abs(after - before) < tol, (
            f"rescale_counts moved perceived controllability by {abs(after - before):.4f} bits "
            f"(before={before:.4f}, after={after:.4f}) -- rescaling must change confidence, not belief"
        )

    def save(self, path: str) -> None:
        """Serialises everything needed to reconstruct this agent exactly
        (counts, learned terminal set, geographic support, hyperparameters)
        -- V/Q/policy are derived, recomputed by plan() after loading, not
        saved (same convention as TabularModelBasedAgent.save)."""
        np.savez(
            path,
            N=self._model.N,
            R_sum=self._model.R_sum,
            R_count=self._model.R_count,
            terminal=self.terminal,
            support=self.support,
            alpha=np.array(self._model.alpha),
            r_default=np.array(self._model.r_default),
            gamma=np.array(self.gamma),
            n_states=np.array(self.n_states),
            n_actions=np.array(self.n_actions),
            use_eta_update=np.array(self.use_eta_update),
            eta=np.array(self.eta),
            T_eta=(self._T_eta if self.use_eta_update else np.zeros((0,))),
        )

    @classmethod
    def load(cls, path: str) -> "OnlineTabularAgent":
        """Reloads an agent from a save() snapshot -- no env needed (support
        and terminal are restored directly, not recomputed), so a snapshot
        alone is enough to regenerate every downstream result without
        retraining (see CONTRIBUTING.md, results and configs). Call plan()
        after loading to (re)populate V/Q/policy."""
        data = np.load(path)
        agent = cls.__new__(cls)
        agent.n_states = int(data["n_states"])
        agent.n_actions = int(data["n_actions"])
        agent.gamma = float(data["gamma"])
        agent.support = data["support"]
        agent.terminal = data["terminal"]
        agent.use_eta_update = bool(data["use_eta_update"])
        agent.eta = float(data["eta"])

        agent._model = TabularModelBasedAgent(
            agent.n_states, agent.n_actions, alpha=float(data["alpha"]),
            alpha_support=agent.support, r_default=float(data["r_default"]),
        )
        agent._model.N = data["N"]
        agent._model.R_sum = data["R_sum"]
        agent._model.R_count = data["R_count"]
        agent._T_eta = data["T_eta"].copy() if agent.use_eta_update else None

        agent.V = agent.Q = agent.policy = None
        return agent
