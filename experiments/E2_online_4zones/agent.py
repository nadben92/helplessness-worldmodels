"""OnlineTabularAgentKnownReward: E2's online agent.

Deliberately NOT helplessness.agents.online.agent.OnlineTabularAgent
(reused, not modified -- see docs/E2_online_4zones.md and CLAUDE.md's
Agent interface section): that class learns R_hat AND terminal[s] from
experience, which is exactly backwards for E2's spec ("l'agent planifie
avec le vrai R et la vraie semantique de terminaison -- seules les
transitions sont apprises"). This class plans directly on the TRUE reward
vector and TRUE terminal mask, passed in at construction, and only ever
learns T_hat (via the same counts/alpha_support machinery
TabularModelBasedAgent already provides -- reused, not reimplemented).
"""

from __future__ import annotations

import numpy as np

from helplessness.agents.base import Agent
from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.evaluation.tiebreak import argmax_random_tie
from helplessness.planning.value_iteration import GAMMA, value_iteration


class OnlineTabularAgentKnownReward(Agent):
    """Owns the T-only counts estimator (TabularModelBasedAgent, used
    purely for N/T_hat -- its own R_hat/R_sum/R_count bookkeeping is
    populated by observe() for convenience but never read) and the
    periodically-replanned greedy policy, planned against the TRUE reward
    and TRUE terminal mask.

    support: (n_states, n_states) bool, the Dirichlet alpha_support mask
    (see helplessness.envs.zones.geographic_support_mask) -- required, no
    default, since E2's spec forbids a flat prior over the whole grid
    (CLAUDE.md online task: an untried action must not look like it can
    teleport anywhere).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        R_true: np.ndarray,
        terminal_true: np.ndarray,
        support: np.ndarray,
        alpha: float,
        gamma: float = GAMMA,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.R_true = np.asarray(R_true, dtype=np.float64)
        self.terminal_true = np.asarray(terminal_true, dtype=bool)
        self.support = support
        self.gamma = gamma

        self._model = TabularModelBasedAgent(n_states, n_actions, alpha=alpha, alpha_support=support)

        self.V: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.policy: np.ndarray | None = None  # collection policy: argmax_random_tie, see plan()
        self.last_plan_n_iter: int | None = None  # value_iteration's iteration count at the last plan() call

    def observe(self, s: int, a: int, r: float, s_next: int, terminated: bool) -> None:
        """Records the transition for T_hat's counts. r/terminated are
        NOT used to update any belief (both are already known exactly,
        see class docstring) -- only sanity-checked against what the
        agent already believes true, so an env/agent mismatch fails
        loudly here rather than silently biasing planning."""
        expected_r = self.R_true[s_next]
        assert abs(r - expected_r) < 1e-9, (
            f"observed reward {r} at s_next={s_next} != known R_true[s_next]={expected_r} "
            "-- env and agent disagree on the (supposedly known) reward"
        )
        expected_terminated = bool(self.terminal_true[s_next])
        assert terminated == expected_terminated, (
            f"observed terminated={terminated} at s_next={s_next} != known terminal_true[s_next]="
            f"{expected_terminated} -- env and agent disagree on the (supposedly known) terminal set"
        )
        self._model.observe(s, a, r, s_next)

    @property
    def N(self) -> np.ndarray:
        return self._model.N

    @property
    def T_hat(self) -> np.ndarray:
        return self._model.T_hat

    @property
    def visited_mask(self) -> np.ndarray:
        return self._model.visited_mask

    def transition_distribution(self, s: int, a: int) -> np.ndarray:
        """Agent.transition_distribution: a single row of T_hat."""
        return self.T_hat[s, a]

    def plan(self, tiebreak_rng: np.random.Generator) -> None:
        """Full value iteration on (T_hat, R_true, terminal_true) -- never
        R_hat, never a learned terminal set. The COLLECTION policy is
        extracted with a random tie-break (argmax_random_tie): under the
        prior alone, every action at a state ties exactly, and
        np.argmax's lowest-index default would funnel every exploitation
        step onto action 0 before any data exists.

        This is the ACTING policy only -- Delta's evaluation policy uses
        a different convention (uniform_tie_policy_probs +
        policy_evaluation_stochastic, see measurement.py), computed
        separately from this method's Q, never from self.policy.

        Warm-started from self.V (the previous plan's converged V, per
        spec: "demarree depuis le V precedent") -- self.V is None only on
        the very first call (planning on the prior), in which case
        value_iteration falls back to its own zero-init. Warm-starting is
        exact, not an approximation: value iteration is a contraction
        mapping with a unique fixed point regardless of V_init, so the
        CONVERGED V is unaffected -- only n_iter (self.last_plan_n_iter,
        kept for the pilot's reporting requirement) drops once T_hat has
        only changed by a handful of counts since the last replan."""
        T = self.T_hat
        self.V, _, self.last_plan_n_iter = value_iteration(
            T, self.R_true, self.terminal_true, gamma=self.gamma, V_init=self.V, return_n_iter=True,
        )
        self.Q = np.einsum("sap,p->sa", T, self.R_true + self.gamma * self.V * (~self.terminal_true))
        self.policy = argmax_random_tie(self.Q, tiebreak_rng)

    def save(self, path: str) -> None:
        """Only N (plus hyperparameters/support/R_true/terminal_true) is
        saved -- V/Q/policy are derived, recomputed by plan() after
        loading. Matches the "jumeau tabulaire" design: everything
        downstream is reconstructible from N alone (see twin.py)."""
        np.savez(
            path,
            N=self._model.N,
            R_true=self.R_true,
            terminal_true=self.terminal_true,
            support=self.support,
            alpha=np.array(self._model.alpha),
            gamma=np.array(self.gamma),
            n_states=np.array(self.n_states),
            n_actions=np.array(self.n_actions),
        )

    @classmethod
    def load(cls, path: str) -> "OnlineTabularAgentKnownReward":
        data = np.load(path)
        agent = cls(
            n_states=int(data["n_states"]),
            n_actions=int(data["n_actions"]),
            R_true=data["R_true"],
            terminal_true=data["terminal_true"],
            support=data["support"],
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
        )
        agent._model.N = data["N"]
        agent.V = agent.Q = agent.policy = None
        return agent
