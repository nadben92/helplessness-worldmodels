"""The online training loop: policy determines the data it collects, full
value iteration replanned periodically. Budget is counted in transitions,
never episodes (episode length varies with p and with policy quality, so
counting episodes would confound data volume with performance).

Three separate RNGs are threaded through explicitly and must never be
merged: explore_rng (epsilon-greedy action choice) and tiebreak_rng (argmax
ties at each replan) are exploration-side randomness; the environment's own
dynamics RNG (env.seed_dynamics, untouched here) is kept fully independent so
that noise and exploration can be varied one at a time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from helplessness.agents.collect import collect
from helplessness.agents.online.agent import OnlineTabularAgent


def train_online(
    env,
    agent: OnlineTabularAgent,
    n_steps: int,
    epsilon: float,
    replan_every: int,
    explore_rng: np.random.Generator,
    tiebreak_rng: np.random.Generator,
    learning: bool = True,
    on_transition: Callable[[int, int, int, float, int, bool, bool], None] | None = None,
) -> int:
    """Runs exactly n_steps transitions, resetting episodes as they end.

    Returns the final state (so a caller can chain another train_online call
    on the same env without losing position, though every phase of this
    protocol resets between phases anyway).

    on_transition(t, s, a, r, s_next, terminated, truncated), if given, is
    called once per transition after the model update and before the replan
    check -- the hook a caller uses for periodic out-of-loop evaluation
    (Step 4) without this loop needing to know about evaluation at all.
    """
    s, _ = env.reset()
    agent.plan(tiebreak_rng)

    for t in range(1, n_steps + 1):
        if explore_rng.random() < epsilon:
            a = int(explore_rng.integers(agent.n_actions))
        else:
            a = int(agent.policy[s])

        s_next, r, terminated, truncated, _info = env.step(a)

        if learning:
            # Truncation is not termination: never marks terminal[s_next],
            # which would fabricate an absorbing state in the middle of the
            # grid and bias every subsequent plan() (see CLAUDE.md).
            agent.observe(s, a, r, s_next, terminated)

        if on_transition is not None:
            on_transition(t, s, a, r, s_next, terminated, truncated)

        if terminated or truncated:
            s, _ = env.reset()
        else:
            s = s_next

        # Replan strictly after this step's count update (learning branch
        # above already ran), so the agent never acts on a policy that is
        # stale by a variable amount.
        if t % replan_every == 0:
            agent.plan(tiebreak_rng)

    return s


def _uniform_policy_over(n_actions: int):
    """A collect()-compatible policy_fn(s, rng) -> action, uniform over
    n_actions. Not helplessness.agents.collect.uniform_policy: that one
    hardcodes 4 actions (the offline environment never had "stay"), and
    collect.py is offline-owned code this experiment must not edit."""
    def policy_fn(s: int, rng: np.random.Generator) -> int:
        return int(rng.integers(n_actions))
    return policy_fn


def train_online_with_burnin(
    env,
    agent: OnlineTabularAgent,
    n_steps: int,
    epsilon: float,
    replan_every: int,
    burn_in_steps: int,
    explore_rng: np.random.Generator,
    tiebreak_rng: np.random.Generator,
    burn_in_rng: np.random.Generator,
    learning: bool = True,
    on_transition: Callable[[int, int, int, float, int, bool, bool], None] | None = None,
) -> int:
    """Phase-1 training with a burn-in preamble: burn_in_steps transitions
    under a uniform random policy over all n_actions -- reusing
    helplessness.agents.collect.collect (imported, not copy-pasted) exactly
    as the offline pipeline does -- before switching to the epsilon-greedy,
    periodically-replanned loop for the remainder of the budget.

    Why: at the task's specified defaults (epsilon=0.1), a fresh agent can,
    ~40% of the time in this layout, permanently fail to sample the one
    untried action that would disconfirm an over-generalised pessimism near
    a trap (both routes to the goal pass a trap-adjacent cell, and the
    restricted-support prior cannot distinguish a safe untried action from
    an unsafe one at that cell until it is actually tried) -- an artefact of
    undirected low-epsilon exploration, not a property of p_train. The
    burn-in guarantees (s, a) coverage before policy-driven exploration
    begins, and is applied identically to every p_train condition in Phase
    1, so it introduces no asymmetry between them.

    burn_in_steps counts against n_steps, never on top of it -- the total
    transition budget is unchanged from a plain train_online() call (see
    CLAUDE.md: budgets are counted in transitions and must stay comparable
    across conditions).

    Deliberately NOT used for Phase 2 (recovery): the same deadlock there is
    part of what is being measured, and papering over it would remove the
    signal (see online/README.md). burn_in_steps=0 makes this call exactly
    equivalent to train_online().
    """
    if not 0 <= burn_in_steps <= n_steps:
        raise ValueError(f"burn_in_steps ({burn_in_steps}) must be in [0, n_steps={n_steps}]")

    if burn_in_steps > 0 and learning:
        policy_fn = _uniform_policy_over(agent.n_actions)
        data = collect(env, policy_fn, burn_in_steps, burn_in_rng)
        for s, a, r, s_next, terminated in data:
            agent.observe(s, a, r, s_next, terminated)

    return train_online(
        env, agent, n_steps - burn_in_steps, epsilon, replan_every,
        explore_rng, tiebreak_rng, learning=learning, on_transition=on_transition,
    )
