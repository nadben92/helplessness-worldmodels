"""E2's own training loop -- NOT helplessness.agents.online.loop.train_online,
reused as-is (see docs/E2_online_4zones.md's Temps-1 audit: that loop was
written for the 5x5 prototype and fails two of the spec's structural
requirements without a heavy rewrite):

  - start states: train_online's env.reset() takes no argument at each
    episode boundary, always returning to env.start_position (fixed) --
    E2 requires a uniform draw over non-terminal states from a DEDICATED
    RNG at every reset.
  - replanning: train_online replans on a step-count interval
    (t % replan_every == 0) -- E2 requires a replan at the END OF EVERY
    EPISODE (plus the one initial planning on the prior), never on a
    fixed step cadence.

Everything else (no burn-in, terminated/truncated never conflated, the
reset transition never counted in N) already held and is preserved here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def train_online_e2(
    env,
    agent,
    n_steps: int,
    epsilon: float,
    non_terminal_states: np.ndarray,
    explore_rng: np.random.Generator,
    tiebreak_rng: np.random.Generator,
    start_rng: np.random.Generator,
    on_transition: Callable[[int, int, int, float, int, bool, bool], None] | None = None,
) -> int:
    """Runs exactly n_steps environment transitions (never counting a
    reset as a step, per the spec). Draws start states uniformly among
    non_terminal_states from start_rng, at the very first reset and at
    every subsequent episode boundary (terminated OR truncated).
    Replans once on the prior before the first step, then again at the
    end of every episode (never on a step-count interval).

    on_transition(t, s, a, r, s_next, terminated, truncated), if given, is
    called once per transition, after the model update -- the hook an
    orchestration script uses for checkpointing (a copy of agent.N at
    logarithmically-spaced t) and trajectory logging, without this loop
    needing to know about either.
    """

    def _draw_start() -> int:
        return int(non_terminal_states[start_rng.integers(len(non_terminal_states))])

    s, _ = env.reset(start_state=_draw_start())
    agent.plan(tiebreak_rng)  # initial planning on the prior, before any data

    for t in range(1, n_steps + 1):
        if explore_rng.random() < epsilon:
            a = int(explore_rng.integers(agent.n_actions))
        else:
            a = int(agent.policy[s])

        s_next, r, terminated, truncated, _info = env.step(a)
        agent.observe(s, a, r, s_next, terminated)

        if on_transition is not None:
            on_transition(t, s, a, r, s_next, terminated, truncated)

        if terminated or truncated:
            agent.plan(tiebreak_rng)  # replan at the end of every episode
            s, _ = env.reset(start_state=_draw_start())
        else:
            s = s_next

    return s
