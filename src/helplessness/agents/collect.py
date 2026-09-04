"""Trajectory collection for the offline protocol.

The collection policy is injected (never hard-coded) so that the same
machinery can serve the online protocol later.
"""

from __future__ import annotations

import numpy as np


def uniform_policy(s: int, rng: np.random.Generator) -> int:
    """Default collection policy: uniform action, independent of s."""
    return int(rng.integers(4))


def collect(env, policy_fn, n_transitions: int, rng: np.random.Generator) -> list[tuple]:
    """Collects exactly n_transitions, resetting at the end of each episode.

    policy_fn(s, rng) -> action. rng is used only for the choice of actions;
    the dynamics' noise comes from env's own generator (seeded separately by
    the caller via env.seed_dynamics), never from this rng. A reset does not
    consume a transition: the budget is counted in transitions, never in
    episodes.
    """
    transitions = []
    s, _ = env.reset()
    while len(transitions) < n_transitions:
        a = policy_fn(s, rng)
        s_next, r, terminated, truncated, _ = env.step(a)
        transitions.append((s, a, r, s_next, terminated))
        if terminated or truncated:
            s, _ = env.reset()
        else:
            s = s_next
    return transitions
