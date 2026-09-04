"""Metrics on an arbitrary transition tensor -- neither depends on GridWorld
nor on any agent class, so they can be applied both to the exact tensor and
to a T_hat learned by an agent (and, later, to a neural world model's
learned transition belief). Array in, scalar/array out, no side effects.
"""

from __future__ import annotations

import numpy as np


def controllability_from_tensor(
    P: np.ndarray, terminal: np.ndarray, action_prior: np.ndarray | None = None
) -> float:
    """Average I(a ; s' | s) in bits over non-terminal states.

    P[s, a, s'] : transition tensor (exact or learned). terminal[s] : boolean
    mask of states to exclude from the average. action_prior : law p(a) used
    to weight the actions (uniform by default).
    """
    n_states, n_actions, _ = P.shape
    if action_prior is None:
        action_prior = np.full(n_actions, 1.0 / n_actions)
    action_prior = np.asarray(action_prior, dtype=np.float64)

    vals = []
    for s in range(n_states):
        if terminal[s]:
            continue
        joint = P[s] * action_prior[:, None]  # p(a, s') = p(a) * P(s'|s,a)
        marg = joint.sum(axis=0)  # p(s')
        denom = action_prior[:, None] * marg[None, :]
        nz = joint > 0
        mi = np.sum(joint[nz] * np.log2(joint[nz] / denom[nz]))
        vals.append(mi)
    return float(np.mean(vals)) if vals else 0.0


def value_gap(Q: np.ndarray) -> np.ndarray:
    """max_a Q(s,a) - mean_a Q(s,a), per state."""
    return Q.max(axis=1) - Q.mean(axis=1)


def value_gap_second(Q: np.ndarray) -> np.ndarray:
    """max_a Q(s,a) - second max_a Q(s,a), per state.

    Unlike value_gap (vs the mean), this is the gap to the closest
    competitor: near 0 signals a near-tie between the two best actions
    (arbitrary tie-break of the argmax), regardless of what the remaining
    actions are worth.
    """
    top2 = np.sort(Q, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]
