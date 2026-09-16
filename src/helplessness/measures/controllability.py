"""Metrics on an arbitrary transition tensor -- neither depends on GridWorld
nor on any agent class, so they can be applied both to the exact tensor and
to a T_hat learned by an agent (and, later, to a neural world model's
learned transition belief). Array in, scalar/array out, no side effects.
"""

from __future__ import annotations

import numpy as np


def controllability_per_state(P: np.ndarray, action_prior: np.ndarray | None = None) -> np.ndarray:
    """I(a ; s' | s) in bits, one value per state -- the primitive every
    per-zone quantity in E1 aggregates by indexing with a zone mask, rather
    than building per-zone sub-tensors. controllability_from_tensor() below
    is exactly this array's terminal-masked mean; nothing about the
    per-state formula changed when it was extracted from that function (see
    tests/measures/test_controllability.py's regression pin).

    P[s, a, s'] : transition tensor (exact or learned). action_prior : law
    p(a) used to weight the actions (uniform by default).
    """
    n_states, n_actions, _ = P.shape
    if action_prior is None:
        action_prior = np.full(n_actions, 1.0 / n_actions)
    action_prior = np.asarray(action_prior, dtype=np.float64)

    vals = np.empty(n_states, dtype=np.float64)
    for s in range(n_states):
        joint = P[s] * action_prior[:, None]  # p(a, s') = p(a) * P(s'|s,a)
        marg = joint.sum(axis=0)  # p(s')
        denom = action_prior[:, None] * marg[None, :]
        nz = joint > 0
        vals[s] = np.sum(joint[nz] * np.log2(joint[nz] / denom[nz]))
    return vals


def controllability_from_tensor(
    P: np.ndarray, terminal: np.ndarray, action_prior: np.ndarray | None = None
) -> float:
    """Average I(a ; s' | s) in bits over non-terminal states -- the
    terminal-masked mean of controllability_per_state().

    P[s, a, s'] : transition tensor (exact or learned). terminal[s] : boolean
    mask of states to exclude from the average. action_prior : law p(a) used
    to weight the actions (uniform by default).
    """
    per_state = controllability_per_state(P, action_prior)
    mask = ~np.asarray(terminal, dtype=bool)
    return float(per_state[mask].mean()) if mask.any() else 0.0


def conditional_entropy_per_state(P: np.ndarray, action_prior: np.ndarray | None = None) -> np.ndarray:
    """H(s' | s, a) in bits, one value per state: the action-prior-weighted
    average, over actions, of the entropy of P(.|s,a). Same
    per-state/scalar-mean split as controllability_per_state(), for the
    same reason -- every E1 quantity is per-zone, i.e. per state subset.
    """
    n_states, n_actions, _ = P.shape
    if action_prior is None:
        action_prior = np.full(n_actions, 1.0 / n_actions)
    action_prior = np.asarray(action_prior, dtype=np.float64)

    vals = np.empty(n_states, dtype=np.float64)
    for s in range(n_states):
        row = P[s]  # (n_actions, n_states)
        h_per_action = np.empty(n_actions, dtype=np.float64)
        for a in range(n_actions):
            p_a = row[a]
            nz = p_a > 0
            h_per_action[a] = -np.sum(p_a[nz] * np.log2(p_a[nz]))
        vals[s] = np.sum(action_prior * h_per_action)
    return vals


def conditional_entropy_from_tensor(
    P: np.ndarray, terminal: np.ndarray, action_prior: np.ndarray | None = None
) -> float:
    """Average H(s' | s, a) in bits over non-terminal states -- the
    terminal-masked mean of conditional_entropy_per_state()."""
    per_state = conditional_entropy_per_state(P, action_prior)
    mask = ~np.asarray(terminal, dtype=bool)
    return float(per_state[mask].mean()) if mask.any() else 0.0


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
