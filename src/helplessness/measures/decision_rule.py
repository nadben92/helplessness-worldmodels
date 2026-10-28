"""What a decision rule makes of a Q row (E6/E7's diagnostics, docs/e6/DECISIONS.md (52)): which actions are directed,
the gap between the best action and the second, the share of directed actions in the greedy tie set, the softmax
reading of Q, and the binning of a quantity along b. Arrays in, arrays out (numpy); nothing here knows a model.

  directed_action_mask   (n_states, A) bool: the action's successor is strictly closer to the goal (E6's directed step,
                         measures.zone_phenotypes.directed_steps, read per action instead of per transition).
  best_second_gap        max Q - the second largest Q, per row.
  tie_set                the actions within `tol` of the row's max (agents.worldmodel.policy.tie_set's rule, numpy).
  greedy_directed_share  per row, the share of directed actions in the tie set: the probability that the training
                         policy's greedy branch (random tie-break in the tie set) picks a directed action.
  softmax_probs          softmax(Q / tau) per row (tau > 0), max-shifted.
  binned                 per bin of b: the count, the median and the mean of a quantity (NaN where empty).
"""

from __future__ import annotations

import numpy as np


def directed_action_mask(succ: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(n_states, A) bool: succ (n_states, A) the successors, dist (n_states,) the distance to the goal."""
    d = np.asarray(dist)
    return d[np.asarray(succ)] < d[:, None]


def best_second_gap(Q: np.ndarray) -> np.ndarray:
    """(...,): the largest minus the second largest value of the last axis."""
    top = np.sort(np.asarray(Q, dtype=float), axis=-1)
    return top[..., -1] - top[..., -2]


def tie_set(Q: np.ndarray, tol: float) -> np.ndarray:
    """(..., A) bool: Q >= max Q - tol (the training's tie set, absolute tolerance)."""
    Q = np.asarray(Q, dtype=float)
    return Q >= Q.max(axis=-1, keepdims=True) - float(tol)


def greedy_directed_share(Q: np.ndarray, directed: np.ndarray, tol: float) -> np.ndarray:
    """(...,): the share of directed actions in each row's tie set. directed: (..., A) bool, the row's directed actions."""
    ties = tie_set(Q, tol)
    return (ties & np.asarray(directed, dtype=bool)).sum(axis=-1) / ties.sum(axis=-1)


def argmax_directed(Q: np.ndarray, directed: np.ndarray) -> np.ndarray:
    """(...,) bool: the strict argmax (first maximum) is a directed action."""
    a = np.argmax(np.asarray(Q, dtype=float), axis=-1)
    return np.take_along_axis(np.asarray(directed, dtype=bool), a[..., None], axis=-1)[..., 0]


def softmax_probs(Q: np.ndarray, tau: float) -> np.ndarray:
    """(..., A): softmax(Q / tau) over the last axis, tau > 0."""
    if not tau > 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    x = np.asarray(Q, dtype=float) / float(tau)
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def softmax_choice(Q: np.ndarray, tau: float, u: float) -> int:
    """The action of a softmax(Q / tau) draw by the inverse CDF of the uniform u in [0, 1). Q: (A,)."""
    cdf = np.cumsum(softmax_probs(Q, tau))
    return int(min(np.searchsorted(cdf, u * cdf[-1], side="right"), len(cdf) - 1))


def binned(b: np.ndarray, values: np.ndarray, edges: np.ndarray) -> dict[str, np.ndarray]:
    """Per bin [edges[i], edges[i + 1]) of b: n, the median and the mean of `values` (finite pairs only)."""
    b, values = np.asarray(b, dtype=float), np.asarray(values, dtype=float)
    keep = np.isfinite(b) & np.isfinite(values)
    b, values = b[keep], values[keep]
    edges = np.asarray(edges, dtype=float)
    k = len(edges) - 1
    n, med, mean = np.zeros(k, dtype=np.int64), np.full(k, np.nan), np.full(k, np.nan)
    which = np.digitize(b, edges) - 1
    for i in range(k):
        v = values[which == i]
        n[i] = v.size
        if v.size:
            med[i], mean[i] = np.median(v), v.mean()
    return {"lo": edges[:-1], "hi": edges[1:], "n": n, "median": med, "mean": mean}
