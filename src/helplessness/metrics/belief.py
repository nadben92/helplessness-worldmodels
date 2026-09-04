"""Belief metrics -- operate on an agent's learned transition belief T_hat
(and the raw counts derived from it), never on the environment or agent
classes themselves. See module docstring pattern in controllability.py.
"""

from __future__ import annotations

import numpy as np

from helplessness.metrics.controllability import controllability_from_tensor


def perceived_controllability(T_hat: np.ndarray, terminal: np.ndarray) -> float:
    """Average I(a ; s' | s) in bits under T_hat. Wraps controllability_from_tensor."""
    return controllability_from_tensor(T_hat, terminal)


def coverage(visited_mask: np.ndarray, terminal: np.ndarray) -> float:
    """Fraction of (s,a) pairs visited among those actually reachable.

    A terminal state (walls included) can never serve as an acting state:
    either the agent cannot enter it (wall), or the episode ends before the
    next step (goal/trap). These pairs must therefore not count in the
    denominator, otherwise the ceiling of the fraction depends on the number
    of terminal states in the layout and is no longer comparable across
    layouts (see roadmap step 2, where the layout varies). Here, 1.0 always
    means complete coverage of what is actually observable.
    """
    reachable = ~terminal
    if not reachable.any():
        return float("nan")
    return float(visited_mask[reachable].mean())


def model_error(
    T_hat: np.ndarray, T_true: np.ndarray, visited_mask: np.ndarray
) -> tuple[float, float]:
    """Average total-variation distance per (s,a) pair.

    Returns (tv_visited, tv_all): the first restricted to the (s,a) pairs
    actually observed (visited_mask, shape (S,A)), the second over all
    pairs -- including those never visited, where T_hat is only the prior
    and the error is structurally large without being informative about the
    quality of learning.
    """
    tv = 0.5 * np.abs(T_hat - T_true).sum(axis=2)  # (S, A)
    tv_all = float(tv.mean())
    tv_visited = float(tv[visited_mask].mean()) if visited_mask.any() else float("nan")
    return tv_visited, tv_all
