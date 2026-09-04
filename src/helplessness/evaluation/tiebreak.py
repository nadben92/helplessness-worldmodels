"""Tie-break policy extraction from an already-computed Q."""

from __future__ import annotations

import numpy as np


def argmax_random_tie(Q: np.ndarray, rng: np.random.Generator, tol: float = 1e-9) -> np.ndarray:
    """argmax with a random pick among ties (up to tolerance), per state.

    np.argmax (used internally by value_iteration) always returns the lowest
    index in case of a tie -- harmless when values genuinely differ, but
    when several actions are tied (or nearly so, cf. value_gap_second near
    0), this indexing bias becomes an artefact: it systematically favours
    one direction (here "up", index 0) over another equally valid choice.
    Does not affect value_iteration itself; serves to extract an alternative
    evaluation policy from an already-computed Q.
    """
    Q = np.asarray(Q)
    row_max = Q.max(axis=1, keepdims=True)
    tied = Q >= (row_max - tol)
    policy = np.empty(Q.shape[0], dtype=int)
    for s in range(Q.shape[0]):
        candidates = np.flatnonzero(tied[s])
        policy[s] = candidates[rng.integers(len(candidates))]
    return policy
