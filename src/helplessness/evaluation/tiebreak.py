"""Tie-break policy extraction from an already-computed Q."""

from __future__ import annotations

import numpy as np


def uniform_tie_policy_probs(Q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """(S, A) row-stochastic: uniform distribution over the actions tied
    (up to tol) with the row max, zero elsewhere. A deterministic
    counterpart of argmax_random_tie(): instead of drawing one action per
    state, returns the full mixture so a stochastic policy can be
    evaluated exactly (planning.value_iteration.policy_evaluation_stochastic)
    rather than by Monte Carlo sampling of the tie-break itself.

    Used for E2's Delta(s) = V*(s) - V^pi_hat(s): under an online agent's
    Dirichlet prior, many (s, a) pairs are never visited and carry exactly
    the same Q under the prior alone -- a plain np.argmax (E1's convention,
    reused verbatim by planning.value_iteration.value_iteration) would
    always pick the lowest action index, structuring Delta by action order
    rather than by anything the agent believes. Reduces to a one-hot row
    exactly where Q has a unique max within tol (see
    tests/evaluation/test_tiebreak.py's regression pin against
    argmax_random_tie's own tie set).
    """
    Q = np.asarray(Q)
    n_states, n_actions = Q.shape
    row_max = Q.max(axis=1, keepdims=True)
    tied = Q >= (row_max - tol)
    counts = tied.sum(axis=1, keepdims=True)
    return tied.astype(np.float64) / counts


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
