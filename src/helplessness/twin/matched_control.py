"""Matched-counts collection: generalises
experiments/E1_offline_4zones/estimation.py::collect_generative (exactly
n_per_pair draws for EVERY (s, a) pair, a single scalar) to a per-pair
target count N_target[s, a] -- E2's two matched-counts conditions both
need this:

  - the matched control: N_target = the online run's own final N[s,a],
    drawn from the EXPERIMENTAL kernel P_exp (isolates the effect of
    heterogeneous visitation from the effect of adaptive sampling -- see
    docs/E2_online_4zones.md).
  - the matched reference: same N_target, drawn from the paired-control
    kernel P_ref (never the analytic value -- see module docstring on
    Ihat_ref below). The SAME realisation is reused for both the online
    and matched-control conditions of a given seed, by design (the
    online-vs-matched comparison then isolates Ihat alone, per spec).

Not a modification of E1's collect_generative (frozen, scalar-only, still
used unchanged by E1's own pipeline) -- a new, more general function.

Moved from experiments/E2_online_4zones/matched_control.py on 2026-10-14 (a
re-export shim stays there); body unchanged.
"""

from __future__ import annotations

import numpy as np


def collect_matched(P: np.ndarray, N_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exactly N_target[s, a] draws from P[s, a], for every (s, a) pair --
    (S, A, S) int64 counts, vectorized per (s,a) via a single multinomial
    draw (same rationale as collect_generative: exact, not an
    approximation, and avoids a per-sample Python loop).

    N_target: (S, A) int array (typically an online run's own N.sum(axis=2)
    -- read as counts, not concentrations: N_target[s,a]==0 draws nothing,
    correctly leaving that pair's row at all zero counts, matching the
    online run's own unvisited pairs exactly).
    """
    n_states, n_actions, _ = P.shape
    N_target = np.asarray(N_target)
    if N_target.shape != (n_states, n_actions):
        raise ValueError(f"N_target must have shape ({n_states}, {n_actions}), got {N_target.shape}")

    counts = np.zeros((n_states, n_actions, n_states), dtype=np.int64)
    for s in range(n_states):
        for a in range(n_actions):
            n = int(N_target[s, a])
            if n > 0:
                counts[s, a] = rng.multinomial(n, P[s, a])
    return counts
