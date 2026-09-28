"""Per-state and per-zone summaries of a measure taken in several real
contexts of the same state (docs/e3/DECISIONS.md, 2026-09-28: the main E3
measure is computed in the real contexts (h_t, z_t) of the replay, not
from the learned initial state h0 alone).

Pure numpy, array-in/array-out (CLAUDE.md's measures/ rule). The
per-context values themselves (I_z, KL, ...) are computed elsewhere, by
the one implementation of each quantity; this module only aggregates.
"""

from __future__ import annotations

import numpy as np


def per_state_context_summary(values: np.ndarray, states: np.ndarray, n_states: int, mc_se: np.ndarray | None = None) -> dict:
    """values (n,): one estimate per context; states (n,): the true state
    of each context. Returns, per state (arrays of length n_states):
      n     number of contexts (0 = not measured);
      mean  mean over the state's contexts (NaN when n = 0);
      sd    standard deviation over them, ddof=1 (NaN when n < 2);
      se    standard error of the mean: sd / sqrt(n) when n >= 2 -- the
            spread over contexts already contains each estimate's own
            Monte-Carlo noise -- and the single context's MC standard
            error (mc_se) when n = 1 (NaN if mc_se is not given); NaN
            when n = 0.
    NaN values (a context where the quantity does not exist) are left
    out of their state's count."""
    values = np.asarray(values, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    mc_se = None if mc_se is None else np.asarray(mc_se, dtype=np.float64)
    n = np.zeros(n_states, dtype=np.int64)
    mean = np.full(n_states, np.nan)
    sd = np.full(n_states, np.nan)
    se = np.full(n_states, np.nan)
    valid = ~np.isnan(values)
    for s in range(n_states):
        idx = np.flatnonzero((states == s) & valid)
        n[s] = idx.size
        if idx.size == 0:
            continue
        mean[s] = values[idx].mean()
        if idx.size >= 2:
            sd[s] = values[idx].std(ddof=1)
            se[s] = sd[s] / np.sqrt(idx.size)
        elif mc_se is not None:
            se[s] = mc_se[idx[0]]
    return {"n": n, "mean": mean, "sd": sd, "se": se}


def zone_mean_with_se(mean: np.ndarray, se: np.ndarray, members: np.ndarray | list[int]) -> dict:
    """Mean of per-state values over a zone's MEASURED states (NaN states
    are left out) and its standard error sqrt(sum se_s^2) / n_measured --
    the zone's states are a fixed set, not a sample, so only the per-state
    errors count. Returns mean, se, n_measured, n_states; mean and se are
    NaN when no state of the zone is measured."""
    members = np.asarray(members, dtype=np.int64)
    m, e = np.asarray(mean, dtype=np.float64)[members], np.asarray(se, dtype=np.float64)[members]
    measured = ~np.isnan(m)
    k = int(measured.sum())
    if k == 0:
        return {"mean": float("nan"), "se": float("nan"), "n_measured": 0, "n_states": int(members.size)}
    e = np.where(np.isnan(e[measured]), 0.0, e[measured])
    return {"mean": float(m[measured].mean()), "se": float(np.sqrt(np.sum(e**2)) / k), "n_measured": k,
            "n_states": int(members.size)}
