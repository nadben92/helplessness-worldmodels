"""Kernel-in-s-space probe (spec section 5.3, secondary measure): decode
K samples of z' ~ prior(h_1_a), match each decoded image to the nearest
of the 144 canonical state templates by pixel L2 distance, classify a
draw as a "ghost" when the margin between the best and second-best
template distance is below a threshold, and report a kernel p_hat(s'|s,a)
renormalised over the NON-ghost draws only (never silently folding ghost
mass into an existing state, and never silently dropping it without
reporting the rate -- spec: "la masse des fantomes n'est pas
renormalisee en silence").

Pure numpy, array-in/array-out: takes already-decoded images and
templates (plain arrays), never a model or an Env/Agent class. The
decoding itself (a torch forward pass through the RSSM's decoder) lives
in experiments/E3_worldmodel_4zones/probe.py, outside measures/ (see
docs/e3/DECISIONS.md, adjustment B).
"""

from __future__ import annotations

import numpy as np


def match_templates(decoded_images: np.ndarray, templates: np.ndarray, margin_threshold: float) -> dict[str, np.ndarray]:
    """decoded_images: (K, ...) -- K decoded samples for one (s, a)
    pair, any per-image shape (flattened internally). templates: (n_states,
    ...), same per-image shape. margin_threshold: minimum (second-best -
    best) squared-L2 distance gap required to accept the nearest match;
    below it, the draw is classified a ghost.

    Returns nearest_state (K,) int, is_ghost (K,) bool, margin (K,)
    float (second-best minus best distance, always >= 0)."""
    K = decoded_images.shape[0]
    n_states = templates.shape[0]
    flat_dec = decoded_images.reshape(K, -1).astype(np.float64)
    flat_tmpl = templates.reshape(n_states, -1).astype(np.float64)

    # (K, n_states) squared L2 distances via |a|^2 + |b|^2 - 2 a.b: memory
    # O(K * n_states). The direct (K, n_states, D) broadcast was 2.6 GB per
    # call at 72x72x3 with 144 templates (docs/e3/DECISIONS.md, 2026-09-24).
    dists = (flat_dec**2).sum(axis=1)[:, None] + (flat_tmpl**2).sum(axis=1)[None, :] - 2.0 * (flat_dec @ flat_tmpl.T)
    dists = np.maximum(dists, 0.0)  # clip the expansion's float rounding below zero
    order = np.argsort(dists, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if n_states > 1 else order[:, 0]
    best_d = np.take_along_axis(dists, best_idx[:, None], axis=1)[:, 0]
    second_d = np.take_along_axis(dists, second_idx[:, None], axis=1)[:, 0]
    margin = second_d - best_d
    is_ghost = margin < margin_threshold

    return {"nearest_state": best_idx, "is_ghost": is_ghost, "margin": margin}


def kernel_from_matches(nearest_state: np.ndarray, is_ghost: np.ndarray, n_states: int) -> dict[str, np.ndarray | float | int]:
    """Aggregates match_templates()'s per-draw output into a renormalised
    kernel p_hat(s'|.) over the n_states valid states -- renormalised
    over the NON-ghost draws only, with the ghost rate reported
    separately (never silently absorbed).

    Returns: p_hat (n_states,), ghost_fraction (float in [0, 1], NaN if
    K == 0), n_ghosts (int), n_samples (int)."""
    K = nearest_state.shape[0]
    non_ghost = ~is_ghost
    n_non_ghost = int(non_ghost.sum())

    p_hat = np.zeros(n_states, dtype=np.float64)
    if n_non_ghost > 0:
        counts = np.bincount(nearest_state[non_ghost], minlength=n_states)[:n_states]
        p_hat = counts / n_non_ghost

    ghost_fraction = float("nan") if K == 0 else (K - n_non_ghost) / K
    return {
        "p_hat": p_hat,
        "ghost_fraction": ghost_fraction,
        "n_ghosts": K - n_non_ghost,
        "n_samples": K,
    }


def kernel_probe(
    decoded_images: np.ndarray, templates: np.ndarray, margin_threshold: float
) -> dict[str, np.ndarray | float | int]:
    """One-call convenience: match_templates() + kernel_from_matches()."""
    matched = match_templates(decoded_images, templates, margin_threshold)
    n_states = templates.shape[0]
    result = kernel_from_matches(matched["nearest_state"], matched["is_ghost"], n_states)
    result["nearest_state"] = matched["nearest_state"]
    result["is_ghost"] = matched["is_ghost"]
    result["margin"] = matched["margin"]
    return result


def expected_l1_sampling_floor(p: np.ndarray, K: int) -> np.ndarray:
    """E[ L1(p_emp_K, p) ]: the expected L1 error of the empirical law of K
    exact draws from p -- what a kernel estimated from K decoded samples
    scores against the truth even when the model IS the truth (docs/e3/
    DECISIONS.md, 2026-09-29: C-b's sampling floor). Exact, not simulated:
    the expectation is linear, so it is the sum over outcomes of the
    binomial mean absolute deviation E|k_i/K - p_i|, k_i ~ Bin(K, p_i).
    p: (..., n_outcomes); returns the floor for every leading index."""
    from scipy.stats import binom

    p = np.asarray(p, dtype=np.float64)
    k = np.arange(K + 1)
    pmf = binom.pmf(k[:, None], K, p.reshape(-1)[None, :])  # (K+1, n)
    mad = (pmf * np.abs(k[:, None] / K - p.reshape(-1)[None, :])).sum(axis=0)
    return mad.reshape(p.shape).sum(axis=-1)


def draws_for_l1_floor(P: np.ndarray, target: float, k_max: int = 100_000) -> int:
    """The smallest number of draws K such that, for every row of P
    (..., n_outcomes) -- i.e. in the noisiest kernel it contains -- the
    expected L1 sampling floor is <= target at K and at every K' in
    [K, 2K] (the floor is not exactly monotone in K on the binomial
    lattice). Doubling search, then bisection, then the lattice check."""
    rows = np.unique(np.asarray(P, dtype=np.float64).reshape(-1, np.shape(P)[-1]).round(12), axis=0)

    def worst(K: int) -> float:
        return float(expected_l1_sampling_floor(rows, K).max())

    hi = 1
    while worst(hi) > target:
        hi *= 2
        if hi > k_max:
            raise ValueError(f"no K <= {k_max} brings the floor to {target}")
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if worst(mid) <= target else (mid, hi)
    K = hi
    while any(worst(k) > target for k in range(K, 2 * K + 1)):
        K += 1
    return K
