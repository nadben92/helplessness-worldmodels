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


def template_distances(decoded_images: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """(K, n_states) squared L2 distances between K decoded images and the
    n_states templates (any per-image shape, flattened)."""
    K = decoded_images.shape[0]
    n_states = templates.shape[0]
    flat_dec = decoded_images.reshape(K, -1).astype(np.float64)
    flat_tmpl = templates.reshape(n_states, -1).astype(np.float64)

    # (K, n_states) squared L2 distances via |a|^2 + |b|^2 - 2 a.b: memory
    # O(K * n_states). The direct (K, n_states, D) broadcast was 2.6 GB per
    # call at 72x72x3 with 144 templates (docs/e3/DECISIONS.md, 2026-09-24).
    dists = (flat_dec**2).sum(axis=1)[:, None] + (flat_tmpl**2).sum(axis=1)[None, :] - 2.0 * (flat_dec @ flat_tmpl.T)
    return np.maximum(dists, 0.0)  # clip the expansion's float rounding below zero


def match_from_distances(dists: np.ndarray, margin_threshold: float, allowed: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """THE matching rule, on (K, n_states) distances: the nearest template,
    and a ghost when the margin (second-best minus best distance) is below
    margin_threshold. allowed: (K, n_states) bool -- the templates each
    draw may be matched to (the projection on a support of
    docs/e3/DECISIONS.md, 2026-09-30 (5): the nearest and the second-best are
    taken among the allowed templates only); None = all of them. A draw
    with a single allowed template has an infinite margin (never a ghost);
    one with none is a ghost.

    Returns nearest_state (K,) int, is_ghost (K,) bool, margin (K,) float."""
    n_states = dists.shape[1]
    if allowed is not None:
        dists = np.where(allowed, dists, np.inf)
    order = np.argsort(dists, axis=1)
    best_idx = order[:, 0]
    second_idx = order[:, 1] if n_states > 1 else order[:, 0]
    best_d = np.take_along_axis(dists, best_idx[:, None], axis=1)[:, 0]
    second_d = np.take_along_axis(dists, second_idx[:, None], axis=1)[:, 0]
    with np.errstate(invalid="ignore"):
        margin = np.where(np.isfinite(best_d), second_d - best_d, -np.inf)  # no allowed template: a ghost
    is_ghost = margin < margin_threshold
    return {"nearest_state": best_idx, "is_ghost": is_ghost, "margin": margin}


def match_templates(decoded_images: np.ndarray, templates: np.ndarray, margin_threshold: float) -> dict[str, np.ndarray]:
    """decoded_images: (K, ...) -- K decoded samples for one (s, a)
    pair, any per-image shape (flattened internally). templates: (n_states,
    ...), same per-image shape. margin_threshold: minimum (second-best -
    best) squared-L2 distance gap required to accept the nearest match;
    below it, the draw is classified a ghost.

    Returns nearest_state (K,) int, is_ghost (K,) bool, margin (K,)
    float (second-best minus best distance, always >= 0)."""
    return match_from_distances(template_distances(decoded_images, templates), margin_threshold)


def off_support_mass(p_hat: np.ndarray, support: np.ndarray) -> np.ndarray:
    """The share of a kernel's mass on states outside the reachable
    successors (docs/e3/DECISIONS.md, 2026-09-30 (5)): p_hat (n_states,
    n_actions, n_states), support (n_states, n_states) bool (state s's
    reachable successors, e.g. envs.zones.geographic_support_mask -- the
    support of the tabular twin's prior). Returns (n_states, n_actions);
    NaN where the row is NaN (state not measured)."""
    outside = np.where(support[:, None, :], 0.0, p_hat).sum(axis=-1)
    return np.where(np.isfinite(p_hat).all(axis=-1), outside, np.nan)


def action_independent_weight(p_hat: np.ndarray, P: np.ndarray) -> np.ndarray:
    """The weight alpha of an action-independent component in a predicted kernel (docs/e3/DECISIONS.md, 2026-10-07 (29)):
    per state s, the least-squares fit of p_hat(.|s, a) = (1 - alpha) P(.|s, a) + alpha m(.), m one distribution
    shared by every action. The shared part drops out of the action contrasts d_a = p_hat_a - mean_a p_hat and
    D_a = P_a - mean_a P, so 1 - alpha = sum_a <d_a, D_a> / sum_a |D_a|^2 (the predicted contrast projected on the true
    one). alpha = w exactly for p_hat = (1 - w) P + w m, whatever m; 0 for p_hat = P; not clipped (< 0: contrasts
    sharper than P's; > 1: reversed). p_hat, P: (n_states, n_actions, n_states). Returns (n_states,); NaN where a row
    of p_hat is NaN (state not measured) or where P has no action contrast (an uncontrollable state: nothing to lose)."""
    d = p_hat - p_hat.mean(axis=1, keepdims=True)
    D = P - P.mean(axis=1, keepdims=True)
    denom = (D ** 2).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = 1.0 - (d * D).sum(axis=(1, 2)) / denom
    ok = np.isfinite(p_hat).all(axis=(1, 2)) & (denom > 1e-12)
    return np.where(ok, alpha, np.nan)


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
