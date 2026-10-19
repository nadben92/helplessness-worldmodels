"""Position/appearance diagnostics (spec section 8): alpha_app (D1) and
the Jensen-Shannon latent-distance diagnostic (D2). Pure numpy,
array-in/array-out -- takes canonical-probe outputs (already computed by
experiments/E3_worldmodel_4zones/probe.py) as plain arrays.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def alpha_app(I_z_swap: float, I_L: float, I_Lprime: float, contrast_threshold: float) -> float | None:
    """alpha_app(s) = (I_z^swap(s) - I_L) / (I_L' - I_L), spec section 8.
    Returns None (undefined) when |I_L' - I_L| <= contrast_threshold --
    "l'indice n'est defini que pour les paires... avec un contraste de
    controlabilite", never a division that happens to not crash on a
    near-zero denominator."""
    contrast = I_Lprime - I_L
    if abs(contrast) <= contrast_threshold:
        return None
    return (I_z_swap - I_L) / contrast


def jensen_shannon_categorical(p: np.ndarray, q: np.ndarray) -> float:
    """Standard Jensen-Shannon divergence between two categorical
    distributions (last axis is the class axis), in bits: JS(p,q) =
    0.5*KL(p||m) + 0.5*KL(q||m), m = (p+q)/2. Bounded in [0, 1] bit for
    two-outcome-equivalent normalisation... actually bounded in [0, 1]
    in NATS-normalised JS; here (log2-based) it is bounded in [0, 1] bit
    when p, q are point masses on disjoint categories (the maximal
    case), 0 when p == q."""
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m), axis=-1)
    kl_qm = np.sum(q * np.log2(q / m), axis=-1)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    # JS divergence is mathematically >= 0 always; clip away the sliver
    # of floating-point noise (~1e-8) that can otherwise appear when p
    # and q are nearly identical, rather than let it leak into callers
    # as a spurious tiny negative.
    return np.maximum(0.0, js)


def jensen_shannon_product(p_probs: np.ndarray, q_probs: np.ndarray) -> float:
    """p_probs, q_probs: (n_vars, n_classes) -- the canonical probe's
    per-variable categorical posterior probabilities at two cells.
    Returns the SUM, over the n_vars independent variables, of each
    variable's Jensen-Shannon divergence -- the natural product-
    distribution decomposition (each variable treated as its own
    categorical channel), used for D2's neighbour-distance comparison
    (spec section 8). Not a claim that the FULL joint-distribution JS
    divergence has this closed form in general -- it is the tractable,
    documented choice for this per-variable-independent latent."""
    per_var = jensen_shannon_categorical(p_probs, q_probs)  # (n_vars,)
    return float(np.sum(per_var))


def mean_pairwise_jensen_shannon(product_probs: np.ndarray) -> float:
    """Mean over all pairs of distinct items of jensen_shannon_product --
    product_probs: (n_items, n_vars, n_classes), e.g. one posterior per
    state. A posterior-collapse diagnostic: ~0 when every state gets the
    same posterior, n_vars bits when every pair has disjoint point-mass
    codes on every variable."""
    n = product_probs.shape[0]
    if n < 2:
        raise ValueError("need at least two items")
    # (n, n, n_vars) per-variable JS for every pair, via broadcasting
    per_var = jensen_shannon_categorical(product_probs[:, None, :, :], product_probs[None, :, :, :])
    pairwise = per_var.sum(axis=-1)
    off_diagonal = ~np.eye(n, dtype=bool)
    return float(pairwise[off_diagonal].mean())


def mean_random_pair_jensen_shannon(product_probs: np.ndarray, rng: np.random.Generator, n_pairs: int = 64) -> float:
    """Label-free separation statistic: mean jensen_shannon_product over
    n_pairs pairs of DISTINCT items drawn uniformly at random from
    product_probs (n_items, n_vars, n_classes) -- e.g. the posteriors of
    the images in the current training batch. Takes no state label (two
    random images almost always show different states), so it can drive
    a training-time decision without the label ever reaching the
    training loop (docs/e3/DECISIONS.md, 2026-09-24)."""
    n = product_probs.shape[0]
    if n < 2:
        raise ValueError("need at least two items")
    i = rng.integers(0, n, size=n_pairs)
    j = (i + rng.integers(1, n, size=n_pairs)) % n  # uniform over j != i
    per_var = jensen_shannon_categorical(product_probs[i], product_probs[j])  # (n_pairs, n_vars)
    return float(per_var.sum(axis=-1).mean())


def mean_within_group_pair_jensen_shannon(product_probs: np.ndarray, groups: np.ndarray, rng: np.random.Generator,
                                          n_pairs: int = 64) -> float:
    """mean_random_pair_jensen_shannon restricted to pairs of DISTINCT items of the SAME group (E5, docs/e5/DECISIONS.md
    (17); an option, off by default): groups (n_items,) labels the items whose images share their background (computed
    from the images, no state label), so that the statistic measures the separation of the agent's states, not of the
    backgrounds. i uniform among the items that have a partner, j uniform among i's partners. NaN when no item has one.
    With ONE group (a single background: E3, E4, E5's baseline and immunisation, a shuffled batch that happens to hold one
    assignment), every pair is a same-group pair and the result is mean_random_pair_jensen_shannon's, the same draws
    (bit for bit: tested)."""
    groups = np.asarray(groups)
    n = product_probs.shape[0]
    if groups.shape != (n,):
        raise ValueError(f"groups must have shape ({n},), got {groups.shape}")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if counts.size == 1:  # one background: every pair is a same-background pair -- E3/E4's statistic, its very draws
        return mean_random_pair_jensen_shannon(product_probs, rng, n_pairs)
    eligible = np.flatnonzero(counts[inverse] >= 2)
    if eligible.size == 0:
        return float("nan")
    i = eligible[rng.integers(0, eligible.size, size=n_pairs)]
    j = np.empty_like(i)
    for t, item in enumerate(i):
        partners = np.flatnonzero((inverse == inverse[item]) & (np.arange(n) != item))
        j[t] = partners[rng.integers(0, partners.size)]
    per_var = jensen_shannon_categorical(product_probs[i], product_probs[j])
    return float(per_var.sum(axis=-1).mean())


def group_pair_counts(groups: np.ndarray) -> dict[str, int]:
    """The items that have a same-group partner (n_eligible), the groups and the groups of two items or more, of a batch's
    groups (mean_within_group_pair_jensen_shannon's pool)."""
    _, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    return {"n_items": int(inverse.size), "n_eligible": int((counts[inverse] >= 2).sum()), "n_groups": int(counts.size),
            "n_groups_with_pairs": int((counts >= 2).sum())}


# ---------------------------------------------------------------------
# D1 acceptance, uncertainty and anomalies (docs/e3/DECISIONS.md, 2026-09-25)
# ---------------------------------------------------------------------


def d1_contrast(I_L: float, se_L: float, I_Lprime: float, se_Lprime: float, min_bits: float = 0.25, n_se: float = 3.0) -> dict:
    """Controllability contrast between zone L (own) and zone L' (target):
    accepted only if |I_L' - I_L| > max(min_bits, n_se * combined se),
    combined se = sqrt(se_L^2 + se_L'^2) (Monte-Carlo standard errors of
    the two zone means). Below it, alpha_app is not defined for the pair."""
    contrast = float(I_Lprime - I_L)
    combined_se = float(np.sqrt(se_L**2 + se_Lprime**2))
    threshold = float(max(min_bits, n_se * combined_se))
    return {"contrast": contrast, "combined_se": combined_se, "threshold": threshold, "accepted": abs(contrast) > threshold}


def alpha_app_with_se(
    I_swap: np.ndarray, se_swap: np.ndarray, I_L: float, se_L: float, I_Lprime: float, se_Lprime: float
) -> tuple[np.ndarray, np.ndarray]:
    """alpha = (I_swap - I_L) / (I_L' - I_L) per state, with its standard
    error by first-order error propagation over the three MC estimates
    (independent draws):
      d alpha / d I_swap = 1 / c,  d alpha / d I_L = (I_swap - I_L') / c^2,
      d alpha / d I_L' = -(I_swap - I_L) / c^2,  c = I_L' - I_L.
    Only call it on an accepted contrast (d1_contrast): c is then far from 0."""
    I_swap, se_swap = np.asarray(I_swap, dtype=np.float64), np.asarray(se_swap, dtype=np.float64)
    c = I_Lprime - I_L
    alpha = (I_swap - I_L) / c
    se = np.sqrt((se_swap / c) ** 2 + (se_L * (I_swap - I_Lprime) / c**2) ** 2 + (se_Lprime * (I_swap - I_L) / c**2) ** 2)
    return alpha, se


def summarize_alpha_app(alpha: np.ndarray, se: np.ndarray, n_se: float = 3.0) -> dict:
    """Per-zone summary of per-state alpha_app values. A value outside
    [0, 1] by more than n_se of its own standard error is an ANOMALY: it is
    listed (index, value, se) and never averaged. The others are summarised
    by their mean over states with an interval mean +/- n_se * sd / sqrt(n)
    (the spread over the zone's states, not the MC error); None when fewer
    than 2 values remain."""
    alpha, se = np.asarray(alpha, dtype=np.float64), np.asarray(se, dtype=np.float64)
    anomalous = (alpha < -n_se * se) | (alpha > 1.0 + n_se * se)
    kept = alpha[~anomalous]
    out = {
        "n_states": int(alpha.size),
        "n_anomalies": int(anomalous.sum()),
        "anomalies": [{"index": int(i), "alpha": float(alpha[i]), "se": float(se[i])} for i in np.flatnonzero(anomalous)],
        "n_kept": int(kept.size),
        "mean": float(kept.mean()) if kept.size else None,
        "ci_low": None,
        "ci_high": None,
    }
    if kept.size >= 2:
        half = n_se * float(kept.std(ddof=1)) / np.sqrt(kept.size)
        out["ci_low"], out["ci_high"] = out["mean"] - half, out["mean"] + half
    return out
