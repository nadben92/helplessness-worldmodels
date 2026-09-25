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
