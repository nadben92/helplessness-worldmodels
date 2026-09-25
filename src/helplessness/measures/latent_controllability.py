"""I_z(s), the principal E3 measure (spec section 5.2): I(a; z' | s) in
bits, computed directly in the RSSM's categorical latent space, under a
uniform reference over actions -- never estimated by counting z samples
(spec's own explicit prohibition: "Interdit: estimer une entropie en
espace z par comptage d'echantillons").

Pure numpy, array-in/array-out (CLAUDE.md's measures/ rule) -- takes
ALREADY-COMPUTED prior probability arrays (e.g. from
experiments/E3_worldmodel_4zones/probe.py's canonical probe, itself
torch-dependent and living outside measures/), never a model or torch
tensor.

H(z'|s,a) is exact (a product-of-independent-categoricals distribution
per action: entropy decomposes exactly as a sum over the n_vars
variables). H(z'|s) (the uniform-action MIXTURE over those 5 product
distributions) has no closed form in general, so it is estimated by
Monte Carlo exactly as spec 5.2 describes: draw a uniform action, sample
z from that action's distribution, evaluate log m(z) = logsumexp_a log
p_a(z) - log(n_actions) via the exact per-action log-densities (never by
counting how often a given z recurs).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def per_variable_entropy(probs: np.ndarray) -> np.ndarray:
    """probs: (..., n_vars, n_classes), a probability simplex per
    variable. Returns (...,): exact entropy in bits, summed over the
    n_vars independent variables (valid because each (..., n_vars, :)
    slice is a PRODUCT distribution over the n_vars variables)."""
    p = np.clip(probs, _EPS, 1.0)
    h_per_var = -(p * np.log2(p)).sum(axis=-1)
    return h_per_var.sum(axis=-1)


def conditional_entropy_z_per_action(prior_probs_per_action: np.ndarray) -> np.ndarray:
    """prior_probs_per_action: (n_actions, n_vars, n_classes). Returns
    (n_actions,): exact H(z'|s,a) for each action."""
    return per_variable_entropy(prior_probs_per_action)


def conditional_entropy_z_mean(prior_probs_per_action: np.ndarray, action_prior: np.ndarray | None = None) -> float:
    """mean_a H(z'|s,a) under action_prior (uniform by default) -- the
    second term of I_z(s)."""
    h = conditional_entropy_z_per_action(prior_probs_per_action)
    n_actions = h.shape[0]
    if action_prior is None:
        action_prior = np.full(n_actions, 1.0 / n_actions)
    return float(np.sum(np.asarray(action_prior) * h))


def _logsumexp2(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log2(np.sum(2.0 ** (x - m), axis=axis, keepdims=True))).squeeze(axis)


def marginal_entropy_z_mc(
    prior_probs_per_action: np.ndarray,
    M: int,
    rng: np.random.Generator,
    action_prior: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """H(z'|s) (the mixture over the uniform-action reference),
    estimated via M Monte Carlo samples exactly as spec 5.2 describes:
    draw a ~ action_prior, z ~ p_a, then log m(z) = logsumexp_a log
    p_a(z) - log(n_actions), using the EXACT per-action log-density
    (never a frequency count of z). Returns H_hat, its MC standard
    error, and the raw per-sample log_m values (for downstream
    diagnostics, e.g. checking se ~ 1/sqrt(M))."""
    n_actions, n_vars, n_classes = prior_probs_per_action.shape
    if action_prior is None:
        action_prior = np.full(n_actions, 1.0 / n_actions)
    action_prior = np.asarray(action_prior, dtype=np.float64)

    probs = np.clip(prior_probs_per_action, _EPS, 1.0)
    log_probs_all = np.log2(probs)  # (n_actions, n_vars, n_classes)

    actions = rng.choice(n_actions, size=M, p=action_prior)
    z = np.empty((M, n_vars), dtype=np.int64)
    for v in range(n_vars):
        cdf = probs[actions, v].cumsum(axis=-1)  # (M, n_classes)
        u = rng.random(M)
        z[:, v] = np.clip((u[:, None] > cdf).sum(axis=-1), 0, n_classes - 1)

    # log p_{a'}(z_i) for every a', for every sample i: (n_actions, M)
    per_var_logp = np.take_along_axis(
        log_probs_all[:, None, :, :],  # (n_actions, 1, n_vars, n_classes)
        z[None, :, :, None],  # (1, M, n_vars, 1)
        axis=-1,
    )[..., 0]  # (n_actions, M, n_vars)
    log_p_all_actions = per_var_logp.sum(axis=-1)  # (n_actions, M)

    log_weighted = log_p_all_actions + np.log2(action_prior)[:, None]
    log_m = _logsumexp2(log_weighted, axis=0)  # (M,)

    H_hat = float(-log_m.mean())
    se = float(log_m.std(ddof=1) / np.sqrt(M)) if M > 1 else float("nan")
    return {"H_hat": H_hat, "se": se, "log_m": log_m}


def mixture_information(
    component_probs: np.ndarray,
    M: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
    bounds_tol_se_multiplier: float = 6.0,
) -> dict[str, float]:
    """The one core behind every latent mutual information in this
    module: I(c; z) = H(sum_c w_c p_c) - sum_c w_c H(p_c), where each
    component p_c (first axis of component_probs, shape (n_components,
    n_vars, n_classes)) is a product of categoricals. The component
    entropies are exact; the mixture entropy is estimated by Monte Carlo
    with exact densities (marginal_entropy_z_mc). Asserts 0 <= I <=
    log2(n_components) within MC tolerance -- a violation beyond it is a
    bug or too small an M, never silently accepted."""
    n_components = component_probs.shape[0]
    H_cond = conditional_entropy_z_mean(component_probs, weights)
    marg = marginal_entropy_z_mc(component_probs, M, rng, weights)
    information = marg["H_hat"] - H_cond

    ceiling = np.log2(n_components)
    tol = bounds_tol_se_multiplier * marg["se"] + 1e-9
    if not (-tol <= information <= ceiling + tol):
        raise AssertionError(
            f"I={information:.6f} outside [0, log2({n_components})={ceiling:.6f}] beyond MC "
            f"tolerance {tol:.6f} (se={marg['se']:.6f}) -- likely a real bug, not MC noise"
        )
    return {"I": information, "H_marginal": marg["H_hat"], "H_conditional_mean": H_cond, "se": marg["se"]}


def latent_controllability(
    prior_probs_per_action: np.ndarray,
    M: int,
    rng: np.random.Generator,
    action_prior: np.ndarray | None = None,
    bounds_tol_se_multiplier: float = 6.0,
) -> dict[str, float]:
    """I_z(s) = H(z'|s) - mean_a H(z'|s,a), spec 5.2: mixture_information
    with the actions as components (uniform reference over actions)."""
    result = mixture_information(prior_probs_per_action, M, rng, action_prior, bounds_tol_se_multiplier)
    return {
        "I_z": result["I"],
        "H_marginal": result["H_marginal"],
        "H_conditional_mean": result["H_conditional_mean"],
        "se": result["se"],
    }


def latent_state_information(
    posterior_probs_per_state: np.ndarray,
    M: int,
    rng: np.random.Generator,
    state_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """I(z; s) in bits, a posterior-collapse diagnostic:
    mixture_information with the STATES as components (one posterior per
    state, e.g. the canonical probe's q(z | h0, enc(render(s)))), uniform
    over states by default. log2(n_states) (7.17 bits for 144 states)
    means z identifies the state exactly; ~0 means the posterior ignores
    the observation (collapse)."""
    result = mixture_information(posterior_probs_per_state, M, rng, state_weights)
    return {
        "I_zs": result["I"],
        "H_marginal": result["H_marginal"],
        "H_conditional_mean": result["H_conditional_mean"],
        "se": result["se"],
    }
