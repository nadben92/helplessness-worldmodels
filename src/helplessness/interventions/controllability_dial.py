"""Causal intervention on the transition belief T_hat -- a tabular rehearsal
of the latent manipulation planned for roadmap step 3. The protocol and the
figures produced here (experiments/03_intervention/) are the ones reused
as-is in the neural phase.

Mathematical justification (verified empirically in
tests/interventions/test_controllability_dial.py, never merely assumed): the
environment's transition tensor decomposes exactly as

    T(p)[s,a,:] = (1-p) * T_det[s,a,:] + p * M[s,:]

where T_det = T(0) is the deterministic dynamics and M[s,:] = mean_a T_det[s,a,:]
is the action-independent marginal -- the cell one lands on when moving in a
uniformly random direction. M is invariant in p: mean_a T(p)[s,a,:] = M[s,:]
for every p. The direction of "p" in tensor space is therefore known in
closed form: no probe, regression, or learned decoder is needed to
manipulate it.

Array in, array out: does not import any environment or agent class, so it
transfers unchanged to a learned belief from any future world model.
"""

from __future__ import annotations

import numpy as np


def marginalise(T: np.ndarray) -> np.ndarray:
    """M[s,:] = average over actions. Returns (S, 1, S) for broadcasting."""
    return T.mean(axis=1, keepdims=True)


def controllability_dial(T: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """T_lam = (1-lam)*T + lam*M. lam=0 intact, lam=1 zero controllability,
    lam<0 extrapolates towards more control.

    Returns (T_lam, clipped_fraction).

    For lam in [0,1], T_lam is a convex combination of valid distributions:
    the simplex constraints (sum to 1, coefficients >= 0) hold automatically
    -- checked via assert rather than assumed. For lam < 0 (or lam > 1), the
    extrapolation can produce negative entries; they are then clipped to 0
    and each (s,a) row is renormalised. clipped_fraction reports the
    fraction of (s,a) rows that needed a clip -- an intervention where most
    rows are clipped is no longer a clean manipulation of controllability
    and must never be silently reported as such.

    Touches neither R_hat nor the terminal mask: only the transition belief
    changes, that is the whole point of the intervention.
    """
    M = marginalise(T)
    T_lam = (1.0 - lam) * T + lam * M

    if 0.0 <= lam <= 1.0:
        assert np.all(T_lam >= -1e-9), "convex combination expected to be nonnegative"
        assert np.allclose(T_lam.sum(axis=2), 1.0, atol=1e-9), "rows must sum to 1"
        return T_lam, 0.0

    negative_rows = (T_lam < 0).any(axis=2)  # (S, A)
    clipped_fraction = float(negative_rows.mean())
    if clipped_fraction > 0.0:
        T_lam = np.clip(T_lam, 0.0, None)
        T_lam = T_lam / T_lam.sum(axis=2, keepdims=True)
    return T_lam, clipped_fraction


def p_effective(p_train: float, lam: float) -> float:
    """p_eff = p_train + lam * (1 - p_train). Value of p that T_lam
    corresponds to, if T_hat approximates T(p_train)."""
    return p_train + lam * (1.0 - p_train)


def clip_diagnostics(T: np.ndarray, lam: float) -> dict:
    """Full clip diagnostic for a given lam (typically < 0).

    clipped_fraction (already returned by controllability_dial) is a step
    function: it takes the same value for any strictly negative lam,
    however small in magnitude (verified empirically), and therefore says
    nothing about the severity of the clip. clipped_mass -- the average
    total-variation distance between the raw tensor (before clipping) and
    the clipped-then-renormalised tensor -- does grow continuously with
    |lam| and measures how much the renormalisation actually distorts each
    row. clipped_mass is what must be looked at to judge whether a lam<0
    intervention stays clean, not clipped_fraction alone.
    """
    M = marginalise(T)
    T_raw = (1.0 - lam) * T + lam * M
    T_clipped, clipped_fraction = controllability_dial(T, lam)
    tv = 0.5 * np.abs(T_clipped - T_raw).sum(axis=2)  # (S, A)
    return {"clipped_fraction": clipped_fraction, "clipped_mass": float(tv.mean())}


def structure_injection(T_hat: np.ndarray, T_det: np.ndarray, mu: float) -> np.ndarray:
    """T_mu = (1-mu)*T_hat + mu*T_det. mu=0 intact belief, mu=1 perfectly
    deterministic belief (the true tensor T(p=0), not a learned object).

    Convex combination of two valid distributions (T_hat and T_det are each
    already simplex rows): no clip is ever necessary, the simplex
    constraints hold for every mu in [0,1] without having to enforce them.

    Different from the lam dial (controllability_dial): that one blends
    towards M, the marginal of T_hat itself -- it AMPLIFIES or ATTENUATES
    the structure already present in the belief, without ever injecting new
    structure (the marginal M is preserved by construction).
    structure_injection instead blends towards T_det, the true deterministic
    tensor -- it progressively REPLACES the (potentially wrong or noisy)
    structure of T_hat with the true structure, whatever it may be. This is
    NOT a direct substitute for lam: the two answer different questions
    (rescaling the existing belief vs. supplying it with the correct
    geometry again), and must not be conflated in the analysis.
    """
    return (1.0 - mu) * T_hat + mu * T_det
