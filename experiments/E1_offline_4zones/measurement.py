"""Etape C -- measurements. Everything routes through measures/ and
planning/ (controllability_per_state, conditional_entropy_per_state,
l1_error_per_pair, policy_evaluation): no controllability or L1
computation is duplicated here, only per-zone/per-region aggregation of
arrays those functions already produce.

Delta procedure is exactly the spec's (planner values from
value_iteration(T_hat, ...) are DISCARDED for Delta -- V^pi_hat is
policy_evaluation(pi_hat, P_true, ...), never the planner's own return
value, which would silently evaluate under T_hat instead of P_true).
V_believed (the planner's own discarded values) is kept separately for the
pessimism report, per spec 3.4 -- it is a real output quantity there, not
an intermediate to throw away.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.zones import interior_mask
from helplessness.measures.belief import l1_error_per_pair
from helplessness.measures.controllability import controllability_per_state
from helplessness.planning.value_iteration import policy_evaluation, value_iteration


def compute_state_and_pair_arrays(
    P_true: np.ndarray,
    R_true: np.ndarray,
    terminal: np.ndarray,
    gamma: float,
    T_hat: np.ndarray,
    R_hat: np.ndarray,
) -> dict:
    """One environment, one (budget, seed) draw's worth of raw arrays --
    everything downstream (per-zone/per-region aggregation) indexes these
    with a mask, never recomputes them.
    """
    I_hat = controllability_per_state(T_hat)
    I_star = controllability_per_state(P_true)
    l1 = l1_error_per_pair(T_hat, P_true)  # (S, A)

    V_believed, pi_hat = value_iteration(T_hat, R_hat, terminal, gamma)  # planner's OWN values: kept for pessimism, NOT used as V^pi_hat
    V_pi_true = policy_evaluation(pi_hat, P_true, R_true, terminal, gamma)  # the real V^pi_hat, under the TRUE kernel, pi_hat fixed
    V_star, _ = value_iteration(P_true, R_true, terminal, gamma)

    delta = V_star - V_pi_true
    # Delta >= 0 is an exact MDP fact (V* is optimal under P, see
    # tests/planning/test_value_iteration.py's positive control), but
    # V_star and V_pi_true come from two independent value_iteration/
    # policy_evaluation fixed-point convergences (tol=1e-10 each), so
    # their difference carries floating-point noise on the order of
    # 1e-10 to 1e-15 that can print as -0.0000 or a tiny negative value
    # with no physical meaning. Clamp only that numerical floor (strictly
    # below 1e-9, far under the existing Delta >= -1e-6 test tolerance) to
    # exactly 0.0 -- a real violation (corrupted T_hat: mean > 5, see the
    # positive-control tests) is orders of magnitude larger and unaffected.
    delta = np.where(np.abs(delta) < 1e-9, 0.0, delta)
    pessimism = V_believed - V_pi_true

    return {
        "I_hat": I_hat,
        "I_star": I_star,
        "l1": l1,
        "delta": delta,
        "v_pi_true": V_pi_true,
        "v_pi_believed": V_believed,
        "pessimism": pessimism,
    }


def _region_state_indices(zone, grid_shape: tuple[int, int], region: str) -> np.ndarray:
    if region == "all":
        return np.flatnonzero(zone.mask)
    interior = interior_mask(zone, grid_shape)
    if region == "interior":
        return np.flatnonzero(interior)
    if region == "boundary":
        return np.flatnonzero(zone.mask & ~interior)
    raise ValueError(f"region must be 'interior', 'boundary', or 'all', got {region!r}")


def aggregate_zone_region(arrays: dict, zone, grid_shape: tuple[int, int], region: str) -> dict:
    """Mean of each per-state/per-(s,a) array over the given zone/region.

    Per spec: Î and L1 are meaningful over the whole zone (region == "all"
    only -- NaN in interior/boundary rows, not repeated, so a naive
    cross-region mean doesn't silently include a value that was never
    region-specific to begin with). Value-derived quantities (delta,
    v_pi_true, v_pi_believed, pessimism) are reported for every region:
    "interior" is primary (spec: aggregate on the interior mask),
    "boundary" is descriptive, "all" is provided too for joins that don't
    care about the split.
    """
    idx = _region_state_indices(zone, grid_shape, region)
    row = {
        "n": int(idx.size),
        "delta": float(arrays["delta"][idx].mean()) if idx.size else float("nan"),
        "v_pi_true": float(arrays["v_pi_true"][idx].mean()) if idx.size else float("nan"),
        "v_pi_believed": float(arrays["v_pi_believed"][idx].mean()) if idx.size else float("nan"),
        "pessimism": float(arrays["pessimism"][idx].mean()) if idx.size else float("nan"),
    }
    if region == "all":
        row["I_hat"] = float(arrays["I_hat"][idx].mean()) if idx.size else float("nan")
        row["I_star"] = float(arrays["I_star"][idx].mean()) if idx.size else float("nan")
        row["l1"] = float(arrays["l1"][idx].mean()) if idx.size else float("nan")
    else:
        row["I_hat"] = float("nan")
        row["I_star"] = float("nan")
        row["l1"] = float("nan")
    return row
