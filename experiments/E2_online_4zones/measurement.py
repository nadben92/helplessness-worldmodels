"""Builds E2's two output tables (per-state, per-pair -- see artifact.py)
from a twin_from_counts() result and the ZoneLayout that produced P_true.
Pure row-building/labeling: every actual number comes from twin.py
(routing through measures/planning), nothing is recomputed here.
"""

from __future__ import annotations

from helplessness.envs.zones import interior_mask


def _region_lookup(layout, grid_shape: tuple[int, int]) -> dict:
    """state -> ("interior" | "boundary"), computed once per zone (not
    once per state) -- interior_mask() is a whole-grid array per call."""
    n_rows, n_cols = grid_shape
    region_of_state = {}
    zone_of_state = {}
    for zone in layout.zones:
        interior = interior_mask(zone, grid_shape)
        for r in range(n_rows):
            for c in range(n_cols):
                if zone.mask[r, c]:
                    s = r * n_cols + c
                    zone_of_state[s] = zone
                    region_of_state[s] = "interior" if interior[r, c] else "boundary"
    return zone_of_state, region_of_state


def per_state_rows(twin_result: dict, layout, grid_shape: tuple[int, int], meta: dict) -> list[dict]:
    """meta: exp, env, condition, alpha, epsilon, seed, assignment,
    checkpoint -- the join-key fields shared by every row this call
    produces."""
    zone_of_state, region_of_state = _region_lookup(layout, grid_shape)
    n_states = grid_shape[0] * grid_shape[1]
    rows = []
    for s in range(n_states):
        zone = zone_of_state[s]
        rows.append({
            **meta,
            "zone": zone.appearance.category,
            "zone_type": type(zone.dynamics).__name__,
            "region": region_of_state[s],
            "state": s,
            "I_hat": float(twin_result["I_hat"][s]),
            "I_star": float(twin_result["I_star"][s]),
            "b": float(twin_result["b"][s]),
            "delta": float(twin_result["delta"][s]),
            "v_pi_true": float(twin_result["v_pi_true"][s]),
            "v_pi_believed": float(twin_result["v_pi_believed"][s]),
        })
    return rows


def per_pair_rows(twin_result: dict, layout, grid_shape: tuple[int, int], n_actions: int, meta: dict) -> list[dict]:
    zone_of_state, region_of_state = _region_lookup(layout, grid_shape)
    n_states = grid_shape[0] * grid_shape[1]
    N_target = twin_result["N_target"]
    l1 = twin_result["l1"]
    is_optimal = twin_result["is_optimal_under_P"]

    rows = []
    for s in range(n_states):
        zone = zone_of_state[s]
        for a in range(n_actions):
            rows.append({
                **meta,
                "zone": zone.appearance.category,
                "zone_type": type(zone.dynamics).__name__,
                "region": region_of_state[s],
                "state": s,
                "action": a,
                "N": int(N_target[s, a]),
                "l1": float(l1[s, a]),
                "is_optimal_under_P": bool(is_optimal[s, a]),
            })
    return rows
