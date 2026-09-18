"""Two spot-verifications requested alongside the corrected figures:
whether the corridor's visitation anisotropy (column 0 >> row 0, column 6
almost empty) is consistent with the uncontrollable zones' drift axis
(row-only, by construction -- helplessness.envs.zones.resolve_drift) and
the goal's position, and how many pairs are still undersampled (N<10,
N<100) at the final checkpoint. Facts and numbers only -- no causal claim
stronger than what the resolved drift vectors and the raw counts
themselves show.
"""

from __future__ import annotations

import numpy as np

import analysis_data as ad
import figures as fg

from helplessness.envs.zones import resolve_drift

FINAL_T = ad.FINAL_CHECKPOINT
GRID_SHAPE = ad.GRID_SHAPE
GOAL_RC = (0, 0)


def _reference_run_dirs():
    return [d for d in ad.all_run_dirs() if d.name.startswith(f"{ad.REFERENCE_CONDITION}_seed")]


def resolved_drift_facts() -> list[dict]:
    """Row-axis drift resolved for each geometric quadrant BAND (rows
    1-5: nw/ne; rows 7-11: sw/se) under direction="away_from_goal" --
    this is rotation-INVARIANT: drift depends only on the geometric
    quadrant's own row range (env.py::_build_block_dynamics), never on
    which zone type currently occupies it, so it is the same in every
    one of the 20 seeds regardless of counterbalancing rotation."""
    rows = []
    for band, row_range in (("rows_1_5 (nw/ne)", (1, 5)), ("rows_7_11 (sw/se)", (7, 11))):
        drift = resolve_drift(row_range, GRID_SHAPE, GOAL_RC, "away_from_goal")
        rows.append({"band": band, "row_range": row_range, "row_drift": drift[0], "col_drift": drift[1]})
    return rows


def corridor_branch_visitation() -> list[dict]:
    """N(s) summed over each of the corridor's 4 branches (row 0, row 6,
    col 0, col 6 -- corridor cells never move under counterbalancing, so
    no canonical remap is needed here), mean +/- 95%CI across the
    reference condition's 20 seeds, at the final checkpoint."""
    from stats_utils import mean_ci95

    run_dirs = _reference_run_dirs()
    per_seed = {"row0": [], "row6": [], "col0": [], "col6": []}
    for run_dir in run_dirs:
        with np.load(run_dir / "arrays.npz") as arrays:
            N = arrays[f"N_checkpoint_{FINAL_T}"]
        N_state = N.sum(axis=(1, 2)).astype(float).reshape(GRID_SHAPE)
        per_seed["row0"].append(N_state[0, :].sum())
        per_seed["row6"].append(N_state[6, :].sum())
        per_seed["col0"].append(N_state[:, 0].sum())
        per_seed["col6"].append(N_state[:, 6].sum())

    rows = []
    for branch, vals in per_seed.items():
        vals = np.array(vals)
        mean, lo, hi = mean_ci95(vals)
        rows.append({"branch": branch, "n_seeds": len(vals), "mean_N": mean, "ci95_low": lo, "ci95_high": hi})
    return rows


def run_anisotropy_check() -> dict:
    drift_rows = resolved_drift_facts()
    branch_rows = corridor_branch_visitation()
    fg.save_csv(branch_rows, ["branch", "n_seeds", "mean_N", "ci95_low", "ci95_high"], "verif_corridor_branch_visitation")
    fg.save_csv(drift_rows, ["band", "row_range", "row_drift", "col_drift"], "verif_resolved_drift")
    return {"drift": drift_rows, "branches": branch_rows}


def undersampled_pairs_at_final_checkpoint(per_pair: np.ndarray) -> list[dict]:
    """Count of (state, action) pairs with N<10 and N<100 by zone type,
    at the final checkpoint, alpha=1 epsilon=0.1 online -- mean across
    the 20 seeds. The matched condition's N is IDENTICAL to online's by
    construction (matched_control.py::collect_matched draws exactly the
    online run's own N_target), so this count applies to both
    conditions without needing a separate pass."""
    mask_base = (
        (per_pair["condition"] == "online") & (per_pair["alpha"] == 1.0) & (per_pair["epsilon"] == 0.1)
        & (per_pair["checkpoint"] == FINAL_T)
    )
    rows = []
    for zone in ad.ZONE_CATEGORIES:
        zone_mask = mask_base & (per_pair["zone"] == zone)
        seeds = np.unique(per_pair["seed"][zone_mask])
        n_lt_10, n_lt_100, n_total = [], [], []
        for s in seeds:
            seed_mask = zone_mask & (per_pair["seed"] == s)
            Ns = per_pair["N"][seed_mask]
            n_lt_10.append(int((Ns < 10).sum()))
            n_lt_100.append(int((Ns < 100).sum()))
            n_total.append(int(seed_mask.sum()))
        rows.append({
            "zone": zone, "n_seeds": len(seeds),
            "mean_n_pairs_total": float(np.mean(n_total)),
            "mean_n_pairs_N_lt_10": float(np.mean(n_lt_10)),
            "mean_n_pairs_N_lt_100": float(np.mean(n_lt_100)),
        })
    fg.save_csv(
        rows, ["zone", "n_seeds", "mean_n_pairs_total", "mean_n_pairs_N_lt_10", "mean_n_pairs_N_lt_100"],
        "verif_undersampled_pairs_final_checkpoint",
    )
    return rows
