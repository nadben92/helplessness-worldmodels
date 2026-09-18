"""Spot-verifications requested alongside the corrected figures:
whether the corridor's visitation anisotropy (column 0 >> row 0, column 6
almost empty) is geometric (persists with no policy at all, under the
epsilon=1 bridge, and matches the exact DP-optimal-path prediction) or
loop-induced (only shows up under the epsilon=0.1 goal-seeking policy),
and how many pairs are still undersampled (N<10, N<100) at the final
checkpoint. Facts and numbers only -- no causal claim stronger than what
the resolved drift vectors, the DP computation and the raw counts
themselves show.
"""

from __future__ import annotations

import numpy as np

import analysis_data as ad
import env as e2_env
import figures as fg
import stats_utils as su

from helplessness.config import load_yaml_block
from helplessness.evaluation.tiebreak import uniform_tie_policy_probs
from helplessness.envs.zones import resolve_drift
from helplessness.planning.value_iteration import value_iteration

FINAL_T = ad.FINAL_CHECKPOINT
GRID_SHAPE = ad.GRID_SHAPE
GOAL_RC = (0, 0)
CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GAMMA = 0.9
GOAL_REWARD = 50.0
BRIDGE_CONDITION = "a1_e1.0_bridge"


def _condition_run_dirs(condition_key: str):
    return [d for d in ad.all_run_dirs() if d.name.startswith(f"{condition_key}_seed")]


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


def corridor_branch_visitation(condition_key: str) -> list[dict]:
    """N(s) summed over each of the corridor's 4 branches (row 0, row 6,
    col 0, col 6 -- corridor cells never move under counterbalancing, so
    no canonical remap is needed here), mean +/- 95%CI across the given
    condition's 20 seeds (online sub-condition), at the final
    checkpoint. Reused for both the epsilon=0.1 reference and the
    epsilon=1 bridge (whose online arm plays a UNIFORM random policy --
    "ou la politique ne joue aucun role")."""
    run_dirs = _condition_run_dirs(condition_key)
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
        mean, lo, hi = su.mean_ci95(vals)
        rows.append({"branch": branch, "n_seeds": len(vals), "mean_N": mean, "ci95_low": lo, "ci95_high": hi})
    return rows


def _expected_visits_from_optimal_policy(rotation: int) -> np.ndarray:
    """(N_STATES,) expected number of visits to each non-goal state,
    averaged UNIFORMLY over all 143 non-terminal start states, under the
    value-iteration-optimal (uniform-tie) policy for the TRUE kernel P --
    exact absorbing-Markov-chain algebra ((I-Q)^-1, Q = the induced
    chain restricted to non-goal states), no data collection, no RNG.
    This is the pure geometric/policy prediction: what optimal,
    goal-directed navigation alone would produce."""
    assignment = e2_env.quadrant_assignment_for_rotation(rotation)
    layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)
    env = e2_env.build_env(layout, CFG, goal_reward=GOAL_REWARD)
    P_true = env.transition_tensor()
    R_true, terminal = e2_env.reward_and_terminal(CFG, goal_reward=GOAL_REWARD)

    V_star, _ = value_iteration(P_true, R_true, terminal, gamma=GAMMA)
    Q_star = np.einsum("sap,p->sa", P_true, R_true + GAMMA * V_star * (~terminal))
    policy_probs = uniform_tie_policy_probs(Q_star)
    P_pi = np.einsum("sa,sap->sp", policy_probs, P_true)  # (S, S) induced Markov chain

    goal_s = ad.GOAL_S
    non_goal = np.array([s for s in range(ad.N_STATES) if s != goal_s])
    Q = P_pi[np.ix_(non_goal, non_goal)]
    M = np.linalg.inv(np.eye(len(non_goal)) - Q)  # fundamental matrix: M[i,j] = E[visits to j | start i]

    expected_visits = np.zeros(ad.N_STATES)
    expected_visits[non_goal] = M.mean(axis=0)  # uniform average over start states (rows)
    return expected_visits


def dp_optimal_branch_visits() -> list[dict]:
    """Branch totals of the DP-exact expected visit counts under the
    optimal policy for P, averaged over the 4 counterbalancing rotations
    (equal weight, matching the sweep's 5-seeds-per-rotation design)."""
    branch_totals = {"row0": [], "row6": [], "col0": [], "col6": []}
    for rotation in range(4):
        ev = _expected_visits_from_optimal_policy(rotation).reshape(GRID_SHAPE)
        branch_totals["row0"].append(ev[0, :].sum())
        branch_totals["row6"].append(ev[6, :].sum())
        branch_totals["col0"].append(ev[:, 0].sum())
        branch_totals["col6"].append(ev[:, 6].sum())
    rows = []
    for branch, vals in branch_totals.items():
        vals = np.array(vals)
        mean, lo, hi = su.mean_ci95(vals)
        rows.append({"branch": branch, "n_rotations": len(vals), "mean_expected_visits": mean, "ci95_low": lo, "ci95_high": hi})
    return rows


def _shares(rows: list[dict], value_field: str) -> dict[str, float]:
    total = sum(r[value_field] for r in rows)
    return {r["branch"]: r[value_field] / total for r in rows}


def branch_comparison_table(dp_rows, bridge_rows, reference_rows) -> list[dict]:
    """Normalized branch SHARES (each branch's fraction of the 4-branch
    total) side by side: DP-optimal (pure policy+geometry, no
    collection), epsilon=1 bridge (pure dynamics, no policy), epsilon=0.1
    reference (observed). Comparable across sources despite very
    different absolute scales (per-episode expected visits vs. total
    counts over a 1e6-step budget) because each is normalized to its own
    4-branch total."""
    dp_shares = _shares(dp_rows, "mean_expected_visits")
    bridge_shares = _shares(bridge_rows, "mean_N")
    reference_shares = _shares(reference_rows, "mean_N")
    rows = []
    for branch in ("row0", "row6", "col0", "col6"):
        rows.append({
            "branch": branch,
            "dp_optimal_share": dp_shares[branch],
            "epsilon1_bridge_share": bridge_shares[branch],
            "epsilon0.1_reference_share": reference_shares[branch],
        })
    return rows


def run_anisotropy_check() -> dict:
    drift_rows = resolved_drift_facts()
    reference_rows = corridor_branch_visitation(ad.REFERENCE_CONDITION)
    bridge_rows = corridor_branch_visitation(BRIDGE_CONDITION)
    dp_rows = dp_optimal_branch_visits()
    comparison_rows = branch_comparison_table(dp_rows, bridge_rows, reference_rows)

    fg.save_csv(reference_rows, ["branch", "n_seeds", "mean_N", "ci95_low", "ci95_high"], "verif_corridor_branch_visitation")
    fg.save_csv(bridge_rows, ["branch", "n_seeds", "mean_N", "ci95_low", "ci95_high"], "verif_corridor_branch_visitation_epsilon1_bridge")
    fg.save_csv(dp_rows, ["branch", "n_rotations", "mean_expected_visits", "ci95_low", "ci95_high"], "verif_corridor_branch_dp_optimal")
    fg.save_csv(
        comparison_rows, ["branch", "dp_optimal_share", "epsilon1_bridge_share", "epsilon0.1_reference_share"],
        "verif_corridor_branch_comparison",
    )
    fg.save_csv(drift_rows, ["band", "row_range", "row_drift", "col_drift"], "verif_resolved_drift")
    return {"drift": drift_rows, "reference": reference_rows, "bridge": bridge_rows, "dp": dp_rows, "comparison": comparison_rows}


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
