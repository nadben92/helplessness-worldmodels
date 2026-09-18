"""Phase A -- status table of predicted forms, on the sweep's saved
artifacts (results/E2_online_4zones_sweep/), never rerunning training
except where explicitly noted (the jumeau check, which needs
live-agent V/policy the sweep never saved -- see jumeau_status()'s
docstring). Facts and statuses only, no interpretation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import analysis_data as ad

FLOAT_TOL = 1e-9
DELTA_TOL = -1e-8


def _group_indices(arr: np.ndarray, fields: tuple[str, ...]) -> dict[tuple, np.ndarray]:
    """Groups row INDICES of a structured array by the given field
    values -- returns {key_tuple: index_array}."""
    keys = list(zip(*(arr[f] for f in fields)))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return {k: np.array(v) for k, v in groups.items()}


def b_exact_zero_status(per_state: np.ndarray) -> dict:
    """b == 0 exact (float precision) in corridor and
    controllable_deterministic, both conditions, all alpha, all epsilon."""
    out = {}
    for zone in ("corridor", "controllable_deterministic"):
        for condition in ("online", "matched"):
            mask = (per_state["zone"] == zone) & (per_state["condition"] == condition)
            b_vals = per_state["b"][mask]
            n = int(mask.sum())
            max_abs_b = float(np.max(np.abs(b_vals))) if n else float("nan")
            out[(zone, condition)] = {
                "n_rows": n,
                "max_abs_b": max_abs_b,
                "status": "observed" if (n > 0 and max_abs_b < FLOAT_TOL) else "non observe",
            }
    return out


def t_hat_equal_via_l1_status(per_pair: np.ndarray) -> dict:
    """T_hat online == T_hat matched in the conveyor and
    controllable_deterministic zones, at equal counts. The raw matched-
    condition N table is not persisted by the sweep (only its DERIVED
    measurement outputs are, in per_pair.csv) -- checked here via l1
    (||T_hat - P_true||_1, saved for both conditions): a deterministic
    zone's T_hat is a pure function of the pair's total count alone
    (every draw of a deterministic pair lands on the SAME single
    successor, regardless of which random draws produced it), and
    matched's N_target exactly equals online's own N per pair by
    construction (matched_control.py::collect_matched) -- so T_hat_online
    == T_hat_matched is a mathematical consequence, not something that
    needs a fresh rerun to observe; l1_online == l1_matched is the
    necessary, saved-data-only proxy checked here, exact to float
    precision."""
    out = {}
    for zone in ("uncontrollable_deterministic", "controllable_deterministic"):
        online_mask = (per_pair["zone"] == zone) & (per_pair["condition"] == "online")
        matched_mask = (per_pair["zone"] == zone) & (per_pair["condition"] == "matched")
        # join on (alpha, epsilon, seed, assignment, checkpoint, state, action)
        join_fields = ("alpha", "epsilon", "seed", "assignment", "checkpoint", "state", "action")
        online_keys = list(zip(*(per_pair[f][online_mask] for f in join_fields)))
        matched_keys = list(zip(*(per_pair[f][matched_mask] for f in join_fields)))
        online_l1 = per_pair["l1"][online_mask]
        matched_l1 = per_pair["l1"][matched_mask]
        matched_lookup = dict(zip(matched_keys, matched_l1))
        diffs = []
        for k, l1_o in zip(online_keys, online_l1):
            l1_m = matched_lookup.get(k)
            if l1_m is not None:
                diffs.append(abs(l1_o - l1_m))
        max_diff = float(max(diffs)) if diffs else float("nan")
        out[zone] = {
            "n_pairs_compared": len(diffs),
            "max_l1_diff": max_diff,
            "status": "observed" if (diffs and max_diff < FLOAT_TOL) else "non observe",
        }
    return out


def delta_nonneg_status(per_state: np.ndarray) -> dict:
    for condition in ("online", "matched"):
        mask = per_state["condition"] == condition
        delta = per_state["delta"][mask]
        min_delta = float(np.min(delta)) if len(delta) else float("nan")
        n_violations = int((delta < DELTA_TOL).sum())
        yield condition, {
            "n_rows": int(mask.sum()), "min_delta": min_delta, "n_violations": n_violations,
            "status": "observed" if n_violations == 0 else "non observe",
        }


def b_variance_status(per_state: np.ndarray) -> dict:
    """Variance of b (online) across the 20 seeds must be > 0 in both
    noisy zones, at the final checkpoint, for the reference condition
    (alpha=1, epsilon=0.1) -- checked at every (alpha, epsilon)
    combination present, not only the reference, since the spec applies
    generally."""
    out = {}
    conditions_present = sorted(set(zip(per_state["alpha"].tolist(), per_state["epsilon"].tolist())))
    for alpha, epsilon in conditions_present:
        for zone in ad.NOISY_ZONES:
            mask = (
                (per_state["zone"] == zone) & (per_state["condition"] == "online")
                & (per_state["alpha"] == alpha) & (per_state["epsilon"] == epsilon)
                & (per_state["checkpoint"] == ad.FINAL_CHECKPOINT)
            )
            # mean b per seed (averaging over states in the zone), then variance across seeds
            seeds = sorted(set(per_state["seed"][mask].tolist()))
            per_seed_mean_b = []
            for seed in seeds:
                seed_mask = mask & (per_state["seed"] == seed)
                per_seed_mean_b.append(float(np.mean(per_state["b"][seed_mask])))
            variance = float(np.var(per_seed_mean_b)) if len(per_seed_mean_b) > 1 else float("nan")
            out[(alpha, epsilon, zone)] = {
                "n_seeds": len(per_seed_mean_b),
                "variance": variance,
                "status": "observed" if variance > 0 else "non observe",
            }
    return out


def n_start_leq_n_status(run_dirs: list) -> dict:
    """N_start[s,a] <= N[s,a] for every pair, at every checkpoint, every
    run -- read directly from arrays.npz (both fields already saved by
    sweep.py)."""
    n_checked = 0
    n_violations = 0
    max_violation = 0
    for run_dir in run_dirs:
        with np.load(run_dir / "arrays.npz") as arrays:
            checkpoint_steps = arrays["checkpoint_steps"].tolist()
            for t in checkpoint_steps:
                N_pair = arrays[f"N_checkpoint_{t}"].sum(axis=2)
                N_start = arrays[f"N_start_checkpoint_{t}"]
                n_checked += N_pair.size
                violation = N_start - N_pair
                n_violations += int((violation > 0).sum())
                if violation.max() > max_violation:
                    max_violation = int(violation.max())
    return {
        "n_pairs_checked": n_checked,
        "n_violations": n_violations,
        "max_violation": max_violation,
        "status": "observed" if n_violations == 0 else "non observe",
    }


def epsilon_one_no_difference_status(per_state: np.ndarray, per_pair: np.ndarray) -> dict:
    """At epsilon=1: no paired online-matched difference on b, L1, or
    Delta -- checked via a Wilcoxon signed-rank test on the intra-seed
    (online - matched) difference, per zone, at the final checkpoint
    (alpha=1, the only alpha swept at epsilon=1)."""
    from scipy.stats import wilcoxon

    out = {}
    mask_base = (per_state["epsilon"] == 1.0) & (per_state["checkpoint"] == ad.FINAL_CHECKPOINT)
    for zone in ad.ZONE_CATEGORIES:
        online_mask = mask_base & (per_state["zone"] == zone) & (per_state["condition"] == "online")
        matched_mask = mask_base & (per_state["zone"] == zone) & (per_state["condition"] == "matched")
        seeds = sorted(set(per_state["seed"][online_mask].tolist()))
        diffs_b = []
        diffs_delta = []
        for seed in seeds:
            o_mask = online_mask & (per_state["seed"] == seed)
            m_mask = matched_mask & (per_state["seed"] == seed)
            diffs_b.append(float(np.mean(per_state["b"][o_mask])) - float(np.mean(per_state["b"][m_mask])))
            diffs_delta.append(float(np.mean(per_state["delta"][o_mask])) - float(np.mean(per_state["delta"][m_mask])))

        pair_mask_base = (per_pair["epsilon"] == 1.0) & (per_pair["checkpoint"] == ad.FINAL_CHECKPOINT) & (per_pair["zone"] == zone)
        online_pair_mask = pair_mask_base & (per_pair["condition"] == "online")
        matched_pair_mask = pair_mask_base & (per_pair["condition"] == "matched")
        diffs_l1 = []
        for seed in seeds:
            o_mask = online_pair_mask & (per_pair["seed"] == seed)
            m_mask = matched_pair_mask & (per_pair["seed"] == seed)
            diffs_l1.append(float(np.mean(per_pair["l1"][o_mask])) - float(np.mean(per_pair["l1"][m_mask])))

        result = {}
        for name, diffs in (("b", diffs_b), ("delta", diffs_delta), ("l1", diffs_l1)):
            diffs_arr = np.array(diffs)
            if np.allclose(diffs_arr, 0.0):
                stat, p = float("nan"), float("nan")
            else:
                stat, p = wilcoxon(diffs_arr)
            result[name] = {
                "n_seeds": len(diffs), "mean_diff": float(np.mean(diffs_arr)),
                "wilcoxon_stat": float(stat), "wilcoxon_p": float(p),
                "status": "observed (no diff)" if (np.isnan(p) or p > 0.05) else "non observe (diff detectee)",
            }
        out[zone] = result
    return out


def counterbalancing_dispersion_status(per_state: np.ndarray) -> dict:
    """For each zone type, at the reference condition and final
    checkpoint: dispersion of mean b (and mean N, from per_pair
    elsewhere) across the 4 rotations vs. across the 20 seeds."""
    out = {}
    mask_base = (
        (per_state["alpha"] == 1.0) & (per_state["epsilon"] == 0.1)
        & (per_state["checkpoint"] == ad.FINAL_CHECKPOINT) & (per_state["condition"] == "online")
    )
    for zone in ad.ZONE_CATEGORIES:
        zone_mask = mask_base & (per_state["zone"] == zone)
        seeds = sorted(set(per_state["seed"][zone_mask].tolist()))
        per_seed_b = []
        per_seed_rotation = []
        for seed in seeds:
            seed_mask = zone_mask & (per_state["seed"] == seed)
            per_seed_b.append(float(np.mean(per_state["b"][seed_mask])))
            per_seed_rotation.append(int(per_state["assignment"][seed_mask][0]))
        per_seed_b = np.array(per_seed_b)
        per_seed_rotation = np.array(per_seed_rotation)

        rotation_means = [per_seed_b[per_seed_rotation == r].mean() for r in range(4) if (per_seed_rotation == r).any()]
        out[zone] = {
            "inter_seed_std": float(np.std(per_seed_b)),
            "inter_rotation_std_of_rotation_means": float(np.std(rotation_means)) if len(rotation_means) > 1 else float("nan"),
            "n_rotations_present": len(rotation_means),
        }
    return out
