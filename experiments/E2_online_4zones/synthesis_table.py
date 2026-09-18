"""Final-checkpoint synthesis table: per zone type x alpha, b_hat online,
b_hat matched, their paired (intra-seed) difference with 95% CI, and
mean Delta outside the 4 blocks (corridor only)."""

from __future__ import annotations

import numpy as np

import analysis_data as ad
import figures as fg
import stats_utils as su

EPSILON_REF = 0.1


def build_synthesis_table(per_state: np.ndarray) -> list[dict]:
    rows = []
    final_t = ad.FINAL_CHECKPOINT
    for alpha in (0.5, 1.0, 5.0):
        for zone in ad.ZONE_CATEGORIES:
            b_online = {}
            b_matched = {}
            for condition, store in (("online", b_online), ("matched", b_matched)):
                mask = (
                    (per_state["zone"] == zone) & (per_state["condition"] == condition)
                    & (per_state["alpha"] == alpha) & (per_state["epsilon"] == EPSILON_REF)
                    & (per_state["checkpoint"] == final_t)
                )
                for s in np.unique(per_state["seed"][mask]):
                    store[int(s)] = float(per_state["b"][mask & (per_state["seed"] == s)].mean())
            common = sorted(set(b_online) & set(b_matched))
            online_vals = np.array([b_online[s] for s in common])
            diffs = np.array([b_online[s] - b_matched[s] for s in common])
            mean_online, _, _ = su.mean_ci95(online_vals)
            matched_vals = np.array([b_matched[s] for s in common])
            mean_matched, _, _ = su.mean_ci95(matched_vals)
            diff_stats = su.paired_comparison(diffs)

            rows.append({
                "alpha": alpha, "zone": zone, "n_seeds": len(common),
                "b_online_mean": mean_online, "b_matched_mean": mean_matched,
                "b_diff_mean": diff_stats["mean"], "b_diff_ci95_low": diff_stats["ci_low"],
                "b_diff_ci95_high": diff_stats["ci_high"], "b_diff_cohens_d": diff_stats["cohens_d"],
                "b_diff_wilcoxon_p": diff_stats["wilcoxon_p"],
            })

    # Delta outside the 4 blocks (corridor only), per alpha, both conditions
    exterior_rows = []
    for alpha in (0.5, 1.0, 5.0):
        for condition in ("online", "matched"):
            mask = (
                (per_state["zone"] == "corridor") & (per_state["condition"] == condition)
                & (per_state["alpha"] == alpha) & (per_state["epsilon"] == EPSILON_REF)
                & (per_state["checkpoint"] == final_t)
            )
            seeds = np.unique(per_state["seed"][mask])
            seed_means = [float(per_state["delta"][mask & (per_state["seed"] == s)].mean()) for s in seeds]
            mean, lo, hi = su.mean_ci95(np.array(seed_means))
            exterior_rows.append({
                "alpha": alpha, "condition": condition, "n_seeds": len(seed_means),
                "delta_exterior_mean": mean, "ci95_low": lo, "ci95_high": hi,
            })

    fg.save_csv(
        rows,
        ["alpha", "zone", "n_seeds", "b_online_mean", "b_matched_mean", "b_diff_mean",
         "b_diff_ci95_low", "b_diff_ci95_high", "b_diff_cohens_d", "b_diff_wilcoxon_p"],
        "synthesis_table_b",
    )
    fg.save_csv(
        exterior_rows, ["alpha", "condition", "n_seeds", "delta_exterior_mean", "ci95_low", "ci95_high"],
        "synthesis_table_delta_exterior",
    )
    return rows, exterior_rows
