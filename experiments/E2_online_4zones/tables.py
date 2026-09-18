"""Numeric tables requested alongside the figures (post-pilot review,
corrections pass): paired online-matched differences at three reference
checkpoints, their ventilation in the controllable-noisy zone, a
budget-trend power-law fit, the exterior Delta ceiling, final-checkpoint
b_hat vs its analytic value, and the paired-zone contrast's budget to
reach 90% of its analytic value. Every table reuses the SAME paired-
comparison primitive (stats_utils.paired_comparison) the figures use --
no second implementation of the Wilcoxon/effect-size computation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import linregress
from scipy.stats import t as student_t

import analysis_data as ad
import fig_f3 as f3
import fig_f5 as f5
import figures as fg
import stats_utils as su

ALPHA_REF, EPSILON_REF = 1.0, 0.1
REFERENCE_CHECKPOINTS = (10_000, 100_000, 1_000_000)
COUNT_BINS = (("N<10", 0, 10), ("10<=N<100", 10, 100), ("N>=100", 100, np.inf))


def _diff_source(quantity: str, per_state: np.ndarray, per_pair: np.ndarray) -> np.ndarray:
    return per_pair if quantity == "l1" else per_state


def table1_paired_diff_by_zone(per_state: np.ndarray, per_pair: np.ndarray) -> list[dict]:
    """Paired (online - matched) difference on b, L1, Delta, by zone
    type, at alpha=1 epsilon=0.1, checkpoints 1e4/1e5/1e6."""
    rows = []
    for quantity in ("b", "l1", "delta"):
        source = _diff_source(quantity, per_state, per_pair)
        for zone in ad.ZONE_CATEGORIES:
            diffs_by_cp = fg.paired_diff_by_checkpoint(source, zone, ALPHA_REF, EPSILON_REF, quantity, REFERENCE_CHECKPOINTS)
            for cp in REFERENCE_CHECKPOINTS:
                stats = su.paired_comparison(diffs_by_cp[cp])
                rows.append({
                    "checkpoint": cp, "zone": zone, "quantity": quantity,
                    "n_comparisons": stats["n"], "mean_diff": stats["mean"],
                    "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                    "wilcoxon_stat": stats["wilcoxon_stat"], "wilcoxon_p": stats["wilcoxon_p"],
                    "cohens_d": stats["cohens_d"],
                })
    fg.save_csv(
        rows,
        ["checkpoint", "zone", "quantity", "n_comparisons", "mean_diff", "ci95_low", "ci95_high",
         "wilcoxon_stat", "wilcoxon_p", "cohens_d"],
        "table1_paired_diff_by_zone",
    )
    return rows


def table2_controllable_noisy_ventilated(per_pair: np.ndarray) -> list[dict]:
    """Same paired L1 comparison, restricted to controllable-noisy,
    ventilated by action-optimal status and by count bin -- the two
    ventilations F3 already plots, reused here (never recomputed)."""
    rows = []
    zone = "controllable_noisy"

    for is_optimal, label in ((True, "optimal_under_P"), (False, "non_optimal")):
        mask_extra = per_pair["is_optimal_under_P"] == is_optimal
        diffs_by_cp = f3._paired_diff_masked(per_pair, mask_extra, zone, REFERENCE_CHECKPOINTS)
        for cp in REFERENCE_CHECKPOINTS:
            stats = su.paired_comparison(diffs_by_cp[cp])
            rows.append({
                "checkpoint": cp, "ventilation": "optimal_status", "subgroup": label,
                "n_comparisons": stats["n"], "mean_l1_diff": stats["mean"],
                "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                "wilcoxon_stat": stats["wilcoxon_stat"], "wilcoxon_p": stats["wilcoxon_p"],
                "cohens_d": stats["cohens_d"],
            })

    for bin_label, lo_n, hi_n in COUNT_BINS:
        mask_extra = (per_pair["N"] >= lo_n) & (per_pair["N"] < hi_n)
        diffs_by_cp = f3._paired_diff_masked(per_pair, mask_extra, zone, REFERENCE_CHECKPOINTS)
        for cp in REFERENCE_CHECKPOINTS:
            stats = su.paired_comparison(diffs_by_cp[cp])
            rows.append({
                "checkpoint": cp, "ventilation": "count_bin", "subgroup": bin_label,
                "n_comparisons": stats["n"], "mean_l1_diff": stats["mean"],
                "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                "wilcoxon_stat": stats["wilcoxon_stat"], "wilcoxon_p": stats["wilcoxon_p"],
                "cohens_d": stats["cohens_d"],
            })

    fg.save_csv(
        rows,
        ["checkpoint", "ventilation", "subgroup", "n_comparisons", "mean_l1_diff", "ci95_low", "ci95_high",
         "wilcoxon_stat", "wilcoxon_p", "cohens_d"],
        "table2_controllable_noisy_ventilated",
    )
    return rows


def _power_law_fit(xs: list[int], ys: list[float]) -> dict:
    log_x = np.log10(np.asarray(xs, dtype=float))
    log_y = np.log10(np.abs(np.asarray(ys, dtype=float)))
    result = linregress(log_x, log_y)
    n = len(xs)
    df = n - 2
    tcrit = float(student_t.ppf(0.975, df)) if df > 0 else float("nan")
    stderr = result.stderr if result.stderr is not None else float("nan")
    return {
        "n_points": n, "exponent": float(result.slope),
        "exponent_ci95_low": float(result.slope - tcrit * stderr),
        "exponent_ci95_high": float(result.slope + tcrit * stderr),
        "intercept": float(result.intercept), "r_value": float(result.rvalue),
    }


def table3_budget_trend_power_law(per_state: np.ndarray, per_pair: np.ndarray) -> list[dict]:
    """Power-law fit (|mean paired diff| ~ budget^exponent, OLS in
    log-log) on checkpoints >= 1e4, controllable-noisy zone, for b and L1
    -- reports whether the gap shrinks with budget and at what rate."""
    zone = "controllable_noisy"
    high_checkpoints = tuple(cp for cp in ad.CHECKPOINTS if cp >= 10_000)
    rows = []
    for quantity in ("b", "l1"):
        source = _diff_source(quantity, per_state, per_pair)
        diffs_by_cp = fg.paired_diff_by_checkpoint(source, zone, ALPHA_REF, EPSILON_REF, quantity, high_checkpoints)
        xs = list(high_checkpoints)
        ys = [float(np.mean(diffs_by_cp[cp])) for cp in high_checkpoints]
        fit = _power_law_fit(xs, ys)
        rows.append({"quantity": quantity, "zone": zone, **fit})
    fg.save_csv(
        rows, ["quantity", "zone", "n_points", "exponent", "exponent_ci95_low", "exponent_ci95_high",
               "intercept", "r_value"],
        "table3_budget_trend_power_law",
    )
    return rows


def table4_delta_exterior_max(per_state: np.ndarray) -> list[dict]:
    """Max over exterior (corridor) states of the seed-mean Delta, with
    its own 95% CI, per condition -- the quantified ceiling the F4
    exterior map (visually blank) doesn't itself carry."""
    rows = []
    for condition in ("online", "matched"):
        mask = (
            (per_state["zone"] == "corridor") & (per_state["condition"] == condition)
            & (per_state["alpha"] == ALPHA_REF) & (per_state["epsilon"] == EPSILON_REF)
            & (per_state["checkpoint"] == ad.FINAL_CHECKPOINT)
        )
        states = np.unique(per_state["state"][mask])
        best = None
        for st in states:
            st_mask = mask & (per_state["state"] == st)
            mean, lo, hi = su.mean_ci95(per_state["delta"][st_mask])
            n_seeds = int(st_mask.sum())
            if best is None or mean > best["max_mean_delta"]:
                best = {
                    "condition": condition, "state": int(st), "n_seeds": n_seeds,
                    "max_mean_delta": mean, "ci95_low": lo, "ci95_high": hi,
                }
        rows.append(best)
    fg.save_csv(
        rows, ["condition", "state", "n_seeds", "max_mean_delta", "ci95_low", "ci95_high"],
        "table4_delta_exterior_max",
    )
    return rows


def table5_b_final_checkpoint_vs_analytic(synthesis_rows: list[dict]) -> list[dict]:
    """Extends synthesis_table_b.csv with each zone's analytic b_hat and
    the online/matched gap to it -- never recomputes b_online_mean/
    b_matched_mean (reused from synthesis_table.py's own rows)."""
    rows = []
    for r in synthesis_rows:
        analytic = fg.ANALYTIC_B.get(r["zone"])
        rows.append({
            **r,
            "analytic_b": analytic,
            "online_gap_to_analytic": (r["b_online_mean"] - analytic) if analytic is not None else float("nan"),
            "matched_gap_to_analytic": (r["b_matched_mean"] - analytic) if analytic is not None else float("nan"),
        })
    fg.save_csv(
        rows,
        ["alpha", "zone", "n_seeds", "b_online_mean", "b_matched_mean", "b_diff_mean",
         "b_diff_ci95_low", "b_diff_ci95_high", "b_diff_cohens_d", "b_diff_wilcoxon_p",
         "analytic_b", "online_gap_to_analytic", "matched_gap_to_analytic"],
        "table5_b_final_checkpoint_vs_analytic",
    )
    return rows


def table6_contrast_budget_to_90pct(per_state: np.ndarray) -> list[dict]:
    """For each paired-zone contrast and alpha: the final-checkpoint
    contrast (already in F5b) plus the first checkpoint at which the
    mean contrast reaches 90% of its analytic value (None if never
    reached over the swept budget)."""
    rows = []
    for zone_a, zone_b, label in f5.CONTRASTS:
        analytic_contrast = fg.ANALYTIC_B[zone_a] - fg.ANALYTIC_B[zone_b]
        target = 0.9 * analytic_contrast
        for alpha in f5.ALPHAS:
            budget_reached = None
            final_contrast = None
            for cp in ad.CHECKPOINTS:
                means_a = f5._per_seed_zone_mean(per_state, zone_a, alpha, cp)
                means_b = f5._per_seed_zone_mean(per_state, zone_b, alpha, cp)
                common = sorted(set(means_a) & set(means_b))
                contrasts = np.array([means_a[s] - means_b[s] for s in common])
                mean_contrast = float(contrasts.mean())
                if budget_reached is None and mean_contrast >= target:
                    budget_reached = cp
                if cp == ad.FINAL_CHECKPOINT:
                    final_contrast = mean_contrast
            rows.append({
                "contrast": label, "alpha": alpha, "analytic_contrast": analytic_contrast,
                "target_90pct": target, "final_checkpoint_contrast": final_contrast,
                "budget_to_90pct": budget_reached if budget_reached is not None else "not_reached",
            })
    fg.save_csv(
        rows,
        ["contrast", "alpha", "analytic_contrast", "target_90pct", "final_checkpoint_contrast", "budget_to_90pct"],
        "table6_contrast_budget_to_90pct",
    )
    return rows
