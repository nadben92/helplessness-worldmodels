"""F3 -- localisation of the adaptive-sampling effect: paired L1 diff
ventilated (a) by zone x action-optimal-status, (b) by count bin, per
zone. Reference condition (alpha=1, epsilon=0.1)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import figures as fg
import stats_utils as su

ALPHA_REF, EPSILON_REF = 1.0, 0.1
COUNT_BINS = (("N<10", 0, 10), ("10<=N<100", 10, 100), ("N>=100", 100, np.inf))


def _paired_diff_masked(per_pair, mask_extra, zone, checkpoints=ad.CHECKPOINTS):
    online_mask = fg.zone_condition_mask(per_pair, zone, "online", ALPHA_REF, EPSILON_REF) & mask_extra
    matched_mask = fg.zone_condition_mask(per_pair, zone, "matched", ALPHA_REF, EPSILON_REF) & mask_extra
    out = {}
    for cp in checkpoints:
        o_mask = online_mask & (per_pair["checkpoint"] == cp)
        m_mask = matched_mask & (per_pair["checkpoint"] == cp)
        seeds = np.unique(per_pair["seed"][o_mask])
        diffs = []
        for s in seeds:
            o_vals = per_pair["l1"][o_mask & (per_pair["seed"] == s)]
            m_vals = per_pair["l1"][m_mask & (per_pair["seed"] == s)]
            if len(o_vals) and len(m_vals):
                diffs.append(o_vals.mean() - m_vals.mean())
        out[cp] = np.array(diffs)
    return out


def make_f3(per_pair: np.ndarray) -> None:
    csv_rows = []

    # --- Row 1: by zone x optimal-status ---
    fig1, axes1 = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, zone in zip(axes1, ad.ZONE_CATEGORIES):
        fg.setup_log_x_axis(ax)
        ax.set_title(fg.ZONE_LABELS[zone])
        for is_optimal, label, style in (
            (True, "action optimale sous P", {"linestyle": "-", "marker": "o"}),
            (False, "action non optimale", {"linestyle": "--", "marker": "x"}),
        ):
            mask_extra = per_pair["is_optimal_under_P"] == is_optimal
            diffs_by_cp = _paired_diff_masked(per_pair, mask_extra, zone)
            xs, means, los, his = [], [], [], []
            for cp in ad.CHECKPOINTS:
                d = diffs_by_cp[cp]
                if len(d) == 0:
                    continue
                mean, lo, hi = su.mean_ci95(d)
                xs.append(cp)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                csv_rows.append({
                    "ventilation": "zone_x_optimal_status", "zone": zone, "is_optimal_under_P": is_optimal,
                    "checkpoint": cp, "n_seeds": len(d), "mean_l1_diff": mean, "ci95_low": lo, "ci95_high": hi,
                })
            ax.plot(xs, means, color="black", markersize=5, label=label, **style)
            ax.fill_between(xs, los, his, alpha=0.15, color="black")
    axes1[0].set_ylabel("L1 online − apparié")
    axes1[0].legend(fontsize=7, loc="best")
    fig1.suptitle(f"F3a — écart L1 par zone × statut de l'action (α={ALPHA_REF}, ε={EPSILON_REF})")
    fig1.tight_layout()
    fg.save_figure(fig1, "F3a_l1_diff_by_zone_and_optimal_status")

    # --- Row 2: by count bin, per zone (lines) ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (bin_label, lo_n, hi_n) in zip(axes2, COUNT_BINS):
        fg.setup_log_x_axis(ax)
        ax.set_title(bin_label)
        for zone in ad.ZONE_CATEGORIES:
            mask_extra = (per_pair["N"] >= lo_n) & (per_pair["N"] < hi_n)
            diffs_by_cp = _paired_diff_masked(per_pair, mask_extra, zone)
            xs, means, los, his = [], [], [], []
            for cp in ad.CHECKPOINTS:
                d = diffs_by_cp[cp]
                if len(d) == 0:
                    continue
                mean, clo, chi = su.mean_ci95(d)
                xs.append(cp)
                means.append(mean)
                los.append(clo)
                his.append(chi)
                csv_rows.append({
                    "ventilation": "count_bin", "zone": zone, "count_bin": bin_label,
                    "checkpoint": cp, "n_seeds": len(d), "mean_l1_diff": mean, "ci95_low": clo, "ci95_high": chi,
                })
            style = fg.ZONE_STYLE[zone]
            if xs:
                ax.plot(xs, means, marker=style["marker"], linestyle=style["linestyle"],
                        color=style["color"], label=fg.ZONE_LABELS[zone], markersize=4)
                ax.fill_between(xs, los, his, alpha=0.1, color=style["color"])
    axes2[0].set_ylabel("L1 online − apparié")
    axes2[0].legend(fontsize=6, loc="best")
    fig2.suptitle(f"F3b — écart L1 par tranche de comptage (α={ALPHA_REF}, ε={EPSILON_REF})")
    fig2.tight_layout()
    fg.save_figure(fig2, "F3b_l1_diff_by_count_bin")

    fg.save_csv(
        csv_rows,
        ["ventilation", "zone", "is_optimal_under_P", "count_bin", "checkpoint", "n_seeds",
         "mean_l1_diff", "ci95_low", "ci95_high"],
        "F3_l1_diff_localisation",
    )
