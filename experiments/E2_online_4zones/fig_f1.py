"""F1 -- b_hat per zone type vs budget, online vs matched, one figure
per alpha (epsilon=0.1 only -- F1 does not cover the epsilon=1 bridge,
see F6 for that)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import figures as fg
import stats_utils as su

F1_ZONES = (
    "uncontrollable_deterministic", "controllable_noisy",
    "controllable_deterministic", "uncontrollable_noisy",
)


def make_f1(per_state: np.ndarray, alpha: float) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    csv_rows = []

    for ax, zone in zip(axes, F1_ZONES):
        fg.setup_log_x_axis(ax)
        ax.set_title(fg.ZONE_LABELS[zone])
        for condition in ("online", "matched"):
            mask = fg.zone_condition_mask(per_state, zone, condition, alpha, 0.1)
            series = fg.per_seed_values_by_checkpoint(per_state, mask, "b", ad.CHECKPOINTS)
            xs, means, los, his = [], [], [], []
            for cp in ad.CHECKPOINTS:
                vals = series[cp]
                mean, lo, hi = su.mean_ci95(vals)
                xs.append(cp)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                csv_rows.append({
                    "alpha": alpha, "zone": zone, "condition": condition, "checkpoint": cp,
                    "n_seeds": len(vals), "mean_b": mean, "ci95_low": lo, "ci95_high": hi,
                })
            style = fg.CONDITION_STYLE[condition]
            ax.plot(xs, means, marker=style["marker"], linestyle=style["linestyle"],
                    fillstyle=style["fillstyle"], color="black", label=condition, markersize=5)
            ax.fill_between(xs, los, his, alpha=0.15, color="black")
            # individual seeds, thin lines, at the final checkpoint only
            # (to keep the axis readable) -- shown as a jittered scatter
            final_vals = series[ad.CHECKPOINTS[-1]]
            jitter = np.random.default_rng(0).uniform(-0.05, 0.05, size=len(final_vals))
            ax.scatter(
                [ad.CHECKPOINTS[-1] * (1 + j) for j in jitter], final_vals,
                s=8, color="black", alpha=0.3, zorder=1,
            )

        if zone in fg.ANALYTIC_B:
            ax.axhline(fg.ANALYTIC_B[zone], color="red", linewidth=0.8, linestyle=":", zorder=0)
        ax.set_ylabel("b̂ (bits)" if zone == F1_ZONES[0] else "")

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"F1 — b̂ par type de zone vs budget (α={alpha}, ε=0.1)")
    fig.tight_layout()
    fg.save_figure(fig, f"F1_b_vs_budget_alpha{alpha}")
    fg.save_csv(
        csv_rows,
        ["alpha", "zone", "condition", "checkpoint", "n_seeds", "mean_b", "ci95_low", "ci95_high"],
        f"F1_b_vs_budget_alpha{alpha}",
    )
