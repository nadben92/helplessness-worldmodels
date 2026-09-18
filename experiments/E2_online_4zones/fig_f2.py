"""F2 -- the central figure: paired (intra-seed) online - matched
difference on b_hat, L1, and Delta, per zone type, vs budget. Reference
condition (alpha=1, epsilon=0.1) only."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import analysis_data as ad
import figures as fg
import stats_utils as su

ALPHA_REF, EPSILON_REF = 1.0, 0.1


def make_f2(per_state: np.ndarray, per_pair: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    csv_rows = []

    panels = [
        ("b", per_state, "b̂ online − apparié (bits)"),
        ("delta", per_state, "Δ online − apparié"),
        ("l1", per_pair, "L1 online − apparié"),
    ]

    for ax, (field, source, ylabel) in zip(axes, panels):
        fg.setup_log_x_axis(ax)
        ax.set_ylabel(ylabel)
        for zone in ad.ZONE_CATEGORIES:
            diffs_by_cp = fg.paired_diff_by_checkpoint(source, zone, ALPHA_REF, EPSILON_REF, field, ad.CHECKPOINTS)
            xs, means, los, his = [], [], [], []
            for cp in ad.CHECKPOINTS:
                mean, lo, hi = su.mean_ci95(diffs_by_cp[cp])
                xs.append(cp)
                means.append(mean)
                los.append(lo)
                his.append(hi)
                csv_rows.append({
                    "quantity": field, "zone": zone, "checkpoint": cp,
                    "n_seeds": len(diffs_by_cp[cp]), "mean_diff": mean, "ci95_low": lo, "ci95_high": hi,
                })
            style = fg.ZONE_STYLE[zone]
            ax.plot(xs, means, marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], label=fg.ZONE_LABELS[zone], markersize=5)
            ax.fill_between(xs, los, his, alpha=0.12, color=style["color"])

    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(f"F2 — écart apparié online − apparié (α={ALPHA_REF}, ε={EPSILON_REF})")
    fig.tight_layout()
    fg.save_figure(fig, "F2_null_model_paired_diff")
    fg.save_csv(
        csv_rows, ["quantity", "zone", "checkpoint", "n_seeds", "mean_diff", "ci95_low", "ci95_high"],
        "F2_null_model_paired_diff",
    )
