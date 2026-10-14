"""E3 final campaign, secondary analysis: b by visit class at the budget (300,000 environment steps), by zone. A state's
visits are those of its least-visited action; in each zone and class, the mean I_s over the environment run's states
of that class minus the mean over the reference run's states of that class. One marker per seed; a cell with no state
on one side is absent. PENDING until the campaign is extracted. Writes paper/figures/out/fig_e3_campaign_visits.pdf.

Source: the campaign's run directories, through experiments/E3_worldmodel_4zones/campaign_analysis.analyse
(secondary[].checkpoints[].b_by_visit_class; classes campaign_analysis.VISIT_CLASSES).
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

NAME = "fig_e3_campaign_visits"
SEED_COLORS = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"], common.TOL_BRIGHT["purple"])
CLASS_MARKER = {"A": "o", "B": "s"}


def main() -> None:
    result, out_dir = src.campaign_analysis(__doc__.splitlines()[0])
    analysis = src.e3_module("campaign_analysis.py")
    classes = [analysis.class_label(lo, hi) for lo, hi in analysis.VISIT_CLASSES]
    pairs = sorted(result["secondary"], key=lambda p: p["seed"])
    common.apply_style()
    fig, axes = common.figure(aspect=0.62, nrows=2, ncols=2, sharex=True)
    x = np.arange(len(classes))
    for i, pair in enumerate(pairs):
        cp = src.budget_checkpoint(pair, analysis.BUDGET_ENV_STEPS)
        offset = (i - (len(pairs) - 1) / 2) * 0.12
        for ax, zone in zip(axes.flat, src.ZONES):
            cells = [cp["b_by_visit_class"][zone][c] for c in classes]
            ax.plot(x + offset, [c["b"] for c in cells], CLASS_MARKER[pair["zone_class"]], ls="none", ms=3.5,
                    color=SEED_COLORS[i % len(SEED_COLORS)], label=f"seed {pair['seed']}")
    for ax, zone in zip(axes.flat, src.ZONES):
        ax.set_title(src.ZONE_ABBR[zone], fontsize=common.SMALL_FONT_SIZE)
        ax.axhline(0.0, color="gray", lw=0.5, zorder=0)
    for ax in axes[1]:
        ax.set_xticks(x, classes, rotation=30)
        ax.set_xlabel("visits of the least-visited action")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\hat b$ (bits)")
    axes[0, 0].legend(fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME, out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
