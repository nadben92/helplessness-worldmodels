"""E1 Step A: the interior gap V*(environment) - V*(control) inside each uncontrollable block, against gamma, for each
goal reward R and both drift directions (36 combinations, perfect model). Writes paper/figures/out/fig_e1_calibration.pdf.

Source: experiments/E1_offline_4zones/artifacts/calibration_report.csv (calibration.py): interior_gap_nw (the
uncontrollable-deterministic block) and interior_gap_se (the uncontrollable-noisy block), plotted as recorded.
"""

from __future__ import annotations

import common
import e1_sources as src

NAME = "fig_e1_calibration"
PANELS = (("interior_gap_nw", "(a) UD block"), ("interior_gap_se", "(b) UN block"))
DIRECTION_STYLE = {"away_from_goal": ("-", "o", "away from the goal"), "toward_goal": ("--", "s", "toward the goal")}
R_COLORS = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"])


def main() -> None:
    rows = src.calibration_rows()
    q = src.step_a()
    common.apply_style()
    fig, axes = common.figure(aspect=0.55, ncols=2, sharey=True)
    for ax, (column, title) in zip(axes, PANELS):
        for R, color in zip(q["rewards"], R_COLORS):
            for direction, (ls, marker, label) in DIRECTION_STYLE.items():
                sub = sorted((r for r in rows if r["R"] == R and r["direction"] == direction), key=lambda r: r["gamma"])
                ax.plot([r["gamma"] for r in sub], [r[column] for r in sub], ls=ls, marker=marker, ms=3, lw=0.9,
                        color=color, label=f"R = {R:g}, {label}")
        ax.axhline(0.0, color="gray", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=common.SMALL_FONT_SIZE)
        ax.set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"interior $V^*_{\mathrm{env}} - V^*_{\mathrm{ctrl}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2, rect=(0, 0.17, 1, 1))
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
