"""E3 final campaign, main analysis: b = I_s(environment) - I_s(reference) by zone at the budget (300,000 environment
steps), one marker per seed (filled: b on I_s; open: b on I_s projected on the 5 reachable successors), with the mean
over seeds and its 95 % Student-t interval (b on I_s). PENDING until the campaign is extracted. Writes
paper/figures/out/fig_e3_campaign_b.pdf.

Source: the campaign's run directories (e3_sources.CAMPAIGN_ROOT), through experiments/E3_worldmodel_4zones/
campaign_analysis.analyse (main.aggregate_b.pairs) and aggregate_b.mean_ci95 for the interval. No other computation.
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

NAME = "fig_e3_campaign_b"
SEED_COLORS = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"], common.TOL_BRIGHT["purple"])
CLASS_MARKER = {"A": "o", "B": "s"}


def main() -> None:
    result, out_dir = src.campaign_analysis(__doc__.splitlines()[0])
    aggregate_b = src.e3_module("aggregate_b.py")
    pairs = sorted(result["main"]["aggregate_b"]["pairs"], key=lambda p: p["seed"])
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.85, aspect=0.6)
    x = np.arange(len(src.ZONES))
    for i, pair in enumerate(pairs):
        offset = (i - (len(pairs) - 1) / 2) * 0.12
        color, marker = SEED_COLORS[i % len(SEED_COLORS)], CLASS_MARKER[pair["zone_class"]]
        ax.plot(x + offset, [pair["I_s"][z]["b"] for z in src.ZONES], marker, ls="none", ms=4, color=color,
                label=f"seed {pair['seed']} ({pair['zone_class']})")
        if "I_s_projected" in pair:
            ax.plot(x + offset, [pair["I_s_projected"][z]["b"] for z in src.ZONES], marker, ls="none", ms=4,
                    mfc="none", color=color)
    for xi, zone in zip(x, src.ZONES):
        stats = aggregate_b.mean_ci95([p["I_s"][zone]["b"] for p in pairs])
        ax.errorbar(xi + 0.32, stats["mean"], yerr=[[stats["mean"] - stats["ci95"][0]], [stats["ci95"][1] - stats["mean"]]],
                    fmt="D", ms=3.5, color="black", capsize=2, lw=0.9, label="mean, 95 % CI" if xi == 0 else None)
    ax.axhline(0.0, color="gray", lw=0.5, zorder=0)
    ax.set_xticks(x, [src.ZONE_ABBR[z] for z in src.ZONES])
    ax.set_ylabel(r"$\hat b$ (bits)")
    ax.legend(loc="lower right", fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME, out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
