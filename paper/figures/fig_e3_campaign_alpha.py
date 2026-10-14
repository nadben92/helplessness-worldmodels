"""E3 final campaign, secondary analysis: alpha, the weight of an action-independent component in the predicted kernel
(zone mean), in the two controllable zones, environment and reference, at every checkpoint measured in both runs of a
pair. One line per run. PENDING until the campaign is extracted. Writes paper/figures/out/fig_e3_campaign_alpha.pdf.

Source: the campaign's run directories, through experiments/E3_worldmodel_4zones/campaign_analysis.analyse
(secondary[].checkpoints[].alpha; measures.kernel_probe.action_independent_weight).
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_campaign_alpha"
ZONES = ("controllable_noisy", "controllable_deterministic")
SEED_COLORS = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"], common.TOL_BRIGHT["purple"])


def main() -> None:
    result, out_dir = src.campaign_analysis(__doc__.splitlines()[0])
    pairs = sorted(result["secondary"], key=lambda p: p["seed"])
    common.apply_style()
    fig, axes = common.figure(aspect=0.42, ncols=2, sharey=True)
    for i, pair in enumerate(pairs):
        steps = [c["env_step"] for c in pair["checkpoints"]]
        color = SEED_COLORS[i % len(SEED_COLORS)]
        for ax, zone in zip(axes, ZONES):
            for role, ls in (("env", "-"), ("ref", "--")):
                ax.plot(steps, [c["alpha"][zone][role] for c in pair["checkpoints"]], ls=ls, marker="o", ms=2.5, lw=0.9,
                        color=color, label=f"seed {pair['seed']} {role}")
    for ax, zone in zip(axes, ZONES):
        ax.set_title(src.ZONE_ABBR[zone], fontsize=common.SMALL_FONT_SIZE)
        ax.set_xscale("log")
        ax.set_xlabel("environment steps")
        ax.axhline(0.0, color="gray", lw=0.5, zorder=0)
    axes[0].set_ylabel(r"$\alpha$ (zone mean)")
    axes[1].legend(fontsize=common.SMALL_FONT_SIZE - 2, ncols=2)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME, out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
