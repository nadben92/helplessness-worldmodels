"""E3 final campaign, secondary analysis: the zones' separation in the latent, held-out accuracy of a linear probe
latent -> zone (4 categories) and latent -> controllable vs uncontrollable category, environment and reference (in the
reference every zone is controllable: the probe separates appearances only, a control), at every checkpoint measured
in both runs of a pair. PENDING until the campaign is extracted. Writes paper/figures/out/fig_e3_campaign_separation.pdf.

Source: the campaign's run directories, through experiments/E3_worldmodel_4zones/campaign_analysis.analyse
(secondary[].checkpoints[].latent_zone_separation: recorded in the run on h and z).
"""

from __future__ import annotations

import common
import e3_sources as src

NAME = "fig_e3_campaign_separation"
PROBES = (("probe_zone", "(a) zone probe"), ("probe_controllability", "(b) controllability probe"))
SEED_COLORS = (common.TOL_BRIGHT["blue"], common.TOL_BRIGHT["red"], common.TOL_BRIGHT["green"], common.TOL_BRIGHT["purple"])


def main() -> None:
    result, out_dir = src.campaign_analysis(__doc__.splitlines()[0])
    pairs = sorted(result["secondary"], key=lambda p: p["seed"])
    common.apply_style()
    fig, axes = common.figure(aspect=0.42, ncols=2, sharey=True)
    for i, pair in enumerate(pairs):
        steps = [c["env_step"] for c in pair["checkpoints"]]
        for role, ls in (("experimental", "-"), ("reference", "--")):
            seps = [c["latent_zone_separation"][role] for c in pair["checkpoints"]]
            if any(s is None for s in seps):
                raise SystemExit(f"seed {pair['seed']} {role}: the zones' separation is missing at a checkpoint")
            for ax, (probe, _) in zip(axes, PROBES):
                ax.plot(steps, [s[probe]["test_accuracy"] for s in seps], ls=ls, marker="o", ms=2.5, lw=0.9,
                        color=SEED_COLORS[i % len(SEED_COLORS)], label=f"seed {pair['seed']} {src.CONDITION_ABBR[role]}")
    for ax, (_, title) in zip(axes, PROBES):
        ax.set_title(title, fontsize=common.SMALL_FONT_SIZE)
        ax.set_xscale("log")
        ax.set_xlabel("environment steps")
    axes[0].set_ylabel("held-out accuracy")
    axes[1].legend(fontsize=common.SMALL_FONT_SIZE - 2, ncols=2)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME, out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
