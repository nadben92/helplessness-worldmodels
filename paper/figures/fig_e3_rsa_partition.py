"""E3 final campaign, RSA of the latent: partition of the variance of the state-level model distances explained by the
theoretical matrices, at the budget (300,000 environment steps), mean over the 4 seeds, for h and z, environment and
reference. Each bar stacks the unique share of each predictor (position, theme, stochasticity, controllability) and the
shared share; its height is the explained variance R^2. In the reference the controllability predictor is constant and
removed before the regression (DECISIONS (32)). Writes paper/figures/out/fig_e3_rsa_partition.pdf.

Source: docs/e3/campaign_tables/analysis.json (the campaign pipeline's own output, copied from
results/E3_worldmodel_4zones/campaign/), rsa_over_seeds.300000.<latent>.partition, plotted as recorded.
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

NAME = "fig_e3_rsa_partition"
CHECKPOINT = "300000"
PARTS = (("position", common.TOL_BRIGHT["blue"]), ("theme", common.TOL_BRIGHT["yellow"]),
         ("stochasticity", common.TOL_BRIGHT["green"]), ("controllability", common.TOL_BRIGHT["red"]),
         ("shared", common.TOL_BRIGHT["grey"]))
BARS = (("h", "experimental", "h, env"), ("h", "reference", "h, ref"), ("z", "experimental", "z, env"),
        ("z", "reference", "z, ref"))


def main() -> None:
    rsa = src.read_json(src.CAMPAIGN_JSON)["rsa_over_seeds"]
    if CHECKPOINT not in rsa:
        raise SystemExit(f"{src.CAMPAIGN_JSON}: no rsa_over_seeds.{CHECKPOINT}")
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.75, aspect=0.6)
    x = np.arange(len(BARS))
    for i, (latent, run, _) in enumerate(BARS):
        part = rsa[CHECKPOINT][latent]["partition"][run]
        bottom = 0.0
        for name, color in PARTS:
            value = part["shared"] if name == "shared" else part["unique"].get(name)
            if value is None:  # the reference's controllability predictor: removed (constant)
                continue
            ax.bar(x[i], value, bottom=bottom, color=color, width=0.6, label=(name if name == "shared" else f"unique: {name}") if i == 0 else None)
            bottom += value
    ax.set_xticks(x, [label for _, _, label in BARS])
    ax.set_ylabel(r"explained variance ($R^2$)")
    ax.legend(fontsize=common.SMALL_FONT_SIZE - 1, loc="upper right")
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
