"""E3 final campaign, secondary analysis (descriptive): the difference of differences
[I_WM - I_twin]_env - [I_WM - I_twin]_ref = b_WM - b_twin by zone at the budget (300,000 environment steps), with its
two parts, b_WM (the world model's b on I_s) and b_twin (the tabular twin's, same counts). One marker per seed and
quantity. PENDING until the campaign is extracted. Writes paper/figures/out/fig_e3_campaign_did.pdf.

Source: the campaign's run directories, through experiments/E3_worldmodel_4zones/campaign_analysis.analyse
(secondary[].checkpoints[].difference_of_differences and .decomposition).
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

NAME = "fig_e3_campaign_did"
QUANTITIES = (("b_wm", r"$\hat b_{\mathrm{WM}}$", common.TOL_BRIGHT["blue"]),
              ("b_twin", r"$\hat b_{\mathrm{twin}}$", common.TOL_BRIGHT["grey"]),
              ("did", "difference of differences", common.TOL_BRIGHT["red"]))
CLASS_MARKER = {"A": "o", "B": "s"}


def main() -> None:
    result, out_dir = src.campaign_analysis(__doc__.splitlines()[0])
    analysis = src.e3_module("campaign_analysis.py")
    pairs = sorted(result["secondary"], key=lambda p: p["seed"])
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.85, aspect=0.6)
    x = np.arange(len(src.ZONES))
    labelled = set()
    for i, pair in enumerate(pairs):
        cp = src.budget_checkpoint(pair, analysis.BUDGET_ENV_STEPS)
        for j, (key, label, color) in enumerate(QUANTITIES):
            values = [cp["difference_of_differences"][z]["did"] if key == "did" else cp["decomposition"][z][key] for z in src.ZONES]
            offset = (j - 1) * 0.25 + (i - (len(pairs) - 1) / 2) * 0.05
            ax.plot(x + offset, values, CLASS_MARKER[pair["zone_class"]], ls="none", ms=3.5, color=color,
                    label=label if key not in labelled else None)
            labelled.add(key)
    ax.axhline(0.0, color="gray", lw=0.5, zorder=0)
    ax.set_xticks(x, [src.ZONE_ABBR[z] for z in src.ZONES])
    ax.set_ylabel("bits")
    ax.legend(loc="lower right", fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME, out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
