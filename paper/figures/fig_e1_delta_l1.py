"""E1 Step B: deciding against modelling. (a) Delta by zone against n (mean over the 20 seeds of the zone's interior
states, 9 per zone); (b) the L1 error of T_hat against P by zone (whole zone). Two separate axes: Delta is a value, L1
a probability mass. Writes paper/figures/out/fig_e1_delta_l1.pdf.

Source: experiments/E1_offline_4zones/artifacts/measurements.csv, through e1_sources.step_b() (report.py's
delta_l1_reading filters: delta on the interior region, l1 on the whole zone; checked against etape_b_report.txt).
"""

from __future__ import annotations

import common
import e1_sources as src

NAME = "fig_e1_delta_l1"


def main() -> None:
    q = src.step_b()
    sizes = src.region_sizes()
    ns = q["budgets"]
    common.apply_style()
    fig, (ax_d, ax_l) = common.figure(aspect=0.42, ncols=2)
    for zone in src.ZONES:
        color = common.ZONE_COLOR[zone]
        ax_d.plot(ns, [q["delta"][zone][n] for n in ns], marker="o", ms=3, lw=0.9, color=color, label=src.ZONE_ABBR[zone])
        ax_l.plot(ns, [q["l1"][zone][n] for n in ns], marker="o", ms=3, lw=0.9, color=color, label=src.ZONE_ABBR[zone])
    for ax in (ax_d, ax_l):
        ax.set_xscale("log")
        ax.set_xlabel("$n$ (draws per pair)")
    ax_d.set_ylabel(r"$\Delta$ (value)")
    ax_d.set_title(f"(a) interior states ({sizes['interior']} per zone)", fontsize=common.SMALL_FONT_SIZE)
    ax_l.set_ylabel(r"$L_1(\hat T, P)$")
    ax_l.set_title(f"(b) whole zone ({sizes['zone']} states)", fontsize=common.SMALL_FONT_SIZE)
    ax_d.legend(fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
