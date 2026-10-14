"""E1 Step B: accuracy and bias cancellation. (a) b by zone against n (whole zone: mean and across-seed standard
deviation over the 20 seeds), with each zone's predicted value. (b) Uncontrollable-noisy zone: the bias of I_hat in the
environment and in its control, and the error that remains in b, against n. Writes paper/figures/out/fig_e1_b_vs_n.pdf.

Source: experiments/E1_offline_4zones/artifacts/measurements.csv, through e1_sources.step_b() (report.py's filters
and functions; checked against artifacts/etape_b_report.txt).
"""

from __future__ import annotations

import common
import e1_sources as src

NAME = "fig_e1_b_vs_n"


def main() -> None:
    q = src.step_b()
    ns = q["budgets"]
    common.apply_style()
    fig, (ax_b, ax_bias) = common.figure(aspect=0.42, ncols=2)
    for zone in src.ZONES:
        color = common.ZONE_COLOR[zone]
        ax_b.errorbar(ns, [q["b"][zone][n]["mean"] for n in ns], yerr=[q["b"][zone][n]["sd"] for n in ns], marker="o",
                      ms=3, lw=0.9, capsize=2, color=color, label=src.ZONE_ABBR[zone])
        ax_b.axhline(q["predicted_b"][zone], color=color, lw=0.6, ls="--")
    ax_b.set_xscale("log")
    ax_b.set_xlabel("$n$ (draws per pair)")
    ax_b.set_ylabel(r"$\hat b$ (bits)")
    ax_b.set_title("(a) whole zone; dashed: predicted", fontsize=common.SMALL_FONT_SIZE)
    ax_b.legend(fontsize=common.SMALL_FONT_SIZE - 1, loc="center right")

    series = ((lambda n: q["bias"][n]["exp"], r"bias of $\hat I$", common.TOL_BRIGHT["red"], "o"),
              (lambda n: q["bias"][n]["ctrl"], r"bias of $\hat I_{\mathrm{ref}}$", common.TOL_BRIGHT["blue"], "s"),
              (lambda n: q["bias"][n]["residual"], r"$|\hat b - b^*|$", "black", "D"))
    for get, label, color, marker in series:
        ax_bias.plot(ns, [get(n) for n in ns], marker=marker, ms=3, lw=0.9, color=color, label=label)
    ax_bias.set_xscale("log")
    ax_bias.set_xlabel("$n$ (draws per pair)")
    ax_bias.set_ylabel("bits")
    ax_bias.set_title("(b) UN zone", fontsize=common.SMALL_FONT_SIZE)
    ax_bias.legend(fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
