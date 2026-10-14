"""E1 Step B, detectability in the controllable-noisy zone (where H0: b = 0 is true): the threshold, the 95th
percentile of |b| under the bootstrap null (mean over the 20 seeds), against n in log-log, with its fitted power law;
the across-seed standard deviation of the observed b; and a slope
-0.5 reference anchored at n = 10. Writes paper/figures/out/fig_e1_detectability.pdf.

Source: experiments/E1_offline_4zones/artifacts/permutation_null.csv and measurements.csv, through
e1_sources.step_b() and report.py's fit_power_law_slope (checked against etape_b_report.txt).
"""

from __future__ import annotations

import numpy as np

import common
import e1_sources as src

NAME = "fig_e1_detectability"
ZONE = "controllable_noisy"


def main() -> None:
    q = src.step_b()
    rep = src.report()
    ns = np.array(q["budgets"], dtype=float)
    p95 = np.array([q["p95"][ZONE][n] for n in q["budgets"]])
    sd_b = np.array([q["b"][ZONE][n]["sd"] for n in q["budgets"]])
    slope, intercept = rep.fit_power_law_slope(list(q["budgets"]), list(p95))
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.75, aspect=0.65)
    ax.plot(ns, p95, "o", ms=4, color=common.TOL_BRIGHT["red"], label="threshold, bootstrap null")
    ax.plot(ns, np.exp(intercept) * ns ** slope, "-", lw=0.9, color=common.TOL_BRIGHT["red"],
            label=f"fitted slope {slope:.3f}")
    ax.plot(ns, p95[0] * (ns / ns[0]) ** -0.5, ":", lw=0.9, color="black", label="slope $-0.5$")
    ax.plot(ns, sd_b, "^", ms=3.5, color=common.TOL_BRIGHT["green"], label=r"sd of observed $\hat b$ over seeds")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$ (draws per pair)")
    ax.set_ylabel("bits")
    ax.legend(fontsize=common.SMALL_FONT_SIZE - 1, loc="lower left")
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
