"""E1 Step B, the recommended n. Uncontrollable-noisy zone: each seed's error |b - b*| against n (dots), the
across-seed standard deviation of b (squares), the fitted A/sqrt(n) curve and its k-sigma versions (k = 2, 3) against
the 0.03-bit tolerance; vertical marks at the fitted crossings. Writes paper/figures/out/fig_e1_precision.pdf.

Source: experiments/E1_offline_4zones/artifacts/measurements.csv, through e1_sources.step_b() (report.py's
across_seed_sd, fit_inverse_sqrt_n, PRECISION_TOL, SAFETY_MARGIN_SIGMAS; checked against etape_b_report.txt).
"""

from __future__ import annotations

import numpy as np

import common
import e1_sources as src

NAME = "fig_e1_precision"
ZONE = "uncontrollable_noisy"


def main() -> None:
    q = src.step_b()
    ns = q["budgets"]
    prec = q["precision"][ZONE]
    pred = q["predicted_b"][ZONE]
    common.apply_style()
    fig, ax = common.figure(width_fraction=1.0, aspect=0.5)
    rng = np.random.default_rng(0)  # horizontal jitter of the dots only (display)
    for n in ns:
        errs = np.abs(np.array(src.zone_values(ZONE, n, "b")) - pred)
        ax.plot(n * np.exp(rng.uniform(-0.08, 0.08, errs.size)), errs, ".", ms=3, color=common.TOL_BRIGHT["grey"],
                label="each seed, $|\\hat b - b^*|$" if n == ns[0] else None)
    ax.plot(ns, [prec["sd"][n] for n in ns], "s", ms=4, color=common.TOL_BRIGHT["blue"], label=r"sd of $\hat b$ over seeds")
    grid = np.geomspace(ns[0], ns[-1], 100)
    ax.plot(grid, prec["A"] / np.sqrt(grid), "-", lw=0.9, color=common.TOL_BRIGHT["blue"], label=r"fitted $A/\sqrt{n}$")
    for k, ls in zip(q["sigmas"], ("--", ":")):
        ax.plot(grid, k * prec["A"] / np.sqrt(grid), ls, lw=0.9, color=common.TOL_BRIGHT["red"], label=f"{k} sd")
        ax.axvline(q["n_needed"][k], color=common.TOL_BRIGHT["red"], lw=0.5, ls=ls)
    ax.axhline(q["tolerance"], color="black", lw=0.8, label="tolerance")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$ (draws per pair)")
    ax.set_ylabel("bits")
    ax.legend(fontsize=common.SMALL_FONT_SIZE - 1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
