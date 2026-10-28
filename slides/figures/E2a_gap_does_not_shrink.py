"""E2a -- Online minus matched-counts control, zone CN, over the 16 checkpoints (alpha = 1, epsilon = 0.1, 20
seeds): |mean paired difference| of b-hat and of the model's L1 error, log-log, with E2's power-law fits (checkpoints
>= 1e4, table3) and their bootstrap intervals."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E2a_gap_does_not_shrink"
CSV = ("experiments", "E2_online_4zones", "artifacts", "figures", "csv")
FLOOR = 3e-4  # display floor of the log axis where a 95% interval crosses 0


def series(rows, q):
    sel = sorted((r for r in rows if r["quantity"] == q and r["zone"] == "controllable_noisy"),
                 key=lambda r: int(r["checkpoint"]))
    x = np.array([int(r["checkpoint"]) for r in sel], float)
    m = np.array([C.num(r["mean_diff"]) for r in sel])
    lo = np.array([C.num(r["ci95_low"]) for r in sel])
    hi = np.array([C.num(r["ci95_high"]) for r in sel])
    sign = np.sign(np.median(m))
    a, b = np.sort(np.stack([sign * lo, sign * hi]), axis=0)
    return x, sign * m, np.maximum(a, FLOOR), np.maximum(b, FLOOR)


def main():
    rows = C.read_csv(C.path(*CSV, "F2_null_model_paired_diff.csv"))
    fits = {r["quantity"]: r for r in C.read_csv(C.path(*CSV, "table3_budget_trend_power_law.csv"))}
    fig, axes = C.plt.subplots(1, 2, figsize=C.FULL, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.2, wspace=0.28)
    C.title(fig, "More data closes the gap in model error, not the bias in b",
            "Online agent minus matched-counts control, zone CN, 20 seeds, 16 checkpoints")
    spec = [("b", "|online − matched|  on  b̂  (bits)", P.EXPOSED, "bias in b̂"),
            ("l1", "|online − matched|  on  L1 error", P.REFERENCE, "gap in L1 error")]
    for ax, (q, ylab, color, name) in zip(axes, spec):
        x, m, lo, hi = series(rows, q)
        ax.fill_between(x, lo, hi, color=color, alpha=P.ALPHA_CI, lw=0)
        ax.plot(x, np.maximum(m, FLOOR), "o-", color=color, ms=6, zorder=4)
        f = fits[q]
        xf = np.geomspace(1e4, 1e6, 20)
        ax.plot(xf, 10 ** C.num(f["intercept"]) * xf ** C.num(f["exponent"]), color=P.TEXT, lw=2.6, zorder=5)
        e, lo, hi = (C.fmt(C.num(f[k]), sign=True)
                     for k in ("exponent", "exponent_bootstrap_ci95_low", "exponent_bootstrap_ci95_high"))
        ax.text(0.04, 0.95, f"{name}: exponent {e}\n[{lo}, {hi}]",
                transform=ax.transAxes, va="top", fontsize=P.SIZE_AXIS, color=P.TEXT, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.9, 0.15)
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylab)
        C.hgrid(ax)
    C.footnote(fig, "Dots: |mean paired difference|; band: 95% interval (clipped at the axis floor where it "
                    "crosses 0); black: power-law fit on checkpoints ≥ 10⁴ with bootstrap interval of the exponent "
                    "(E2 table 3). Source: experiments/E2_online_4zones/artifacts/figures/csv/F2_null_model_paired_diff.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
