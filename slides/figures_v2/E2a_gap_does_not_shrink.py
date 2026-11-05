"""E2a (v2) -- Online - matched, zone CN, 16 checkpoints, 20 seeds: |mean paired difference| of b-hat (left) and of
the L1 model error (right), 95% interval band (clipped at the floor where it crosses 0), E2's power-law fit (black)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E2a_gap_does_not_shrink"
CSV = ("experiments", "E2_online_4zones", "artifacts", "figures", "csv")
FLOOR = 3e-4


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
    fig, axes = C.full(1, 2, sharex=True)
    for ax, (q, ylab, col) in zip(axes, (("b", "|Δb̂| online − matched (bits)", P.EXPOSED),
                                          ("l1", "|ΔL1| online − matched", P.REFERENCE))):
        x, m, lo, hi = series(rows, q)
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, np.maximum(m, FLOOR), "o-", color=col)
        xf = np.geomspace(1e4, 1e6, 20)
        ax.plot(xf, 10 ** C.num(fits[q]["intercept"]) * xf ** C.num(fits[q]["exponent"]), color="black", lw=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR * 0.9, 0.1)
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylab)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
