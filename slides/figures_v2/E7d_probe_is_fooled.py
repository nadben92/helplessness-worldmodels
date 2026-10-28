"""E7d (v2) -- Delta b vs Delta C1 (main z-bar): correction A, one point per (agent, setting) with 95% intervals
(d7_correction.csv; p1-b ringed in black); induction B, one point per run (data_v2/e7d_induction_runs.csv, extracted
read-only by _extract_e7d_induction.py; directed B runs filled, random-direction controls open). Dotted: the slope
between the natural groups (d7_bins.csv)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7d_probe_is_fooled"
D7 = ("results", "E7_dissociation", "d7", "analysis")
B_RUNS = ("slides", "data_v2", "e7d_induction_runs.csv")
ZBAR = "draw"


def natural_slope():
    bins = C.read_csv(C.path(*D7, "d7_bins.csv"))

    def grp(name):
        sel = [r for r in bins if r["zbar"] == ZBAR and r["group"] == name and r["measure"] == "C1"
               and not C.truthy(r["masked"])]
        w = np.array([C.num(r["n_steps"]) for r in sel])
        return (np.average([C.num(r["b_lo"]) + 0.125 for r in sel], weights=w),
                np.average([C.num(r["value"]) for r in sel], weights=w))
    (be, ce), (br, cr) = grp("exposed"), grp("references, no intervention")
    return (cr - ce) / (br - be)


def main():
    corr = [r for r in C.read_csv(C.path(*D7, "d7_correction.csv")) if r["zbar"] == ZBAR]
    brun = C.read_csv(C.path(*B_RUNS))
    fig, ax = C.full()
    x = np.array([-2.0, 2.0])
    ax.plot(x, natural_slope() * x, color=P.THEORY, ls=P.THEORY_LS, lw=2.4)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    for r in brun:
        rand = r["mode"] == "reverse_random"
        ax.scatter(C.num(r["delta_b"]), C.num(r["delta_C1"]), s=90, marker="D",
                   color="white" if rand else P.INDUCTION, edgecolor=P.INDUCTION, linewidth=2, zorder=3)
    for r in corr:
        db, dc = C.num(r["db"]), C.num(r["dC1"])
        ax.errorbar(db, dc, xerr=[[db - C.num(r["db_lo"])], [C.num(r["db_hi"]) - db]],
                    yerr=[[dc - C.num(r["dC1_lo"])], [C.num(r["dC1_hi"]) - dc]], fmt="none", ecolor=P.CORRECTION,
                    lw=2, alpha=0.5)
        p1b = r["agent"].startswith("p1-b")
        ax.scatter(db, dc, s=170 if p1b else 120, color=P.CORRECTION, edgecolor="black" if p1b else P.CORRECTION,
                   linewidth=2.5, zorder=4)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Δb (bits)")
    ax.set_ylabel("ΔC1")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
