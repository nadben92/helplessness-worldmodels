"""E7d -- "The probe is fooled" (D7): for each latent-correction run (A, 10 agent x setting pairs, main z-bar), the
gain in b against the gain in C1 (controllability read by the critic at a common z: share of directed argmax of
V(h'_a, z-bar)), with 95% intervals. Induced references (B) are pooled by b class (the per-run values are not in the
light outputs): Delta b = class centre - 0, Delta C1 = class C1 - C1 of the references without intervention near
b = 0. Expected line: the slope between the natural groups (exposed vs references)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E7d_probe_is_fooled"
D = ("results", "E7_dissociation", "d7", "analysis")
ZBAR = "draw"
HALF_BIN = 0.125


def group(bins, name):
    sel = [r for r in bins if r["zbar"] == ZBAR and r["group"] == name and r["measure"] == "C1"
           and not C.truthy(r["masked"])]
    b = np.array([C.num(r["b_lo"]) + HALF_BIN for r in sel])
    v = np.array([C.num(r["value"]) for r in sel])
    n = np.array([C.num(r["n_steps"]) for r in sel])
    return b, v, n


def main():
    bins = C.read_csv(C.path(*D, "d7_bins.csv"))
    corr = [r for r in C.read_csv(C.path(*D, "d7_correction.csv")) if r["zbar"] == ZBAR]
    be, ve, ne = group(bins, "exposed")
    br, vr, nr = group(bins, "references, no intervention")
    bi, vi, _ = group(bins, "induced references")
    b_exp, c_exp = np.average(be, weights=ne), np.average(ve, weights=ne)
    b_ref, c_ref = np.average(br, weights=nr), np.average(vr, weights=nr)
    slope = (c_ref - c_exp) / (b_ref - b_exp)
    c0 = vr[np.argmin(np.abs(br))]

    fig, ax = C.full()
    fig.subplots_adjust(left=0.1, right=0.74, top=0.8, bottom=0.21)
    C.title(fig, "The probe is fooled: b moves, the critic's controllability does not",
            "Latent manipulation in h: gain in b vs gain in controllability read by the critic (C1)")
    x = np.linspace(-2.4, 2.0, 2)
    ax.plot(x, slope * x, color=P.THEORY, ls=P.THEORY_LS, lw=1.8, zorder=1)
    ax.text(1.55, slope * 1.55 + 0.05, f"natural agents\n(slope {slope:.2f} per bit)", fontsize=P.SIZE_SMALL,
            color=P.THEORY, ha="right", va="bottom")
    ax.axhline(0, color=P.AXIS, lw=0.9)
    ax.axvline(0, color=P.AXIS, lw=0.9)
    for r in corr:
        db, dc = C.num(r["db"]), C.num(r["dC1"])
        real = r["agent"].startswith("p1-b")
        col = P.CORRECTION
        ax.errorbar(db, dc, xerr=[[db - C.num(r["db_lo"])], [C.num(r["db_hi"]) - db]],
                    yerr=[[dc - C.num(r["dC1_lo"])], [C.num(r["dC1_hi"]) - dc]], fmt="none", ecolor=col,
                    alpha=P.ALPHA_CI, lw=1.5, zorder=2)
        ax.scatter(db, dc, s=110 if real else 80, color=col, edgecolor=P.TEXT if real else col, lw=1.8 if real else 0,
                   zorder=4)
    p1b = [r for r in corr if r["agent"].startswith("p1-b")]
    xb, yb = np.mean([C.num(r["db"]) for r in p1b]), np.mean([C.num(r["dC1"]) for r in p1b])
    ax.annotate("p1-b: the only real change", (xb, yb), xytext=(-330, 30), textcoords="offset points",
                fontsize=P.SIZE_ANNOT, color=P.TEXT, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=P.TEXT_2, lw=1.2))
    ax.annotate("3 of 4 agents: b +1.2 to +1.8 bit,\ncritic unchanged", (1.6, -0.1), xytext=(-40, -70),
                textcoords="offset points", fontsize=P.SIZE_ANNOT, color=P.CORRECTION, fontweight="bold", ha="right")
    ax.scatter(bi, vi - c0, marker="D", s=60, color=P.INDUCTION, zorder=3)
    ax.set_xlim(-2.4, 2.0)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel("Δb  (bits)")
    ax.set_ylabel("ΔC1  (critic's directed share)")
    C.hgrid(ax)
    fig.text(0.76, 0.66, "correction A\n(exposed, 10 runs)", color=P.CORRECTION, fontsize=P.SIZE_ANNOT + 1,
             fontweight="bold")
    fig.text(0.76, 0.52, "induction B\n(references, pooled\nby b class)", color=P.INDUCTION,
             fontsize=P.SIZE_ANNOT + 1, fontweight="bold")
    C.footnote(fig, "C1: share of directed argmax of V(h′ₐ, z̄), z̄ drawn from the mixture of the 5 priors (D6/D7). "
                    "Bars: 95% intervals over episodes. Expected line through the natural exposed (b ≈ "
                    f"{C.fmt(b_exp)}, C1 {c_exp:.2f}) and references (b ≈ {C.fmt(b_ref)}, C1 {c_ref:.2f}). Source: "
                    "results/E7_dissociation/d7/analysis/d7_correction.csv, d7_bins.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
