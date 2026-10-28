"""E7b -- Dose-response of the anti-oracle at k = 5: references whose look-ahead particles follow the conveyor with
probability p (0 to 1); directed share against the effective b. E6's natural agents (training image) overlaid."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E7b_anti_oracle_dose"
RUNS = ("results", "E7_dissociation", "dev_e4_cpu", "analysis", "t1_runs.csv")
POINTS = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
K = "5"


def main():
    runs = [r for r in C.read_csv(C.path(*RUNS))
            if r["model"] == "reference" and r["h_mode"] == "mean" and r["k"] == K]
    agents = sorted({r["agent"] for r in runs})
    ps = sorted({C.num(r["p"]) for r in runs})
    fig, ax = C.full()
    fig.subplots_adjust(left=0.09, right=0.78, top=0.8, bottom=0.2)
    C.title(fig, "Forcing the conveyor belief into a reference: behaviour falls with it",
            "Anti-oracle: look-ahead particles follow the conveyor with probability p (k = 5, 4 references)")
    B, D = [], []
    for a in agents:
        sel = sorted((r for r in runs if r["agent"] == a), key=lambda r: C.num(r["p"]))
        b = np.array([C.num(r["b_eff"]) for r in sel])
        d = np.array([C.num(r["directed_share"]) for r in sel])
        ax.plot(b, d, color=P.ANTI_ORACLE, lw=1.0, alpha=0.5, zorder=2)
        B.append(b)
        D.append(d)
    bm, dm = np.mean(B, axis=0), np.mean(D, axis=0)
    ax.plot(bm, dm, "-o", color=P.ANTI_ORACLE, lw=3.2, ms=7, zorder=4)
    for p, x, y in zip(ps, bm, dm):
        if p in (0.0, 0.6, 1.0):
            ax.annotate(f"p = {p:g}\n{y:.2f} at b {C.fmt(x)}", (x, y), xytext={0.0: (10, 12), 0.6: (18, -46)}.get(p, (60, -6)),
                        textcoords="offset points", fontsize=P.SIZE_ANNOT, color=P.TEXT, fontweight="bold")
    pts = [r for r in C.read_csv(C.path(*POINTS)) if C.truthy(r["valid"])]
    for role, col in (("exposed", P.EXPOSED), ("reference", P.REFERENCE)):
        sel = [r for r in pts if r["role"] == role]
        ax.scatter([C.num(r["b_step1"]) for r in sel], [C.num(r["directed_share"]) for r in sel], s=70,
                   color=col if role == "exposed" else P.BG, edgecolor=col, lw=1.8, alpha=0.75, zorder=3)
    C.chance_line(ax, 0.4, label="chance", where=-1.2)
    ax.set_xlim(-2.5, 0.25)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("b  (bits; effective b for the anti-oracle, b at step 1 for E6 agents)")
    ax.set_ylabel("directed share")
    C.hgrid(ax)
    fig.text(0.8, 0.66, "anti-oracle\n(references)", color=P.ANTI_ORACLE, fontsize=P.SIZE_ANNOT + 1,
             fontweight="bold")
    fig.text(0.8, 0.53, "E6 exposed\n(natural)", color=P.EXPOSED, fontsize=P.SIZE_ANNOT + 1, fontweight="bold")
    fig.text(0.8, 0.40, "E6 reference\n(natural)", color=P.REFERENCE, fontsize=P.SIZE_ANNOT + 1, fontweight="bold")
    C.footnote(fig, "Thin: each reference; thick: mean over the 4, p = 0, 0.1, …, 1. Effective b is computed on the "
                    "mixed kernel (not the same measure as the natural b). Sources: results/E7_dissociation/dev_e4_cpu/"
                    "analysis/t1_runs.csv; results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
