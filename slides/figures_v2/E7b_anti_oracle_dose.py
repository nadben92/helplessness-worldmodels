"""E7b (v2) -- Anti-oracle at k = 5 (4 references, p = 0 .. 1): directed share vs effective b (thin per agent, thick
mean, orange), with E6's natural agents (exposed filled vermilion, references open blue); dashed: chance."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7b_anti_oracle_dose"
RUNS = ("results", "E7_dissociation", "dev_e4_cpu", "analysis", "t1_runs.csv")
POINTS = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")


def main():
    runs = [r for r in C.read_csv(C.path(*RUNS)) if r["model"] == "reference" and r["h_mode"] == "mean"
            and r["k"] == "5"]
    fig, ax = C.full()
    C.chance(ax, 0.4)
    B, Y = [], []
    for a in sorted({r["agent"] for r in runs}):
        sel = sorted((r for r in runs if r["agent"] == a), key=lambda r: C.num(r["p"]))
        B.append([C.num(r["b_eff"]) for r in sel])
        Y.append([C.num(r["directed_share"]) for r in sel])
        ax.plot(B[-1], Y[-1], color=P.ANTI_ORACLE, lw=1.3, alpha=0.55)
    ax.plot(np.mean(B, 0), np.mean(Y, 0), "o-", color=P.ANTI_ORACLE, lw=3.4)
    pts = [r for r in C.read_csv(C.path(*POINTS)) if C.truthy(r["valid"])]
    for role, col in (("exposed", P.EXPOSED), ("reference", P.REFERENCE)):
        sel = [r for r in pts if r["role"] == role]
        ax.scatter([C.num(r["b_step1"]) for r in sel], [C.num(r["directed_share"]) for r in sel], s=110,
                   color=col if role == "exposed" else "white", edgecolor=col, linewidth=2.5, zorder=4)
    ax.set_xlim(-2.5, 0.25)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("b (bits)")
    ax.set_ylabel("directed share")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
