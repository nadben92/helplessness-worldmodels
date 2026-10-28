"""E7c (v2) -- Static share of directed greedy choices (100 zone contexts, d4_static.csv) vs look-ahead k, one measure
for all four variants: reference, exposed + oracle, exposed + latent correction, exposed (own model); mean of 4 agents,
band = range; dashed: chance."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7c_lookahead"
SRC = ("results", "E7_latent_manipulation", "dev_e4_cpu_diag", "analysis", "d4_static.csv")
KS = [1, 2, 3, 5, 10]
SERIES = [("reference", "model", P.REFERENCE), ("exposed", "oracle", P.ORACLE),
          ("exposed", "model_corrected", P.CORRECTION), ("exposed", "model", P.EXPOSED)]


def main():
    st = C.read_csv(C.path(*SRC))
    fig, ax = C.full()
    C.chance(ax, 0.4)
    for model, variant, col in SERIES:
        v = np.array([[C.num(r["greedy_directed"]) for r in st if r["model"] == model and r["variant"] == variant
                       and int(r["k"]) == k] for k in KS])
        ax.fill_between(KS, v.min(1), v.max(1), color=col, alpha=0.2, lw=0)
        ax.plot(KS, v.mean(1), "o-", color=col, lw=3.2)
    ax.set_xscale("log")
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    ax.set_ylim(0.25, 1.04)
    ax.set_xlabel("look-ahead k")
    ax.set_ylabel("directed share")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
