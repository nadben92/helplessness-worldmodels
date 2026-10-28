"""E4d (v2) -- b in the old UD quadrant by step since entry (training image): 4 configurations (thin), mean (thick);
dotted: -log2 5."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E4d_recovery_in_episode"
SRC = ("results", "E4_generalization_gradient", "final", "tables", "recovery_curves_extremes.csv")


def main():
    curves = [r for r in C.read_csv(C.path(*SRC)) if r["probe"] == "training" and r["zone_role"] == "ud"]
    k = np.arange(1, 26)
    B = np.array([[C.num(r[f"b_k{i}"]) for i in k] for r in curves])
    fig, ax = C.full()
    C.theory(ax, -np.log2(5))
    for b in B:
        ax.plot(k, b, color=P.EXPOSED, lw=1.3, alpha=0.5)
    ax.plot(k, B.mean(axis=0), color=P.EXPOSED, lw=3.4)
    ax.set_xlim(0.5, 25.5)
    ax.set_xlabel("step in quadrant")
    ax.set_ylabel("b (bits)")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
