"""E6b (v2) -- Directed share vs b at step 1, one point per valid (agent, image, model): exposed filled vermilion,
reference open blue; marker = test image (circle training, square position only, triangle theme only, diamond neutral);
bars: bootstrap 95% interval; dashed: chance 2/5."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E6b_directed_vs_b"
SRC = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
MARK = {"training": "o", "position_only": "s", "theme_only": "^", "neutral": "D"}


def main():
    rows = [r for r in C.read_csv(C.path(*SRC)) if C.truthy(r["valid"])]
    fig, ax = C.full()
    C.chance(ax, 2 / 5)
    for role, col in (("exposed", P.EXPOSED), ("reference", P.REFERENCE)):
        for img, mk in MARK.items():
            sel = [r for r in rows if r["role"] == role and r["image"] == img]
            if not sel:
                continue
            x = np.array([C.num(r["b_step1"]) for r in sel])
            y = np.array([C.num(r["directed_share"]) for r in sel])
            lo = np.array([C.num(r["directed_share_lo"]) for r in sel])
            hi = np.array([C.num(r["directed_share_hi"]) for r in sel])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=col, lw=2, alpha=0.6)
            ax.scatter(x, y, marker=mk, s=150, color=col if role == "exposed" else "white", edgecolor=col,
                       linewidth=2.5, zorder=3 if role == "exposed" else 4)
    ax.set_xlim(-2.5, 0.35)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("b at step 1 (bits)")
    ax.set_ylabel("directed share")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
