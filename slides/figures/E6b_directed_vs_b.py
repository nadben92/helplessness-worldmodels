"""E6b -- Directed share against b at step 1, one point per (agent, image, model), valid pairs only (E4's four agents,
four test images). Chance = 2/5 (inside a quadrant two of the five actions bring the agent closer to the goal)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E6b_directed_vs_b"
SRC = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
CHANCE = 2 / 5
MARK = {"training": "o", "position_only": "s", "theme_only": "^", "neutral": "D"}
NAMES = {"training": "training image", "position_only": "position only", "theme_only": "theme only",
         "neutral": "neutral quadrant"}


def main():
    rows = [r for r in C.read_csv(C.path(*SRC)) if C.truthy(r["valid"])]
    fig, ax = C.full()
    fig.subplots_adjust(left=0.09, right=0.76, top=0.8, bottom=0.16)
    C.title(fig, "Where b is low, behaviour drops to chance",
            "Share of directed steps in the probed quadrant vs b at step 1, one point per (agent, image)")
    for role, col in (("exposed", P.EXPOSED), ("reference", P.REFERENCE)):
        for img, mk in MARK.items():
            sel = [r for r in rows if r["role"] == role and r["image"] == img]
            if not sel:
                continue
            x = np.array([C.num(r["b_step1"]) for r in sel])
            y = np.array([C.num(r["directed_share"]) for r in sel])
            lo = np.array([C.num(r["directed_share_lo"]) for r in sel])
            hi = np.array([C.num(r["directed_share_hi"]) for r in sel])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=col, alpha=P.ALPHA_CI, lw=1.5, zorder=2)
            ax.scatter(x, y, marker=mk, s=95, color=col if role == "exposed" else P.BG, edgecolor=col,
                       linewidth=2, zorder=3 if role == "exposed" else 4)
    C.chance_line(ax, CHANCE, label="chance (2 of 5 actions)", where=-1.6)
    exp_low = [r for r in rows if r["role"] == "exposed" and C.num(r["b_step1"]) < -1]
    high = [r for r in rows if not (r["role"] == "exposed" and C.num(r["b_step1"]) < -1)]
    for grp, xy in ((exp_low, (-1.45, 0.58)), (high, (-0.6, 0.78))):
        b = np.array([C.num(r["b_step1"]) for r in grp])
        d = np.array([C.num(r["directed_share"]) for r in grp])
        ax.annotate(f"b ≈ {C.fmt(b.mean())}, directed {d.min():.2f}–{d.max():.2f}\n(n = {len(grp)})", xy,
                    fontsize=P.SIZE_ANNOT, color=P.TEXT, ha="center", fontweight="bold")
    ax.set_xlim(-2.5, 0.35)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("b at step 1  (bits)")
    ax.set_ylabel("directed share")
    C.hgrid(ax)
    y = 0.95
    for role, col in (("exposed", P.EXPOSED), ("reference", P.REFERENCE)):
        fig.text(0.785, y - 0.17, role, fontsize=P.SIZE_ANNOT + 1, color=col, fontweight="bold",
                 transform=fig.transFigure)
        y -= 0.05
    yy = 0.62
    for img, mk in MARK.items():
        ax.scatter([], [], marker=mk, s=80, color=P.BG, edgecolor=P.TEXT_2, linewidth=1.8, label=NAMES[img])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, yy), fontsize=P.SIZE_SMALL + 0.5, handletextpad=0.4,
              labelcolor=P.TEXT)
    C.footnote(fig, "Filled: exposed model; open: reference (b = 0 by construction). Bars: bootstrap 95% interval over "
                    "episodes. Invalid pairs (p2-a position only, theme only) left out. E4's four agents, 180 episodes "
                    "per point. Source: results/E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
