"""E4b -- Main E4 result: b at step 1 against the distance of the probed colour to the training colour of the probed
quadrant (/D), one curve per position, valid probes only (4 configurations, 3 axes; duplicated probe images counted
once). Binned means with the individual probes; share of valid probes per position."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E4b_belief_follows_position"
SRC = ("results", "E4_generalization_gradient", "final", "tables", "probes_b_step1.csv")
POSITIONS = [("ud", "old UD quadrant", P.EXPOSED), ("cd", "old CD quadrant", P.REFERENCE),
             ("neutral", "neutral quadrant", P.ZONE_LINE["corridor"])]
BINS = np.round(np.arange(0, 1.0001, 0.1), 2)


def main():
    rows = C.read_csv(C.path(*SRC))
    uniq = {}
    for r in rows:
        if r["axis"] == "position_only":
            continue
        uniq[(r["configuration"], r["image"], r["slot"])] = r
    fig, ax = C.full()
    fig.subplots_adjust(left=0.09, right=0.78, top=0.8, bottom=0.16)
    C.title(fig, "The belief is tied to position, not colour",
            "b at the first step in the probed quadrant, frozen models, every zone controllable")
    rng = np.random.default_rng(1)  # jitter of the probe dots only
    for pos, name, col in POSITIONS:
        sel = [r for r in uniq.values() if r["position"] == pos]
        ok = [r for r in sel if C.truthy(r["valid"])]
        x = np.array([C.num(r["dist_to_training_color_of_zone_over_D"]) for r in ok])
        b = np.array([C.num(r["b_step1"]) for r in ok])
        ax.scatter(x + rng.uniform(-0.008, 0.008, len(x)), b, s=16, color=col, alpha=0.35, lw=0, zorder=2)
        idx = np.clip(np.digitize(x, BINS[1:-1]), 0, len(BINS) - 2)
        xs, ms = [], []
        for i in range(len(BINS) - 1):
            m = idx == i
            if m.sum() >= 2:
                xs.append(x[m].mean())
                ms.append(b[m].mean())
        ax.plot(xs, ms, "-o", color=col, lw=3, ms=6, zorder=4)
        C.end_label(ax, xs[-1], ms[-1], name, col, dy=10 if pos == "cd" else (-26 if pos == "neutral" else 0))
        ax.text(1.02, {"ud": 0.38, "cd": 0.31, "neutral": 0.24}[pos],
                f"{name}: {len(ok)}/{len(sel)} valid", transform=ax.transAxes, fontsize=P.SIZE_SMALL, color=col)
    C.theory_line(ax, -np.log2(5))
    ax.text(0.005, -np.log2(5) - 0.08, "−log₂5 (full conveyor belief)", fontsize=P.SIZE_SMALL, color=P.THEORY,
            va="top")
    ax.axhline(0, color=P.AXIS, lw=0.9, zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-2.55, 0.35)
    ax.set_xlabel("distance of the probed colour to the quadrant's training colour  (/ D)")
    ax.set_ylabel("b at step 1  (bits)")
    C.hgrid(ax)
    ax.text(1.02, 0.45, "probes kept", transform=ax.transAxes, fontsize=P.SIZE_SMALL, color=P.TEXT_2,
            fontweight="bold")
    C.footnote(fig, "The neutral quadrant's training colour is the corridor's, so its probes start at 0.38 D. Valid probes only (reference mass on the true successor ≥ 0.90 and |I_ref − log₂5| ≤ 0.15 at step 1). "
                    "Lines: means in bins of 0.1 D (≥ 2 probes); dots: probes (4 configurations × 3 colour axes). "
                    "Source: results/E4_generalization_gradient/final/tables/probes_b_step1.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
