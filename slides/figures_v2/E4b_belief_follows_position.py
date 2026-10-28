"""E4b (v2) -- b at step 1 vs distance of the probed colour to the quadrant's training colour (/D), valid probes only
(duplicated probe images counted once), binned means (0.1 D, >= 2 probes) and probes; old UD (vermilion), old CD
(blue), neutral (grey). Dotted: -log2 5."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E4b_belief_follows_position"
SRC = ("results", "E4_generalization_gradient", "final", "tables", "probes_b_step1.csv")
POSITIONS = [("ud", P.EXPOSED), ("cd", P.REFERENCE), ("neutral", P.ZONE_LINE["corridor"])]
BINS = np.round(np.arange(0, 1.0001, 0.1), 2)


def main():
    uniq = {(r["configuration"], r["image"], r["slot"]): r for r in C.read_csv(C.path(*SRC))
            if r["axis"] != "position_only"}
    fig, ax = C.full()
    C.theory(ax, -np.log2(5))
    rng = np.random.default_rng(1)  # jitter of the probe dots only
    for pos, col in POSITIONS:
        ok = [r for r in uniq.values() if r["position"] == pos and C.truthy(r["valid"])]
        x = np.array([C.num(r["dist_to_training_color_of_zone_over_D"]) for r in ok])
        b = np.array([C.num(r["b_step1"]) for r in ok])
        ax.scatter(x + rng.uniform(-0.008, 0.008, len(x)), b, s=28, color=col, alpha=0.4)
        idx = np.clip(np.digitize(x, BINS[1:-1]), 0, len(BINS) - 2)
        pts = [(x[idx == i].mean(), b[idx == i].mean()) for i in range(len(BINS) - 1) if (idx == i).sum() >= 2]
        ax.plot(*zip(*pts), "o-", color=col, lw=3.2)
    ax.set_xlabel("distance to training colour (/ D)")
    ax.set_ylabel("b at step 1 (bits)")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
