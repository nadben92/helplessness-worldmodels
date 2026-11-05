"""E7a (v2) -- The 2 x 2 (mean of 4 agents, table_2x2.csv): rows = belief (true, conveyor), columns = critic
(reference, exposed); left k = 1, right k = 5; colour = directed share (thin bar)."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7a_belief_critic_2x2"
SRC = ("results", "E7_dissociation", "dev_e4_cpu", "analysis", "table_2x2.csv")
CELLS = {(0, 0): "true_belief__ref_critic", (0, 1): "true_belief__exp_critic (oracle D4)",
         (1, 0): "conveyor__ref_critic (anti-oracle p=1)", (1, 1): "conveyor__exp_critic"}


def main():
    rows = C.read_csv(C.path(*SRC))
    cmap = LinearSegmentedColormap.from_list("d", ["white", P.REFERENCE])
    norm = Normalize(0.0, 1.0)
    fig, axes = C.full(1, 2)
    for ax, k in zip(axes, ("1", "5")):
        mean = next(r for r in rows if r["k"] == k and r["agent"] == "mean")
        img = np.array([[C.num(mean[CELLS[(i, j)]]) for j in range(2)] for i in range(2)])
        im = ax.imshow(img, cmap=cmap, norm=norm)
        ax.set_xticks([0, 1], ["reference\ncritic", "exposed\ncritic"])
        ax.set_yticks([0, 1], ["true\nbelief", "conveyor\nbelief"] if k == "1" else ["", ""])
        ax.set_xlabel(f"k = {k}")
    cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.03)
    cb.set_label("directed share", fontsize=P.SIZE_LABEL)
    cb.ax.tick_params(labelsize=P.SIZE_TICK)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
