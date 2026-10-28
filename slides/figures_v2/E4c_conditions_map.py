"""E4c (v2) -- b at step 1, 4 configurations x conditions (training image at old UD, position only, theme only,
training image at neutral, training image at old CD); invalid cells hatched; thin colour bar."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E4c_conditions_map"
T = ("results", "E4_generalization_gradient", "final", "tables")
CONFIGS = ["p1-a-d1", "p1-b-d1", "p2-a-d1", "p2-b-d1"]
COLS = ["UD", "position\nonly", "theme\nonly", "neutral", "CD"]


def main():
    vz = C.read_csv(C.path(*T, "validity_image_zone.csv"))
    to = {r["configuration"]: r for r in C.read_csv(C.path(*T, "theme_only_position_only.csv"))}
    cmap = LinearSegmentedColormap.from_list("b", [P.EXPOSED, "white"])
    norm = Normalize(-np.log2(5), 0)
    fig, ax = C.full()
    for i, cfg in enumerate(CONFIGS):
        tr = {r["role"]: r for r in vz if r["configuration"] == cfg and C.truthy(r["is_training"])}
        cells = [(tr["ud"]["b_step1"], tr["ud"]["valid"]),
                 (to[cfg]["position_only_b_step1"], to[cfg]["position_only_valid"]),
                 (to[cfg]["theme_only_b_step1"], to[cfg]["theme_only_valid"]),
                 (tr["neutral"]["b_step1"], tr["neutral"]["valid"]), (tr["cd"]["b_step1"], tr["cd"]["valid"])]
        for j, (b, v) in enumerate(cells):
            if C.truthy(v):
                ax.add_patch(C.plt.Rectangle((j, i), 1, 1, fc=cmap(norm(C.num(b))), ec="#BBBBBB", lw=1.5))
            else:
                ax.add_patch(C.plt.Rectangle((j, i), 1, 1, fc="white", ec="#BBBBBB", lw=1.5))
                ax.add_patch(C.plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fill=False, ec=P.CHANCE, lw=0,
                                             hatch="///"))
    ax.set_xlim(0, 5)
    ax.set_ylim(4, 0)
    ax.set_xticks(np.arange(5) + 0.5, COLS)
    ax.set_yticks(np.arange(4) + 0.5, CONFIGS)
    ax.tick_params(length=0)
    cb = fig.colorbar(C.plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("b (bits)", fontsize=P.SIZE_LABEL)
    cb.ax.tick_params(labelsize=P.SIZE_TICK)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
