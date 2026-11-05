"""E4c -- b at step 1 for the 4 configurations x the extreme conditions: training image read at the old UD, old CD and
neutral quadrants, position only (old UD quadrant in corridor colour), theme only (neutral quadrant in T_inc).
Invalid cells hatched (value shown in grey)."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E4c_conditions_map"
T = ("results", "E4_generalization_gradient", "final", "tables")
CONFIGS = ["p1-a-d1", "p1-b-d1", "p2-a-d1", "p2-b-d1"]
COLS = ["training image\nold UD", "position only\nold UD", "theme only\nneutral", "training image\nneutral",
        "training image\nold CD"]


def main():
    vz = C.read_csv(C.path(*T, "validity_image_zone.csv"))
    to = {r["configuration"]: r for r in C.read_csv(C.path(*T, "theme_only_position_only.csv"))}
    val = np.full((4, 5), np.nan)
    ok = np.zeros((4, 5), bool)
    for i, cfg in enumerate(CONFIGS):
        tr = {r["role"]: r for r in vz if r["configuration"] == cfg and C.truthy(r["is_training"])}
        cells = [(tr["ud"]["b_step1"], tr["ud"]["valid"]),
                 (to[cfg]["position_only_b_step1"], to[cfg]["position_only_valid"]),
                 (to[cfg]["theme_only_b_step1"], to[cfg]["theme_only_valid"]),
                 (tr["neutral"]["b_step1"], tr["neutral"]["valid"]),
                 (tr["cd"]["b_step1"], tr["cd"]["valid"])]
        for j, (b, v) in enumerate(cells):
            val[i, j], ok[i, j] = C.num(b), C.truthy(v)
    cmap = LinearSegmentedColormap.from_list("b", [P.EXPOSED, "#E8B9A6", P.BG])
    norm = Normalize(-np.log2(5), 0)

    fig, ax = C.full()
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76, bottom=0.12)
    C.title(fig, "Remove the colour, keep the place: the belief stays",
            "b at step 1 (bits), 4 configurations × probe conditions")
    for i in range(4):
        for j in range(5):
            v = val[i, j]
            fc = cmap(norm(v)) if ok[i, j] else P.BG
            ax.add_patch(C.plt.Rectangle((j, i), 1, 1, fc=fc, ec=P.BG, lw=4, hatch=None if ok[i, j] else "///",
                                         zorder=1))
            if not ok[i, j]:
                ax.add_patch(C.plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fill=False, ec=P.CHANCE, lw=0,
                                             hatch="///", zorder=2))
            dark = ok[i, j] and v < -1.2
            ax.text(j + 0.5, i + 0.5, C.fmt(v) + ("" if ok[i, j] else "\ninvalid"), ha="center", va="center",
                    fontsize=P.SIZE_AXIS if ok[i, j] else P.SIZE_SMALL,
                    color=(P.BG if dark else P.TEXT) if ok[i, j] else P.TEXT_2,
                    fontweight="bold" if ok[i, j] else "normal", zorder=3)
    ax.set_xlim(0, 5)
    ax.set_ylim(4, 0)
    ax.set_xticks(np.arange(5) + 0.5, COLS)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(4) + 0.5, CONFIGS)
    ax.tick_params(length=0, labelsize=P.SIZE_ANNOT)
    for s in ax.spines.values():
        s.set_visible(False)
    C.footnote(fig, "Hatched: invalid probe (the reference no longer predicts that image well; p2-a's near-black "
                    "colour). Sources: results/E4_generalization_gradient/final/tables/validity_image_zone.csv "
                    "(training image) and theme_only_position_only.csv.", y=0.025)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
