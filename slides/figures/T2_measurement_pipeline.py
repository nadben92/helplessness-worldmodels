"""T2 -- How b is read from a world model: one GRU step per action, the prior's draws decoded and matched to cells,
I(a; s'|s) over the five predicted-cell distributions, minus the paired reference's. Two illustrative cases (the
values are the analytic endpoints, log2 5 and 0, not data)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _schema as S  # noqa: E402

FIG_ID = "T2_measurement_pipeline"
LOG2_5 = float(np.log2(5))


def mini(ax, x0, y0, cell, targets, color, title, value, verdict):
    """A 3 x 3 neighbourhood, the five actions' predicted cells as arrows from the centre."""
    for i in range(3):
        for j in range(3):
            ax.add_patch(C.plt.Rectangle((x0 + j * cell, y0 + (2 - i) * cell), cell, cell, fc=P.ZONE_CORRIDOR,
                                         ec=P.BG, lw=2, zorder=2))
    cx, cy = x0 + 1.5 * cell, y0 + 1.5 * cell
    for (di, dj), n in targets.items():
        tx, ty = cx + dj * cell, cy - di * cell
        if (di, dj) == (0, 0):
            ax.add_patch(C.plt.Circle((cx, cy), cell * 0.22, fill=False, ec=color, lw=2.4, zorder=4))
        else:
            S.conn(ax, (cx, cy), (tx, ty), color=color, lw=2.4, head=9, zorder=4)
        if n > 1:
            ax.text(tx + cell * 0.28, ty + cell * 0.28, f"×{n}", fontsize=P.SIZE_ANNOT, color=color,
                    fontweight="bold", ha="center", va="center", zorder=5)
    ax.text(x0 + 1.5 * cell, y0 + 3 * cell + 0.035, title, ha="center", va="bottom", fontsize=P.SIZE_ANNOT,
            color=P.TEXT, fontweight="bold")
    ax.text(x0 + 3 * cell + 0.03, cy + 0.04, value, ha="left", va="center", fontsize=P.SIZE_AXIS, color=P.TEXT)
    ax.text(x0 + 3 * cell + 0.03, cy - 0.045, verdict, ha="left", va="center", fontsize=P.SIZE_AXIS, color=color,
            fontweight="bold")


def main():
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "b asks the model: do my actions lead to different places?")
    ax = S.canvas(fig, (0.02, 0.02, 0.96, 0.84))

    y = 0.74
    steps = [
        (0.075, "(h, z)", "current state"),
        (0.215, "GRU(h, z, a)", "once per action"),
        (0.355, "prior p(z′ | h′ₐ)", "16 draws"),
        (0.495, "decoder", "draws → images"),
        (0.635, "predicted cell", "nearest template"),
        (0.775, "I(a ; s′ | s)", "bits, 0 to log₂5"),
        (0.915, "b = I − I_ref", "minus reference"),
    ]
    for i, (x, t, sub) in enumerate(steps):
        last = i == len(steps) - 1
        S.box(ax, x, y, 0.122, 0.15, t, sub=sub, fc="#EFE8DA" if not last else P.BG,
              ec=P.TEXT if last else P.AXIS, weight="bold", lw=2.0 if last else 1.2)
        if i:
            S.conn(ax, (steps[i - 1][0] + 0.061, y), (x - 0.061, y))
    ax.text(0.215, y - 0.13, "a ∈ {up, right, down, left, stay}", ha="center", va="top", fontsize=P.SIZE_SMALL,
            color=P.TEXT_2)

    ax.plot([0.03, 0.97], [0.50, 0.50], color=P.GRID, lw=1.2)
    cell = 0.085
    mini(ax, 0.07, 0.09, cell, {(-1, 0): 1, (0, 1): 1, (1, 0): 1, (0, -1): 1, (0, 0): 1}, P.REFERENCE,
         "Actions lead to 5 different cells", f"I = log₂5 = {LOG2_5:.2f}", "b ≈ 0")
    mini(ax, 0.56, 0.09, cell, {(1, 0): 5}, P.EXPOSED,
         "All actions lead to the same cell", "I = 0", f"b ≈ −{LOG2_5:.1f}")
    C.footnote(fig, "Illustration with the analytic endpoints (reference: I_ref = log₂5 in a controllable zone). "
                    "b is averaged over the states of a zone.", y=0.03)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
