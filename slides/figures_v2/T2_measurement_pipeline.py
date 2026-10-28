"""T2 (v2) -- How b is read from the world model: block names only, and the two endpoint cases (5 different cells vs
the same cell for all 5 actions)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "T2_measurement_pipeline"


def mini(ax, x0, y0, cell, targets, color, name):
    """3 x 3 cells, square on the page (the canvas spans the 16:9 figure)."""
    ch = cell * C.FULL[0] / C.FULL[1]
    for i in range(3):
        for j in range(3):
            ax.add_patch(C.plt.Rectangle((x0 + j * cell, y0 + (2 - i) * ch), cell, ch, fc=P.ZONE_CORRIDOR,
                                         ec="white", lw=3, zorder=2))
    cx, cy = x0 + 1.5 * cell, y0 + 1.5 * ch
    for di, dj in targets:
        if (di, dj) == (0, 0):
            from matplotlib.patches import Ellipse
            ax.add_patch(Ellipse((cx, cy), cell * 0.4, ch * 0.4, fill=False, ec=color, lw=3, zorder=4))
        else:
            D.conn(ax, (cx, cy), (cx + dj * cell, cy - di * ch), color=color, lw=3.2)
    ax.text(cx, y0 - 0.05, name, ha="center", va="top", fontsize=P.SIZE_BLOCK, fontweight="bold", color=color)


def main():
    fig = C.plt.figure(figsize=C.FULL)
    ax = D.canvas(fig)
    steps = ["(h, z)", "GRU", "prior", "decoder", "cell", "I(a; s′|s)", "b"]
    xs = [0.07 + i * 0.143 for i in range(len(steps))]
    y = 0.8
    for i, (x, t) in enumerate(zip(xs, steps)):
        last = i == len(steps) - 1
        D.box(ax, x, y, 0.115, 0.15, t, lw=3.2 if last else 2.0)
        if i:
            D.conn(ax, (xs[i - 1] + 0.0575, y), (x - 0.0575, y))
    cell = 0.075
    mini(ax, 0.2, 0.12, cell, [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)], P.REFERENCE, "controllable")
    mini(ax, 0.6, 0.12, cell, [(1, 0)], P.EXPOSED, "conveyor")
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
