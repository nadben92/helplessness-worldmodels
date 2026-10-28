"""E7e (v2) -- Three readers of the same predicted z': decoder (-> b), critic (-> action), KL (-> h correction).
Block names only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "E7e_three_readers"


def main():
    fig = C.plt.figure(figsize=C.FULL)
    ax = D.canvas(fig)
    D.box(ax, 0.1, 0.5, 0.14, 0.14, "h′ₐ")
    D.box(ax, 0.32, 0.5, 0.14, 0.14, "z′", lw=3.4)
    D.conn(ax, (0.17, 0.5), (0.25, 0.5))
    for y, name, out, col in ((0.82, "decoder", "b", P.REFERENCE), (0.5, "critic", "action", P.EXPOSED),
                              (0.18, "KL", "h correction", P.CORRECTION)):
        D.box(ax, 0.6, y, 0.18, 0.14, name, ec=col, lw=3)
        D.box(ax, 0.86, y, 0.18, 0.14, out, ec=col, lw=3, fc=col, tc="white")
        D.conn(ax, (0.39, 0.5), (0.51, y), color=col, lw=3)
        D.conn(ax, (0.69, y), (0.77, y), color=col, lw=3)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
