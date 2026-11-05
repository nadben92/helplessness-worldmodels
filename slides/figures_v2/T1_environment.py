"""T1 (v2) -- E1-E3's environment: four zones, corridor, goal; only the conveyor (UD) arrows, read from the tensor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "T1_environment"


def main():
    types, T = D.four_zone_types(), D.four_zone_tensor()
    fig = C.plt.figure(figsize=C.FULL)
    ax = D.canvas(fig, (0.24, 0.03, 0.52, 0.94))
    ax.set_xlim(-0.5, 11.5)
    D.draw_cells(ax, types)
    D.conveyor_arrows(ax, T, types, "UD")
    for code in ("UD", "CN", "CD", "UN"):
        rr, cc = (types == code).nonzero()
        ax.text(cc.mean(), rr.min() - 1.0 if rr.min() < 6 else rr.max() + 0.75, code, ha="center", va="center", fontsize=P.SIZE_BLOCK + 3,
                fontweight="bold", zorder=7)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
