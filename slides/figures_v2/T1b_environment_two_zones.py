"""T1b (v2) -- E4's two-zone environment, exposed and paired reference; only the conveyor arrows (exposed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "T1b_environment_two_zones"


def main():
    fig = C.plt.figure(figsize=C.FULL)
    exp = D.two_zone_types(1)
    for k, (cond, name, col) in enumerate((("experimental", "exposed", P.EXPOSED),
                                            ("reference", "reference", P.REFERENCE))):
        ax = D.canvas(fig, (0.04 + 0.49 * k, 0.04, 0.43, 0.82))
        ax.set_xlim(-0.5, 11.5)
        types = exp.copy()
        if cond == "reference":
            types[types == "UD"] = "CD"
        D.draw_cells(ax, types)
        if cond == "experimental":
            D.conveyor_arrows(ax, D.two_zone_tensor(1, cond), types, "UD")
        for code in ("UD", "CD"):
            rr, cc = (exp == code).nonzero()
            ax.text(cc.mean(), rr.min() - 1.0 if rr.min() < 6 else rr.max() + 0.75, code if cond == "experimental" or code == "CD" else "CD",
                    ha="center", va="center", fontsize=P.SIZE_BLOCK + 2, fontweight="bold", zorder=7)
        fig.text(0.04 + 0.49 * k + 0.215, 0.93, name, ha="center", fontsize=P.SIZE_BLOCK + 4, fontweight="bold",
                 color=col)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
