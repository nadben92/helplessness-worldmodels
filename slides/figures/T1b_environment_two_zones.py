"""T1b -- E4's two-zone environment: UD and CD on one diagonal, the two other slots with the corridor's dynamics and
colour. The paired reference has the same image with UD made controllable (derive_control). Arrows from the tensors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _grid as G  # noqa: E402

FIG_ID = "T1b_environment_two_zones"


def panel(ax, condition: str, name: str, color: str):
    types = G.two_zone_types(1)
    T = G.two_zone_tensor(1, condition)
    if condition == "reference":
        types = types.copy()
        types[types == "UD"] = "CD"
    G.draw_cells(ax, types)
    exp = G.two_zone_types(1)
    for code in ("UD", "CD"):
        rr, cc = (exp == code).nonzero()
        r, c = int(round(rr.mean())), int(round(cc.mean()))
        mass = G.action_arrows(ax, T, r * 12 + c, color=P.TEXT)
        label = code if condition == "experimental" or code == "CD" else "UD → CD"
        ax.text(cc.min() - 0.35, rr.min() - 0.3, label, fontsize=P.SIZE_ANNOT + 1, fontweight="bold", color=P.TEXT,
                ha="left", va="top", zorder=7)
        if len(mass) == 1:
            ax.annotate("×5", (c, r + 0.5), xytext=(10, 0), textcoords="offset points", fontsize=P.SIZE_ANNOT,
                        fontweight="bold", color=P.TEXT, va="center")
    ax.set_title(name, color=color, fontsize=P.SIZE_AXIS + 1, loc="center", pad=8)


def main():
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "E4: one conveyor, one controllable zone",
            "The two other quadrants take the corridor's dynamics and colour")
    panel(fig.add_axes([0.05, 0.12, 0.40, 0.66]), "experimental", "Exposed model trains here", P.EXPOSED)
    panel(fig.add_axes([0.53, 0.12, 0.40, 0.66]), "reference", "Paired reference trains here", P.REFERENCE)
    C.footnote(fig, "Same image, geometry and goal in both; only the conveyor differs (made controllable by the "
                    "pairing). Fills show dynamics roles, not the rendered colour themes.", y=0.03)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
