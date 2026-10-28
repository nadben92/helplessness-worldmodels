"""T1 -- The environment of E1-E3: 12 x 12 torus, four 5 x 5 zones, cross corridor, goal at an intersection. The
arrows are read from the library's transition tensor at the centre of each zone (width and opacity = probability)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _grid as G  # noqa: E402

FIG_ID = "T1_environment"

LEGEND = [
    ("CD", "Controllable", "each action moves where it says"),
    ("UD", "Uncontrollable (conveyor)", "all 5 actions → the same cell"),
    ("CN", "Controllable, noisy", "the action works 90% of the time"),
    ("UN", "Uncontrollable, noisy", "conveyor 90% of the time"),
    ("corridor", "Corridor", "controllable, links the zones"),
]


def main():
    types = G.four_zone_types()
    T = G.four_zone_tensor()
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "Four zones, four kinds of control",
            "12 × 12 torus (edges wrap) · 5 actions: 4 moves + stay · same task everywhere: reach the goal")
    ax = fig.add_axes([0.05, 0.11, 0.47, 0.71])
    G.draw_cells(ax, types)
    for code in ("CD", "UD", "CN", "UN"):
        rr, cc = (types == code).nonzero()
        r, c = int(round(rr.mean())), int(round(cc.mean()))
        # Deterministic zones: all five actions. Noisy zones: one action ("right"), so the 90 / 2.5 % split shows.
        acts = None if code in ("CD", "UD") else [1]
        mass = G.action_arrows(ax, T, r * 12 + c, color=P.TEXT, actions=acts)
        if acts is not None:
            ax.text(c, r + 1.75, "action: right" if code == "CN" else "any action", ha="center", va="center",
                    fontsize=P.SIZE_SMALL, color=P.TEXT, zorder=7)
        if code == "UD":
            (s2,) = mass
            r2, c2 = divmod(s2, 12)
            ax.annotate("×5", ((c + c2) / 2, (r + r2) / 2), xytext=(10, 0), textcoords="offset points",
                        fontsize=P.SIZE_ANNOT, fontweight="bold", color=P.TEXT, va="center")
        ax.text(cc.min() - 0.35, rr.min() - 0.3, code, fontsize=P.SIZE_ANNOT + 1, fontweight="bold", color=P.TEXT,
                ha="left", va="top", zorder=7)
    ax.annotate("goal", (0, 0), xytext=(0.7, -0.05), textcoords="data", fontsize=P.SIZE_SMALL, color=P.TEXT,
                fontweight="bold", va="center", zorder=7)

    y = 0.76
    for code, name, what in LEGEND:
        fig.patches.append(C.plt.Rectangle((0.585, y - 0.028), 0.03, 0.05, transform=fig.transFigure,
                                           facecolor=P.ZONE_COLORS[code], edgecolor=P.AXIS, lw=0.8))
        fig.text(0.63, y + 0.012, f"{code if code != 'corridor' else ''}{'  ' if code != 'corridor' else ''}{name}",
                 fontsize=P.SIZE_AXIS, fontweight="bold", color=P.TEXT, va="center")
        fig.text(0.63, y - 0.03, what, fontsize=P.SIZE_ANNOT, color=P.TEXT_2, va="center")
        y -= 0.125
    fig.text(0.6, y - 0.005, "★", fontsize=34, color=P.GOAL, ha="center", va="center")
    fig.text(0.63, y + 0.012, "Goal", fontsize=P.SIZE_AXIS, fontweight="bold", color=P.TEXT, va="center")
    fig.text(0.63, y - 0.03, "+50, ends the episode; −1 per step", fontsize=P.SIZE_ANNOT, color=P.TEXT_2, va="center")
    C.footnote(fig, "Arrows: successors of each zone's centre cell, read from the environment's transition tensor "
                    "(width ∝ probability). Conveyors drift away from the goal.", y=0.03)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
