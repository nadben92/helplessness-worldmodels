"""E4a -- Design: exposed and reference trained, then frozen and probed with zero data in an all-controllable world
where one quadrant's colour is changed along a colour continuum, at three positions, plus "position only" and "theme
only". Colours are p1-a's real themes and its segment's 11 points (probes_b_step1.csv)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _grid as G  # noqa: E402
import _schema as S  # noqa: E402

FIG_ID = "E4a_design"
SRC = ("results", "E4_generalization_gradient", "final", "tables", "probes_b_step1.csv")
CONFIG = "p1-a-d1"


def rgb(s: str):
    return tuple(int(v) / 255 for v in s.split("-"))


def mini(fig, rect, slot_colors: dict, title: str, mark: str | None = None, tc=P.TEXT):
    """A torus rendered with given slot colours (slot -> RGB), others corridor-coloured."""
    from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
    from helplessness.envs.layouts import two_zones
    corridor = tuple(c / 255 for c in CORRIDOR_COLOR)
    regions = two_zones.region_states(two_zones.build_layout(1, "reference"))
    types = np.full((12, 12), "corridor", dtype=object)
    cols = {"corridor": corridor}
    for slot, color in slot_colors.items():
        r, c = np.divmod(regions[slot], 12)
        types[r, c] = slot
        cols[slot] = color
    ax = fig.add_axes(rect)
    G.draw_cells(ax, types, colors=cols, lw=0.4, outline=False, goal_size=0.4)
    if mark:
        r, c = np.divmod(regions[mark], 12)
        ax.add_patch(C.plt.Rectangle((c.min() - 0.5, r.min() - 0.5), 5, 5, fill=False, ec=P.TEXT, lw=2.4, zorder=8))
    ax.set_title(title, fontsize=P.SIZE_ANNOT, color=tc, loc="center", pad=6, fontweight="bold")
    return ax


def main():
    from helplessness.envs.layouts import two_zones
    roles = two_zones.roles(1)
    rows = [r for r in C.read_csv(C.path(*SRC)) if r["configuration"] == CONFIG]
    seg = sorted((r for r in rows if r["axis"] == "segment" and r["position"] == "ud"), key=lambda r: int(r["k"]))
    t_inc, t_ctrl = rgb(rows[0]["T_inc"]), rgb(rows[0]["T_ctrl"])
    from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
    corridor = tuple(c / 255 for c in CORRIDOR_COLOR)
    ud, cd, neu = roles["ud"], roles["cd"], roles["neutral"]

    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "Train, freeze, then move colours and positions apart",
            "Probes: frozen weights, zero data, a world where every zone is controllable")
    bg = S.canvas(fig)
    # Step 1: training
    bg.text(0.045, 0.80, "1  Train (300k steps)", fontsize=P.SIZE_AXIS, fontweight="bold", color=P.TEXT)
    mini(fig, [0.04, 0.42, 0.15, 0.3], {ud: t_inc, cd: t_ctrl}, "exposed", tc=P.EXPOSED)
    mini(fig, [0.21, 0.42, 0.15, 0.3], {ud: t_inc, cd: t_ctrl}, "reference", tc=P.REFERENCE)
    bg.text(0.045, 0.37, "conveyor in the red quadrant;\nreference: same image, all controllable",
            fontsize=P.SIZE_SMALL, color=P.TEXT_2, va="top")
    S.conn(bg, (0.375, 0.57), (0.425, 0.57), color=P.TEXT, lw=2.2, head=12)
    bg.text(0.4, 0.61, "freeze", ha="center", fontsize=P.SIZE_SMALL, color=P.TEXT_2)
    # Step 2: probes
    bg.text(0.445, 0.80, "2  Probe: recolour one quadrant", fontsize=P.SIZE_AXIS, fontweight="bold", color=P.TEXT)
    mid = rgb(seg[5]["color"])
    mini(fig, [0.44, 0.47, 0.12, 0.25], {ud: mid, cd: t_ctrl}, "old UD", mark=ud)
    mini(fig, [0.575, 0.47, 0.12, 0.25], {ud: t_inc, cd: mid}, "old CD", mark=cd)
    mini(fig, [0.71, 0.47, 0.12, 0.25], {ud: t_inc, cd: t_ctrl, neu: mid}, "neutral", mark=neu)
    mini(fig, [0.845, 0.47, 0.12, 0.25], {ud: corridor, cd: t_ctrl}, "position only", mark=ud)
    # colour continuum
    n = len(seg)
    x0, w = 0.47, 0.33
    for i, r in enumerate(seg):
        bg.add_patch(C.plt.Rectangle((x0 + i * w / n, 0.3), w / n, 0.06, fc=rgb(r["color"]), ec=P.BG, lw=1))
    bg.text(x0, 0.39, "training colour of\nthe conveyor (T_inc)", fontsize=P.SIZE_SMALL, color=P.TEXT, va="bottom")
    bg.text(x0 + w, 0.39, "colour of the\ncontrollable zone", fontsize=P.SIZE_SMALL, color=P.TEXT, va="bottom",
            ha="right")
    bg.text(x0 + w / 2, 0.27, "11 colours per axis (segment shown; 2 arcs too)", ha="center", va="top",
            fontsize=P.SIZE_SMALL, color=P.TEXT_2)
    bg.text(0.905, 0.42, "conveyor's quadrant\nin corridor colour", ha="center", va="top", fontsize=P.SIZE_SMALL,
            color=P.TEXT_2)
    bg.text(0.905, 0.3, "theme only: neutral\nquadrant in T_inc", ha="center", va="top", fontsize=P.SIZE_SMALL,
            color=P.TEXT_2)
    # Step 3: read-out
    bg.text(0.045, 0.16, "3  Read b at the first step in the recoloured quadrant (exposed − reference);\n"
                         "    keep only probes where the reference still predicts well (validity).", fontsize=P.SIZE_ANNOT, color=P.TEXT)
    C.footnote(fig, f"Colours of configuration {CONFIG}; 4 configurations (2 colour pairs × which colour is the "
                    "conveyor's), 100 probes each. Source: results/E4_generalization_gradient/final/tables/"
                    "probes_b_step1.csv.", y=0.03)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
