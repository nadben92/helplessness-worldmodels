"""E4a (v2) -- Design: train (exposed, reference) -> freeze -> probe by recolouring one quadrant (old UD, old CD,
neutral, position only) along the colour continuum. Real colours of p1-a (probes_b_step1.csv); no dynamics arrows."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "E4a_design"
SRC = ("results", "E4_generalization_gradient", "final", "tables", "probes_b_step1.csv")
CONFIG = "p1-a-d1"


def rgb(s):
    return tuple(int(v) / 255 for v in s.split("-"))


def mini(fig, rect, slot_colors, name, mark=None, color="black"):
    from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
    from helplessness.envs.layouts import two_zones
    regions = two_zones.region_states(two_zones.build_layout(1, "reference"))
    types = np.full((12, 12), "corridor", dtype=object)
    cols = {"corridor": tuple(c / 255 for c in CORRIDOR_COLOR)}
    for slot, col in slot_colors.items():
        r, c = np.divmod(regions[slot], 12)
        types[r, c] = slot
        cols[slot] = col
    ax = fig.add_axes(rect)
    D.draw_cells(ax, types, colors=cols, lw=0.5, goal_size=13)
    if mark:
        r, c = np.divmod(regions[mark], 12)
        ax.add_patch(C.plt.Rectangle((c.min() - 0.5, r.min() - 0.5), 5, 5, fill=False, ec="black", lw=3.5, zorder=8))
    ax.text(5.5, -1.2, name, ha="center", va="bottom", fontsize=P.SIZE_BLOCK, fontweight="bold", color=color)


def main():
    from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
    from helplessness.envs.layouts import two_zones
    roles = two_zones.roles(1)
    ud, cd, neu = roles["ud"], roles["cd"], roles["neutral"]
    rows = [r for r in C.read_csv(C.path(*SRC)) if r["configuration"] == CONFIG]
    seg = sorted((r for r in rows if r["axis"] == "segment" and r["position"] == "ud"), key=lambda r: int(r["k"]))
    t_inc, t_ctrl, mid = rgb(rows[0]["T_inc"]), rgb(rows[0]["T_ctrl"]), rgb(seg[5]["color"])
    corridor = tuple(c / 255 for c in CORRIDOR_COLOR)

    fig = C.plt.figure(figsize=C.FULL)
    bg = D.canvas(fig)
    mini(fig, [0.02, 0.3, 0.18, 0.52], {ud: t_inc, cd: t_ctrl}, "exposed", color=P.EXPOSED)
    mini(fig, [0.21, 0.3, 0.18, 0.52], {ud: t_inc, cd: t_ctrl}, "reference", color=P.REFERENCE)
    D.conn(bg, (0.395, 0.56), (0.44, 0.56), lw=3)
    bg.text(0.4175, 0.61, "freeze", ha="center", fontsize=P.SIZE_BLOCK - 2)
    mini(fig, [0.45, 0.4, 0.13, 0.42], {ud: mid, cd: t_ctrl}, "old UD", mark=ud)
    mini(fig, [0.585, 0.4, 0.13, 0.42], {ud: t_inc, cd: mid}, "old CD", mark=cd)
    mini(fig, [0.72, 0.4, 0.13, 0.42], {ud: t_inc, cd: t_ctrl, neu: mid}, "neutral", mark=neu)
    mini(fig, [0.855, 0.4, 0.13, 0.42], {ud: corridor, cd: t_ctrl}, "position only", mark=ud)
    x0, w = 0.47, 0.49
    for i, r in enumerate(seg):
        bg.add_patch(C.plt.Rectangle((x0 + i * w / len(seg), 0.16), w / len(seg), 0.11, fc=rgb(r["color"]),
                                     ec="white", lw=1.5))
    bg.text(x0, 0.13, "T_inc", ha="left", va="top", fontsize=P.SIZE_BLOCK)
    bg.text(x0 + w, 0.13, "T_ctrl", ha="right", va="top", fontsize=P.SIZE_BLOCK)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
