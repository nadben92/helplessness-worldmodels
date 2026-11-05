"""Drawing helpers of the v2 figures: the 12 x 12 torus (zone membership and transition tensors from the library's
layouts, never hand-written) and schematic boxes. Cell (r, c) is centred on (x=c, y=r), row 0 at the top."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from style_v2 import palette as P

GRID = (12, 12)


# ------------------------------------------------------------------------------------------------------- layouts
def _cfg():
    from helplessness.config import load_yaml_block
    return load_yaml_block("zones/e1_4zones.yaml")


def four_zone_types() -> np.ndarray:
    code = {"UncontrollableDeterministic": "UD", "ControllableNoisy": "CN", "ControllableDeterministic": "CD",
            "UncontrollableNoisy": "UN"}
    types = np.full(GRID, "corridor", dtype=object)
    for block in _cfg()["blocks"].values():
        (r0, r1), (c0, c1) = block["rows"], block["cols"]
        types[r0:r1 + 1, c0:c1 + 1] = code[block["dynamics_type"]]
    return types


def four_zone_tensor() -> np.ndarray:
    from helplessness.envs.layouts import four_zones
    cfg = _cfg()
    return four_zones.build_env(four_zones.build_layout(cfg, "away_from_goal"), cfg, 50.0).transition_tensor()


def two_zone_types(diagonal: int = 1) -> np.ndarray:
    from helplessness.envs.layouts import two_zones
    regions = two_zones.region_states(two_zones.build_layout(diagonal, "experimental"))
    roles = two_zones.roles(diagonal)
    types = np.full(GRID, "corridor", dtype=object)
    for role, code in (("ud", "UD"), ("cd", "CD")):
        r, c = np.divmod(regions[roles[role]], GRID[1])
        types[r, c] = code
    return types


def two_zone_tensor(diagonal: int = 1, condition: str = "experimental") -> np.ndarray:
    from helplessness.envs.layouts import four_zones, two_zones
    return four_zones.build_env(two_zones.build_layout(diagonal, condition), _cfg(), 50.0).transition_tensor()


# --------------------------------------------------------------------------------------------------------- torus
def draw_cells(ax, types: np.ndarray, colors: dict | None = None, goal=(0, 0), lw: float = 1.0, goal_size: float = 26,
               edge: str = "white"):
    colors = colors or P.ZONE_COLORS
    R, C = types.shape
    for r in range(R):
        for c in range(C):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colors[types[r, c]], edgecolor=edge, lw=lw))
    if goal is not None:
        ax.scatter([goal[1]], [goal[0]], marker="*", s=goal_size ** 2, color=P.GOAL, edgecolor="black",
                   linewidth=1.0, zorder=6, clip_on=False)
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(R - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def arrow(ax, rc0, rc1, color="black", lw=3.0, mutation=28, shrink=6, zorder=5):
    (r0, c0), (r1, c1) = rc0, rc1
    ax.add_patch(FancyArrowPatch((c0, r0), (c1, r1), arrowstyle="-|>", mutation_scale=mutation, color=color, lw=lw,
                                 shrinkA=shrink, shrinkB=shrink, zorder=zorder))


def conveyor_arrows(ax, T: np.ndarray, types: np.ndarray, code: str = "UD", color="black"):
    """One arrow per state of the zone, to its successor under the conveyor (read from the tensor: deterministic and
    the same for every action)."""
    for r, c in zip(*np.nonzero(types == code)):
        s = r * GRID[1] + c
        succ = {int(np.argmax(T[s, a])) for a in range(T.shape[1])}
        assert len(succ) == 1 and all(T[s, a].max() == 1.0 for a in range(T.shape[1])), "not a conveyor state"
        r1, c1 = divmod(succ.pop(), GRID[1])
        if abs(r1 - r) + abs(c1 - c) == 1:
            arrow(ax, (r, c), (r1, c1), color=color, lw=2.2, mutation=18, shrink=2)


# -------------------------------------------------------------------------------------------------------- schema
def canvas(fig, rect=(0, 0, 1, 1)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, text="", fc="white", ec="black", lw=2.0, ls="-", size=P.SIZE_BLOCK, tc="black",
        weight="bold"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.015", fc=fc,
                                ec=ec, lw=lw, ls=ls, zorder=3))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=tc, fontweight=weight, zorder=4)


def conn(ax, p0, p1, color="black", lw=2.2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=22, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=2))
