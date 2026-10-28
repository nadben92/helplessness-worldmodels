"""Drawing the 12 x 12 torus: zone fills, goal, arrows between cells. Cell (r, c) is centred on (x=c, y=r), row 0 at
the top. Zone membership comes from the library's layouts, never from a hand-written map."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from style import palette as P
from style.common import REPO  # noqa: F401  (the library is importable once style.common is)

GRID = (12, 12)


def four_zone_types() -> np.ndarray:
    """(12, 12) zone codes of E1-E3's 4-zone layout (configs/zones/e1_4zones.yaml, canonical assignment)."""
    from helplessness.config import load_yaml_block
    cfg = load_yaml_block("zones/e1_4zones.yaml")
    code = {"UncontrollableDeterministic": "UD", "ControllableNoisy": "CN", "ControllableDeterministic": "CD",
            "UncontrollableNoisy": "UN"}
    types = np.full(GRID, "corridor", dtype=object)
    for block in cfg["blocks"].values():
        (r0, r1), (c0, c1) = block["rows"], block["cols"]
        types[r0:r1 + 1, c0:c1 + 1] = code[block["dynamics_type"]]
    return types


def four_zone_tensor() -> np.ndarray:
    """E1-E3's experimental transition tensor (S, A, S), from the library's environment."""
    from helplessness.config import load_yaml_block
    from helplessness.envs.layouts import four_zones
    cfg = load_yaml_block("zones/e1_4zones.yaml")
    env = four_zones.build_env(four_zones.build_layout(cfg, "away_from_goal"), cfg, 50.0)
    return env.transition_tensor()


def two_zone_types(diagonal: int = 1) -> np.ndarray:
    """(12, 12) zone codes of E4's two-zone layout: UD and CD on the diagonal, the other slots corridor-like."""
    from helplessness.envs.layouts import two_zones
    regions = two_zones.region_states(two_zones.build_layout(diagonal, "experimental"))
    roles = two_zones.roles(diagonal)
    types = np.full(GRID, "corridor", dtype=object)
    for role, code in (("ud", "UD"), ("cd", "CD")):
        r, c = np.divmod(regions[roles[role]], GRID[1])
        types[r, c] = code
    return types


def two_zone_tensor(diagonal: int = 1, condition: str = "experimental") -> np.ndarray:
    from helplessness.envs.layouts import two_zones, four_zones
    from helplessness.config import load_yaml_block
    cfg = load_yaml_block("zones/e1_4zones.yaml")
    env = four_zones.build_env(two_zones.build_layout(diagonal, condition), cfg, 50.0)
    return env.transition_tensor()


def draw_cells(ax, types: np.ndarray, colors: dict | None = None, goal=(0, 0), lw: float = 0.8, outline: bool = True,
               goal_size: float = 0.42) -> None:
    colors = colors or P.ZONE_COLORS
    R, C = types.shape
    for r in range(R):
        for c in range(C):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colors[types[r, c]], edgecolor=P.BG, lw=lw,
                                   zorder=1))
    if outline:
        for code in set(types.ravel()) - {"corridor"}:
            rr, cc = np.nonzero(types == code)
            for rows, cols in _blocks(rr, cc):
                ax.add_patch(Rectangle((cols[0] - 0.5, rows[0] - 0.5), cols[1] - cols[0] + 1, rows[1] - rows[0] + 1,
                                       fill=False, edgecolor=P.AXIS, lw=1.0, zorder=2))
    if goal is not None:
        draw_goal(ax, *goal, size=goal_size)
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(R - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def _blocks(rr, cc):
    """Split the cells of one zone code into 5 x 5 blocks (the quadrants)."""
    out = []
    for rb in (range(1, 6), range(7, 12)):
        for cb in (range(1, 6), range(7, 12)):
            m = np.isin(rr, list(rb)) & np.isin(cc, list(cb))
            if m.any():
                out.append(((rr[m].min(), rr[m].max()), (cc[m].min(), cc[m].max())))
    return out


def draw_goal(ax, r, c, size: float = 0.42) -> None:
    ax.scatter([c], [r], marker="*", s=(size * 62) ** 2, color=P.GOAL, edgecolor=P.TEXT, linewidth=0.8, zorder=6)


def arrow(ax, rc0, rc1, color=P.TEXT, lw=2.0, alpha=1.0, head=12, shrink=6, zorder=5, style="-|>") -> None:
    (r0, c0), (r1, c1) = rc0, rc1
    ax.add_patch(FancyArrowPatch((c0, r0), (c1, r1), arrowstyle=f"{style},head_length={head * 0.05:.2f},"
                                 f"head_width={head * 0.032:.2f}", mutation_scale=20, color=color, lw=lw, alpha=alpha,
                                 shrinkA=shrink, shrinkB=shrink, zorder=zorder))


def action_arrows(ax, T: np.ndarray, s: int, color=P.TEXT, min_p: float = 0.02, lw_max: float = 3.4,
                  actions=None) -> dict:
    """One arrow per (action, successor) of state s, width and opacity by probability; a ring for the self-loop.
    Returns {successor: summed probability over actions} for labelling."""
    r0, c0 = divmod(s, GRID[1])
    mass = {}
    actions = range(T.shape[1]) if actions is None else actions
    for a in actions:
        for s2 in np.nonzero(T[s, a] > min_p)[0]:
            p = float(T[s, a, s2])
            mass[int(s2)] = mass.get(int(s2), 0.0) + p
    for s2, m in mass.items():
        p = m / len(actions)
        r1, c1 = divmod(s2, GRID[1])
        if s2 == s:
            ax.add_patch(_ring(c0, r0, color, alpha=0.35 + 0.65 * p, lw=0.8 + lw_max * p))
        else:
            arrow(ax, (r0, c0), (r1, c1), color=color, lw=0.8 + lw_max * p, alpha=0.35 + 0.65 * p, head=7 + 9 * p,
                  shrink=4)
    return mass


def _ring(x, y, color, alpha, lw):
    from matplotlib.patches import Circle
    return Circle((x, y), 0.2, fill=False, edgecolor=color, lw=lw, alpha=alpha, zorder=5)
