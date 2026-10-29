"""E5 shuffle layouts: the 12 zone assignments of p1-b-d1 (agent-free render, UD conveyor arrows from the dynamics)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from helplessness.envs.layouts import two_zones  # noqa: E402
from helplessness.envs.observation.allocentric_pixel_backend import render_background  # noqa: E402
from helplessness.pipeline import episode_sources as es, two_zones_setup as setup  # noqa: E402

OUT = Path(__file__).parent
CONFIG = "p1-b-d1"
cfg = setup.load_config(REPO / "configs/e4/e4.yaml")
conf = setup.configuration(cfg, CONFIG)
R, C = two_zones.GRID_SHAPE
k, gl = cfg["render"]["k"], cfg["render"]["grid_lines"]
canon = es.canonical_assignment(conf["diagonal"])
rows = []


def draw(ax, i):
    w = es.assignment_world(i, conf, "experimental", cfg["render"], cfg["episode"], conf["seed"])
    img = render_background(w.layout, w.themes, k=k, grid_lines=gl)
    ax.imshow(img, extent=(0, C, R, 0), interpolation="nearest")
    for q in es.QUADRANT_ORDER:
        m = two_zones.quadrant_mask(q)
        rr, cc = np.nonzero(m)
        ax.add_patch(plt.Rectangle((cc.min(), rr.min()), cc.max() - cc.min() + 1, rr.max() - rr.min() + 1,
                                   fill=False, ec="0.75", lw=0.5))
    succ = w.env.transition_tensor().argmax(axis=-1)
    ud = es.ASSIGNMENTS[i][0]
    for s in two_zones.region_states(w.layout)[ud]:
        r, c = divmod(int(s), C)
        # the conveyor is action-independent: all actions must give the same successor
        assert len(set(succ[s].tolist())) == 1
        r2, c2 = divmod(int(succ[s, 0]), C)
        dr, dc = ((r2 - r + R // 2) % R) - R // 2, ((c2 - c + C // 2) % C) - C // 2  # shortest signed step on the torus
        ax.annotate("", xy=(c + .5 + .35 * dc, r + .5 + .35 * dr), xytext=(c + .5 - .35 * dc, r + .5 - .35 * dr),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.set_xlim(0, C); ax.set_ylim(R, 0); ax.set_aspect("equal"); ax.axis("off")
    return w, (dr, dc)


for i in range(12):
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, i)
    fig.savefig(OUT / f"layout_{i:02d}.png"); plt.close(fig)

fig, axs = plt.subplots(3, 4, figsize=(12, 9), dpi=200)
fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.04, 0.04)
dirs = set()
for i, ax in enumerate(axs.ravel()):
    w, d = draw(ax, i)
    ud, cd = es.ASSIGNMENTS[i]
    rows.append(f"| {i:02d} | {ud} | {cd} | {', '.join(q for q in es.QUADRANT_ORDER if q not in (ud, cd))} | {d} | {'**canonical**' if i == canon else ''} |")
fig.savefig(OUT / "E5_layouts_grid.png"); plt.close(fig)

col = es.assignment_colors(canon, conf["diagonal"], conf["t_inc"], conf["t_ctrl"])
(OUT / "MANIFEST_e5_layouts.md").write_text(f"""# E5 shuffle layouts ({CONFIG})

Files: layout_00..11.png (one per assignment, ASSIGNMENTS order of `pipeline/episode_sources.py`), E5_layouts_grid.png (3x4, row-major).
Script: `slides/out_e5_layouts/plot_e5_layouts.py`.

| idx | UD | CD | neutral | arrow (drow, dcol) | |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

- Canonical (E4 and probes): index {canon} (UD={es.ASSIGNMENTS[canon][0]}, CD={es.ASSIGNMENTS[canon][1]}).
- RGB: T_inc / UD = {conf['t_inc']}, T_ctrl / CD = {conf['t_ctrl']}, neutral/corridor = {col[[q for q in es.QUADRANT_ORDER if q not in es.ASSIGNMENTS[canon]][0]]}.
- Goal: (row, col) = {two_zones.GOAL_RC}, state {two_zones.GOAL_STATE}, drawn as the render does (no agent marker).
- Arrows: white, one per UD cell, from the cell towards its successor given by `env.transition_tensor().argmax(-1)` (same for every action, asserted), shortest signed step on the torus, drawn at the cell. Drift convention: `{two_zones.DIRECTION}`.
- Gray outlines (0.75, lw 0.5) mark the 4 quadrants (the render does not distinguish the neutral ones from the corridor).
- Render: `envs.observation.allocentric_pixel_backend.render_background` via `pipeline.episode_sources.assignment_world` (k={k}, grid_lines={gl}).
""", encoding="utf-8")
