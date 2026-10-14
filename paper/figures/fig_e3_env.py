"""E3: the pixel environment. (a) the experimental world of seed 0 (rotation 0, class A), with its zone types;
(b) one observation: the same render with the agent's marker; (c) the reference world of the same seed, with its
zones' dynamics after the pairing. Writes paper/figures/out/fig_e3_env.pdf.

The images are E3's own renders (helplessness.envs.observation.allocentric_pixel_backend.render / render_background)
of E3's own layouts (experiments/E3_worldmodel_4zones/env.py), with the theme colours a run recorded in its
world.json (gpu_e30b win64, seed 0: the theme stream depends on the seed only).
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

from helplessness.envs.observation.allocentric_pixel_backend import render, render_background

NAME = "fig_e3_env"
SEED = 0
AGENT_CELL = (3, 9)   # an example cell of the north-east block (the controllable-noisy zone at rotation 0)


def world(condition: str) -> dict:
    runs = [r for r in src.e30b_runs("win64") if src.run_identity(r) == (SEED, condition)]
    if len(runs) != 1:
        raise SystemExit(f"Expected one gpu_e30b win64 run for seed {SEED}, {condition}; found {len(runs)}")
    return src.read_json(runs[0] / "world.json")


def main() -> None:
    e3_env = src.e3_module("env.py")
    run_mini = src.e3_module("run_mini.py")
    worlds = {c: world(c) for c in ("experimental", "reference")}
    if worlds["experimental"]["themes"] != worlds["reference"]["themes"]:
        raise SystemExit("The experimental and reference worlds of a seed must share their themes (spec 2.3)")
    themes = run_mini.themes_from_dict(worlds["experimental"]["themes"])
    assignment = e3_env.quadrant_assignment_for_rotation(worlds["experimental"]["rotation"])
    layouts = {"experimental": e3_env.build_layout(assignment), "reference": e3_env.build_control_layout(assignment)}
    k = worlds["experimental"]["render"]["k"]
    n_rows, n_cols = e3_env.GRID_SHAPE
    agent_state = AGENT_CELL[0] * n_cols + AGENT_CELL[1]

    common.apply_style()
    fig, axes = common.figure(aspect=0.38, ncols=3)
    panels = [
        ("experimental", render_background(layouts["experimental"], themes, k=k), "(a) environment"),
        ("experimental", render(agent_state, layouts["experimental"], themes, k=k), "(b) one observation"),
        ("reference", render_background(layouts["reference"], themes, k=k), "(c) reference"),
    ]
    for ax, (condition, image, title) in zip(axes, panels):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if image is panels[1][1]:
            continue
        kinds = worlds[condition]["zone_kinds"]  # appearance category -> dynamics type in this world
        for zone in layouts[condition].zones:
            category = zone.appearance.category
            if category == "corridor":
                continue
            rows, cols = np.nonzero(zone.mask)
            ax.text((cols.mean() + 0.5) * k - 0.5, (rows.mean() + 0.5) * k - 0.5, src.ZONE_ABBR[kinds[category]],
                    ha="center", va="center", color="white", fontsize=common.SMALL_FONT_SIZE,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "black", "ec": "none", "alpha": 0.55})
        g_r, g_c = divmod(themes.goal_state, n_cols)
        ax.annotate("goal", xy=((g_c + 0.5) * k, (g_r + 0.5) * k), xytext=((g_c + 2.2) * k, (g_r + 2.0) * k),
                    fontsize=common.SMALL_FONT_SIZE, color="black",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
                    arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "black"})
    fig.tight_layout(pad=0.2)
    path = common.save_pdf(fig, NAME)
    print(f"wrote {path.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
