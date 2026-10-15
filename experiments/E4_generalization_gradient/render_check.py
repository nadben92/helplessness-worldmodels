"""Renders E4's images for a visual check (never into the repository: --out is a new directory outside it):

  configurations.png   for every configuration, the experimental and the reference environment as the model sees them
                       (the observation with the agent in the corridor), each slot labelled with its role and dynamics,
                       the conveyor's drift drawn as an arrow;
  axes_<pair>.png      the three axes of each pair, 11 swatches each, with the agent, goal and corridor colours;
  probes_<config>.png  for one configuration of each pair: the training image, the extreme probes (theme only: T_inc at
                       the neutral slot; position only: the corridor's colour at the UD slot) and a few axis points;
  <config>_<name>.png  each of those images alone, at the model's resolution (72 x 72) and enlarged.

Usage: python experiments/E4_generalization_gradient/render_check.py --out <new directory outside the repo>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import e4_setup
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR, GOAL_COLOR
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.observation.allocentric_pixel_backend import render
from helplessness.probing.theme_probes import probe_images

AGENT_STATE = 6  # (0, 6): a corridor cell, away from the goal


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def image_of(conf: dict, slot_colors: dict, condition: str = "reference") -> np.ndarray:
    layout = two_zones.build_layout(conf["diagonal"], condition)
    return render(AGENT_STATE, layout, two_zones.themes(slot_colors, conf["agent_color"]), k=6)


def annotate(ax, conf: dict, condition: str) -> None:
    layout = two_zones.build_layout(conf["diagonal"], condition)
    role_of = {slot: role for role, slot in conf["roles"].items()}
    for z in layout.zones:
        slot = z.appearance.category
        if slot not in QUADRANT_ORDER:
            continue
        rows, cols = np.nonzero(z.mask)
        cy, cx = rows.mean() * 6 + 3, cols.mean() * 6 + 3
        kind = "UD" if isinstance(z.dynamics, UncontrollableDeterministic) else "CD"
        ax.text(cx, cy, f"{slot}\n{role_of[slot]}\n{kind}", ha="center", va="center", fontsize=7, color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})
        if kind == "UD":
            dr, dc = z.dynamics.drift
            ax.annotate("", xy=(cx + 12 * dc, cy + 12 * dr), xytext=(cx, cy), arrowprops={"color": "white", "width": 2})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=e4_setup.DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.resolve().is_relative_to(e4_setup.REPO_ROOT):
        parser.error("--out must be outside the repository")
    args.out.mkdir(parents=True, exist_ok=False)
    plt = _plt()
    cfg = e4_setup.load_config(args.config)
    ids = list(cfg["e4"]["configurations"])
    fig, axes = plt.subplots(len(ids), 2, figsize=(7, 3.4 * len(ids)))
    for row, cid in zip(axes, ids):
        conf = e4_setup.configuration(cfg, cid)
        for ax, condition in zip(row, ("experimental", "reference")):
            ax.imshow(image_of(conf, conf["training_colors"], condition), interpolation="nearest")
            annotate(ax, conf, condition)
            ax.set_title(f"{cid} {condition}\nT_inc {conf['t_inc']}  T_ctrl {conf['t_ctrl']}", fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "configurations.png", dpi=110)
    plt.close(fig)

    themes = e4_setup.load_themes(cfg)
    for name, pair in themes["pairs"].items():
        fig, ax = plt.subplots(figsize=(9, 2.6))
        for r, key in enumerate(("segment", "arc_center_b", "arc_center_a")):
            ax.imshow(np.array(pair[key], dtype=np.uint8)[None], extent=(0, 11, 2.8 - r, 3.6 - r), interpolation="nearest")
            ax.text(-0.2, 3.2 - r, key, ha="right", va="center", fontsize=8)
        refs = np.array([themes["agent_marker_color"], GOAL_COLOR, CORRIDOR_COLOR], dtype=np.uint8)
        ax.imshow(refs[None], extent=(0, 3, -0.3, 0.5), interpolation="nearest")
        ax.text(3.2, 0.1, "agent, goal, corridor", va="center", fontsize=8)
        ax.set_xlim(-3, 11.5)
        ax.set_ylim(-0.5, 3.8)
        ax.axis("off")
        ax.set_title(f"{name}: a={pair['a']} b={pair['b']} D={pair['D']:.1f}", fontsize=9)
        fig.savefig(args.out / f"axes_{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    for cid in ("p1-a-d1", "p2-a-d1"):
        conf = e4_setup.configuration(cfg, cid)
        images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
        def find(axis, k, position):
            return next(im for im in images for p in im["probes"] if (p["axis"], p["k"], p["position"]) == (axis, k, position))
        picks = {"training": images[0], "theme_only_Tinc_at_neutral": find("segment", 0, "neutral"),
                 "position_only_corridor_at_ud": find("position_only", 0, "ud"), "segment_t0.5_at_ud": find("segment", 5, "ud"),
                 "arc_ctrl_k10_at_ud": find("arc_ctrl", 10, "ud"), "arc_inc_k10_at_cd": find("arc_inc", 10, "cd"),
                 "Tctrl_at_ud": find("segment", 10, "ud"), "Tinc_at_cd": find("segment", 0, "cd")}
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
        for ax, (label, im) in zip(axes.ravel(), picks.items()):
            img = image_of(conf, im["slot_colors"])
            ax.imshow(img, interpolation="nearest")
            ax.set_title(f"{label}\nimage {im['index']}", fontsize=8)
            ax.axis("off")
            plt.imsave(args.out / f"{cid}_{label}.png", np.kron(img, np.ones((4, 4, 1), dtype=np.uint8)))
        fig.tight_layout()
        fig.savefig(args.out / f"probes_{cid}.png", dpi=110)
        plt.close(fig)
    print(f"[render_check] images in {args.out}: " + ", ".join(sorted(p.name for p in args.out.iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
