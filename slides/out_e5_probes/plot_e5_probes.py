"""E5 probe images (E4's zero-data set, 93 distinct images of p1-b-d1): raw agent-free render, E5's own colours."""
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
from helplessness.pipeline import two_zones_setup as setup  # noqa: E402
from helplessness.probing.theme_probes import probe_images  # noqa: E402

OUT = Path(__file__).parent
CONFIG = "p1-b-d1"
cfg = setup.load_config(REPO / "configs/e4/e4.yaml")
conf = setup.configuration(cfg, CONFIG)
k, gl = cfg["render"]["k"], cfg["render"]["grid_lines"]
layout = two_zones.build_layout(conf["diagonal"], "reference")
images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
rgb = [render_background(layout, two_zones.themes(im["slot_colors"], conf["agent_color"]), k=k, grid_lines=gl) for im in images]
for i, a in enumerate(rgb):
    plt.imsave(OUT / f"probe_{i:02d}.png", np.kron(a, np.ones((10, 10, 1), dtype=np.uint8)))

nc = 11
nr = -(-len(rgb) // nc)
fig, axs = plt.subplots(nr, nc, figsize=(nc * 1.5, nr * 1.5), dpi=200)
fig.subplots_adjust(0.005, 0.005, 0.995, 0.995, 0.04, 0.04)
for j, ax in enumerate(axs.ravel()):
    ax.axis("off")
    if j < len(rgb):
        ax.imshow(rgb[j], interpolation="nearest")
fig.savefig(OUT / "E5_probes_grid.png"); plt.close(fig)

lines = []
for im in images:
    pr = "; ".join(f"{p['axis']}:{p['k']}@{p['position']}" for p in im["probes"]) or "training image"
    sc = ", ".join(f"{q}={tuple(c)}" for q, c in im["slot_colors"].items())
    lines.append(f"| {im['index']:02d} | {'yes' if im['is_training'] else ''} | {','.join(im['zones'])} | {pr} | {sc} |")
(OUT / "MANIFEST_e5_probes.md").write_text(f"""# E5 probe images ({CONFIG})

{len(images)} distinct images of `probing.theme_probes.probe_images` (E4's zero-data set, reused by E5), in its order; the training image is index 0.
Raw render, agent-free (`render_background`, k={k}, grid_lines={gl}), reference dynamics layout, E5's own colours: T_inc/UD = {conf['t_inc']}, T_ctrl/CD = {conf['t_ctrl']} (not the red/blue swap of the layouts figure).
Files: probe_00..{len(images)-1:02d}.png (720 px), E5_probes_grid.png ({nr}x{nc}, row-major). The same probe images apply to every shuffle agent. Script: `plot_e5_probes.py`.
Probe notation: axis:k@position (k = 0..10 along the axis; position = former UD / CD / neutral slot of the canonical layout).

| idx | training | zones of interest | probes | slot colours |
|---|---|---|---|---|
""" + "\n".join(lines) + "\n", encoding="utf-8")
print(len(images))
