"""Scripted example trajectory in E4's p1-b-d1 environment (real dynamics, real renderer) -> GIF + JSON.
Usage: python slides/out_setup/make_trajectory.py"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "E4_generalization_gradient"))
import e4_setup  # noqa: E402
from helplessness.envs.dynamics import ACTION_NAMES  # noqa: E402
from helplessness.envs.layouts import two_zones  # noqa: E402

UP, RIGHT, DOWN, LEFT, STAY = range(5)
SCALE, TRAIL = 8, 6
VARIED = [RIGHT, DOWN, DOWN, LEFT, UP]  # the same five actions in CD and in UD
ACTIONS = ([LEFT] * 1 + [DOWN]            # corridor (6,11) -> (6,10) -> CD (7,10)
           + VARIED                        # CD: the agent goes where it chooses
           + [UP, UP]                       # back to the corridor
           + [LEFT] * 4 + [UP] * 6 + [LEFT] * 3 + [DOWN]  # corridor to (0,3), then into UD
           + VARIED                        # UD: the same actions, the conveyor pushes down
           + [LEFT] * 3 + [UP] * 6)        # corridor to the goal (0,0)

cfg = e4_setup.load_config()
conf = e4_setup.configuration(cfg, "p1-b-d1")
colors = dict(conf["training_colors"])
colors["nw"], colors["se"] = colors["se"], colors["nw"]  # swapped for the slide: UD red, CD blue
w = e4_setup.world(cfg, conf, "experimental", slot_colors=colors)
env, layout = w.env, w.layout
obs, info = env.reset(start_state=6 * 12 + 11)
states, rewards, regions = [info["state"]], [], [two_zones.region_of_state(layout, info["state"])]
used = []
for a in ACTIONS:
    obs, r, term, trunc, info = env.step(a)
    used.append(a); states.append(info["state"]); rewards.append(r)
    regions.append(two_zones.region_of_state(layout, info["state"]))
    if term:
        break
assert term and states[-1] == two_zones.GOAL_STATE, states

def center(s):
    r, c = divmod(s, 12)
    return ((c + .5) * 6 * SCALE, (r + .5) * 6 * SCALE)

def arrow(d, s, a):
    x, y = center(s)
    if a == STAY:
        d.ellipse([x - 7, y - 7, x + 7, y + 7], outline="white", width=4); return
    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[a]
    L = 34
    tip = (x + dc * L, y + dr * L)
    px, py = -dr, dc  # perpendicular
    head = [tip, (x + dc * (L - 16) + px * 11, y + dr * (L - 16) + py * 11), (x + dc * (L - 16) - px * 11, y + dr * (L - 16) - py * 11)]
    for col, wd in (("black", 9), ("white", 5)):
        d.line([(x, y), (x + dc * (L - 12), y + dr * (L - 12))], fill=col, width=wd)
    d.polygon(head, fill="white", outline="black")

frames = []
for t, s in enumerate(states):
    img = Image.fromarray(w.templates_np[s]).resize((72 * SCALE,) * 2, Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(img)
    pts = [center(q) for q in states[max(0, t - TRAIL):t + 1]]
    for p, q in zip(pts, pts[1:]):
        if abs(p[0] - q[0]) < 100 and abs(p[1] - q[1]) < 100:  # skip torus wraps
            d.line([p, q], fill=(255, 255, 255), width=3)
    frames.append(img)
dur = [250] * len(frames); dur[-1] = 1000
frames[0].save(HERE / "E4_trajectory.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0, disposal=2)
json.dump({"config": "p1-b-d1", "start_state": states[0], "actions": used, "action_names": [ACTION_NAMES[a] for a in used],
           "states": states, "positions_rc": [list(divmod(s, 12)) for s in states], "regions": regions, "rewards": rewards},
          open(HERE / "E4_trajectory.json", "w"))
print(len(frames), "frames", sum(dur) / 1000, "s")
print([(i, regions[i]) for i in range(len(regions)) if i == 0 or regions[i] != regions[i - 1]])
