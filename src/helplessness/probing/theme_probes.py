"""E4's zero-data probe set (docs/E4_generalization_gradient.md), reused by E5 and E7: the probe images of one
training configuration. A probe environment has the reference's dynamics (everything controllable) and the training
image, except the colour of ONE slot:

  themes     the 11 points of each of the three axes (theme_axes.axes_for_orientation: segment T_inc -> T_ctrl, arc
             centred on T_ctrl from T_inc, arc centred on T_inc from T_ctrl);
  positions  the former UD slot, the former CD slot, the neutral slot (two_zones.roles);
  plus       "position only": the corridor's colour at the former UD slot.

That is 3 x 11 x 3 + 1 = 100 probes. Images are identified by their slot -> colour map (the image is a pure function
of it, the agent colour being fixed), so identical images are merged and keep the list of every probe they stand for.
The axes share their endpoints (T_inc starts the segment and arc_ctrl, T_ctrl ends the segment and starts arc_inc),
and T_inc at the UD slot and T_ctrl at the CD slot both give back the training image: 93 distinct images. The zone of
interest of a probe is the slot it changes; the training image has the four slots, each in turn.
"""

from __future__ import annotations

from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER

AXES = ("segment", "arc_ctrl", "arc_inc")
POSITIONS = ("ud", "cd", "neutral")


def probe_images(diagonal: int, t_inc, t_ctrl, axes: dict) -> list[dict]:
    """The distinct probe images of a configuration, the training image first. Each: {"index", "slot_colors" {slot:
    (r, g, b)}, "is_training", "probes" [{"axis", "k", "t", "position", "slot"}], "zones" [slots of interest]}.
    axes: {"segment", "arc_ctrl", "arc_inc"} -> (11, 3) int arrays, each from its own start. t = k / 10."""
    roles = two_zones.roles(diagonal)
    training = two_zones.training_colors(diagonal, t_inc, t_ctrl)
    key_of = lambda colors: tuple(tuple(colors[q]) for q in QUADRANT_ORDER)  # noqa: E731
    images: dict[tuple, dict] = {}

    def add(colors: dict, probe: dict | None, zone: str | None) -> None:
        key = key_of(colors)
        if key not in images:
            images[key] = {"slot_colors": dict(colors), "is_training": key == key_of(training), "probes": [], "zones": []}
        image = images[key]
        if probe is not None:
            image["probes"].append(probe)
        if zone is not None and zone not in image["zones"]:
            image["zones"].append(zone)

    add(training, None, None)
    for q in QUADRANT_ORDER:  # the training image: every slot in turn
        add(training, None, q)
    for axis in AXES:
        points = axes[axis]
        n = len(points)
        for k in range(n):
            color = tuple(int(v) for v in points[k])
            for position in POSITIONS:
                slot = roles[position]
                colors = dict(training)
                colors[slot] = color
                add(colors, {"axis": axis, "k": k, "t": k / (n - 1), "position": position, "slot": slot}, slot)
    colors = dict(training)
    colors[roles["ud"]] = tuple(int(v) for v in CORRIDOR_COLOR)
    add(colors, {"axis": "position_only", "k": 0, "t": 0.0, "position": "ud", "slot": roles["ud"]}, roles["ud"])
    out = []
    for index, image in enumerate(images.values()):
        image["index"] = index
        image["zones"] = [q for q in QUADRANT_ORDER if q in image["zones"]]
        out.append(image)
    return out
