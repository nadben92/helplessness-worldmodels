"""E5's theme-swap probe set (docs/e5/DECISIONS.md (38)): the probe images of one training configuration in which T_inc
is moved to another quadrant slot, then E4's colour axes are run in that slot. A probe environment has the reference's
dynamics (everything controllable), as every E4/E5 probe.

  swaps      T_inc exchanged with the colour of one other slot m of the canonical training image: for m the neutral
             slots, the old UD slot takes the corridor's colour; for m the CD slot, T_inc and T_ctrl exchange. Each base
             image is the image of a shuffle training assignment (UD at m; CD at the canonical CD slot, or at the old
             UD slot when m is the CD slot), built by two_zones.training_colors as the shuffle's episodes build it.
  themes     in the slot m only, the 11 points of each of E4's three axes (theme_axes.axes_for_orientation), the other
             three slots unchanged. segment and arc_ctrl start at T_inc (t = 0 is the base image); arc_inc starts at
             T_ctrl and stays at distance D from T_inc (a control axis: no slot carries T_inc). t = k / 10 on every axis.

The axes share their endpoints (T_inc starts the segment and arc_ctrl, T_ctrl ends the segment and starts arc_inc): 31
distinct colours per swap, 93 images in all. Every slot of every image is a zone of interest. A point is flagged
near_duplicate when the colour of m lies within NEAR_DUPLICATE_D * D (RGB) of the colour of another slot of the image, up
to the rounding of an axis point to integers (ROUNDING = sqrt(3) / 2 RGB units: segment t >= 0.8 and arc_inc t <= 0.2
with p1-b-d1, whose rounded points sit at 0.2004 D and 0.2005 D from T_ctrl).
Image indices start at SWAP_IMAGE_BASE, so that a swap image never shares a measurement seed with E4's probe images
(0-92) or with the assignments' checkpoint records (pipeline.assignment_records.ASSIGNMENT_IMAGE_BASE + k).
"""

from __future__ import annotations

from helplessness.envs.appearance.theme_axes import rgb_distance
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.probing.theme_probes import AXES

SWAP_IMAGE_BASE = 2000
NEAR_DUPLICATE_D = 0.2
ROUNDING = 3 ** 0.5 / 2


def swap_targets(diagonal: int) -> tuple[str, ...]:
    """The slots T_inc is moved to: every slot but the canonical UD slot, in QUADRANT_ORDER."""
    ud = two_zones.roles(diagonal)["ud"]
    return tuple(q for q in QUADRANT_ORDER if q != ud)


def swap_base(diagonal: int, t_inc, t_ctrl, m: str) -> tuple[dict, str, str]:
    """(slot -> colour, UD slot, CD slot) of the base image of the swap to m: the training image of the assignment with UD
    at m and CD at the canonical CD slot (at the old UD slot when m is the CD slot)."""
    roles = two_zones.roles(diagonal)
    cd = roles["ud"] if m == roles["cd"] else roles["cd"]
    return two_zones.training_colors(diagonal, t_inc, t_ctrl, ud_slot=m, cd_slot=cd), m, cd


def swap_probe_images(diagonal: int, t_inc, t_ctrl, axes: dict) -> list[dict]:
    """The swap images of a configuration, by swap (swap_targets), axis (AXES) and k, identical images merged. Each:
    {"index", "slot_colors", "is_training" (False: not the canonical training image), "probes" [{"axis", "k", "t",
    "position": "swap", "slot": m, "swap": m, "near_duplicate"}], "zones" (the four slots), "swap": m, "ud_slot",
    "cd_slot", "old_ud_slot"}. axes: {"segment", "arc_ctrl", "arc_inc"} -> (11, 3) int arrays."""
    old_ud = two_zones.roles(diagonal)["ud"]
    D = float(rgb_distance(t_inc, t_ctrl))
    key_of = lambda colors: tuple(tuple(colors[q]) for q in QUADRANT_ORDER)  # noqa: E731
    images: dict[tuple, dict] = {}
    for m in swap_targets(diagonal):
        base, ud, cd = swap_base(diagonal, t_inc, t_ctrl, m)
        for axis in AXES:
            points = axes[axis]
            n = len(points)
            for k in range(n):
                colors = dict(base)
                colors[m] = tuple(int(v) for v in points[k])
                near = min(float(rgb_distance(colors[m], colors[q])) for q in QUADRANT_ORDER if q != m) <= NEAR_DUPLICATE_D * D + ROUNDING
                key = key_of(colors)
                if key not in images:
                    images[key] = {"slot_colors": colors, "is_training": False, "probes": [], "zones": list(QUADRANT_ORDER),
                                   "swap": m, "ud_slot": ud, "cd_slot": cd, "old_ud_slot": old_ud}
                images[key]["probes"].append({"axis": axis, "k": k, "t": k / (n - 1), "position": "swap", "slot": m, "swap": m,
                                              "near_duplicate": near})
    out = []
    for i, image in enumerate(images.values()):
        image["index"] = SWAP_IMAGE_BASE + i
        out.append(image)
    return out


def zone_role(image: dict, slot: str) -> str:
    """The role of a slot in the swap image's assignment: "ud" (the slot m), "cd" or "neutral"."""
    return "ud" if slot == image["ud_slot"] else "cd" if slot == image["cd_slot"] else "neutral"
