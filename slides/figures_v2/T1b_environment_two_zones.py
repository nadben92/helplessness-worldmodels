"""T1b (v2) -- E4's two-zone environment, exposed (left) and paired reference (right), rendered with the real training
themes of configuration p1-a-d1 (T_inc, T_ctrl read from probes_b_step1.csv; slot colours from
two_zones.training_colors). The reference keeps exactly the exposed model's appearance: the panels differ only by the
dynamics, i.e. the conveyor arrows (read from the tensor) drawn in the exposed panel's UD slot."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "T1b_environment_two_zones"
SRC = ("results", "E4_generalization_gradient", "final", "tables", "probes_b_step1.csv")
CONFIG = "p1-a-d1"


def rgb(s: str) -> tuple[int, int, int]:
    return tuple(int(v) for v in s.split("-"))


def slot_types_and_colors():
    """(12, 12) slot names (corridor cells: 'corridor') and slot -> RGB (0-1) of the training image."""
    from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
    from helplessness.envs.layouts import two_zones
    row = next(r for r in C.read_csv(C.path(*SRC)) if r["configuration"] == CONFIG)
    slot_rgb = two_zones.training_colors(1, rgb(row["T_inc"]), rgb(row["T_ctrl"]))
    regions = two_zones.region_states(two_zones.build_layout(1, "experimental"))
    types = np.full((12, 12), "corridor", dtype=object)
    for slot in slot_rgb:
        r, c = np.divmod(regions[slot], 12)
        types[r, c] = slot
    colors = {slot: tuple(v / 255 for v in col) for slot, col in slot_rgb.items()}
    colors["corridor"] = tuple(v / 255 for v in CORRIDOR_COLOR)
    return types, colors, slot_rgb


def main():
    types, colors, slot_rgb = slot_types_and_colors()
    from helplessness.envs.layouts import two_zones
    ud = two_zones.roles(1)["ud"]
    fig = C.plt.figure(figsize=C.FULL)
    for k, (cond, name, col) in enumerate((("experimental", "exposed", P.EXPOSED),
                                            ("reference", "reference", P.REFERENCE))):
        ax = D.canvas(fig, (0.04 + 0.49 * k, 0.04, 0.43, 0.82))
        ax.set_xlim(-0.5, 11.5)
        D.draw_cells(ax, types, colors=colors)
        if cond == "experimental":
            dyn = np.where(types == ud, "UD", "other")
            D.conveyor_arrows(ax, D.two_zone_tensor(1, cond), dyn, "UD", color="black")
        fig.text(0.04 + 0.49 * k + 0.215, 0.93, name, ha="center", fontsize=P.SIZE_BLOCK + 4, fontweight="bold",
                 color=col)
    print(f"T1b v2: {CONFIG} slot colours {slot_rgb}")
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
