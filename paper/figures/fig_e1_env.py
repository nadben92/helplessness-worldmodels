"""E1: the environment. The 12x12 torus with the corridor, the goal and the four 5x5 blocks, each labelled with its
dynamics type; arrows show the drift of the two uncontrollable blocks for the retained direction (away from the goal).
Writes paper/figures/out/fig_e1_env.pdf.

Source: configs/zones/e1_4zones.yaml and experiments/E1_offline_4zones/config.yaml, through E1's own
env.build_layout (the drift of each uncontrollable block is the one its dynamics object resolved).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb

import common
import e1_sources as src

NAME = "fig_e1_env"
CORRIDOR_COLOR = "#DDDDDD"


def main() -> None:
    cfg = src.zones_config()
    lay = src.layout()
    n_rows, n_cols = cfg["grid_shape"]
    image = np.ones((n_rows, n_cols, 3))
    common.apply_style()
    fig, ax = common.figure(width_fraction=0.5, aspect=1.0)
    for zone in lay.zones:
        category = zone.appearance.category
        color = CORRIDOR_COLOR if category == "corridor" else common.ZONE_COLOR[category]
        image[zone.mask] = to_rgb(color)
    ax.imshow(image, interpolation="nearest")
    for zone in lay.zones:
        category = zone.appearance.category
        if category == "corridor":
            continue
        rows, cols = np.nonzero(zone.mask)
        r0, c0 = rows.mean(), cols.mean()
        drift = getattr(zone.dynamics, "drift", None)
        label_y = r0 + (1.2 if drift is not None else 0)
        ax.text(c0, label_y, src.ZONE_ABBR[category], ha="center", va="center", color="white",
                fontsize=common.SMALL_FONT_SIZE)
        if drift is not None:
            dr, dc = drift
            ax.annotate("", xy=(c0 + 1.3 * dc, r0 - 0.4 + 1.3 * dr), xytext=(c0, r0 - 0.4),
                        arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 1.2})
    g_r, g_c = cfg["goal"]
    ax.plot(g_c, g_r, marker="*", color="black", ms=9)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.tick_params(labelsize=common.SMALL_FONT_SIZE - 2, length=0)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
