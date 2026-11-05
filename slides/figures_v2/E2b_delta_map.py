"""E2b (v2) -- Map of Delta (online - matched, final checkpoint, mean of 20 seeds, canonical zone frame): the corridor
cells are exactly 0 (checked), zone cells coloured on a diverging scale; zone codes and a thin colour bar."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "E2b_delta_map"
CSV = ("experiments", "E2_online_4zones", "artifacts", "figures", "csv")


def grid(name):
    g = np.full((12, 12), np.nan)
    for r in C.read_csv(C.path(*CSV, name)):
        g[int(r["row"]), int(r["col"])] = C.num(r["value"])
    return g


def main():
    full_map, ext = grid("F4_delta_diff.csv"), grid("F4_delta_diff_exterior.csv")
    types = D.four_zone_types()
    corridor = types == "corridor"
    corridor[0, 0] = False
    assert np.all(ext[corridor] == 0.0), "exterior Delta is not exactly 0"
    cmap = LinearSegmentedColormap.from_list("div", [P.REFERENCE, "white", P.EXPOSED])
    vmax = np.nanmax(np.abs(full_map))
    norm = TwoSlopeNorm(0.0, -vmax, vmax)
    fig = C.plt.figure(figsize=C.FULL)
    ax = fig.add_axes([0.25, 0.04, 0.46, 0.92])
    colors = {k: "white" for k in P.ZONE_COLORS}
    colors["corridor"] = cmap(norm(0.0))
    D.draw_cells(ax, types, colors=colors, lw=0.6, edge="#BBBBBB")
    for r in range(12):
        for c in range(12):
            if types[r, c] != "corridor" and np.isfinite(full_map[r, c]):
                ax.add_patch(C.plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=cmap(norm(full_map[r, c])), ec="#BBBBBB",
                                             lw=0.6))
    for code in ("UD", "CN", "CD", "UN"):
        rr, cc = (types == code).nonzero()
        ax.add_patch(C.plt.Rectangle((cc.min() - 0.5, rr.min() - 0.5), 5, 5, fill=False, ec="black", lw=2.2,
                                     zorder=5))
        ax.text(cc.mean(), rr.mean(), code, ha="center", va="center", fontsize=P.SIZE_BLOCK + 2, fontweight="bold",
                color="black", alpha=0.35, zorder=6)
    cax = fig.add_axes([0.75, 0.2, 0.015, 0.6])
    cb = fig.colorbar(C.plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("Δ online − matched", fontsize=P.SIZE_LABEL)
    cb.ax.tick_params(labelsize=P.SIZE_TICK)
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
