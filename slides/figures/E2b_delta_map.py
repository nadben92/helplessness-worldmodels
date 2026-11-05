"""E2b -- Map of Delta (online minus matched, final checkpoint, 20 seeds, canonical zone frame): exactly 0 on every
corridor state, non-zero only inside the zones."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _grid as G  # noqa: E402

FIG_ID = "E2b_delta_map"
CSV = ("experiments", "E2_online_4zones", "artifacts", "figures", "csv")


def grid(name):
    g = np.full((12, 12), np.nan)
    for r in C.read_csv(C.path(*CSV, name)):
        g[int(r["row"]), int(r["col"])] = C.num(r["value"])
    return g


def main():
    full_map, ext = grid("F4_delta_diff.csv"), grid("F4_delta_diff_exterior.csv")
    types = G.four_zone_types()
    corridor = types == "corridor"
    corridor[0, 0] = False  # the goal: terminal, no Delta
    vals = ext[corridor]
    assert np.all(vals == 0.0), "exterior Delta is not exactly 0"
    cmap = LinearSegmentedColormap.from_list("div", [P.REFERENCE, P.BG, P.EXPOSED])
    vmax = np.nanmax(np.abs(full_map))
    norm = TwoSlopeNorm(0.0, -vmax, vmax)

    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "Outside the zones, online and matched agents act identically",
            "Δ (value lost by the greedy policy), online − matched, final checkpoint, mean of 20 seeds")
    ax = fig.add_axes([0.06, 0.1, 0.48, 0.72])
    colors = {k: P.ZONE_CORRIDOR for k in P.ZONE_COLORS}
    G.draw_cells(ax, types, colors=colors, outline=True)
    for r in range(12):
        for c in range(12):
            if types[r, c] != "corridor" and np.isfinite(full_map[r, c]):
                ax.add_patch(C.plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=cmap(norm(full_map[r, c])), ec=P.BG,
                                             lw=0.8, zorder=1.5))
            elif corridor[r, c]:
                ax.text(c, r, "0", ha="center", va="center", fontsize=P.SIZE_SMALL - 1.5, color=P.TEXT_2, zorder=3)
    for code in ("UD", "CN", "CD", "UN"):
        rr, cc = (types == code).nonzero()
        ax.text(cc.min() - 0.35, rr.min() - 0.3, code, fontsize=P.SIZE_ANNOT, fontweight="bold", color=P.TEXT,
                ha="left", va="top", zorder=7)
    cax = fig.add_axes([0.58, 0.18, 0.015, 0.5])
    sm = C.plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=P.SIZE_SMALL)
    cb.set_label("Δ online − matched (inside zones)", fontsize=P.SIZE_ANNOT)
    fig.text(0.68, 0.62, f"{corridor.sum()} corridor states", fontsize=P.SIZE_AXIS + 2, fontweight="bold",
             color=P.TEXT)
    fig.text(0.68, 0.55, "Δ = 0 exactly in every one", fontsize=P.SIZE_AXIS + 2, color=P.TEXT)
    fig.text(0.68, 0.43, f"inside zones: {C.fmt(np.nanmin(full_map[~(types == 'corridor')]), 3, True)}"
                         f" to {C.fmt(np.nanmax(full_map[~(types == 'corridor')]), 3, True)}",
             fontsize=P.SIZE_ANNOT, color=P.TEXT_2)
    C.footnote(fig, "Canonical zone-type frame (each seed remapped to its zone type's quadrant before averaging; "
                    "4 rotations × 5 seeds). Goal state: terminal, no Δ. Source: experiments/E2_online_4zones/"
                    "artifacts/figures/csv/F4_delta_diff.csv and F4_delta_diff_exterior.csv.", y=0.025)
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
