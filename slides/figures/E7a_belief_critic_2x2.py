"""E7a -- The 2 x 2 dissociation (E4's four agents, training image, directed share epsilon-greedy in the zone): rows =
transition belief (true / conveyor), columns = critic (reference / exposed), for look-ahead k = 1 and k = 5. Mean of
the 4 agents, range in brackets (table_2x2.csv)."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import common as C, palette as P  # noqa: E402

FIG_ID = "E7a_belief_critic_2x2"
SRC = ("results", "E7_dissociation", "dev_e4_cpu", "analysis", "table_2x2.csv")
CELLS = {(0, 0): "true_belief__ref_critic", (0, 1): "true_belief__exp_critic (oracle D4)",
         (1, 0): "conveyor__ref_critic (anti-oracle p=1)", (1, 1): "conveyor__exp_critic"}


def main():
    rows = C.read_csv(C.path(*SRC))
    cmap = LinearSegmentedColormap.from_list("d", [P.BG, P.REFERENCE])
    norm = Normalize(0.3, 1.0)
    fig, axes = C.plt.subplots(1, 2, figsize=C.FULL)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.72, bottom=0.17, wspace=0.12)
    C.title(fig, "Behaviour follows the transition belief, not the critic",
            "Share of directed steps in the old UD quadrant, mean of 4 agents [range]")
    for ax, k in zip(axes, (1, 5)):
        agents = [r for r in rows if r["k"] == str(k) and r["agent"] != "mean"]
        mean = next(r for r in rows if r["k"] == str(k) and r["agent"] == "mean")
        for (i, j), col in CELLS.items():
            v = C.num(mean[col])
            vals = np.array([C.num(r[col]) for r in agents])
            ax.add_patch(C.plt.Rectangle((j, i), 1, 1, fc=cmap(norm(v)), ec=P.BG, lw=5))
            dark = v > 0.75
            ax.text(j + 0.5, i + 0.45, f"{v:.2f}", ha="center", va="center", fontsize=30, fontweight="bold",
                    color=P.BG if dark else P.TEXT)
            ax.text(j + 0.5, i + 0.72, f"[{vals.min():.2f}–{vals.max():.2f}]", ha="center", va="center",
                    fontsize=P.SIZE_ANNOT, color=P.BG if dark else P.TEXT_2)
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_xticks([0.5, 1.5], ["reference\ncritic", "exposed\ncritic"])
        ax.xaxis.tick_top()
        ax.set_yticks([0.5, 1.5], ["true\nbelief", "conveyor\nbelief"] if k == 1 else ["", ""])
        ax.tick_params(length=0, labelsize=P.SIZE_AXIS)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_aspect("equal")
        ax.set_title(f"look-ahead k = {k}", loc="center", fontsize=P.SIZE_AXIS + 1, pad=64, color=P.TEXT)
    C.footnote(fig, "Conveyor row: chance floor by construction (under the conveyor belief every action predicts the same "
                    "cell, Q values tie, choice is uniform: 2 of 5 actions are directed). True belief with the exposed "
                    "critic: oracle transitions (D4). Source: results/E7_dissociation/dev_e4_cpu/analysis/table_2x2.csv.")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
