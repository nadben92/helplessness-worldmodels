"""Renders the GridWorld layout (walls, start, goal, traps) to a PNG.

General utility, not tied to one experiment (see scripts/regenerate_agents.py
for the same rationale) -- the layout is identical across the offline and
online protocols (only the action count differs), so one figure documents
both. Output: reports/gridworld_layout.png, alongside the other report
figures (see CONTRIBUTING.md, "what gets versioned").
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from helplessness.envs.gridworld import GridWorld

REPO_ROOT = Path(__file__).resolve().parent.parent

CELL_COLORS = {
    "S": "#cfe3f7",  # light blue
    "G": "#bfe6c4",  # light green
    "T": "#f3b6b6",  # light red
    "#": "#4a4a4a",  # wall
    ".": "#fafafa",  # empty
}
CELL_LABELS = {"S": "S", "G": "G", "T": "T", "#": "", ".": ""}


def render_gridworld(env: GridWorld, ax=None, annotate_states: bool = True, title: str | None = None):
    """Draws env's layout on a matplotlib Axes (created if not given).
    Returns the Axes. Reads only env.grid/n_rows/n_cols -- no side effects
    on the environment (consistent with the analysis-functions invariant in
    CLAUDE.md, even though this is a plotting helper, not a metric)."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=(env.n_cols, env.n_rows))

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            ch = str(env.grid[r, c])
            ax.add_patch(Rectangle((c, env.n_rows - 1 - r), 1, 1,
                                    facecolor=CELL_COLORS[ch], edgecolor="#999999", linewidth=1))
            label = CELL_LABELS[ch]
            if label:
                ax.text(c + 0.5, env.n_rows - 1 - r + 0.5, label,
                        ha="center", va="center", fontsize=18, fontweight="bold", color="#222222")
            if annotate_states and ch != "#":
                s = env.rc_to_s((r, c))
                ax.text(c + 0.08, env.n_rows - 1 - r + 0.88, str(s),
                        ha="left", va="top", fontsize=8, color="#666666")

    ax.set_xlim(0, env.n_cols)
    ax.set_ylim(0, env.n_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, wrap=True)
    return ax


def main() -> None:
    env = GridWorld()
    ax = render_gridworld(
        env,
        title=f"GridWorld default layout  (goal_reward={env.goal_reward:+.2f}, trap_reward={env.trap_reward:+.2f}, "
              f"step_reward={env.step_reward:+.2f})",
    )
    out_path = REPO_ROOT / "reports" / "gridworld_layout.png"
    ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
