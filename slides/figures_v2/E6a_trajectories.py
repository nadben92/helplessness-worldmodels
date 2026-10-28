"""E6a (v2) -- Exposed (left) and reference (right) episodes from the same start in the old UD quadrant (training
image, controllable test world). Same choice rule as v1: the agent whose exposed directed share is closest to the mean
of the four; start = quadrant centre; per model, the episode from that start whose directed share is closest to the
model's pooled share (ties: lowest index). Directed shares from measures.zone_phenotypes."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402

FIG_ID = "E6a_trajectories"
POINTS = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
EPISODES = ("scratch", "e6dev", "results", "E6_phenotypes_latent")
STAMP = "20261023T134444Z"


def choose():
    from helplessness.envs.layouts import two_zones
    from helplessness.measures.zone_phenotypes import directed_share
    pts = [r for r in C.read_csv(C.path(*POINTS)) if r["image"] == "training"]
    pooled = {(r["agent"], r["role"]): C.num(r["directed_share"]) for r in pts}
    exp = {a: v for (a, role), v in pooled.items() if role == "exposed"}
    agent = min(sorted(exp), key=lambda a: abs(exp[a] - np.mean(list(exp.values()))))
    ep = np.load(C.path(*EPISODES, f"e6__{agent}__{STAMP}", "episodes.npz"))
    zone_states = two_zones.region_states(two_zones.build_layout(1, "reference"))[two_zones.roles(1)["ud"]]
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[zone_states] = True
    dist = np.array([two_zones.toric_distance_to_goal(s) for s in range(two_zones.N_STATES)])
    r, c = np.divmod(zone_states, 12)
    start = int(round(r.mean())) * 12 + int(round(c.mean()))
    out = {"agent": agent}
    for role in ("exposed", "reference"):
        S, L = ep[f"training__{role}__states"], ep[f"training__{role}__length"]
        idx = np.flatnonzero(S[:, 0] == start)
        ds = directed_share(S[idx], L[idx], zone, dist)
        share = ds["n_directed"] / np.maximum(ds["n_zone_steps"], 1)
        i = int(idx[int(np.argmin(np.abs(share - pooled[(agent, role)])))])
        out[role] = S[i, :L[i] + 1]
    return out


def draw_path(ax, states, color):
    rc = np.array([divmod(int(s), 12) for s in states], float)
    pts = rc + np.random.default_rng(3).uniform(-0.07, 0.07, rc.shape) * (np.arange(len(rc)) > 0)[:, None]
    cmap = LinearSegmentedColormap.from_list("t", ["white", color])
    segs, cols = [], []
    for t in range(len(rc) - 1):
        if np.abs(rc[t + 1] - rc[t]).sum() > 1:  # wrap around the torus
            continue
        segs.append([[pts[t, 1], pts[t, 0]], [pts[t + 1, 1], pts[t + 1, 0]]])
        cols.append(cmap(0.3 + 0.7 * t / max(len(rc) - 2, 1)))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=5, capstyle="round", zorder=7))
    ax.scatter([pts[0, 1]], [pts[0, 0]], s=220, color="white", edgecolor=color, linewidth=3.5, zorder=8)


def main():
    ch = choose()
    types = D.two_zone_types(1)
    fig = C.plt.figure(figsize=C.FULL)
    for k, (role, col) in enumerate((("exposed", P.EXPOSED), ("reference", P.REFERENCE))):
        ax = fig.add_axes([0.04 + 0.5 * k, 0.04, 0.42, 0.92])
        D.draw_cells(ax, types, colors={**P.ZONE_COLORS, "UD": "#FFC2B3", "CD": "#BFDDFF"})
        draw_path(ax, ch[role], col)
    print(f"E6a v2: agent {ch['agent']}")
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
