"""E6a -- Typical episodes from the same start in the old UD quadrant (training image, all-controllable test world,
frozen weights): one exposed (wanders), one reference (heads for the goal).

Choice, fixed before looking at the paths: the agent whose exposed directed share (points.csv) is closest to the mean
of the four agents; the start = the centre of the quadrant; for each model, among its 20 episodes from that start, the
one whose own directed share is closest to the model's pooled share on the image (ties: lowest episode index).
Directed shares from measures.zone_phenotypes."""
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import common as C, palette as P  # noqa: E402
import _grid as G  # noqa: E402

FIG_ID = "E6a_trajectories"
POINTS = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
EPISODES = ("scratch", "e6dev", "results", "E6_phenotypes_latent")
STAMP = "20261023T134444Z"  # the v2 run whose light outputs are in results/E6_phenotypes_latent/dev_e4_cpu_v2/runs


def choose():
    from helplessness.envs.layouts import two_zones
    from helplessness.measures.zone_phenotypes import directed_share
    pts = [r for r in C.read_csv(C.path(*POINTS)) if r["image"] == "training"]
    exp = {r["agent"]: C.num(r["directed_share"]) for r in pts if r["role"] == "exposed"}
    ref = {r["agent"]: C.num(r["directed_share"]) for r in pts if r["role"] == "reference"}
    target = np.mean(list(exp.values()))
    agent = min(sorted(exp), key=lambda a: abs(exp[a] - target))
    ep = np.load(C.path(*EPISODES, f"e6__{agent}__{STAMP}", "episodes.npz"))
    layout = two_zones.build_layout(1, "reference")
    zone_states = two_zones.region_states(layout)[two_zones.roles(1)["ud"]]
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[zone_states] = True
    dist = np.array([two_zones.toric_distance_to_goal(s) for s in range(two_zones.N_STATES)])
    r, c = np.divmod(zone_states, 12)
    start = int(round(r.mean())) * 12 + int(round(c.mean()))
    out = {"agent": agent, "start": start, "pooled": {"exposed": exp[agent], "reference": ref[agent]}}
    for role in ("exposed", "reference"):
        S, L = ep[f"training__{role}__states"], ep[f"training__{role}__length"]
        idx = np.flatnonzero(S[:, 0] == start)
        ds = directed_share(S[idx], L[idx], zone, dist)
        share = ds["n_directed"] / np.maximum(ds["n_zone_steps"], 1)
        k = int(np.argmin(np.abs(share - out["pooled"][role])))
        i = int(idx[k])
        out[role] = {"index": i, "states": S[i, :L[i] + 1], "share": float(share[k]),
                     "terminated": bool(ep[f"training__{role}__terminated"][i]), "length": int(L[i]),
                     "median_share": float(np.median(share))}
    return out


def draw_path(ax, states, color):
    rc = np.array([divmod(int(s), 12) for s in states], float)
    n = len(rc)
    cmap = LinearSegmentedColormap.from_list("t", ["#F1E4DA" if color == P.EXPOSED else "#DCE6EE", color])
    rng = np.random.default_rng(3)  # small offsets so revisits stay visible
    jit = rng.uniform(-0.07, 0.07, rc.shape)
    jit[0] = 0
    pts = rc + jit
    segs, cols = [], []
    for t in range(n - 1):
        if np.abs(rc[t + 1] - rc[t]).sum() > 1:  # wrap around the torus: do not draw across the grid
            continue
        segs.append([[pts[t, 1], pts[t, 0]], [pts[t + 1, 1], pts[t + 1, 0]]])
        cols.append(cmap(0.25 + 0.75 * t / max(n - 2, 1)))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=4, capstyle="round", zorder=7))
    ax.scatter([pts[0, 1]], [pts[0, 0]], s=170, color=P.BG, edgecolor=color, linewidth=3, zorder=8)
    ax.scatter([pts[-1, 1]], [pts[-1, 0]], s=60, color=color, zorder=8)


def main():
    ch = choose()
    types = G.two_zone_types(1)
    types[types == "UD"] = "UD"
    fig = C.plt.figure(figsize=C.FULL)
    C.title(fig, "Same start, same controllable world: one wanders, one goes home",
            f"Training image, start at the centre of the old conveyor quadrant (agent {ch['agent'].replace('_e4', '')})")
    for k, (role, col, name) in enumerate((("exposed", P.EXPOSED, "Exposed"), ("reference", P.REFERENCE, "Reference"))):
        ax = fig.add_axes([0.05 + 0.48 * k, 0.15, 0.42, 0.62])
        G.draw_cells(ax, types, colors={**P.ZONE_COLORS, "UD": "#EED3C6"})
        draw_path(ax, ch[role]["states"], col)
        e = ch[role]
        ax.set_title(f"{name}: {e['length']} steps, {'reached the goal' if e['terminated'] else 'did not reach the goal'}",
                     color=col, fontsize=P.SIZE_AXIS, loc="center", pad=26, fontweight="bold")
        ax.text(5.5, -0.75, f"directed share {e['share']:.2f} (model's pooled share {ch['pooled'][role]:.2f})",
                ha="center", va="bottom", fontsize=P.SIZE_SMALL, color=P.TEXT_2)
    C.footnote(fig, "Old UD quadrant shaded (fully controllable at test). Light → dark: time. Chosen by rule: agent closest to the "
                    "mean exposed directed share; per model, the episode from this start closest to its pooled share. Source: scratch/e6dev/results/E6_phenotypes_latent/"
                    f"e6__{ch['agent']}__{STAMP}/episodes.npz.", y=0.02)
    print(f"E6a: agent {ch['agent']}, start {ch['start']}, exposed ep {ch['exposed']['index']} "
          f"share {ch['exposed']['share']:.3f}, reference ep {ch['reference']['index']} share {ch['reference']['share']:.3f}")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
