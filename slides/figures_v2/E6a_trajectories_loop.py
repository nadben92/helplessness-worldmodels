"""E6a, loop version (v2) -- the same panels as E6a_trajectories (same agent p2-b, same start: the old UD quadrant's centre,
training image, controllable test world; the reference's episode unchanged, 6 steps), with the exposed episode chosen to
SHOW the loop in the zone (user's request, 2026-10-28): among the exposed episodes from that start that end at the goal,
those whose path after the first exit from the zone goes straight to the goal (steps after the exit = the exit cell's
distance to the goal, no return to the zone) in at most MAX_CORRIDOR steps (an exit on the goal's side: no long walk
round the torus in the corridor); among them the most loops in the zone (stays removed -- a stay draws nothing --, a loop =
a return to a cell last visited >= 3 moves earlier), then the most steps in the zone, then the length closest to 20
(E6a's), then the lowest index. An illustration chosen after looking at the episodes, not a typical one: the
manifest gives how many episodes qualify. E6a_trajectories (the typical episode) is not overwritten."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import _draw as D  # noqa: E402
import E6a_trajectories as E6A  # noqa: E402  (its choice of agent and reference episode, its drawing)

FIG_ID = "E6a_trajectories_loop"
LENGTH = 20
MAX_CORRIDOR = 4


def n_loops(states) -> int:
    """Returns to a cell last visited >= 3 moves earlier, stays removed."""
    moves = [int(x) for k, x in enumerate(states) if k == 0 or x != states[k - 1]]
    last, loops = {}, 0
    for k, x in enumerate(moves):
        loops += x in last and k - last[x] >= 3
        last[x] = k
    return loops


def choose_loop() -> dict:
    from helplessness.envs.layouts import two_zones
    ch = E6A.choose()  # agent p2-b and the reference's episode, unchanged
    ep = np.load(C.path(*E6A.EPISODES, f"e6__{ch['agent']}__{E6A.STAMP}", "episodes.npz"))
    zone_states = two_zones.region_states(two_zones.build_layout(1, "reference"))[two_zones.roles(1)["ud"]]
    zone = np.zeros(two_zones.N_STATES, bool)
    zone[zone_states] = True
    dist = np.array([two_zones.toric_distance_to_goal(s) for s in range(two_zones.N_STATES)])
    start = int(ch["exposed"][0])
    S, L, T = ep["training__exposed__states"], ep["training__exposed__length"], ep["training__exposed__terminated"]
    rows = []
    for i in np.flatnonzero(S[:, 0] == start):
        s = S[i, :L[i] + 1]
        inz = zone[s]
        exit_t = int(np.argmax(~inz)) if (~inz).any() else len(s)
        direct = bool(T[i]) and exit_t < len(s) and not inz[exit_t:].any() and (len(s) - 1 - exit_t) == dist[s[exit_t]]
        rows.append({"index": int(i), "length": int(L[i]), "goal": bool(T[i]), "direct_exit": direct, "zone_steps": exit_t,
                     "distinct_zone_cells": len(set(s[:exit_t].tolist())), "loops": n_loops(s[:exit_t])})
    ok = [r for r in rows if r["goal"] and r["direct_exit"] and r["length"] - r["zone_steps"] <= MAX_CORRIDOR]
    if not ok:
        raise SystemExit(f"no exposed episode with a direct exit in <= {MAX_CORRIDOR} corridor steps")
    best = min(ok, key=lambda r: (-r["loops"], -r["zone_steps"], abs(r["length"] - LENGTH), r["index"]))
    s = S[best["index"], :L[best["index"]] + 1]
    return {**ch, "exposed": s, "exposed_length": best["length"], "exposed_index": best["index"], "rows": rows, "best": best}


def main():
    ch = choose_loop()
    types = D.two_zone_types(1)
    fig = C.plt.figure(figsize=C.FULL)
    for k, (role, col) in enumerate((("exposed", P.EXPOSED), ("reference", P.REFERENCE))):
        ax = fig.add_axes([0.04 + 0.5 * k, 0.03, 0.42, 0.84])
        D.draw_cells(ax, types, colors={**P.ZONE_COLORS, "UD": "#FFC2B3", "CD": "#BFDDFF"})
        E6A.draw_path(ax, ch[role], col)
        fig.text(0.04 + 0.5 * k + 0.21, 0.92, f"{ch[role + '_length']} steps", ha="center", fontsize=P.SIZE_LABEL)
    rows = ch["rows"]
    print(f"E6a loop: agent {ch['agent']}, exposed ep {ch['exposed_index']} ({ch['best']}), path {ch['exposed'].tolist()}; "
          f"reference ep {ch['reference_index']} {ch['reference_length']} steps")
    print(f"  exposed episodes from the start: {len(rows)}; at the goal: {sum(r['goal'] for r in rows)}; direct exit: "
          f"{sum(r['direct_exit'] for r in rows)}; direct exit in <= {MAX_CORRIDOR} corridor steps: "
          f"{sum(r['direct_exit'] and r['length'] - r['zone_steps'] <= MAX_CORRIDOR for r in rows)}; "
          f"zone steps median {np.median([r['zone_steps'] for r in rows]):.1f}")
    return C.save(fig, FIG_ID, tight=False)


if __name__ == "__main__":
    main()
