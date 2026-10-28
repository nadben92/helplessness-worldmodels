"""Context walks ending at a given state of a zone (E7's situations, docs/E7_latent_manipulation.md): numpy only, no
model. New module (2026-10-23); probing/walks.py (E4's fixed walks) is unchanged.

A context walk for a target state s of the zone, with deterministic successors succ (n_states, n_actions):
  1. a start drawn uniformly in the corridor (goal excluded), then n_pre corridor steps (n_pre uniform in
     [n_pre_min, n_pre_max]) of one of two kinds: "random" (uniform actions that stay in the corridor, `stay` allowed) or
     "goal" (each step an action that lowers the distance to the goal while staying in the corridor if one exists, else
     a uniform corridor action -- a walk heading for the goal);
  2. an entry SIDE of the zone (north, east, south, west, by the context's index in turn): the shortest corridor path
     (breadth-first, actions in index order, goal excluded) to a corridor cell from which one action enters the zone on
     that side, at the side's border cell nearest to s (ties: the first in state order);
  3. the shortest path inside the zone from the entry cell to s (breadth-first within the zone).
These contexts start in the corridor and reach s after a varied history -- unlike E6's episodes, which start in the
zone from the initial state. Output as probing.walks.generate_walks (states, actions, phase, ...), so
probing.fixed_trajectory.fixed_trajectory_context filters them unchanged; the last state of every walk is its target.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from helplessness.probing.walks import PHASE_PAD, PHASE_PATH, PHASE_PRE, PHASE_ZONE

SIDES = ("north", "east", "south", "west")
SIDE_STEP = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}


def _bfs(succ: np.ndarray, start: int, allowed: set, goal_test) -> list[tuple[int, int]]:
    parent = {start: None}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if goal_test(s):
            path = []
            while parent[s] is not None:
                prev, a = parent[s]
                path.append((prev, a))
                s = prev
            return path[::-1]
        for a in range(succ.shape[1]):
            nxt = int(succ[s, a])
            if nxt in allowed and nxt not in parent:
                parent[nxt] = (s, a)
                queue.append(nxt)
    raise ValueError(f"no path from {start}")


def side_entries(succ: np.ndarray, zone: np.ndarray, corridor: np.ndarray, grid_cols: int) -> dict[str, list[tuple[int, int, int]]]:
    """side -> [(corridor cell c, action a, zone cell e = succ[c, a])] for the moves entering the zone across that side
    (the move's direction is opposite to the side: entering across the north side is a move south)."""
    zone_set, corr = set(int(s) for s in zone), set(int(s) for s in corridor)
    out = {side: [] for side in SIDES}
    for c in sorted(corr):
        rc, cc = divmod(c, grid_cols)
        for a in range(succ.shape[1]):
            e = int(succ[c, a])
            if e in zone_set:
                re, ce = divmod(e, grid_cols)
                dr, dc = re - rc, ce - cc
                dr, dc = (int(np.sign(dr)) if abs(dr) <= 1 else -int(np.sign(dr))), (int(np.sign(dc)) if abs(dc) <= 1 else -int(np.sign(dc)))
                side = {(1, 0): "north", (-1, 0): "south", (0, 1): "west", (0, -1): "east"}.get((dr, dc))
                if side is not None:
                    out[side].append((c, a, e))
    return out


def context_walks(succ: np.ndarray, zone: np.ndarray, corridor: np.ndarray, dist: np.ndarray, targets: np.ndarray, n_per_target: int,
                  rng: np.random.Generator, grid_cols: int, n_pre_min: int = 8, n_pre_max: int = 32) -> dict[str, np.ndarray]:
    """n_per_target walks for each target state (module docstring); context index c of a target: side SIDES[c % 4], kind
    "random" for even c // 4 + c, "goal" otherwise (the kinds alternate and every side meets both kinds). Returns the
    walks' arrays (as probing.walks.generate_walks) plus target, side and kind per walk."""
    zone_set, corr_set = set(int(s) for s in zone), set(int(s) for s in corridor)
    entries = side_entries(succ, zone, corridor, grid_cols)
    walks, meta = [], []
    for s_target in np.asarray(targets):
        for c in range(n_per_target):
            side = SIDES[c % 4]
            kind = "random" if (c // 4 + c) % 2 == 0 else "goal"
            s = int(corridor[rng.integers(len(corridor))])
            states, actions, phases = [s], [], [PHASE_PRE]
            for _ in range(int(rng.integers(n_pre_min, n_pre_max + 1))):
                options = [a for a in range(succ.shape[1]) if int(succ[s, a]) in corr_set]
                closer = [a for a in options if dist[int(succ[s, a])] < dist[s]]
                a = int(rng.choice(closer)) if (kind == "goal" and closer) else int(rng.choice(options))
                actions.append(a)
                s = int(succ[s, a])
                states.append(s)
                phases.append(PHASE_PRE)
            cands = entries[side]
            d_to_target = [abs(divmod(e, grid_cols)[0] - divmod(int(s_target), grid_cols)[0]) +
                           abs(divmod(e, grid_cols)[1] - divmod(int(s_target), grid_cols)[1]) for _, _, e in cands]
            best = min(range(len(cands)), key=lambda i: (d_to_target[i], cands[i][2], cands[i][0]))
            c_in, a_in, e_in = cands[best]
            for prev, a in _bfs(succ, s, corr_set, lambda x: x == c_in):
                actions.append(a)
                s = int(succ[prev, a])
                states.append(s)
                phases.append(PHASE_PATH)
            actions.append(a_in)
            s = e_in
            states.append(s)
            phases.append(PHASE_ZONE)
            for prev, a in _bfs(succ, s, zone_set, lambda x: x == int(s_target)):
                actions.append(a)
                s = int(succ[prev, a])
                states.append(s)
                phases.append(PHASE_ZONE)
            assert s == int(s_target)
            actions.append(-1)
            walks.append((states, actions, phases))
            meta.append((int(s_target), side, kind))
    T = max(len(w[0]) for w in walks)
    n = len(walks)
    out = {k: np.full((n, T), fill, dtype=np.int64) for k, fill in (("states", -1), ("actions", -1), ("phase", PHASE_PAD))}
    out["length"] = np.zeros(n, dtype=np.int64)
    for i, (states, actions, phases) in enumerate(walks):
        m = len(states)
        out["states"][i, :m], out["actions"][i, :m], out["phase"][i, :m] = states, actions, phases
        out["length"][i] = m
    out["target"] = np.array([m[0] for m in meta])
    out["side"] = np.array([m[1] for m in meta])
    out["kind"] = np.array([m[2] for m in meta])
    return out
