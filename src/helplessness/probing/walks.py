"""Fixed walks for zero-data probes (E4, docs/E4_generalization_gradient.md): numpy only, no model. A walk is a
sequence of states and actions in an environment with deterministic dynamics P (n_states, n_actions, n_states), in
two phases:

  1. warm-up in the corridor: a start drawn uniformly among the corridor states (goal excluded), then n_pre steps of
     uniform actions that stay in the corridor -- an action whose successor is outside the corridor or the goal is
     redrawn (`stay` is always allowed);
  2. entry and walk in the zone of interest: an entry state drawn uniformly among the zone's border states (zone states
     with a corridor neighbour); a shortest path through the corridor (breadth-first, actions in index order, goal
     excluded) to a corridor state from which one action enters the zone at that entry state; that action; then
     n_in - 1 steps of uniform actions that stay in the zone (an action whose successor leaves the zone is redrawn).

Step numbering: steps_since_start counts every state from 0 (the start); steps_since_entry is 1 at the entry state
(the first step in the zone) and n_in at the last one, 0 outside the zone. A walk therefore has n_in states in the
zone of interest, and never visits the goal nor any other quadrant. Walks are a pure function of (P, regions, the
parameters, the generator), so a seed reproduces them exactly and the same walks serve every image whose dynamics is
P (the probe images of a configuration differ only by their colours).
"""

from __future__ import annotations

from collections import deque

import numpy as np

PHASE_PRE, PHASE_PATH, PHASE_ZONE, PHASE_PAD = 0, 1, 2, -1


def successors(P: np.ndarray) -> np.ndarray:
    """(n_states, n_actions) successor of every pair of a deterministic P; raises if P is not deterministic."""
    if not np.allclose(P.max(axis=-1), 1.0):
        raise ValueError("walks need deterministic dynamics (every P[s, a] a point mass)")
    return P.argmax(axis=-1)


def border_states(succ: np.ndarray, zone: np.ndarray, corridor: np.ndarray) -> np.ndarray:
    """States of the zone with at least one action leading into the corridor (or reached from it)."""
    zone_set, corr_set = set(zone.tolist()), set(corridor.tolist())
    entries = {int(succ[c, a]) for c in corr_set for a in range(succ.shape[1]) if int(succ[c, a]) in zone_set}
    return np.array(sorted(entries), dtype=np.int64)


def corridor_path(succ: np.ndarray, start: int, corridor: set, target_pairs: dict) -> list[tuple[int, int]]:
    """Shortest list of (state, action) from `start` through corridor states to a state c of target_pairs, ending with
    target_pairs[c] (the action entering the zone). Breadth-first, actions in index order."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if s in target_pairs:
            path = [(s, target_pairs[s])]
            while parent[s] is not None:
                prev, a = parent[s]
                path.append((prev, a))
                s = prev
            return path[::-1]
        for a in range(succ.shape[1]):
            nxt = int(succ[s, a])
            if nxt in corridor and nxt not in parent:
                parent[nxt] = (s, a)
                queue.append(nxt)
    raise ValueError(f"no corridor path from state {start} to the zone")


def generate_walks(P: np.ndarray, corridor: np.ndarray, zone: np.ndarray, n_walks: int, n_pre: int, n_in: int,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """n_walks walks (module docstring). corridor: corridor states WITHOUT the goal; zone: the zone of interest.
    Returns arrays padded to the longest walk:
      states (n_walks, T) int, -1 after the end;      actions (n_walks, T) int, the action taken AT step t, -1 at the
      phase (n_walks, T) PHASE_* codes;               last step and after;
      steps_since_start (n_walks, T); steps_since_entry (n_walks, T); length (n_walks,); entry_state (n_walks,)."""
    if n_in < 1 or n_pre < 0:
        raise ValueError("n_in >= 1 and n_pre >= 0")
    succ = successors(P)
    n_actions = succ.shape[1]
    corridor_set, zone_set = set(int(s) for s in corridor), set(int(s) for s in zone)
    border = border_states(succ, np.asarray(zone), np.asarray(corridor))
    walks = []
    for _ in range(n_walks):
        s = int(corridor[rng.integers(len(corridor))])
        states, actions, phases = [s], [], [PHASE_PRE]
        for _ in range(n_pre):
            while True:
                a = int(rng.integers(n_actions))
                if int(succ[s, a]) in corridor_set:
                    break
            actions.append(a)
            s = int(succ[s, a])
            states.append(s)
            phases.append(PHASE_PRE)
        entry = int(border[rng.integers(len(border))])
        targets = {}  # corridor state -> the smallest action entering the zone at `entry`
        for c in sorted(corridor_set):
            entering = [a for a in range(n_actions) if int(succ[c, a]) == entry]
            if entering:
                targets[c] = entering[0]
        for c, a in corridor_path(succ, s, corridor_set, targets):
            assert c == s
            actions.append(a)
            s = int(succ[s, a])
            states.append(s)
            phases.append(PHASE_PATH if s not in zone_set else PHASE_ZONE)
        assert s == entry and phases[-1] == PHASE_ZONE
        for _ in range(n_in - 1):
            while True:
                a = int(rng.integers(n_actions))
                if int(succ[s, a]) in zone_set:
                    break
            actions.append(a)
            s = int(succ[s, a])
            states.append(s)
            phases.append(PHASE_ZONE)
        actions.append(-1)
        walks.append((states, actions, phases, entry))
    T = max(len(w[0]) for w in walks)
    out = {k: np.full((n_walks, T), fill, dtype=np.int64) for k, fill in
           (("states", -1), ("actions", -1), ("phase", PHASE_PAD), ("steps_since_start", -1), ("steps_since_entry", 0))}
    out["length"] = np.zeros(n_walks, dtype=np.int64)
    out["entry_state"] = np.zeros(n_walks, dtype=np.int64)
    for i, (states, actions, phases, entry) in enumerate(walks):
        n = len(states)
        out["states"][i, :n], out["actions"][i, :n], out["phase"][i, :n] = states, actions, phases
        out["steps_since_start"][i, :n] = np.arange(n)
        in_zone = np.asarray(phases) == PHASE_ZONE
        out["steps_since_entry"][i, :n][in_zone] = np.arange(1, int(in_zone.sum()) + 1)
        out["length"][i], out["entry_state"][i] = n, entry
    return out
