"""Where a world model draws the agent, against where it is, along episodes (E6, docs/e6/DECISIONS.md (45)): arrays in,
arrays and scalars out. Nothing here imports an environment, an agent or torch.

Inputs (one row per episode, as measures.zone_phenotypes): states (n, H + 1), length (n,), the zone (n_states,) bool,
decoded (n, H) the decoded position at each step (a state index, NaN where unmeasured), and, for the errors, conveyor
(n_states,) the successor of every state under the exposed TRAINING dynamics (the conveyor; it does not depend on the
action) and neighbours (n_states, 4) the four torus neighbours of every state.

  steps_by_localisation  the steps taken in the zone, split by whether the decoded position was right: the directed
                         share of each part (zone_phenotypes.directed_steps).
  error_nature           among the steps t >= 1 in the zone where the decoded position is wrong: the share on the cell
                         where the conveyor would have taken the agent from its previous state, conveyor[s_{t-1}] (steps
                         where conveyor[s_{t-1}] == s_t, i.e. the agent moved as the conveyor would have, are left out:
                         no error can land there), the share on any of the four neighbours of s_t, and among the errors on
                         a neighbour while conveyor[s_{t-1}] is a neighbour of s_t, the share on it against the chance
                         level 1/4 (errors spread uniformly over the four neighbours), with a binomial interval; and the
                         share of the errors on the HOMOLOGOUS cell of each other quadrant (the same position within the
                         quadrant: offsets (0, 6), (6, 0), (6, 6) on the 12 x 12 torus); with `regions` (n_states,) labels,
                         the share of the errors in each region (the zone itself, another quadrant, the corridor): a drift
                         stays in the zone, a confusion of quadrants does not.
"""

from __future__ import annotations

import numpy as np

from helplessness.measures import zone_phenotypes as zp


def steps_by_localisation(states: np.ndarray, length: np.ndarray, zone: np.ndarray, dist: np.ndarray, decoded: np.ndarray) -> dict:
    """n and directed share of the steps in the zone, localised and not (steps without a measure left out)."""
    inz = zp.in_zone_steps(states, length, zone)
    directed = zp.directed_steps(states, length, dist)
    measured = np.isfinite(decoded)
    right = measured & (np.where(measured, decoded, -1) == states[:, :-1])
    out = {}
    for name, mask in (("localised", inz & right), ("mislocalised", inz & measured & ~right)):
        n = int(mask.sum())
        out[f"n_{name}"] = n
        out[f"directed_{name}"] = float(directed[mask].mean()) if n else float("nan")
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial share."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


HOMOLOGOUS_OFFSETS = {"right": (0, 6), "below": (6, 0), "diagonal": (6, 6)}


def error_nature(states: np.ndarray, length: np.ndarray, zone: np.ndarray, decoded: np.ndarray, conveyor: np.ndarray,
                 neighbours: np.ndarray, grid_shape: tuple[int, int] = (12, 12), regions: np.ndarray | None = None) -> dict:
    """Module docstring. Counts and shares, with the chance level of the neighbour test."""
    inz = zp.in_zone_steps(states, length, zone)
    H = decoded.shape[1]
    n_err = n_eligible = on_conv = on_neigh = n_neigh_test = conv_in_neigh = 0
    R, C = grid_shape
    homologous = {name: 0 for name in HOMOLOGOUS_OFFSETS}
    in_region: dict[str, int] = {}
    for i in range(states.shape[0]):
        for t in range(1, min(int(length[i]), H)):
            d = decoded[i, t]
            if not inz[i, t] or not np.isfinite(d) or int(d) == states[i, t]:
                continue
            s, prev, d = int(states[i, t]), int(states[i, t - 1]), int(d)
            n_err += 1
            neigh = set(int(v) for v in neighbours[s])
            on_neigh += d in neigh
            if regions is not None:
                in_region[str(regions[d])] = in_region.get(str(regions[d]), 0) + 1
            r, col = divmod(s, C)
            for name, (dr, dc) in HOMOLOGOUS_OFFSETS.items():
                homologous[name] += d == ((r + dr) % R) * C + (col + dc) % C
            c = int(conveyor[prev])
            if c == s:
                continue
            n_eligible += 1
            on_conv += d == c
            if c in neigh and d in neigh:
                n_neigh_test += 1
                conv_in_neigh += d == c
    share = lambda k, n: float(k / n) if n else float("nan")  # noqa: E731
    return {"n_errors": n_err, "share_on_neighbour": share(on_neigh, n_err), "n_eligible": n_eligible,
            "share_on_conveyor": share(on_conv, n_eligible), "share_on_conveyor_ci": wilson(on_conv, n_eligible),
            "n_neighbour_test": n_neigh_test, "share_on_conveyor_among_neighbours": share(conv_in_neigh, n_neigh_test),
            "share_on_conveyor_among_neighbours_ci": wilson(conv_in_neigh, n_neigh_test), "chance_among_neighbours": 0.25,
            **{f"share_on_homologous_{name}": share(k, n_err) for name, k in homologous.items()},
            **{f"share_in_{name}": share(k, n_err) for name, k in sorted(in_region.items())}}


def torus_neighbours(grid_shape: tuple[int, int]) -> np.ndarray:
    """(n_states, 4): up, right, down, left neighbours on the torus."""
    R, C = grid_shape
    out = np.empty((R * C, 4), dtype=np.int64)
    for s in range(R * C):
        r, c = divmod(s, C)
        out[s] = [((r - 1) % R) * C + c, r * C + (c + 1) % C, ((r + 1) % R) * C + c, r * C + (c - 1) % C]
    return out
