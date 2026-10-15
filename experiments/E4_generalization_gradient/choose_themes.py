"""E4's training colours: two pairs (A, B) and (C, D), the fixed agent-marker colour, and the three axes of each pair
(helplessness.envs.appearance.theme_axes). Writes configs/e4/themes.yaml; prints the report. Deterministic (seed 0).

Constraints (docs/E4_generalization_gradient.md, section 4 of the E4 prompt):
  - every point of every axis, for both orientations, at least MIN_DIST from the agent, the goal and the corridor
    colours. MIN_DIST is E3's minimum distance between theme colours and from the agent marker
    (configs/e3/*.yaml render.cielab_min_dist = cielab_marker_min_dist = 30), which E3 measured in CIELAB (deltaE76);
    E4 keeps that metric for this constraint. D and the axes themselves are Euclidean in RGB;
  - the three axes stay in the RGB cube (the arcs' planes: theme_axes.choose_arc_plane);
  - pair 2 is distinct from pair 1: its two colours at least MIN_DIST from pair 1's two colours, and its direction at
    least MIN_ANGLE_DEG from pair 1's direction (not a near-parallel copy of pair 1);
  - under these constraints each pair maximises D = |T_a - T_b|.
The agent colour is the grid colour (step 5) farthest (smallest deltaE76, maximised) from every axis point, the goal
and the corridor; the pairs' arc planes depend on it (they avoid it), so pairs and agent are recomputed until the
agent colour no longer changes.

Search: 400,000 random integer pairs (plus every pair of E3's campaign theme colours, seeds 0, 1, 5, 6), sorted by D,
tested in that order until the first feasible one; then a hill climb on the six coordinates (steps 16, 8, 4, 2, 1)
that accepts any move increasing D while feasible.

Usage: python experiments/E4_generalization_gradient/choose_themes.py [--out configs/e4/themes.yaml]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from helplessness.config import REPO_ROOT
from helplessness.envs.appearance import theme_axes as ta
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR, GOAL_COLOR, build_themes
from helplessness.envs.layouts import four_zones_pixel
from helplessness.streams import worldmodel as e3_streams

MIN_DIST = 30.0
MIN_ANGLE_DEG = 45.0
N_RANDOM = 400_000
SEARCH_ANGLES = 72   # plane grid during the search (coarse); the final axes use theme_axes' 360
FINAL_ANGLES = 360
AGENT_GRID_STEP = 5


def e3_campaign_colors() -> list[tuple[int, int, int]]:
    """The zone colours of E3's campaign seeds (0, 1, 5, 6), recomputed from their theme stream as run_mini.build_world
    draws them (configs/e3/gpu_campaign.yaml: min_dist 30, marker_min_dist 30)."""
    colors = []
    for seed in (0, 1, 5, 6):
        themes = build_themes(four_zones_pixel.ZONE_CATEGORIES, goal_state=four_zones_pixel.goal_state(),
                              rng=e3_streams.spawn_streams(seed)["theme"], min_dist=30.0, marker_min_dist=30.0)
        colors += list(themes.zone_colors.values())
    return colors


def feasible(a, b, forbidden, n_angles: int) -> dict | None:
    """The pair's axes when every constraint of a single pair holds, else None."""
    if ta.lab_distances(ta.to_rgb(ta.segment_axis(a, b)), forbidden).min() < MIN_DIST:
        return None
    axes = ta.pair_axes(a, b, forbidden, n_angles=n_angles)
    if axes is None or ta.axes_min_lab_distance(axes, forbidden) < MIN_DIST:
        return None
    return axes


def distinct_from(a, b, first: tuple | None) -> bool:
    if first is None:
        return True
    fa, fb = first
    if ta.lab_distances(np.array([a, b]), np.array([fa, fb])).min() < MIN_DIST:
        return False
    d1, d2 = np.subtract(fb, fa, dtype=float), np.subtract(b, a, dtype=float)
    cos = abs(np.dot(d1, d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return bool(cos <= np.cos(np.radians(MIN_ANGLE_DEG)))


def best_pair(forbidden: np.ndarray, rng: np.random.Generator, first: tuple | None, seeds_extra: list) -> tuple:
    cand = rng.integers(0, 256, size=(N_RANDOM, 2, 3))
    extra = np.array([[p, q] for p, q in itertools.permutations(seeds_extra, 2)], dtype=np.int64).reshape(-1, 2, 3)
    cand = np.concatenate([extra, cand])
    D = np.linalg.norm(cand[:, 0] - cand[:, 1], axis=1)
    found = None
    for i in np.argsort(-D, kind="stable"):
        a, b = tuple(int(v) for v in cand[i, 0]), tuple(int(v) for v in cand[i, 1])
        if distinct_from(a, b, first) and feasible(a, b, forbidden, SEARCH_ANGLES) is not None:
            found = (a, b)
            break
    if found is None:
        raise RuntimeError("no feasible pair among the candidates")
    a, b = np.array(found[0]), np.array(found[1])
    improved = True
    while improved:
        improved = False
        for step in (16, 8, 4, 2, 1):
            for idx in range(6):
                for sign in (1, -1):
                    na, nb = a.copy(), b.copy()
                    (na if idx < 3 else nb)[idx % 3] += sign * step
                    if not (ta.in_cube(na) and ta.in_cube(nb)):
                        continue
                    if ta.rgb_distance(na, nb) <= ta.rgb_distance(a, b) + 1e-9:
                        continue
                    if distinct_from(tuple(na), tuple(nb), first) and feasible(na, nb, forbidden, SEARCH_ANGLES) is not None:
                        a, b, improved = na, nb, True
    return tuple(int(v) for v in a), tuple(int(v) for v in b)


def choose_agent(points: np.ndarray) -> tuple[tuple[int, int, int], float]:
    grid = np.arange(0, 256, AGENT_GRID_STEP)
    cand = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
    targets = np.concatenate([points, np.array([CORRIDOR_COLOR, GOAL_COLOR])])
    lab_c, lab_t = ta.lab(cand), ta.lab(targets)
    best_i, best_d = 0, -1.0
    for first in range(0, len(cand), 20_000):
        d = np.linalg.norm(lab_c[first:first + 20_000, None, :] - lab_t[None, :, :], axis=-1).min(axis=1)
        i = int(np.argmax(d))
        if d[i] > best_d + 1e-12:
            best_i, best_d = first + i, float(d[i])
    return tuple(int(v) for v in cand[best_i]), best_d


def solve() -> dict:
    agent = None
    e3_colors = e3_campaign_colors()
    for _ in range(5):
        forbidden = np.array([CORRIDOR_COLOR, GOAL_COLOR] + ([agent] if agent else []), dtype=np.float64)
        pair1 = best_pair(forbidden, np.random.default_rng(0), None, e3_colors)
        pair2 = best_pair(forbidden, np.random.default_rng(1), pair1, e3_colors)
        axes = [ta.pair_axes(*p, forbidden, n_angles=FINAL_ANGLES) for p in (pair1, pair2)]
        points = np.concatenate([np.concatenate([ax["segment"], ax["arc_center_b"], ax["arc_center_a"]]) for ax in axes])
        new_agent, agent_margin = choose_agent(points)
        if new_agent == agent:
            break
        agent = new_agent
    forbidden = np.array([CORRIDOR_COLOR, GOAL_COLOR, agent], dtype=np.float64)
    axes = [ta.pair_axes(*p, forbidden, n_angles=FINAL_ANGLES) for p in (pair1, pair2)]
    out = {"min_dist_cielab": MIN_DIST, "min_angle_deg": MIN_ANGLE_DEG, "agent_marker_color": list(agent),
           "corridor_color": list(CORRIDOR_COLOR), "goal_color": list(GOAL_COLOR), "pairs": {}}
    for name, pair, ax in zip(("pair1", "pair2"), (pair1, pair2), axes):
        pts = np.concatenate([ax["segment"], ax["arc_center_b"], ax["arc_center_a"]])
        out["pairs"][name] = {
            "a": list(pair[0]), "b": list(pair[1]), "D": ax["D"],
            "phi_center_b": ax["phi_center_b"], "phi_center_a": ax["phi_center_a"],
            "segment": ax["segment"].tolist(), "arc_center_b": ax["arc_center_b"].tolist(), "arc_center_a": ax["arc_center_a"].tolist(),
            "min_cielab_to": {name_: float(ta.lab_distances(pts, [col]).min()) for name_, col in
                              (("agent", agent), ("goal", GOAL_COLOR), ("corridor", CORRIDOR_COLOR))},
            "min_rgb_to": {name_: float(ta.rgb_distance(pts, col).min()) for name_, col in
                           (("agent", agent), ("goal", GOAL_COLOR), ("corridor", CORRIDOR_COLOR))},
        }
    out["pair2_vs_pair1"] = {
        "min_cielab_endpoints": float(ta.lab_distances(np.array(pair2), np.array(pair1)).min()),
        "angle_deg": float(np.degrees(np.arccos(abs(np.dot(np.subtract(pair1[1], pair1[0]), np.subtract(pair2[1], pair2[0])))
                                                / (ta.rgb_distance(*pair1) * ta.rgb_distance(*pair2))))),
    }
    best_e3 = max((ta.rgb_distance(p, q), p, q) for p, q in itertools.combinations(e3_colors, 2)
                  if feasible(p, q, forbidden, SEARCH_ANGLES) is not None) if e3_colors else None
    out["best_feasible_e3_pair"] = None if best_e3 is None else {"D": float(best_e3[0]), "a": list(best_e3[1]), "b": list(best_e3[2])}
    out["agent_min_cielab_to_everything"] = float(choose_agent(np.concatenate(
        [np.concatenate([ax["segment"], ax["arc_center_b"], ax["arc_center_a"]]) for ax in axes]))[1])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "configs" / "e4" / "themes.yaml")
    args = parser.parse_args(argv)
    out = solve()
    header = ("# E4's training colours and theme axes, generated by experiments/E4_generalization_gradient/choose_themes.py\n"
              "# (deterministic; never edited by hand). RGB 0-255. Distances: D Euclidean in RGB; min_cielab_to in deltaE76.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + yaml.safe_dump(out, sort_keys=False, default_flow_style=None, width=200), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1))
    for name, p in out["pairs"].items():
        print(name, {k: p[k] for k in ("a", "b", "D", "phi_center_b", "phi_center_a", "min_cielab_to", "min_rgb_to")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
