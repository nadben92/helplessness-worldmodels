"""Temps-3 report addition: from the sweep's logged per-episode records
(start state, length, truncated -- see sweep.py::run_condition_seed),
buckets episode length (median, p90, p99) and truncation rate by the
STARTING zone's type, aggregated across all seeds of each condition.

Run separately from the sweep itself, after it completes:
    cd experiments/E2_online_4zones
    python episode_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import env as e2_env

from helplessness.config import load_yaml_block, REPO_ROOT

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]

RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_ID = "E2_online_4zones_sweep"

_ZONE_OF_STATE_CACHE: dict[int, np.ndarray] = {}


def zone_of_state_for_assignment(rotation: int) -> np.ndarray:
    """(N_STATES,) array of zone-category strings for the given
    counterbalancing rotation -- cached, since the same handful of
    rotations (0..3) recur across all 20 seeds of a condition."""
    if rotation not in _ZONE_OF_STATE_CACHE:
        assignment = e2_env.quadrant_assignment_for_rotation(rotation)
        layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)
        out = np.empty(N_STATES, dtype=object)
        n_rows, n_cols = GRID_SHAPE
        for zone in layout.zones:
            for r in range(n_rows):
                for c in range(n_cols):
                    if zone.mask[r, c]:
                        out[r * n_cols + c] = zone.appearance.category
        _ZONE_OF_STATE_CACHE[rotation] = out
    return _ZONE_OF_STATE_CACHE[rotation]


def condition_keys_present(results_root: Path | None = None, experiment_id: str | None = None) -> list[str]:
    # RESULTS_ROOT/EXPERIMENT_ID read as module globals HERE, at call
    # time (None defaults, resolved in the body) rather than bound as
    # default parameter values at function-definition time -- a caller
    # that reassigns the module-level globals after import (e.g. a test
    # driver pointing this script at an isolated results directory)
    # would otherwise see no effect, since default parameter expressions
    # are evaluated exactly once, at def-time (see sweep.py's own
    # analogous fix for run_and_save/BUDGET/CHECKPOINTS).
    results_root = RESULTS_ROOT if results_root is None else results_root
    experiment_id = EXPERIMENT_ID if experiment_id is None else experiment_id
    run_dirs = (results_root / experiment_id).glob("*_seed*")
    return sorted({p.name.rsplit("_seed", 1)[0] for p in run_dirs})


def analyze_condition(condition_key: str, results_root: Path | None = None, experiment_id: str | None = None) -> dict:
    """Aggregates episode length/truncation, bucketed by starting zone
    type, across every seed's run directory for this condition."""
    results_root = RESULTS_ROOT if results_root is None else results_root
    experiment_id = EXPERIMENT_ID if experiment_id is None else experiment_id
    run_dirs = sorted((results_root / experiment_id).glob(f"{condition_key}_seed*"))
    lengths_by_zone: dict[str, list[int]] = {}
    truncated_by_zone: dict[str, list[bool]] = {}

    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        rotation = summary["assignment"]
        zone_of_state = zone_of_state_for_assignment(rotation)

        with np.load(run_dir / "arrays.npz") as arrays:
            starts = arrays["episode_start_state"]
            lengths = arrays["episode_lengths"]
            truncated = arrays["episode_truncated"]

        zones = zone_of_state[starts]
        for z in np.unique(zones):
            mask = zones == z
            lengths_by_zone.setdefault(z, []).extend(lengths[mask].tolist())
            truncated_by_zone.setdefault(z, []).extend(truncated[mask].tolist())

    report = {}
    for zone, lens_list in lengths_by_zone.items():
        lens = np.array(lens_list)
        trunc = np.array(truncated_by_zone[zone])
        report[zone] = {
            "n_episodes": int(len(lens)),
            "median_length": float(np.median(lens)),
            "p90_length": float(np.percentile(lens, 90)),
            "p99_length": float(np.percentile(lens, 99)),
            "truncation_rate": float(trunc.mean()),
        }
    return report


def main() -> None:
    keys = condition_keys_present()
    if not keys:
        print(f"No sweep runs found under {RESULTS_ROOT / EXPERIMENT_ID}")
        return
    for key in keys:
        print(f"\n=== {key} ===")
        report = analyze_condition(key)
        for zone, stats in sorted(report.items()):
            print(
                f"  {zone:28s} n={stats['n_episodes']:7d}  "
                f"median={stats['median_length']:6.1f}  p90={stats['p90_length']:6.1f}  "
                f"p99={stats['p99_length']:6.1f}  truncation_rate={stats['truncation_rate']:.4f}"
            )


if __name__ == "__main__":
    main()
