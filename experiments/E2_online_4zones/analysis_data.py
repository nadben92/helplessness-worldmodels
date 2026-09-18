"""Loads the E2 sweep's artifacts (results/E2_online_4zones_sweep/) into
plain numpy structured arrays -- never pandas (see E1's own
artifact.py docstring: "No pandas dependency (not in pyproject.toml's
tabular-phase allowlist)" -- the same convention is kept here). Never
reruns training: every quantity here is read from what sweep.py already
wrote to disk.

Zone-type reconstruction is done per run from that run's OWN
counterbalancing rotation (summary.json's "assignment"), never assumed
to be the identity assignment -- CLAUDE.md's own review note for this
task: "verifie explicitement que la reconstruction du type de zone se
fait depuis la rotation propre a chaque run." The per_state/per_pair CSV
rows already carry their own "zone"/"zone_type" columns (written by
measurement.py at collection time, already rotation-correct) so this
module does not even need to re-derive it for the CSV-backed data --
only analysis that goes beyond the CSVs (e.g. spatial maps built
directly from arrays.npz's raw N tables) needs the explicit per-run
zone lookup, provided here as zone_of_state_for_run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import env as e2_env

from helplessness.config import load_yaml_block, REPO_ROOT

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_ROWS, N_COLS = GRID_SHAPE
N_STATES = N_ROWS * N_COLS
N_ACTIONS = 5
GOAL_S = e2_env.goal_state(CFG)

SWEEP_ROOT = REPO_ROOT / "results" / "E2_online_4zones_sweep"

ZONE_CATEGORIES = (
    "corridor", "uncontrollable_deterministic", "controllable_noisy",
    "controllable_deterministic", "uncontrollable_noisy",
)
NOISY_ZONES = ("controllable_noisy", "uncontrollable_noisy")
DETERMINISTIC_ZONES = ("controllable_deterministic", "uncontrollable_deterministic")

PER_STATE_DTYPE = np.dtype([
    ("condition", "<U8"), ("alpha", "f8"), ("epsilon", "f8"), ("seed", "i4"),
    ("assignment", "i4"), ("checkpoint", "i8"), ("zone", "<U32"), ("zone_type", "<U32"),
    ("region", "<U8"), ("state", "i4"), ("I_hat", "f8"), ("I_star", "f8"), ("b", "f8"),
    ("delta", "f8"), ("v_pi_true", "f8"), ("v_pi_believed", "f8"),
])

PER_PAIR_DTYPE = np.dtype([
    ("condition", "<U8"), ("alpha", "f8"), ("epsilon", "f8"), ("seed", "i4"),
    ("assignment", "i4"), ("checkpoint", "i8"), ("zone", "<U32"), ("zone_type", "<U32"),
    ("region", "<U8"), ("state", "i4"), ("action", "i4"), ("N", "i8"), ("l1", "f8"),
    ("is_optimal_under_P", "?"),
])

_ZONE_OF_STATE_CACHE: dict[int, np.ndarray] = {}


def zone_of_state_for_rotation(rotation: int) -> np.ndarray:
    """(N_STATES,) array of zone-category strings for the given
    counterbalancing rotation -- cached across calls (only 4 distinct
    rotations exist)."""
    if rotation not in _ZONE_OF_STATE_CACHE:
        assignment = e2_env.quadrant_assignment_for_rotation(rotation)
        layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)
        out = np.empty(N_STATES, dtype="<U32")
        for zone in layout.zones:
            for r in range(N_ROWS):
                for c in range(N_COLS):
                    if zone.mask[r, c]:
                        out[r * N_COLS + c] = zone.appearance.category
        _ZONE_OF_STATE_CACHE[rotation] = out
    return _ZONE_OF_STATE_CACHE[rotation]


def all_run_dirs(sweep_root: Path = SWEEP_ROOT) -> list[Path]:
    return sorted(p for p in sweep_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text())


def load_per_state(run_dirs: list[Path] | None = None) -> np.ndarray:
    """Concatenates every run's per_state.csv into one structured array.
    Excludes the goal state (never a measurement subject) -- it never
    appears in these CSVs to begin with (measurement.py's per_state_rows
    iterates every state including the goal, but region is interior/boundary
    only; the goal itself, being in the corridor zone, DOES appear as a
    row -- explicitly filtered out here, see module docstring "etat but
    exclu de toutes les mesures")."""
    if run_dirs is None:
        run_dirs = all_run_dirs()
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "per_state.csv"
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                # exp, env, condition, alpha, epsilon, seed, assignment,
                # checkpoint, zone, zone_type, region, state, I_hat,
                # I_star, b, delta, v_pi_true, v_pi_believed
                rows.append((
                    row[2], float(row[3]), float(row[4]), int(row[5]), int(row[6]),
                    int(row[7]), row[8], row[9], row[10], int(row[11]),
                    float(row[12]), float(row[13]), float(row[14]), float(row[15]),
                    float(row[16]), float(row[17]),
                ))
    arr = np.array(rows, dtype=PER_STATE_DTYPE)
    return arr[arr["state"] != GOAL_S]


def load_per_pair(run_dirs: list[Path] | None = None) -> np.ndarray:
    if run_dirs is None:
        run_dirs = all_run_dirs()
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "per_pair.csv"
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                # exp, env, condition, alpha, epsilon, seed, assignment,
                # checkpoint, zone, zone_type, region, state, action, N,
                # l1, is_optimal_under_P
                rows.append((
                    row[2], float(row[3]), float(row[4]), int(row[5]), int(row[6]),
                    int(row[7]), row[8], row[9], row[10], int(row[11]), int(row[12]),
                    int(row[13]), float(row[14]), row[15] == "True",
                ))
    arr = np.array(rows, dtype=PER_PAIR_DTYPE)
    return arr[arr["state"] != GOAL_S]


def load_summaries(run_dirs: list[Path] | None = None) -> list[dict]:
    if run_dirs is None:
        run_dirs = all_run_dirs()
    return [_read_summary(d) for d in run_dirs]


def condition_keys_present(sweep_root: Path = SWEEP_ROOT) -> list[str]:
    return sorted({p.name.rsplit("_seed", 1)[0] for p in all_run_dirs(sweep_root)})


CONDITION_LABELS = {
    "a0.5_e0.1": {"alpha": 0.5, "epsilon": 0.1},
    "a1_e0.1": {"alpha": 1.0, "epsilon": 0.1},
    "a5_e0.1": {"alpha": 5.0, "epsilon": 0.1},
    "a1_e1.0_bridge": {"alpha": 1.0, "epsilon": 1.0},
}
REFERENCE_CONDITION = "a1_e0.1"
CHECKPOINTS = tuple(sorted(set(int(round(x)) for x in np.logspace(3, 6, 16))))
FINAL_CHECKPOINT = CHECKPOINTS[-1]
N_SEEDS = 20
