"""E2's output schema: TWO tables (per-state, per-pair) rather than E1's
single per-zone-aggregate table (spec section 6) -- E2 needs per-(s,a)
granularity (is_optimal_under_P, N, l1) that a zone mean would erase.

Join key (both tables): (exp, env, condition, alpha, epsilon, seed,
assignment, checkpoint, zone, state[, action]). Keeps E1's `region`
column and `env` values ({exp, ctrl}); `condition` is new
({online, matched}, spec section: "Conditions"); `assignment` records
which counterbalancing rotation produced this row (see
experiments/E2_online_4zones/env.py::quadrant_assignment_for_rotation).

to_e1_zone_format() derives E1-schema-compatible zone-aggregate rows FROM
the per-state table (never recomputed independently -- spec: "Les
agregats par zone au format E1 se derivent de la table par etat"), so
experiments/E1_offline_4zones/artifact.py::join_exp_ctrl can be reused
unchanged for zone-level exp/ctrl comparisons. E2's own n_budget-shaped
slot is filled by `checkpoint` (the step count at which N was snapshotted)
-- the two are the same kind of quantity (how much data went into this
row), just named differently because E1's collection is generative and
E2's is a running online count.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PER_STATE_COLUMNS = [
    "exp", "env", "condition", "alpha", "epsilon", "seed", "assignment", "checkpoint",
    "zone", "zone_type", "region", "state",
    "I_hat", "I_star", "b", "delta", "v_pi_true", "v_pi_believed",
]

PER_PAIR_COLUMNS = [
    "exp", "env", "condition", "alpha", "epsilon", "seed", "assignment", "checkpoint",
    "zone", "zone_type", "region", "state", "action",
    "N", "l1", "is_optimal_under_P",
]

_NUMERIC_PER_STATE = ("alpha", "epsilon", "seed", "assignment", "checkpoint", "state",
                       "I_hat", "I_star", "b", "delta", "v_pi_true", "v_pi_believed")
_NUMERIC_PER_PAIR = ("alpha", "epsilon", "seed", "assignment", "checkpoint", "state", "action", "N", "l1")


def write_rows(rows: list[dict], columns: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path


def write_per_state_artifact(rows: list[dict], path: str | Path) -> Path:
    return write_rows(rows, PER_STATE_COLUMNS, path)


def write_per_pair_artifact(rows: list[dict], path: str | Path) -> Path:
    return write_rows(rows, PER_PAIR_COLUMNS, path)


def _coerce_row(row: dict, numeric_cols: tuple[str, ...]) -> dict:
    out = dict(row)
    int_cols = ("seed", "assignment", "checkpoint", "state", "action", "N")
    for col in numeric_cols:
        v = out.get(col, "")
        if v == "" or v is None:
            out[col] = float("nan")
        else:
            try:
                out[col] = int(v) if col in int_cols else float(v)
            except ValueError:
                out[col] = float("nan")
    if "is_optimal_under_P" in out:
        out["is_optimal_under_P"] = out["is_optimal_under_P"] in ("True", "true", "1", True)
    return out


def load_per_state_artifact(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return [_coerce_row(row, _NUMERIC_PER_STATE) for row in csv.DictReader(f)]


def load_per_pair_artifact(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return [_coerce_row(row, _NUMERIC_PER_PAIR) for row in csv.DictReader(f)]


def to_e1_zone_format(per_state_rows: list[dict], per_pair_rows: list[dict]) -> list[dict]:
    """Derives E1-schema zone-aggregate rows (see
    experiments/E1_offline_4zones/artifact.py::MEASUREMENT_COLUMNS) from
    E2's per-state table, grouping by every join-key field EXCEPT state
    (which collapses into the mean) and by region in {interior, boundary,
    all} (E2's own per-state rows are tagged interior/boundary only --
    "all" is the whole-zone mean, computed here, never a 3rd per-state
    label -- see module docstring).

    per_pair_rows supplies l1 (a per-(s,a) quantity with no per-state
    counterpart): E1's own l1 is the mean of l1_error_per_pair over every
    (s,a) pair whose s lies in the zone/region -- computed here from
    per_pair_rows the same way, never left null at region=="all" (E1's
    own load_measurement_artifact asserts l1 is non-null there, exactly
    like I_hat/I_star -- see tests/agents/test_e2_artifact.py's
    round-trip-through-E1's-own-loader check).

    I_hat/I_star/b/l1 are populated only at region=="all" and (for b)
    env=="exp", exactly matching E1's own invariant.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in per_state_rows:
        key = (row["exp"], row["env"], row["condition"], row["alpha"], row["epsilon"], row["seed"],
               row["assignment"], row["checkpoint"], row["zone"], row["zone_type"])
        groups.setdefault(key, []).append(row)

    pair_groups: dict[tuple, list[dict]] = {}
    for row in per_pair_rows:
        key = (row["exp"], row["env"], row["condition"], row["alpha"], row["epsilon"], row["seed"],
               row["assignment"], row["checkpoint"], row["zone"], row["zone_type"])
        pair_groups.setdefault(key, []).append(row)

    out = []
    for key, rows in groups.items():
        exp, env, condition, alpha, epsilon, seed, assignment, checkpoint, zone, zone_type = key
        pair_rows = pair_groups.get(key, [])
        for region in ("interior", "boundary", "all"):
            if region == "all":
                region_rows = rows
            else:
                region_rows = [r for r in rows if r["region"] == region]
            if not region_rows:
                continue
            delta = float(np.mean([r["delta"] for r in region_rows]))
            v_pi_true = float(np.mean([r["v_pi_true"] for r in region_rows]))
            v_pi_believed = float(np.mean([r["v_pi_believed"] for r in region_rows]))
            if region == "all":
                I_hat = float(np.mean([r["I_hat"] for r in rows]))
                I_star = float(np.mean([r["I_star"] for r in rows]))
                b = float(np.mean([r["b"] for r in rows])) if env == "exp" else float("nan")
                l1 = float(np.mean([r["l1"] for r in pair_rows])) if pair_rows else float("nan")
            else:
                I_hat = I_star = b = l1 = float("nan")
            out.append({
                "exp": exp, "env": env, "condition": condition, "alpha": alpha, "epsilon": epsilon,
                "seed": seed, "assignment": assignment,
                "n_budget": checkpoint,  # E1-schema join key uses n_budget; E2's checkpoint fills that slot
                "zone": zone, "zone_type": zone_type, "region": region,
                "I_hat": I_hat, "I_star": I_star, "b": b, "delta": delta,
                "l1": l1, "v_pi_true": v_pi_true, "v_pi_believed": v_pi_believed,
            })
    return out
