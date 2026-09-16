"""The E1/E2/E3-shared output artifact: schema, writer, loader, and join
function (spec 3.6 -- "ship the loader and join function too, E2 and E3
must not reimplement the comparison"). No pandas dependency (not in
pyproject.toml's tabular-phase allowlist): rows are plain lists of dicts,
loaded and joined with stdlib csv + dict indexing.

Schema:
    exp, env, zone, zone_type, n_budget, seed, region,
    I_hat, I_star, b, delta, l1, v_pi_true, v_pi_believed
    exp in {E1, E2, E3}; env in {exp, ctrl}; region in {interior, boundary, all}
    join key: (exp, env, zone, n_budget, seed, region)

INVARIANT (asserted by load_measurement_artifact, not merely assumed):
I_hat, I_star, and l1 are non-null if and only if region == "all". They
are NOT repeated across interior/boundary/all -- duplicating a
region-independent whole-zone quantity into every region row would let a
consumer that naively groups-and-means over region silently triple-count
it, which is a worse silent failure than a NaN a reader has to notice.
b follows the same rule and is additionally populated only on the "exp"
row (b = Ihat_exp - Ihat_ctrl, already exp-referenced by definition; NaN
on the "ctrl" row and on every non-"all" region regardless of env).

Permutation-null results are a SEPARATE file, with an extended schema
(see PERMUTATION_COLUMNS below and its own docstring note): zone,
quantity, n_budget, seed, p95, observed, detectable.
"""

from __future__ import annotations

import csv
from pathlib import Path

MEASUREMENT_COLUMNS = [
    "exp", "env", "zone", "zone_type", "n_budget", "seed", "region",
    "I_hat", "I_star", "b", "delta", "l1", "v_pi_true", "v_pi_believed",
]

# Extended from the spec's literal (zone, n_budget, seed, p95, I_hat_observed,
# detectable) with:
#   - `quantity` ("I_hat" | "b") so the same file can hold both nulls (spec
#     3.5 explicitly adds a second, wider null for b alongside the original
#     Ihat null) -- I_hat_observed generalised to `observed` accordingly.
#   - `method` ("shuffle" | "bootstrap"): the two answer different
#     questions, neither a correction of the other (see
#     permutation_null.py's module docstring). The shuffle null is
#     conditional on the observed data and gives an EXACT test of
#     H0: "action has no effect" -- use it to decide whether Ihat is
#     distinguishable from zero. The bootstrap null is unconditional
#     (per-action resample-with-replacement from each action's own
#     observed distribution) and describes the spread expected on a
#     fresh dataset -- use it to quote a precision for E2/E4. Both are
#     reported, not one silently dropped, so a reader can see both.
PERMUTATION_COLUMNS = ["zone", "quantity", "method", "n_budget", "seed", "p95", "observed", "detectable"]


def write_measurement_artifact(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MEASUREMENT_COLUMNS})
    return path


def write_permutation_artifact(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERMUTATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PERMUTATION_COLUMNS})
    return path


_NUMERIC_MEASUREMENT_COLUMNS = ("n_budget", "seed", "I_hat", "I_star", "b", "delta", "l1", "v_pi_true", "v_pi_believed")
_NUMERIC_PERMUTATION_COLUMNS = ("n_budget", "seed", "p95", "observed")


def _coerce_row(row: dict, numeric_cols: tuple[str, ...]) -> dict:
    out = dict(row)
    for col in numeric_cols:
        v = out.get(col, "")
        if v == "" or v is None:
            out[col] = float("nan")
        else:
            try:
                out[col] = int(v) if col in ("n_budget", "seed") else float(v)
            except ValueError:
                out[col] = float("nan")
    if "detectable" in out:
        out["detectable"] = out["detectable"] in ("True", "true", "1", True)
    return out


_REGION_ALL_ONLY_COLUMNS = ("I_hat", "I_star", "l1")


def _assert_region_all_invariant(rows: list[dict]) -> None:
    """I_hat/I_star/l1 must be non-null iff region == 'all' (either env);
    b must be non-null iff region == 'all' AND env == 'exp'. Asserted on
    load, not assumed, so a hand-edited or externally-produced file that
    violates it fails loudly here rather than silently mis-joining or
    triple-counting downstream (see module docstring)."""
    for row in rows:
        is_all = row["region"] == "all"
        for col in _REGION_ALL_ONLY_COLUMNS:
            if col not in row:
                continue
            is_nan = row[col] != row[col]  # NaN != NaN
            if is_all and is_nan:
                raise ValueError(f"row {join_key(row)}: {col} is null at region=='all' -- invariant violated")
            if not is_all and not is_nan:
                raise ValueError(f"row {join_key(row)}: {col} is non-null outside region=='all' -- invariant violated")

        b_is_nan = row["b"] != row["b"]
        b_should_be_populated = is_all and row.get("env") == "exp"
        if b_should_be_populated and b_is_nan:
            raise ValueError(f"row {join_key(row)}: b is null at region=='all', env=='exp' -- invariant violated")
        if not b_should_be_populated and not b_is_nan:
            raise ValueError(f"row {join_key(row)}: b is non-null outside (region=='all', env=='exp') -- invariant violated")


def load_measurement_artifact(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = [_coerce_row(row, _NUMERIC_MEASUREMENT_COLUMNS) for row in csv.DictReader(f)]
    _assert_region_all_invariant(rows)
    return rows


def load_permutation_artifact(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return [_coerce_row(row, _NUMERIC_PERMUTATION_COLUMNS) for row in csv.DictReader(f)]


def join_key(row: dict) -> tuple:
    return (row["exp"], row["env"], row["zone"], row["n_budget"], row["seed"], row["region"])


def join_exp_ctrl(rows: list[dict]) -> list[dict]:
    """Joins each 'exp' row to its paired 'ctrl' row on
    (exp, zone, n_budget, seed, region) -- the comparison E2/E3 must reuse,
    not reimplement. Returns one merged dict per matched pair, with
    ctrl_-prefixed columns for the control side's own I_hat/I_star/delta/
    l1/v_pi_true/v_pi_believed, plus the exp row's own columns (including
    b) unprefixed.
    """
    ctrl_by_key = {}
    for row in rows:
        if row["env"] == "ctrl":
            key = (row["exp"], row["zone"], row["n_budget"], row["seed"], row["region"])
            ctrl_by_key[key] = row

    joined = []
    for row in rows:
        if row["env"] != "exp":
            continue
        key = (row["exp"], row["zone"], row["n_budget"], row["seed"], row["region"])
        ctrl_row = ctrl_by_key.get(key)
        if ctrl_row is None:
            continue
        merged = dict(row)
        for col in ("I_hat", "I_star", "delta", "l1", "v_pi_true", "v_pi_believed"):
            merged[f"ctrl_{col}"] = ctrl_row.get(col)
        joined.append(merged)
    return joined
