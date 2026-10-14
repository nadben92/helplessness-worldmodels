"""The declaration of a generated number and the readers its getters use. Data access only: the formatting (rounding)
of every number happens in gen_numbers.fmt, nowhere else.

A getter returns the raw value or raises:
  - SourceError: a source is missing or inconsistent -- the build stops (never a fallback value);
  - PendingSource: the source belongs to a batch that has not run yet (the E3 final campaign) -- allowed only for a
    Number declared pending=True, which is then written as \\pending{} and listed by gen_numbers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parent


class SourceError(SystemExit):
    pass


class PendingSource(Exception):
    pass


@dataclass(frozen=True)
class Number:
    macro: str
    src: str
    get: Callable[[], float]
    decimals: int
    signed: bool = False
    style: str = "fixed"     # "fixed", "rounded", "percent" (value x 100) or "sci" (m x 10^e); see gen_numbers.fmt
    pending: bool = False    # may raise PendingSource (written as \pending{}); any other failure stops the build


def read_rows(rel_path: str, **match: str) -> list[dict]:
    """Rows of a CSV (path relative to the repo root) whose columns equal `match`."""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        raise SourceError(f"Missing source file: {rel_path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for key in match:
        if rows and key not in rows[0]:
            raise SourceError(f"{rel_path}: no column {key!r}")
    rows = [r for r in rows if all(r[k] == v for k, v in match.items())]
    if not rows:
        raise SourceError(f"{rel_path}: no row with {match}")
    return rows


def one_value(rel_path: str, column: str, **match: str) -> float:
    rows = read_rows(rel_path, **match)
    if len(rows) != 1:
        raise SourceError(f"{rel_path}: expected one row with {match}, found {len(rows)}")
    if column not in rows[0]:
        raise SourceError(f"{rel_path}: no column {column!r}")
    return float(rows[0][column])


def constant_value(rel_path: str, column: str, **match: str) -> float:
    """The value of `column`, required to be identical on every matching row."""
    values = {r[column] for r in read_rows(rel_path, **match)}
    if len(values) != 1:
        raise SourceError(f"{rel_path}: {column} is not constant over rows {match}: {sorted(values)}")
    return float(values.pop())


def n_distinct(rel_path: str, column: str, **match: str) -> int:
    return len({r[column] for r in read_rows(rel_path, **match)})
