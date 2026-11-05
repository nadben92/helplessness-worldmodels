"""POST HOC (written after the data of e5_probes_swap were seen; NOT part of (38)'s reading rule, docs/e5/DECISIONS.md).
(38)'s rule leaves every axis "not readable" (REPORT.md); these variants show what the data say where the rule cannot
count, and why. They reuse analyse_swap.py's functions; only the parameters named below change.

  delta_1_swap     (38)'s rule on Delta, an agent counting at a point with Delta in 1 of the 3 swaps (instead of 2)
  moved_2_swaps    (38)'s rule on b(moved) itself instead of Delta (2 of 3 swaps)
  moved_1_swap     the same, 1 of 3 swaps
  far_levels       per agent and axis, the mean b of each class over the kept points at t >= 0.5 (1 swap enough), with
                   the number of (t, swap) cells: does the belief leave the slot m, or spread over the image?

Usage: python experiments/E5_belief_training/analysis_swap/posthoc_swap.py --tables <analyse_swap tables>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyse_swap as rule


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def typed_points(tables: Path) -> list[dict]:
    return [{**p, "t": float(p["t"]), "b": float(p["b"]), "near_duplicate": rule.truthy(p["near_duplicate"])}
            for p in read(tables / "points.csv")]


def typed_deltas(tables: Path) -> list[dict]:
    return [{**d, "t": float(d["t"]), "delta": float(d["delta"]), "near_duplicate": rule.truthy(d["near_duplicate"])}
            for d in read(tables / "delta.csv")]


def readings(crv: list[dict], labels: list[str], cls: str) -> dict:
    out = {}
    for axis in rule.AXES:
        per = {x: rule.verdict(rule.counted(crv, x, axis, cls), axis) for x in labels}
        out[axis] = {"agents": {x: {k: v[k] for k in ("plateau", "plateau_t", "n_far_points", "counted", "verdict") if k in v}
                                for x, v in per.items()}, "conclusion": rule.conclusion(per)}
    return out


def far_levels(pts: list[dict], labels: list[str]) -> dict:
    cells = defaultdict(list)
    for p in pts:
        if p["label"] in labels and p["t"] >= rule.FAR_T:
            cells[(p["label"], p["axis"], p["class"])].append(p["b"])
    return {x: {axis: {cls: ({"mean": round(float(np.mean(cells[(x, axis, cls)])), 3), "n_cells": len(cells[(x, axis, cls)])}
                             if cells[(x, axis, cls)] else None) for cls in rule.CLASSES} for axis in rule.AXES} for x in labels}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args(argv)
    pts, dls = typed_points(args.tables), typed_deltas(args.tables)
    shuffle = sorted({p["label"] for p in pts if p["label"].startswith("shuffle_")})
    crv1, crv2 = rule.curves(pts, dls, min_swaps=1), rule.curves(pts, dls)
    out = {"post_hoc": True, "delta_1_swap": readings(crv1, shuffle, "delta"), "moved_2_swaps": readings(crv2, shuffle, "moved"),
           "moved_1_swap": readings(crv1, shuffle, "moved"), "far_levels": far_levels(pts, shuffle)}
    (args.tables / "posthoc.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key in ("delta_1_swap", "moved_2_swaps", "moved_1_swap"):
        for axis, d in out[key].items():
            ag = {x.split("_")[1]: (None if v["plateau"] is None else round(v["plateau"], 2), v["verdict"], v["plateau_t"])
                  for x, v in d["agents"].items()}
            print(f"{key} {axis}: {d['conclusion']} {ag}", flush=True)
    for x, axes in out["far_levels"].items():
        for axis, c in axes.items():
            print(x, axis, {cls: (v["mean"], v["n_cells"]) if v else None for cls, v in c.items()}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
