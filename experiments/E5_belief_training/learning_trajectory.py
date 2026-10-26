"""The learning trajectory of E5's runs (docs/e5/DECISIONS.md (33)), from their light files (checkpoints.jsonl): at every
checkpoint of the grid, the prior's mass on the true successor by quadrant of the canonical layout (and, for the shuffle
runs, by zone type averaged over the 12 assignments where every assignment was measured), and b at the first step in
the zone of the UD slot in the training world, per pair (check_wave.py's pairing: a run's own reference, or
batches.REFERENCE_SEED_OF's); E4's runs (the baseline A) are read the same way.

Also, at the last checkpoint measuring every assignment: the mass by zone type for each assignment, with its symmetry
class (e5.assignment_symmetry: rotation class, the CD slot's relation to UD), to locate the deficit; and, per run, the
trend of the smallest quadrant mass over the last checkpoints (a linear fit in environment steps from --trend-from on)
and, if it rises, the step at which it would reach E4's threshold (0.97).

Usage: python experiments/E5_belief_training/learning_trajectory.py --runs <run dir> ... --out <dir>
Outputs: trajectory_runs.csv, trajectory_pairs.csv, deficit_by_assignment.csv, summary.json, trajectory.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import check_wave as cw

MASS_MIN = cw.MASS_MIN


def label(run: dict) -> str:
    kind = "base" if run["tag"] in ("base", "d1") else run["tag"]
    origin = "E4" if run["tag"] == "d1" else "E5"
    return f"{origin} {kind} s{run['seed']} {'exp' if run['role'] == 'experimental' else 'ref'}"


def run_rows(run: dict) -> list[dict]:
    rows = []
    for r in run["records"]:
        mass = cw.mass_of(r)
        measured = [v for v in mass.values() if v is not None]
        mo = (r.get("e5") or {}).get("mean_over_assignments") or {}
        by_type = {t: cw.number(v) for t, v in (mo.get("mass_true_by_type") or {}).items()}
        rows.append({"run": label(run), "env_step": r["env_step"], "grad_step": r["grad_step"], **{f"mass_{q}": mass[q] for q in cw.QUADRANTS},
                     "mass_min": min(measured) if measured else None, **{f"mass_type_{t}": by_type.get(t) for t in ("ud", "cd", "neutral")},
                     "success_rate": cw.number(r.get("success_rate"))})
    return rows


def pair_rows(exp: dict, ref: dict) -> list[dict]:
    by_step = {r["env_step"]: r for r in ref["records"]}
    rows = []
    for r in exp["records"]:
        q = by_step.get(r["env_step"])
        if q is None:
            continue
        e, f = cw.step1(r, cw.ROLES["ud"]), cw.step1(q, cw.ROLES["ud"])
        mo_e, mo_r = ((x.get("e5") or {}).get("mean_over_assignments") or {} for x in (r, q))
        b_type = None
        if mo_e and mo_r:
            a, c = cw.number(mo_e["I_step1_by_type"]["ud"]), cw.number(mo_r["I_step1_by_type"]["ud"])
            b_type = None if a is None or c is None else a - c
        rows.append({"pair": f"{label(exp)} vs {label(ref)}", "env_step": r["env_step"], "b_ud": None if e is None or f is None else e - f,
                     "b_ud_by_type_over_assignments": b_type})
    return rows


def deficit_rows(run: dict) -> list[dict]:
    for r in reversed(run["records"]):
        e5 = r.get("e5") or {}
        if len(e5.get("assignments_measured") or []) > 1:
            sym = e5.get("assignment_symmetry") or {}
            out = []
            for k, rec in sorted(e5["by_assignment"].items(), key=lambda kv: int(kv[0])):
                m = rec["real_context"]["mass_true_by_type"]
                s = sym.get(k, {})
                out.append({"run": label(run), "env_step": r["env_step"], "assignment": int(k), "ud": s.get("ud"), "cd": s.get("cd"),
                            "rotation_class": s.get("rotation_class"), "ud_cd": s.get("ud_cd"),
                            **{f"mass_{t}": cw.number(m.get(t)) for t in ("ud", "cd", "neutral", "corridor")}})
            return out
    return []


def trend(rows: list[dict], key: str, start: int) -> dict:
    pts = [(r["env_step"], r[key]) for r in rows if r["env_step"] >= start and r[key] is not None]
    if len(pts) < 3:
        return {"points": len(pts)}
    x, y = np.array(pts, dtype=float).T
    slope, intercept = np.polyfit(x, y, 1)
    out = {"points": len(pts), "from": int(x[0]), "to": int(x[-1]), "first": float(y[0]), "last": float(y[-1]),
           "slope_per_100k_steps": float(slope * 1e5)}
    if slope > 0 and y[-1] < MASS_MIN:
        out["steps_to_threshold_linear"] = float((MASS_MIN - intercept) / slope)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def plot(runs_rows: dict, pairs: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, rows in runs_rows.items():
        x = [r["env_step"] for r in rows if r["mass_min"] is not None]
        axes[0].plot(x, [r["mass_min"] for r in rows if r["mass_min"] is not None], marker=".", label=name,
                     ls="--" if "base" in name else "-")
        sel = [r for r in rows if r["mass_type_ud"] is not None]
        if sel:
            for t, mk in (("ud", "o"), ("cd", "s"), ("neutral", "^")):
                axes[1].plot([r["env_step"] for r in sel], [r[f"mass_type_{t}"] for r in sel], marker=mk, label=f"{name} {t}")
    for name, rows in pairs.items():
        sel = [r for r in rows if r["b_ud"] is not None]
        axes[2].plot([r["env_step"] for r in sel], [r["b_ud"] for r in sel], marker=".", label=name, ls="--" if "base" in name else "-")
    axes[0].axhline(MASS_MIN, color="k", ls=":")
    axes[1].axhline(MASS_MIN, color="k", ls=":")
    axes[2].axhline(-math.log2(5), color="k", ls=":")
    for ax, title in zip(axes, ("smallest quadrant mass (canonical layout)", "shuffle: mass by zone type, mean over the 12 assignments",
                                "b at step 1, UD slot (canonical), training world")):
        ax.set_xscale("log")
        ax.set_xlabel("environment steps")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=5.5, ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--trend-from", type=int, default=200000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = [cw.read_run(d) for d in args.runs]
    checks = {r["name"]: cw.check_run(r) for r in runs}
    healthy = [r for r in runs if checks[r["name"]]["lock_in"] is None and checks[r["name"]]["switch"]["grad_step"] is not None
               and checks[r["name"]]["status"] == "done"]
    runs_rows = {label(r): run_rows(r) for r in healthy}
    pairs = {}
    for exp in (r for r in healthy if r["role"] == "experimental"):
        ref_seed = cw.reference_seed(args.repo_dir.resolve(), exp["tag"], exp["seed"]) if exp["tag"] != "d1" else exp["seed"]
        ref = next((r for r in healthy if r["role"] == "reference" and r["tag"] == exp["tag"] and r["seed"] == ref_seed), None)
        if ref is not None:
            pairs[f"{label(exp)} vs {label(ref)}"] = pair_rows(exp, ref)
    deficit = [row for r in healthy for row in deficit_rows(r)]
    summary = {"healthy_runs": list(runs_rows), "excluded": [r["name"] for r in runs if r not in healthy],
               "trend_min_mass": {n: trend(rows, "mass_min", args.trend_from) for n, rows in runs_rows.items()},
               "trend_mass_type_ud": {n: trend(rows, "mass_type_ud", args.trend_from) for n, rows in runs_rows.items()
                                      if any(r["mass_type_ud"] is not None for r in rows)},
               "b_ud_last": {n: rows[-1] for n, rows in pairs.items() if rows}}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "trajectory_runs.csv", [row for rows in runs_rows.values() for row in rows])
    write_csv(args.out / "trajectory_pairs.csv", [row for rows in pairs.values() for row in rows])
    write_csv(args.out / "deficit_by_assignment.csv", deficit)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    plot(runs_rows, pairs, args.out / "trajectory.png")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
