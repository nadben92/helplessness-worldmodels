"""E5's final probes of the shuffle pairs at 650,000 against the baselines (docs/e5/DECISIONS.md (37)).

The tables of analysis_450k/analyse_450k.py, by its own functions, on the final measurements, plus pair A (seed 2's
shuffle pair, never probed before) and both comparisons of (37): the primary, each shuffle pair against the baseline of
its reference seed at the matched checkpoint (100,000), and the secondary, against the baselines at 300,000. Nothing of
the measure is recomputed here; invalid probes are counted, never imputed.

Measurements and groups: baseline A and B at 300,000 (`baseline`), at --matched-k (`baseline_matched`); shuffle f = 1 A,
E and F at --final-k (`shuffle`); any other shuffle checkpoint given (E and F at 475,000) as `shuffle_k<k>`, for the
comparison with the preliminary probes. A measurement's label is its condition and seed label (A = 2, B = 102, E = 402,
F = 502) and its k: seed 2 is both baseline A and shuffle A.

Tables in --out: analyse_450k's (measurements, validity_by_position, extremes, e4_replication, gradient_probes,
gradient_binned, distribution, recovery), and
  matched_by_probe.csv, matched_aggregated.csv, matched_summary.csv
      every comparison of a final shuffle measurement against its baseline at --matched-k ("matched") and at 300,000
      ("300k"), and of E and F at --final-k against their earlier checkpoint ("vs_k<k>"); a difference only where the
      probe is valid in both (analyse_probes.matched)
  ud_vs_neutral.csv, ud_vs_neutral_summary.csv
      b at the old UD slot against the neutral slot for the same probed colour (the same axis and k), probes valid at
      both slots; per measurement: the mean, SD and number of the differences UD minus neutral

Usage: python experiments/E5_belief_training/analysis_650k/analyse_650k.py <probe dir> ... --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis_450k"))

import analyse_450k as A450  # noqa: E402
import analyse_probes as AP  # noqa: E402

SEED_LABEL = {2: "A", 102: "B", 402: "E", 502: "F"}
BASELINE_K = 300000


def label_of(m: dict) -> str:
    return ("baseline" if m["condition"] == "baseline" else "shuffle") + f"_{SEED_LABEL[m['seed']]}_k{m['k']}"


def group_of(m: dict, matched_k: int, final_k: int) -> str:
    if m["condition"] == "baseline":
        return {BASELINE_K: "baseline", matched_k: "baseline_matched"}[m["k"]]
    return "shuffle" if m["k"] == final_k else f"shuffle_k{m['k']}"


def load(d: Path, matched_k: int, final_k: int) -> dict:
    m = A450.load(d, matched_k)
    m.update(label=label_of(m), group=group_of(m, matched_k, final_k))
    return m


def key(m: dict) -> str:
    return f"{m['condition']}:{m['seed']}@{m['k']}"


def comparisons(ms: list[dict], matched_k: int, final_k: int) -> list[tuple[str, str]]:
    """(name, "<a>=<b>"): each final shuffle measurement against the baseline of its reference seed at matched_k and at
    300,000, and against the same pair at its earlier checkpoints."""
    index = {(m["condition"], m["seed"], m["k"]): m for m in ms}
    out = []
    for m in ms:
        if m["group"] != "shuffle":
            continue
        ref = int(m["meta"]["reference_seed"])
        for name, k in (("matched", matched_k), ("300k", BASELINE_K)):
            b = index.get(("baseline", ref, k))
            if b is None:
                raise SystemExit(f"{m['label']}: no baseline of seed {ref} at {k}")
            out.append((name, f"{key(m)}={key(b)}"))
        for other in ms:
            if other["condition"] == m["condition"] and other["seed"] == m["seed"] and other["k"] != final_k:
                out.append((f"vs_k{other['k']}", f"{key(m)}={key(other)}"))
    return out


def ud_vs_neutral(ms: list[dict]) -> tuple[list[dict], list[dict]]:
    rows, summary = [], []
    for m in ms:
        by = {(p["axis"], p["k"], p["position"]): p for p in m["probes"]}
        colour = {(int(r["image"]), r["slot"]): r["color"] for r in AP.read_csv(m["dir"] / "probes.csv")}
        d = []
        for (axis, k, pos), p in sorted(by.items()):
            q = by.get((axis, k, "neutral"))
            if pos != "ud" or q is None:
                continue
            assert colour[(p["image"], p["slot"])] == colour[(q["image"], q["slot"])], (m["label"], axis, k)
            both = p["valid"] and q["valid"]
            rows.append({"label": m["label"], "group": m["group"], "axis": axis, "k": k, "color": colour[(p["image"], p["slot"])],
                         "dist_to_inc_over_D": p["dist_to_inc_over_D"], "b_ud": p["b_step1"], "valid_ud": p["valid"],
                         "b_neutral": q["b_step1"], "valid_neutral": q["valid"],
                         "ud_minus_neutral": p["b_step1"] - q["b_step1"] if both else float("nan")})
            if both:
                d.append(p["b_step1"] - q["b_step1"])
        d = np.array(d)
        summary.append({"label": m["label"], "group": m["group"], "pairs": sum(r["label"] == m["label"] for r in rows),
                        "valid_both": len(d), "mean": float(d.mean()) if len(d) else float("nan"),
                        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                        "min": float(d.min()) if len(d) else float("nan"), "max": float(d.max()) if len(d) else float("nan")})
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--matched-k", type=int, default=100000, help="the baselines' checkpoint matched to the pairs' level ((37))")
    parser.add_argument("--final-k", type=int, default=650000)
    parser.add_argument("--e4-table", type=Path, default=A450.E4_TABLE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ms = sorted((load(d, args.matched_k, args.final_k) for d in args.dirs), key=lambda m: (m["group"], m["seed"], m["k"]))
    if len({m["label"] for m in ms}) != len(ms):
        raise SystemExit("two measurements with the same label")
    args.out.mkdir(parents=True, exist_ok=True)
    AP.write_csv(args.out / "measurements.csv", A450.measurement_rows(ms))
    AP.write_csv(args.out / "validity_by_position.csv", A450.validity_rows(ms))
    AP.write_csv(args.out / "extremes.csv", A450.extreme_rows(ms))
    base_a = [m for m in ms if m["condition"] == "baseline" and m["seed"] == 2 and m["k"] == BASELINE_K]
    rep = A450.e4_replication(base_a[0], args.e4_table) if base_a else []
    AP.write_csv(args.out / "e4_replication.csv", rep)
    probes, binned = A450.gradient_rows(ms)
    AP.write_csv(args.out / "gradient_probes.csv", probes)
    AP.write_csv(args.out / "gradient_binned.csv", binned)
    AP.write_csv(args.out / "distribution.csv", A450.distribution_rows(ms))
    AP.write_csv(args.out / "recovery.csv", A450.recovery_rows(ms))
    comps = comparisons(ms, args.matched_k, args.final_k)
    by_probe, agg = AP.matched(ms, [c for _, c in comps])
    kind = dict((c, n) for n, c in comps)
    by_probe = [{"kind": kind[r["comparison"]], **r} for r in by_probe]
    AP.write_csv(args.out / "matched_by_probe.csv", by_probe)
    AP.write_csv(args.out / "matched_aggregated.csv", agg)
    AP.write_csv(args.out / "matched_summary.csv", [{"kind": kind[r["comparison"]], **r} for r in A450.matched_summary(by_probe)])
    uvn, uvn_summary = ud_vs_neutral(ms)
    AP.write_csv(args.out / "ud_vs_neutral.csv", uvn)
    AP.write_csv(args.out / "ud_vs_neutral_summary.csv", uvn_summary)
    d = np.array([r["difference"] for r in rep if r["valid_both"]])
    summary = {"measurements": {m["label"]: m["dir"].name for m in ms}, "comparisons": [{"kind": n, "comparison": c} for n, c in comps],
               "e4_replication": {"probes_valid_both": int(len(d)), "mean_abs_difference": float(np.abs(d).mean()) if len(d) else None,
                                  "max_abs_difference": float(np.abs(d).max()) if len(d) else None,
                                  "n_above_0.05": int((np.abs(d) > 0.05).sum()),
                                  "validity_agreement": int(sum(r["valid_e5"] == r["valid_e4"] for r in rep)), "probes": len(rep)}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["e4_replication"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
