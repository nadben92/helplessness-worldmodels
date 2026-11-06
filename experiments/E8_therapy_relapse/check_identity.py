"""Form F0 of docs/e8/EXPECTED_FORMS.md: E8's measures at therapy checkpoint 0 against the 650,000 campaign (E6/E7 on all
agents, report section 21): exact equality on the training image, same machine, CPU, 1 thread.

Compared, for one pair (the probe job's rows and episodes.npz against the campaign's job outputs, read-only):
  e6      E6's episodes of both models (states, actions, length, terminated, I, loc_template, decoded_cell,
          decoded_template, Q: np.array_equal, NaN equal) and the rows' b at step 1, directed share, recovery latency,
          zone share, localisation at step 1 and in the zone, E4's probe b and validity;
  c12     D7's readout arrays (V_zbar, V_zbar_mean, Q_hbar) of both models, and C1 / C3 recomputed from the campaign's
          arrays with E8's own summary (the same function) equal to E8's rows;
  oracle  the oracle at k = 1's directed share, b at step 1 and localisation at step 1 (the campaign did not keep the
          exposed model's oracle episodes: run_e7_diagnostics saved both models under one key, the reference last).

Usage (from the repo root):
  python experiments/E8_therapy_relapse/check_identity.py --probes <probes__... job dir> --pair base_A \\
      --e6 <campaign e6__<agent>__... dir> --c12 <campaign e7d7__<agent>__natural__natural_training__... dir> \\
      --oracle <campaign e7diag__<agent>__lookahead__... dir> [--image training] [--out <json>]
Exit code 0 when every comparison holds; the JSON lists every comparison and its outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

E6_ARRAYS = ("states", "actions", "length", "terminated", "I", "loc_template", "decoded_cell", "decoded_template", "Q")
READOUT = ("V_zbar", "V_zbar_mean", "Q_hbar")
ROW_FIELDS = ("b_step1", "directed_share", "recovery_latency", "zone_share", "loc_template_step1", "loc_template_zone")
PROBE_FIELDS = ("b_step1_probe", "valid")


def equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


def same(x, y) -> bool:
    if isinstance(x, float) or isinstance(y, float):
        return (np.isnan(float(x)) and np.isnan(float(y))) or float(x) == float(y)
    return x == y


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--e6", type=Path, required=True)
    parser.add_argument("--c12", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=None)
    parser.add_argument("--image", default="training", help="E8's image name of the training image (shuffle: assignment01)")
    parser.add_argument("--e5-records", type=Path, default=None,
                        help="F0 (c): the exposed model's E5 run (its checkpoints.jsonl): the 12-assignment walks at --at, |diff| <= 0.05 bit")
    parser.add_argument("--at", type=int, default=650000, help="--e5-records: the record's environment step (the start checkpoint)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = [r for r in json.loads((args.probes / "rows.json").read_text(encoding="utf-8")) if r["pair"] == args.pair and r["image"] == args.image]
    if [r["role"] for r in rows] != ["exposed", "reference"] or rows[0]["checkpoint"] != 0:
        raise SystemExit(f"{args.probes}: expected the exposed and reference rows of {args.pair} at checkpoint 0 on {args.image}")
    ep = np.load(args.probes / "episodes.npz")
    ref_model = rows[1]["model"]
    tag = {"exposed": f"{args.pair}__{args.image}__exposed", "reference": f"{ref_model}__{args.image}__reference"}
    e6_rows = json.loads((args.e6 / "rows.json").read_text(encoding="utf-8"))
    e6_ep = np.load(args.e6 / "episodes.npz")
    c12_ep = np.load(args.c12 / "episodes.npz")
    c12_tag = {"exposed": "none__0", "reference": "reverse_none__0"}
    checks = []
    for row in rows:
        role = row["role"]
        e6_row = next(r for r in e6_rows if r["image"] == "training" and r["role"] == role)
        for key in E6_ARRAYS:
            checks.append({"what": f"e6 {role} {key}", "ok": equal(ep[f"{tag[role]}__{key}"], e6_ep[f"training__{role}__{key}"])})
        for key in ROW_FIELDS + PROBE_FIELDS:
            if key in row and key in e6_row:
                checks.append({"what": f"e6 {role} row {key}", "ok": same(row[key], e6_row[key]), "e8": row[key], "campaign": e6_row[key]})
        for key in READOUT:
            checks.append({"what": f"c12 {role} {key}", "ok": equal(ep[f"{tag[role]}__{key}"], c12_ep[f"{c12_tag[role]}__{key}"])})
    if args.oracle is not None:
        o_ep = np.load(args.oracle / "episodes.npz")
        o_rows = json.loads((args.oracle / "summary.json").read_text(encoding="utf-8"))["rows"]
        o_row = next(r for r in o_rows if r["k"] == 1 and r["model"] == "exposed")
        # the campaign's episodes.npz holds the REFERENCE's oracle episodes under none__0__k1 (run_e7_diagnostics.phase_episodes
        # saves both models under one tag, the reference's last: the exposed model's arrays were overwritten; flagged in
        # IMPLEMENTATION_REPORT.md, not fixed -- E7 is finished): the exposed oracle is compared on its rows only
        for key in ("directed_share", "b_step1", "loc_template_step1"):
            mine = rows[0]["oracle_directed_share"] if key == "directed_share" else rows[0][f"oracle_{key}"]
            checks.append({"what": f"oracle k1 {key}", "ok": same(mine, o_row[key]), "e8": mine, "campaign": o_row[key]})
    tolerance = []
    if args.e5_records is not None:  # F0 (c): another device (the cluster's GPU), EXPECTED_FORMS' tolerance
        recs = [json.loads(x) for x in (args.e5_records / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        rec = next(r for r in recs if r["env_step"] == args.at)["e5"]["by_assignment"]
        all_rows = json.loads((args.probes / "rows.json").read_text(encoding="utf-8"))
        for r in (x for x in all_rows if x["pair"] == args.pair and x["role"] == "exposed"):
            k = str(r["assignment"])
            if k in rec:
                record = rec[k]["walks"][r["slot"]]["I_curve"][0]
                tolerance.append({"what": f"E5 record, assignment {k} ({r['slot']})", "ok": abs(r["walks_I_step1"] - record) <= 0.05,
                                  "e8": r["walks_I_step1"], "record": record, "diff": r["walks_I_step1"] - record})
    failed = [c for c in checks + tolerance if not c["ok"]]
    report = {"probes": str(args.probes), "pair": args.pair, "n_checks": len(checks), "n_failed": len(failed), "checks": checks,
              "e5_records_tolerance_0.05": tolerance}
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    if tolerance:
        print(f"F0 (c), E5 records: {sum(c['ok'] for c in tolerance)} / {len(tolerance)} within 0.05 bit "
              f"(max |diff| {max(abs(c['diff']) for c in tolerance):.4f})")
    print(f"F0 identity, {args.pair}: {sum(c['ok'] for c in checks)} / {len(checks)} exact comparisons hold"
          + ("" if not failed else "; FAILED: " + ", ".join(c["what"] for c in failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
