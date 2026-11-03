"""The swap probes at t = 0 (GPU, the batch's own measurement) against the training records of the same assignment
(docs/e5/DECISIONS.md (38), "expected shape"): no weights, read-only. The base image of each swap (segment k = 0) IS a
training assignment (4, 11, 6 for p1-b-d1); its record is b = I_curve[0](experimental) - I_curve[0](reference) of
e5.by_assignment at the same checkpoint. check_point0.py is the CPU counterpart on local weights (before launch).

Usage: python experiments/E5_belief_training/analysis_swap/point0_records.py --rows <tables>/rows.csv
           --pair <label> <experimental run dir> <reference run dir> [--pair ...] --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from helplessness.pipeline.episode_sources import assignment_index


def record_at(run_dir: Path, k_step: int) -> dict:
    for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(r.get("env_step", -1)) == k_step and r.get("e5"):
                return r
    raise SystemExit(f"{run_dir}: no E5 record at {k_step}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--pair", nargs=3, action="append", required=True, metavar=("LABEL", "EXPERIMENTAL", "REFERENCE"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with open(args.rows, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["axis"] == "segment" and int(r["k"]) == 0]
    out = []
    for label, exp_dir, ref_dir in args.pair:
        k_step = int(label.rsplit("_k", 1)[1])
        rec = {name: record_at(Path(d), k_step)["e5"]["by_assignment"] for name, d in (("exp", exp_dir), ("ref", ref_dir))}
        for r in (r for r in rows if r["label"] == label):
            k = str(assignment_index(r["ud_slot"], r["cd_slot"]))
            i_exp, i_ref = (float(rec[name][k]["walks"][r["zone"]]["I_curve"][0]) for name in ("exp", "ref"))
            out.append({"label": label, "swap": r["swap"], "assignment": int(k), "zone": r["zone"], "zone_role": r["zone_role"],
                        "is_old_ud": r["is_old_ud"], "b_probe": float(r["b_step1"]), "b_record": i_exp - i_ref,
                        "difference": float(r["b_step1"]) - (i_exp - i_ref), "valid": r["valid"], "kept": r["kept"],
                        "mass_ref_step1": float(r["mass_ref_step1"]),
                        "mass_ref_record": float(rec["ref"][k]["walks"][r["zone"]]["mass_true_mean"])})
    if not out:
        raise SystemExit("no row matched the labels given")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(f"{o['label']} {o['swap']} (assignment {o['assignment']}) {o['zone']} {o['zone_role']:7}: probe {o['b_probe']:+.3f}, "
              f"record {o['b_record']:+.3f}, diff {o['difference']:+.3f}, kept {o['kept']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
