"""The 12 training assignments in the shuffle pairs' own checkpoint records (e5.by_assignment, the walks' I_curve[0] at
step 1, experimental - reference; the records' protocol: e4.checkpoint.n_walks walks, n_in = e4.walks.n_in, purpose
"checkpoint"), read-only, no weights: b at the UD slot (the T_inc slot of that assignment) against the CD slot and the
neutral slots, per agent. THEME_EVIDENCE.md, item 1. The reference's mass on the true successor (mass_true_mean) is given
beside every slot; E4's validity threshold (0.90) is applied to it only to COUNT the slots that would pass (the records'
walks are not the probes', so this is not the probes' validity).

Usage: python experiments/E5_belief_training/analysis_swap/assignments_records.py
           --pair <label> <experimental run dir> <reference run dir> [--pair ...] --checkpoint 650000 --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from helplessness.pipeline.episode_sources import ASSIGNMENTS
from point0_records import record_at

SLOTS = ("nw", "ne", "se", "sw")
MASS_MIN = 0.90


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=3, action="append", required=True, metavar=("LABEL", "EXPERIMENTAL", "REFERENCE"))
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for label, exp_dir, ref_dir in args.pair:
        rec = {name: record_at(Path(d), args.checkpoint)["e5"]["by_assignment"] for name, d in (("exp", exp_dir), ("ref", ref_dir))}
        for k, (ud, cd) in enumerate(ASSIGNMENTS):
            for slot in SLOTS:
                walks = {name: rec[name][str(k)]["walks"][slot] for name in rec}
                rows.append({"label": label, "assignment": k, "ud_slot": ud, "cd_slot": cd, "slot": slot,
                             "role": "ud" if slot == ud else "cd" if slot == cd else "neutral",
                             "b_step1": float(walks["exp"]["I_curve"][0]) - float(walks["ref"]["I_curve"][0]),
                             "mass_ref": float(walks["ref"]["mass_true_mean"])})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for label, *_ in args.pair:
        mine = [r for r in rows if r["label"] == label]
        for role in ("ud", "cd", "neutral"):
            b = np.array([r["b_step1"] for r in mine if r["role"] == role])
            ok = np.array([r["mass_ref"] >= MASS_MIN for r in mine if r["role"] == role])
            print(f"{label} {role:7}: n {b.size}, mean {b.mean():+.3f}, min {b.min():+.3f}, max {b.max():+.3f}; "
                  f"reference mass >= {MASS_MIN}: {ok.sum()}/{ok.size} (their mean {b[ok].mean() if ok.any() else float('nan'):+.3f})", flush=True)
        ud = {r["assignment"]: r["b_step1"] for r in mine if r["role"] == "ud"}
        others = {k: max(r["b_step1"] for r in mine if r["assignment"] == k and r["role"] != "ud") for k in ud}
        print(f"{label}: UD below every other slot in {sum(ud[k] < others[k] for k in ud)}/12 assignments; "
              f"min gap (other max - UD) {min(others[k] - ud[k] for k in ud):+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
