"""LOCALISATION_CRITERION.md applied to the analysis tables: a probe is kept if valid and the exposed model's localisation
at step 1 (direct rule, localisation_probes.py) is >= 0.90. Reads tables/gradient_probes.csv, tables/extremes.csv and
tables/localisation_<label>.csv (one per measurement); writes tables/gradient_probes_loc.csv, gradient_binned_loc.csv
(analyse_450k's bins, unit = condition group and measurement), extremes_loc.csv and loc_removed.csv (probes removed by
the localisation term per measurement, position and distance bin).

Usage: python experiments/E5_belief_training/analysis_450k/apply_localisation.py [--tables <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyse_probes as AP  # noqa: E402
from analyse_450k import BINS  # noqa: E402

LOC_MIN = 0.90
LABELS = ("baseline_A_k300000", "baseline_B_k300000", "shuffle_E_k475000", "shuffle_F_k475000")


def bin_index(x: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, BINS[1:-1]), 0, len(BINS) - 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, default=Path(__file__).resolve().parent / "tables")
    args = parser.parse_args(argv)
    loc = {}
    for label in LABELS:
        for r in AP.read_csv(args.tables / f"localisation_{label}.csv"):
            loc[(label, int(r["image"]), r["slot"])] = float(r["loc_agent_cell_step1"])
    probes = []
    for r in AP.read_csv(args.tables / "gradient_probes.csv"):
        if r["label"] not in LABELS:
            continue
        lv = loc[(r["label"], int(r["image"]), r["slot"])]
        probes.append({**r, "loc_step1": lv, "kept": AP.truthy(r["valid"]) and lv >= LOC_MIN})
    AP.write_csv(args.tables / "gradient_probes_loc.csv", probes)
    binned, removed = [], []
    for unit in ("group", "label"):
        for key in sorted({r[unit] for r in probes}):
            for pos in ("ud", "cd", "neutral"):
                sel = [r for r in probes if r[unit] == key and r["position"] == pos and r["kept"]]
                x = np.array([float(r["dist_to_inc_over_D"]) for r in sel])
                b = np.array([float(r["b_step1"]) for r in sel])
                idx = bin_index(x)
                for i in range(len(BINS) - 1):
                    if (idx == i).any():
                        binned.append({"unit": unit, "key": key, "position": pos, "bin_from": BINS[i], "bin_to": BINS[i + 1],
                                       "n": int((idx == i).sum()), "x_mean": float(x[idx == i].mean()), "b_mean": float(b[idx == i].mean())})
    for label in LABELS:
        for pos in ("ud", "cd", "neutral"):
            valid = [r for r in probes if r["label"] == label and r["position"] == pos and AP.truthy(r["valid"])]
            idx = bin_index(np.array([float(r["dist_to_inc_over_D"]) for r in valid]))
            for i in sorted(set(idx.tolist())):
                cell = [r for r, j in zip(valid, idx) if j == i]
                gone = [r for r in cell if not r["kept"]]
                removed.append({"label": label, "position": pos, "bin_centre": round(float(BINS[i]) + 0.05, 2), "valid": len(cell),
                                "removed_by_localisation": len(gone),
                                "b_removed": " ".join(f"{float(r['b_step1']):.2f}" for r in gone),
                                "loc_removed": " ".join(f"{r['loc_step1']:.2f}" for r in gone)})
    AP.write_csv(args.tables / "gradient_binned_loc.csv", binned)
    AP.write_csv(args.tables / "loc_removed.csv", removed)
    extremes = []
    for r in AP.read_csv(args.tables / "extremes.csv"):
        if r["label"] in LABELS and r["probe"] in ("position_only", "theme_only"):
            image = 92 if r["probe"] == "position_only" else 2
            lv = loc[(r["label"], image, r["slot"])]
            extremes.append({**r, "loc_step1": lv, "kept": AP.truthy(r["valid"]) and lv >= LOC_MIN})
    AP.write_csv(args.tables / "extremes_loc.csv", extremes)
    for label in LABELS:
        sel = [r for r in probes if r["label"] == label]
        print(f"{label}: valid {sum(AP.truthy(r['valid']) for r in sel)}, kept {sum(r['kept'] for r in sel)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
