"""E5's probes of the shuffle pairs at 475,000 against the baselines (EXPECTED_SHAPES.md, docs/e5/DECISIONS.md (36)).

Reads measure_probes.py outputs (read-only) and E4's final probe table; writes CSV tables and summary.json into --out.
b at step 1 and the validity come from analyse_probes.load_probe_dir (curves.npz, probes.csv's `valid`); nothing of the
measure is recomputed here. Invalid probes are counted, never imputed.

Measurements: baseline A and B at 300,000 (condition `baseline`), baseline B at its matched checkpoint (`baseline_matched`),
shuffle f = 1 E and F at 475,000 (`shuffle`). Tables:
  measurements.csv         one row per measurement: pairing, checkpoint, walks, probes and valid probes
  validity_by_position.csv valid probes by axis and position
  extremes.csv             the training image (every zone), position only, theme only: b at step 1, validity, reference
  e4_replication.csv       baseline A against E4's probes_b_step1.csv (p1-b-d1), probe by probe
  gradient_probes.csv      every (image, slot) once: distance of the probed colour to T_inc over D, b at step 1, validity
  gradient_binned.csv      valid probes binned by distance (0.1 D wide, centred on multiples of 0.1 D), by condition and position (seeds pooled), and by seed
  distribution.csv         valid probes by measurement: quantiles and the counts by class of b
  recovery.csv             b by step since entry, training image, UD slot (E4d's curve), and its gain steps 6-10 vs 1
  matched_*.csv            the matched comparison of (36) (analyse_probes.matched), and its summary by position

Usage: python experiments/E5_belief_training/analysis_450k/analyse_450k.py <probe dir> ... --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyse_probes as AP  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.envs.appearance.theme_axes import rgb_distance  # noqa: E402

E4_TABLE = REPO_ROOT / "results" / "E4_generalization_gradient" / "final" / "tables" / "probes_b_step1.csv"
LOG2_5 = float(np.log2(5))
BINS = np.round(np.arange(-0.05, 2.0501, 0.1), 2)  # centred on multiples of 0.1 D, where the probes' distances fall
CLASSES = [("helpless", -np.inf, -1.5), ("intermediate", -1.5, -0.5), ("in_control", -0.5, np.inf)]
TRAINING_ROLES = {"ud": "training_ud", "cd": "training_cd", "neutral": "training_neutral"}


def label_of(m: dict) -> str:
    return {402: "shuffle_E", 502: "shuffle_F", 2: "baseline_A", 102: "baseline_B"}[m["seed"]] + f"_k{m['k']}"


def group_of(m: dict, matched_k: int) -> str:
    if m["condition"] == "baseline":
        return "baseline_matched" if m["k"] == matched_k else "baseline"
    return "shuffle"


def load(d: Path, matched_k: int) -> dict:
    m = AP.load_probe_dir(d)
    meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
    t_inc, t_ctrl = tuple(meta["t_inc"]), tuple(meta["t_ctrl"])
    D = float(rgb_distance(t_inc, t_ctrl))
    rows = AP.read_csv(d / "probes.csv")
    for p, r in zip(m["probes"], rows):  # load_probe_dir keeps probes.csv's order
        assert (p["image"], p["slot"]) == (int(r["image"]), r["slot"])
        p.update(dist_to_inc_over_D=float(r["dist_to_inc"]) / D, mass_ref=float(r["mass_ref_step1"]), I_ref=float(r["I_ref_step1"]))
    curves = np.load(d / "curves.npz")
    b_curve = curves["curve_exp"] - curves["curve_ref"]
    images = AP.read_csv(d / "images.csv")
    m.update(meta=meta, D=D, label=label_of(m), group=group_of(m, matched_k), images=images, b_curve=b_curve)
    return m


def unique_probes(m: dict) -> list[dict]:
    """Every (image, slot) once (an image shared by several axes is one measurement); position only left out."""
    seen = {}
    for p in m["probes"]:
        if p["axis"] != "position_only":
            seen.setdefault((p["image"], p["slot"]), p)
    return list(seen.values())


def measurement_rows(ms: list[dict]) -> list[dict]:
    out = []
    for m in ms:
        u = unique_probes(m)
        out.append({"label": m["label"], "group": m["group"], "condition": m["condition"], "seed": m["seed"],
                    "reference_seed": m["meta"]["reference_seed"], "checkpoint_k": m["k"],
                    "n_walks_used": m["meta"]["walks"]["n_walks_used"], "probes": len(m["probes"]),
                    "valid": sum(p["valid"] for p in m["probes"]), "unique_image_slots": len(u),
                    "unique_valid": sum(p["valid"] for p in u), "dir": m["dir"].name})
    return out


def validity_rows(ms: list[dict]) -> list[dict]:
    out = []
    for m in ms:
        for axis in ("segment", "arc_ctrl", "arc_inc", "position_only", "all"):
            for pos in ("ud", "cd", "neutral", "all"):
                sel = [p for p in m["probes"] if axis in ("all", p["axis"]) and pos in ("all", p["position"])]
                if sel:
                    out.append({"label": m["label"], "group": m["group"], "axis": axis, "position": pos, "probes": len(sel),
                                "valid": sum(p["valid"] for p in sel), "share_valid": sum(p["valid"] for p in sel) / len(sel)})
    return out


def extreme_rows(ms: list[dict]) -> list[dict]:
    out = []
    for m in ms:
        for i, r in enumerate(m["images"]):
            if AP.truthy(r["is_training"]):
                out.append({"label": m["label"], "group": m["group"], "seed": m["seed"], "probe": TRAINING_ROLES.get(r["role"], "training_" + r["role"]),
                            "slot": r["zone"], "b_step1": float(m["b_curve"][i, 0]), "valid": AP.truthy(r["valid"]),
                            "mass_ref_step1": float(r["mass_ref_step1"]), "I_ref_step1": float(r["I_ref_step1"]),
                            "mass_exp_step1": float(r["mass_exp_step1"])})
        for name, (axis, k, pos) in AP.EXTREMES.items():
            p = next(p for p in m["probes"] if (p["axis"], p["k"], p["position"]) == (axis, k, pos))
            img = next(r for r in m["images"] if int(r["image"]) == p["image"] and r["zone"] == p["slot"])
            out.append({"label": m["label"], "group": m["group"], "seed": m["seed"], "probe": name, "slot": p["slot"],
                        "b_step1": p["b_step1"], "valid": p["valid"], "mass_ref_step1": p["mass_ref"], "I_ref_step1": p["I_ref"],
                        "mass_exp_step1": float(img["mass_exp_step1"])})
    return out


def e4_replication(m: dict, e4_table: Path) -> list[dict]:
    if not e4_table.is_file():
        raise SystemExit(f"missing E4 table: {e4_table}")
    e4 = {(r["axis"], int(r["k"]), r["position"]): r for r in AP.read_csv(e4_table) if r["configuration"] == "p1-b-d1"}
    out = []
    for p in m["probes"]:
        q = e4[(p["axis"], p["k"], p["position"])]
        assert (int(q["image"]), q["slot"]) == (p["image"], p["slot"])
        both = p["valid"] and AP.truthy(q["valid"])
        out.append({"axis": p["axis"], "k": p["k"], "position": p["position"], "image": p["image"], "slot": p["slot"],
                    "b_e5": p["b_step1"], "valid_e5": p["valid"], "b_e4": float(q["b_step1"]), "se_e4": float(q["se_b_step1"]),
                    "valid_e4": AP.truthy(q["valid"]), "difference": p["b_step1"] - float(q["b_step1"]), "valid_both": both})
    return out


def gradient_rows(ms: list[dict]) -> tuple[list[dict], list[dict]]:
    probes = [{"label": m["label"], "group": m["group"], "seed": m["seed"], "image": p["image"], "slot": p["slot"],
               "position": p["position"], "axis_first": p["axis"], "dist_to_inc_over_D": p["dist_to_inc_over_D"],
               "b_step1": p["b_step1"], "valid": p["valid"]} for m in ms for p in unique_probes(m)]
    binned = []
    for unit in ("group", "label"):
        for key in sorted({r[unit] for r in probes}):
            for pos in ("ud", "cd", "neutral"):
                sel = [r for r in probes if r[unit] == key and r["position"] == pos and r["valid"]]
                x = np.array([r["dist_to_inc_over_D"] for r in sel])
                b = np.array([r["b_step1"] for r in sel])
                idx = np.clip(np.digitize(x, BINS[1:-1]), 0, len(BINS) - 2)
                for i in range(len(BINS) - 1):
                    if (idx == i).any():
                        binned.append({"unit": unit, "key": key, "position": pos, "bin_from": BINS[i], "bin_to": BINS[i + 1],
                                       "n": int((idx == i).sum()), "x_mean": float(x[idx == i].mean()),
                                       "b_mean": float(b[idx == i].mean()),
                                       "b_sd": float(b[idx == i].std(ddof=1)) if (idx == i).sum() > 1 else float("nan")})
    return probes, binned


def distribution_rows(ms: list[dict]) -> list[dict]:
    out = []
    for m in ms:
        b = np.array([p["b_step1"] for p in unique_probes(m) if p["valid"]])
        row = {"label": m["label"], "group": m["group"], "n_valid_unique": len(b), "min": b.min(), "q25": np.quantile(b, 0.25),
               "median": np.median(b), "q75": np.quantile(b, 0.75), "max": b.max()}
        for name, lo, hi in CLASSES:
            row[f"n_{name}"] = int(((b > lo) & (b <= hi)).sum()) if np.isfinite(lo) else int((b <= hi).sum())
        out.append(row)
    return out


def recovery_rows(ms: list[dict]) -> list[dict]:
    out = []
    for m in ms:
        i = next(i for i, r in enumerate(m["images"]) if AP.truthy(r["is_training"]) and r["role"] == "ud")
        c = m["b_curve"][i]
        out.append({"label": m["label"], "group": m["group"], "seed": m["seed"], "valid": AP.truthy(m["images"][i]["valid"]),
                    "gain_steps6_10_minus_step1": float(c[5:10].mean() - c[0]), **{f"b_k{k + 1}": float(c[k]) for k in range(len(c))}})
    return out


def matched_summary(rows: list[dict]) -> list[dict]:
    out = []
    for comp in sorted({r["comparison"] for r in rows}):
        for pos in ("ud", "cd", "neutral", "all"):
            sel = [r for r in rows if r["comparison"] == comp and pos in ("all", r["position"])]
            d = np.array([r["difference"] for r in sel if r["valid_first"] and r["valid_second"]])
            out.append({"comparison": comp, "position": pos, "probes": len(sel),
                        "valid_both": int(sum(r["valid_first"] and r["valid_second"] for r in sel)),
                        "valid_first_only": int(sum(r["valid_first"] and not r["valid_second"] for r in sel)),
                        "valid_second_only": int(sum(r["valid_second"] and not r["valid_first"] for r in sel)),
                        "valid_neither": int(sum(not r["valid_first"] and not r["valid_second"] for r in sel)),
                        "difference_mean": float(d.mean()) if len(d) else float("nan"),
                        "difference_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan")})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--matched-k", type=int, default=63096, help="baseline B's checkpoint matched to the pairs' level ((36))")
    parser.add_argument("--e4-table", type=Path, default=E4_TABLE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ms = sorted((load(d, args.matched_k) for d in args.dirs), key=lambda m: (m["group"], m["seed"], m["k"]))
    if len({m["label"] for m in ms}) != len(ms):
        raise SystemExit("two measurements with the same label")
    args.out.mkdir(parents=True, exist_ok=True)
    AP.write_csv(args.out / "measurements.csv", measurement_rows(ms))
    AP.write_csv(args.out / "validity_by_position.csv", validity_rows(ms))
    AP.write_csv(args.out / "extremes.csv", extreme_rows(ms))
    base_a = [m for m in ms if m["seed"] == 2 and m["k"] == 300000]
    rep = e4_replication(base_a[0], args.e4_table) if base_a else []
    AP.write_csv(args.out / "e4_replication.csv", rep)
    probes, binned = gradient_rows(ms)
    AP.write_csv(args.out / "gradient_probes.csv", probes)
    AP.write_csv(args.out / "gradient_binned.csv", binned)
    AP.write_csv(args.out / "distribution.csv", distribution_rows(ms))
    AP.write_csv(args.out / "recovery.csv", recovery_rows(ms))
    comparisons = [f"{m['condition']}:{m['seed']}@{m['k']}=baseline:102@{args.matched_k}" for m in ms if m["group"] == "shuffle"]
    by_probe, agg = AP.matched(ms, comparisons)
    AP.write_csv(args.out / "matched_by_probe.csv", by_probe)
    AP.write_csv(args.out / "matched_aggregated.csv", agg)
    AP.write_csv(args.out / "matched_summary.csv", matched_summary(by_probe))
    d = np.array([r["difference"] for r in rep if r["valid_both"]])
    summary = {"measurements": {m["label"]: m["dir"].name for m in ms}, "matched": comparisons,
               "e4_replication": {"probes_valid_both": int(len(d)), "mean_abs_difference": float(np.abs(d).mean()) if len(d) else None,
                                  "max_abs_difference": float(np.abs(d).max()) if len(d) else None,
                                  "n_above_0.05": int((np.abs(d) > 0.05).sum()),
                                  "validity_agreement": int(sum(r["valid_e5"] == r["valid_e4"] for r in rep)), "probes": len(rep)}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["e4_replication"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
