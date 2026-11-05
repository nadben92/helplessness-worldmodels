"""The theme-swap probes (docs/e5/DECISIONS.md (38)): tables and the reading rule, from measure_probes.py --probe-set swap
outputs (probes-swap__<...> directories), read-only.

Zone classes of a swap image (swap_probes.csv): "moved" (the slot m that carries the swapped colour), "old_ud" (nw, the
canonical UD slot), "cd" (the assignment's CD slot, but the old UD slot), "neutral" (the assignment's neutral slots, but
the old UD slot). Only KEPT zones (valid and the exposed model localised at step 1, (38)) enter a value; b is b at step 1.

Tables (--out):
  rows.csv        every (probe, zone) row of every measurement, with its label (<condition>_<seed label>_k<k>) and class
  validity.csv    rows, valid and kept, by label, swap, axis and zone class
  points.csv      by label, axis, t, swap and class: b of the kept zone (neutral: the mean of its kept neutral slots)
  delta.csv       by label, axis, t and swap: Delta = b(moved) - b(neutral), where both are kept
  curves.csv      by label, axis, t and class (moved, neutral, old_ud, cd, delta): the mean over the swaps with a value, the
                  number of swaps with a value, whether the point COUNTS (at least MIN_SWAPS swaps) and near_duplicate
  reading.json    per shuffle agent and axis: its plateau (the mean of its PLATEAU_POINTS furthest counted Delta points,
                  all at t >= FAR_T; also the same without the near-duplicate points), whether it counts and its verdict ("colour" / "arrangement" /
                  "both"; also without the near-duplicate points); the witnesses (nw and the neutral slots: "about 0" or
                  "departs", with the t); the conclusion per axis (at least 2 counted agents with the same verdict, else
                  "not readable"). PRIMARY: arc_ctrl and arc_inc; the segment is SECONDARY, with and without its
                  near-duplicate points (t >= 0.8: two CD-coloured slots, no assignment matches, (38)). A measurement
                  named after its reference (measure_probes.py --name-reference, label <condition>_<seed>-ref<L>_k<k>)
                  SUPERSEDES the same agent's measurement against its own reference (SUPERSEDES, (39)): when both are
                  given, both are read, but only the superseding one enters the conclusions ("superseded" lists the other)
  shared_image.csv  (--compare-probes) the one swap image that is also an E4 probe image (T_ctrl at nw and se): b at step
                  1 at nw in the swap measurement against the E4-set measurement of the same model and checkpoint, and
                  the difference (a reproducibility control, (38)): the same model and walks, other draws

Usage: python experiments/E5_belief_training/analysis_swap/analyse_swap.py <probes-swap dir> ... --out <tables dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

SEED_LABELS = {2: "A", 102: "B", 402: "E", 502: "F"}  # configs/e5/e5.yaml, p1-b-d1 (e5_setup.configuration)
AXES = ("segment", "arc_ctrl", "arc_inc")
CLASSES = ("moved", "neutral", "old_ud", "cd")
MIN_SWAPS = 2  # (38): an agent counts at a point if Delta is computed in at least 2 of its 3 swaps
MIN_FAR_POINTS = 3  # ... and for an axis if it has >= 3 counted points with t >= 0.5 (and counts at t = 0, but on arc_inc)
FAR_T = 0.5
PRIMARY_AXES, SECONDARY_AXES = ("arc_ctrl", "arc_inc"), ("segment",)
PLATEAU_POINTS = 3  # the plateau: the mean of the 3 furthest counted points (all at t >= FAR_T)
ABOUT_MINUS_2, ABOUT_0 = -1.5, -0.5  # E5c's classes: b <= -1.5, -1.5 < b <= -0.5, b > -0.5
# docs/e5/DECISIONS.md (39): seed A's pair against B's reference is A's measurement; against its own, it is reported only
SUPERSEDES = {"A-refB": "A"}


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def zone_class(r: dict) -> str:
    if r["zone_role"] == "ud":
        return "moved"
    if truthy(r["is_old_ud"]):
        return "old_ud"
    return r["zone_role"]  # cd or neutral


def load(dirs: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for d in dirs:
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        if meta.get("probe_set") != "swap":
            raise SystemExit(f"{d}: not a swap probe measurement (meta.yaml probe_set)")
        status = json.loads((d / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "done":
            raise SystemExit(f"{d}: status {status.get('status')}")
        condition = "shuffle" if meta["training"] == "shuffle" else meta["training"]
        label = label_of(meta)
        if label in seen:
            raise SystemExit(f"{label} given twice ({d})")
        seen.add(label)
        for r in read_csv(d / "swap_probes.csv"):
            rows.append({"label": label, "condition": condition, "seed": int(meta["seed"]), "k_checkpoint": int(meta["checkpoint_k"]),
                         "class": zone_class(r), **r})
    return rows


def points(rows: list[dict]) -> list[dict]:
    """By (label, axis, t, swap, class): the kept zones' b (the mean of the kept slots of that class)."""
    cells, near = defaultdict(list), {}
    for r in rows:
        if truthy(r["kept"]):
            key = (r["label"], r["axis"], round(float(r["t"]), 2), r["swap"], r["class"])
            cells[key].append(float(r["b_step1"]))
            near[key] = truthy(r["near_duplicate"])
    return [{"label": key[0], "axis": key[1], "t": key[2], "swap": key[3], "class": key[4], "b": float(np.mean(v)), "n_slots": len(v),
             "near_duplicate": near[key]} for key, v in sorted(cells.items())]


def deltas(pts: list[dict]) -> list[dict]:
    by = {(p["label"], p["axis"], p["t"], p["swap"], p["class"]): p for p in pts}
    out = []
    for (label, axis, t, swap, cls), p in sorted(by.items()):
        if cls == "moved" and (label, axis, t, swap, "neutral") in by:
            out.append({"label": label, "axis": axis, "t": t, "swap": swap, "delta": p["b"] - by[(label, axis, t, swap, "neutral")]["b"],
                        "near_duplicate": p["near_duplicate"]})
    return out


def curves(pts: list[dict], dls: list[dict], min_swaps: int = MIN_SWAPS) -> list[dict]:
    cells, near = defaultdict(dict), defaultdict(bool)
    for p in pts:
        cells[(p["label"], p["axis"], p["t"], p["class"])][p["swap"]] = p["b"]
        near[(p["label"], p["axis"], p["t"], p["class"])] |= p["near_duplicate"]
    for d in dls:
        cells[(d["label"], d["axis"], d["t"], "delta")][d["swap"]] = d["delta"]
        near[(d["label"], d["axis"], d["t"], "delta")] |= d["near_duplicate"]
    return [{"label": key[0], "axis": key[1], "t": key[2], "class": key[3], "mean": float(np.mean(list(v.values()))), "n_swaps": len(v),
             "swaps": " ".join(sorted(v)), "counts": len(v) >= min_swaps, "near_duplicate": near[key]}
            for key, v in sorted(cells.items())]


def counted(crv: list[dict], label: str, axis: str, cls: str) -> dict[float, tuple[float, bool]]:
    """t -> (mean, near_duplicate) of the counted points of one agent, axis and class."""
    return {r["t"]: (r["mean"], r["near_duplicate"]) for r in crv if r["label"] == label and r["axis"] == axis and r["class"] == cls and r["counts"]}


def verdict(pts: dict[float, tuple[float, bool]], axis: str) -> dict:
    """(38)'s reading rule on one agent's counted Delta points of one axis: the plateau (the mean of the PLATEAU_POINTS
    furthest counted points, all at t >= FAR_T)."""
    far = {t: v for t, (v, _) in pts.items() if t >= FAR_T}
    plateau_t = sorted(far)[-PLATEAU_POINTS:]
    clean_t = sorted(t for t in far if not pts[t][1])[-PLATEAU_POINTS:]
    out = {"points": {str(t): round(v, 4) for t, (v, _) in sorted(pts.items())}, "n_far_points": len(far),
           "plateau": float(np.mean([far[t] for t in plateau_t])) if far else None, "plateau_t": plateau_t,
           "plateau_without_near_duplicates": float(np.mean([far[t] for t in clean_t])) if clean_t else None,
           "plateau_without_near_duplicates_t": clean_t}
    anchored = axis == "arc_inc" or 0.0 in pts
    out["delta_0"] = pts[0.0][0] if 0.0 in pts and axis != "arc_inc" else None
    out["counted"] = anchored and len(far) >= MIN_FAR_POINTS and (axis == "arc_inc" or pts[0.0][0] <= ABOUT_MINUS_2)
    if not out["counted"]:
        out["verdict"] = "not counted"
        if anchored and axis != "arc_inc" and pts[0.0][0] > ABOUT_MINUS_2:
            out["note"] = "Delta(0) not about -2"
        return out
    out["verdict"] = classify(out["plateau"])
    clean = out["plateau_without_near_duplicates"]
    out["verdict_without_near_duplicates"] = classify(clean) if len(out["plateau_without_near_duplicates_t"]) >= MIN_FAR_POINTS else "not counted"
    return out


def classify(plateau: float) -> str:
    return "colour" if plateau > ABOUT_0 else "arrangement" if plateau <= ABOUT_MINUS_2 else "both"


def witness(pts: dict[float, tuple[float, bool]]) -> dict:
    departs = sorted(t for t, (v, _) in pts.items() if v <= ABOUT_0)
    return {"points": {str(t): round(v, 4) for t, (v, _) in sorted(pts.items())},
            "reading": "no counted point" if not pts else "about 0" if not departs else "departs", "departs_at": departs}


def superseded(labels: list[str]) -> list[str]:
    """The labels whose agent has a superseding measurement among labels (SUPERSEDES, (39))."""
    out = []
    for x in labels:
        condition, agent, k = x.split("_")
        for new, old in SUPERSEDES.items():
            if agent == old and f"{condition}_{new}_{k}" in labels:
                out.append(x)
    return out


def conclusion(readings: dict[str, dict], key: str = "verdict", exclude: tuple[str, ...] = ()) -> str:
    counted_ = [r[key] for x, r in readings.items() if x not in exclude and r["counted"] and r.get(key, "not counted") != "not counted"]
    if len(counted_) < 2:
        return "not readable"
    best = max(set(counted_), key=counted_.count)
    return best if counted_.count(best) >= 2 else "no majority"


def reading(crv: list[dict], labels: list[str]) -> dict:
    shuffle = [x for x in labels if x.startswith("shuffle_")]
    out = {"rule": {"min_swaps": MIN_SWAPS, "min_far_points": MIN_FAR_POINTS, "far_t": FAR_T, "plateau_points": PLATEAU_POINTS,
                    "about_minus_2": ABOUT_MINUS_2, "about_0": ABOUT_0, "primary_axes": list(PRIMARY_AXES),
                    "secondary_axes": list(SECONDARY_AXES)}, "delta": {}, "witnesses": {}}
    old = tuple(superseded(shuffle))
    if old:  # (39): reported, out of the conclusions
        out["superseded"] = list(old)
    for axis in AXES:
        per = {x: verdict(counted(crv, x, axis, "delta"), axis) for x in shuffle}
        out["delta"][axis] = {"role": "primary" if axis in PRIMARY_AXES else "secondary", "agents": per, "conclusion": conclusion(per, exclude=old)}
        if axis in SECONDARY_AXES:
            out["delta"][axis]["conclusion_without_near_duplicates"] = conclusion(per, "verdict_without_near_duplicates", exclude=old)
        out["witnesses"][axis] = {x: {cls: witness(counted(crv, x, axis, cls)) for cls in ("old_ud", "neutral")} for x in shuffle}
    return out


def validity(rows: list[dict]) -> list[dict]:
    cells = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for key in ((r["label"], "all", "all", "all"), (r["label"], r["swap"], r["axis"], r["class"])):
            c = cells[key]
            c[0] += 1
            c[1] += truthy(r["valid"])
            c[2] += truthy(r["kept"])
    return [{"label": k[0], "swap": k[1], "axis": k[2], "class": k[3], "rows": v[0], "valid": v[1], "kept": v[2],
             "share_valid": v[1] / v[0], "share_kept": v[2] / v[0]} for k, v in sorted(cells.items())]


def label_of(meta: dict) -> str:
    condition = "shuffle" if meta["training"] == "shuffle" else meta["training"]
    ref = f"-ref{meta['reference_seed_label']}" if meta.get("reference_seed_label") else ""  # --name-reference, (39)
    return f"{condition}_{SEED_LABELS[int(meta['seed'])]}{ref}_k{int(meta['checkpoint_k']):06d}"


def shared_image(rows: list[dict], probe_dirs: list[Path]) -> list[dict]:
    """The swap to se at T_ctrl (segment k = 10: T_ctrl at nw and se) is also E4's probe "T_ctrl at the old UD slot". For
    every swap measurement with an E4-set measurement of the same label among probe_dirs (probes__<...>, measure_probes.py
    --probe-set e4): b at step 1 at nw in both (the E4 set's from curves.npz, its images.csv row matched by colours)."""
    swap = {r["label"]: r for r in rows if r["swap"] == "se" and r["axis"] == "segment" and int(r["k"]) == 10 and r["zone"] == "nw"}
    out = []
    for d in probe_dirs:
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        if meta.get("probe_set", "e4") != "e4":
            raise SystemExit(f"{d}: not an E4-set probe measurement")
        label = label_of(meta)
        if label not in swap:
            continue
        s = swap[label]
        colour = s["zone_color"]  # T_ctrl
        images = read_csv(d / "images.csv")
        curves_ = np.load(d / "curves.npz")
        match = [i for i, r in enumerate(images) if r["zone"] == "nw" and r["color_nw"] == colour and r["color_se"] == colour]
        if len(match) != 1:
            raise SystemExit(f"{d}: {len(match)} rows of the shared image at nw")
        i = match[0]
        b_e4 = float(curves_["curve_exp"][i, 0] - curves_["curve_ref"][i, 0])
        out.append({"label": label, "probe_dir": d.name, "b_step1_swap": float(s["b_step1"]), "b_step1_e4_set": b_e4,
                    "difference": float(s["b_step1"]) - b_e4, "valid_swap": truthy(s["valid"]), "kept_swap": truthy(s["kept"]),
                    "valid_e4_set": truthy(images[i].get("valid", "")), "n_walks_e4_set": int(images[i]["n_walks"])})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare-probes", type=Path, nargs="*", default=[],
                        help="E4-set probe directories (probes__<...>) of the same models and checkpoints: shared_image.csv")
    args = parser.parse_args(argv)
    rows = load(args.dirs)
    args.out.mkdir(parents=True, exist_ok=True)
    pts = points(rows)
    dls = deltas(pts)
    crv = curves(pts, dls)
    labels = sorted({r["label"] for r in rows})
    write_csv(args.out / "rows.csv", rows)
    write_csv(args.out / "validity.csv", validity(rows))
    write_csv(args.out / "points.csv", pts)
    write_csv(args.out / "delta.csv", dls)
    write_csv(args.out / "curves.csv", crv)
    rd = reading(crv, labels)
    (args.out / "reading.json").write_text(json.dumps(rd, indent=1), encoding="utf-8")
    for v in validity(rows):
        if v["swap"] == "all":
            print(f"{v['label']}: {v['valid']}/{v['rows']} valid, {v['kept']} kept", flush=True)
    for axis in PRIMARY_AXES + SECONDARY_AXES:
        d = rd["delta"][axis]
        plateaus = {x: (None if v["plateau"] is None else round(v["plateau"], 2), v["verdict"]) for x, v in d["agents"].items()}
        extra = f"; without t >= 0.8: {d['conclusion_without_near_duplicates']}" if "conclusion_without_near_duplicates" in d else ""
        print(f"{d['role']} {axis}: {d['conclusion']}{extra} {plateaus}", flush=True)
    if rd.get("superseded"):
        print(f"superseded (reported, out of the conclusions, (39)): {rd['superseded']}", flush=True)
    if args.compare_probes:
        shared = shared_image(rows, args.compare_probes)
        write_csv(args.out / "shared_image.csv", shared)
        for s in shared:
            print(f"shared image (T_ctrl at nw and se), {s['label']}: b swap {s['b_step1_swap']:+.3f}, E4 set {s['b_step1_e4_set']:+.3f} "
                  f"({s['probe_dir']}), difference {s['difference']:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
