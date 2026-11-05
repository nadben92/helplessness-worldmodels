"""EXPLORATORY (not planned by docs/e5/DECISIONS.md (38); REPORT.md section 10, whose reading was written first): the start
of the colour gradient, where the agents stay localised. Does b(m) change when only the colour of the swapped slot m
changes, the arrangement staying fixed? Kept zones only (valid and localised >= 0.90, unchanged); nothing relaxed.

Interval: a bootstrap over the walks. The walks of a slot are the same on every image (E4's fixed walks), so one resample
of a slot's walks is applied at every t: the differences t_last - 0 are paired. Each resample recomputes the measure's own
estimator at step 1 (the mean over states of each state's mean, fixed_trajectory.summarize_zone), exposed and reference
separately, from steps.npz (measure_probes.py --save-steps).

Portion (REPORT.md section 10): per agent and axis, the swaps where slot m is kept at t = 0; t_last = the largest t such
that at least NEED of them (2, or 1 when the agent has a single one) keep m at every t' <= t; Delta b(m) = pooled b(m) at
t_last - pooled b(m) at 0 over those same swaps. b(m) - neutral likewise, a swap also needing a neutral slot kept at
both ends (the neutral slots kept at both ends, averaged).

Tables (--out):
  gradient_points.csv  per label, axis, t, class (moved, neutral, old_ud, delta): the mean over the swaps KEPT at that t
                       and its bootstrap SE, the number of swaps; the mean over the swaps NOT kept (drawn hollow, never
                       removed); the localisation of slot m (mean and minimum over the three swaps) and of the whole image
  gradient_start.json  per label and axis: t_last, the swaps, Delta b(m) and Delta (b(m) - neutral) with their 95 %
                       percentile intervals, the witnesses' Delta on the same portion (neutral slots, nw), the first t
                       where slot m's localisation falls below 0.90 (min over the portion's swaps; also over all three),
                       and the reading (REPORT.md section 10's rule). The interval is the walks' sampling only: neither
                       the measurement draws (about 0.05 bit, shared_image.csv) nor the seed are in it

Usage: python experiments/E5_belief_training/analysis_swap/gradient_start.py <probes-swap dirs> --out <tables dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import analyse_swap as swap_tables

N_BOOT = 2000
BOOT_SEED = 20261103  # the bootstrap's own stream (no measurement shares it)
LOC_MIN = 0.90  # analysis_450k/LOCALISATION_CRITERION.md (the `kept` criterion), unchanged
MIN_RISE = 0.2  # REPORT.md section 10: a rise below 0.2 bit is read as "stays at about -2"
NEIGHBOURS = {"ne": ("sw",), "sw": ("ne",), "se": ("ne", "sw")}  # each swap's neutral slots ((38), without nw)
TS = [k / 10 for k in range(11)]


def estimator(values: np.ndarray, states: np.ndarray, walks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """summarize_zone's step-1 value under walk weights (B, n_walks): the mean over states of each state's weighted mean."""
    ok = np.isfinite(values)
    values, states, walks = values[ok], states[ok], walks[ok]
    uniq, inv = np.unique(states, return_inverse=True)
    onehot = np.zeros((values.size, uniq.size))
    onehot[np.arange(values.size), inv] = 1.0
    w = weights[:, walks]
    n = w @ onehot
    with np.errstate(invalid="ignore", divide="ignore"):
        per_state = (w * values) @ onehot / n
    return np.nanmean(np.where(n > 0, per_state, np.nan), axis=1)


def load_label(d: Path, rows: list[dict], rng: np.random.Generator) -> dict:
    """(axis, t, swap, zone) -> {"b": (B + 1,) b at step 1 (index 0: all walks), "kept", "loc"} for one measurement."""
    steps = np.load(d / "steps.npz")
    out, boot = {}, {}
    for r in rows:
        key = f"i{r['image']}_{r['zone']}"
        mask = steps[f"{key}_steps_since_entry"] == 1
        walks = steps[f"{key}_walk"][mask].astype(np.int64)
        n = int(walks.max()) + 1
        if r["zone"] not in boot:  # one resample per slot, shared by every image (paired across t)
            boot[r["zone"]] = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT)])
        states = steps[f"{key}_states"][mask]
        b = (estimator(steps[f"{key}_experimental_I"][mask].astype(np.float64), states, walks, boot[r["zone"]])
             - estimator(steps[f"{key}_reference_I"][mask].astype(np.float64), states, walks, boot[r["zone"]]))
        out[(r["axis"], round(float(r["t"]), 2), r["swap"], r["zone"])] = {
            "b": b, "b_step1": float(r["b_step1"]), "kept": swap_tables.truthy(r["kept"]), "loc": float(r["loc_exp_step1"]),
            "class": r["class"]}
    return out


def neutral_value(cells: dict, axis: str, t: float, swap: str, slots: tuple[str, ...]) -> np.ndarray:
    return np.mean([cells[(axis, t, swap, z)]["b"] for z in slots], axis=0)


def points(label: str, cells: dict) -> list[dict]:
    out = []
    for axis in swap_tables.AXES:
        for t in TS:
            per = {sw: {z: cells[(axis, t, sw, z)] for z in ("nw", "ne", "se", "sw")} for sw in NEIGHBOURS}
            moved = {sw: per[sw][sw] for sw in NEIGHBOURS}
            loc_m = [moved[sw]["loc"] for sw in NEIGHBOURS]
            loc_image = min(c["loc"] for sw in NEIGHBOURS for c in per[sw].values())
            series = {"moved": {sw: (moved[sw]["b"], moved[sw]["kept"]) for sw in NEIGHBOURS},
                      "old_ud": {sw: (per[sw]["nw"]["b"], per[sw]["nw"]["kept"]) for sw in ("ne", "sw")}}  # nw is CD in the swap to se
            neutral, delta = {}, {}
            for sw, slots in NEIGHBOURS.items():
                kept = tuple(z for z in slots if per[sw][z]["kept"])
                neutral[sw] = (neutral_value(cells, axis, t, sw, kept or slots), bool(kept))
                delta[sw] = (moved[sw]["b"] - neutral[sw][0], moved[sw]["kept"] and bool(kept))
            series["neutral"], series["delta"] = neutral, delta
            for cls, s in series.items():
                kept = [v for v, k in s.values() if k]
                lost = [v[0] for v, k in s.values() if not k]
                mean = np.mean(kept, axis=0) if kept else None
                out.append({"label": label, "axis": axis, "t": t, "class": cls, "n_kept_swaps": len(kept),
                            "mean_kept": float(mean[0]) if kept else "", "se_kept": float(np.std(mean[1:])) if kept else "",
                            "mean_not_kept": float(np.mean(lost)) if lost else "", "n_not_kept_swaps": len(lost),
                            "loc_m_mean": float(np.mean(loc_m)), "loc_m_min": float(min(loc_m)), "loc_image_min": loc_image})
    return out


def interval(diff: np.ndarray) -> dict:
    lo, hi = np.percentile(diff[1:], [2.5, 97.5])
    return {"estimate": float(diff[0]), "se": float(np.std(diff[1:])), "ci95": [float(lo), float(hi)]}


def portion(cells: dict, axis: str, with_neutral: bool) -> dict | None:
    """REPORT.md section 10's portion: t_last and the swaps, then the paired difference t_last - 0."""
    def ok(sw, t):
        if not cells[(axis, t, sw, sw)]["kept"]:
            return False
        return not with_neutral or any(cells[(axis, t, sw, z)]["kept"] for z in NEIGHBOURS[sw])

    start = [sw for sw in NEIGHBOURS if ok(sw, 0.0)]
    if not start:
        return None
    need = 2 if len(start) >= 2 else 1
    t_last, swaps = 0.0, start
    for t in TS[1:]:
        run = [sw for sw in start if all(ok(sw, u) for u in TS if u <= t)]
        if len(run) < need:
            break
        t_last, swaps = t, run
    if with_neutral:  # the neutral slots kept at both ends
        slots = {sw: tuple(z for z in NEIGHBOURS[sw] if cells[(axis, 0.0, sw, z)]["kept"] and cells[(axis, t_last, sw, z)]["kept"])
                 for sw in swaps}
        swaps = [sw for sw in swaps if slots[sw]]
        if not swaps:
            return None
        value = lambda t: np.mean([cells[(axis, t, sw, sw)]["b"] - neutral_value(cells, axis, t, sw, slots[sw]) for sw in swaps], axis=0)
    else:
        value = lambda t: np.mean([cells[(axis, t, sw, sw)]["b"] for sw in swaps], axis=0)
    return {"t_last": t_last, "swaps": swaps, "need": need, "at_0": float(value(0.0)[0]), "at_t_last": float(value(t_last)[0]),
            **interval(value(t_last) - value(0.0))}


def witnesses(cells: dict, axis: str, part: dict | None) -> dict:
    """The witnesses on b(m)'s portion (same swaps, same t_last, paired): the neutral slots (those kept at both ends) and
    the old UD slot nw (swaps to ne and sw only: in the swap to se nw is the CD slot), each where kept at both ends."""
    if part is None or part["t_last"] == 0.0:
        return {"neutral": None, "old_ud": None}
    t1 = part["t_last"]
    both = lambda sw, z: cells[(axis, 0.0, sw, z)]["kept"] and cells[(axis, t1, sw, z)]["kept"]
    out = {}
    for name, zones in (("neutral", NEIGHBOURS), ("old_ud", {"ne": ("nw",), "sw": ("nw",)})):
        use = {sw: tuple(z for z in zones.get(sw, ()) if both(sw, z)) for sw in part["swaps"]}
        use = {sw: z for sw, z in use.items() if z}
        if not use:
            out[name] = None
            continue
        value = lambda t: np.mean([neutral_value(cells, axis, t, sw, z) for sw, z in use.items()], axis=0)
        out[name] = {"swaps": sorted(use), "at_0": float(value(0.0)[0]), "at_t_last": float(value(t1)[0]), **interval(value(t1) - value(0.0))}
    return out


def loc_drop(cells: dict, axis: str, swaps) -> float | None:
    """The first t where slot m's localisation (minimum over the swaps given) falls below LOC_MIN."""
    for t in TS:
        if swaps and min(cells[(axis, t, sw, sw)]["loc"] for sw in swaps) < LOC_MIN:
            return t
    return None


def reading(moved: dict | None, delta: dict | None) -> str:
    """REPORT.md section 10's rule, written before the numbers."""
    if moved is None or moved["t_last"] == 0.0:
        return "no portion"
    if moved["ci95"][0] > 0 and moved["estimate"] >= MIN_RISE:
        if delta is not None and delta["ci95"][0] > 0:
            return "b(m) rises, slot-specific: reads the colour (at least in part)"
        return "b(m) rises, not slot-specific (image-wide shift)"
    return "b(m) stays: not conclusive"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(BOOT_SEED)
    pts, summary = [], {"exploratory": True, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "loc_min": LOC_MIN, "min_rise": MIN_RISE, "agents": {}}
    for d in args.dirs:
        rows = swap_tables.load([d])
        label = rows[0]["label"]
        cells = load_label(d, rows, rng)
        check = max(abs(c["b"][0] - c["b_step1"]) for c in cells.values())
        if check > 1e-4:  # steps.npz holds float32 values
            raise SystemExit(f"{d}: the estimator does not reproduce b_step1 (max |difference| {check})")
        pts += points(label, cells)
        summary["agents"][label] = {}
        for axis in swap_tables.AXES:
            moved, delta = portion(cells, axis, False), portion(cells, axis, True)
            summary["agents"][label][axis] = {
                "delta_b_moved": moved, "delta_b_moved_minus_neutral": delta, "witnesses": witnesses(cells, axis, moved),
                "loc_m_below_090_from_t": loc_drop(cells, axis, moved["swaps"] if moved else []),  # the portion's swaps
                "loc_m_below_090_from_t_any_swap": loc_drop(cells, axis, list(NEIGHBOURS)), "reading": reading(moved, delta)}
        print(label, {a: (v["delta_b_moved"] and (v["delta_b_moved"]["t_last"], round(v["delta_b_moved"]["estimate"], 3),
                                                   [round(x, 3) for x in v["delta_b_moved"]["ci95"]]), v["reading"])
                      for a, v in summary["agents"][label].items()}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    swap_tables.write_csv(args.out / "gradient_points.csv", pts)
    (args.out / "gradient_start.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
