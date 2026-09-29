"""b by zone from environment / reference pairs (docs/e3/DECISIONS.md,
2026-09-30): for every pair of finished runs with the same config and the
same seed, one in the experimental environment and one in its reference,

    b(zone) = I(experimental run, zone) - I(reference run, zone),

with I_s (the main measure: controllability of the decoded s-space kernel,
real contexts) and I_z (secondary, an upper bound). The standard error of a
pair's b combines the two runs' zone se's; a class mean (block class A or B,
e3_env.block_class_for_seed) averages its seeds' b with se
sqrt(sum se^2) / n (the seeds are the replicates; their spread is also
reported). Each run's calibration criteria are listed next to it (C-a:
replay template accuracy >= 0.99; C-b: per zone, over pairs with
N[s,a] >= 100, L1(WM, P) <= L1(B, P) + 0.05, with the L1 floor of the
kernel's draws; C-c: z-only probe >= 0.99), so the b of a configuration that
fails them can be read as such.

Inputs: run directories, or directories holding a recomputation
(recomputed/<label>/), each with results.json and arrays.npz. Usage, from the
repo root:
    python experiments/E3_worldmodel_4zones/aggregate_b.py results/E3_worldmodel_4zones/*gpu_e30b* [--out <dir>]
Output: B_BY_ZONE.md and b_by_zone.json in --out (default: printed only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "aggregate_b_env")

CATEGORIES = tuple(e3_env.ZONE_CATEGORIES)
C_A, C_C, C_B_MARGIN, C_B_MIN_COUNT = 0.99, 0.99, 0.05, 100
MEASURES = ("I_s", "I_z")


def config_of(run_dir: str) -> str:
    """The config name from a run directory's name (runs.make_run_name: ...__worldmodel-<config>__seedNN__...)."""
    match = re.search(r"__worldmodel-(.+?)__seed\d+__", Path(run_dir).name)
    if not match:
        raise ValueError(f"not an E3 run directory name: {run_dir}")
    return match.group(1)


def calibration(results: dict, arrays) -> dict:
    """C-a, C-b (per zone) and C-c of one run (docs/e3/DECISIONS.md, 2026-09-25 and 2026-09-29)."""
    zone = np.array(results["zone_of_state"])
    N_sa = arrays["N"].sum(axis=-1)
    l1_wm = np.abs(arrays["real_p_hat"] - arrays["P_true"]).sum(axis=-1)
    l1_b = np.abs(arrays["T_hat_B"] - arrays["P_true"]).sum(axis=-1)
    draws = results["real_context"].get("kernel_draws")
    floor = results["real_context"].get("l1_floor_by_zone", {})
    c_b = {}
    for cat in CATEGORIES:
        mask = (zone == cat)[:, None] & (N_sa >= C_B_MIN_COUNT) & np.isfinite(l1_wm)
        wm, b = (float(l1_wm[mask].mean()), float(l1_b[mask].mean())) if mask.any() else (float("nan"), float("nan"))
        c_b[cat] = {"l1_wm": wm, "l1_b": b, "floor": floor.get(cat), "n_pairs": int(mask.sum()),
                    "ok": bool(mask.any() and wm <= b + C_B_MARGIN)}
    acc, z_only = results["accuracies"]["acc_replay_hard"], results["z_only_probe"]["test_accuracy"]
    return {"c_a": {"value": acc, "ok": bool(acc >= C_A)}, "c_c": {"value": z_only, "ok": bool(z_only >= C_C)},
            "c_b": c_b, "kernel_draws": draws, "all_ok": bool(acc >= C_A and z_only >= C_C and all(v["ok"] for v in c_b.values()))}


def load_run(path: Path) -> dict:
    results = json.loads((path / "results.json").read_text(encoding="utf-8"))
    arrays = np.load(path / "arrays.npz")
    rc = results["real_context"]
    run_dir = results["run_dir"] if path.parent.name == "recomputed" else str(path)  # a recomputation names its run
    return {
        "path": str(path), "config": config_of(run_dir),
        "seed": int(results["seed"]), "condition": results.get("condition", "experimental"),
        "zone_class": e3_env.block_class_for_seed(int(results["seed"])),
        "zone_kinds": results.get("zone_kinds"),
        **{m: {cat: {"mean": rc[f"{m}_by_zone"][cat]["mean"], "se": rc[f"{m}_by_zone"][cat]["se"]} for cat in CATEGORIES}
           for m in MEASURES if f"{m}_by_zone" in rc},
        "calibration": calibration(results, arrays),
    }


def _nan(x):
    return float("nan") if x is None else float(x)


def pair_b(exp: dict, ref: dict, measure: str) -> dict[str, dict]:
    out = {}
    for cat in CATEGORIES:
        e, r = exp[measure][cat], ref[measure][cat]
        out[cat] = {"b": _nan(e["mean"]) - _nan(r["mean"]), "se": float(np.hypot(_nan(e["se"]), _nan(r["se"]))),
                    "exp": _nan(e["mean"]), "ref": _nan(r["mean"])}
    return out


def aggregate(runs: list[dict]) -> dict:
    """Pairs (config, seed) -> b per zone for each measure; class means per config."""
    by_key: dict[tuple, dict] = {}
    for run in runs:
        slot = by_key.setdefault((run["config"], run["seed"]), {})
        if run["condition"] in slot:
            raise ValueError(f"two {run['condition']} runs for config {run['config']} seed {run['seed']}: "
                             f"{slot[run['condition']]['path']} and {run['path']}")
        slot[run["condition"]] = run
    pairs, unpaired = [], []
    for (config, seed), slot in sorted(by_key.items()):
        if set(slot) != {"experimental", "reference"}:
            unpaired += [run["path"] for run in slot.values()]
            continue
        exp, ref = slot["experimental"], slot["reference"]
        pairs.append({"config": config, "seed": seed, "zone_class": exp["zone_class"],
                      "calibration": {"experimental": exp["calibration"], "reference": ref["calibration"]},
                      **{m: pair_b(exp, ref, m) for m in MEASURES if m in exp and m in ref}})
    classes = {}
    for pair in pairs:
        classes.setdefault((pair["config"], pair["zone_class"]), []).append(pair)
    class_means = {}
    for (config, cls), members in sorted(classes.items()):
        entry = {"config": config, "zone_class": cls, "seeds": [p["seed"] for p in members]}
        for m in MEASURES:
            if not all(m in p for p in members):
                continue
            entry[m] = {}
            for cat in CATEGORIES:
                b = np.array([p[m][cat]["b"] for p in members])
                se = np.array([p[m][cat]["se"] for p in members])
                entry[m][cat] = {"b": float(b.mean()), "se": float(np.sqrt((se ** 2).sum()) / len(b)),
                                 "sd_over_seeds": float(b.std(ddof=1)) if len(b) > 1 else float("nan"), "n_seeds": len(b)}
        entry["all_calibrated"] = all(p["calibration"][c]["all_ok"] for p in members for c in ("experimental", "reference"))
        class_means[f"{config}/{cls}"] = entry
    return {"pairs": pairs, "class_means": class_means, "unpaired": unpaired}


def _f(x: float) -> str:
    return "NaN" if not np.isfinite(x) else f"{x:+.4f}"


def markdown(agg: dict) -> str:
    lines = ["# b by zone: experimental minus reference (docs/e3/DECISIONS.md, 2026-09-30)", "",
             "b = I(experimental) - I(reference), same config and seed. **I_s is the main measure**, I_z secondary (upper "
             "bound). Zones are named by their appearance category; their dynamics in each environment are listed in the "
             "runs' reports. Class means: mean over the class's seeds, se = sqrt(sum se^2) / n.", ""]
    head = "| " + " | ".join(CATEGORIES) + " |"
    rule = "|---" * (len(CATEGORIES) + 3) + "|"
    for m in MEASURES:
        lines += [f"## {m} ({'MAIN' if m == 'I_s' else 'secondary'}): class means", "",
                  f"| config / class | seeds | all runs calibrated (C-a, C-b, C-c) {head}", rule]
        for key, entry in agg["class_means"].items():
            if m in entry:
                cells = " | ".join(f"{_f(entry[m][c]['b'])} [{entry[m][c]['se']:.4f}]" for c in CATEGORIES)
                lines.append(f"| {key} | {entry['seeds']} | {'yes' if entry['all_calibrated'] else 'no'} | {cells} |")
        lines += ["", f"## {m} ({'MAIN' if m == 'I_s' else 'secondary'}): per pair", "",
                  f"| config | class | seed {head}", "|---" * (len(CATEGORIES) + 3) + "|"]
        for p in agg["pairs"]:
            if m in p:
                cells = " | ".join(f"{_f(p[m][c]['b'])} [{p[m][c]['se']:.4f}] ({p[m][c]['exp']:.3f} vs {p[m][c]['ref']:.3f})"
                                   for c in CATEGORIES)
                lines.append(f"| {p['config']} | {p['zone_class']} | {p['seed']} | {cells} |")
        lines.append("")
    lines += ["## Calibration of each run", "", "| config | seed | condition | C-a | C-c | C-b by zone (L1_WM vs L1_B + 0.05; floor) |",
              "|---|---|---|---|---|---|"]
    for p in agg["pairs"]:
        for cond, cal in p["calibration"].items():
            cb = "; ".join(f"{c[:16]} {v['l1_wm']:.3f} vs {v['l1_b'] + C_B_MARGIN:.3f} ({_nan(v['floor']):.3f}) "
                           f"{'ok' if v['ok'] else 'FAIL'}" for c, v in cal["c_b"].items())
            lines.append(f"| {p['config']} | {p['seed']} | {cond} | {cal['c_a']['value']:.4f} {'ok' if cal['c_a']['ok'] else 'FAIL'} | "
                         f"{cal['c_c']['value']:.4f} {'ok' if cal['c_c']['ok'] else 'FAIL'} | {cb} |")
    if agg["unpaired"]:
        lines += ["", "Runs without their pair (left out): " + ", ".join(f"`{Path(u).name}`" for u in agg["unpaired"])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="run (or recomputation) directories with results.json and arrays.npz")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    paths = [p for p in args.runs if (p / "results.json").exists() and (p / "arrays.npz").exists()]
    skipped = [p for p in args.runs if p not in paths]
    agg = aggregate([load_run(p) for p in paths])
    agg["skipped_without_results"] = [str(p) for p in skipped]
    text = markdown(agg) + ("".join(f"\nNo results.json / arrays.npz (not finished?): `{p.name}`" for p in skipped) + "\n"
                            if skipped else "")
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "B_BY_ZONE.md").write_text(text, encoding="utf-8")
        (args.out / "b_by_zone.json").write_text(json.dumps(agg, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
