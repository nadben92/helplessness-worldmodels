"""b by zone from environment / reference pairs (docs/e3/DECISIONS.md,
2026-09-30): for every pair of finished runs with the same config and the
same seed, one in the experimental environment and one in its reference,

    b(zone) = I(experimental run, zone) - I(reference run, zone),

with I_s (the main measure: controllability of the decoded s-space kernel,
real contexts, computed on the kernel as it is), I_s projected on the
support (secondary, docs/e3/DECISIONS.md 2026-09-30 (5): for runs that have it)
and I_z (secondary, an upper bound). The off-support mass of each run's
kernel (its share on states that are not reachable successors) is listed by
zone next to its calibration.

BIAS CONTROL (docs/e3/DECISIONS.md, 2026-10-02 (16); replaces the rules of
2026-09-30 (7) and (9)). For every pair and zone: b on I_s and on I_s
projected, their difference, and the difference of off-support mass between
the two environments (experimental - reference), reported as they are.
  * Aberration control (E3-0's ranges, configs/e3/e30_aberration_control.yaml,
    --e30-control): a value outside them -- in CN / CD
    |b(I_s) - b(I_s projected)| > tau_b or |off-support difference| > tau_off,
    in UN / UD the environment-side off-support mass outside its interval -- is
    FLAGGED, to be explained before interpretation; never excluded.
  * Campaign rule (campaign_rule; N, FRACTION and REFERENCE EFFECT fixed by the
    user before the campaign, --n-seeds --fraction --reference-effect): per
    controllable zone, over the N seeds (N even, classes A and B in equal
    numbers), the mean of b(I_s) - b(I_s projected) and of the off-support
    difference with their 95 % Student-t intervals; b is interpretable in the
    zone if both intervals lie within +- FRACTION x REFERENCE EFFECT. Unset
    parameters, or two seeds (the pilot): not applied, the differences are
    reported as a description.
The standard error of a
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

from helplessness.envs.zones import geographic_support_mask
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import off_support_mass

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "aggregate_b_env")

CATEGORIES = tuple(e3_env.ZONE_CATEGORIES)
C_A, C_C, C_B_MARGIN, C_B_MIN_COUNT = 0.99, 0.99, 0.05, 100
MEASURES = ("I_s", "I_s_projected", "I_z")


def is_controllable(category: str) -> bool:
    """Zone categories are named <dynamics kind>_...: "uncontrollable_*" vs "controllable_*"."""
    return not category.startswith("uncontrollable")


def zone_value(x: float | dict, category: str) -> float:
    """A threshold given as one number for every zone, or as {zone: number}."""
    return float(x[category] if isinstance(x, dict) else x)


def thresholds_from_pairs(pairs: list[dict]) -> dict:
    """DECISIONS 2026-09-30 (9), fixed before the values were seen. For each controllable zone, over the E3-0
    pairs (the tested quantity itself, one value per pair): tau_b = 2 x sd of b(I_s) - b(I_s projected),
    tau_off = 2 x sd of off-support(env) - off-support(ref) (sample sd, ddof = 1; needs >= 2 pairs). For each
    uncontrollable zone: tau_off_floor = 2 x sd, over the E3-0 environment runs, of the environment-side
    off-support mass (the quantity that the floor interval bounds). Also returns the values it came from."""
    out = {"tau_b": {}, "tau_off": {}, "tau_off_floor": {}, "values": {}, "n_pairs": len(pairs)}
    for cat in CATEGORIES:
        bc = [p["bias_control"][cat] for p in pairs]
        if is_controllable(cat):
            diff = np.array([v["b_difference"] for v in bc], dtype=float)
            off = np.array([v["off_support_difference"] for v in bc], dtype=float)
            out["values"][cat] = {"b_difference": diff.tolist(), "off_support_difference": off.tolist()}
            out["tau_b"][cat] = 2.0 * float(np.std(diff, ddof=1)) if np.isfinite(diff).sum() > 1 else float("nan")
            out["tau_off"][cat] = 2.0 * float(np.std(off, ddof=1)) if len(off) > 1 else float("nan")
        else:
            env = np.array([v["off_support_experimental"] for v in bc], dtype=float)
            out["values"][cat] = {"off_support_experimental": env.tolist()}
            out["tau_off_floor"][cat] = 2.0 * float(np.std(env, ddof=1)) if len(env) > 1 else float("nan")
    return out


def floor_from_pairs(pairs: list[dict], tau_off_floor: dict) -> dict:
    """The E3-0 floor of the uncontrollable zones: b(I_s) of the E3-0 pairs (mean and values) and the interval of the
    environment-side off-support mass (min, max over the E3-0 environment runs)."""
    floor = {"n_pairs": len(pairs), "tau_off_floor": tau_off_floor, "zones": {}}
    for cat in CATEGORIES:
        if is_controllable(cat):
            continue
        b = [p["bias_control"][cat]["b_I_s"] for p in pairs]
        off = [p["bias_control"][cat]["off_support_experimental"] for p in pairs]
        floor["zones"][cat] = {"b_I_s": float(np.mean(b)), "b_I_s_values": b, "off_support_min": float(min(off)),
                               "off_support_max": float(max(off))}
    return floor
LABEL = {"I_s": "I_s (MAIN)", "I_s_projected": "I_s projected on the support (secondary)", "I_z": "I_z (secondary)"}
GRID_SHAPE = (12, 12)
E30_CONTROL = Path(__file__).resolve().parents[2] / "configs" / "e3" / "e30_aberration_control.yaml"


def load_e30_control(path: Path = E30_CONTROL) -> dict:
    """E3-0's ranges for the aberration control (DECISIONS 2026-10-02 (16)): tau_b, tau_off (CN, CD),
    off_support_interval (UN, UD)."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def aberrations(v: dict, cat: str, control: dict) -> list[str]:
    """The aberration flags of one pair and zone against E3-0's ranges (empty: within them)."""
    flags = []
    if is_controllable(cat):
        if np.isfinite(v["b_difference"]) and abs(v["b_difference"]) > control["tau_b"][cat]:
            flags.append(f"|b(I_s) - b(I_s projected)| = {abs(v['b_difference']):.4f} > {control['tau_b'][cat]:.4f}")
        if abs(v["off_support_difference"]) > control["tau_off"][cat]:
            flags.append(f"|off-support env - ref| = {abs(v['off_support_difference']):.4f} > {control['tau_off'][cat]:.4f}")
    else:
        lo, hi = control["off_support_interval"][cat]
        if not lo <= v["off_support_experimental"] <= hi:
            flags.append(f"off-support (env) {v['off_support_experimental']:.4f} outside [{lo:.4f}, {hi:.4f}]")
    return flags


def mean_ci95(x: np.ndarray) -> dict:
    """Mean and 95 % Student-t interval over the seeds (n - 1 degrees of freedom)."""
    from scipy.stats import t

    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return {"n": n, "mean": m, "ci95": [float("nan"), float("nan")]}
    half = float(t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return {"n": n, "mean": m, "ci95": [m - half, m + half]}


def campaign_rule(pairs: list[dict], n_seeds: int | None, fraction: float | None, reference_effect: float | None) -> dict:
    """DECISIONS 2026-10-02 (16), 3. Per controllable zone, over the campaign's pairs: the mean and 95 % interval of
    b(I_s) - b(I_s projected) and of off-support(env) - off-support(ref); interpretable if both intervals lie within
    +- fraction x |reference_effect|. Not applied (descriptive) when a parameter is unset or with two seeds (the
    pilot). The campaign must have exactly n_seeds pairs, n_seeds even, block classes A and B in equal numbers."""
    unset = [k for k, v in (("n_seeds", n_seeds), ("fraction", fraction), ("reference_effect", reference_effect)) if v is None]
    classes = [p["zone_class"] for p in pairs]
    out = {"n_pairs": len(pairs), "classes": {c: classes.count(c) for c in sorted(set(classes))}, "zones": {}}
    if unset:
        out["status"] = f"not applied: {', '.join(unset)} not fixed (set by the user before the campaign); descriptive only"
    elif len(pairs) <= 2:
        out["status"] = "not applied: two seeds (pilot); descriptive only"
    else:
        if len(pairs) != n_seeds or n_seeds % 2 or len(set(out["classes"].values())) != 1 or len(out["classes"]) != 2:
            raise ValueError(f"campaign rule: {len(pairs)} pairs with classes {out['classes']}, expected N = {n_seeds} (even), "
                             "classes A and B in equal numbers")
        out["status"] = "applied"
        out["bound"] = fraction * abs(reference_effect)
    for cat in CATEGORIES:
        if not is_controllable(cat):
            continue
        b = mean_ci95([p["bias_control"][cat]["b_difference"] for p in pairs])
        off = mean_ci95([p["bias_control"][cat]["off_support_difference"] for p in pairs])
        z = {"b_difference": b, "off_support_difference": off}
        if out["status"] == "applied":
            inside = [bool(-out["bound"] <= q["ci95"][0] and q["ci95"][1] <= out["bound"]) for q in (b, off)]
            z["verdict"] = "interpretable" if all(inside) else "NOT interpretable"
        out["zones"][cat] = z
    return out


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
    off = off_support_mass(arrays["real_p_hat"], geographic_support_mask(GRID_SHAPE))  # (S, A), from the saved kernel
    off_by_zone = {cat: float(np.nanmean(off[zone == cat])) if np.isfinite(off[zone == cat]).any() else float("nan") for cat in CATEGORIES}
    return {"off_support_mass": off_by_zone, "c_a": {"value": acc, "ok": bool(acc >= C_A)}, "c_c": {"value": z_only, "ok": bool(z_only >= C_C)},
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


def aggregate(runs: list[dict], e30_control: dict | None = None, campaign: dict | None = None) -> dict:
    """Pairs (config, seed) -> b per zone for each measure; class means per config; for every pair and zone the
    bias-control quantities, with E3-0's aberration flags when e30_control is given (load_e30_control); the
    campaign rule when campaign = {n_seeds, fraction, reference_effect} is given (values may be None: descriptive)."""
    by_key: dict[tuple, dict] = {}
    for run in runs:
        slot = by_key.setdefault((run["config"], run["seed"]), {})
        if run["condition"] in slot:
            raise ValueError(f"two {run['condition']} runs for config {run['config']} seed {run['seed']}: "
                             f"{slot[run['condition']]['path']} and {run['path']}")
        slot[run["condition"]] = run
    pairs, unpaired = [], []

    def bias_control(exp: dict, ref: dict) -> dict:
        out = {}
        for cat in CATEGORIES:
            off_e, off_r = exp["calibration"]["off_support_mass"][cat], ref["calibration"]["off_support_mass"][cat]
            b_s = _nan(exp["I_s"][cat]["mean"]) - _nan(ref["I_s"][cat]["mean"]) if "I_s" in exp and "I_s" in ref else float("nan")
            has_proj = "I_s_projected" in exp and "I_s_projected" in ref
            b_p = _nan(exp["I_s_projected"][cat]["mean"]) - _nan(ref["I_s_projected"][cat]["mean"]) if has_proj else float("nan")
            v = {"b_I_s": b_s, "b_I_s_projected": b_p, "b_difference": b_s - b_p if has_proj else float("nan"),
                 "off_support_experimental": off_e, "off_support_reference": off_r, "off_support_difference": off_e - off_r}
            if e30_control is not None:
                v["aberrations"] = aberrations(v, cat, e30_control)
            out[cat] = v
        return out

    for (config, seed), slot in sorted(by_key.items()):
        if set(slot) != {"experimental", "reference"}:
            unpaired += [run["path"] for run in slot.values()]
            continue
        exp, ref = slot["experimental"], slot["reference"]
        pairs.append({"config": config, "seed": seed, "zone_class": exp["zone_class"],
                      "calibration": {"experimental": exp["calibration"], "reference": ref["calibration"]},
                      "bias_control": bias_control(exp, ref),
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
    result = {"pairs": pairs, "class_means": class_means, "unpaired": unpaired, "e30_control": e30_control}
    if campaign is not None:
        result["campaign_rule"] = campaign_rule(pairs, campaign.get("n_seeds"), campaign.get("fraction"), campaign.get("reference_effect"))
    return result


def _f(x: float) -> str:
    return "NaN" if not np.isfinite(x) else f"{x:+.4f}"


def markdown(agg: dict) -> str:
    lines = ["# b by zone: experimental minus reference (docs/e3/DECISIONS.md, 2026-09-30)", "",
             "b = I(experimental) - I(reference), same config and seed. **I_s is the main measure** (on the kernel as it "
             "is); I_s projected on the support and I_z (upper bound) are secondary. Zones are named by their appearance category; their dynamics in each environment are listed in the "
             "runs' reports. Class means: mean over the class's seeds, se = sqrt(sum se^2) / n.", ""]
    head = "| " + " | ".join(CATEGORIES) + " |"
    rule = "|---" * (len(CATEGORIES) + 3) + "|"
    for m in MEASURES:
        if not any(m in p for p in agg["pairs"]):
            continue
        lines += [f"## {LABEL[m]}: class means", "",
                  f"| config / class | seeds | all runs calibrated (C-a, C-b, C-c) {head}", rule]
        for key, entry in agg["class_means"].items():
            if m in entry:
                cells = " | ".join(f"{_f(entry[m][c]['b'])} [{entry[m][c]['se']:.4f}]" for c in CATEGORIES)
                lines.append(f"| {key} | {entry['seeds']} | {'yes' if entry['all_calibrated'] else 'no'} | {cells} |")
        lines += ["", f"## {LABEL[m]}: per pair", "",
                  f"| config | class | seed {head}", "|---" * (len(CATEGORIES) + 3) + "|"]
        for p in agg["pairs"]:
            if m in p:
                cells = " | ".join(f"{_f(p[m][c]['b'])} [{p[m][c]['se']:.4f}] ({p[m][c]['exp']:.3f} vs {p[m][c]['ref']:.3f})"
                                   for c in CATEGORIES)
                lines.append(f"| {p['config']} | {p['zone_class']} | {p['seed']} | {cells} |")
        lines.append("")
    lines += ["## Bias control (DECISIONS 2026-10-02 (16))", "",
              "Per pair and zone, as measured. Aberration flags: values outside E3-0's ranges (configs/e3/e30_aberration_control.yaml) "
              "-- to be explained before interpretation, never excluded." if agg.get("e30_control") else
              "Per pair and zone, as measured (no E3-0 aberration control given).", "",
              "| config | seed | zone | b (I_s) | b (I_s projected) | difference | off-support env / ref (difference) | aberration flags |",
              "|---|---|---|---|---|---|---|---|"]
    for p in agg["pairs"]:
        for c, v in p["bias_control"].items():
            flags = "; ".join(v["aberrations"]) if v.get("aberrations") else ("none" if "aberrations" in v else "-")
            lines.append(f"| {p['config']} | {p['seed']} | {c} | {_f(v['b_I_s'])} | {_f(v['b_I_s_projected'])} | {_f(v['b_difference'])} | "
                         f"{v['off_support_experimental']:.3f} / {v['off_support_reference']:.3f} ({v['off_support_difference']:+.3f}) | {flags} |")
    lines.append("")
    if "campaign_rule" in agg:
        cr = agg["campaign_rule"]
        lines += ["## Campaign rule (DECISIONS 2026-10-02 (16), 3)", "", f"Status: {cr['status']}. Pairs: {cr['n_pairs']}, classes {cr['classes']}"
                  + (f"; bound +- {cr['bound']:.4f}" if "bound" in cr else "") + ".", "",
                  "| zone | b(I_s) - b(I_s projected): mean [95 % CI] | off-support env - ref: mean [95 % CI] | verdict |", "|---|---|---|---|"]
        for c, z in cr["zones"].items():
            cells = [f"{q['mean']:+.4f} [{q['ci95'][0]:+.4f}, {q['ci95'][1]:+.4f}]" for q in (z["b_difference"], z["off_support_difference"])]
            lines.append(f"| {c} | {cells[0]} | {cells[1]} | {z.get('verdict', 'descriptive')} |")
        lines.append("")
    lines += ["## Calibration and off-support mass of each run", "",
              "| config | seed | condition | C-a | C-c | C-b by zone (L1_WM vs L1_B + 0.05; floor) | off-support mass by zone |",
              "|---|---|---|---|---|---|---|"]
    for p in agg["pairs"]:
        for cond, cal in p["calibration"].items():
            cb = "; ".join(f"{c[:16]} {v['l1_wm']:.3f} vs {v['l1_b'] + C_B_MARGIN:.3f} ({_nan(v['floor']):.3f}) "
                           f"{'ok' if v['ok'] else 'FAIL'}" for c, v in cal["c_b"].items())
            lines.append(f"| {p['config']} | {p['seed']} | {cond} | {cal['c_a']['value']:.4f} {'ok' if cal['c_a']['ok'] else 'FAIL'} | "
                         f"{cal['c_c']['value']:.4f} {'ok' if cal['c_c']['ok'] else 'FAIL'} | {cb} | "
                         + "; ".join(f"{c[:16]} {v:.3f}" for c, v in cal["off_support_mass"].items()) + " |")
    if agg["unpaired"]:
        lines += ["", "Runs without their pair (left out): " + ", ".join(f"`{Path(u).name}`" for u in agg["unpaired"])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="run (or recomputation) directories with results.json and arrays.npz")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--e30-control", type=Path, default=E30_CONTROL,
                        help="E3-0's ranges for the aberration control (default configs/e3/e30_aberration_control.yaml)")
    parser.add_argument("--no-e30-control", action="store_true", help="report without E3-0's aberration flags")
    parser.add_argument("--n-seeds", type=int, default=None, help="campaign rule: N (set by the user before the campaign)")
    parser.add_argument("--fraction", type=float, default=None, help="campaign rule: FRACTION (set by the user before the campaign)")
    parser.add_argument("--reference-effect", type=float, default=None, help="campaign rule: REFERENCE EFFECT, bits (set by the user)")
    args = parser.parse_args(argv)
    paths = [p for p in args.runs if (p / "results.json").exists() and (p / "arrays.npz").exists()]
    skipped = [p for p in args.runs if p not in paths]
    control = None if args.no_e30_control else load_e30_control(args.e30_control)
    agg = aggregate([load_run(p) for p in paths], control,
                    {"n_seeds": args.n_seeds, "fraction": args.fraction, "reference_effect": args.reference_effect})
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
