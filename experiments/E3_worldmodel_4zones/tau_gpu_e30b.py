"""The reading rule's thresholds from the gpu_e30b re-measurement, and the rule's complete
verdict on gpu_e30b (docs/e3/DECISIONS.md, 2026-09-30 (7), (9) and 2026-10-01 (10)). One command,
once handoff/gpu_e30b_measures/ (the measurement batch's archives) is in the checkout:

    python experiments/E3_worldmodel_4zones/tau_gpu_e30b.py

What it does, in this order, without changing any definition:
1. Reads the four re-measured win64 runs (handoff/gpu_e30b_measures/<run>.recomputed.tgz) and
   their original measurements (handoff/gpu_e30b/<run>.tgz), in a temporary directory: nothing
   is extracted into results/ and nothing existing is written over.
2. Checks the batch: the four runs of submit_kubeflow.MEASURE_BATCHES["gpu_e30b_measures"], each
   measured from the expected weights (SHA-256), all four with the same code (source_sha256) on the
   same device and GPU, TF32 off. A failure stops the script: the batch exists so that both sides
   of each pair are measured on one platform with one version of the code.
3. Consistency: the re-measurement must find the stored values again (same GPU, same streams). By
   run and zone: I_s, I_z (real contexts) and the off-support mass, original vs re-measured, and the
   largest difference of the per-state arrays (real_I_s, real_I_z, real_p_hat). A difference above
   CONSISTENCY_TOL is flagged "DIFFERS -- to be reported" (the code of measurement or the platform
   differ); it is reported, never corrected here.
4. Thresholds, by the definitions fixed in (9), B (aggregate_b.thresholds_from_pairs) over the two
   win64 pairs (seeds 0 and 5): for CN and CD, tau_b = 2 x sd of b(I_s) - b(I_s projected), tau_off =
   2 x sd of off-support(env) - off-support(ref); for UN and UD, tau_off_floor (the implementer's
   reading of "+- TAU_OFF of the zone", to be confirmed) and the E3-0 floor (aggregate_b.floor_from_pairs).
5. The rule's verdict on every pair and zone (the rule of (9), e30_rule_verdict; since 2026-10-02 (16) E3-0's
   values are an aberration control only, and this verdict is the record of (15)).

Outputs, in --out (default reports/tau_gpu_e30b/, refused if it already holds files):
TAU.md (batch check, consistency, thresholds with the values they come from, verdict),
B_BY_ZONE.md (aggregate_b's full table), thresholds.json (aggregate_b's --thresholds format).
Exit code 1 when the batch check fails, or when a consistency difference is flagged (the thresholds
and the verdict are still written, marked as such).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module

agg = load_experiment_module("E3_worldmodel_4zones", "aggregate_b.py", "tau_gpu_e30b_aggregate")
cluster = load_experiment_module("E3_worldmodel_4zones", "submit_kubeflow.py", "tau_gpu_e30b_cluster")

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH = "gpu_e30b_measures"
DEFAULT_MEASURES = REPO_ROOT / "handoff" / BATCH
DEFAULT_ORIGINALS = REPO_ROOT / "handoff" / "gpu_e30b"
DEFAULT_OUT = REPO_ROOT / "reports" / "tau_gpu_e30b"
CONSISTENCY_TOL = 1e-6  # bits / probability mass: above this, the re-measurement does not reproduce the stored value
COMPARED = (("I_s", "real_I_s"), ("I_z", "real_I_z"))


def _safe_members(tar: tarfile.TarFile, wanted) -> list[tarfile.TarInfo]:
    members = [m for m in tar.getmembers() if wanted(m.name)]
    for m in members:
        if m.name.startswith("/") or ".." in Path(m.name).parts:
            raise ValueError(f"unsafe path in archive: {m.name}")
    return members


def extract(measures: Path, originals: Path, tmp: Path) -> dict[str, dict[str, Path]]:
    """{job: {"recomputed": dir with results.json + arrays.npz, "original": same}} for the batch's runs."""
    runs = cluster.MEASURE_BATCHES[BATCH]["runs"]
    out = {}
    for job, (name, _) in runs.items():
        rec_archive, org_archive = measures / f"{name}.recomputed.tgz", originals / f"{name}.tgz"
        for a in (rec_archive, org_archive):
            if not a.is_file():
                raise FileNotFoundError(f"{job}: {a} is missing")
        with tarfile.open(rec_archive) as tar:
            members = _safe_members(tar, lambda n: "/recomputed/final__" in n)
            tar.extractall(tmp / "recomputed", members=members)
        finals = sorted((tmp / "recomputed" / name / "recomputed").glob("final__*"))
        if not finals:
            raise FileNotFoundError(f"{job}: no final__* recomputation in {rec_archive}")
        with tarfile.open(org_archive) as tar:
            members = _safe_members(tar, lambda n: n.endswith(("/results.json", "/arrays.npz")) and "/recomputed/" not in n
                                    and n.count("/") == 3)
            tar.extractall(tmp / "original", members=members)
        out[job] = {"recomputed": finals[-1], "original": tmp / "original" / "results" / "E3_worldmodel_4zones" / name}
    return out


def batch_check(paths: dict[str, dict[str, Path]]) -> tuple[list[str], dict]:
    runs = cluster.MEASURE_BATCHES[BATCH]["runs"]
    problems, seen = [], {}
    for job, p in paths.items():
        res = json.loads((p["recomputed"] / "results.json").read_text(encoding="utf-8"))
        rep = res.get("reproducibility") or {}
        seen[job] = {"weights_sha256": res.get("weights_sha256"), "source_sha256": rep.get("source_sha256"),
                     "device": rep.get("device"), "gpu_name": rep.get("gpu_name"), "allow_tf32_matmul": rep.get("allow_tf32_matmul"),
                     "allow_tf32_cudnn": rep.get("allow_tf32_cudnn"), "kernel_draws": res["real_context"].get("kernel_draws")}
        if seen[job]["weights_sha256"] != runs[job][1]:
            problems.append(f"{job}: measured weights {seen[job]['weights_sha256']}, expected {runs[job][1]}")
        if seen[job]["allow_tf32_matmul"] or seen[job]["allow_tf32_cudnn"]:
            problems.append(f"{job}: measured with TF32 on")
    for key in ("source_sha256", "device", "gpu_name", "kernel_draws"):
        values = {v[key] for v in seen.values()}
        if len(values) != 1 or None in values:
            problems.append(f"the four runs differ in {key}: {values}")
    return problems, seen


def consistency(paths: dict[str, dict[str, Path]]) -> list[dict]:
    rows = []
    for job, p in paths.items():
        rec, org = agg.load_run(p["recomputed"]), agg.load_run(p["original"])
        a_rec, a_org = np.load(p["recomputed"] / "arrays.npz"), np.load(p["original"] / "arrays.npz")
        arrays = {k: float(np.nanmax(np.abs(a_rec[k] - a_org[k]))) if a_rec[k].shape == a_org[k].shape else float("inf")
                  for k in ("real_I_s", "real_I_z", "real_p_hat")}
        for cat in agg.CATEGORIES:
            row = {"job": job, "zone": cat}
            for m, _ in COMPARED:
                row[f"{m}_original"], row[f"{m}_recomputed"] = org[m][cat]["mean"], rec[m][cat]["mean"]
            row["off_original"] = org["calibration"]["off_support_mass"][cat]
            row["off_recomputed"] = rec["calibration"]["off_support_mass"][cat]
            diffs = [abs(row[f"{m}_recomputed"] - row[f"{m}_original"]) for m, _ in COMPARED] + \
                [abs(row["off_recomputed"] - row["off_original"])]
            row["max_abs_diff"] = float(max(diffs))
            row["arrays_max_abs_diff"] = arrays
            row["flag"] = "DIFFERS -- to be reported" if (row["max_abs_diff"] > CONSISTENCY_TOL
                                                           or max(arrays.values()) > CONSISTENCY_TOL) else "same"
            rows.append(row)
    return rows


def compute(paths: dict[str, dict[str, Path]]) -> dict:
    runs = [agg.load_run(p["recomputed"]) for p in paths.values()]
    base = agg.aggregate(runs)
    if len(base["pairs"]) != 2 or base["unpaired"]:
        raise ValueError(f"expected the two win64 pairs, got {[(p['config'], p['seed']) for p in base['pairs']]}, unpaired {base['unpaired']}")
    th = agg.thresholds_from_pairs(base["pairs"])
    floor = agg.floor_from_pairs(base["pairs"], th["tau_off_floor"])
    rules = {"tau_b": th["tau_b"], "tau_off": th["tau_off"], "floor": floor, "derivation": th,
             "source": f"{BATCH}: gpu_e30b win64 pairs, seeds 0 and 5, re-measured"}
    result = agg.aggregate(runs)
    for p in result["pairs"]:
        for cat, v in p["bias_control"].items():
            v["verdict"] = e30_rule_verdict(v, cat, th["tau_b"], th["tau_off"], floor)
    return {"rules": rules, "aggregate": result}


def e30_rule_verdict(v: dict, cat: str, tau_b: dict, tau_off: dict, floor: dict) -> str:
    """The reading rule of DECISIONS 2026-09-30 (9) as applied to gpu_e30b in (15) -- kept here for that record only:
    since 2026-10-02 (16) E3-0's thresholds are an aberration control, not a rule (aggregate_b.py)."""
    if agg.is_controllable(cat):
        ok = abs(v["off_support_difference"]) <= tau_off[cat] and np.isfinite(v["b_difference"]) and abs(v["b_difference"]) <= tau_b[cat]
        return "interpretable" if ok else "NOT interpretable"
    fz, tol = floor["zones"][cat], floor["tau_off_floor"][cat]
    within = fz["off_support_min"] - tol <= v["off_support_experimental"] <= fz["off_support_max"] + tol
    return "read against the floor" if within else "diagnostic only (off-support outside the floor's interval)"


def report(check: tuple[list[str], dict], rows: list[dict], result: dict) -> str:
    problems, seen = check
    th = result["rules"]["derivation"]
    lines = ["# gpu_e30b re-measured: thresholds of the reading rule and its verdict", "",
             "Definitions fixed in docs/e3/DECISIONS.md 2026-09-30 (9), B, before the values were computed; not changed here.", "",
             "## 1. Batch check (one platform, one code version, the expected weights)", ""]
    lines += [f"- PROBLEM: {p}" for p in problems] or ["- passed"]
    lines += ["", "| job | weights SHA-256 | code source_sha256 | device / GPU | TF32 matmul / cudnn | K |", "|---|---|---|---|---|---|"]
    lines += [f"| {j} | {v['weights_sha256'][:12]}... | {(v['source_sha256'] or '?')[:12]}... | {v['device']} / {v['gpu_name']} | "
              f"{v['allow_tf32_matmul']} / {v['allow_tf32_cudnn']} | {v['kernel_draws']} |" for j, v in seen.items()]
    flagged = [r for r in rows if r["flag"] != "same"]
    lines += ["", f"## 2. Consistency with the original gpu_e30b measurements (tolerance {CONSISTENCY_TOL:g})", "",
              f"{len(flagged)} of {len(rows)} (run, zone) rows differ." + (" **To be reported: not corrected here.**" if flagged else ""), "",
              "| job | zone | I_s original / re-measured | I_z original / re-measured | off-support original / re-measured | max diff | flag |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['job']} | {r['zone']} | {r['I_s_original']:.6f} / {r['I_s_recomputed']:.6f} | {r['I_z_original']:.6f} / "
                     f"{r['I_z_recomputed']:.6f} | {r['off_original']:.6f} / {r['off_recomputed']:.6f} | {r['max_abs_diff']:.2e} | {r['flag']} |")
    lines += ["", "Per-state arrays, largest absolute difference: "
              + "; ".join(f"{r['job']}: " + ", ".join(f"{k} {v:.2e}" for k, v in r["arrays_max_abs_diff"].items())
                          for r in rows[::len(agg.CATEGORIES)]), ""]
    lines += ["## 3. Thresholds (two pairs: seeds 0 and 5, different block classes -- an sd over n = 2, one degree of freedom)", "",
              "| zone | values over the two pairs | threshold |", "|---|---|---|"]
    for cat in agg.CATEGORIES:
        vals = th["values"][cat]
        if agg.is_controllable(cat):
            lines.append(f"| {cat} | b(I_s) - b(I_s projected): {', '.join(f'{x:+.4f}' for x in vals['b_difference'])} | "
                         f"tau_b = {th['tau_b'][cat]:.4f} |")
            lines.append(f"| {cat} | off-support env - ref: {', '.join(f'{x:+.4f}' for x in vals['off_support_difference'])} | "
                         f"tau_off = {th['tau_off'][cat]:.4f} |")
        else:
            lines.append(f"| {cat} | off-support (environment): {', '.join(f'{x:.4f}' for x in vals['off_support_experimental'])} | "
                         f"tau_off_floor = {th['tau_off_floor'][cat]:.4f} (implementer's reading, to be confirmed) |")
    lines += ["", "## 4. Verdict of the rule on gpu_e30b", "",
              "Applied to the same two pairs that set the thresholds (see DECISIONS 2026-10-01 (10) on what this verdict can mean).", "",
              "| seed | zone | b (I_s) | b (I_s projected) | off-support env / ref | verdict |", "|---|---|---|---|---|---|"]
    for p in result["aggregate"]["pairs"]:
        for cat, v in p["bias_control"].items():
            lines.append(f"| {p['seed']} | {cat} | {v['b_I_s']:+.4f} | {v['b_I_s_projected']:+.4f} | {v['off_support_experimental']:.4f} / "
                         f"{v['off_support_reference']:.4f} | {v['verdict']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measures", type=Path, default=DEFAULT_MEASURES, help=f"the measurement batch's handoff (default handoff/{BATCH})")
    parser.add_argument("--originals", type=Path, default=DEFAULT_ORIGINALS, help="gpu_e30b's handoff (the original measurements)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} already holds files: choose a new directory")
    with tempfile.TemporaryDirectory(prefix="tau_") as tmp:
        paths = extract(args.measures, args.originals, Path(tmp))
        check = batch_check(paths)
        rows = consistency(paths)
        result = compute(paths)
        text = report(check, rows, result)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "TAU.md").write_text(text, encoding="utf-8")
        (args.out / "B_BY_ZONE.md").write_text(agg.markdown(result["aggregate"]), encoding="utf-8")
        (args.out / "thresholds.json").write_text(json.dumps(result["rules"], indent=1, default=float), encoding="utf-8")
    print(text)
    flagged = any(r["flag"] != "same" for r in rows)
    if check[0] or flagged:
        print("STOP: " + ("batch check failed; " if check[0] else "") + ("consistency differences to report" if flagged else ""), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
