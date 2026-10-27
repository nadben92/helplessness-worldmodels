"""The learning check of an E5 wave, between two waves (docs/e5/DECISIONS.md (31)): one command, one line per run and
one per pair, from the runs' light files only (meta.yaml, checkpoints.jsonl; standard library only, so it runs in the
notebook as well as on an extracted export). No probe.

Per run:
  status            meta.yaml's (done / failed, with the error)
  switch            the estimator switch: gradient and environment steps, the rule (threshold / late_threshold); a
                    continuation (run_e5.py --extend, meta.yaml branched_from) carries its original's switch: before its
                    first checkpoint (or when it failed at once) it is read from the original, found by name next to it,
                    up the continuation chain (docs/e5/DECISIONS.md (34))
  collapse          after the switch, the longest stretch of consecutive evaluations of the switch statistic below the
                    collapse guard (1.0 bit; the guard stops a run at 5) and the lowest value; the last value
  mass              at the last checkpoint, the prior's mass on the true successor in each quadrant of the canonical
                    layout (the smallest of the four; E4's convergence threshold 0.97); shuffle runs: also by zone type,
                    averaged over the 12 assignments, at the last checkpoint measuring every assignment; where
                    checkpoints_recomputed.jsonl holds a reproduced recomputation of the canonical record
                    (recompute_checkpoints.py, docs/e5/DECISIONS.md (35)), its values replace the recorded ones
  success           the agent's success rate and episode length over the last episodes (at the last checkpoint)
Per pair (experimental - reference, same training; the reference's seed is the run's own or batches.REFERENCE_SEED_OF's --
seed B's f = 1 reference for the replacements E and F, docs/e5/DECISIONS.md (32); with --batch, such a reference from
another wave is added):
  b                 b at the first step in the zone in the training world, by slot of the canonical layout (fixed walks
                    on the training image, E4's checkpoint record) at the last common checkpoint; shuffle pairs: also by
                    zone type averaged over the 12 assignments; E4's criterion: |b_UD + log2 5| <= 0.15, |b| <= 0.1 at CD
                    and in the corridor-like slots, mass >= 0.97 in the four quadrants of both runs
  converged at      the checkpoints where that criterion holds (the candidates for the common probe checkpoint)
A run is flagged ATTENTION when it failed (a run still training or stopped by the user stays 'running': not flagged), never switched, had a stretch of 5 evaluations below 1.0 bit after its switch,
or misses the mass threshold at its last checkpoint; LOCK-IN when it has not switched and its statistic stayed below 0.01
bit on at least 95% of the evaluations from gradient step 6,100 on (at least 20) -- a run locked in, never switched or
failed is EXCLUDED from every pair (and so from the probes), listed apart; NaN/inf when a training loss present in a record or
the statistic is not finite (the agent heads' losses are ABSENT before the switch -- the console prints them as nan --
and are not flagged); a pair when it misses E4's criterion at its last checkpoint.

Usage, from the e5 checkout (notebook) or the repo root (an extracted export):
  python experiments/E5_belief_training/check_wave.py --batch e5_wave1 [--repo-dir .] [--out <file.json>]
  python experiments/E5_belief_training/check_wave.py --runs <run dir> ... [--out <file.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import batches
import e5_names
from export_batch import meta_fields, seeds

LOG2_5 = math.log2(5)
# the canonical roles of diagonal 1 (envs.layouts.two_zones.roles(1); tested equal): p1-b-d1
ROLES = {"ud": "nw", "cd": "se", "neutral": "ne", "other": "sw"}
QUADRANTS = ("nw", "ne", "se", "sw")
MASS_MIN, B_UD_TOL, B_CD_MAX, COLLAPSE_BITS, COLLAPSE_CONSECUTIVE = 0.97, 0.15, 0.1, 1.0, 5
# lock-in (docs/e5/DECISIONS.md (32), (33)): no switch, and the switch statistic below LOCK_BITS on at least LOCK_SHARE of
# the evaluations from the late threshold's activation on (LOCK_AFTER), at least LOCK_MIN_EVALS of them (isolated spikes
# allowed: seed B's locked-in run had 19 of 1,375 evaluations between 0.01 and 0.042 bit)
LOCK_BITS, LOCK_AFTER, LOCK_MIN_EVALS, LOCK_SHARE = 0.01, 6100, 20, 0.95


def read_run(run_dir: Path) -> dict:
    meta = meta_fields(run_dir / "meta.yaml")
    lines = [x for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    records = []
    for i, x in enumerate(lines):
        try:
            records.append(json.loads(x))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            # a pod stopped while appending a record leaves its last line truncated: that checkpoint's record is lost (its
            # weights, written atomically, are whole or absent), the others are read (docs/e5/DECISIONS.md (33))
            print(f"{run_dir.name}: last record truncated (a stopped run), ignored", flush=True)
    records = [r for r in records if r.get("e4")]
    recomputed, refused = apply_recomputed(run_dir, records)
    name = run_dir.name
    preset = name.split("__")[1]  # 2zones-<configuration>-<tag>
    return {"dir": run_dir, "name": name, "tag": preset.rsplit("-", 1)[1], "role": "experimental" if "__pair-experimental__" in name else "reference",
            "seed": int(name.split("__seed")[1].split("__")[0]), "meta": meta, "records": records, "recomputed": recomputed,
            "recompute_refused": refused}


RECOMPUTED = "checkpoints_recomputed.jsonl"


def apply_recomputed(run_dir: Path, records: list[dict]) -> tuple[list[int], list[int]]:
    """The canonical assignment's real-context records recomputed by recompute_checkpoints.py (docs/e5/DECISIONS.md (35):
    the checkpoints a restored run measured against another assignment's dynamics), read from checkpoints_recomputed.jsonl
    beside checkpoints.jsonl (never rewritten) and put in place of the recorded ones in `records` (in place): e4's
    real_context, e5's by_assignment of the canonical assignment and the means over assignments of the mass and I_s. Only a
    recomputation that reproduced the record's other fields is used. Returns (steps replaced, steps refused)."""
    path = run_dir / RECOMPUTED
    if not path.is_file():
        return [], []
    lines = {int(x["env_step"]): x for x in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if x}
    done, refused = [], []
    for r in records:
        line = lines.get(int(r["env_step"]))
        if line is None:
            continue
        if not line.get("reproduced"):
            refused.append(int(r["env_step"]))
            continue
        canonical = str(r["e5"]["canonical"])
        r["e4"]["real_context"] = line["e4"]["real_context"]
        r["e5"]["by_assignment"][canonical]["real_context"] = line["e5"]["by_assignment"][canonical]["real_context"]
        if line["e5"].get("mean_over_assignments") and r["e5"].get("mean_over_assignments"):
            r["e5"]["mean_over_assignments"].update(line["e5"]["mean_over_assignments"])
        done.append(int(r["env_step"]))
    return done, refused


def original_of(run: dict) -> Path | None:
    """The run a continuation was branched from (meta.yaml branched_from: <original>/checkpoints/resume.pt, a path of the
    machine that ran it): the directory of that name next to the continuation, None for a run that is not a continuation."""
    src = run["meta"].get("branched_from")
    if not src:
        return None
    parts = [p for p in re.split(r"[\\/]", src) if p]
    return run["dir"].parent / parts[-3] if len(parts) >= 3 and parts[-2] == "checkpoints" else None


def switch_of(run: dict) -> dict:
    for r in reversed(run["records"]):
        sw = (r.get("e5") or {}).get("estimator_switch") or {}
        if sw.get("grad_step") is not None or r.get("switch_step") is not None:
            return {"grad_step": sw.get("grad_step", r.get("switch_step")), "env_step": sw.get("env_step"), "rule": sw.get("rule")}
    origin = original_of(run)
    if origin is not None:  # a continuation with no record yet: its state, switch included, is its original's
        if not (origin / "checkpoints.jsonl").is_file():
            return {"grad_step": None, "env_step": None, "rule": None, "from": origin.name, "original_missing": True}
        return {**switch_of(read_run(origin)), "from": origin.name}
    return {"grad_step": None, "env_step": None, "rule": None}


def collapse_of(run: dict, switch_grad: int | None) -> dict:
    evals = [(int(e[0]), float(e[1])) for r in run["records"] for e in (r.get("switch_pairs") or {}).get("evals", [])]
    after = [v for s, v in evals if switch_grad is not None and s >= switch_grad]
    streak = longest = 0
    for v in after:
        streak = streak + 1 if v < COLLAPSE_BITS else 0
        longest = max(longest, streak)
    return {"evaluations_after_switch": len(after), "longest_below_1bit": longest,
            "min_after_switch": min(after) if after else None, "last": evals[-1][1] if evals else None}


def lock_in_of(run: dict, switched: bool) -> dict | None:
    """Not switched, and the statistic below LOCK_BITS on at least LOCK_SHARE of the evaluations from LOCK_AFTER on: the
    latent locked on the background (docs/e5/DECISIONS.md (31)-(33))."""
    after = [float(e[1]) for r in run["records"] for e in (r.get("switch_pairs") or {}).get("evals", []) if int(e[0]) >= LOCK_AFTER]
    if switched or len(after) < LOCK_MIN_EVALS:
        return None
    share = sum(v < LOCK_BITS for v in after) / len(after)
    return {"evaluations": len(after), "share_below": share, "max": max(after)} if share >= LOCK_SHARE else None


def non_finite_of(run: dict) -> list[str]:
    """Values that are NaN or infinite where a number was COMPUTED: the training losses present in a checkpoint record and
    the switch statistic. The agent heads' losses are absent before the switch (the heads train from it on; the console
    prints them as nan): absent is not flagged."""
    out = []
    for r in run["records"]:
        bad = sorted(k for k, v in (r.get("train") or {}).items() if isinstance(v, (int, float)) and not math.isfinite(v))
        if r.get("batch_pair_js") is not None and not math.isfinite(float(r["batch_pair_js"])):
            bad.append("batch_pair_js")
        if bad:
            out.append(f"{r['env_step']}: {', '.join(bad)}")
    return out


def number(v) -> float | None:
    """A recorded number, or None when it was not measured: the records write a NaN as null (pipeline.runtime._jsonable),
    e.g. a quadrant with no measured state at the first checkpoints (no context of that region in the replay yet)."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def mass_of(record: dict) -> dict:
    """Quadrant -> the prior's mass on the true successor, None when not measured."""
    m = record["e4"]["real_context"]["mass_true_by_region"]
    return {q: number(m.get(q)) for q in QUADRANTS}


def missing_of(run: dict) -> list[str]:
    """The checkpoints where a quadrant's mass or a slot's I at step 1 was not measured (null in the record)."""
    out = []
    for r in run["records"]:
        gaps = [f"mass {q}" for q, v in mass_of(r).items() if v is None] + [f"I step 1 {q}" for q in QUADRANTS if step1(r, q) is None]
        if gaps:
            out.append(f"{r['env_step']}: {', '.join(gaps)}")
    return out


def mean_over_assignments(run: dict) -> tuple[int | None, dict | None]:
    for r in reversed(run["records"]):
        mo = (r.get("e5") or {}).get("mean_over_assignments")
        if mo:
            return r["env_step"], mo
    return None, None


def step1(record: dict, slot: str) -> float | None:
    return number(record["e4"]["walks"][slot]["I_curve"][0])


def criterion(exp: dict, ref: dict) -> dict:
    """E4's convergence criterion at one checkpoint; a value not measured makes it unmet (missing lists what)."""
    e1, r1 = ({q: step1(rec, q) for q in QUADRANTS} for rec in (exp, ref))
    b = {q: None if e1[q] is None or r1[q] is None else e1[q] - r1[q] for q in QUADRANTS}
    masses = [v for rec in (exp, ref) for v in mass_of(rec).values()]
    missing = [f"mass ({sum(v is None for v in masses)} quadrant value(s))"] if any(v is None for v in masses) else []
    missing += [f"b {q}" for q in QUADRANTS if b[q] is None]
    mass_ok = not any(v is None for v in masses) and all(v >= MASS_MIN for v in masses)
    corridor_like = [ROLES["cd"], ROLES["neutral"], ROLES["other"]]
    ok = (mass_ok and not missing and abs(b[ROLES["ud"]] + LOG2_5) <= B_UD_TOL
          and all(abs(b[q]) <= B_CD_MAX for q in corridor_like))
    return {"b_step1_by_slot": b, "mass_ok": mass_ok, "converged": ok, "missing": missing}


def check_run(run: dict) -> dict:
    sw = switch_of(run)
    col = collapse_of(run, sw["grad_step"])
    last = run["records"][-1] if run["records"] else None
    mass = mass_of(last) if last else {}
    k_all, mo = mean_over_assignments(run)
    lock = lock_in_of(run, sw["grad_step"] is not None)
    nan = non_finite_of(run)
    missing = missing_of(run)
    flags = []
    if missing:
        flags.append(f"NOT MEASURED (null) at checkpoint(s) {'; '.join(missing[:4])}" + (" ..." if len(missing) > 4 else ""))
    if lock is not None:
        flags.append(f"LOCK-IN: no switch, statistic < {LOCK_BITS} bit on {lock['share_below']:.1%} of the {lock['evaluations']} "
                     f"evaluations from {LOCK_AFTER} on (max {lock['max']:.4f})")
    if nan:
        flags.append(f"NaN/inf at checkpoint(s) {'; '.join(nan[:3])}" + (" ..." if len(nan) > 3 else ""))
    if run["meta"].get("status") == "failed":
        flags.append(f"status failed: {run['meta'].get('error', '')[:120]}")
    if run.get("recompute_refused"):
        flags.append(f"recomputation NOT reproduced at {run['recompute_refused']} (recorded values kept; recompute_checkpoints.py)")
    if sw.get("original_missing"):
        flags.append(f"continuation of {sw['from']}, not found here: switch unknown")
    elif sw["grad_step"] is None:
        flags.append("never switched")
    if col["longest_below_1bit"] >= COLLAPSE_CONSECUTIVE:
        flags.append(f"{col['longest_below_1bit']} consecutive evaluations below {COLLAPSE_BITS} bit after the switch")
    measured = [v for v in mass.values() if v is not None]
    if measured and min(measured) < MASS_MIN:
        flags.append(f"quadrant mass {min(measured):.3f} < {MASS_MIN}")
    return {"run": run["name"], "tag": run["tag"], "role": run["role"], "seed": run["seed"], "status": run["meta"].get("status"),
            "last_env_step": last["env_step"] if last else None, "switch": sw, "collapse": col, "mass_by_quadrant": mass,
            "mass_by_type_mean_over_assignments": None if mo is None else {"env_step": k_all, **mo.get("mass_true_by_type", {})},
            "success_rate": last.get("success_rate") if last else None, "episode_length": last.get("episode_length") if last else None,
            "lock_in": lock, "non_finite": nan, "not_measured": missing, "recomputed": run.get("recomputed", []), "attention": flags}


def check_pair(exp: dict, ref: dict) -> dict:
    by_step = {r["env_step"]: r for r in ref["records"]}
    common = [r for r in exp["records"] if r["env_step"] in by_step]
    if not common:
        return {"pair": f"{exp['tag']} seed {exp['seed']}", "attention": ["no common checkpoint"]}
    crit = {r["env_step"]: criterion(r, by_step[r["env_step"]]) for r in common}
    last = common[-1]["env_step"]
    _, mo_e = mean_over_assignments(exp)
    _, mo_r = mean_over_assignments(ref)
    by_type = None
    if mo_e and mo_r:
        by_type = {t: None if number(mo_e["I_step1_by_type"][t]) is None or number(mo_r["I_step1_by_type"][t]) is None
                   else number(mo_e["I_step1_by_type"][t]) - number(mo_r["I_step1_by_type"][t]) for t in mo_e["I_step1_by_type"]}
    out = {"pair": f"{exp['tag']} seed {exp['seed']}", "last_common_env_step": last, "b_step1_by_slot": crit[last]["b_step1_by_slot"],
           "b_step1_by_type_mean_over_assignments": by_type, "converged_at_last": crit[last]["converged"],
           "converged_at": [s for s, c in crit.items() if c["converged"]], "attention": []}
    unmeasured = [f"{s}: {', '.join(c['missing'])}" for s, c in crit.items() if c["missing"]]
    if unmeasured:
        out["attention"].append(f"criterion not computable (value not measured) at {'; '.join(unmeasured[:4])}" + (" ..." if len(unmeasured) > 4 else ""))
    if not crit[last]["converged"]:
        out["attention"].append("E4's convergence criterion not met at the last checkpoint")
    return out


def newest(repo: Path, label: str, training: str, condition: str, fraction) -> Path | None:
    seed_of = seeds(repo, batches.CONFIGURATION)
    found = sorted((repo / "results" / e5_names.EXPERIMENT_ID).glob(
        e5_names.run_glob(batches.CONFIGURATION, training, condition, seed_of[label], fraction)))
    return found[-1] if found else None


def batch_run_dirs(repo: Path, batch: str) -> list[Path]:
    """The batch's runs, plus the references of another wave its experimental runs are paired with (REFERENCE_SEED_OF)."""
    out = []
    for job, (label, training, fraction, condition) in batches.batch_runs(batch).items():
        d = newest(repo, label, training, condition, fraction)
        if d is None:
            print(f"{job}: no run found", flush=True)
        else:
            out.append(d)
        ref_label = batches.REFERENCE_SEED_OF.get((label, training))
        if condition == "experimental" and ref_label is not None:
            ref = newest(repo, ref_label, training, "reference", fraction)
            if ref is not None and ref not in out:
                out.append(ref)
    return out


def reference_seed(repo: Path, tag: str, seed: int) -> int:
    """The seed of the reference an experimental run is measured against (its own, or REFERENCE_SEED_OF's)."""
    seed_of = seeds(repo, batches.CONFIGURATION)
    label = next((lab for lab, s in seed_of.items() if s == seed), None)
    training = "baseline" if tag == "base" else "shuffle" if tag.startswith("sh") else "immunisation"
    ref_label = batches.REFERENCE_SEED_OF.get((label, training))
    return seed if ref_label is None else seed_of[ref_label]


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    dirs = list(args.runs) + (batch_run_dirs(args.repo_dir.resolve(), args.batch) if args.batch else [])
    if not dirs:
        parser.error("give --batch or --runs")
    runs = [read_run(d) for d in dirs]
    run_checks = [check_run(r) for r in runs]
    pairs, excluded = [], []
    # a run in no pair (docs/e5/DECISIONS.md (32), (33)): locked in, never switched or failed -- as the probe plan; a run
    # still training or stopped by the user (status running) stays in its pairs
    never = {c["run"] for c in run_checks if c["switch"]["grad_step"] is None and not c["switch"].get("original_missing")}
    locked = {c["run"]: ("LOCK-IN" if c["lock_in"] is not None else "never switched" if c["run"] in never else "failed")
              for c in run_checks if c["lock_in"] is not None or c["run"] in never or c["status"] == "failed"}
    repo = args.repo_dir.resolve()
    for exp in (r for r in runs if r["role"] == "experimental"):
        ref_seed = reference_seed(repo, exp["tag"], exp["seed"])
        ref = next((r for r in runs if r["role"] == "reference" and r["tag"] == exp["tag"] and r["seed"] == ref_seed), None)
        if ref is None:
            continue
        if exp["name"] in locked or ref["name"] in locked:  # docs/e5/DECISIONS.md (32): a locked-in run is in no pair
            excluded.append({"pair": f"{exp['tag']} seed {exp['seed']} (reference seed {ref_seed})",
                             "reason": "; ".join(f"{r['role']} {locked[r['name']]}" for r in (exp, ref) if r["name"] in locked)})
            continue
        pair = check_pair(exp, ref)
        pair["reference_seed"] = ref_seed
        pairs.append(pair)
    print("RUNS")
    for c in run_checks:
        sw, col = c["switch"], c["collapse"]
        print(f"  {c['tag']} seed {c['seed']} {c['role']}: {c['status']} at {c['last_env_step']} | switch {fmt(sw['grad_step'])} "
              f"(env {fmt(sw['env_step'])}, {sw['rule'] or '-'}{', from the original run' if sw.get('from') else ''}) | after it: longest {col['longest_below_1bit']} below 1 bit, min "
              f"{fmt(col['min_after_switch'])}, last {fmt(col['last'])} | mass min {fmt(min([v for v in c['mass_by_quadrant'].values() if v is not None], default=None))} "
              f"| success {fmt(c['success_rate'])} "
              + (f"| canonical record recomputed at {len(c['recomputed'])} checkpoint(s) " if c["recomputed"] else "")
              + f"| {'ATTENTION: ' + '; '.join(c['attention']) if c['attention'] else 'OK'}")
    print("PAIRS")
    for p in pairs:
        b = p.get("b_step1_by_slot", {})
        print(f"  {p['pair']}: at {p.get('last_common_env_step')} b step 1 UD {fmt(b.get(ROLES['ud']))} CD {fmt(b.get(ROLES['cd']))} "
              f"neutral {fmt(b.get(ROLES['neutral']))} other {fmt(b.get(ROLES['other']))}"
              + (f" | by type over assignments {({t: None if v is None else round(v, 3) for t, v in p['b_step1_by_type_mean_over_assignments'].items()})}"
                 if p.get("b_step1_by_type_mean_over_assignments") else "")
              + f" | converged at {p.get('converged_at')} | {'ATTENTION: ' + '; '.join(p['attention']) if p['attention'] else 'OK'}")
    common = sorted(set.intersection(*(set(p.get("converged_at", [])) for p in pairs))) if pairs else []
    print(f"ALL {len(pairs)} PAIRS MEET E4's CRITERION AT: {common if common else 'none yet'}")
    if excluded:
        print("EXCLUDED (a run locked in, never switched or failed: no pair, no probe)")
        for e in excluded:
            print(f"  {e['pair']}: {e['reason']}")
    if args.out:
        args.out.write_text(json.dumps({"runs": run_checks, "pairs": pairs, "excluded_pairs": excluded, "all_pairs_converged_at": common},
                                       indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
