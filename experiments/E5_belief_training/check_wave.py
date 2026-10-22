"""The learning check of an E5 wave, between two waves (docs/e5/DECISIONS.md (31)): one command, one line per run and
one per pair, from the runs' light files only (meta.yaml, checkpoints.jsonl; standard library only, so it runs in the
notebook as well as on an extracted export). No probe.

Per run:
  status            meta.yaml's (done / failed, with the error)
  switch            the estimator switch: gradient and environment steps, the rule (threshold / late_threshold)
  collapse          after the switch, the longest stretch of consecutive evaluations of the switch statistic below the
                    collapse guard (1.0 bit; the guard stops a run at 5) and the lowest value; the last value
  mass              at the last checkpoint, the prior's mass on the true successor in each quadrant of the canonical
                    layout (the smallest of the four; E4's convergence threshold 0.97); shuffle runs: also by zone type,
                    averaged over the 12 assignments, at the last checkpoint measuring every assignment
  success           the agent's success rate and episode length over the last episodes (at the last checkpoint)
Per pair (experimental - reference, same training; the reference's seed is the run's own or batches.REFERENCE_SEED_OF's --
seed B's f = 1 reference for the replacements E and F, docs/e5/DECISIONS.md (32); with --batch, such a reference from
another wave is added):
  b                 b at the first step in the zone in the training world, by slot of the canonical layout (fixed walks
                    on the training image, E4's checkpoint record) at the last common checkpoint; shuffle pairs: also by
                    zone type averaged over the 12 assignments; E4's criterion: |b_UD + log2 5| <= 0.15, |b| <= 0.1 at CD
                    and in the corridor-like slots, mass >= 0.97 in the four quadrants of both runs
  converged at      the checkpoints where that criterion holds (the candidates for the common probe checkpoint)
A run is flagged ATTENTION when it failed, never switched, had a stretch of 5 evaluations below 1.0 bit after its switch,
or misses the mass threshold at its last checkpoint; LOCK-IN when it has not switched and its statistic stayed below 0.01
bit on every evaluation from gradient step 6,100 on (at least 20) -- a locked-in run is EXCLUDED from every pair (and so
from the probes), listed apart; NaN/inf when a training loss present in a record or
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
# lock-in (docs/e5/DECISIONS.md (32)): no switch, and the switch statistic below LOCK_BITS on every evaluation from the
# late threshold's activation on (LOCK_AFTER) over at least LOCK_MIN_EVALS evaluations
LOCK_BITS, LOCK_AFTER, LOCK_MIN_EVALS = 0.01, 6100, 20


def read_run(run_dir: Path) -> dict:
    meta = meta_fields(run_dir / "meta.yaml")
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    name = run_dir.name
    preset = name.split("__")[1]  # 2zones-<configuration>-<tag>
    return {"dir": run_dir, "name": name, "tag": preset.rsplit("-", 1)[1], "role": "experimental" if "__pair-experimental__" in name else "reference",
            "seed": int(name.split("__seed")[1].split("__")[0]), "meta": meta, "records": [r for r in records if r.get("e4")]}


def switch_of(run: dict) -> dict:
    for r in reversed(run["records"]):
        sw = (r.get("e5") or {}).get("estimator_switch") or {}
        if sw.get("grad_step") is not None or r.get("switch_step") is not None:
            return {"grad_step": sw.get("grad_step", r.get("switch_step")), "env_step": sw.get("env_step"), "rule": sw.get("rule")}
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
    """Not switched, and the statistic below LOCK_BITS on every evaluation from LOCK_AFTER on: the latent locked on the
    background (docs/e5/DECISIONS.md (31), (32))."""
    after = [float(e[1]) for r in run["records"] for e in (r.get("switch_pairs") or {}).get("evals", []) if int(e[0]) >= LOCK_AFTER]
    if switched or len(after) < LOCK_MIN_EVALS:
        return None
    return {"evaluations": len(after), "max": max(after)} if max(after) < LOCK_BITS else None


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


def mass_of(record: dict) -> dict:
    m = record["e4"]["real_context"]["mass_true_by_region"]
    return {q: m[q] for q in QUADRANTS}


def mean_over_assignments(run: dict) -> tuple[int | None, dict | None]:
    for r in reversed(run["records"]):
        mo = (r.get("e5") or {}).get("mean_over_assignments")
        if mo:
            return r["env_step"], mo
    return None, None


def step1(record: dict, slot: str) -> float:
    return float(record["e4"]["walks"][slot]["I_curve"][0])


def criterion(exp: dict, ref: dict) -> dict:
    b = {q: step1(exp, q) - step1(ref, q) for q in QUADRANTS}
    mass_ok = all(v >= MASS_MIN for rec in (exp, ref) for v in mass_of(rec).values())
    corridor_like = [ROLES["cd"], ROLES["neutral"], ROLES["other"]]
    ok = mass_ok and abs(b[ROLES["ud"]] + LOG2_5) <= B_UD_TOL and all(abs(b[q]) <= B_CD_MAX for q in corridor_like)
    return {"b_step1_by_slot": b, "mass_ok": mass_ok, "converged": ok}


def check_run(run: dict) -> dict:
    sw = switch_of(run)
    col = collapse_of(run, sw["grad_step"])
    last = run["records"][-1] if run["records"] else None
    mass = mass_of(last) if last else {}
    k_all, mo = mean_over_assignments(run)
    lock = lock_in_of(run, sw["grad_step"] is not None)
    nan = non_finite_of(run)
    flags = []
    if lock is not None:
        flags.append(f"LOCK-IN: no switch, statistic < {LOCK_BITS} bit on all {lock['evaluations']} evaluations from {LOCK_AFTER} on "
                     f"(max {lock['max']:.4f})")
    if nan:
        flags.append(f"NaN/inf at checkpoint(s) {'; '.join(nan[:3])}" + (" ..." if len(nan) > 3 else ""))
    if run["meta"].get("status") != "done":
        flags.append(f"status {run['meta'].get('status')}: {run['meta'].get('error', '')[:120]}")
    if sw["grad_step"] is None:
        flags.append("never switched")
    if col["longest_below_1bit"] >= COLLAPSE_CONSECUTIVE:
        flags.append(f"{col['longest_below_1bit']} consecutive evaluations below {COLLAPSE_BITS} bit after the switch")
    if mass and min(mass.values()) < MASS_MIN:
        flags.append(f"quadrant mass {min(mass.values()):.3f} < {MASS_MIN}")
    return {"run": run["name"], "tag": run["tag"], "role": run["role"], "seed": run["seed"], "status": run["meta"].get("status"),
            "last_env_step": last["env_step"] if last else None, "switch": sw, "collapse": col, "mass_by_quadrant": mass,
            "mass_by_type_mean_over_assignments": None if mo is None else {"env_step": k_all, **mo.get("mass_true_by_type", {})},
            "success_rate": last.get("success_rate") if last else None, "episode_length": last.get("episode_length") if last else None,
            "lock_in": lock, "non_finite": nan, "attention": flags}


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
        by_type = {t: mo_e["I_step1_by_type"][t] - mo_r["I_step1_by_type"][t] for t in mo_e["I_step1_by_type"]}
    out = {"pair": f"{exp['tag']} seed {exp['seed']}", "last_common_env_step": last, "b_step1_by_slot": crit[last]["b_step1_by_slot"],
           "b_step1_by_type_mean_over_assignments": by_type, "converged_at_last": crit[last]["converged"],
           "converged_at": [s for s, c in crit.items() if c["converged"]], "attention": []}
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
    locked = {c["run"] for c in run_checks if c["lock_in"] is not None}
    repo = args.repo_dir.resolve()
    for exp in (r for r in runs if r["role"] == "experimental"):
        ref_seed = reference_seed(repo, exp["tag"], exp["seed"])
        ref = next((r for r in runs if r["role"] == "reference" and r["tag"] == exp["tag"] and r["seed"] == ref_seed), None)
        if ref is None:
            continue
        if exp["name"] in locked or ref["name"] in locked:  # docs/e5/DECISIONS.md (32): a locked-in run is in no pair
            excluded.append({"pair": f"{exp['tag']} seed {exp['seed']} (reference seed {ref_seed})",
                             "reason": "LOCK-IN: " + ", ".join(r["name"] for r in (exp, ref) if r["name"] in locked)})
            continue
        pair = check_pair(exp, ref)
        pair["reference_seed"] = ref_seed
        pairs.append(pair)
    print("RUNS")
    for c in run_checks:
        sw, col = c["switch"], c["collapse"]
        print(f"  {c['tag']} seed {c['seed']} {c['role']}: {c['status']} at {c['last_env_step']} | switch {fmt(sw['grad_step'])} "
              f"(env {fmt(sw['env_step'])}, {sw['rule'] or '-'}) | after it: longest {col['longest_below_1bit']} below 1 bit, min "
              f"{fmt(col['min_after_switch'])}, last {fmt(col['last'])} | mass min {fmt(min(c['mass_by_quadrant'].values()) if c['mass_by_quadrant'] else None)} "
              f"| success {fmt(c['success_rate'])} | {'ATTENTION: ' + '; '.join(c['attention']) if c['attention'] else 'OK'}")
    print("PAIRS")
    for p in pairs:
        b = p.get("b_step1_by_slot", {})
        print(f"  {p['pair']}: at {p.get('last_common_env_step')} b step 1 UD {fmt(b.get(ROLES['ud']))} CD {fmt(b.get(ROLES['cd']))} "
              f"neutral {fmt(b.get(ROLES['neutral']))} other {fmt(b.get(ROLES['other']))}"
              + (f" | by type over assignments {({t: round(v, 3) for t, v in p['b_step1_by_type_mean_over_assignments'].items()})}"
                 if p.get("b_step1_by_type_mean_over_assignments") else "")
              + f" | converged at {p.get('converged_at')} | {'ATTENTION: ' + '; '.join(p['attention']) if p['attention'] else 'OK'}")
    if excluded:
        print("EXCLUDED (locked in: no pair, no probe)")
        for e in excluded:
            print(f"  {e['pair']}: {e['reason']}")
    if args.out:
        args.out.write_text(json.dumps({"runs": run_checks, "pairs": pairs, "excluded_pairs": excluded}, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
