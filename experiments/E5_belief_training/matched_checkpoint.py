"""The baseline checkpoint matched to a shuffle pair's level of learning (docs/e5/DECISIONS.md (36)). Standard library
only (check_wave.py's reading: the recomputed canonical records of (35) replace the recorded ones), so it runs in the
notebook. No probe, nothing written but --out.

For each shuffle pair of the batch (its experimental run against its reference: its own seed's, or REFERENCE_SEED_OF's)
at checkpoint --at:
  level      L = the smallest canonical quadrant mass on the true successor of the pair's two runs (the 8 values of E4's
             mass sub-criterion), with the pair's level declared: each run's four masses, b at step 1 by slot and the
             sub-criteria of E4's criterion that fail;
  baseline   the baseline of the pair's reference seed (E and F: seed B's; A: E4's own pair, under --e4-results);
  matched    its FIRST checkpoint (common to its two runs, in increasing steps) where the smallest canonical quadrant mass
             of its two runs is >= L; a checkpoint with a quadrant not measured does not count. Printed with the baseline's
             trajectory up to it, the weights' presence and the baseline runs' estimator switch (flagged if the matched
             checkpoint precedes it). No match: an explicit error, never a fallback.
The submission line of e5_probes_baselines with the matched checkpoints closes the output.

Usage, from the e5 checkout (PYTHONPATH=src):
  python experiments/E5_belief_training/matched_checkpoint.py --batch e5_extend_b --at 475000 [--e4-results <E4 results root>]
      [--repo-dir .] [--out <file.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import batches
import check_wave
import e5_names


def masses(record: dict) -> list[float | None]:
    return list(check_wave.mass_of(record).values())


def level(*records: dict) -> float | None:
    """The smallest canonical quadrant mass over the runs' records; None when a quadrant was not measured."""
    values = [v for r in records for v in masses(r)]
    return None if any(v is None for v in values) else min(values)


def matched_step(exp_records: list[dict], ref_records: list[dict], target: float) -> tuple[int | None, list[tuple[int, float | None]]]:
    """The first common checkpoint whose level (both runs) is >= target, and the trajectory (step, level) up to it."""
    by_step = {r["env_step"]: r for r in ref_records}
    trajectory = []
    for r in sorted(exp_records, key=lambda x: x["env_step"]):
        s = r["env_step"]
        if s not in by_step:
            continue
        lv = level(r, by_step[s])
        trajectory.append((s, lv))
        if lv is not None and lv >= target:
            return s, trajectory
    return None, trajectory


def record_at(run: dict, step: int) -> dict:
    found = [r for r in run["records"] if r["env_step"] == step]
    if not found:
        raise SystemExit(f"{run['name']}: no checkpoint record at {step}")
    return found[0]


def declared_level(exp_rec: dict, ref_rec: dict) -> dict:
    """The pair's level at the checkpoint: masses by quadrant, b at step 1 by slot, the failing sub-criteria of E4's criterion."""
    c = check_wave.criterion(exp_rec, ref_rec)
    b = c["b_step1_by_slot"]
    fail = [f"mass {who} {min(v for v in masses(r) if v is not None):.3f} < {check_wave.MASS_MIN}"
            for who, r in (("experimental", exp_rec), ("reference", ref_rec))
            if any(v is None or v < check_wave.MASS_MIN for v in masses(r))]
    ud = b[check_wave.ROLES["ud"]]
    if ud is None or abs(ud + check_wave.LOG2_5) > check_wave.B_UD_TOL:
        fail.append(f"|b_UD + log2 5| = {'-' if ud is None else f'{abs(ud + check_wave.LOG2_5):.3f}'} > {check_wave.B_UD_TOL}")
    for role in ("cd", "neutral", "other"):
        v = b[check_wave.ROLES[role]]
        if v is None or abs(v) > check_wave.B_CD_MAX:
            fail.append(f"|b_{role}| = {'-' if v is None else f'{abs(v):.3f}'} > {check_wave.B_CD_MAX}")
    return {"mass_experimental": check_wave.mass_of(exp_rec), "mass_reference": check_wave.mass_of(ref_rec),
            "b_step1_by_slot": b, "converged": c["converged"], "failing": fail}


def baseline_dirs(repo: Path, label: str, configuration: str, e4_results: Path | None) -> dict[str, Path]:
    if label == "A":  # seed A's baseline is E4's own pair
        if e4_results is None:
            raise SystemExit("seed A's baseline is E4's own pair: --e4-results <E4 results root>")
        return {c: e5_names.find_e4_run(e4_results, configuration, c) for c in ("experimental", "reference")}
    out = {c: check_wave.newest(repo, label, "baseline", c, None) for c in ("experimental", "reference")}
    missing = [c for c, d in out.items() if d is None]
    if missing:
        raise SystemExit(f"no baseline {', '.join(missing)} run of seed {label}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", required=True, choices=list(batches.EXTEND_BATCHES) + ["e5_wave1", "e5_wave2"])
    parser.add_argument("--at", type=int, required=True, help="the shuffle pairs' probe checkpoint")
    parser.add_argument("--configuration", default=batches.CONFIGURATION)
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--e4-results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    repo = args.repo_dir.resolve()
    out, ks = [], []
    for job, (label, training, fraction, condition) in batches.batch_runs(args.batch).items():
        if training != "shuffle" or condition != "experimental":
            continue
        ref_label = batches.REFERENCE_SEED_OF.get((label, training), label)
        dirs = {"experimental": check_wave.newest(repo, label, training, "experimental", fraction),
                "reference": check_wave.newest(repo, ref_label, training, "reference", fraction)}
        if any(d is None for d in dirs.values()):
            raise SystemExit(f"{job}: run not found ({dirs})")
        exp, ref = (check_wave.read_run(dirs[c]) for c in ("experimental", "reference"))
        e_rec, r_rec = record_at(exp, args.at), record_at(ref, args.at)
        target = level(e_rec, r_rec)
        if target is None:
            raise SystemExit(f"{job}: a quadrant mass not measured at {args.at}")
        declared = declared_level(e_rec, r_rec)
        base = {c: check_wave.read_run(d) for c, d in baseline_dirs(repo, ref_label, args.configuration, args.e4_results).items()}
        k, trajectory = matched_step(base["experimental"]["records"], base["reference"]["records"], target)
        if k is None:
            raise SystemExit(f"{job}: the baseline of seed {ref_label} never reaches the level {target:.4f} "
                             f"(its highest: {max((lv for _, lv in trajectory if lv is not None), default=None)})")
        switches = {c: check_wave.switch_of(r)["env_step"] for c, r in base.items()}
        weights = {c: (r["dir"] / "checkpoints" / f"agent_env{k:06d}.pt").is_file() for c, r in base.items()}
        before_switch = [c for c, s in switches.items() if s is None or k < int(s)]
        pair = f"{label} vs reference {ref_label}"
        print(f"PAIR {pair} at {args.at} ({exp['name']} / {ref['name']}"
              + (f"; canonical records recomputed at {exp['recomputed']} / {ref['recomputed']}" if exp["recomputed"] or ref["recomputed"] else "") + ")")
        for who in ("experimental", "reference"):
            m = declared[f"mass_{who}"]
            print(f"  mass {who:12s} " + " ".join(f"{q} {'-' if v is None else f'{v:.3f}'}" for q, v in m.items()))
        b = declared["b_step1_by_slot"]
        print("  b step 1 by slot " + " ".join(f"{role} ({q}) " + ("-" if b[q] is None else f"{b[q]:+.3f}") for role, q in check_wave.ROLES.items()))
        print(f"  E4's criterion {'MET' if declared['converged'] else 'not met: ' + '; '.join(declared['failing'])}")
        print(f"  level L = {target:.4f}; baseline {ref_label} ({base['experimental']['name']} / {base['reference']['name']}):")
        for s, lv in trajectory:
            print(f"    {s:>7}  level {'-' if lv is None else f'{lv:.4f}'}" + ("   <- MATCHED" if s == k else ""))
        print(f"  matched checkpoint {k}; weights agent_env{k:06d}.pt: {weights}; estimator switch (env step): {switches}"
              + (f"; ATTENTION: before the switch of {before_switch}" if before_switch else ""))
        out.append({"pair": pair, "at": args.at, "level": target, "declared": declared, "baseline_seed_label": ref_label,
                    "matched_checkpoint": k, "trajectory": trajectory, "weights": weights, "baseline_switch_env_step": switches,
                    "before_switch": before_switch})
        ks.append(k)
    if not out:
        raise SystemExit(f"{args.batch}: no shuffle experimental run")
    print("MATCHED: " + ", ".join(f"{o['pair']} -> baseline {o['baseline_seed_label']} at {o['matched_checkpoint']}" for o in out))
    labels = sorted({o["baseline_seed_label"] for o in out})
    if labels == list(batches.MATCHED_BASELINE_SEEDS["e5_probes_baselines"]):
        print("SUBMIT: python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_baselines --phase probes "
              f"--checkpoint {' '.join(str(k) for k in sorted(set(ks)))} --e4-results <E4 results root> --repo-dir {repo} --dry-run")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
