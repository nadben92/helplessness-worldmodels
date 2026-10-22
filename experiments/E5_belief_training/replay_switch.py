"""Replays the estimator switch on the switch statistic a run recorded (docs/e5/DECISIONS.md (28), (29)): when would the
run have switched, and by which rule, under a given criterion -- exactly, since the training is the same up to the
switch whatever the rule (the rule only reads the statistic). Also: when the statistic collapsed (for a run that never
switched), and the switch's margin before it.

The series: every evaluation of the same-background statistic (every switch_eval_every = 50 gradient steps), recorded
by run_one in checkpoints.jsonl (`switch_pairs.evals`: gradient step, statistic, items with a partner, groups) since
the same-background statistic (2026-10-19); a run without it is refused. The criterion: E4's main rule (3 bits on 5
consecutive evaluations from step 500), plus E5's late threshold (`--late-threshold` bits on 5 consecutive evaluations
from `--late-after`); `--grid` replays a grid of late thresholds and activations as well.

Collapse: the first evaluation of the last stretch below `--collapse-bits` (1.0, the collapse guard's) that lasts to the
end of the series, if the statistic had been above it before (none otherwise); the reconstruction error on the agent's
patch is read from the checkpoint records around it.

Usage: python experiments/E5_belief_training/replay_switch.py <run dir> ... [--late-threshold 1.5 --late-after 6100]
           [--grid] [--out <file.json>] [--series-out <file.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from helplessness.agents.worldmodel.loop import SeparationSwitch
from helplessness.pipeline.runtime import switch_thresholds_bits
GRID_THRESHOLDS = (1.0, 1.25, 1.5, 1.75, 2.0)
GRID_AFTER = (6100, 8000, 10000)


def series(run_dir: Path) -> tuple[list[tuple[int, float]], list[dict]]:
    """(gradient step, statistic) of every recorded evaluation, in order, and the checkpoint records."""
    records = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    evals = [(int(e[0]), float(e[1])) for r in records for e in (r.get("switch_pairs") or {}).get("evals", [])]
    if not evals:
        raise SystemExit(f"{run_dir}: no recorded evaluations (switch_pairs.evals)")
    return evals, records


def main_rule(run_dir: Path) -> dict:
    """The run's own main rule (its config: E4's 3 bits on 5 consecutive evaluations every 50 steps from 500)."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    est = cfg["train"]["latent_estimator"]
    threshold, collapse = switch_thresholds_bits(est, cfg["model"]["n_vars"])
    return {"threshold_bits": threshold, "consecutive": est["switch_consecutive"], "eval_every": est["switch_eval_every"],
            "min_steps": est["switch_min_steps"], "collapse_threshold_bits": collapse, "max_steps": est["switch_max_steps"]}


def replay(evals: list[tuple[int, float]], rule: dict, late_threshold: float | None, late_after: int | None) -> dict:
    sw = SeparationSwitch(late_threshold_bits=late_threshold, late_after_steps=late_after, **rule)
    for step, stat in evals:
        sw.update(step, stat)
        if sw.switched:
            break
    return {"switch_step": sw.switch_step, "rule": sw.switch_rule}


def collapse_onset(evals: list[tuple[int, float]], bits: float) -> int | None:
    if not any(stat >= bits for _, stat in evals):
        return None
    onset = None
    for step, stat in evals:
        if stat < bits:
            onset = step if onset is None else onset
        else:
            onset = None
    return onset


def gaps(evals: list[tuple[int, float]], every: int = 50) -> list[tuple[int, int]]:
    steps = [s for s, _ in evals]
    return [(a, b) for a, b in zip(steps, steps[1:]) if b - a != every]


def analyse(run_dir: Path, late_threshold: float, late_after: int, collapse_bits: float, grid: bool) -> dict:
    evals, records = series(run_dir)
    rule = main_rule(run_dir)
    onset = collapse_onset(evals, collapse_bits)
    main = replay(evals, rule, late_threshold, late_after)
    out = {"run": run_dir.name, "main_rule": rule, "n_evals": len(evals), "first_step": evals[0][0], "last_step": evals[-1][0],
           "gaps": gaps(evals, rule["eval_every"]), "max_statistic": max(s for _, s in evals), "e4_rule_only": replay(evals, rule, None, None),
           "recorded_switch_step": records[-1].get("switch_step") if records else None,
           "criterion": {"late_threshold": late_threshold, "late_after": late_after, **main}, "collapse_onset": onset,
           "margin_before_collapse": None if onset is None or main["switch_step"] is None else onset - main["switch_step"]}
    if onset is not None:
        before = [s for s, v in evals if s < onset][-5:]
        out["statistic_before_collapse"] = [(s, v) for s, v in evals if s in before or onset <= s < onset + 250]
        out["recon_patch_around_collapse"] = [(r["grad_step"], (r.get("train") or {}).get("recon_patch")) for r in records
                                              if abs(r["grad_step"] - onset) <= 6000]
    if grid:
        out["grid"] = []
        for t in GRID_THRESHOLDS:
            for a in GRID_AFTER:
                r = replay(evals, rule, t, a)
                out["grid"].append({"late_threshold": t, "late_after": a, **r,
                                    "before_collapse": None if onset is None else r["switch_step"] is not None and r["switch_step"] < onset})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--late-threshold", type=float, default=1.5)
    parser.add_argument("--late-after", type=int, default=6100)
    parser.add_argument("--collapse-bits", type=float, default=1.0)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--series-out", type=Path, default=None, help="also write the runs' series (light: for a test)")
    args = parser.parse_args(argv)
    results = [analyse(d, args.late_threshold, args.late_after, args.collapse_bits, args.grid) for d in args.runs]
    text = json.dumps(results, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.series_out:
        args.series_out.write_text(json.dumps({d.name: series(d)[0] for d in args.runs}), encoding="utf-8")
    for r in results:
        c = r["criterion"]
        print(f"{r['run']}: {r['n_evals']} evaluations ({r['first_step']}-{r['last_step']}, gaps {len(r['gaps'])}), max {r['max_statistic']:.3f} bits; "
              f"E4's rule alone: {r['e4_rule_only']['switch_step']}; criterion (late {c['late_threshold']} from {c['late_after']}): "
              f"{c['switch_step']} [{c['rule']}]; collapse onset {r['collapse_onset']}, margin {r['margin_before_collapse']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
