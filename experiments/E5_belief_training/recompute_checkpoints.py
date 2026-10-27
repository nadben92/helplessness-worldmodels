"""Recomputes the real-context records of a shuffle run at its checkpoints (docs/e5/DECISIONS.md (35)), into
<run>/checkpoints_recomputed.jsonl beside checkpoints.jsonl (never rewritten); check_wave.py reads it.

Why: a run restored from a resume state (a continuation, --extend; a resumed run, --resume) before the fix of (35) had
its run's world's environment replaced by the environment of the episode in progress at the restore (one assignment
of 12); the canonical assignment's world IS the run's world, so from there on its checkpoint records measured the
canonical assignment's mass on the true successor against that other assignment's dynamics. The weights, the replay
and every other field are right; in the experimental condition the canonical mass is wrong (the reference's 12
assignments share one dynamics: unchanged).

What depends on the world in a checkpoint record: the canonical assignment's real-context record (e4.real_context, the
same dict as e5.by_assignment[canonical].real_context) through its successors (mass_true_by_region, mass_true_by_type),
and e5.mean_over_assignments.mass_true_by_type. Its I_s, decodability and ghost fraction use the kernel and the
templates only; the walks (e4.walks, b) use probe worlds built afresh. e4.final (a run's last checkpoint) is not
recomputed: no stopped continuation has one, and a run finished after the fix measured it right.

How: at checkpoint N, the weights agent_env<N>.pt, the replay truncated to the record's stored_steps (its first
stored_steps steps: the replay never wraps around, so these are exactly the steps stored at N -- no later step), the
12 assignments' worlds rebuilt as the run builds them (run_e5.assignment_worlds), and pipeline.assignment_records.
real_context_by_assignment called as the checkpoint hook calls it: the same assignments (the record's
assignments_measured, in the same order: the kernels draw in turn from one generator), the same streams (keyed by the
seed, condition, checkpoint and slot), the same settings. Its replay: checkpoints/resume.pt when its step is >= N (the
last resume point of a stopped run), else replay_final.npz (a finished run).

The check: every field that does NOT depend on the world -- all fields of the other assignments, the canonical
assignment's I_s, decodability, contexts, ghost fraction -- must come out as recorded, and so must the record's visits by
zone type (from the truncated replay): `reproduced` when the largest difference is <= TOLERANCE. On the device the run
measured on (cuda: the generators' draws are the device's), it is exact on the same GPU model; TOLERANCE allows the
float summation order of another GPU model (differences of order 1e-7), while different draws (another device) give
Monte-Carlo differences of order 1e-3 or more, which it refuses. check_wave.py uses a recomputed record only when
reproduced. A record the bug did not touch (any original run's) comes out with its canonical mass unchanged too: that is
the tool's validation (RUN_GPU.md, step 6b).

Usage, from the e5 checkout (cuda: in a pod, submit_kubeflow.py --phase recompute; cpu: a local check only):
  python experiments/E5_belief_training/recompute_checkpoints.py <run dir> [--steps 300000 325000] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import e5_setup
import run_e5
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.cluster.resume import load_for_check
from helplessness.pipeline.assignment_records import mean_over_assignments, real_context_by_assignment, type_of_row, visit_shares
from helplessness.pipeline.checkpoints import build_rssm
from helplessness.pipeline.episode_sources import ASSIGNMENTS, ZoneAssignmentShuffle, canonical_assignment
from helplessness.pipeline.runtime import _jsonable, configure_runtime
from helplessness.streams.assignment import assignment_generator

OUT_NAME = "checkpoints_recomputed.jsonl"
TOLERANCE = 1e-6
WORLD_FIELDS = ("mass_true_by_region", "mass_true_by_type")  # the canonical record's fields that depend on its world


def records_by_step(run_dir: Path) -> dict[int, dict]:
    out = {}
    for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a stopped run's truncated last line (check_wave.read_run)
            continue
        if r.get("e4") and r.get("e5"):
            out[int(r["env_step"])] = r
    return out


def largest_difference(a, b) -> float:
    """The largest absolute difference between two JSON values of the same shape (0.0 when equal; inf when their shapes,
    keys or missing values differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf
        return max((largest_difference(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        return math.inf if len(a) != len(b) else max((largest_difference(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return 0.0 if a == b else math.inf
    return abs(float(a) - float(b))


class RunReplay:
    """The run's replay, truncated on demand to the steps stored at a checkpoint (with the run's 12-assignment tables)."""

    def __init__(self, run_dir: Path, tables: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device):
        self.run_dir, self.tables, self.device = run_dir, tables, device
        self._compact, self.source, self.until_step = None, None, None

    def _load(self, step: int) -> None:
        resume = self.run_dir / "checkpoints" / "resume.pt"
        if resume.is_file():
            state, replay, _ = load_for_check(resume)
            if int(state["progress"]["env_step"]) >= step:
                if not np.array_equal(replay._table, self.tables[0]):
                    raise SystemExit(f"{resume}: its replay table is not the run's 12 assignments' (rebuilt from its seed)")
                self._compact, self.source, self.until_step = replay.export_compact(), "checkpoints/resume.pt", int(state["progress"]["env_step"])
                return
        final = self.run_dir / "replay_final.npz"
        if final.is_file():
            self._compact, self.source, self.until_step = dict(np.load(final)), "replay_final.npz", None
            return
        raise SystemExit(f"{self.run_dir}: no replay holding step {step} (resume.pt before it, no replay_final.npz)")

    def at(self, step: int, stored_steps: int) -> ReplayBuffer:
        if self._compact is None or (self.until_step is not None and self.until_step < step):
            self._load(step)
        if stored_steps > len(self._compact["s"]):
            raise SystemExit(f"{self.source} holds {len(self._compact['s'])} steps, fewer than the {stored_steps} stored at {step}")
        buf = ReplayBuffer.from_compact({k: v[:stored_steps] for k, v in self._compact.items()}, *self.tables)
        return buf.to_device(self.device) if self.device.type == "cuda" else buf


def recompute(run_dir: Path, steps: list[int] | None, device_name: str, log=print) -> list[dict]:
    """The recomputed lines of run_dir at `steps` (default: every checkpoint of a continuation), merged into its
    checkpoints_recomputed.jsonl (written atomically). Returns the new lines."""
    run_dir = Path(run_dir)
    cfg = e5_setup.load_config(run_dir / "config.yaml")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    if cfg["e5"]["run"]["training"] != "shuffle":
        raise SystemExit(f"{run_dir}: not a shuffle run")
    cfg["runtime"]["device"] = device_name
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = configure_runtime(cfg)
    conf, condition, hooks, _ = run_e5.build(cfg)
    seed, fraction = int(meta["seed"]), float(cfg["e5"]["run"]["fraction"])
    canonical = canonical_assignment(conf["diagonal"])
    records = records_by_step(run_dir)
    if steps is None:
        if not meta.get("branched_from"):
            raise SystemExit(f"{run_dir}: not a continuation -- give --steps")
        steps = sorted(records)
    world = hooks.build_world(cfg, seed, condition)
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    worlds = run_e5.assignment_worlds(world, conf, condition, cfg, seed)
    source = ZoneAssignmentShuffle([worlds[k] for k in range(len(ASSIGNMENTS))], canonical, fraction, assignment_generator(seed))
    replays = RunReplay(run_dir, (source.table, source.background_table, source.background_of_row), device)
    rssm = build_rssm(cfg).to(device)
    ck, types = cfg["e4"]["checkpoint"], type_of_row()
    lines = []
    for step in steps:
        if step not in records:
            raise SystemExit(f"{run_dir}: no record at {step}")
        rec, t0 = records[step], time.perf_counter()
        ks = [int(k) for k in rec["e5"]["assignments_measured"]]
        replay = replays.at(step, int(rec["stored_steps"]))
        rssm.load_state_dict(torch.load(run_dir / "checkpoints" / f"agent_env{step:06d}.pt", map_location=device, weights_only=False)["rssm"])
        rssm.eval()
        rc = real_context_by_assignment(rssm, replay, {k: worlds[k] for k in ks}, cfg, seed, condition, step, ck["real_context_max_per_state"],
                                        ck["real_context_draws"], e5_setup.SLOT_ASSIGNMENT_REAL_CONTEXT)
        rc = _jsonable(rc)
        rc = {int(k): v for k, v in json.loads(json.dumps(rc)).items()}  # as the record stores it
        recorded = {k: rec["e5"]["by_assignment"][str(k)]["real_context"] for k in ks}
        unaffected = max([largest_difference(rc[k], recorded[k]) for k in ks if k != canonical]
                         + [largest_difference({f: v for f, v in rc[canonical].items() if f not in WORLD_FIELDS},
                                               {f: v for f, v in recorded[canonical].items() if f not in WORLD_FIELDS})], default=0.0)
        visits = largest_difference(json.loads(json.dumps(_jsonable(visit_shares(replay.obs_rows_and_flags()[0], types)))),
                                    rec["e5"]["visits_by_type"])
        worst = max(unaffected, visits)
        e5_out = {"by_assignment": {str(canonical): {"real_context": rc[canonical]}}}
        if len(ks) > 1:
            e5_out["mean_over_assignments"] = json.loads(json.dumps(_jsonable(
                {key: mean_over_assignments(rc, key) for key in ("mass_true_by_type", "I_s_by_type")})))
        line = {"env_step": step, "reproduced": bool(worst <= TOLERANCE), "largest_difference_unaffected": worst,
                "largest_difference_canonical_mass": largest_difference(rc[canonical]["mass_true_by_region"],
                                                                        recorded[canonical]["mass_true_by_region"]),
                "canonical_mass_recorded": recorded[canonical]["mass_true_by_region"],
                "e4": {"real_context": rc[canonical]}, "e5": e5_out, "assignments_measured": ks, "stored_steps": int(rec["stored_steps"]),
                "replay_source": replays.source, "device": str(device),
                "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": time.perf_counter() - t0}
        lines.append(line)
        q = ("nw", "ne", "se", "sw")
        log(f"{run_dir.name} {step}: {'REPRODUCED' if line['reproduced'] else 'NOT REPRODUCED'} (largest difference on the "
            f"fields independent of the world {worst:.2e}); canonical mass by quadrant recorded "
            f"{[recorded[canonical]['mass_true_by_region'][x] for x in q]} -> recomputed {[rc[canonical]['mass_true_by_region'][x] for x in q]} "
            f"({len(ks)} assignment(s), {line['seconds']:.0f} s)")
    out = run_dir / OUT_NAME
    merged = {}
    if out.is_file():
        merged = {int(x["env_step"]): x for x in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if x}
    merged.update({x["env_step"]: x for x in lines})
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(merged[s]) + "\n" for s in sorted(merged)), encoding="utf-8")
    os.replace(tmp, out)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="shuffle run directories")
    parser.add_argument("--steps", type=int, nargs="+", default=None, help="checkpoints (default: every checkpoint of a continuation)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    bad = 0
    for run in args.runs:
        bad += sum(not x["reproduced"] for x in recompute(run, args.steps, args.device, log=lambda s: print(s, flush=True)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
