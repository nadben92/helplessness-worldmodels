"""Preflight of a measurement batch (submit_kubeflow.MEASURE_BATCHES; docs/e3/DECISIONS.md,
2026-10-01 (10)): the existing runs are measured again by recompute_measures.py, read in place,
with no training. Run on the GPU node by `submit_kubeflow.py --phase preflight --batch <batch>`.

Checks, each a PROBLEM (never a warning) when it fails:
1. Device: CUDA available; after configure_runtime with the runs' own config, TF32 is off
   (matmul and cuDNN) and the deterministic algorithms are on, as in the runs that are re-measured.
2. CPUs: the CPUs requested for the job (--cpus) are the batch's prescribed number
   (submit_kubeflow.BATCH_CPUS) AND the container can use exactly that many (cgroup quota /
   affinity, preflight.effective_cpus).
3. Code version: the checkout's source_tree_hash is the one RUN_GPU.md states.
4. Every run: its directory exists under the batch's source root with config.yaml, meta.yaml,
   world.json, replay.npz and checkpoints/weights_final.pt; its name, its meta.yaml (seed,
   condition) and its config (window shape) are the job's (win<T>, seed, condition); the
   checkpoint's SHA-256 is the expected one.
5. The measurement itself, on the GPU: the first selected run is recomputed in full (its own K_s)
   into <out-dir>/smoke/ and timed; its outputs must be complete and record TF32 off. Its
   comparison with the run's original arrays (comparison_with_run.json) is reported, not judged.
Projection: the smoke run's duration x the number of rounds (runs / --parallel) x 1.25 (several
processes share the GPU and the CPUs). Verdict file <out-dir>/preflight_verdict.json and a last
line "PREFLIGHT DECISION: launch | stop" (launch: no problem and projection <= --max-batch-hours).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import yaml

from helplessness.experiment_loader import load_experiment_module
from helplessness.runs import source_tree_hash

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "preflight_measures_run_mini")
cluster = load_experiment_module("E3_worldmodel_4zones", "submit_kubeflow.py", "preflight_measures_cluster")
preflight = load_experiment_module("E3_worldmodel_4zones", "preflight.py", "preflight_measures_preflight")
recompute = load_experiment_module("E3_worldmodel_4zones", "recompute_measures.py", "preflight_measures_recompute")

HERE = Path(__file__).resolve().parent
RUN_GPU = HERE / "RUN_GPU.md"
REQUIRED_FILES = ("config.yaml", "meta.yaml", "world.json", "replay.npz", "checkpoints/weights_final.pt")
# a closed-loop run re-measured at its checkpoints (kind "checkpoints", remeasure_checkpoints.py)
REQUIRED_FILES_CHECKPOINTS = ("config.yaml", "meta.yaml", "checkpoints.jsonl", "counts.npz", "checkpoints/resume.pt")
# a closed-loop run's final weights measured repeatedly (kind "noise", measure_noise.py)
REQUIRED_FILES_NOISE = ("config.yaml", "meta.yaml", "checkpoints.jsonl", "checkpoints/resume.pt", "checkpoints/agent_final.pt")
CONTENTION = 1.25
ROLE = {"experimental": "experimental", "reference": "control"}


def stated_hash(doc: Path = RUN_GPU) -> str | None:
    """The source hash RUN_GPU.md says the checkout must have (the line after "must print exactly:")."""
    text = doc.read_text(encoding="utf-8") if doc.exists() else ""
    tail = text.split("must print exactly:")[1] if "must print exactly:" in text else ""
    m = re.search(r"'source_sha256': '([0-9a-f]{64})'", tail)
    return m.group(1) if m else None


def parse_job(job: str) -> tuple[int, int, str]:
    """win64_seed5_reference -> (64, 5, "reference")."""
    m = re.fullmatch(r"win(\d+)_seed(\d+)_(experimental|reference)", job)
    if not m:
        raise ValueError(f"job name {job!r} is not win<T>_seed<N>_<condition>")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def check_run(job: str, run_dir: Path, expected_sha256: str | None, kind: str = "final", env_steps: list[int] | None = None) -> list[str]:
    """Check 4 for one run (no measurement): presence, identity (name, meta.yaml, config), weights checksum (when known).
    kind "checkpoints": a closed-loop run, its resume point, counts and the weights of every requested checkpoint."""
    problems = []
    if not run_dir.is_dir():
        return [f"{job}: {run_dir} does not exist"]
    required = {"final": REQUIRED_FILES, "checkpoints": REQUIRED_FILES_CHECKPOINTS, "noise": REQUIRED_FILES_NOISE}[kind]
    missing = [f for f in required if not (run_dir / f).is_file()]
    if kind == "checkpoints":
        meta0 = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) if (run_dir / "meta.yaml").is_file() else {}
        for c in env_steps or []:
            if not (run_dir / "checkpoints" / f"agent_env{c:06d}.pt").is_file() and not (
                    c == int((meta0 or {}).get("env_steps", -1)) and (run_dir / "checkpoints" / "agent_final.pt").is_file()):
                missing.append(f"checkpoints/agent_env{c:06d}.pt")
    if missing:
        return [f"{job}: {run_dir.name} lacks {missing}"]
    window, seed, condition = parse_job(job)
    name = run_dir.name
    name_ok = f"__pair-{ROLE[condition]}__" in name and (f"_win{window}__seed{seed:02d}__" in name if kind == "final" else f"__seed{seed:02d}__" in name)
    if not name_ok:
        problems.append(f"{job}: directory name {name} is not window {window}, seed {seed}, {condition}")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    if int(meta.get("seed", -1)) != seed or meta.get("condition", "experimental") != condition:
        problems.append(f"{job}: meta.yaml says seed {meta.get('seed')}, {meta.get('condition')}")
    if meta.get("status") != "done":
        problems.append(f"{job}: meta.yaml status {meta.get('status')!r}, not done")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if int(cfg["train"]["seq_len"]) != window:
        problems.append(f"{job}: config.yaml seq_len {cfg['train']['seq_len']}, not {window}")
    if expected_sha256 is not None:
        found = recompute.file_sha256(run_dir / "checkpoints" / "weights_final.pt")
        if found != expected_sha256:
            problems.append(f"{job}: weights_final.pt SHA-256 {found}, expected {expected_sha256}")
    return problems


def cpu_problems(requested: int | None, prescribed: int, cpus: dict) -> list[str]:
    """Check 2: requested == prescribed == usable, explicitly."""
    problems = []
    if requested != prescribed:
        problems.append(f"CPUs requested {requested}, the batch is prescribed {prescribed}")
    if cpus.get("usable") != prescribed:
        problems.append(f"CPUs usable in the container {cpus.get('usable')} ({cpus}), the batch is prescribed {prescribed}")
    return problems


def decide(problems: list[str], hours: float | None, max_hours: float) -> str:
    return "launch" if not problems and hours is not None and math.isfinite(hours) and hours <= max_hours else "stop"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(cluster.MEASURE_BATCHES), required=True)
    parser.add_argument("--runs", nargs="+", default=None, help="the batch's runs to check (default: all)")
    parser.add_argument("--parallel", type=int, default=4, help="processes sharing the GPU in the launch")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs requested for the job (submit_kubeflow.py --cpu)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-batch-hours", type=float, default=4.0)
    parser.add_argument("--skip-smoke", action="store_true", help=argparse.SUPPRESS)  # tests only
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    batch = cluster.MEASURE_BATCHES[args.batch]
    jobs = cluster.select_runs(args.runs, args.batch)
    problems: list[str] = []
    summary: dict = {"batch": args.batch, "runs": jobs, "source_root": batch["source_root"], "parallel": args.parallel}

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        problems.append(f"--device cuda but CUDA is unavailable (torch {torch.__version__})")
    summary["cpus"] = {"requested": args.cpus, **preflight.effective_cpus()}
    problems += cpu_problems(args.cpus, cluster.BATCH_CPUS[args.batch], summary["cpus"])
    found = source_tree_hash(run_mini.REPO_ROOT)
    summary["code"] = {"found": found, "stated_in_RUN_GPU": stated_hash()}
    if found["source_sha256"] != summary["code"]["stated_in_RUN_GPU"]:
        problems.append(f"source hash {found['source_sha256']} is not the one RUN_GPU.md states ({summary['code']['stated_in_RUN_GPU']})")
    for job in jobs:
        run_name, sha = batch["runs"][job]
        problems += check_run(job, Path(batch["source_root"]) / run_name, sha, batch.get("kind", "final"), batch.get("env_steps"))
    if batch.get("goal_control"):  # the goal control's checkpoint (2026-10-06 (26))
        job, c, _ = batch["goal_control"]
        w = Path(batch["source_root"]) / batch["runs"][job][0] / "checkpoints" / f"agent_env{c:06d}.pt"
        if not w.is_file():
            problems.append(f"goal control: {w} is missing")
    for job, c in batch.get("value_diagnostics", []):  # the value diagnostic's checkpoints (2026-10-05 (24))
        w = Path(batch["source_root"]) / batch["runs"][job][0] / "checkpoints" / f"agent_env{c:06d}.pt"
        if not w.is_file():
            problems.append(f"value diagnostic: {w} is missing")
    print(f"[preflight-measures] {json.dumps(run_mini._jsonable(summary))}", flush=True)
    for p in problems:
        print(f"[preflight-measures] PROBLEM: {p}", flush=True)

    hours = None
    first = Path(batch["source_root"]) / batch["runs"][jobs[0]][0]
    if not problems and not args.skip_smoke:
        cfg = run_mini.load_mini_config(first / "config.yaml")
        cfg["runtime"]["device"] = args.device
        device = run_mini.configure_runtime(cfg)
        flags = run_mini.environment_metadata(device)
        summary["runtime"] = {k: flags[k] for k in ("device", "gpu_name", "allow_tf32_matmul", "allow_tf32_cudnn", "deterministic_algorithms")}
        if flags["allow_tf32_matmul"] or flags["allow_tf32_cudnn"] or not flags["deterministic_algorithms"]:
            problems.append(f"runtime not as in the measured runs (TF32 off, deterministic on): {summary['runtime']}")
        print(f"[preflight-measures] smoke: full recomputation of {jobs[0]} on {device} ...", flush=True)
        t0 = time.perf_counter()
        try:
            if batch.get("kind") == "noise":  # one repeat at the largest multiplier, an upper bound for every measurement of a job
                mn = load_experiment_module("E3_worldmodel_4zones", "measure_noise.py", "preflight_measures_noise")
                smoke = mn.noise(first, 1, [max(batch["contexts_multipliers"])], args.device)
                seconds = (time.perf_counter() - t0) * batch["repeats"] * len(batch["contexts_multipliers"])
                out = None
                rep = smoke["reproducibility"]
                if rep.get("allow_tf32_matmul") or rep.get("allow_tf32_cudnn"):
                    problems.append(f"smoke measurement ran with TF32 on: {rep}")
                summary["smoke"] = {"seconds": seconds, "one_repeat_at_the_largest_multiplier": smoke["by_multiplier"]}
            elif batch.get("kind") == "checkpoints":  # one checkpoint of the first run; the job measures len(env_steps)
                remeasure = load_experiment_module("E3_worldmodel_4zones", "remeasure_checkpoints.py", "preflight_measures_remeasure")
                out = remeasure.remeasure(first, batch["env_steps"][:1], args.device, args.out_dir / "smoke")[0]
                seconds = (time.perf_counter() - t0) * len(batch["env_steps"])
            else:
                out = recompute.recompute(first, ["final"], args.device, out_dir=args.out_dir / "smoke",
                                          weights_sha256=batch["runs"][jobs[0]][1])[0]
                seconds = time.perf_counter() - t0
            if out is not None:
                missing = [f for f in ("results.json", "arrays.npz", "SUMMARY.md") if not (out / f).is_file()]
                if missing:
                    problems.append(f"smoke recomputation wrote no {missing}")
                res = json.loads((out / "results.json").read_text(encoding="utf-8"))
                rep = res.get("reproducibility", {})
                if rep.get("allow_tf32_matmul") or rep.get("allow_tf32_cudnn"):
                    problems.append(f"smoke recomputation ran with TF32 on: {rep}")
                comparison = out / "comparison_with_run.json"
                summary["smoke"] = {"seconds": seconds, "out": str(out),
                                    "identical_to_original": json.loads(comparison.read_text()) if comparison.exists() else None}
            rounds = math.ceil(len(jobs) / max(1, args.parallel))
            hours = seconds * rounds * CONTENTION / 3600.0
            print(f"[preflight-measures] smoke: {seconds:.0f} s; identical to the run's original arrays: "
                  f"{summary['smoke'].get('identical_to_original')} (reported only: the aggregation compares them)", flush=True)
        except Exception as err:  # the measurement fails on this machine: a problem, reported with its message
            problems.append(f"smoke recomputation failed: {type(err).__name__}: {err}")
    summary["projected_batch_hours"] = hours
    max_hours = batch.get("max_batch_hours", args.max_batch_hours)  # a batch may carry its own ceiling (gpu_pilot_noise: 8 h)
    decision = decide(problems, hours, max_hours)
    verdict = {"passed": not problems, "problems": problems, "warnings": [], "parallel": args.parallel, "cpus": summary["cpus"],
               "projected_batch_hours": hours, "max_batch_hours": max_hours, "decision": decision, "summary": summary}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preflight_verdict.json").write_text(json.dumps(run_mini._jsonable(verdict), indent=1), encoding="utf-8")
    print("PREFLIGHT PASSED" if not problems else "PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems), flush=True)
    if hours is not None:
        print(f"[preflight-measures] projected wall time of the batch: {hours:.2f} h ({len(jobs)} runs, {args.parallel} at a time)", flush=True)
    print(f"[preflight-measures] CPUs: {args.cpus} requested, {summary['cpus']['usable']} usable", flush=True)
    print(f"PREFLIGHT DECISION: {decision}  (rule of RUN_GPU.md step 3; details in {args.out_dir / 'preflight_verdict.json'})", flush=True)
    return 0 if decision == "launch" else 1


if __name__ == "__main__":
    sys.exit(main())
