"""Preflight of a closed-loop agent batch (submit_kubeflow.AGENT_BATCHES; docs/e3/DECISIONS.md, 2026-10-02 (19)),
run on the GPU node by `submit_kubeflow.py --phase preflight --batch <batch>`.

Checks, each a PROBLEM when it fails (decision "stop"):
1. Device: CUDA available; after configure_runtime with the batch's config, TF32 off and deterministic algorithms on.
2. CPUs: requested = the batch's prescribed number (submit_kubeflow.BATCH_CPUS) = usable in the container.
3. Code version: the checkout's source_tree_hash is the one RUN_GPU.md states.
4. Execution mode: no runtime.compile, no stacked seeds (preflight.execution_mode_problems; the pilot runs plain).
5. The real entry point, truncated: run_agent.py, seed 0, experimental, --max-env-steps (default 1,100: one checkpoint
   at 1,000 and the end), into <out-dir>/truncated/; its outputs must be complete (meta.yaml done on cuda, weights at
   the checkpoint, agent_final.pt, results.json and arrays.npz from the final measures, the checkpoint goal
   diagnostics in checkpoints.jsonl).
Informative, never in the decision:
6. The speed bench (bench_agent.py: world-model, heads and environment steps, stacked seeds 1-8, torch.compile behind
   its equivalence gate), --bench-minutes (default 30), into <out-dir>/bench_agent.json.
Parallelism (decides the launch's --max-parallel): the closed loop's inner loop run for --loop-seconds in 2, then 4
processes side by side (bench_agent.py --loop); 4 is chosen if the 4 processes' total environment steps per second is
at least the 2 processes' total, else 2 (the supervisor starts the jobs in order: the two experimental runs first).
Projection: the planned environment steps / the per-process rate at the chosen parallelism, x 2 rounds when 2 run at
once. The verdict (<out-dir>/preflight_verdict.json) ends with "PREFLIGHT DECISION: launch-max-parallel-4 |
launch-max-parallel-2 | stop"; stop also when the projection exceeds --max-batch-hours (default 336).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from helplessness.experiment_loader import load_experiment_module
from helplessness.runs import source_tree_hash

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "preflight_agent_run_mini")
cluster = load_experiment_module("E3_worldmodel_4zones", "submit_kubeflow.py", "preflight_agent_cluster")
preflight = load_experiment_module("E3_worldmodel_4zones", "preflight.py", "preflight_agent_preflight")
pm = load_experiment_module("E3_worldmodel_4zones", "preflight_measures.py", "preflight_agent_measures")

HERE = Path(__file__).resolve().parent
OUTPUTS = ("config.yaml", "meta.yaml", "console.log", "metrics.csv", "checkpoints.jsonl", "results.json", "arrays.npz", "counts.npz",
           "checkpoints/agent_final.pt", "checkpoints/agent_env001000.pt")


def check_truncated(run_dir: Path) -> list[str]:
    problems = [f"truncated run: missing {f}" for f in OUTPUTS if not (run_dir / f).is_file()]
    if (run_dir / "meta.yaml").is_file():
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
        if meta.get("status") != "done":
            problems.append(f"truncated run: meta.yaml status {meta.get('status')!r}")
        if not str(meta.get("device", "")).startswith("cuda"):
            problems.append(f"truncated run: device {meta.get('device')!r}")
    if (run_dir / "checkpoints.jsonl").is_file():
        records = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
        if not records or "goal_diagnostics" not in records[0]:
            problems.append("truncated run: no goal_diagnostics at the checkpoints")
    return problems


def choose_parallel(total_2: float, total_4: float) -> int:
    """4 processes if they make at least as many environment steps per second in total as 2, else 2."""
    return 4 if total_4 >= total_2 else 2


def projected_hours(planned_env_steps: int, per_process_rate: float, parallel: int, n_runs: int) -> float:
    rounds = math.ceil(n_runs / parallel)
    return planned_env_steps / per_process_rate * rounds / 3600.0 if per_process_rate > 0 else float("inf")


def decide(problems: list[str], hours: float | None, max_hours: float, parallel: int | None) -> str:
    if problems or hours is None or not math.isfinite(hours) or hours > max_hours or parallel not in (2, 4):
        return "stop"
    return f"launch-max-parallel-{parallel}"


def _loops(config: Path, n: int, seconds: float, out: Path) -> dict:
    procs = [subprocess.Popen([sys.executable, str(HERE / "bench_agent.py"), "--config", str(config), "--device", "cuda", "--loop", str(seconds)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(n)]
    rates, errors = [], []
    for p in procs:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            errors.append((stderr.strip().splitlines() or ["?"])[-1])
        else:
            rates.append(json.loads(stdout.strip().splitlines()[-1])["env_steps_per_s"])
    return {"processes": n, "per_process": rates, "total": float(sum(rates)), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(cluster.AGENT_BATCHES), required=True)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-env-steps", type=int, default=1100)
    parser.add_argument("--bench-minutes", type=float, default=30.0)
    parser.add_argument("--loop-seconds", type=float, default=120.0)
    parser.add_argument("--max-batch-hours", type=float, default=336.0)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    batch = cluster.AGENT_BATCHES[args.batch]
    config = run_mini.REPO_ROOT / batch["config"]
    cfg = run_mini.load_mini_config(config)
    problems: list[str] = []
    summary: dict = {"batch": args.batch, "config": batch["config"], "runs": list(batch["runs"])}

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        problems.append(f"--device cuda but CUDA is unavailable (torch {torch.__version__})")
    summary["cpus"] = {"requested": args.cpus, **preflight.effective_cpus()}
    problems += pm.cpu_problems(args.cpus, cluster.BATCH_CPUS[args.batch], summary["cpus"])
    found = source_tree_hash(run_mini.REPO_ROOT)
    summary["code"] = {"found": found, "stated_in_RUN_GPU": pm.stated_hash()}
    if found["source_sha256"] != summary["code"]["stated_in_RUN_GPU"]:
        problems.append(f"source hash {found['source_sha256']} is not the one RUN_GPU.md states ({summary['code']['stated_in_RUN_GPU']})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hours, parallel = None, None
    if not problems:
        cfg["runtime"]["device"] = args.device
        device = run_mini.configure_runtime(cfg)
        flags = run_mini.environment_metadata(device)
        if flags["allow_tf32_matmul"] or flags["allow_tf32_cudnn"] or not flags["deterministic_algorithms"]:
            problems.append(f"runtime: TF32 on or determinism off ({flags})")
        mode_problems, _ = preflight.execution_mode_problems(cfg, device, {config.name: cfg})
        problems += mode_problems
        # 6. the speed bench (informative)
        print("[preflight-agent] speed bench (informative) ...", flush=True)
        bench_mod = load_experiment_module("E3_worldmodel_4zones", "bench_agent.py", "preflight_agent_bench")
        try:
            bench = run_mini._jsonable(bench_mod.bench(cfg, device, budget_minutes=args.bench_minutes))
        except Exception as err:
            bench = {"error": f"{type(err).__name__}: {err}"}
        (args.out_dir / "bench_agent.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
        summary["bench"] = bench
        print(f"[preflight-agent] bench: {json.dumps(bench)[:2000]}", flush=True)
        # 5. the truncated real run
        print(f"[preflight-agent] truncated run: run_agent.py --max-env-steps {args.max_env_steps} ...", flush=True)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(HERE / "run_agent.py"), "--config", str(config), "--device", args.device, "--seeds", "0",
                        "--conditions", "experimental", "--max-env-steps", str(args.max_env_steps), "--out-dir", str(args.out_dir / "truncated")],
                       check=False)
        summary["truncated_run_seconds"] = time.perf_counter() - t0
        runs = sorted((args.out_dir / "truncated" / run_mini.EXPERIMENT_ID).glob("*__seed00__*"))
        problems += check_truncated(runs[-1]) if runs else ["the truncated run wrote no run directory"]
        # parallelism and projection
        if not problems and args.device == "cuda":
            print("[preflight-agent] parallelism: 2 then 4 processes side by side ...", flush=True)
            two, four = _loops(config, 2, args.loop_seconds, args.out_dir), _loops(config, 4, args.loop_seconds, args.out_dir)
            summary["parallel"] = {"2": two, "4": four}
            if two["errors"] or four["errors"]:
                problems.append(f"parallel loop failed: {two['errors'] + four['errors']}")
            else:
                parallel = choose_parallel(two["total"], four["total"])
                rate = min((four if parallel == 4 else two)["per_process"])
                planned = cfg["budget"]["total_steps"] + cfg["budget"]["prefill_steps"]
                hours = projected_hours(planned, rate, parallel, len(batch["runs"]))
                summary["projection"] = {"max_parallel": parallel, "slowest_process_env_steps_per_s": rate, "planned_env_steps": planned,
                                         "projected_batch_hours": hours}
                print(f"[preflight-agent] env steps/s in total: 2 processes {two['total']:.2f}, 4 processes {four['total']:.2f} -> "
                      f"--max-parallel {parallel}; projected batch {hours:.1f} h", flush=True)
    decision = decide(problems, hours, args.max_batch_hours, parallel)
    verdict = {"passed": not problems, "problems": problems, "warnings": [], "parallel": parallel, "cpus": summary["cpus"],
               "projected_batch_hours": hours, "max_batch_hours": args.max_batch_hours, "decision": decision, "summary": summary}
    (args.out_dir / "preflight_verdict.json").write_text(json.dumps(run_mini._jsonable(verdict), indent=1), encoding="utf-8")
    print("PREFLIGHT PASSED" if not problems else "PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems), flush=True)
    print(f"[preflight-agent] CPUs: {args.cpus} requested, {summary['cpus']['usable']} usable", flush=True)
    print(f"PREFLIGHT DECISION: {decision}  (rule of RUN_GPU.md step 3; details in {args.out_dir / 'preflight_verdict.json'})", flush=True)
    return 0 if decision != "stop" else 1


if __name__ == "__main__":
    sys.exit(main())
