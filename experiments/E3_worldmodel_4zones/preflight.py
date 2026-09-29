"""Short check to run on the GPU machine BEFORE launching the runs of a
config (a few minutes). One command:

    python experiments/E3_worldmodel_4zones/preflight.py --config configs/e3/gpu_e30.yaml         --extra-config configs/e3/gpu_e30_bridge.yaml --device cuda

1. Device: CUDA present, the model's parameters, the pre-rendered table and
   a training minibatch all on the GPU; TF32 off; determinism flags.
2. Determinism: the same 5 training steps run twice from the same weights
   and generator give bit-identical losses and weights (GPU determinism
   policy, docs/e3/DECISIONS.md 2026-09-28).
3. Throughput of the real training step at the config's size (both
   estimator phases), alone and with --parallel processes side by side
   (default 5, the number of runs launched together: the --extra-config
   runs, one process each, and --config for the rest), peak GPU memory,
   and the projected duration of one run of each config.
3b. Memory with the same --parallel processes (docs/e3/DECISIONS.md, 2026-09-29):
   each trains the config's model a few steps, waits for the others, then
   takes one training-diagnostic point and one final measurement at the
   same time as the others; the peak GPU memory of each phase is printed.
   Fails if a process fails (out of memory) or if the processes' worst
   peaks (reserved) add up to more than 90% of the GPU's memory. A launch
   with a subset of runs or a smaller --max-parallel is tested with its
   own --parallel (submit_kubeflow.py --phase preflight builds it).
4. The real entry point, truncated: run_mini.py on seed 0 / experimental
   with --max-steps (default 200) into <out-dir>/preflight/, then checks
   that every output is written (meta.yaml says device cuda and status
   done, replay.npz, world.json, weights_final.pt, resume.pt, arrays,
   report), and recomputes the final measures from the directory with
   recompute_measures.py (identical numbers expected).
Prints PREFLIGHT PASSED or PREFLIGHT FAILED (exit code 1) and writes
preflight_summary.json next to the run. Always ends with the machine-readable
verdict (docs/e3/DECISIONS.md, 2026-09-30 (3)): <out-dir>/preflight_verdict.json
(passed, problems, memory_problems, warnings, parallel, projected_batch_hours,
decision) and a last line "PREFLIGHT DECISION: launch | retry-max-parallel-<N> |
stop", the launch rule of RUN_GPU.md step 3 (decide()).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.rssm import LatentEstimator
from helplessness.experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "preflight_run_mini")

HERE = Path(__file__).resolve().parent
TARGET_HOURS = 6.0
# The truncated run's model is barely trained, so its D1 accepts few contrasts; a trained model's D1
# in real contexts costs up to ~12 pairs x ~30 states x 2 x 50 contexts x ~8 ms: this allowance is added.
D1_ALLOWANCE_SECONDS = 900.0
# Memory test: replay steps (> 256 windows of 65 steps, one full block of the real-context and z-only probes)
# and the share of the GPU's memory left free when every process is at its peak.
MEMORY_PROBE_ENV_STEPS = 20_000
MEMORY_PROBE_KERNEL_DRAWS = 64
MEMORY_HEADROOM = 0.10


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(cfg: dict, device: torch.device, n_env_steps: int = 3000, with_counts: bool = False):
    world = run_mini.build_world(cfg, 0, "experimental")
    replay = run_mini.ReplayBuffer.for_run(budget=n_env_steps, prefill_steps=0, table=world.templates_np)
    streams = run_mini.seed_streams(0, device)
    N = run_mini.collect_and_count(world.env, replay, n_env_steps, streams["collect"], run_mini.N_STATES, run_mini.N_ACTIONS,
                                   streams["start_state"])
    if device.type == "cuda":
        replay.to_device(device)
    return (world, replay, N) if with_counts else (world, replay)


def bench(cfg: dict, device: torch.device, n_steps: int) -> dict:
    """Steps per second of the real training step (sync=False, as run_mini),
    in the probabilities phase and after the switch, plus peak memory."""
    world, replay = setup(cfg, device)
    torch.manual_seed(0)
    rssm = run_mini.build_rssm(cfg).to(device)
    opt = torch.optim.Adam(rssm.parameters(), lr=1e-4, eps=1e-8)
    loss_cfg = run_mini.WorldModelLossConfig(k=cfg["render"]["k"], w_patch=cfg["train"]["w_patch"])
    bg = world.background_t.to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    rng = np.random.default_rng(0)
    out = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for phase, est in (("probabilities", LatentEstimator("probs")), ("straight-through", LatentEstimator("straight_through"))):
        for _ in range(3):  # warm-up
            mb, _ = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], rng)
            run_mini.train_world_model_step(rssm, opt, mb, bg, loss_cfg, clip_norm=None, estimator=est, generator=gen, sync=False)
        _sync(device)
        t0 = time.perf_counter()
        for i in range(n_steps):
            mb, _ = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], rng)
            res = run_mini.train_world_model_step(rssm, opt, mb, bg, loss_cfg, clip_norm=None, estimator=est, generator=gen,
                                                  sync=False, return_posteriors=(i % 50 == 0))
        _sync(device)
        out[phase] = n_steps / (time.perf_counter() - t0)
        assert all(np.isfinite(float(v)) for k, v in res.items() if k != "post_probs"), "non-finite loss"
    out["peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
    return out


def _wait_barrier(barrier: Path | None, n: int, timeout_s: float = 3600.0) -> None:
    """File barrier: announce readiness, then wait until n processes have (no-op for barrier None)."""
    if barrier is None:
        return
    barrier.mkdir(parents=True, exist_ok=True)
    (barrier / f"ready_{os.getpid()}").touch()
    t0 = time.perf_counter()
    while len(list(barrier.glob("ready_*"))) < n:
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"memory barrier: {len(list(barrier.glob('ready_*')))}/{n} processes ready after {timeout_s:.0f}s")
        time.sleep(0.2)


def memory_probe(cfg: dict, device: torch.device, out_dir: Path, barrier: Path | None = None, n_barrier: int = 1,
                 n_env_steps: int = MEMORY_PROBE_ENV_STEPS, train_steps: int = 20) -> dict:
    """Peak GPU memory of what a run holds while it measures (docs/e3/DECISIONS.md, 2026-09-29): the
    config's model trained a few steps (weights, Adam state, the training step's own peak), then one
    training-diagnostic point (run_mini.training_diagnostics, M = iz_mc_samples_training) and one
    final measurement (run_mini.measure_final), each between run_mini.gpu_peak_start / gpu_peak_end as
    in a run. The replay holds MEMORY_PROBE_ENV_STEPS steps: more than one full block of the
    real-context and z-only probes (256 windows), so their block-size peak is reached. With a barrier,
    every process waits for the others before measuring, so the peaks of the side-by-side processes
    coincide (the first batch's out-of-memory errors happened in concurrent measurements)."""
    world, replay, N = setup(cfg, device, n_env_steps=n_env_steps, with_counts=True)
    streams = run_mini.seed_streams(0, device)
    torch.manual_seed(0)
    rssm = run_mini.build_rssm(cfg).to(device)
    opt = torch.optim.Adam(rssm.parameters(), lr=1e-4, eps=1e-8)
    loss_cfg = run_mini.WorldModelLossConfig(k=cfg["render"]["k"], w_patch=cfg["train"]["w_patch"])
    bg = world.background_t.to(device)
    world.templates_t, world.background_t = world.templates_t.to(device), bg
    gen = torch.Generator(device=device).manual_seed(0)
    rng = np.random.default_rng(0)
    run_mini.gpu_peak_start(device)
    for i in range(train_steps):
        mb, _ = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], rng)
        est = LatentEstimator("probs" if i < train_steps // 2 else "straight_through")
        run_mini.train_world_model_step(rssm, opt, mb, bg, loss_cfg, clip_norm=None, estimator=est, generator=gen, sync=False)
    out = {"training": run_mini.gpu_peak_end(device)}
    _wait_barrier(barrier, n_barrier)
    T_hat_B = run_mini.build_tabular_twin(N, run_mini.N_STATES, run_mini.N_ACTIONS)
    P_true = world.env.transition_tensor()
    d_rng, d_gen = run_mini.measure_streams(0, run_mini.SLOT_DIAG, streams)
    run_mini.gpu_peak_start(device)
    t0 = time.perf_counter()
    run_mini.training_diagnostics(rssm, world, P_true, T_hat_B, replay, cfg, d_rng, d_gen, cfg["measure"]["iz_mc_samples_training"],
                                  final=False)
    out["diagnostic_point"] = run_mini.gpu_peak_end(device)
    out["diagnostic_point_seconds"] = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    # The kernels are decoded MEMORY_PROBE_KERNEL_DRAWS draws per (s, a) here instead of the final K: their peak is set by the
    # decoding chunk (1,024 images), not by K, and the full K (CPU template matching) would make the test long. The final
    # measure's duration is measured by the truncated real run (step 4) at the config's own K.
    cfg_final = copy.deepcopy(cfg)
    cfg_final["measure"]["kernel_k_s"] = cfg_final["measure"]["real_context"]["kernel_draws"] = MEMORY_PROBE_KERNEL_DRAWS
    t0 = time.perf_counter()
    fields, _ = run_mini.measure_final(rssm, world, replay, N, cfg_final, streams, out_dir, run_mini.RunLog(None, "[memory probe]"),
                                       train_steps)
    out["final_measures"] = fields["gpu_memory_final"]
    out["final_measures_seconds"] = time.perf_counter() - t0
    out["total_mib"] = torch.cuda.get_device_properties(device).total_memory / 2**20 if device.type == "cuda" else None
    return out


def memory_verdict(results: list[dict], total_mib: float, headroom: float = MEMORY_HEADROOM) -> dict:
    """Whether the side-by-side processes fit: each process's worst peak (reserved by its caching
    allocator: training, diagnostic point or final measures), summed over the processes, must stay
    under (1 - headroom) of the GPU's memory -- the case where every process is at its peak at once."""
    worst = [max(r[k]["peak_reserved_mib"] for k in ("training", "diagnostic_point", "final_measures")) for r in results]
    budget = (1.0 - headroom) * total_mib
    return {"per_process_worst_peak_reserved_mib": worst, "sum_mib": float(sum(worst)), "budget_mib": budget,
            "total_mib": total_mib, "fits": bool(sum(worst) <= budget)}


def determinism_check(cfg: dict, device: torch.device, n_steps: int = 5) -> dict:
    world, replay = setup(cfg, device, n_env_steps=1000)
    bg = world.background_t.to(device)
    loss_cfg = run_mini.WorldModelLossConfig(k=cfg["render"]["k"], w_patch=cfg["train"]["w_patch"])
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        rssm = run_mini.build_rssm(cfg).to(device)
        opt = torch.optim.Adam(rssm.parameters(), lr=1e-3, eps=1e-8)
        gen = torch.Generator(device=device).manual_seed(0)
        rng = np.random.default_rng(0)
        losses = []
        for i in range(n_steps):
            mb, _ = replay.sample(cfg["train"]["batch_size"], cfg["train"]["seq_len"], rng)
            est = LatentEstimator("probs" if i < n_steps // 2 else "straight_through")
            losses.append(run_mini.train_world_model_step(rssm, opt, mb, bg, loss_cfg, clip_norm=None, estimator=est, generator=gen))
        runs.append((losses, [p.detach().cpu().clone() for p in rssm.parameters()]))
    same_losses = runs[0][0] == runs[1][0]
    same_weights = all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
    return {"bit_identical_losses": same_losses, "bit_identical_weights": same_weights}


def device_check(cfg: dict, device: torch.device) -> dict:
    world, replay = setup(cfg, device, n_env_steps=500)
    rssm = run_mini.build_rssm(cfg).to(device)
    mb, _ = replay.sample(2, 8, np.random.default_rng(0))
    on_device = {
        "parameters": all(p.device.type == device.type for p in rssm.parameters()),
        "replay_table": replay._device_table is None if device.type == "cpu" else replay._device_table.device.type == device.type,
        "minibatch_images": (not isinstance(mb.x, np.ndarray)) and mb.x.device.type == device.type if device.type == "cuda" else True,
    }
    return on_device


def effective_cpus() -> dict:
    """The CPUs this process can use: the container's cgroup quota when there is one (cgroup v2 cpu.max, or
    v1 cfs quota / period), the scheduler affinity, and os.cpu_count() (the node's, inside a container)."""
    import math

    quota = None
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            quota = int(text[0]) / int(text[1])
    except (OSError, ValueError, IndexError):
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            quota = q / period if q > 0 else None
        except (OSError, ValueError):
            pass
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    usable = [v for v in (quota, affinity) if v]
    return {"cgroup_quota": quota, "affinity": affinity, "cpu_count": os.cpu_count(),
            "usable": math.floor(min(usable) + 1e-9) if usable else os.cpu_count()}


def batch_hours(hours_per_run: list[float], parallel: int) -> float:
    """Wall time of a batch whose runs start as slots free up, at most `parallel` at once: the runs taken
    longest first, each into the slot that frees first (the supervisor's queue)."""
    slots = [0.0] * max(1, parallel)
    for h in sorted(hours_per_run, reverse=True):
        slots[slots.index(min(slots))] += h
    return max(slots)


def decide(verdict: dict, max_batch_hours: float, fallback_parallel: int) -> str:
    """RUN_GPU.md step 3 (docs/e3/DECISIONS.md, 2026-09-30 (3)):
      "launch"                    passed, no warning, projected_batch_hours finite and <= max_batch_hours;
      "retry-max-parallel-<N>"    the only problems are the memory test's, at a parallelism above N;
      "stop"                      anything else (another failure, a warning, a missing or longer projection)."""
    hours = verdict.get("projected_batch_hours")
    if verdict["passed"] and not verdict["warnings"]:
        return "launch" if hours is not None and np.isfinite(hours) and hours <= max_batch_hours else "stop"
    only_memory = verdict["memory_problems"] and len(verdict["memory_problems"]) == len(verdict["problems"])
    if only_memory and verdict["parallel"] > fallback_parallel:
        return f"retry-max-parallel-{fallback_parallel}"
    return "stop"


def write_verdict(out_dir: Path, problems: list[str], memory_problems: list[str], warnings: list[str], parallel: int,
                  hours: float | None, max_batch_hours: float, fallback_parallel: int) -> str:
    verdict = {"passed": not problems, "problems": problems, "memory_problems": memory_problems, "warnings": warnings,
               "parallel": parallel, "projected_batch_hours": hours, "max_batch_hours": max_batch_hours,
               "fallback_parallel": fallback_parallel}
    verdict["decision"] = decide(verdict, max_batch_hours, fallback_parallel)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_verdict.json").write_text(json.dumps(run_mini._jsonable(verdict), indent=1), encoding="utf-8")
    return verdict["decision"]


def projected_hours(cfg: dict, steps_per_s: float, diag_seconds: float, final_seconds: float) -> float:
    planned = int(round((cfg["budget"]["total_steps"] + cfg["budget"]["prefill_steps"]) * cfg["train"]["replay_ratio"]))
    n_points = len(run_mini.checkpoint_steps(cfg, planned)) if cfg["measure"].get("diagnostics_at_checkpoints") else 0
    return (planned / steps_per_s + n_points * diag_seconds + final_seconds + D1_ALLOWANCE_SECONDS) / 3600.0


def check_run_dir(run_dir: Path, device: torch.device) -> list[str]:
    problems = []
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    if meta.get("status") != "done":
        problems.append(f"meta.yaml status is {meta.get('status')!r}")
    if meta.get("device") != str(device) and not str(meta.get("device", "")).startswith(device.type):
        problems.append(f"meta.yaml device is {meta.get('device')!r}")
    if meta.get("allow_tf32_matmul") or meta.get("allow_tf32_cudnn"):
        problems.append("TF32 is on")
    for name in ("config.yaml", "world.json", "replay.npz", "console.log", "diagnostics.jsonl", "results.json", "arrays.npz",
                 "REPORT.md", "checkpoints/weights_final.pt", "checkpoints/resume.pt", "checkpoints/latest.json",
                 "artifacts/loss_curves.png", "artifacts/iz_by_zone.png"):
        if not (run_dir / name).is_file():
            problems.append(f"missing {name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=200, help="gradient steps of the truncated real run")
    parser.add_argument("--bench-steps", type=int, default=30)
    parser.add_argument("--parallel", type=int, default=5,
                        help="processes side by side for the throughput and memory tests: the launch's number of simultaneous runs")
    parser.add_argument("--extra-config", type=Path, nargs="*", default=[],
                        help="configs launched alongside (one process each in the parallel measure), e.g. the bridge run")
    parser.add_argument("--out-dir", type=Path, default=run_mini.REPO_ROOT / "results" / "preflight")
    parser.add_argument("--cpus", type=int, default=None,
                        help="CPUs requested for the job (submit_kubeflow.py --cpu): recorded, checked against the container's "
                             "limit, and stated with the projection (measured under this allocation)")
    parser.add_argument("--batch-runs", nargs="*", default=None, metavar="CONFIG=N",
                        help="the launch's runs per config (default: the --parallel mix); with more runs than --parallel, the "
                             "batch's wall time counts the queue")
    parser.add_argument("--max-batch-hours", type=float, default=20.0, help="launch rule: projected batch wall time at most this")
    parser.add_argument("--fallback-parallel", type=int, default=4, help="launch rule: parallelism to retry at after a memory failure")
    parser.add_argument("--bench-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--memory-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--barrier", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--barrier-n", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = run_mini.load_mini_config(args.config)
    cfg["runtime"]["device"] = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        problem = (f"--device cuda but torch.cuda.is_available() is False (torch {torch.__version__}, "
                   f"built for CUDA {torch.version.cuda})")
        print(f"PREFLIGHT FAILED: {problem}", flush=True)
        if not (args.bench_only or args.memory_only):
            decision = write_verdict(args.out_dir, [problem], [], [], args.parallel, None, args.max_batch_hours, args.fallback_parallel)
            print(f"PREFLIGHT DECISION: {decision}", flush=True)
        return 1
    device = run_mini.configure_runtime(cfg)
    if args.bench_only:
        print(json.dumps(bench(cfg, device, args.bench_steps)), flush=True)
        return 0
    if args.memory_only:
        out_dir = args.out_dir / "memory_probe" / f"pid{os.getpid()}"
        print(json.dumps(run_mini._jsonable(memory_probe(cfg, device, out_dir, args.barrier, args.barrier_n))), flush=True)
        return 0

    summary: dict = {"config": str(args.config), "device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    summary["cpus"] = {"requested": args.cpus, **effective_cpus()}
    print(f"[preflight] {summary}", flush=True)
    problems: list[str] = []
    memory_problems: list[str] = []
    warnings: list[str] = []
    if args.cpus is not None and summary["cpus"]["usable"] != args.cpus:
        warnings.append(f"{args.cpus} CPU(s) requested but {summary['cpus']['usable']} usable here ({summary['cpus']}): the "
                        "throughput and the projection are measured on the latter")
        print(f"[preflight] WARNING: {warnings[-1]}", flush=True)

    summary["on_device"] = device_check(cfg, device)
    print(f"[preflight] on {device}: {summary['on_device']}", flush=True)
    problems += [f"{k} not on {device}" for k, v in summary["on_device"].items() if not v]

    summary["determinism"] = determinism_check(cfg, device)
    print(f"[preflight] determinism (same 5 steps twice): {summary['determinism']}", flush=True)
    if not all(summary["determinism"].values()):
        problems.append("two identical runs of 5 steps differ (GPU determinism)")

    summary["throughput_single"] = bench(cfg, device, args.bench_steps)
    print(f"[preflight] throughput, one process: {summary['throughput_single']}", flush=True)
    configs = [args.config] * max(args.parallel - len(args.extra_config), 0) + list(args.extra_config)
    if len(configs) > 1:
        procs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(c), "--device", args.device,
                                   "--bench-only", "--bench-steps", str(args.bench_steps)], stdout=subprocess.PIPE, text=True)
                 for c in configs]
        outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
        per_config = {}
        for c, o in zip(configs, outs):
            per_config.setdefault(str(c), []).append(o)
        summary["throughput_parallel"] = {
            "n_processes": len(configs),
            "per_process": {c: {ph: float(np.mean([o[ph] for o in os_])) for ph in ("probabilities", "straight-through")}
                            for c, os_ in per_config.items()}}
        print(f"[preflight] throughput, {len(configs)} processes side by side: {summary['throughput_parallel']}", flush=True)

    # Memory with the planned number of processes: each trains a few steps, waits for the others, then
    # takes one diagnostic point and one final measurement at the same time as the others.
    barrier = args.out_dir / "memory_probe" / f"barrier_{os.getpid()}_{int(time.time())}"
    if device.type == "cuda":
        print(f"[preflight] memory: {len(configs)} process(es) side by side, diagnostic point + final measures at the same "
              "time ...", flush=True)
        procs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config", str(c), "--device", args.device,
                                   "--memory-only", "--barrier", str(barrier), "--barrier-n", str(len(configs)),
                                   "--out-dir", str(args.out_dir)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for c in configs]
    else:
        print("[preflight] memory test skipped (no GPU)", flush=True)
        procs = []
    mem_results, mem_failed = [], []
    for c, p in zip(configs, procs):
        stdout, stderr = p.communicate()  # the probe's JSON is the last line of its stdout
        if p.returncode != 0:
            mem_failed.append(f"{Path(c).name}: exit {p.returncode}: {(stderr.strip().splitlines() or ['?'])[-1]}")
        else:
            mem_results.append({"config": str(c), **json.loads(stdout.strip().splitlines()[-1])})
    summary["memory_parallel"] = {"n_processes": len(configs), "per_process": mem_results, "failed": mem_failed}
    for r in mem_results:
        peaks = {k: r[k] and round(r[k]["peak_reserved_mib"]) for k in ("training", "diagnostic_point", "final_measures")}
        print(f"[preflight] memory {Path(r['config']).name}: peak reserved MiB {peaks}; diagnostic point "
              f"{r['diagnostic_point_seconds']:.0f} s, final measures {r['final_measures_seconds']:.0f} s", flush=True)
    memory_problems += [f"memory test process failed (out of memory?): {m}" for m in mem_failed]
    if device.type == "cuda" and mem_results and not mem_failed:
        verdict = memory_verdict(mem_results, mem_results[0]["total_mib"])
        summary["memory_parallel"]["verdict"] = verdict
        print(f"[preflight] memory, {len(configs)} processes at their peaks: {verdict['sum_mib']:.0f} MiB of a "
              f"{verdict['budget_mib']:.0f} MiB budget ({1 - MEMORY_HEADROOM:.0%} of {verdict['total_mib']:.0f}): "
              f"{'fits' if verdict['fits'] else 'DOES NOT FIT'}", flush=True)
        if not verdict["fits"]:
            memory_problems.append(f"{len(configs)} processes at their measurement peaks need {verdict['sum_mib']:.0f} MiB, above "
                                   f"{verdict['budget_mib']:.0f} MiB: lower --parallel (launcher --max-parallel) or "
                                   "measure.max_images_per_forward")
    problems += memory_problems

    print(f"[preflight] truncated real run: run_mini.py --max-steps {args.max_steps} ...", flush=True)
    t0 = time.perf_counter()
    cmd = [sys.executable, str(HERE / "run_mini.py"), "--config", str(args.config), "--device", args.device, "--seeds", "0",
           "--conditions", "experimental", "--max-steps", str(args.max_steps), "--out-dir", str(args.out_dir)]
    subprocess.run(cmd, check=False)
    summary["truncated_run_seconds"] = time.perf_counter() - t0
    runs = sorted((args.out_dir / run_mini.EXPERIMENT_ID).glob("grid12x12__*seed00__*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        problems.append("the truncated run wrote no run directory")
    else:
        run_dir = runs[-1]
        summary["run_dir"] = str(run_dir)
        problems += check_run_dir(run_dir, device)
        diag = [json.loads(line) for line in (run_dir / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()] \
            if (run_dir / "diagnostics.jsonl").exists() else []
        final_seconds = diag[-1]["seconds"] if diag else float("nan")
        recompute = load_experiment_module("E3_worldmodel_4zones", "recompute_measures.py", "preflight_recompute")
        t1 = time.perf_counter()
        out = recompute.recompute(run_dir, ["final"], args.device)[0]
        summary["recompute_seconds"] = time.perf_counter() - t1
        same = json.loads((out / "comparison_with_run.json").read_text(encoding="utf-8"))
        summary["recompute_identical"] = same
        if not all(same.values()):
            problems.append(f"recomputed final measures differ from the run's: {same}")
        # Projection per config. The diagnostic-point and final-measure costs of the extra configs are taken
        # from --config's truncated run (the spec-size model: an upper bound for the smaller bridge network).
        summary["projected_hours_per_run"] = {}
        for c in dict.fromkeys([args.config, *args.extra_config]):
            rate = summary.get("throughput_parallel", {}).get("per_process", {}).get(str(c)) or summary["throughput_single"]
            slowest = min(rate["probabilities"], rate["straight-through"])
            hours = projected_hours(run_mini.load_mini_config(c), slowest, diag_seconds=final_seconds,
                                    final_seconds=summary["recompute_seconds"])
            summary["projected_hours_per_run"][str(c)] = hours
            n = summary.get("throughput_parallel", {}).get("n_processes", 1)
            print(f"[preflight] projected duration of one run of {Path(c).name}: {hours:.1f} h ({slowest:.2f} gradient steps/s with "
                  f"{n} process(es) side by side; one diagnostic point {final_seconds:.0f} s)", flush=True)
            if hours > TARGET_HOURS * 1.25:  # informational since 2026-09-30: the launch rule bears on the whole batch
                print(f"[preflight] note: one run of {Path(c).name} is above ~{TARGET_HOURS:g} h (the budget is fixed in the "
                      "config, not adjusted to the machine)", flush=True)
        # The whole batch: --batch-runs (default: the parallel mix), at most --parallel at once.
        per_run = {Path(c).as_posix(): h for c, h in summary["projected_hours_per_run"].items()}
        counts = ({Path(c).as_posix(): int(n) for c, n in (item.rsplit("=", 1) for item in args.batch_runs)} if args.batch_runs
                  else {Path(c).as_posix(): configs.count(c) for c in dict.fromkeys(configs)})
        # a config of the launch left out of the parallel mix is costed as the slowest measured one (upper bound)
        runs_hours = [per_run.get(c, max(per_run.values())) for c, n in counts.items() for _ in range(n)]
        unmeasured = [c for c in counts if c not in per_run]
        if unmeasured:
            print(f"[preflight] not in the parallel mix, costed as the slowest measured config: {unmeasured}", flush=True)
        n_runs = len(runs_hours)
        summary["projected_batch_hours"] = batch_hours(runs_hours, len(configs))
        cpus = summary["cpus"]
        print(f"[preflight] projected wall time of the batch: {summary['projected_batch_hours']:.1f} h for {n_runs} run(s), "
              f"{len(configs)} at a time, on {cpus['usable']} usable CPU(s)"
              + (f" ({args.cpus} requested)" if args.cpus is not None else "")
              + "; upper bound: every diagnostic point is costed as the final one (the training points use the reduced K)",
              flush=True)
        (run_dir / "preflight_summary.json").write_text(json.dumps(run_mini._jsonable(summary), indent=1), encoding="utf-8")

    if not problems and "projected_batch_hours" not in summary:
        problems.append("no duration projection (the truncated run did not finish)")
    if problems:
        print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems), flush=True)
    else:
        print("PREFLIGHT PASSED", flush=True)
    decision = write_verdict(args.out_dir, problems, memory_problems, warnings, len(configs), summary.get("projected_batch_hours"),
                             args.max_batch_hours, args.fallback_parallel)
    print(f"PREFLIGHT DECISION: {decision}  (rule of RUN_GPU.md step 3; details in {args.out_dir / 'preflight_verdict.json'})",
          flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
