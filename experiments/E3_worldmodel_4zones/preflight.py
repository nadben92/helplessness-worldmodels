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
4. The real entry point, truncated: run_mini.py on seed 0 / experimental
   with --max-steps (default 200) into <out-dir>/preflight/, then checks
   that every output is written (meta.yaml says device cuda and status
   done, replay.npz, world.json, weights_final.pt, resume.pt, arrays,
   report), and recomputes the final measures from the directory with
   recompute_measures.py (identical numbers expected).
Prints PREFLIGHT PASSED or PREFLIGHT FAILED (exit code 1) and writes
preflight_summary.json next to the run.
"""

from __future__ import annotations

import argparse
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def setup(cfg: dict, device: torch.device, n_env_steps: int = 3000):
    world = run_mini.build_world(cfg, 0, "experimental")
    replay = run_mini.ReplayBuffer.for_run(budget=n_env_steps, prefill_steps=0, table=world.templates_np)
    streams = run_mini.seed_streams(0, device)
    run_mini.collect_and_count(world.env, replay, n_env_steps, streams["collect"], run_mini.N_STATES, run_mini.N_ACTIONS,
                               streams["start_state"])
    if device.type == "cuda":
        replay.to_device(device)
    return world, replay


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
    parser.add_argument("--parallel", type=int, default=5, help="processes side by side for the throughput measure")
    parser.add_argument("--extra-config", type=Path, nargs="*", default=[],
                        help="configs launched alongside (one process each in the parallel measure), e.g. the bridge run")
    parser.add_argument("--out-dir", type=Path, default=run_mini.REPO_ROOT / "results" / "preflight")
    parser.add_argument("--bench-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = run_mini.load_mini_config(args.config)
    cfg["runtime"]["device"] = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("PREFLIGHT FAILED: --device cuda but torch.cuda.is_available() is False "
              f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})", flush=True)
        return 1
    device = run_mini.configure_runtime(cfg)
    if args.bench_only:
        print(json.dumps(bench(cfg, device, args.bench_steps)), flush=True)
        return 0

    summary: dict = {"config": str(args.config), "device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    print(f"[preflight] {summary}", flush=True)
    problems: list[str] = []

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
            if hours > TARGET_HOURS * 1.25:
                print(f"[preflight] WARNING: {Path(c).name} is above the ~{TARGET_HOURS:g} h target -- report it before launching "
                      "(the budget is fixed in the config, not adjusted to the machine)", flush=True)
        (run_dir / "preflight_summary.json").write_text(json.dumps(run_mini._jsonable(summary), indent=1), encoding="utf-8")

    if problems:
        print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems), flush=True)
        return 1
    print("PREFLIGHT PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
