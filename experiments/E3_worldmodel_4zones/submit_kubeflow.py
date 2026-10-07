"""Submit E3 to UCL ARC Kubeflow; default to preflight, never implicit training.

Two kinds of batch: training batches (BATCHES: phases preflight, train, resume; run_mini.py) and
measurement batches (MEASURE_BATCHES: phases preflight and measure only; recompute_measures.py on
existing runs read in place, no training -- docs/e3/DECISIONS.md, 2026-10-01 (10))."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

IMAGE = "ghcr.io/ucl-arc-unified-ai/kubeflow-trainer-images-torch:v0.0.1"
SCRIPT = "experiments/E3_worldmodel_4zones"
SPEC_CONFIG = "configs/e3/gpu_e30.yaml"
BRIDGE_CONFIG = "configs/e3/gpu_e30_bridge.yaml"
WIN64_CONFIG = "configs/e3/gpu_e30b_win64.yaml"
WIN16_CONFIG = "configs/e3/gpu_e30b_win16.yaml"
CONDITIONS = ("experimental", "reference")
# Every run of each batch: name -> (config, seed, condition), in launch order. Within a batch the configs
# first appear from the heaviest on the GPU to the lightest (spec network before the bridge; 64-step
# windows before 16-step ones, whose measurement probes unroll 4 times fewer steps per window).
BATCHES = {
    "gpu_e30": {
        **{f"seed{seed}_{condition}": (SPEC_CONFIG, seed, condition) for seed in (0, 5) for condition in CONDITIONS},
        "bridge_seed0_experimental": (BRIDGE_CONFIG, 0, "experimental"),
    },
    "gpu_e30b": {f"{tag}_seed{seed}_{condition}": (config, seed, condition)
                 for tag, config in (("win64", WIN64_CONFIG), ("win16", WIN16_CONFIG)) for seed in (0, 5) for condition in CONDITIONS},
}
# Measurement batches: existing runs measured again by recompute_measures.py, read in place where they were
# trained (the weights never leave the volume), outputs into the job's own directory. Each run: its directory
# name under source_root and the SHA-256 of its checkpoints/weights_final.pt (handoff/gpu_e30b/handoff.md).
GPU_E30B_RESULTS = "/scratch/helplessness-worldmodels-gpu_e30b/results/E3_worldmodel_4zones"
MEASURE_BATCHES = {
    "gpu_e30b_measures": {
        "source_root": GPU_E30B_RESULTS,
        "runs": {
            "win64_seed0_experimental": ("grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z",
                                         "f60377e0e7a8e46f9e24fd7af203eb724b6d35a7f56ce1033476358daf2f9ae2"),
            "win64_seed0_reference": ("grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z",
                                      "fa60ea69d694db6fd062171296f7af107f9b71f4609491353831a0f5effecbb9"),
            "win64_seed5_experimental": ("grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z",
                                         "9cc40755cc6110ac0a5e15805f3cf6853fda1bdd1528d34f2b52cc0d4bef6038"),
            "win64_seed5_reference": ("grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z",
                                      "690ea27d22a4a87eb8a4f310c0eac0fd3925525fd668b6ff006610c948e96fa2"),
        },
    },
}
# Closed-loop agent batches (run_agent.py; docs/e3/DECISIONS.md, 2026-10-02 (19)): phases preflight (preflight_agent.py,
# with the informative speed bench), train and resume. Jobs in launch order: the experimental runs first, so a launch
# at --max-parallel 2 runs them before the references.
AGENT_BATCHES = {
    "gpu_pilot_cl": {
        "config": "configs/e3/gpu_pilot_closed_loop.yaml",
        "runs": {"seed0_experimental": (0, "experimental"), "seed5_experimental": (5, "experimental"),
                 "seed0_reference": (0, "reference"), "seed5_reference": (5, "reference")},
    },
}
# The pilot's checkpoint weights re-measured (docs/e3/DECISIONS.md, 2026-10-05 (23)): remeasure_checkpoints.py on each
# run, read in place where the pilot wrote it; the weights' checksums are not known off the cluster (never handed off).
PILOT_RESULTS = "/scratch/helplessness-worldmodels/results/E3_worldmodel_4zones"
MEASURE_BATCHES["gpu_pilot_remeasure"] = {
    "kind": "checkpoints",
    "source_root": PILOT_RESULTS,
    "env_steps": [63096, 100000, 158489, 251189, 300000],
    # a fifth job, after the four (DECISIONS 2026-10-05 (24)): the read-only value diagnostic at these (run, checkpoint)
    "value_diagnostics": [("win64_seed0_reference", 10000), ("win64_seed0_reference", 15849), ("win64_seed5_experimental", 39811)],
    "runs": {
            "win64_seed0_experimental": ("grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z", None),
            "win64_seed5_experimental": ("grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z", None),
            "win64_seed0_reference": ("grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z", None),
            "win64_seed5_reference": ("grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z", None),
    },
}
# Measurement noise of I_s and the goal control (docs/e3/DECISIONS.md, 2026-10-06 (26)): measure_noise.py on the pilot's
# final weights (8 independent context draws x 1, 2, 4 times the contexts per state), and the goal control at REF 0 /
# 15,849 (10,000 draws per pair into the goal), runs read in place.
MEASURE_BATCHES["gpu_pilot_noise"] = {
    "kind": "noise",
    "source_root": PILOT_RESULTS,
    "repeats": 8,
    "contexts_multipliers": [1, 2, 4],
    "goal_control": ("win64_seed0_reference", 15849, 10_000),
    # 2026-10-06 (27): after its repeats, each job re-measures I_s at the 5 checkpoints with the candidate protocols
    # (x2 first, then x4); if the whole exceeds max_batch_hours, the fallback batch keeps x2 only
    "checkpoint_steps": list(MEASURE_BATCHES["gpu_pilot_remeasure"]["env_steps"]),
    "checkpoint_multipliers": [2, 4],
    "fallback_batch": "gpu_pilot_noise_x2",
    "max_batch_hours": 8.0,
    "runs": dict(MEASURE_BATCHES["gpu_pilot_remeasure"]["runs"]),
}
MEASURE_BATCHES["gpu_pilot_noise_x2"] = {**MEASURE_BATCHES["gpu_pilot_noise"], "checkpoint_multipliers": [2], "fallback_batch": None}
# The E3 campaign (docs/e3/DECISIONS.md, 2026-10-07 (29), (30)): N = 4 seeds. Seeds 1 (class A) and 6 (class B): new runs,
# each in the environment and its reference, 4 runs in one round (at most 4 at a time). Seeds 0 (A) and 5 (B): the
# pilot's runs, read in place and measured with the campaign's protocol by campaign_measure_pilot.py, queued after the
# training jobs (whole batch only). The preflight times the campaign's measures on the pilot's REF seed 0 (read only)
# and skips the speed bench (measured for the pilot).
AGENT_BATCHES["gpu_campaign"] = {
    "config": "configs/e3/gpu_campaign.yaml",
    "runs": {f"seed{seed}_{condition}": (seed, condition) for seed in (1, 6) for condition in CONDITIONS},
    "pilot_runs": {f"pilot_{job}": f"{PILOT_RESULTS}/{run_dir}" for job, (run_dir, _) in MEASURE_BATCHES["gpu_pilot_remeasure"]["runs"].items()},
    "measure_smoke_run": f"{PILOT_RESULTS}/{MEASURE_BATCHES['gpu_pilot_remeasure']['runs']['win64_seed0_reference'][0]}",
    "bench_minutes": 0,
}
DEFAULT_BATCH = "gpu_e30"
# The job's CPU request, PER BATCH (docs/e3/DECISIONS.md, 2026-09-30 (5)): gpu_e30b was prescribed 5 CPUs and ran on 3,
# because --cpu defaulted to 3 and a command without it (or with another value) was accepted silently. The
# request now defaults to the batch's own value, and a different --cpu is refused unless --allow-cpu-override.
BATCH_CPUS = {"gpu_e30": 3, "gpu_e30b": 5, "gpu_e30b_measures": 3, "gpu_pilot_cl": 3, "gpu_pilot_remeasure": 3, "gpu_pilot_noise": 3,
              "gpu_pilot_noise_x2": 3, "gpu_campaign": 3}
DEFAULT_CPUS = BATCH_CPUS[DEFAULT_BATCH]


def resolve_cpus(batch: str, cpu: int | None, allow_override: bool = False) -> int:
    """The CPUs to request for a batch: its prescribed number; an explicit
    --cpu must equal it (ValueError otherwise) unless allow_override."""
    prescribed = BATCH_CPUS[batch]
    if cpu is None or cpu == prescribed:
        return prescribed
    if not allow_override:
        raise ValueError(f"batch {batch} is prescribed {prescribed} CPUs, got --cpu {cpu}; pass --allow-cpu-override to "
                         "request another number on purpose (and record why)")
    return cpu
RUNS = BATCHES[DEFAULT_BATCH]


def all_batches() -> list[str]:
    return [*BATCHES, *MEASURE_BATCHES, *AGENT_BATCHES]


def batch_runs(batch: str) -> dict:
    """The batch's runs, name -> definition (training: (config, seed, condition); measurement: (run dir, sha256))."""
    if batch in BATCHES:
        return BATCHES[batch]
    if batch in MEASURE_BATCHES:
        return MEASURE_BATCHES[batch]["runs"]
    if batch in AGENT_BATCHES:
        return AGENT_BATCHES[batch]["runs"]
    raise ValueError(f"unknown batch {batch!r}; choose among {all_batches()}")


def select_runs(runs: list[str] | None, batch: str = DEFAULT_BATCH) -> list[str]:
    """The requested subset of the batch's runs, in launch order (all of them for None)."""
    known = batch_runs(batch)
    if not runs:
        return list(known)
    unknown = sorted(set(runs) - set(known))
    if unknown:
        raise ValueError(f"unknown run(s) {unknown} in batch {batch}; choose among {list(known)}")
    return [name for name in known if name in runs]


def n_concurrent(runs: list[str] | None, max_parallel: int | None, batch: str = DEFAULT_BATCH) -> int:
    """How many processes share the GPU at most: every selected run, or
    max_parallel if smaller."""
    n = len(select_runs(runs, batch))
    if max_parallel is not None and max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    return n if max_parallel is None else min(n, max_parallel)


def preflight_command(python: str, output: Path, runs: list[str] | None, max_parallel: int | None,
                      batch: str = DEFAULT_BATCH, cpus: int | None = None) -> list[str]:
    """preflight.py (preflight_measures.py for a measurement batch) for the planned launch: --parallel is the number of
    processes that will share the GPU, filled with the heaviest config
    first (its first config in the batch's order), plus one process of each
    other selected config when there is room, so the memory and throughput
    tests see the worst mix the launch can produce and every config's
    duration is projected."""
    selected = select_runs(runs, batch)
    parallel = n_concurrent(runs, max_parallel, batch)
    cpus = BATCH_CPUS[batch] if cpus is None else cpus
    if batch in AGENT_BATCHES:
        return [python, "-u", f"{SCRIPT}/preflight_agent.py", "--batch", batch, "--cpus", str(cpus), "--device", "cuda",
                "--out-dir", str(output / "runs")]
    if batch in MEASURE_BATCHES:
        return [python, "-u", f"{SCRIPT}/preflight_measures.py", "--batch", batch, "--runs", *selected, "--parallel", str(parallel),
                "--cpus", str(cpus), "--device", "cuda", "--out-dir", str(output / "runs")]
    known = BATCHES[batch]
    configs = list(dict.fromkeys(known[name][0] for name in selected))
    primary, others = configs[0], configs[1:]
    n_primary = sum(known[name][0] == primary for name in selected)
    command = [python, "-u", f"{SCRIPT}/preflight.py", "--config", primary]
    if others and parallel > n_primary:
        command += ["--extra-config", *others[:parallel - 1]]
    counts = [f"{c}={sum(known[name][0] == c for name in selected)}" for c in configs]
    return command + ["--parallel", str(parallel), "--batch-runs", *counts, "--cpus", str(cpus), "--device", "cuda",
                      "--out-dir", str(output / "runs")]


def commands(phase: str, python: str, output: Path, resume: str = "", runs: list[str] | None = None,
             max_parallel: int | None = None, batch: str = DEFAULT_BATCH, cpus: int | None = None) -> list[tuple[str, list[str]]]:
    """Return foreground commands, preserving the scientific configs unchanged.
    batch: a key of BATCHES; runs: a subset of its runs (default: all);
    max_parallel: the most processes run at once (default: all), which the
    preflight tests; cpus: the job's CPU request, passed to the preflight (its
    duration projection is measured under that allocation)."""
    base = [python, "-u", f"{SCRIPT}/run_mini.py"]
    if phase == "preflight":
        return [("preflight", preflight_command(python, output, runs, max_parallel, batch, cpus))]
    if batch in AGENT_BATCHES:
        agent = [python, "-u", f"{SCRIPT}/run_agent.py"]
        if phase == "resume":
            if not resume:
                raise ValueError("resume requires a run directory")
            return [("resume", agent + ["--resume", resume, "--device", "cuda"])]
        if phase != "train":
            raise ValueError(f"{batch} is a closed-loop batch: phases preflight, train and resume, never {phase}")
        known = AGENT_BATCHES[batch]["runs"]
        jobs = [(name, agent + ["--config", AGENT_BATCHES[batch]["config"], "--device", "cuda", "--seeds", str(known[name][0]),
                                "--conditions", known[name][1]]) for name in select_runs(runs, batch)]
        if not runs:  # the pilot's runs measured as campaign seeds (2026-10-07 (30)), after the training jobs, whole batch only
            jobs += [(name, [python, "-u", f"{SCRIPT}/campaign_measure_pilot.py", run_dir, "--config", AGENT_BATCHES[batch]["config"],
                             "--device", "cuda", "--out-dir", str(output / "runs")])
                     for name, run_dir in AGENT_BATCHES[batch].get("pilot_runs", {}).items()]
        return jobs
    if batch in MEASURE_BATCHES:
        if phase != "measure":
            raise ValueError(f"{batch} is a measurement batch: phases preflight and measure only, never {phase}")
        root, known = MEASURE_BATCHES[batch]["source_root"], MEASURE_BATCHES[batch]["runs"]
        if MEASURE_BATCHES[batch].get("kind") == "noise":
            b = MEASURE_BATCHES[batch]
            jobs = [(name, [python, "-u", f"{SCRIPT}/measure_noise.py", "--noise", f"{root}/{known[name][0]}", "--repeats", str(b["repeats"]),
                            "--contexts-multipliers", *[str(m) for m in b["contexts_multipliers"]],
                            *(["--checkpoint-steps", *[str(c) for c in b["checkpoint_steps"]], "--checkpoint-multipliers",
                               *[str(m) for m in b["checkpoint_multipliers"]]] if b.get("checkpoint_steps") else []),
                            "--device", "cuda", "--out-dir", str(output / "runs" / known[name][0])]) for name in select_runs(runs, batch)]
            if not runs:
                job, c, draws = b["goal_control"]
                jobs.append(("goal_control", [python, "-u", f"{SCRIPT}/measure_noise.py", "--goal-control", f"{root}/{known[job][0]}:{c}",
                                              "--draws", str(draws), "--device", "cuda", "--out-dir", str(output / "runs" / "goal_control")]))
            return jobs
        if MEASURE_BATCHES[batch].get("kind") == "checkpoints":
            steps = [str(c) for c in MEASURE_BATCHES[batch]["env_steps"]]
            jobs = [(name, [python, "-u", f"{SCRIPT}/remeasure_checkpoints.py", f"{root}/{known[name][0]}", "--env-steps", *steps,
                            "--device", "cuda", "--out-dir", str(output / "runs")]) for name in select_runs(runs, batch)]
            points = MEASURE_BATCHES[batch].get("value_diagnostics")
            if points and not runs:  # with the whole batch only
                jobs.append(("value_diagnostics", [python, "-u", f"{SCRIPT}/diagnose_value_generalisation.py", "--points",
                                                   *[f"{root}/{known[job][0]}:{c}" for job, c in points], "--device", "cuda",
                                                   "--out-dir", str(output / "runs" / "value_diagnostics")]))
            return jobs
        return [(name, [python, "-u", f"{SCRIPT}/recompute_measures.py", f"{root}/{known[name][0]}", "--checkpoint", "final",
                        "--device", "cuda", "--out-dir", str(output / "runs"), "--weights-sha256", known[name][1]])
                for name in select_runs(runs, batch)]
    if phase == "measure":
        raise ValueError(f"phase measure is for the measurement batches {list(MEASURE_BATCHES)}, not {batch}")
    if phase == "resume":
        if not resume:
            raise ValueError("resume requires a run directory")
        return [("resume", base + ["--resume", resume, "--device", "cuda", "--wandb-artifact"])]
    if phase != "train":
        raise ValueError(f"Unknown phase: {phase}")
    known = BATCHES[batch]
    return [(name, base + ["--config", known[name][0], "--device", "cuda", "--seeds", str(known[name][1]),
                           "--conditions", known[name][2], "--wandb-artifact"])
            for name in select_runs(runs, batch)]


def supervise(jobs: list[tuple[str, list[str]]], repo: Path, output: Path, max_parallel: int | None = None) -> int:
    """Run the jobs, at most max_parallel at a time (default: all at once),
    in order, starting the next queued job when one finishes; wait for all
    of them and persist each exit code, including scientific failures. A
    job never started (supervisor interrupted) is recorded with exit_code
    null and not_started true."""
    if max_parallel is not None and max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    limit = len(jobs) if max_parallel is None else max_parallel
    output.mkdir(parents=True, exist_ok=False)
    (output / "supervisor.json").write_text(json.dumps({"max_parallel": limit, "jobs": [name for name, _ in jobs]}, indent=2) + "\n",
                                            encoding="utf-8")
    processes: dict[str, subprocess.Popen] = {}
    queue = list(jobs)
    status: dict[str, dict] = {name: {"command": command, "log": str(output / f"{name}.log"), "exit_code": None}
                               for name, command in jobs}

    def save() -> None:
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    def interrupted(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    def running() -> int:
        return sum(status[name]["exit_code"] is None for name in processes)

    previous = signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        with ExitStack() as stack:
            try:
                while queue or running():
                    while queue and running() < limit:
                        name, command = queue.pop(0)
                        log = stack.enter_context((output / f"{name}.log").open("w", encoding="utf-8"))
                        processes[name] = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                        status[name].update(pid=processes[name].pid, started_at=datetime.now(timezone.utc).isoformat())
                        save()
                        print(f"started {name}: pid={processes[name].pid}, log={status[name]['log']} "
                              f"({running()}/{limit} running, {len(queue)} queued)", flush=True)
                    for name, process in processes.items():
                        code = process.poll()
                        if code is not None and status[name]["exit_code"] is None:
                            status[name].update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                            save()
                            print(f"finished {name}: exit_code={code}", flush=True)
                    if queue or running():
                        time.sleep(1)
            finally:
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                for name, process in processes.items():
                    try:
                        code = process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        code = process.wait()
                    if status[name]["exit_code"] is None:
                        status[name].update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                for name, _ in queue:
                    status[name]["not_started"] = True
                save()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return int(any(item["exit_code"] != 0 for item in status.values()))


def _run_e3(repo_dir: str, phase: str, output: str, token_dir: str, resume: str, runs: str = "", max_parallel: int = 0,
            batch: str = "gpu_e30", cpus: int = 3) -> None:
    """Serialized by CustomTrainer: imports and constants must be self-contained.
    batch: a key of BATCHES; runs: comma-separated subset of its runs ("" = all); max_parallel: 0 = all at once."""
    import os
    from pathlib import Path
    import subprocess
    import sys

    if int(os.environ.get("RANK", "0")) != 0:
        print("Only rank zero runs this non-distributed experiment.", flush=True)
        return
    repo = Path(repo_dir)
    home = repo / ".cluster-home"
    home.mkdir(exist_ok=True)
    os.environ.update(HOME=str(home), USER="trainjob", LOGNAME="trainjob",
                      PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      MPLBACKEND="Agg", XDG_CACHE_HOME=str(home / ".cache"),
                      # one BLAS/OpenMP thread per process: several processes share the job's few CPUs, and numpy's
                      # template matching would otherwise start one thread per core of the node (2026-09-30)
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # The admission webhook rejects container env patches; credentials arrive as files.
    if token_dir and phase != "preflight":
        token = (Path(token_dir) / "WANDB_API_KEY").read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Mounted WANDB_API_KEY is empty")
        os.environ["WANDB_API_KEY"] = token
    else:
        os.environ.pop("WANDB_API_KEY", None)
        os.environ["WANDB_MODE"] = "disabled"

    # Isolate this project's installs from language_mind while keeping the image's CUDA torch.
    venv = repo / ".venv-cluster"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                    "--system-site-packages", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    subprocess.run([sys.executable, "-m", "pip", "--python", python,
                    "install", "--quiet", "-e", ".[e3,dev,wandb]"],
                   cwd=repo, check=True)
    subprocess.run([python, "-c",
                    "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, flush=True)"],
                   check=True)
    selection = (["--batch", batch, "--cpu", str(cpus)] + (["--runs", *runs.split(",")] if runs else [])
                 + (["--max-parallel", str(max_parallel)] if max_parallel else []))
    subprocess.run([python, "-u", "experiments/E3_worldmodel_4zones/submit_kubeflow.py",
                    "--worker", "--phase", phase, "--repo-dir", repo_dir,
                    "--output", output, "--resume", resume, *selection], cwd=repo, check=True)


def runtime_patch(volume: str, mount: str, secret: str | None) -> dict:
    """Build only the volume patch permitted by the cluster admission policy."""
    volumes = [{"name": "scratch", "persistentVolumeClaim": {"claimName": volume}}]
    mounts = [{"name": "scratch", "mountPath": mount}]
    if secret:
        volumes.append({"name": "wandb-token", "secret": {"secretName": secret}})
        mounts.append({"name": "wandb-token", "mountPath": "/etc/e3-wandb", "readOnly": True})
    return {"template": {"spec": {"replicatedJobs": [
        {"name": "node", "template": {"spec": {"template": {"spec": {
            "volumes": volumes, "containers": [{"name": "node", "volumeMounts": mounts}],
        }}}}}
    ]}}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train", "resume", "measure"), default="preflight",
                        help="measure: a measurement batch's recomputation (MEASURE_BATCHES), no training")
    parser.add_argument("--repo-dir", default="/scratch/helplessness-worldmodels")
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--cpu", type=int, default=None,
                        help="CPUs requested for the job (default: the batch's prescribed number, BATCH_CPUS; another value is "
                             "refused without --allow-cpu-override); the preflight checks and projects under it")
    parser.add_argument("--allow-cpu-override", action="store_true", help="accept a --cpu different from the batch's prescribed number")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--wandb-secret", help="Secret containing WANDB_API_KEY; ignored for preflight")
    parser.add_argument("--resume", default="", help="Existing run directory (phase=resume only)")
    parser.add_argument("--output", help="New persistent directory for supervisor logs/status")
    parser.add_argument("--batch", choices=all_batches(), default=DEFAULT_BATCH, help="which batch's runs (default: gpu_e30)")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="subset of the batch's runs (default: all); pass the same to the preflight and the training")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="most runs sharing the GPU at once (default: all); the preflight tests this number")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase == "resume") != bool(args.resume):
        parser.error("--resume is required exactly when --phase=resume")
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    try:
        select_runs(args.runs, args.batch)
        commands(args.phase, "python", Path("."), args.resume or "x", runs=args.runs, batch=args.batch)  # phase / batch compatibility
        args.cpu = resolve_cpus(args.batch, args.cpu, args.allow_cpu_override or args.worker)  # the worker gets the resolved value
    except ValueError as err:
        parser.error(str(err))
    repo = Path(args.repo_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"{args.phase}-{stamp}"
    if args.worker:
        jobs = commands(args.phase, sys.executable, output, args.resume, runs=args.runs, max_parallel=args.max_parallel,
                        batch=args.batch, cpus=args.cpu)
        return supervise(jobs, repo, output, max_parallel=args.max_parallel if args.phase in ("train", "measure") else None)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not repo.is_relative_to(Path(args.mount).resolve()) or not output.is_relative_to(Path(args.mount).resolve()):
        parser.error("--repo-dir and --output must be on the mounted persistent volume")

    import kubeflow.trainer
    from kubeflow.trainer.options.kubernetes import RuntimePatch

    secret = args.wandb_secret if args.phase in ("train", "resume") else None
    client = kubeflow.trainer.TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        options=[RuntimePatch(training_runtime_spec=runtime_patch(args.volume, args.mount, secret))],
        trainer=kubeflow.trainer.CustomTrainer(
            func=_run_e3, image=args.image, num_nodes=1,
            func_args={"repo_dir": str(repo), "phase": args.phase, "output": str(output),
                       "token_dir": "/etc/e3-wandb" if secret else "", "resume": args.resume,
                       "runs": ",".join(args.runs or []), "max_parallel": args.max_parallel or 0, "batch": args.batch,
                       "cpus": args.cpu},
            resources_per_node={"cpu": str(args.cpu), "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "phase": args.phase, "output": str(output), "repo_dir": str(repo),
              "batch": args.batch, "cpus": args.cpu, "runs": select_runs(args.runs, args.batch),
              "max_parallel": n_concurrent(args.runs, args.max_parallel, args.batch),
              "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
