"""Submit E3 to UCL ARC Kubeflow; default to preflight, never implicit training."""

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


def commands(phase: str, python: str, output: Path, resume: str = "") -> list[tuple[str, list[str]]]:
    """Return foreground commands, preserving the scientific configs unchanged."""
    base = [python, "-u", f"{SCRIPT}/run_mini.py"]
    if phase == "preflight":
        return [("preflight", [python, "-u", f"{SCRIPT}/preflight.py",
                              "--config", "configs/e3/gpu_e30.yaml",
                              "--extra-config", "configs/e3/gpu_e30_bridge.yaml",
                              "--device", "cuda", "--out-dir", str(output / "runs")])]
    if phase == "resume":
        if not resume:
            raise ValueError("resume requires a run directory")
        return [("resume", base + ["--resume", resume, "--device", "cuda", "--wandb-artifact"])]
    if phase != "train":
        raise ValueError(f"Unknown phase: {phase}")
    runs = [(f"seed{seed}_{condition}", base + [
        "--config", "configs/e3/gpu_e30.yaml", "--device", "cuda",
        "--seeds", str(seed), "--conditions", condition, "--wandb-artifact",
    ]) for seed in (0, 5) for condition in ("experimental", "reference")]
    runs.append(("bridge_seed0_experimental", base + [
        "--config", "configs/e3/gpu_e30_bridge.yaml", "--device", "cuda",
        "--seeds", "0", "--conditions", "experimental", "--wandb-artifact",
    ]))
    return runs


def supervise(jobs: list[tuple[str, list[str]]], repo: Path, output: Path) -> int:
    """Wait for all children; persist each exit code, including scientific failures."""
    output.mkdir(parents=True, exist_ok=False)
    processes: list[tuple[str, subprocess.Popen]] = []
    status: dict[str, dict] = {}

    def save() -> None:
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    def interrupted(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with ExitStack() as stack:
            try:
                for name, command in jobs:
                    log_path = output / f"{name}.log"
                    log = stack.enter_context(log_path.open("w", encoding="utf-8"))
                    process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                    processes.append((name, process))
                    status[name] = {"pid": process.pid, "command": command, "log": str(log_path),
                                    "started_at": datetime.now(timezone.utc).isoformat(), "exit_code": None}
                    save()
                    print(f"started {name}: pid={process.pid}, log={log_path}", flush=True)
                while any(item["exit_code"] is None for item in status.values()):
                    for name, process in processes:
                        code = process.poll()
                        if code is not None and status[name]["exit_code"] is None:
                            status[name].update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                            save()
                            print(f"finished {name}: exit_code={code}", flush=True)
                    if any(item["exit_code"] is None for item in status.values()):
                        time.sleep(1)
            finally:
                for _, process in processes:
                    if process.poll() is None:
                        process.terminate()
                for name, process in processes:
                    try:
                        code = process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        code = process.wait()
                    if status[name]["exit_code"] is None:
                        status[name].update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                save()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return int(any(item["exit_code"] != 0 for item in status.values()))


def _run_e3(repo_dir: str, phase: str, output: str, token_dir: str, resume: str) -> None:
    """Serialized by CustomTrainer: imports and constants must be self-contained."""
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
                      MPLBACKEND="Agg", XDG_CACHE_HOME=str(home / ".cache"))
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
    subprocess.run([python, "-u", "experiments/E3_worldmodel_4zones/submit_kubeflow.py",
                    "--worker", "--phase", phase, "--repo-dir", repo_dir,
                    "--output", output, "--resume", resume], cwd=repo, check=True)


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
    parser.add_argument("--phase", choices=("preflight", "train", "resume"), default="preflight")
    parser.add_argument("--repo-dir", default="/scratch/helplessness-worldmodels")
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--cpu", default="3")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--wandb-secret", help="Secret containing WANDB_API_KEY; ignored for preflight")
    parser.add_argument("--resume", default="", help="Existing run directory (phase=resume only)")
    parser.add_argument("--output", help="New persistent directory for supervisor logs/status")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase == "resume") != bool(args.resume):
        parser.error("--resume is required exactly when --phase=resume")
    repo = Path(args.repo_dir).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"{args.phase}-{stamp}"
    if args.worker:
        return supervise(commands(args.phase, sys.executable, output, args.resume), repo, output)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not repo.is_relative_to(Path(args.mount).resolve()) or not output.is_relative_to(Path(args.mount).resolve()):
        parser.error("--repo-dir and --output must be on the mounted persistent volume")

    import kubeflow.trainer
    from kubeflow.trainer.options.kubernetes import RuntimePatch

    secret = args.wandb_secret if args.phase != "preflight" else None
    client = kubeflow.trainer.TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        options=[RuntimePatch(training_runtime_spec=runtime_patch(args.volume, args.mount, secret))],
        trainer=kubeflow.trainer.CustomTrainer(
            func=_run_e3, image=args.image, num_nodes=1,
            func_args={"repo_dir": str(repo), "phase": args.phase, "output": str(output),
                       "token_dir": "/etc/e3-wandb" if secret else "", "resume": args.resume},
            resources_per_node={"cpu": args.cpu, "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "phase": args.phase, "output": str(output), "repo_dir": str(repo),
              "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
