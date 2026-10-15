"""Submit an E4 batch (batches.WAVES) to UCL ARC Kubeflow: one job, one GPU, the batch's four runs in parallel under
the library's supervisor (helplessness.cluster.kubeflow.supervise). Each run pushes its light results to the orphan
branch results/<batch> when it ends (run_e4.py --push-batch; docs/RESULTS_CONVENTION.md), and the supervisor pushes the
batch's status.json and handoff.json at the end. Weights and buffers stay under <repo-dir>/results/ on the volume.

Phases: train (the batch's runs; wave 1 at the full budget, waves 2-4 at B, batches.max_env_steps) and resume (one run
directory, continued from its checkpoints/resume.pt).

The checkout on the volume must be a clone of the branch e4 over https (the push token is an https credential);
the submission refuses another branch or a modified working tree. Credentials: a Kubernetes secret holding the key
RESULTS_PUSH_TOKEN (a GitHub token allowed to push to the repository), mounted read-only at PUSH_TOKEN_DIR; never an
environment variable in the job spec, never a file of the repository.

Usage (cluster login node, from the checkout): see experiments/E4_generalization_gradient/RUN_GPU.md.
  python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --repo-dir /scratch/<checkout> \\
      --push-secret <secret name> [--dry-run]
  python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase resume --resume <run dir> ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:  # the notebook's Python runs this script without the package installed
    sys.path.insert(0, str(_SRC))

import batches
from helplessness.cluster.kubeflow import runtime_patch_secrets, supervise

IMAGE = "ghcr.io/ucl-arc-unified-ai/kubeflow-trainer-images-torch:v0.0.1"  # E3's image (same torch, same CUDA)
SCRIPT = "experiments/E4_generalization_gradient"
PUSH_TOKEN_DIR = "/etc/e4-results-push"
PUSH_TOKEN_KEY = "RESULTS_PUSH_TOKEN"
RESULTS_SUBDIR = "results"


def commands(phase: str, python: str, batch: str, push_repo: str, token_file: str, resume: str = "") -> list[tuple[str, list[str]]]:
    """The supervisor's jobs: one run_e4.py process per run of the batch (train), or the resumed run."""
    push = ["--push-batch", batch, "--push-repo", push_repo, "--push-token-file", token_file]
    base = [python, "-u", f"{SCRIPT}/run_e4.py"]
    if phase == "resume":
        if not resume:
            raise ValueError("resume requires a run directory")
        return [("resume", base + ["--resume", resume, "--device", "cuda", *push])]
    if phase != "train":
        raise ValueError(f"unknown phase {phase!r}")
    stop = batches.max_env_steps(batch)
    return [(name, base + ["--config", batches.CONFIG, "--configuration", cid, "--condition", condition, "--device", "cuda",
                           "--out-dir", RESULTS_SUBDIR, *([] if stop is None else ["--max-env-steps", str(stop)]), *push])
            for name, (cid, condition) in batches.batch_runs(batch).items()]


def batch_run_dirs(repo: Path, batch: str, started_after: str) -> list[Path]:
    """The run directories the batch created (newest per job), for the batch-level push."""
    root = repo / RESULTS_SUBDIR / "E4_generalization_gradient"
    out = []
    for cid, condition in batches.batch_runs(batch).values():
        role = "experimental" if condition == "experimental" else "control"
        found = sorted(d for d in root.glob(f"*__2zones-{cid}__pair-{role}__*") if d.name[-16:] >= started_after)
        if found:
            out.append(found[-1])
    return out


def _run_e4(repo_dir: str, phase: str, output: str, batch: str, push_repo: str, token_dir: str, resume: str = "",
            max_parallel: int = 0) -> None:
    """Serialized by CustomTrainer: imports and constants must be self-contained (as E3's _run_e3)."""
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
    os.environ.update(HOME=str(home), USER="trainjob", LOGNAME="trainjob", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      MPLBACKEND="Agg", XDG_CACHE_HOME=str(home / ".cache"), WANDB_MODE="disabled",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    venv = repo / ".venv-cluster"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)], check=True)
    python = str(venv / "bin" / "python")
    subprocess.run([sys.executable, "-m", "pip", "--python", python, "install", "--quiet", "-e", ".[e3,dev]"], cwd=repo, check=True)
    subprocess.run([python, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
                                  "print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, flush=True)"], check=True)
    subprocess.run([python, "-u", "experiments/E4_generalization_gradient/submit_kubeflow.py", "--worker", "--phase", phase,
                    "--batch", batch, "--repo-dir", repo_dir, "--output", output, "--push-repo", push_repo,
                    "--push-token-file", f"{token_dir}/RESULTS_PUSH_TOKEN", "--resume", resume,
                    *(["--max-parallel", str(max_parallel)] if max_parallel else [])], cwd=repo, check=True)


def worker(args) -> int:
    from helplessness.cluster import results_push

    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo, output = Path(args.repo_dir), Path(args.output)
    jobs = commands(args.phase, sys.executable, args.batch, args.push_repo, args.push_token_file, args.resume)
    code = supervise(jobs, repo, output, max_parallel=args.max_parallel)
    try:  # the batch-level status, after every run pushed its own results
        token = results_push.read_token(args.push_token_file)
        results_push.push_batch_status(output, batch_run_dirs(repo, args.batch, started), "E4_generalization_gradient", args.batch,
                                       args.push_repo, token)
    except Exception as err:
        print(f"[submit_kubeflow] BATCH STATUS PUSH FAILED: {type(err).__name__}: {err}", flush=True)
        code = code or 2
    return code


def check_checkout(repo: Path) -> str:
    """The checkout must be on branch e4, unmodified, with an https remote; returns that remote's URL."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()

    branch, dirty, url = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain", "--untracked-files=no"), git("remote", "get-url", "origin")
    if branch != "e4":
        raise SystemExit(f"{repo} is on branch {branch!r}, not e4")
    if dirty:
        raise SystemExit(f"{repo} has modified tracked files:\n{dirty}")
    if not url.startswith("https://"):
        raise SystemExit(f"origin is {url!r}: the results push needs an https remote")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(batches.WAVES), required=True)
    parser.add_argument("--phase", choices=("train", "resume"), default="train")
    parser.add_argument("--repo-dir", required=True, help="the e4 checkout on the persistent volume")
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--push-secret", help="the Kubernetes secret holding RESULTS_PUSH_TOKEN")
    parser.add_argument("--resume", default="", help="run directory (phase resume)")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--output", help="supervisor directory (default: <repo>/logs/kubeflow/<phase>-<UTC>)")
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and the runtime patch, submit nothing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--push-repo", help=argparse.SUPPRESS)
    parser.add_argument("--push-token-file", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase == "resume") != bool(args.resume):
        parser.error("--resume is required exactly when --phase=resume")
    if args.worker:
        return worker(args)
    repo = Path(args.repo_dir).resolve()
    try:
        commands(args.phase, "python", args.batch, "https://example.invalid/x.git", "TOKEN", args.resume or "x")
    except ValueError as err:
        parser.error(str(err))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"e4-{args.phase}-{stamp}"
    patch = runtime_patch_secrets(args.volume, args.mount, {args.push_secret: PUSH_TOKEN_DIR} if args.push_secret else {})
    push_repo = "https://<origin>" if args.dry_run else check_checkout(repo)
    if args.dry_run:
        jobs = commands(args.phase, "<venv python>", args.batch, push_repo, f"{PUSH_TOKEN_DIR}/{PUSH_TOKEN_KEY}", args.resume)
        print(json.dumps({"batch": args.batch, "cpus": batches.BATCH_CPUS[args.batch], "output": str(output),
                          "jobs": jobs, "runtime_patch": patch}, indent=1))
        return 0
    if not args.push_secret:
        parser.error("--push-secret is required: every run pushes its results")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not repo.is_relative_to(Path(args.mount).resolve()) or not output.is_relative_to(Path(args.mount).resolve()):
        parser.error("--repo-dir and --output must be on the mounted persistent volume")

    import kubeflow.trainer
    from kubeflow.trainer.options.kubernetes import RuntimePatch

    cpus = batches.BATCH_CPUS[args.batch]
    client = kubeflow.trainer.TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        options=[RuntimePatch(training_runtime_spec=patch)],
        trainer=kubeflow.trainer.CustomTrainer(
            func=_run_e4, image=args.image, num_nodes=1,
            func_args={"repo_dir": str(repo), "phase": args.phase, "output": str(output), "batch": args.batch,
                       "push_repo": push_repo, "token_dir": PUSH_TOKEN_DIR, "resume": args.resume, "max_parallel": args.max_parallel or 0},
            resources_per_node={"cpu": str(cpus), "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "batch": args.batch, "phase": args.phase, "output": str(output), "repo_dir": str(repo), "cpus": cpus,
              "runs": list(batches.batch_runs(args.batch)), "push_branch": f"results/{args.batch}",
              "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs (RUN_GPU.md).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
