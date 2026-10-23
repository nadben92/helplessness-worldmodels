"""Submit an E6 / E7 phase (batches.PHASES) of a batch of agents (batches.BATCHES) to UCL ARC Kubeflow: one job on one GPU,
under the library's supervisor (helplessness.cluster.kubeflow.supervise: <output>/status.json, one log per job), the
agents in parallel (batches.MAX_PARALLEL), each process measuring --agents-per-job agents. E5's submission with E6's
phases; frozen weights only, nothing is trained.

Phases: e6 (run_e6.py), e7-directions (run_e7_directions.py), e7-apply (run_e7_apply.py; reads the e7-directions outputs
under results/E7_latent_manipulation/ of the checkout, so it is submitted after them). Batch parameters: --runs-root (the
root the agents file's run globs are relative to: /scratch on the cluster), --agents-per-job, --agent-ids (a subset).

Launched from the e6 checkout (branch e6, no modified tracked file: refused otherwise); results under
<repo-dir>/results/E6_phenotypes_latent/ and results/E7_latent_manipulation/, brought back with export_batch.py.

Usage (Kubeflow notebook, from the e6 checkout):
  python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e6 --runs-root /scratch --repo-dir /scratch/helplessness-worldmodels-e6 --dry-run
  python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-directions ...
  python experiments/E6_phenotypes_latent/submit_kubeflow.py --batch e6_e5 --phase e7-apply ...
  python experiments/E6_phenotypes_latent/submit_kubeflow.py --status <supervisor directory>
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
from helplessness.cluster import batch_status
from helplessness.cluster.kubeflow import runtime_patch_secrets, supervise

IMAGE = "ghcr.io/ucl-arc-unified-ai/kubeflow-trainer-images-torch:v0.0.1"  # E3-E5's image
BRANCH = "e6"
CHECKOUT = Path(__file__).resolve().parents[2]  # the agents files are read from the checkout this script runs from


def commands(python: str, batch: str, phase: str, runs_root: str, repo: Path, device: str = "cuda", agents_per_job: int = 1,
             only: list[str] | None = None) -> list[tuple[str, list[str]]]:
    return [(name, [python, "-u", *args]) for name, args in batches.jobs(CHECKOUT, batch, phase, runs_root, device, agents_per_job, only)]


def _run_e6(repo_dir: str, phase: str, output: str, batch: str, runs_root: str, agents_per_job: int = 1, max_parallel: int = 0,
            agent_ids: str = "") -> None:
    """Serialized by CustomTrainer: imports and constants must be self-contained (as E5's _run_e5)."""
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
    args = [python, "-u", "experiments/E6_phenotypes_latent/submit_kubeflow.py", "--worker", "--phase", phase, "--batch", batch,
            "--repo-dir", repo_dir, "--output", output, "--runs-root", runs_root, "--agents-per-job", str(agents_per_job)]
    if max_parallel:
        args += ["--max-parallel", str(max_parallel)]
    if agent_ids:
        args += ["--agent-ids", *agent_ids.split(",")]
    subprocess.run(args, cwd=repo, check=True)


def check_checkout(repo: Path) -> None:
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()

    branch, dirty = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    if branch != BRANCH:
        raise SystemExit(f"{repo} is on branch {branch!r}, not {BRANCH}")
    if dirty:
        raise SystemExit(f"{repo} has modified tracked files:\n{dirty}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(batches.BATCHES), default=None)
    parser.add_argument("--phase", choices=list(batches.PHASES), default=None)
    parser.add_argument("--status", type=Path, default=None, help="print the state of a submitted batch (its supervisor directory) and exit")
    parser.add_argument("--stale-after", type=float, default=batch_status.STALE_AFTER)
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument("--runs-root", default=None, help="the root of the agents file's run globs (the cluster: /scratch)")
    parser.add_argument("--agents-per-job", type=int, default=1)
    parser.add_argument("--agent-ids", nargs="+", default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output", help="supervisor directory (default: <repo>/logs/kubeflow/e6-<phase>-<batch>-<UTC>)")
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and the runtime patch, submit nothing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.status is not None:
        rows = batch_status.batch_status(args.status, args.stale_after)
        print("\n".join(batch_status.format_status(rows, args.stale_after)), flush=True)
        return 2 if any(r["state"] == "STALE" for r in rows) else 0
    for flag in ("batch", "phase", "repo_dir", "runs_root"):
        if getattr(args, flag) is None:
            parser.error(f"--{flag.replace('_', '-')} is required")
    repo = Path(args.repo_dir).resolve()
    max_parallel = args.max_parallel or batches.MAX_PARALLEL
    if args.worker:
        jobs = commands(sys.executable, args.batch, args.phase, args.runs_root, repo, args.device, args.agents_per_job, args.agent_ids)
        return supervise(jobs, repo, Path(args.output), max_parallel=max_parallel)
    try:
        jobs = commands("<venv python>", args.batch, args.phase, args.runs_root, repo, "cuda", args.agents_per_job, args.agent_ids)
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"e6-{args.phase}-{args.batch}-{stamp}"
    patch = runtime_patch_secrets(args.volume, args.mount, {})
    if args.dry_run:
        print(json.dumps({"batch": args.batch, "phase": args.phase, "agents_file": batches.BATCHES[args.batch], "runs_root": args.runs_root,
                          "cpus": batches.BATCH_CPUS, "max_parallel": max_parallel, "output": str(output), "jobs": jobs,
                          "runtime_patch": patch}, indent=1))
        return 0
    check_checkout(repo)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if not repo.is_relative_to(Path(args.mount).resolve()) or not output.is_relative_to(Path(args.mount).resolve()):
        parser.error("--repo-dir and --output must be on the mounted persistent volume")

    import kubeflow.trainer
    from kubeflow.trainer.options.kubernetes import RuntimePatch

    client = kubeflow.trainer.TrainerClient()
    job_id = client.train(
        runtime=client.get_runtime("torch-distributed"),
        options=[RuntimePatch(training_runtime_spec=patch)],
        trainer=kubeflow.trainer.CustomTrainer(
            func=_run_e6, image=args.image, num_nodes=1,
            func_args={"repo_dir": str(repo), "phase": args.phase, "output": str(output), "batch": args.batch, "runs_root": args.runs_root,
                       "agents_per_job": args.agents_per_job, "max_parallel": max_parallel, "agent_ids": ",".join(args.agent_ids or [])},
            resources_per_node={"cpu": str(batches.BATCH_CPUS), "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "batch": args.batch, "phase": args.phase, "runs_root": args.runs_root, "output": str(output),
              "repo_dir": str(repo), "jobs": [name for name, _ in jobs], "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
