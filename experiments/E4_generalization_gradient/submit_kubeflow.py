"""Submit an E4 batch (batches.BATCHES) to UCL ARC Kubeflow: one job, one GPU, under the library's supervisor
(helplessness.cluster.kubeflow.supervise), which writes <output>/status.json (every job's command, exit code, start and
end) and one log per job.

Phases:
  train    the batch's four runs in parallel (wave 1 at the full budget; the other batches stop at B,
           batches.max_env_steps; checkpoint diagnostics as batches.diagnostics_at);
  resume   one run directory, continued from its checkpoints/resume.pt;
  probes   measure_probes.py at one checkpoint (--checkpoint <environment step>) for every configuration of the batch,
           or those given with --configurations: all the probes, on the exposed model and its reference, on the GPU,
           batches.PROBES_MAX_PARALLEL configurations at a time. Never launched automatically.

Results stay on the volume under <repo-dir>/results/E4_generalization_gradient/ and are brought back by
export_batch.py (one archive per batch). Optional, off by default: --push-secret <Kubernetes secret holding
RESULTS_PUSH_TOKEN> makes every run push its light results to the orphan branch results/<batch> when it ends
(docs/RESULTS_CONVENTION.md); it then needs an https checkout. Without it, no secret, no token and no push are involved.

The checkout on the volume must be on branch e4 with no modified tracked file (refused otherwise).

Usage (Kubeflow notebook, from the checkout): see experiments/E4_generalization_gradient/RUN_GPU.md.
  python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --repo-dir /scratch/<checkout> [--dry-run]
  python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase probes --checkpoint <B> --repo-dir ...
  python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase resume --resume <run dir> --repo-dir ...
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
PHASES = ("train", "resume", "probes")


def commands(phase: str, python: str, batch: str, push: dict | None = None, resume: str = "", checkpoint: int | None = None,
             configurations: list[str] | None = None, device: str = "cuda", n_walks: int | None = None) -> list[tuple[str, list[str]]]:
    """The supervisor's jobs. push: None (default, no push) or {"repo": url, "token_file": path}. device: cuda on the
    cluster; cpu only to try the phases locally (--worker --device cpu)."""
    push_args = [] if push is None else ["--push-batch", batch, "--push-repo", push["repo"], "--push-token-file", push["token_file"]]
    run = [python, "-u", f"{SCRIPT}/run_e4.py"]
    if phase == "resume":
        if not resume:
            raise ValueError("resume requires a run directory")
        return [("resume", run + ["--resume", resume, "--device", device, *push_args])]
    if phase == "probes":
        if checkpoint is None:
            raise ValueError("probes requires --checkpoint <environment step>")
        known = batches.BATCHES[batch]["configurations"]
        chosen = list(configurations or known)
        unknown = sorted(set(chosen) - set(known))
        if unknown:
            raise ValueError(f"configuration(s) {unknown} not in {batch}: {list(known)}")
        return [(f"probes_{cid}_env{int(checkpoint):06d}",
                 [python, "-u", f"{SCRIPT}/measure_probes.py", "--config", batches.CONFIG, "--configuration", cid,
                  "--checkpoint", str(int(checkpoint)), "--find-runs", RESULTS_SUBDIR, "--device", device,
                  "--out-dir", RESULTS_SUBDIR, "--save-steps", *([] if n_walks is None else ["--n-walks", str(n_walks)])])
                for cid in chosen]
    if phase != "train":
        raise ValueError(f"unknown phase {phase!r}")
    stop, diag = batches.max_env_steps(batch), batches.diagnostics_at(batch)
    return [(name, run + ["--config", batches.CONFIG, "--configuration", cid, "--condition", condition, "--device", device,
                          "--out-dir", RESULTS_SUBDIR, *([] if stop is None else ["--max-env-steps", str(stop)]),
                          *([] if diag is None else ["--diagnostics-at", *map(str, diag)]), *push_args])
            for name, (cid, condition) in batches.batch_runs(batch).items()]


def batch_run_dirs(repo: Path, batch: str, started_after: str = "") -> list[Path]:
    """The run directories of the batch (the newest per job, created after started_after if given)."""
    root = repo / RESULTS_SUBDIR / "E4_generalization_gradient"
    out = []
    for cid, condition in batches.batch_runs(batch).values():
        role = "experimental" if condition == "experimental" else "control"
        found = sorted(d for d in root.glob(f"*__2zones-{cid}__pair-{role}__*") if d.name[-16:] >= started_after)
        if found:
            out.append(found[-1])
    return out


def _run_e4(repo_dir: str, phase: str, output: str, batch: str, push_repo: str = "", token_dir: str = "", resume: str = "",
            max_parallel: int = 0, checkpoint: int = 0, configurations: str = "", n_walks: int = 0) -> None:
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
    args = [python, "-u", "experiments/E4_generalization_gradient/submit_kubeflow.py", "--worker", "--phase", phase, "--batch", batch,
            "--repo-dir", repo_dir, "--output", output, "--resume", resume]
    if push_repo:
        args += ["--push-repo", push_repo, "--push-token-file", f"{token_dir}/RESULTS_PUSH_TOKEN"]
    if max_parallel:
        args += ["--max-parallel", str(max_parallel)]
    if checkpoint:
        args += ["--checkpoint", str(checkpoint)]
    if configurations:
        args += ["--configurations", *configurations.split(",")]
    if n_walks:
        args += ["--n-walks", str(n_walks)]
    subprocess.run(args, cwd=repo, check=True)


def worker(args) -> int:
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo, output = Path(args.repo_dir), Path(args.output)
    push = {"repo": args.push_repo, "token_file": args.push_token_file} if args.push_repo else None
    jobs = commands(args.phase, sys.executable, args.batch, push, args.resume, args.checkpoint, args.configurations, args.device,
                    args.n_walks)
    max_parallel = args.max_parallel or (batches.PROBES_MAX_PARALLEL if args.phase == "probes" else None)
    code = supervise(jobs, repo, output, max_parallel=max_parallel)
    if push is not None and args.phase != "probes":  # the batch-level status, after every run pushed its own results
        from helplessness.cluster import results_push

        try:
            token = results_push.read_token(args.push_token_file)
            results_push.push_batch_status(output, batch_run_dirs(repo, args.batch, started), "E4_generalization_gradient", args.batch,
                                           args.push_repo, token)
        except Exception as err:
            print(f"[submit_kubeflow] BATCH STATUS PUSH FAILED: {type(err).__name__}: {err}", flush=True)
            code = code or 2
    return code


def check_checkout(repo: Path, need_https: bool) -> str:
    """The checkout must be on branch e4 with no modified tracked file; with a push, an https remote. Returns its URL."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()

    branch, dirty, url = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain", "--untracked-files=no"), git("remote", "get-url", "origin")
    if branch != "e4":
        raise SystemExit(f"{repo} is on branch {branch!r}, not e4")
    if dirty:
        raise SystemExit(f"{repo} has modified tracked files:\n{dirty}")
    if need_https and not url.startswith("https://"):
        raise SystemExit(f"origin is {url!r}: the results push needs an https remote")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(batches.BATCHES), required=True)
    parser.add_argument("--phase", choices=PHASES, default="train")
    parser.add_argument("--repo-dir", required=True, help="the e4 checkout on the persistent volume")
    parser.add_argument("--checkpoint", type=int, default=None, help="phase probes: the environment step of the weights")
    parser.add_argument("--configurations", nargs="+", default=None, help="phase probes: a subset of the batch's configurations")
    parser.add_argument("--n-walks", type=int, default=None, help="phase probes: the first N walks only (a quick timing)")
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--push-secret", default=None, help="OPTIONAL: Kubernetes secret holding RESULTS_PUSH_TOKEN (push to results/<batch>)")
    parser.add_argument("--resume", default="", help="run directory (phase resume)")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--output", help="supervisor directory (default: <repo>/logs/kubeflow/e4-<phase>-<batch>-<UTC>)")
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and the runtime patch, submit nothing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--push-repo", default="", help=argparse.SUPPRESS)
    parser.add_argument("--push-token-file", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.phase == "resume") != bool(args.resume):
        parser.error("--resume is required exactly when --phase=resume")
    if (args.phase == "probes") != (args.checkpoint is not None):
        parser.error("--checkpoint is required exactly when --phase=probes")
    if args.worker:
        return worker(args)
    repo = Path(args.repo_dir).resolve()
    push = {"repo": "https://<origin>", "token_file": f"{PUSH_TOKEN_DIR}/{PUSH_TOKEN_KEY}"} if args.push_secret else None
    try:
        commands(args.phase, "python", args.batch, push, args.resume or "x", args.checkpoint, args.configurations, "cuda", args.n_walks)
    except ValueError as err:
        parser.error(str(err))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"e4-{args.phase}-{args.batch}-{stamp}"
    patch = runtime_patch_secrets(args.volume, args.mount, {args.push_secret: PUSH_TOKEN_DIR} if args.push_secret else {})
    max_parallel = args.max_parallel or (batches.PROBES_MAX_PARALLEL if args.phase == "probes" else None)
    if args.dry_run:
        jobs = commands(args.phase, "<venv python>", args.batch, push, args.resume, args.checkpoint, args.configurations, "cuda", args.n_walks)
        print(json.dumps({"batch": args.batch, "phase": args.phase, "cpus": batches.BATCH_CPUS[args.batch], "max_parallel": max_parallel,
                          "push": push is not None, "output": str(output), "jobs": jobs, "runtime_patch": patch}, indent=1))
        return 0
    url = check_checkout(repo, need_https=push is not None)
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
                       "push_repo": url if push else "", "token_dir": PUSH_TOKEN_DIR if push else "", "resume": args.resume,
                       "max_parallel": max_parallel or 0, "checkpoint": args.checkpoint or 0,
                       "configurations": ",".join(args.configurations or []), "n_walks": args.n_walks or 0},
            resources_per_node={"cpu": str(cpus), "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "batch": args.batch, "phase": args.phase, "output": str(output), "repo_dir": str(repo), "cpus": cpus,
              "jobs": [name for name, _ in commands(args.phase, "python", args.batch, push, args.resume, args.checkpoint, args.configurations)],
              "push": push is not None, "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs (RUN_GPU.md).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
