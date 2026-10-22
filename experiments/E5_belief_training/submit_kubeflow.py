"""Submit an E5 batch (batches.WAVES) to UCL ARC Kubeflow: one job, one GPU, under the library's supervisor
(helplessness.cluster.kubeflow.supervise), which writes <output>/status.json (every job's command, exit code, start and
end) and one log per job. E4's submission (experiments/E4_generalization_gradient/submit_kubeflow.py) with E5's batch
parameters.

Phases:
  train    the batch's four runs in parallel (batches.train_args);
  resume   with --resume <run dir>: that run, continued from its checkpoints/resume.pt; without --resume (and with
           --configuration): every run of the batch whose meta.yaml status is not "done" (the newest run directory of
           each job), all in parallel in one pod, each with the single run's command and resume point; a done run is
           never relaunched. Before anything is submitted (and in --dry-run), each run's resume point is checked on the
           CPU (helplessness.cluster.resume: latest.json, resume.pt complete and loaded entirely, the same step); on any
           problem nothing is submitted;
  probes   measure_probes.py at one checkpoint k (--checkpoint; post-switch steps for the immunisation) for every pair
           of the batch (batches.PROBE_PAIRS), batches.PROBES_MAX_PARALLEL at a time. Never launched automatically.

Batch parameters: --configuration (every phase: the E4 configuration of E5), --e4-results (probes of the baseline A: E4's
results root). --switch-step / --post-steps serve the immunisation only, deferred to E8 (batches.DEFERRED).

E5 is launched from its own clone (/scratch/helplessness-worldmodels-e5, branch e5; the e4 clone keeps E4's runs and
probes): the checkout must be on branch e5 with no modified tracked file (refused otherwise). Results stay on the volume
under <repo-dir>/results/E5_belief_training/ and come back with export_batch.py. No push (E4's optional push is not
offered here).

Usage (Kubeflow notebook, from the e5 checkout): see experiments/E5_belief_training/RUN_GPU.md.
  python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --configuration <id> --repo-dir /scratch/helplessness-worldmodels-e5 [--dry-run]
  python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --configuration <id> --phase resume --repo-dir ... [--dry-run]
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
import e5_names
from export_batch import seeds
from helplessness.cluster import batch_status
from helplessness.cluster import resume as cluster_resume
from helplessness.cluster.kubeflow import runtime_patch_secrets, supervise

IMAGE = "ghcr.io/ucl-arc-unified-ai/kubeflow-trainer-images-torch:v0.0.1"  # E3's and E4's image
SCRIPT = "experiments/E5_belief_training"
BRANCH = "e5"
RESULTS_SUBDIR = "results"
PHASES = ("train", "resume", "probes")


def commands(phase: str, python: str, batch: str, configuration: str, resume: str = "", checkpoint: int | None = None,
             switch_step: int | None = None, post_steps: int | None = None, e4_results: str | None = None,
             device: str = "cuda", n_walks: int | None = None, resume_runs: dict[str, str] | None = None) -> list[tuple[str, list[str]]]:
    """The supervisor's jobs. device: cuda on the cluster; cpu only to try the phases locally (--worker --device cpu).
    Phase resume: one run (resume, job "resume") or several (resume_runs: job name -> run directory, jobs
    "resume_<job name>"), each with the single run's command."""
    if phase == "resume":
        if bool(resume) == bool(resume_runs):
            raise ValueError("resume requires one run directory or the batch's runs to resume, not both")
        if resume:
            return [("resume", [python, "-u", f"{SCRIPT}/run_e5.py", "--resume", resume, "--device", device])]
        return [(f"resume_{name}", [python, "-u", f"{SCRIPT}/run_e5.py", "--resume", d, "--device", device])
                for name, d in resume_runs.items()]
    if not configuration:
        raise ValueError("--configuration is required (the E4 configuration of E5)")
    if phase == "probes":
        if checkpoint is None:
            raise ValueError("probes requires --checkpoint <k>")
        return [(name, [python, "-u", f"{SCRIPT}/measure_probes.py", *args, "--device", device, "--out-dir", RESULTS_SUBDIR, "--save-steps",
                        *([] if n_walks is None else ["--n-walks", str(n_walks)])])
                for name, args in batches.probe_args(batch, configuration, checkpoint, e4_results)]
    if phase != "train":
        raise ValueError(f"unknown phase {phase!r}")
    return [(job, [python, "-u", f"{SCRIPT}/run_e5.py", *batches.train_args(batch, job, configuration, switch_step, post_steps),
                   "--device", device, "--out-dir", RESULTS_SUBDIR])
            for job in batches.batch_runs(batch)]


def batch_run_dirs_by_job(repo: Path, batch: str, configuration: str) -> dict[str, Path | None]:
    """job name -> the newest run directory of that job under <repo>/results/E5_belief_training/ (None if none)."""
    seed_of = seeds(repo, configuration)
    root = repo / RESULTS_SUBDIR / e5_names.EXPERIMENT_ID
    out = {}
    for job, (label, training, fraction, condition) in batches.batch_runs(batch).items():
        found = sorted(root.glob(e5_names.run_glob(configuration, training, condition, seed_of[label], fraction)))
        out[job] = found[-1] if found else None
    return out


def resume_plan(repo: Path, batch: str, configuration: str, resume: str) -> tuple[dict[str, str] | None, list[dict]]:
    """The runs to resume and their checks (helplessness.cluster.resume)."""
    return cluster_resume.resume_plan(repo, {} if resume else batch_run_dirs_by_job(repo, batch, configuration), resume, batch)


def _run_e5(repo_dir: str, phase: str, output: str, batch: str, configuration: str = "", resume: str = "", max_parallel: int = 0,
            checkpoint: int = 0, switch_step: int = 0, post_steps: int = 0, e4_results: str = "", n_walks: int = 0,
            resume_runs: str = "") -> None:
    """Serialized by CustomTrainer: imports and constants must be self-contained (as E4's _run_e4)."""
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
    args = [python, "-u", "experiments/E5_belief_training/submit_kubeflow.py", "--worker", "--phase", phase, "--batch", batch,
            "--repo-dir", repo_dir, "--output", output, "--resume", resume, "--configuration", configuration]
    for flag, value in (("--max-parallel", max_parallel), ("--checkpoint", checkpoint), ("--switch-step", switch_step),
                        ("--post-steps", post_steps), ("--n-walks", n_walks)):
        if value:
            args += [flag, str(value)]
    if e4_results:
        args += ["--e4-results", e4_results]
    if resume_runs:
        args += ["--resume-runs", resume_runs]
    subprocess.run(args, cwd=repo, check=True)


def worker(args) -> int:
    jobs = commands(args.phase, sys.executable, args.batch, args.configuration, args.resume, args.checkpoint, args.switch_step,
                    args.post_steps, args.e4_results, args.device, args.n_walks, cluster_resume.decode_resume_runs(args.resume_runs))
    max_parallel = args.max_parallel or (batches.PROBES_MAX_PARALLEL if args.phase == "probes" else None)
    return supervise(jobs, Path(args.repo_dir), Path(args.output), max_parallel=max_parallel)


def check_checkout(repo: Path) -> None:
    """The checkout must be on branch e5 with no modified tracked file."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout.strip()

    branch, dirty = git("rev-parse", "--abbrev-ref", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    if branch != BRANCH:
        raise SystemExit(f"{repo} is on branch {branch!r}, not {BRANCH}")
    if dirty:
        raise SystemExit(f"{repo} has modified tracked files:\n{dirty}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(batches.BATCHES), default=None, help="required, except with --status")
    parser.add_argument("--status", type=Path, default=None,
                        help="print the state of a submitted batch (its supervisor directory), with a STALE flag, and exit")
    parser.add_argument("--stale-after", type=float, default=batch_status.STALE_AFTER, help="--status: seconds without a log change")
    parser.add_argument("--phase", choices=PHASES, default="train")
    parser.add_argument("--repo-dir", default=None, help="the e5 checkout on the persistent volume (required, except with --status)")
    parser.add_argument("--configuration", default=batches.CONFIGURATION,
                        help=f"the E4 configuration of E5 (default and only accepted value: {batches.CONFIGURATION})")
    parser.add_argument("--switch-step", type=int, default=None, help="the immunisation's switch step (deferred to E8)")
    parser.add_argument("--post-steps", type=int, default=None, help="the immunisation's steps after the switch (deferred to E8)")
    parser.add_argument("--e4-results", default=None, help="probes: E4's results root (immunisation A's reference)")
    parser.add_argument("--checkpoint", type=int, default=None, help="phase probes: k (post-switch steps for the immunisation)")
    parser.add_argument("--n-walks", type=int, default=None, help="phase probes: the first N walks only (a quick timing)")
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--resume", default="", help="phase resume: one run directory (default: every run of the batch not done)")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--output", help="supervisor directory (default: <repo>/logs/kubeflow/e5-<phase>-<batch>-<UTC>)")
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and the runtime patch, submit nothing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--resume-runs", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.status is not None:  # helplessness.cluster.batch_status: done / failed / running / STALE / not started
        rows = batch_status.batch_status(args.status, args.stale_after)
        print("\n".join(batch_status.format_status(rows, args.stale_after)), flush=True)
        return 2 if any(r["state"] == "STALE" for r in rows) else 0
    if args.batch is None:
        parser.error("--batch is required")
    if args.repo_dir is None:
        parser.error("--repo-dir is required")
    if args.configuration and args.configuration != batches.CONFIGURATION and not args.worker:
        parser.error(f"E5's configuration is {batches.CONFIGURATION} (docs/e5/DECISIONS.md (21)), not {args.configuration!r}")
    if args.phase in ("train", "resume") and not batches.batch_runs(args.batch):
        parser.error(f"{args.batch} trains nothing: it has the probes phase only")
    if args.resume and args.phase != "resume":
        parser.error("--resume goes with --phase resume")
    if args.phase == "resume" and not args.resume and not args.configuration and not args.worker:
        parser.error("--phase resume without --resume needs --configuration (to find the batch's runs)")
    if (args.phase == "probes") != (args.checkpoint is not None):
        parser.error("--checkpoint is required exactly when --phase=probes")
    if args.worker:
        return worker(args)
    repo = Path(args.repo_dir).resolve()
    resume_runs = None
    if args.phase == "resume":  # checked before anything is printed as a plan or submitted
        resume_runs, checks = resume_plan(repo, args.batch, args.configuration, args.resume)
        if not cluster_resume.report_checks(args.batch, checks, log=lambda line: print(line, flush=True)):
            return 1
    try:
        commands(args.phase, "python", args.batch, args.configuration, args.resume, args.checkpoint, args.switch_step,
                 args.post_steps, args.e4_results, "cuda", args.n_walks, resume_runs)
    except ValueError as err:
        parser.error(str(err))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"e5-{args.phase}-{args.batch}-{stamp}"
    patch = runtime_patch_secrets(args.volume, args.mount, {})
    max_parallel = args.max_parallel or (batches.PROBES_MAX_PARALLEL if args.phase == "probes" else None)
    jobs = commands(args.phase, "<venv python>", args.batch, args.configuration, args.resume, args.checkpoint, args.switch_step,
                    args.post_steps, args.e4_results, "cuda", args.n_walks, resume_runs)
    if args.dry_run:
        print(json.dumps({"batch": args.batch, "phase": args.phase, "configuration": args.configuration, "switch_step": args.switch_step,
                          "post_steps": args.post_steps, "cpus": batches.BATCH_CPUS[args.batch], "max_parallel": max_parallel,
                          "output": str(output), "jobs": jobs, "runtime_patch": patch}, indent=1))
        return 0
    check_checkout(repo)
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
            func=_run_e5, image=args.image, num_nodes=1,
            func_args={"repo_dir": str(repo), "phase": args.phase, "output": str(output), "batch": args.batch,
                       "configuration": args.configuration, "resume": args.resume, "max_parallel": max_parallel or 0,
                       "checkpoint": args.checkpoint or 0, "switch_step": args.switch_step or 0, "post_steps": args.post_steps or 0,
                       "e4_results": args.e4_results or "", "n_walks": args.n_walks or 0,
                       "resume_runs": cluster_resume.encode_resume_runs(resume_runs)},
            resources_per_node={"cpu": str(cpus), "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "batch": args.batch, "phase": args.phase, "configuration": args.configuration,
              "switch_step": args.switch_step, "post_steps": args.post_steps, "output": str(output), "repo_dir": str(repo), "cpus": cpus,
              "jobs": [name for name, _ in jobs], "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs (RUN_GPU.md).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
