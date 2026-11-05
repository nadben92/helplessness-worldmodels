"""Submit an E8 batch (batches.BATCHES) to UCL ARC Kubeflow: one pod (3 CPUs, one GPU, E5's), its jobs under the library's
supervisor (helplessness.cluster.kubeflow.supervise: <output>/status.json, one log per job). E5's submission with E8's
batches and phases. `nadim-launcher` must be stopped while a batch runs (one pod at a time, the quota).

Phases:
  train    the batch's four therapy runs in parallel (batches.train_args; run_e8.py --phase train, the runs root = the
           mount: the start runs live in /scratch/helplessness-worldmodels-e4/... and -e5/..., configs/e8/agents.yaml);
  resume   every run of the batch whose meta.yaml status is not "done" (the newest run directory of each job), from its
           checkpoints/resume.pt, after the library's read-only resume check (helplessness.cluster.resume); with --resume
           <run dir>, that run only. On any problem nothing is submitted;
  probes   run_e8.py --phase probes, one job per probe job of the batch (the pairs sharing a reference) and checkpoint
           (--checkpoint k [k ...], or `all`: every checkpoint of the runs' grid), PROBES_MAX_PARALLEL at a time.

Only batches.LAUNCHABLE batches are submitted (wave 2 after wave 1's validation); --dry-run prints the jobs of any batch.
Launched from E8's own clone (/scratch/helplessness-worldmodels-e8, branch e8, no modified tracked file). Results stay
under <repo-dir>/results/E8_therapy_relapse/ and come back with export_batch.py. No push.

Usage (Kubeflow notebook, from the e8 checkout; always a --dry-run first):
  python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --repo-dir /scratch/helplessness-worldmodels-e8 --dry-run
  python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --repo-dir /scratch/helplessness-worldmodels-e8
  python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --phase resume --repo-dir ... [--dry-run]
  python experiments/E8_therapy_relapse/submit_kubeflow.py --batch e8_w1_base --phase probes --checkpoint all --repo-dir ... [--dry-run]
  python experiments/E8_therapy_relapse/submit_kubeflow.py --status <supervisor directory>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:  # the notebook's Python runs this script without the package installed
    sys.path.insert(0, str(_SRC))

import batches  # noqa: E402
import e8_setup  # noqa: E402
from helplessness.cluster import batch_status  # noqa: E402
from helplessness.cluster import resume as cluster_resume  # noqa: E402
from helplessness.cluster.kubeflow import runtime_patch_secrets, supervise  # noqa: E402

IMAGE = "ghcr.io/ucl-arc-unified-ai/kubeflow-trainer-images-torch:v0.0.1"  # E3-E5's image
SCRIPT = "experiments/E8_therapy_relapse/run_e8.py"
BRANCH = "e8"
PHASES = ("train", "resume", "probes")


def batch_run_dirs(repo: Path, batch: str) -> dict[str, Path | None]:
    """model id -> the newest run directory of the batch's run of that model (None: none yet)."""
    models = e8_setup.load_models(repo / "configs" / "e8" / "agents.yaml")["models"]
    out = {}
    for model, q in batches.BATCHES[batch]["runs"].items():
        therapy = e8_setup.therapy_of(models[model], q)
        found = sorted((repo / "results" / e8_setup.EXPERIMENT_ID).glob(e8_setup.run_glob(models[model], therapy)))
        out[model] = found[-1] if found else None
    return out


def checkpoints_of(run_dirs: dict[str, Path | None]) -> list[int]:
    """Every checkpoint of the batch's grid (the resolved configs agree; refused otherwise)."""
    grids = {tuple(int(c) for c in yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["checkpoints"]["grid"])
             for d in run_dirs.values() if d is not None}
    if len(grids) != 1:
        raise ValueError(f"the batch's runs have {len(grids)} checkpoint grids")
    return list(grids.pop())


def commands(phase: str, python: str, batch: str, mount: str, device: str, resume_runs: dict[str, str] | None = None,
             run_dirs: dict[str, str] | None = None, checkpoints: list[int] | None = None) -> list[tuple[str, list[str]]]:
    base = [python, "-u", SCRIPT]
    if phase == "train":
        return [(model, [*base, *args, "--runs-root", mount, "--device", device]) for model, args in batches.train_args(batch).items()]
    if phase == "resume":
        return [(f"resume_{job}", [*base, "--phase", "resume", "--run-dir", d, "--device", device]) for job, d in (resume_runs or {}).items()]
    agents = e8_setup.load_models(e8_setup.REPO_ROOT / "configs" / "e8" / "agents.yaml")
    jobs = []
    for pairs in batches.BATCHES[batch]["probe_jobs"]:
        models = sorted({agents["pairs"][p][r] for p in pairs for r in ("exposed", "reference")})
        missing = [m for m in models if not (run_dirs or {}).get(m)]
        if missing:
            raise ValueError(f"no run directory for {missing}")
        runs = [a for m in models for a in ("--run", f"{m}={run_dirs[m]}")]
        for k in checkpoints or []:
            jobs.append((f"probes_{'-'.join(pairs)}_k{k:06d}", [*base, "--phase", "probes", "--pairs", *pairs, *runs, "--checkpoint", str(k),
                                                               "--device", device]))
    return jobs


def _run_e8(repo_dir: str, phase: str, output: str, batch: str, mount: str, jobs_json: str, max_parallel: int = 0) -> None:
    """Serialized by CustomTrainer: imports and constants must be self-contained (as E5's _run_e5)."""
    import json
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
    jobs = [(name, [python if a == "<venv python>" else a for a in argv]) for name, argv in json.loads(jobs_json)]
    subprocess.run([python, "-u", "experiments/E8_therapy_relapse/submit_kubeflow.py", "--worker", "--batch", batch, "--phase", phase,
                    "--repo-dir", repo_dir, "--output", output, "--mount", mount, "--jobs-json", json.dumps(jobs),
                    "--max-parallel", str(max_parallel)], cwd=repo, check=True)


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
    parser.add_argument("--status", type=Path, default=None, help="print the state of a submitted batch (its supervisor directory) and exit")
    parser.add_argument("--stale-after", type=float, default=batch_status.STALE_AFTER)
    parser.add_argument("--phase", choices=PHASES, default="train")
    parser.add_argument("--repo-dir", default=None, help="the e8 checkout on the persistent volume")
    parser.add_argument("--resume", default="", help="phase resume: one run directory (default: every run of the batch not done)")
    parser.add_argument("--checkpoint", nargs="+", default=None, help="phase probes: k [k ...] or `all`")
    parser.add_argument("--volume", default="scratch-volume")
    parser.add_argument("--mount", default="/scratch")
    parser.add_argument("--memory", default="64Gi")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--output", help="supervisor directory (default: <repo>/logs/kubeflow/e8-<phase>-<batch>-<UTC>)")
    parser.add_argument("--dry-run", action="store_true", help="print the jobs and the runtime patch, submit nothing")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--jobs-json", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.status is not None:
        rows = batch_status.batch_status(args.status, args.stale_after)
        print("\n".join(batch_status.format_status(rows, args.stale_after)), flush=True)
        return 2 if any(r["state"] == "STALE" for r in rows) else 0
    if args.batch is None or args.repo_dir is None:
        parser.error("--batch and --repo-dir are required (except with --status)")
    if args.worker:
        return supervise([tuple(j) for j in json.loads(args.jobs_json)], Path(args.repo_dir), Path(args.output),
                         max_parallel=args.max_parallel or None)
    if (args.phase == "probes") != (args.checkpoint is not None):
        parser.error("--checkpoint is required exactly when --phase=probes")
    repo = Path(args.repo_dir).resolve()
    resume_runs, run_dirs, checkpoints = None, None, None
    if args.phase == "resume":
        resume_runs, checks = cluster_resume.resume_plan(repo, batch_run_dirs(repo, args.batch), args.resume, args.batch)
        if not cluster_resume.report_checks(args.batch, checks, log=lambda line: print(line, flush=True)):
            return 1
        if resume_runs is None:
            resume_runs = {"resume": args.resume}
    if args.phase == "probes":
        dirs = batch_run_dirs(repo, args.batch)
        run_dirs = {m: d.relative_to(repo).as_posix() for m, d in dirs.items() if d is not None}
        checkpoints = checkpoints_of(dirs) if args.checkpoint == ["all"] else [int(k) for k in args.checkpoint]
    jobs = commands(args.phase, "<venv python>", args.batch, args.mount, "cuda", resume_runs, run_dirs, checkpoints)
    max_parallel = args.max_parallel or (batches.PROBES_MAX_PARALLEL if args.phase == "probes" else None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(args.output).resolve() if args.output else repo / "logs" / "kubeflow" / f"e8-{args.phase}-{args.batch}-{stamp}"
    patch = runtime_patch_secrets(args.volume, args.mount, {})
    if args.dry_run:
        print(json.dumps({"batch": args.batch, "phase": args.phase, "launchable": args.batch in batches.LAUNCHABLE,
                          "cpus": batches.BATCH_CPUS[args.batch], "max_parallel": max_parallel, "output": str(output), "jobs": jobs,
                          "runtime_patch": patch, "reminder": "stop nadim-launcher before submitting (one 3-CPU pod: the quota)"}, indent=1))
        return 0
    if args.batch not in batches.LAUNCHABLE:
        parser.error(f"{args.batch} is not in batches.LAUNCHABLE (wave 2 waits for wave 1's validation; the later quadrants for the user)")
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
            func=_run_e8, image=args.image, num_nodes=1,
            func_args={"repo_dir": str(repo), "phase": args.phase, "output": str(output), "batch": args.batch, "mount": args.mount,
                       "jobs_json": json.dumps(jobs), "max_parallel": max_parallel or 0},
            resources_per_node={"cpu": str(cpus), "memory": args.memory, "nvidia.com/gpu": "1"},
        ),
    )
    record = {"job_id": job_id, "batch": args.batch, "phase": args.phase, "output": str(output), "repo_dir": str(repo), "cpus": cpus,
              "jobs": [name for name, _ in jobs], "submitted_at": datetime.now(timezone.utc).isoformat()}
    record_path = output.parent / f"{output.name}.submission.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print("Submitted, not yet confirmed running. Check the pod and its logs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
