"""The results handoff of a GPU batch (docs/e3/DECISIONS.md, 2026-09-30 (3);
RUN_GPU.md step 5), so the person or assistant running the batch has nothing
to write by hand: from the supervisor's output directory (status.json and the
commands of every run) and the run directories under the results root, it
writes into --out

    handoff.md                      status of every run (exit code, meta.yaml status, gradient steps
                                    done, switch step, error), the preflight's decision, the archives;
    <run directory name>.tgz        one light archive per run: everything except checkpoints/,
                                    failure/*.pt, any .pt file and wandb/;
    logs.tgz                        the supervisor's logs and status.json, and the preflight's
                                    directories without their .pt files.

Every archive must stay below MAX_ARCHIVE_MB (GitHub refuses files over 100 MB): the script stops
otherwise. It never writes outside --out, and refuses an --out that already holds files.

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/make_handoff.py --train-dir logs/kubeflow/train-<UTC> \\
        [--preflight-dir logs/kubeflow/preflight-<UTC> ...] --out handoff/gpu_e30b
    python experiments/E3_worldmodel_4zones/make_handoff.py --measure-dir logs/kubeflow/measure-<UTC> \\
        [--preflight-dir logs/kubeflow/preflight-<UTC> ...] --out handoff/gpu_e30b_measures

A measurement batch (submit_kubeflow.MEASURE_BATCHES, docs/e3/DECISIONS.md 2026-10-01 (10)) is
handed off with --measure-dir instead of --train-dir: one light archive per run,
<run directory name>.recomputed.tgz, of the recomputation the job wrote under <measure dir>/runs/
(results.json, arrays.npz, SUMMARY.md, comparison_with_run.json, console.log, artifacts); the
original run directories are not read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "E3_worldmodel_4zones"
MAX_ARCHIVE_MB = 90.0
PAIR_ROLE = {"experimental": "experimental", "reference": "control"}


def _excluded(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(member.name).parts
    if member.name.endswith(".pt") or "checkpoints" in parts or "wandb" in parts:
        return None
    return member


def run_of_job(command: list[str], started_at: str | None, results_root: Path) -> Path | None:
    """The run directory a supervisor job wrote: run_mini.py --config <c> --seeds <n> --conditions <cond>
    -> results_root/*__pair-<role>__worldmodel-<config stem>__seed<nn>__<UTC>, the latest one created at or
    after the job's start (to the minute)."""
    config = Path(command[command.index("--config") + 1]).stem
    seed = int(command[command.index("--seeds") + 1])
    role = PAIR_ROLE[command[command.index("--conditions") + 1]]
    candidates = sorted(results_root.glob(f"*__pair-{role}__worldmodel-{config}__seed{seed:02d}__*"))
    if started_at:
        start = datetime.fromisoformat(started_at).strftime("%Y%m%dT%H%M")
        candidates = [c for c in candidates if re.search(r"__(\d{8}T\d{4})\d{2}Z$", c.name)
                      and re.search(r"__(\d{8}T\d{4})\d{2}Z$", c.name).group(1) >= start]
    return candidates[-1] if candidates else None


def run_row(name: str, job: dict, run_dir: Path | None) -> dict:
    row = {"job": name, "exit_code": job.get("exit_code"), "started_at": job.get("started_at"), "finished_at": job.get("finished_at"),
           "not_started": bool(job.get("not_started")), "run_dir": run_dir.name if run_dir else None}
    if run_dir is not None and (run_dir / "meta.yaml").exists():
        meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
        error = str(meta.get("error") or "")
        row.update(status=meta.get("status"), steps_done=meta.get("steps_done"), switch_step=meta.get("switch_step"),
                   error=error.splitlines()[0][:300] if error else "", finished=(run_dir / "results.json").exists())
    return row


def measure_row(name: str, job: dict, measure_dir: Path) -> tuple[dict, Path | None]:
    """A measurement job (recompute_measures.py <source run> ... --out-dir <measure dir>/runs): its row and the
    directory to archive (<measure dir>/runs/<run name>), None when the job wrote nothing."""
    command = job["command"]
    script = next(i for i, c in enumerate(command) if c.endswith("recompute_measures.py"))
    run_name = Path(command[script + 1]).name
    out = measure_dir / "runs" / run_name
    row = {"job": name, "exit_code": job.get("exit_code"), "started_at": job.get("started_at"), "finished_at": job.get("finished_at"),
           "not_started": bool(job.get("not_started")), "run_dir": run_name, "finished": False}
    outputs = sorted((out / "recomputed").glob("final__*")) if out.exists() else []
    if outputs:
        last = outputs[-1]
        row["output"] = f"{run_name}/recomputed/{last.name}"
        row["finished"] = (last / "results.json").exists() and (last / "arrays.npz").exists()
        if (last / "comparison_with_run.json").exists():
            same = json.loads((last / "comparison_with_run.json").read_text(encoding="utf-8"))
            row["identical_to_original"] = all(same.values())
        if (last / "results.json").exists():
            res = json.loads((last / "results.json").read_text(encoding="utf-8"))
            row["source_sha256"] = (res.get("reproducibility") or {}).get("source_sha256")
            row["weights_sha256"] = res.get("weights_sha256")
    return row, (out if outputs else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=None,
                        help="the training supervisor's output directory (with status.json); omit it when the training was not launched")
    parser.add_argument("--measure-dir", type=Path, default=None,
                        help="a measurement batch's supervisor output directory (instead of --train-dir)")
    parser.add_argument("--preflight-dir", type=Path, nargs="*", default=[], help="the preflight's output directories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} already holds files: choose a new directory")
    if args.train_dir and args.measure_dir:
        parser.error("--train-dir and --measure-dir: one batch at a time")
    supervisor = args.train_dir or args.measure_dir
    status = json.loads((supervisor / "status.json").read_text(encoding="utf-8")) if supervisor else {}
    args.out.mkdir(parents=True, exist_ok=True)

    rows, archives = [], []
    for name, job in (status.items() if args.measure_dir else ()):
        row, out = measure_row(name, job, args.measure_dir) if not job.get("not_started") else ({"job": name, "not_started": True}, None)
        rows.append(row)
        if out is not None:
            archive = args.out / f"{out.name}.recomputed.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(out, arcname=out.name, filter=_excluded)
            archives.append(archive)
    for name, job in (status.items() if args.train_dir else ()):
        run_dir = run_of_job(job["command"], job.get("started_at"), args.results_root) if not job.get("not_started") else None
        rows.append(run_row(name, job, run_dir))
        if run_dir is not None:
            archive = args.out / f"{run_dir.name}.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(run_dir, arcname=f"results/E3_worldmodel_4zones/{run_dir.name}", filter=_excluded)
            archives.append(archive)
    logs = args.out / "logs.tgz"
    with tarfile.open(logs, "w:gz") as tar:
        for d in [*([supervisor] if supervisor else []), *args.preflight_dir]:
            tar.add(d, arcname=f"logs/kubeflow/{d.name}", filter=_excluded)
            submission = d.parent / f"{d.name}.submission.json"  # the submission record (job id, requested CPUs, runs)
            if submission.exists():
                tar.add(submission, arcname=f"logs/kubeflow/{submission.name}")
    archives.append(logs)
    too_big = [(a.name, a.stat().st_size / 2**20) for a in archives if a.stat().st_size / 2**20 > MAX_ARCHIVE_MB]

    decisions = []
    for d in args.preflight_dir:
        for verdict_path in sorted(d.rglob("preflight_verdict.json")):
            v = json.loads(verdict_path.read_text(encoding="utf-8"))
            cpus = v.get("cpus") or {}
            decisions.append(f"`{d.name}`: decision **{v['decision']}**, parallel {v['parallel']}, CPUs requested / usable "
                             f"{cpus.get('requested', '?')} / {cpus.get('usable', '?')}, projected batch hours "
                             f"{v.get('projected_batch_hours')}, problems {v['problems'] or 'none'}, warnings {v['warnings'] or 'none'}")
    lines = [f"# GPU batch handoff -- {args.out.name}", "",
             f"Written {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by make_handoff.py from "
             f"`{supervisor.name if supervisor else 'no training / measurement (not launched)'}`. "
             "No weights (.pt) in any archive.", "", "## Preflight", ""]
    lines += [f"- {d}" for d in decisions] or ["- (no preflight directory given)"]
    lines += ["", "## Runs", ""]
    rows_for_train = rows
    if args.measure_dir:
        rows_for_train = []
        lines += ["Measurement batch: no training; each run re-measured by recompute_measures.py from its own weights.", "",
                  "| job | exit code | run | output | finished | identical to the original arrays | code source_sha256 | weights SHA-256 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['job']} | {'not started' if r.get('not_started') else r.get('exit_code')} | {r.get('run_dir', '-')} | "
                         f"{r.get('output', 'none')} | {'yes' if r.get('finished') else 'no'} | {r.get('identical_to_original', '-')} | "
                         f"{r.get('source_sha256', '-')} | {r.get('weights_sha256', '-')} |")
    elif not args.train_dir:
        lines.append("Training not launched (see the preflight above).")
    else:
        lines += ["| job | exit code | status (meta.yaml) | gradient steps done | switch step | finished (results.json) | error |",
                  "|---|---|---|---|---|---|---|"]
    for r in rows_for_train:
        if r["not_started"]:
            lines.append(f"| {r['job']} | - | not started | - | - | - | - |")
            continue
        lines.append(f"| {r['job']} | {r['exit_code']} | {r.get('status', 'no run directory')} | {r.get('steps_done') or '-'} | "
                     f"{r.get('switch_step') if r.get('switch_step') is not None else 'none'} | {'yes' if r.get('finished') else 'no'} | "
                     f"{r.get('error') or '-'} |")
    lines += ["", "## Archives", ""] + [f"- `{a.name}`: {a.stat().st_size / 2**20:.1f} MB" for a in archives]
    (args.out / "handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "handoff.json").write_text(json.dumps({"runs": rows, "archives": [a.name for a in archives]}, indent=1), encoding="utf-8")
    print("\n".join(lines))
    if too_big:
        print(f"STOP: archive(s) above {MAX_ARCHIVE_MB:g} MB, not committable: {too_big}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
