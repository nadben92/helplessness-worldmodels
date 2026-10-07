"""The results handoff of a GPU batch (docs/e3/DECISIONS.md, 2026-09-30 (3);
RUN_GPU.md step 5), so the person or assistant running the batch has nothing
to write by hand: from the supervisor's output directory (status.json and the
commands of every run) and the run directories under the results root, it
writes into --out

    handoff.md                      status of every run (exit code, meta.yaml status, gradient steps
                                    done, switch step, error), the preflight's decision, the archives;
    <run directory name>.tgz        one light archive per run: everything except checkpoints/,
                                    failure/*.pt, any .pt file, wandb/, measures/ and replay_final.npz;
    <run directory name>.measures.tgz   the run's checkpoint measures (measures/env<NNNNNN>/: results.json,
                                    arrays.npz, artifacts), when it has some (the campaign, DECISIONS 2026-10-07 (29));
    <run directory name>.weights.tgz    with --with-weights: the weights at every checkpoint and the final ones
                                    (checkpoints/agent_env<NNNNNN>.pt, agent_final.pt) and the final replay
                                    (replay_final.npz) -- never resume.pt (optimisers, environment, streams);
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


def _excluded_from_logs(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """The logs of a measurement batch: everything but the per-run outputs (<measure dir>/runs/, archived run by run)
    and the preflight's smoke measurement (runs/smoke/) -- gpu_pilot_remeasure's logs.tgz reached 186 MB with them
    (docs/e3/DECISIONS.md, 2026-10-06 (25))."""
    parts = Path(member.name).parts
    if len(parts) >= 4 and parts[3] == "runs" and parts[2].startswith("measure-"):
        return None
    if "smoke" in parts:
        return None
    return _excluded(member)


def _excluded(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(member.name).parts
    if member.name.endswith(".pt") or "checkpoints" in parts or "wandb" in parts:
        return None
    return member


def _excluded_run(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """The light run archive: _excluded, and the measures and final replay, which have their own archives."""
    parts = Path(member.name).parts
    if "measures" in parts or member.name.endswith("replay_final.npz"):
        return None
    return _excluded(member)


def _weights_only(member: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """The weights archive: the checkpoint and final weights and the final replay; never resume.pt or failure/."""
    name = Path(member.name).name
    if member.isdir():
        return member
    if name.startswith("agent_") and name.endswith(".pt") or name == "replay_final.npz":
        return member
    return None


def run_of_job(command: list[str], started_at: str | None, results_root: Path) -> Path | None:
    """The run directory a supervisor job wrote: run_mini.py --config <c> --seeds <n> --conditions <cond>
    -> results_root/*__pair-<role>__worldmodel-<config stem>__seed<nn>__<UTC>, the latest one created at or
    after the job's start (to the minute)."""
    config = Path(command[command.index("--config") + 1]).stem
    if any(c.endswith("run_agent.py") for c in command):  # closed-loop runs are named worldmodel-agent-<config stem>
        config = f"agent-{config}"
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
        row.update(status=meta.get("status"), steps_done=meta.get("steps_done") or meta.get("env_steps"), switch_step=meta.get("switch_step"),
                   error=error.splitlines()[0][:300] if error else "", finished=(run_dir / "results.json").exists(),
                   measured_checkpoints=len(list((run_dir / "measures").glob("env*/results.json"))))
    return row


def measure_row(name: str, job: dict, measure_dir: Path) -> tuple[dict, Path | None]:
    """A measurement job (recompute_measures.py <source run> ... --out-dir <measure dir>/runs): its row and the
    directory to archive (<measure dir>/runs/<run name>), None when the job wrote nothing."""
    command = job["command"]
    if any(c.endswith("measure_noise.py") for c in command):  # measurement noise / goal control (2026-10-06 (26))
        out = Path(command[command.index("--out-dir") + 1])
        done = out.exists() and any(out.glob("*.json"))
        row = {"job": name, "exit_code": job.get("exit_code"), "not_started": bool(job.get("not_started")), "run_dir": out.name,
               "output": f"{out.name}/*.json" if done else "none", "finished": done}
        return row, (out if done else None)
    if any(c.endswith("diagnose_value_generalisation.py") for c in command):  # the read-only value diagnostic (2026-10-05 (24))
        out = Path(command[command.index("--out-dir") + 1])
        done = (out / "value_diagnostics.json").exists()
        row = {"job": name, "exit_code": job.get("exit_code"), "not_started": bool(job.get("not_started")), "run_dir": "value_diagnostics",
               "output": "value_diagnostics/value_diagnostics.json" if done else "none", "finished": done}
        return row, (out if done else None)
    script = next(i for i, c in enumerate(command) if c.endswith(("recompute_measures.py", "remeasure_checkpoints.py")))
    run_name = Path(command[script + 1]).name
    out = measure_dir / "runs" / run_name
    row = {"job": name, "exit_code": job.get("exit_code"), "started_at": job.get("started_at"), "finished_at": job.get("finished_at"),
           "not_started": bool(job.get("not_started")), "run_dir": run_name, "finished": False}
    if command[script].endswith("remeasure_checkpoints.py"):  # one directory per checkpoint (2026-10-05 (23))
        asked = command[command.index("--env-steps") + 1:command.index("--device")]
        done = [c for c in asked if (out / f"env{int(c):06d}" / "results.json").exists()]
        row.update(output=f"{run_name}/env*" if done else "none", finished=len(done) == len(asked), checkpoints_done=f"{len(done)}/{len(asked)}")
        if done:
            res = json.loads((out / f"env{int(done[0]):06d}" / "results.json").read_text(encoding="utf-8"))
            row["source_sha256"] = (res.get("reproducibility") or {}).get("source_sha256")
        return row, (out if done else None)
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
    parser.add_argument("--with-weights", action="store_true",
                        help="also one <run>.weights.tgz per run: checkpoint and final weights, final replay (the campaign)")
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
        if any(c.endswith("campaign_measure_pilot.py") for c in job["command"]):  # a pilot run measured as a campaign seed (30)
            cmd = job["command"]
            run_dir = Path(cmd[cmd.index("--out-dir") + 1]) / Path(cmd[cmd.index("--config") - 1]).name
            run_dir = run_dir if not job.get("not_started") and run_dir.exists() else None
        else:
            run_dir = run_of_job(job["command"], job.get("started_at"), args.results_root) if not job.get("not_started") else None
        rows.append(run_row(name, job, run_dir))
        if run_dir is not None:
            arcname = f"results/E3_worldmodel_4zones/{run_dir.name}"
            archive = args.out / f"{run_dir.name}.tgz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(run_dir, arcname=arcname, filter=_excluded_run)
            archives.append(archive)
            if (run_dir / "measures").is_dir():
                archive = args.out / f"{run_dir.name}.measures.tgz"
                with tarfile.open(archive, "w:gz") as tar:
                    tar.add(run_dir / "measures", arcname=f"{arcname}/measures", filter=_excluded)
                archives.append(archive)
            if args.with_weights:
                archive = args.out / f"{run_dir.name}.weights.tgz"
                with tarfile.open(archive, "w:gz") as tar:
                    if (run_dir / "checkpoints").is_dir():
                        tar.add(run_dir / "checkpoints", arcname=f"{arcname}/checkpoints", filter=_weights_only)
                    if (run_dir / "replay_final.npz").exists():
                        tar.add(run_dir / "replay_final.npz", arcname=f"{arcname}/replay_final.npz")
                archives.append(archive)
    logs = args.out / "logs.tgz"
    with tarfile.open(logs, "w:gz") as tar:
        for d in [*([supervisor] if supervisor else []), *args.preflight_dir]:
            tar.add(d, arcname=f"logs/kubeflow/{d.name}", filter=_excluded_from_logs if args.measure_dir else _excluded)
            submission = d.parent / f"{d.name}.submission.json"  # the submission record (job id, requested CPUs, runs)
            if submission.exists():
                tar.add(submission, arcname=f"logs/kubeflow/{submission.name}")
    archives.append(logs)
    for d in args.preflight_dir:  # the closed-loop preflight's speed bench, also at the top of the handoff (informative)
        for bench in sorted(d.rglob("bench_agent.json")):
            (args.out / "bench_agent.json").write_bytes(bench.read_bytes())
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
             + ("Weights (.pt) only in the *.weights.tgz archives (checkpoint and final weights, final replay; no resume.pt)."
              if args.with_weights else "No weights (.pt) in any archive."), "", "## Preflight", ""]
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
        lines += ["| job | exit code | status (meta.yaml) | steps done | switch step | finished (results.json) | checkpoints measured | error |",
                  "|---|---|---|---|---|---|---|---|"]
    for r in rows_for_train:
        if r["not_started"]:
            lines.append(f"| {r['job']} | - | not started | - | - | - | - | - |")
            continue
        lines.append(f"| {r['job']} | {r['exit_code']} | {r.get('status', 'no run directory')} | {r.get('steps_done') or '-'} | "
                     f"{r.get('switch_step') if r.get('switch_step') is not None else 'none'} | {'yes' if r.get('finished') else 'no'} | "
                     f"{r.get('measured_checkpoints', '-')} | {r.get('error') or '-'} |")
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
