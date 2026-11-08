"""E8's probes of an extracted batch on the local CPU, every job in one unattended process (E6's chain_all_cpu.py for E8's
probe jobs; docs/e8/DECISIONS.md (16)). Nothing of the measures is here: each job is run_e8.py --phase probes' own command
line (the pairs sharing a reference, one checkpoint), --workers at a time, one thread each.

  --logs/STATUS.txt   the jobs done / running / FAILED / pending, the error that stopped the chain, and a "heartbeat <time>"
                      line rewritten every HEARTBEAT seconds while the chain lives (a heartbeat older than a few minutes
                      with no ERROR line: the chain was killed from outside)
  --logs/state.json   the same, read back on a relaunch: a job marked done is skipped
  --logs/<job>.log    each job's output

A job that fails stops the chain once the running jobs end (nothing new is started); so does any exception of the chain
itself, KeyboardInterrupt included: the running jobs' process trees are killed and the jobs marked FAILED with the cause.
On a relaunch, a job that was not finished is rerun: its output directories with status "running" (never to be read)
are first moved aside to <out-dir>/E8_therapy_relapse/interrupted/<UTC stamp>/, nothing is deleted. The machine is kept
awake while the chain runs (Windows).

--smoke: the first job only, with --n-reps 1 --n-walks 3, into <out-dir>/smoke/ (the end-to-end check before the full
chain: it must finish; delete it afterwards). Its numbers are meaningless.

A batch with its own config (batches.BATCH_CONFIG, e.g. wave 2's) gets --config in every job.

Usage, from the repo root (src of this checkout on PYTHONPATH), on an extracted batch:
  python experiments/E8_therapy_relapse/chain_cpu.py --batch e8_w1_base --batch-root <extracted dir> --logs logs/e8_w1_cpu \\
      --out-dir <outputs root> [--workers 12] [--checkpoints all] [--smoke]
Registered as a Windows scheduled task for long chains (as E6's E7_650k_chain): pythonw, a clean console.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

import batches
import e8_setup

HEARTBEAT = 60  # seconds
_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if sys.platform == "win32":
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()


def write_status(logs: Path, state: dict) -> None:
    with _lock:
        counts = {s: sum(j["status"] == s for j in state["jobs"].values()) for s in ("done", "running", "FAILED", "pending")}
        lines = [f"heartbeat {now()}", f"updated {state.get('updated', '-')}", "  ".join(f"{k} {v}" for k, v in counts.items()), ""]
        lines += [f"{name:44s} {j['status']:8s} start {j.get('start', '-'):26s} end {j.get('end', '-')}" for name, j in state["jobs"].items()]
        if state.get("error"):
            lines += ["", "ERROR: " + state["error"]]
        try:
            (logs / "STATUS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (logs / "state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
        except OSError:  # a reader holding the file: the next heartbeat rewrites it
            pass


def heartbeat(logs: Path, state: dict, stop: threading.Event) -> None:
    while not stop.wait(HEARTBEAT):
        write_status(logs, state)


def batch_runs(batch: str, batch_root: Path) -> dict[str, Path]:
    """model id -> its therapy run directory in the extracted batch (the newest one, by the run's own e8.run.model)."""
    out = {}
    for d in sorted((batch_root / "results" / e8_setup.EXPERIMENT_ID).glob("grid12x12__*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
        model = cfg["e8"]["run"]["model"]
        if model in batches.BATCHES[batch]["runs"]:
            out[model] = d
    missing = sorted(set(batches.BATCHES[batch]["runs"]) - set(out))
    if missing:
        raise SystemExit(f"{batch_root}: no run of {missing}")
    return out


def jobs(batch: str, batch_root: Path, out_dir: Path, checkpoints: list[str], extra: list[str]) -> dict[str, list[str]]:
    runs = batch_runs(batch, batch_root)
    grid = sorted({int(c) for d in runs.values() for c in yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["checkpoints"]["grid"]})
    ks = grid if checkpoints == ["all"] else [int(k) for k in checkpoints]
    agents = e8_setup.load_models(e8_setup.REPO_ROOT / "configs" / "e8" / "agents.yaml")["pairs"]
    out = {}
    for pairs in batches.BATCHES[batch]["probe_jobs"]:
        models = sorted({agents[p][r] for p in pairs for r in ("exposed", "reference")})
        runs_args = [a for m in models for a in ("--run", f"{m}={runs[m]}")]
        for k in ks:
            out[f"probes_{'-'.join(pairs)}_k{k:06d}"] = [sys.executable, "-u", str(Path(__file__).with_name("run_e8.py")), "--phase", "probes",
                                                         "--pairs", *pairs, *runs_args, "--checkpoint", str(k), "--device", "cpu",
                                                         "--out-dir", str(out_dir), *batches.config_args(batch), *extra]
    return out


def set_aside(out_dir: Path) -> Path | None:
    """Output directories left "running" by an interrupted chain, moved aside (never deleted)."""
    found = [d for d in sorted((out_dir / e8_setup.EXPERIMENT_ID).glob("probes__*"))
             if (d / "status.json").is_file() and json.loads((d / "status.json").read_text(encoding="utf-8"))["status"] == "running"]
    if not found:
        return None
    dest = out_dir / e8_setup.EXPERIMENT_ID / "interrupted" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest.mkdir(parents=True, exist_ok=True)
    for d in found:
        shutil.move(str(d), str(dest / d.name))
    return dest


def chain(args: argparse.Namespace, todo: dict[str, list[str]]) -> int:
    args.logs.mkdir(parents=True, exist_ok=True)
    path = args.logs / "state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"jobs": {}}
    for name in todo:
        if state["jobs"].get(name, {}).get("status") != "done":
            state["jobs"][name] = {"status": "pending"}
    state["error"] = None
    moved = set_aside(args.out_dir)
    if moved:
        state["set_aside"] = str(moved)
    keep_awake()
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(args.logs, state, stop), daemon=True).start()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    running: dict[str, tuple[subprocess.Popen, object]] = {}
    failed = False
    try:
        queue = [n for n in todo if state["jobs"][n]["status"] != "done"]
        while queue or running:
            while queue and not failed and len(running) < args.workers:
                name = queue.pop(0)
                log = open(args.logs / f"{name}.log", "w", encoding="utf-8")
                running[name] = (subprocess.Popen(todo[name], cwd=e8_setup.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), log)
                state["jobs"][name] = {"status": "running", "start": now()}
                state["updated"] = now()
                write_status(args.logs, state)
            if failed and not running:
                break
            time.sleep(1)  # short waits: an interrupt is handled at once (Windows)
            for name in [n for n, (p, _) in running.items() if p.poll() is not None]:
                proc, log = running.pop(name)
                log.close()
                ok = proc.returncode == 0
                state["jobs"][name].update(status="done" if ok else "FAILED", end=now())
                state["updated"] = now()
                if not ok:
                    failed = True
                    state["error"] = f"job {name} exited {proc.returncode}; see {args.logs / (name + '.log')}; no new job started"
                write_status(args.logs, state)
    except BaseException as exc:  # KeyboardInterrupt included: never leave a job marked running
        for name, (proc, log) in running.items():
            kill_tree(proc)
            log.close()
            state["jobs"][name].update(status="FAILED", end=now())
        state.update(updated=now(), error=f"chain stopped by {type(exc).__name__} ({exc!r}); running jobs killed\n" + traceback.format_exc())
        write_status(args.logs, state)
        return 1
    finally:
        stop.set()
    state["updated"] = now()
    write_status(args.logs, state)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", choices=list(batches.BATCHES), required=True)
    parser.add_argument("--batch-root", type=Path, required=True, help="the extracted batch (its results/E8_therapy_relapse/ runs)")
    parser.add_argument("--out-dir", type=Path, required=True, help="where the probe jobs write (results/E8_therapy_relapse/probes__*)")
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--checkpoints", nargs="+", default=["all"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.out_dir, args.workers = args.out_dir / "smoke", 1
        todo = jobs(args.batch, args.batch_root, args.out_dir, args.checkpoints, ["--n-reps", "1", "--n-walks", "3"])
        todo = dict(list(todo.items())[:1])
        args.logs = args.logs / "smoke"
    else:
        todo = jobs(args.batch, args.batch_root, args.out_dir, args.checkpoints, [])
    return chain(args, todo)


if __name__ == "__main__":
    sys.exit(main())
