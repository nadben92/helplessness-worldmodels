"""E6 / E7 on all agents, every phase in a row (docs/e6/DECISIONS.md (63)): run_all_cpu.py once per phase, in order, in
one unattended process. Nothing of the measures is here: each phase is run_all_cpu.py's own command line.

  --logs/STATUS.txt       the phase running, each phase's start and end, the error that stopped the chain, and a
                          "heartbeat <time>" line rewritten every HEARTBEAT seconds while the chain lives (a heartbeat
                          older than a few minutes with no ERROR line: the chain was killed from outside)
  --logs/state.json       the same, read back on a relaunch: a phase marked done is skipped
  --logs/pool_<phase>.txt run_all_cpu.py's output; --logs/<phase>/ its per-job logs

A phase that fails stops the chain (the next phases never read incomplete outputs); so does any exception of the chain
itself, KeyboardInterrupt included: the running phase's process tree is killed and the phase marked FAILED with the
cause. On a relaunch, a phase that was not finished is rerun from scratch: its job directories under scratch/all/
(status "running", never to be read) are first moved aside to scratch/all_interrupted/<UTC stamp>/, nothing is deleted.
The machine is kept awake while the chain runs (Windows: SetThreadExecutionState; the screen may still turn off).

--smoke <agent> <image>: the end-to-end check before a launch: every phase on one agent (e6 and c12 on one image), with
the scripts' own quick overrides (--n-reps 1, --contexts-per-state 1, no bootstrap), sequentially, into
scratch/smoke/<job>/ (delete it afterwards). The measures' numbers are meaningless there.

Usage, from the repo root (src of this checkout on PYTHONPATH):
  python experiments/E6_phenotypes_latent/chain_all_cpu.py --runs-root <root> --logs logs/all_cpu_650k [--workers 12]
  python experiments/E6_phenotypes_latent/chain_all_cpu.py --runs-root <root> --logs <dir> --smoke f1_F segment_k1_ud
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

import run_all_cpu as rac

PHASES = ("e6", "c12", "directions", "constrained", "corrected", "oracle", "anti", "free")
PREFIX = {"e6": "e6_", "c12": "c12_", "directions": "dir_", "constrained": "con_", "corrected": "cor_", "oracle": "oracle_",
          "anti": "anti_", "free": "free_"}
QUICK = {"e6": ["--n-reps", "1", "--no-bootstrap"], "c12": ["--n-reps", "1"], "directions": ["--contexts-per-state", "1"],
         "constrained": ["--contexts-per-state", "1"], "corrected": ["--n-reps", "1"], "oracle": ["--n-reps", "1"],
         "anti": ["--n-reps", "1"], "free": ["--n-reps", "1"]}
HEARTBEAT = 60  # seconds
_status_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def keep_awake() -> None:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def write_status(logs: Path, state: dict) -> None:
    with _status_lock:
        lines = [f"heartbeat {now()}", f"updated {state.get('updated', '-')}", f"current: {state.get('current') or '-'}", ""]
        for p in PHASES:
            s = state["phases"].get(p, {})
            lines.append(f"{p:12s} {s.get('status', 'pending'):8s} start {s.get('start', '-'):26s} end {s.get('end', '-')}"
                         + (f"  ({s['jobs']})" if "jobs" in s else ""))
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


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if sys.platform == "win32":
            subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()


def set_aside(phase: str) -> Path | None:
    """An unfinished phase's job directories, moved to scratch/all_interrupted/<stamp>/ (never deleted)."""
    found = sorted((rac.REPO_ROOT / rac.OUT).glob(PREFIX[phase] + "*"))
    if not found:
        return None
    dest = rac.REPO_ROOT / "scratch" / "all_interrupted" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest.mkdir(parents=True, exist_ok=True)
    for d in found:
        shutil.move(str(d), str(dest / d.name))
    return dest


def alive(pid: int) -> bool:
    if sys.platform != "win32":
        return Path(f"/proc/{pid}").exists()
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(handle, ctypes.byref(code))
    k32.CloseHandle(handle)
    return code.value == 259  # STILL_ACTIVE


def chain(args: argparse.Namespace) -> int:
    args.logs.mkdir(parents=True, exist_ok=True)
    lock = args.logs / "chain.pid"
    if lock.is_file() and alive(int(lock.read_text())):
        print(f"a chain is already running (pid {lock.read_text()}); not started", flush=True)
        return 2
    lock.write_text(str(os.getpid()))
    path = args.logs / "state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"phases": {}}
    state["error"] = None
    keep_awake()
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(args.logs, state, stop), daemon=True).start()
    p, proc = None, None
    try:
        for p in PHASES:
            if state["phases"].get(p, {}).get("status") == "done":
                continue
            moved = set_aside(p)
            state.update(current=p, updated=now())
            state["phases"][p] = {"status": "running", "start": now(), **({"set_aside": str(moved)} if moved else {})}
            write_status(args.logs, state)
            cmd = [sys.executable, "-u", str(Path(__file__).with_name("run_all_cpu.py")), "--phase", p, "--runs-root",
                   args.runs_root, "--workers", str(args.workers), "--logs", str(args.logs / p)]
            with open(args.logs / f"pool_{p}.txt", "w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, cwd=rac.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                while (code := proc.poll()) is None:  # short waits: an interrupt is handled at once (Windows)
                    time.sleep(1)
            proc = None
            tail = (args.logs / f"pool_{p}.txt").read_text(encoding="utf-8").strip().splitlines()[-1:]
            state["phases"][p].update(status="done" if code == 0 else "FAILED", end=now(),
                                      jobs=(tail[0][:160] + ("..." if len(tail[0]) > 160 else "")) if tail else "")
            state["updated"] = now()
            if code != 0:
                state["current"] = None
                state["error"] = f"phase {p} exited {code}; see {args.logs / f'pool_{p}.txt'} and {args.logs / p}/; chain stopped"
                write_status(args.logs, state)
                return 1
            write_status(args.logs, state)
    except BaseException as exc:  # KeyboardInterrupt included: never leave a phase marked running
        if proc is not None:
            kill_tree(proc)
        if p is not None and state["phases"].get(p, {}).get("status") == "running":
            state["phases"][p].update(status="FAILED", end=now())
        state.update(current=None, updated=now(),
                     error=f"chain stopped by {type(exc).__name__} ({exc!r}) during phase {p}; its processes were killed\n"
                           + traceback.format_exc())
        write_status(args.logs, state)
        return 1
    finally:
        stop.set()
    state.update(current=None, updated=now())
    write_status(args.logs, state)
    return 0


def smoke(args: argparse.Namespace) -> int:
    agent, image = args.smoke
    rac.OUT = "scratch/smoke"
    # jobs() resolves every agent's earlier phase; only this agent's is run here, the others' are left unresolved
    newest, best_setting = rac.newest, rac.best_setting
    rac.newest = lambda pattern: newest(pattern) if f"__{agent}__" in pattern else "-"
    rac.best_setting = lambda d: best_setting(d) if d != "-" else (0.0, 0.0)
    args.logs.mkdir(parents=True, exist_ok=True)
    for p in PHASES:
        todo = [(n, c) for n, c in rac.jobs(p, sys.executable, args.runs_root, [image] if p in ("e6", "c12") else None)
                if n.startswith(f"{PREFIX[p]}{agent}")]
        if p == "anti":
            todo = todo[:1] + todo[-1:]  # one reference job, the experimental job
        for name, cmd in todo:
            cut = cmd.index("--out-dir")
            cmd = [*cmd[:cut], *QUICK[p], *cmd[cut:]]
            t0 = time.time()
            with open(args.logs / f"{name}.log", "w", encoding="utf-8") as log:
                code = subprocess.call(cmd, cwd=rac.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"{p:12s} {name}: exit {code} in {time.time() - t0:.0f} s", flush=True)
            if code != 0:
                return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", nargs=2, metavar=("AGENT", "IMAGE"), default=None)
    args = parser.parse_args(argv)
    args.logs = args.logs if args.logs.is_absolute() else rac.REPO_ROOT / args.logs
    return smoke(args) if args.smoke else chain(args)


if __name__ == "__main__":
    sys.exit(main())
