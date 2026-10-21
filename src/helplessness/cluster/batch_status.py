"""The status of a batch on the GPU cluster, read from its supervisor directory (helplessness.cluster.kubeflow.supervise:
status.json and one log per job), with a STALE flag: a job still running (no exit code) whose log has not been modified
for more than `stale_after` seconds (default one hour). E4's wave 2 was frozen for 24 hours by a failure of the cluster's
storage with nothing visible in status.json (no exit code, no error): only the logs' modification times showed it.
Standard library only (the notebook's Python runs the submit scripts without the package's dependencies).

Usage: submit_kubeflow.py --status <supervisor directory> [--stale-after 3600] (E4 and E5).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STALE_AFTER = 3600.0
STATES = ("done", "failed", "running", "STALE", "not started")


def batch_status(output: Path, stale_after: float = STALE_AFTER, now: float | None = None) -> list[dict]:
    """One row per job of <output>/status.json: its state (done: exit code 0; failed: another exit code; not started:
    no pid; STALE: running with a log older than stale_after seconds; running otherwise), exit code, pid, start, the
    log's age in seconds and its last line. The log is <output>/<job>.log (status.json's own path is the cluster's)."""
    output = Path(output)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    now = time.time() if now is None else now
    rows = []
    for name, job in status.items():
        log = output / f"{name}.log"
        age = now - log.stat().st_mtime if log.is_file() else None
        last = ""
        if log.is_file():
            lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            last = lines[-1][:200] if lines else ""
        code = job.get("exit_code")
        if code is not None:
            state = "done" if code == 0 else "failed"
        elif job.get("pid") is None or job.get("not_started"):
            state = "not started"
        elif age is None or age > stale_after:
            state = "STALE"
        else:
            state = "running"
        rows.append({"job": name, "state": state, "exit_code": code, "pid": job.get("pid"), "started_at": job.get("started_at"),
                     "log_age_seconds": None if age is None else round(age), "last_line": last})
    return rows


def format_status(rows: list[dict], stale_after: float = STALE_AFTER) -> list[str]:
    """Readable lines, then one warning line per STALE job."""
    out = [f"{r['job']:<28} {r['state']:<11} exit={r['exit_code']!s:<5} log age="
           f"{'-' if r['log_age_seconds'] is None else _age(r['log_age_seconds'])} | {r['last_line']}" for r in rows]
    for r in rows:
        if r["state"] == "STALE":
            out.append(f"STALE {r['job']}: running, but its log has not changed for "
                       f"{'ever (no log)' if r['log_age_seconds'] is None else _age(r['log_age_seconds'])} "
                       f"(> {_age(stale_after)}): check the pod, the GPU and /scratch")
    return out


def _age(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"
