"""Generic Kubeflow tooling for the GPU cluster (UCL ARC): the job supervisor
and the runtime volume patch. Moved from
experiments/E3_worldmodel_4zones/submit_kubeflow.py on 2026-10-14, bodies
unchanged; that script keeps E3's batch tables, its serialised worker
(_run_e3) and its command-line defaults, and calls these. No cluster path is
written here: the repo directory, the mount point and the token's mount
point are always arguments.
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time


def supervise(jobs: list[tuple[str, list[str]]], repo: Path, output: Path, max_parallel: int | None = None) -> int:
    """Run the jobs, at most max_parallel at a time (default: all at once),
    in order, starting the next queued job when one finishes; wait for all
    of them and persist each exit code, including scientific failures. A
    job never started (supervisor interrupted) is recorded with exit_code
    null and not_started true."""
    if max_parallel is not None and max_parallel < 1:
        raise ValueError(f"max_parallel must be >= 1, got {max_parallel}")
    limit = len(jobs) if max_parallel is None else max_parallel
    output.mkdir(parents=True, exist_ok=False)
    (output / "supervisor.json").write_text(json.dumps({"max_parallel": limit, "jobs": [name for name, _ in jobs]}, indent=2) + "\n",
                                            encoding="utf-8")
    processes: dict[str, subprocess.Popen] = {}
    queue = list(jobs)
    status: dict[str, dict] = {name: {"command": command, "log": str(output / f"{name}.log"), "exit_code": None}
                               for name, command in jobs}

    def save() -> None:
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    def interrupted(signum: int, frame: object) -> None:
        raise SystemExit(128 + signum)

    def running() -> int:
        return sum(status[name]["exit_code"] is None for name in processes)

    previous = signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        with ExitStack() as stack:
            try:
                while queue or running():
                    while queue and running() < limit:
                        name, command = queue.pop(0)
                        log = stack.enter_context((output / f"{name}.log").open("w", encoding="utf-8"))
                        processes[name] = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                        status[name].update(pid=processes[name].pid, started_at=datetime.now(timezone.utc).isoformat())
                        save()
                        print(f"started {name}: pid={processes[name].pid}, log={status[name]['log']} "
                              f"({running()}/{limit} running, {len(queue)} queued)", flush=True)
                    for name, process in processes.items():
                        code = process.poll()
                        if code is not None and status[name]["exit_code"] is None:
                            status[name].update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                            save()
                            print(f"finished {name}: exit_code={code}", flush=True)
                    if queue or running():
                        time.sleep(1)
            finally:
                for process in processes.values():
                    if process.poll() is None:
                        process.terminate()
                for name, process in processes.items():
                    try:
                        code = process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        code = process.wait()
                    if status[name]["exit_code"] is None:
                        status[name].update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
                for name, _ in queue:
                    status[name]["not_started"] = True
                save()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return int(any(item["exit_code"] != 0 for item in status.values()))


def runtime_patch(volume: str, mount: str, secret: str | None, token_mount: str) -> dict:
    """Build only the volume patch permitted by the cluster admission policy.
    token_mount: where the secret holding WANDB_API_KEY is mounted in the
    container (E3: "/etc/e3-wandb", passed by its submit_kubeflow.py)."""
    volumes = [{"name": "scratch", "persistentVolumeClaim": {"claimName": volume}}]
    mounts = [{"name": "scratch", "mountPath": mount}]
    if secret:
        volumes.append({"name": "wandb-token", "secret": {"secretName": secret}})
        mounts.append({"name": "wandb-token", "mountPath": token_mount, "readOnly": True})
    return {"template": {"spec": {"replicatedJobs": [
        {"name": "node", "template": {"spec": {"template": {"spec": {
            "volumes": volumes, "containers": [{"name": "node", "volumeMounts": mounts}],
        }}}}}
    ]}}}
