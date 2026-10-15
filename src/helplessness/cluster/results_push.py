"""Automatic push of a run's light results to its batch's orphan branch results/<batch> (docs/RESULTS_CONVENTION.md),
from the cluster, at the end of the run -- success or failure. Added 2026-10-15 for E4.

What is pushed, under results/<experiment_id>/<batch>/<run name>/: the light files of the run directory
(PUSHED_FILES, when present), console.log gzipped, run_status.json (status, error, exit code, host time) and
heavy_files.json (every weight / buffer / array left on the cluster's storage, with its absolute path, SHA-256 and
size: handoff.heavy_files). Nothing heavy is ever pushed. At the end of a batch, push_batch_status adds the
supervisor's status.json and a batch-level handoff.json (every run's status and heavy files).

Credentials: never written in the code nor in any file the push creates. The token is read from a file (a mounted
secret, the cluster's admission policy passes credentials as files) or an environment variable, and handed to git
through a temporary GIT_ASKPASS script that reads it from the environment of the git process only; it never appears
on a command line nor in a log. Network errors are retried (`retries` times, `wait` seconds apart); a push rejected
because another run of the batch pushed first is retried after a rebase (each run writes its own directory, so the
rebase never conflicts).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from helplessness.cluster.handoff import heavy_files

PUSHED_FILES = ("config.yaml", "meta.yaml", "results.json", "summary.json", "metrics.csv", "checkpoints.jsonl",
                "diagnostics.jsonl", "REPORT.md")
TOKEN_ENV = "RESULTS_PUSH_TOKEN"


class PushError(RuntimeError):
    """The results could not be pushed after every retry."""


def read_token(token_file: str | Path | None, token_env: str = TOKEN_ENV) -> str | None:
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"the token file {token_file} is empty")
        return token
    return os.environ.get(token_env) or None


def _askpass_env(token: str | None, workdir: Path) -> dict:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token is None:
        return env
    script = workdir / ("askpass.bat" if sys.platform == "win32" else "askpass.sh")
    if sys.platform == "win32":
        script.write_text("@echo %RESULTS_PUSH_ASKPASS_TOKEN%\n", encoding="utf-8")
    else:
        script.write_text('#!/bin/sh\nprintf "%s\\n" "$RESULTS_PUSH_ASKPASS_TOKEN"\n', encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    env.update(GIT_ASKPASS=str(script), RESULTS_PUSH_ASKPASS_TOKEN=token)
    return env


def _git(args: list[str], cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True)


def stage_run(run_dir: Path, dest: Path, status: dict) -> None:
    """The run's light files into dest (created), with run_status.json and heavy_files.json."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in PUSHED_FILES:
        if (run_dir / name).is_file():
            shutil.copyfile(run_dir / name, dest / name)
    if (run_dir / "console.log").is_file():
        with open(run_dir / "console.log", "rb") as src, gzip.GzipFile(dest / "console.log.gz", "wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
    (dest / "run_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    (dest / "heavy_files.json").write_text(json.dumps(heavy_files(run_dir), indent=1), encoding="utf-8")


def push_tree(stage: Path, branch: str, repo_url: str, message: str, token: str | None, retries: int = 6,
              wait: float = 30.0, author: tuple[str, str] = ("cluster", "cluster@localhost"), log=print) -> str:
    """Commits the content of `stage` (a tree of relative paths) on top of `branch` of repo_url (created as an orphan
    branch when it does not exist yet) and pushes it. Returns the pushed commit. Raises PushError after `retries`
    failed attempts."""
    last = ""
    for attempt in range(1, retries + 1):
        with tempfile.TemporaryDirectory(prefix="results_push_") as tmp:
            tmp = Path(tmp)
            env = _askpass_env(token, tmp)
            repo = tmp / "repo"
            repo.mkdir()
            ident = ["-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}"]
            steps = [["init", "-q"], ["remote", "add", "origin", repo_url]]
            for args in steps:
                done = _git(args, repo, env)
                if done.returncode != 0:
                    raise PushError(f"git {args[0]} failed: {done.stderr.strip()}")
            fetch = _git(["fetch", "-q", "--depth", "1", "origin", branch], repo, env)
            exists = fetch.returncode == 0
            if not exists and "couldn't find remote ref" not in fetch.stderr and "not found" not in fetch.stderr.lower():
                last = f"fetch: {fetch.stderr.strip()}"
                log(f"[results push] attempt {attempt}/{retries}: {last}; retrying in {wait:g} s")
                time.sleep(wait)
                continue
            checkout = (["checkout", "-q", "-b", branch, "FETCH_HEAD"] if exists else ["checkout", "-q", "--orphan", branch])
            done = _git(checkout, repo, env)
            if done.returncode != 0:
                raise PushError(f"git checkout failed: {done.stderr.strip()}")
            shutil.copytree(stage, repo, dirs_exist_ok=True)
            _git(["add", "-A"], repo, env)
            commit = _git([*ident, "commit", "-q", "-m", message], repo, env)
            if commit.returncode != 0 and "nothing to commit" not in commit.stdout + commit.stderr:
                raise PushError(f"git commit failed: {commit.stderr.strip()}")
            push = _git(["push", "-q", "origin", f"HEAD:refs/heads/{branch}"], repo, env)
            if push.returncode == 0:
                sha = _git(["rev-parse", "HEAD"], repo, env).stdout.strip()
                log(f"[results push] {branch} <- {sha} ({message})")
                return sha
            last = f"push: {push.stderr.strip()}"
            log(f"[results push] attempt {attempt}/{retries}: {last}; retrying in {wait:g} s")
        time.sleep(wait)
    raise PushError(f"could not push to {branch} after {retries} attempts: {last}")


def push_run(run_dir: str | Path, experiment_id: str, batch: str, repo_url: str, status: dict, token: str | None,
             retries: int = 6, wait: float = 30.0, log=print) -> str:
    """Stages one run (stage_run) under results/<experiment_id>/<batch>/<run name>/ and pushes it to results/<batch>."""
    run_dir = Path(run_dir)
    with tempfile.TemporaryDirectory(prefix="results_stage_") as tmp:
        stage = Path(tmp)
        stage_run(run_dir, stage / "results" / experiment_id / batch / run_dir.name,
                  {**status, "pushed_at": datetime.now(timezone.utc).isoformat()})
        return push_tree(stage, f"results/{batch}", repo_url, f"{batch} : {run_dir.name} ({status.get('status', '?')})", token,
                         retries=retries, wait=wait, log=log)


def push_batch_status(supervisor_dir: str | Path, run_dirs: list[Path], experiment_id: str, batch: str, repo_url: str,
                      token: str | None, retries: int = 6, wait: float = 30.0, log=print) -> str:
    """The batch-level files: the supervisor's status.json and a handoff.json with every run's status and heavy files."""
    supervisor_dir = Path(supervisor_dir)
    with tempfile.TemporaryDirectory(prefix="results_stage_") as tmp:
        dest = Path(tmp) / "results" / experiment_id / batch
        dest.mkdir(parents=True)
        if (supervisor_dir / "status.json").is_file():
            shutil.copyfile(supervisor_dir / "status.json", dest / "status.json")
        runs = []
        for run_dir in run_dirs:
            meta = run_dir / "meta.yaml"
            runs.append({"run": run_dir.name, "meta": meta.read_text(encoding="utf-8") if meta.is_file() else None})
        handoff = {"batch": batch, "runs": runs, "heavy_files": [e for d in run_dirs for e in heavy_files(d)]}
        (dest / "handoff.json").write_text(json.dumps(handoff, indent=1), encoding="utf-8")
        return push_tree(Path(tmp), f"results/{batch}", repo_url, f"{batch} : statut du batch", token, retries=retries, wait=wait, log=log)
