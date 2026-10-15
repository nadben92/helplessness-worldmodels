"""helplessness.cluster.results_push against a local bare repository (no network): two runs pushed to the same orphan
branch, nothing heavy pushed, the batch status, the retries on an unreachable remote."""

from __future__ import annotations

import json
import subprocess

import pytest

from helplessness.cluster import results_push


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def make_run(root, name: str, status: str = "done"):
    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    (d / "meta.yaml").write_text(f"status: {status}\n", encoding="utf-8")
    (d / "checkpoints.jsonl").write_text('{"env_step": 1}\n', encoding="utf-8")
    (d / "console.log").write_text("hello\n", encoding="utf-8")
    (d / "checkpoints" / "agent_env000001.pt").write_bytes(b"weights")
    (d / "replay_final.npz").write_bytes(b"replay")
    return d


@pytest.fixture()
def remote(tmp_path):
    bare = tmp_path / "remote.git"
    git("init", "-q", "--bare", str(bare))
    return bare


def test_two_runs_on_one_orphan_branch_without_heavy_files(tmp_path, remote):
    runs = tmp_path / "runs"
    a, b = make_run(runs, "run_a"), make_run(runs, "run_b", "failed")
    url = remote.as_uri()
    results_push.push_run(a, "E4_generalization_gradient", "e4_test", url, {"status": "done"}, None, retries=2, wait=0)
    results_push.push_run(b, "E4_generalization_gradient", "e4_test", url, {"status": "failed", "error": "X"}, None, retries=2, wait=0)
    results_push.push_batch_status(tmp_path, [a, b], "E4_generalization_gradient", "e4_test", url, None, retries=2, wait=0)
    files = git("ls-tree", "-r", "--name-only", "results/e4_test", cwd=remote).split()
    base = "results/E4_generalization_gradient/e4_test"
    assert f"{base}/run_a/config.yaml" in files and f"{base}/run_b/run_status.json" in files and f"{base}/handoff.json" in files
    assert f"{base}/run_a/console.log.gz" in files and f"{base}/run_a/heavy_files.json" in files
    assert not any(f.endswith((".pt", ".npz")) for f in files)
    assert git("rev-list", "--max-parents=0", "results/e4_test", cwd=remote).count("\n") == 1  # orphan: one root commit
    heavy = json.loads(git("show", f"results/e4_test:{base}/run_a/heavy_files.json", cwd=remote))
    assert {h["path"].replace("\\", "/").split("/")[-1] for h in heavy} == {"agent_env000001.pt", "replay_final.npz"}
    status = json.loads(git("show", f"results/e4_test:{base}/run_b/run_status.json", cwd=remote))
    assert status["status"] == "failed" and status["error"] == "X"


def test_unreachable_remote_is_retried_then_raises(tmp_path):
    run = make_run(tmp_path / "runs", "run_a")
    logs = []
    with pytest.raises(results_push.PushError):
        results_push.push_run(run, "E", "b", (tmp_path / "missing.git").as_uri(), {"status": "done"}, "SECRET-TOKEN", retries=2,
                              wait=0, log=logs.append)
    assert len(logs) == 2 and not any("SECRET-TOKEN" in line for line in logs)


def test_token_from_file_or_env(tmp_path, monkeypatch):
    f = tmp_path / "tok"
    f.write_text("abc\n", encoding="utf-8")
    assert results_push.read_token(f) == "abc"
    monkeypatch.setenv(results_push.TOKEN_ENV, "xyz")
    assert results_push.read_token(None) == "xyz"
    (tmp_path / "empty").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        results_push.read_token(tmp_path / "empty")
