"""Cluster submission and foreground supervision, without a cluster or CUDA."""

import json
import os
import sys

import pytest

from _experiment_loader import load_experiment_module

cluster = load_experiment_module("E3_worldmodel_4zones", "submit_kubeflow.py", "test_cluster")


def test_preflight_is_separate_from_full_training(tmp_path):
    jobs = cluster.commands("preflight", "python", tmp_path)
    assert len(jobs) == 1
    assert jobs[0][1][2].endswith("preflight.py")
    assert "--extra-config" in jobs[0][1]
    assert "--max-steps" not in jobs[0][1]
    runs = cluster.commands("train", "python", tmp_path)
    assert len(runs) == 5
    assert len({name for name, _ in runs}) == 5
    for _, command in runs:
        assert command[command.index("--device") + 1] == "cuda"
        assert "--max-steps" not in command


def test_resume_requires_directory(tmp_path):
    with pytest.raises(ValueError, match="directory"):
        cluster.commands("resume", "python", tmp_path)
    command = cluster.commands("resume", "python", tmp_path, "/scratch/run")[0][1]
    assert command[command.index("--resume") + 1] == "/scratch/run"


def test_secret_is_a_readonly_file_not_a_container_env():
    patch = cluster.runtime_patch("scratch-volume", "/scratch", "e3-wandb")
    spec = patch["template"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert "env" not in container
    assert container["volumeMounts"][1]["readOnly"] is True
    assert spec["volumes"][1]["secret"]["secretName"] == "e3-wandb"
    assert len(cluster.runtime_patch("scratch-volume", "/scratch", None)
               ["template"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["volumes"]) == 1


def test_supervisor_waits_for_all_runs_and_records_failure(tmp_path):
    output = tmp_path / "logs"
    jobs = [("failure", [sys.executable, "-c", "print('guard fired'); raise SystemExit(1)"]),
            ("success", [sys.executable, "-c", "import time; time.sleep(0.2); print('finished')"])]
    assert cluster.supervise(jobs, tmp_path, output) == 1
    status = json.loads((output / "status.json").read_text())
    assert status["failure"]["exit_code"] == 1
    assert status["success"]["exit_code"] == 0
    assert "finished" in (output / "success.log").read_text()
    assert all("finished_at" in item for item in status.values())
    with pytest.raises(FileExistsError):
        cluster.supervise(jobs, tmp_path, output)


def test_successful_preflight_exit_code(tmp_path):
    assert cluster.supervise([("preflight", [sys.executable, "-c", "print('PREFLIGHT PASSED')"])],
                             tmp_path, tmp_path / "logs") == 0


def test_worker_bootstrap_uses_host_pip_and_disables_tracking(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cluster.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    for name in ("HOME", "USER", "LOGNAME", "PYTHONUNBUFFERED", "CUBLAS_WORKSPACE_CONFIG",
                 "MPLBACKEND", "XDG_CACHE_HOME", "WANDB_MODE"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    cluster._run_e3(str(tmp_path), "preflight", str(tmp_path / "logs"), "/does/not/exist", "")
    assert "--without-pip" in calls[0]
    assert calls[1][:5] == [sys.executable, "-m", "pip", "--python", str(tmp_path / ".venv-cluster/bin/python")]
    assert "WANDB_API_KEY" not in os.environ
    assert os.environ["WANDB_MODE"] == "disabled"
    assert "--worker" in calls[-1]
