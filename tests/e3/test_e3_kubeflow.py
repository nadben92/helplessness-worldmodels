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
                 "MPLBACKEND", "XDG_CACHE_HOME", "WANDB_MODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    cluster._run_e3(str(tmp_path), "preflight", str(tmp_path / "logs"), "/does/not/exist", "")
    assert "--without-pip" in calls[0]
    assert calls[1][:5] == [sys.executable, "-m", "pip", "--python", str(tmp_path / ".venv-cluster/bin/python")]
    assert "WANDB_API_KEY" not in os.environ
    assert os.environ["WANDB_MODE"] == "disabled"
    assert "--worker" in calls[-1]


def test_a_subset_of_runs_is_launched_in_batch_order(tmp_path):
    runs = cluster.commands("train", "python", tmp_path, runs=["bridge_seed0_experimental", "seed5_experimental"])
    assert [name for name, _ in runs] == ["seed5_experimental", "bridge_seed0_experimental"]
    seed5 = runs[0][1]
    assert seed5[seed5.index("--config") + 1] == cluster.SPEC_CONFIG
    assert (seed5[seed5.index("--seeds") + 1], seed5[seed5.index("--conditions") + 1]) == ("5", "experimental")
    assert runs[1][1][runs[1][1].index("--config") + 1] == cluster.BRIDGE_CONFIG
    with pytest.raises(ValueError, match="unknown run"):
        cluster.commands("train", "python", tmp_path, runs=["seed7_experimental"])


def test_preflight_tests_the_planned_number_of_processes(tmp_path):
    def parallel(command):
        return int(command[command.index("--parallel") + 1])

    full = cluster.commands("preflight", "python", tmp_path)[0][1]
    assert parallel(full) == 5 and "--extra-config" in full
    capped = cluster.commands("preflight", "python", tmp_path, max_parallel=2)[0][1]
    assert parallel(capped) == 2 and "--extra-config" not in capped  # worst mix: two spec-size runs
    mixed = cluster.commands("preflight", "python", tmp_path, runs=["seed5_experimental", "bridge_seed0_experimental"])[0][1]
    assert parallel(mixed) == 2 and mixed[mixed.index("--extra-config") + 1] == cluster.BRIDGE_CONFIG
    bridge_only = cluster.commands("preflight", "python", tmp_path, runs=["bridge_seed0_experimental"])[0][1]
    assert parallel(bridge_only) == 1 and bridge_only[bridge_only.index("--config") + 1] == cluster.BRIDGE_CONFIG
    with pytest.raises(ValueError, match="max_parallel"):
        cluster.commands("preflight", "python", tmp_path, max_parallel=0)


def test_supervisor_runs_at_most_max_parallel_jobs_at_once(tmp_path):
    output = tmp_path / "logs"
    job = [sys.executable, "-c", "import time; time.sleep(0.3)"]
    jobs = [("a", job), ("b", [sys.executable, "-c", "raise SystemExit(2)"]), ("c", job)]
    assert cluster.supervise(jobs, tmp_path, output, max_parallel=1) == 1
    status = json.loads((output / "status.json").read_text())
    assert [status[n]["exit_code"] for n in "abc"] == [0, 2, 0]
    for first, second in (("a", "b"), ("b", "c")):  # one at a time: each starts after the previous one finished
        assert status[second]["started_at"] >= status[first]["finished_at"]
    assert json.loads((output / "supervisor.json").read_text()) == {"max_parallel": 1, "jobs": ["a", "b", "c"]}


def test_worker_passes_the_selection_to_the_supervisor(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cluster.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setenv("RANK", "0")
    for name in ("HOME", "USER", "LOGNAME", "PYTHONUNBUFFERED", "CUBLAS_WORKSPACE_CONFIG",
                 "MPLBACKEND", "XDG_CACHE_HOME", "WANDB_MODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "WANDB_API_KEY"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    cluster._run_e3(str(tmp_path), "train", str(tmp_path / "logs"), "", "", runs="seed0_reference,seed5_reference", max_parallel=2)
    worker = calls[-1]
    assert worker[worker.index("--runs") + 1:worker.index("--runs") + 3] == ["seed0_reference", "seed5_reference"]
    assert worker[worker.index("--max-parallel") + 1] == "2"


def test_gpu_e30b_batch_has_eight_runs_in_two_shapes(tmp_path):
    runs = cluster.commands("train", "python", tmp_path, batch="gpu_e30b")
    assert len(runs) == 8 and len({name for name, _ in runs}) == 8
    configs = [command[command.index("--config") + 1] for _, command in runs]
    assert configs.count(cluster.WIN64_CONFIG) == 4 and configs.count(cluster.WIN16_CONFIG) == 4
    pairs = {(command[command.index("--seeds") + 1], command[command.index("--conditions") + 1]) for _, command in runs}
    assert pairs == {(seed, condition) for seed in ("0", "5") for condition in ("experimental", "reference")}
    with pytest.raises(ValueError, match="unknown run"):
        cluster.commands("train", "python", tmp_path, runs=["seed0_experimental"], batch="gpu_e30b")
    with pytest.raises(ValueError, match="unknown batch"):
        cluster.select_runs(None, batch="gpu_e99")


def test_gpu_e30b_preflight_mixes_both_shapes_heaviest_first(tmp_path):
    command = cluster.commands("preflight", "python", tmp_path, batch="gpu_e30b", max_parallel=8)[0][1]
    assert command[command.index("--config") + 1] == cluster.WIN64_CONFIG
    assert command[command.index("--extra-config") + 1] == cluster.WIN16_CONFIG
    assert command[command.index("--parallel") + 1] == "8"
    capped = cluster.commands("preflight", "python", tmp_path, batch="gpu_e30b", max_parallel=4)[0][1]
    assert "--extra-config" not in capped and capped[capped.index("--parallel") + 1] == "4"  # 4 x win64: the worst mix
    only16 = cluster.commands("preflight", "python", tmp_path, batch="gpu_e30b",
                              runs=["win16_seed0_experimental", "win16_seed0_reference"])[0][1]
    assert only16[only16.index("--config") + 1] == cluster.WIN16_CONFIG and "--extra-config" not in only16


def test_worker_passes_the_batch_and_one_blas_thread(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cluster.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setenv("RANK", "0")
    for name in ("HOME", "USER", "LOGNAME", "PYTHONUNBUFFERED", "CUBLAS_WORKSPACE_CONFIG", "MPLBACKEND", "XDG_CACHE_HOME",
                 "WANDB_MODE", "WANDB_API_KEY", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    cluster._run_e3(str(tmp_path), "train", str(tmp_path / "logs"), "", "", max_parallel=4, batch="gpu_e30b")
    worker = calls[-1]
    assert worker[worker.index("--batch") + 1] == "gpu_e30b" and "--runs" not in worker
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_cpu_request_reaches_the_preflight_and_the_worker(tmp_path, monkeypatch):
    command = cluster.commands("preflight", "python", tmp_path, batch="gpu_e30b", max_parallel=4, cpus=5)[0][1]
    assert command[command.index("--cpus") + 1] == "5"
    i = command.index("--batch-runs")
    assert command[i + 1:i + 3] == [f"{cluster.WIN64_CONFIG}=4", f"{cluster.WIN16_CONFIG}=4"]
    assert cluster.commands("preflight", "python", tmp_path)[0][1][cluster.commands("preflight", "python", tmp_path)[0][1].index("--cpus") + 1] == "3"
    calls = []
    monkeypatch.setattr(cluster.subprocess, "run", lambda c, **kwargs: calls.append(c))
    monkeypatch.setenv("RANK", "0")
    for name in ("HOME", "USER", "LOGNAME", "PYTHONUNBUFFERED", "CUBLAS_WORKSPACE_CONFIG", "MPLBACKEND", "XDG_CACHE_HOME",
                 "WANDB_MODE", "WANDB_API_KEY", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    cluster._run_e3(str(tmp_path), "preflight", str(tmp_path / "logs"), "", "", batch="gpu_e30b", cpus=5)
    assert calls[-1][calls[-1].index("--cpu") + 1] == "5"


def test_a_batch_s_prescribed_cpus_cannot_be_missed(tmp_path, capsys):
    """docs/e3/DECISIONS.md, 2026-09-30 (5): gpu_e30b was prescribed 5 CPUs and ran on 3 (the default)."""
    assert cluster.resolve_cpus("gpu_e30b", None) == 5 and cluster.resolve_cpus("gpu_e30", None) == 3
    assert cluster.resolve_cpus("gpu_e30b", 5) == 5
    with pytest.raises(ValueError, match="prescribed 5 CPUs"):
        cluster.resolve_cpus("gpu_e30b", 3)
    assert cluster.resolve_cpus("gpu_e30b", 3, allow_override=True) == 3
    default = cluster.commands("preflight", "python", tmp_path, batch="gpu_e30b")[0][1]
    assert default[default.index("--cpus") + 1] == "5"  # without --cpu, the batch's own number
    with pytest.raises(SystemExit):  # refused at submission, before anything is sent
        cluster.main(["--phase", "preflight", "--batch", "gpu_e30b", "--cpu", "3"])
    assert "prescribed 5 CPUs" in capsys.readouterr().err


def test_measurement_batch_reads_the_runs_in_place_and_never_trains(tmp_path):
    """docs/e3/DECISIONS.md, 2026-10-01 (10): gpu_e30b_measures = recompute_measures.py on the four win64 runs, read where
    they were trained, outputs in the job's own directory, the weights' checksums checked; no training phase."""
    jobs = cluster.commands("measure", "python", tmp_path, batch="gpu_e30b_measures")
    assert [name for name, _ in jobs] == ["win64_seed0_experimental", "win64_seed0_reference", "win64_seed5_experimental",
                                          "win64_seed5_reference"]
    for name, command in jobs:
        run_dir, sha = cluster.MEASURE_BATCHES["gpu_e30b_measures"]["runs"][name]
        assert command[2].endswith("recompute_measures.py") and "run_mini.py" not in " ".join(command)
        assert command[3] == f"{cluster.GPU_E30B_RESULTS}/{run_dir}"
        assert command[command.index("--out-dir") + 1] == str(tmp_path / "runs")
        assert command[command.index("--weights-sha256") + 1] == sha and command[command.index("--device") + 1] == "cuda"
        assert "--kernel-draws" not in command  # the runs' own K_s (1,358): the measurement configuration is the run's
    for phase in ("train", "resume"):
        with pytest.raises(ValueError, match="measurement batch"):
            cluster.commands(phase, "python", tmp_path, resume="x", batch="gpu_e30b_measures")
    with pytest.raises(ValueError, match="phase measure"):
        cluster.commands("measure", "python", tmp_path, batch="gpu_e30b")
    pre = cluster.commands("preflight", "python", tmp_path, batch="gpu_e30b_measures", max_parallel=4)[0][1]
    assert pre[2].endswith("preflight_measures.py") and pre[pre.index("--cpus") + 1] == "3" and pre[pre.index("--parallel") + 1] == "4"
    assert cluster.resolve_cpus("gpu_e30b_measures", None) == 3
    with pytest.raises(ValueError):
        cluster.resolve_cpus("gpu_e30b_measures", 5)
    with pytest.raises(SystemExit):
        cluster.main(["--phase", "train", "--batch", "gpu_e30b_measures"])


def test_closed_loop_pilot_batch():
    """docs/e3/DECISIONS.md, 2026-10-02 (19): gpu_pilot_cl runs run_agent.py, experimental runs first, 3 CPUs; its
    preflight is preflight_agent.py; resume continues one run; no measure phase."""
    from pathlib import Path

    jobs = cluster.commands("train", "python", Path("/o"), batch="gpu_pilot_cl")
    assert [n for n, _ in jobs] == ["seed0_experimental", "seed5_experimental", "seed0_reference", "seed5_reference"]
    for name, cmd in jobs:
        assert cmd[2].endswith("run_agent.py") and cmd[cmd.index("--config") + 1] == "configs/e3/gpu_pilot_closed_loop.yaml"
        assert "--compile" not in cmd and cmd[cmd.index("--device") + 1] == "cuda"
    pre = cluster.commands("preflight", "python", Path("/o"), batch="gpu_pilot_cl")[0][1]
    assert pre[2].endswith("preflight_agent.py") and pre[pre.index("--cpus") + 1] == "3"
    res = cluster.commands("resume", "python", Path("/o"), resume="/r/run", batch="gpu_pilot_cl")[0][1]
    assert res[2].endswith("run_agent.py") and res[res.index("--resume") + 1] == "/r/run"
    with pytest.raises(ValueError, match="closed-loop batch"):
        cluster.commands("measure", "python", Path("/o"), batch="gpu_pilot_cl")
    assert cluster.resolve_cpus("gpu_pilot_cl", None) == 3


def test_pilot_remeasure_batch():
    """docs/e3/DECISIONS.md, 2026-10-05 (23): the pilot's 4 runs re-measured at 5 checkpoints, read in place."""
    from pathlib import Path

    jobs = cluster.commands("measure", "python", Path("/o"), batch="gpu_pilot_remeasure")
    assert [n for n, _ in jobs][-1] == "value_diagnostics" and len(jobs) == 5  # 2026-10-05 (24): the diagnostic last
    diag = jobs[-1][1]
    assert diag[2].endswith("diagnose_value_generalisation.py")
    points = diag[diag.index("--points") + 1:diag.index("--device")]
    assert [p.rsplit(":", 1)[1] for p in points] == ["10000", "15849", "39811"]
    assert "pair-control" in points[0] and "seed00" in points[0] and "pair-experimental" in points[2] and "seed05" in points[2]
    for name, cmd in jobs[:4]:
        assert cmd[2].endswith("remeasure_checkpoints.py") and cmd[3].startswith(cluster.PILOT_RESULTS + "/")
        assert cmd[cmd.index("--env-steps") + 1:cmd.index("--device")] == ["63096", "100000", "158489", "251189", "300000"]
        assert cmd[cmd.index("--out-dir") + 1] == str(Path("/o") / "runs")
    with pytest.raises(ValueError, match="measurement batch"):
        cluster.commands("train", "python", Path("/o"), batch="gpu_pilot_remeasure")
    assert cluster.resolve_cpus("gpu_pilot_remeasure", None) == 3


def test_pilot_noise_batch():
    """docs/e3/DECISIONS.md, 2026-10-06 (26): 4 noise jobs on the pilot's final weights, then the goal control."""
    from pathlib import Path

    jobs = cluster.commands("measure", "python", Path("/o"), batch="gpu_pilot_noise")
    assert [n for n, _ in jobs][-1] == "goal_control" and len(jobs) == 5
    for name, cmd in jobs[:4]:
        assert cmd[2].endswith("measure_noise.py") and cmd[cmd.index("--repeats") + 1] == "8"
        assert cmd[cmd.index("--contexts-multipliers") + 1:cmd.index("--checkpoint-steps")] == ["1", "2", "4"]
        assert cmd[cmd.index("--checkpoint-steps") + 1:cmd.index("--checkpoint-multipliers")] == ["63096", "100000", "158489", "251189", "300000"]
        assert cmd[cmd.index("--checkpoint-multipliers") + 1:cmd.index("--device")] == ["2", "4"]  # 2026-10-06 (27): x2 first
    fallback = cluster.commands("measure", "python", Path("/o"), batch="gpu_pilot_noise_x2")[0][1]
    assert fallback[fallback.index("--checkpoint-multipliers") + 1:fallback.index("--device")] == ["2"]
    assert cluster.MEASURE_BATCHES["gpu_pilot_noise"]["fallback_batch"] == "gpu_pilot_noise_x2"
    g = jobs[4][1]
    assert g[g.index("--goal-control") + 1].endswith("seed00__20261003T144527Z:15849") and "pair-control" in g[g.index("--goal-control") + 1]
    assert g[g.index("--draws") + 1] == "10000"
    assert cluster.resolve_cpus("gpu_pilot_noise", None) == 3 and cluster.MEASURE_BATCHES["gpu_pilot_noise"]["max_batch_hours"] == 8.0
