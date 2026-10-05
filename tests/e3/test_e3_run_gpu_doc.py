"""RUN_GPU.md (docs/e3/DECISIONS.md, 2026-09-30 (3); batch gpu_e30b_measures, 2026-10-01 (10)) is executed as written by
an assistant on the GPU environment: every command it gives must exist and
accept its options, the hash command must run on the standard library alone,
and the decision line it quotes must be the preflight's."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from _experiment_loader import load_experiment_module

cluster = load_experiment_module("E3_worldmodel_4zones", "submit_kubeflow.py", "test_doc_cluster")
handoff = load_experiment_module("E3_worldmodel_4zones", "make_handoff.py", "test_doc_handoff")
REPO = cluster.Path(__file__).resolve().parents[2]
DOC = (REPO / "experiments" / "E3_worldmodel_4zones" / "RUN_GPU.md").read_text(encoding="utf-8")


def _commands(script: str) -> list[list[str]]:
    lines = [line.strip() for line in DOC.splitlines() if line.strip().startswith(f"python experiments/E3_worldmodel_4zones/{script}")]
    return [line.split()[2:] for line in lines]


def test_every_submission_command_parses(tmp_path, monkeypatch):
    """Batch gpu_pilot_remeasure (docs/e3/DECISIONS.md, 2026-10-05 (23)): the preflight and the measurement, 4 jobs, 3 CPUs."""
    commands = _commands("submit_kubeflow.py")
    assert {tuple(c) for c in commands} == {
        ("--phase", "preflight", "--batch", "gpu_pilot_remeasure", "--max-parallel", "4", "--cpu", "3"),
        ("--phase", "measure", "--batch", "gpu_pilot_remeasure", "--max-parallel", "4", "--cpu", "3")}
    mount = tmp_path / "scratch"
    repo = mount / "helplessness-worldmodels"
    repo.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "kubeflow", None)  # the submission itself needs the cluster's SDK
    for argv in commands:
        with pytest.raises(ImportError):  # the arguments passed every check; only the SDK import is left
            cluster.main([*argv, "--repo-dir", str(repo), "--mount", str(mount)])
        phase = argv[argv.index("--phase") + 1]
        jobs = cluster.commands(phase, "python", tmp_path, batch="gpu_pilot_remeasure", max_parallel=4, cpus=3)
        assert len(jobs) == (1 if phase == "preflight" else 5)
        if phase == "measure":
            assert all(cmd[2].endswith("remeasure_checkpoints.py") for _, cmd in jobs[:4])
            assert jobs[4][1][2].endswith("diagnose_value_generalisation.py")
    for run_dir, _ in cluster.MEASURE_BATCHES["gpu_pilot_remeasure"]["runs"].values():
        assert run_dir in DOC  # the doc names every run it reads
    assert cluster.PILOT_RESULTS in DOC


def test_handoff_commands_use_existing_options(capsys):
    commands = _commands("make_handoff.py")
    assert len(commands) == 2
    with pytest.raises(SystemExit):
        handoff.main(["--help"])
    help_text = capsys.readouterr().out
    for argv in commands:
        for flag in (a for a in argv if a.startswith("--")):
            assert flag in help_text, flag
        assert "--out" in argv and argv[argv.index("--out") + 1] == "handoff/gpu_pilot_remeasure"


def test_hash_command_runs_on_the_standard_library_alone():
    line = next(line.strip() for line in DOC.splitlines() if line.strip().startswith('python -c "import sys; sys.path.insert(0, \'src\')'))
    code = re.match(r'python -c "(.*)"$', line).group(1)
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    assert re.fullmatch(r"\{'source_sha256': '[0-9a-f]{64}', 'source_n_files': \d+\}", out), out
    expected = re.search(r"\{'source_sha256': '([^']*)', 'source_n_files': ([^}]*)\}", DOC.split("must print exactly:")[1])
    assert expected, "RUN_GPU.md states the expected hash"
    assert out == f"{{'source_sha256': '{expected.group(1)}', 'source_n_files': {expected.group(2)}}}", \
        "the hash in RUN_GPU.md is not the checkout's: update it when the code changes"


def test_decision_lines_are_the_preflight_s():
    preflight_src = (REPO / "experiments" / "E3_worldmodel_4zones" / "preflight_measures.py").read_text(encoding="utf-8")
    assert 'print(f"PREFLIGHT DECISION: {decision}' in preflight_src
    for decision in ("launch", "stop"):
        assert f"PREFLIGHT DECISION: {decision}" in DOC
    assert "<= 4" in DOC and 'parser.add_argument("--max-batch-hours", type=float, default=4.0)' in preflight_src
