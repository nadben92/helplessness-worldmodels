"""RUN_GPU.md (docs/e3/DECISIONS.md, 2026-09-30 (3); batch gpu_campaign, 2026-10-07 (29)) is executed as written by
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
    """Batch gpu_campaign (docs/e3/DECISIONS.md, 2026-10-07 (29), (30)): the preflight, the 8 jobs at 4 or 2 at
    once, and the resume of one run (only with the user's agreement); 3 CPUs everywhere."""
    commands = _commands("submit_kubeflow.py")
    assert {tuple(c) for c in commands} == {
        ("--phase", "preflight", "--batch", "gpu_campaign", "--max-parallel", "4", "--cpu", "3"),
        ("--phase", "train", "--batch", "gpu_campaign", "--max-parallel", "4", "--cpu", "3"),
        ("--phase", "train", "--batch", "gpu_campaign", "--max-parallel", "2", "--cpu", "3"),
        ("--phase", "resume", "--batch", "gpu_campaign", "--cpu", "3", "--resume", "<the", "run's", "directory>")}
    mount = tmp_path / "scratch"
    repo = mount / "helplessness-worldmodels"
    repo.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "kubeflow", None)  # the submission itself needs the cluster's SDK
    for argv in commands:
        if argv[argv.index("--phase") + 1] == "resume":
            argv = argv[:argv.index("--resume") + 1] + [str(mount / "a_run")]
        with pytest.raises(ImportError):  # the arguments passed every check; only the SDK import is left
            cluster.main([*argv, "--repo-dir", str(repo), "--mount", str(mount)])
        phase = argv[argv.index("--phase") + 1]
        mp = int(argv[argv.index("--max-parallel") + 1]) if "--max-parallel" in argv else None
        jobs = cluster.commands(phase, "python", tmp_path, resume=str(mount / "a_run"), batch="gpu_campaign", max_parallel=mp, cpus=3)
        if phase == "train":  # the 4 new runs, then the 4 pilot runs measured as campaign seeds (2026-10-07 (30))
            assert [name for name, _ in jobs] == [f"seed{s}_{c}" for s in (1, 6) for c in ("experimental", "reference")] +                 [f"pilot_win64_seed{s}_{c}" for c in ("experimental", "reference") for s in (0, 5)]
            assert all(cmd[2].endswith("run_agent.py") and "configs/e3/gpu_campaign.yaml" in cmd for _, cmd in jobs[:4])
            assert all(cmd[2].endswith("campaign_measure_pilot.py") and "configs/e3/gpu_campaign.yaml" in cmd for _, cmd in jobs[4:])
        elif phase == "preflight":
            assert jobs[0][1][2].endswith("preflight_agent.py")
        else:
            assert jobs[0][1][2].endswith("run_agent.py") and "--resume" in jobs[0][1]
    assert cluster.AGENT_BATCHES["gpu_campaign"]["config"] in DOC
    assert cluster.AGENT_BATCHES["gpu_campaign"]["measure_smoke_run"].rsplit("/", 1)[1] in DOC  # the run read by the preflight
    for run_dir in cluster.AGENT_BATCHES["gpu_campaign"]["pilot_runs"].values():
        assert run_dir.rsplit("/", 1)[1] in DOC  # the doc names every pilot run it reads


def test_handoff_commands_use_existing_options(capsys):
    commands = _commands("make_handoff.py")
    assert len(commands) == 2
    with pytest.raises(SystemExit):
        handoff.main(["--help"])
    help_text = capsys.readouterr().out
    for argv in commands:
        for flag in (a for a in argv if a.startswith("--")):
            assert flag in help_text, flag
        assert "--out" in argv and argv[argv.index("--out") + 1] == "handoff/gpu_campaign"
    assert "--with-weights" in commands[0]  # the weights and the final replay come back (2026-10-07 (29))


LAUNCHED_CAMPAIGN_SHA256 = "d84dec84a3bd9542ea4846502ae30b20e9909c9c7b49b6e8ac019a4fd21d1a80"


def test_hash_command_runs_on_the_standard_library_alone():
    line = next(line.strip() for line in DOC.splitlines() if line.strip().startswith('python -c "import sys; sys.path.insert(0, \'src\')'))
    code = re.match(r'python -c "(.*)"$', line).group(1)
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    assert re.fullmatch(r"\{'source_sha256': '[0-9a-f]{64}', 'source_n_files': \d+\}", out), out
    expected = re.search(r"\{'source_sha256': '([^']*)', 'source_n_files': ([^}]*)\}", DOC.split("must print exactly:")[1])
    assert expected, "RUN_GPU.md states the expected hash"
    if expected.group(1) == LAUNCHED_CAMPAIGN_SHA256:
        # gpu_campaign was launched with this code (docs/e3/DECISIONS.md, 2026-10-13 (31)): RUN_GPU.md stays the record
        # of what runs; the analysis code added since changes the checkout's hash. Checked again for the next batch.
        pytest.skip("RUN_GPU.md records the launched gpu_campaign's code; the checkout has moved on since")
    assert out == f"{{'source_sha256': '{expected.group(1)}', 'source_n_files': {expected.group(2)}}}", \
        "the hash in RUN_GPU.md is not the checkout's: update it when the code changes"


def test_decision_lines_are_the_preflight_s():
    preflight_src = (REPO / "experiments" / "E3_worldmodel_4zones" / "preflight_agent.py").read_text(encoding="utf-8")
    assert 'print(f"PREFLIGHT DECISION: {decision}' in preflight_src and 'return f"launch-max-parallel-{parallel}"' in preflight_src
    for decision in ("launch-max-parallel-4", "launch-max-parallel-2", "stop"):
        assert f"PREFLIGHT DECISION: {decision}" in DOC
