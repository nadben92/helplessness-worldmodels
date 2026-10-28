"""E6 / E7's batch table and submission commands (no cluster involved), and the agents files."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline import frozen_setup as fs

batches = load_experiment_module("E6_phenotypes_latent", "batches.py", "e6_batches_under_test")
submit = load_experiment_module("E6_phenotypes_latent", "submit_kubeflow.py", "e6_submit_under_test")


def test_agent_ids_of_both_files():
    assert batches.agent_ids(REPO_ROOT, "e6_dev_e4") == ["p1-a-d1_e4", "p1-b-d1_e4", "p2-a-d1_e4", "p2-b-d1_e4"]
    # every agent of E6/E7 on all agents: E4's four pairs (E5's baseline A is E4's p1-b, listed once) and E5's B, E, F
    assert batches.agent_ids(REPO_ROOT, "e6_e5") == ["p1-a_e4", "base_A", "p2-a_e4", "p2-b_e4", "base_B", "f1_E", "f1_F"]
    raw = yaml.safe_load((REPO_ROOT / "configs" / "e6" / "agents_e5.yaml").read_text(encoding="utf-8"))
    by_id = {a["id"]: a for a in raw["agents"]}
    assert all(by_id[a][r]["checkpoint"] == 475000 for a in ("f1_E", "f1_F") for r in ("experimental", "reference"))
    assert "seed102" in by_id["f1_E"]["reference"]["run"] and by_id["f1_E"]["reference"] == by_id["f1_F"]["reference"]


@pytest.mark.parametrize("phase", list(batches.PHASES))
def test_jobs_of_every_phase(phase):
    jobs = submit.commands("py", "e6_dev_e4", phase, "/scratch", REPO_ROOT, "cuda", 2)
    assert [n for n, _ in jobs] == [f"{phase}_p1-a-d1_e4_p1-b-d1_e4", f"{phase}_p2-a-d1_e4_p2-b-d1_e4"]
    for _, cmd in jobs:
        assert cmd[:3] == ["py", "-u", batches.PHASES[phase]] and (REPO_ROOT / cmd[2]).is_file()
        assert cmd[cmd.index("--runs-root") + 1] == "/scratch" and cmd[cmd.index("--device") + 1] == "cuda"
        assert ("--constrained" in cmd) == (phase == "e7-apply") and ("--directions" in cmd) == (phase in ("e7-apply", "e7-constrained"))
    with pytest.raises(ValueError):
        batches.jobs(REPO_ROOT, "e6_dev_e4", phase, "/scratch", only=["nobody"])


def test_dry_run_prints_without_submitting(capsys):
    assert submit.main(["--batch", "e6_e5", "--phase", "e6", "--runs-root", "/scratch", "--repo-dir", str(REPO_ROOT), "--dry-run"]) == 0
    assert '"jobs"' in capsys.readouterr().out


def test_agents_files_parse_and_resolve(tmp_path):
    for name in ("agents_dev_e4.yaml", "agents_e5.yaml"):  # (load_agents reads YAML: commented entries are not agents)
        with pytest.raises(FileNotFoundError):  # nothing under an empty root: the globs are explicit errors, never silent
            fs.load_agents(REPO_ROOT / "configs" / "e6" / name, tmp_path)
    run = tmp_path / "E4_generalization_gradient" / "grid12x12__2zones-p1-a-d1__pair-experimental__worldmodel-agent-e4__seed01__X" / "checkpoints"
    run.mkdir(parents=True)
    (run / "agent_env300000.pt").write_bytes(b"")
    ref = Path(str(run).replace("pair-experimental", "pair-control"))
    ref.mkdir(parents=True)
    (ref / "agent_env300000.pt").write_bytes(b"")
    agents = fs.load_agents(REPO_ROOT / "configs" / "e6" / "agents_dev_e4.yaml", tmp_path, only=["p1-a-d1_e4"])
    assert agents[0]["weights"]["experimental"] == run / "agent_env300000.pt" and agents[0]["condition"] == "dev_pair1"
