"""preflight_agent.py (docs/e3/DECISIONS.md, 2026-10-02 (19)) without a GPU: the parallelism choice, the projection,
the decision lines, and the check of the truncated run's outputs."""

from __future__ import annotations

import json

import pytest
import yaml

from _experiment_loader import load_experiment_module

pa = load_experiment_module("E3_worldmodel_4zones", "preflight_agent.py", "test_preflight_agent")


def test_parallelism_projection_and_decision():
    assert pa.choose_parallel(2.0, 2.1) == 4 and pa.choose_parallel(2.0, 1.9) == 2
    assert pa.projected_hours(300_000, 1.0, 4, 4) == pytest.approx(300_000 / 3600)  # one round
    assert pa.projected_hours(300_000, 1.0, 2, 4) == pytest.approx(2 * 300_000 / 3600)  # experimental first, then references
    assert pa.decide([], 50.0, 336.0, 4) == "launch-max-parallel-4" and pa.decide([], 50.0, 336.0, 2) == "launch-max-parallel-2"
    assert pa.decide(["x"], 50.0, 336.0, 4) == "stop" and pa.decide([], 400.0, 336.0, 4) == "stop" and pa.decide([], None, 336.0, None) == "stop"


def test_truncated_run_outputs(tmp_path):
    d = tmp_path / "run"
    (d / "checkpoints").mkdir(parents=True)
    for f in pa.OUTPUTS:
        (d / f).write_text("x", encoding="utf-8")
    (d / "meta.yaml").write_text(yaml.safe_dump({"status": "done", "device": "cuda:0"}), encoding="utf-8")
    (d / "checkpoints.jsonl").write_text(json.dumps({"env_step": 1000, "goal_diagnostics": {}}) + "\n", encoding="utf-8")
    assert pa.check_truncated(d) == []
    (d / "checkpoints" / "agent_env001000.pt").unlink()
    (d / "checkpoints.jsonl").write_text(json.dumps({"env_step": 1000}) + "\n", encoding="utf-8")
    problems = pa.check_truncated(d)
    assert any("agent_env001000.pt" in p for p in problems) and any("goal_diagnostics" in p for p in problems)


def test_campaign_projection_and_truncated_outputs(tmp_path):
    """docs/e3/DECISIONS.md, 2026-10-07 (29): the projection adds each run's measures (MEASURED_CHECKPOINTS full
    measurements, GOAL_DIAGNOSTICS goal diagnostics, x CONTENTION); the campaign's truncated run must also write twin.npz,
    replay_final.npz and the agent's goal diagnostics."""
    smoke = {"seconds_full_measures": 600.0, "seconds_goal_diagnostics": 40.0}
    extra = pa.CONTENTION * (pa.MEASURED_CHECKPOINTS * 600.0 + pa.GOAL_DIAGNOSTICS * 40.0) / 3600
    assert pa.run_hours(300_000, 2.0, smoke) == pytest.approx(300_000 / 2.0 / 3600 + extra)
    assert pa.projected_hours(300_000, 2.0, 4, 8, smoke) == pytest.approx(2 * (300_000 / 2.0 / 3600 + extra))
    assert pa.projected_hours(300_000, 2.0, 4, 8) == pytest.approx(2 * 300_000 / 2.0 / 3600)
    cfg = pa.run_mini.load_mini_config(pa.run_mini.REPO_ROOT / "configs" / "e3" / "gpu_campaign.yaml")
    d = tmp_path / "run"
    (d / "checkpoints").mkdir(parents=True)
    for f in pa.OUTPUTS:
        (d / f).write_text("x", encoding="utf-8")
    (d / "meta.yaml").write_text(yaml.safe_dump({"status": "done", "device": "cuda:0"}), encoding="utf-8")
    (d / "checkpoints.jsonl").write_text(json.dumps({"env_step": 1000, "goal_diagnostics": {}}) + "\n", encoding="utf-8")
    problems = pa.check_truncated(d, cfg)
    assert any("twin.npz" in p for p in problems) and any("replay_final.npz" in p for p in problems)
    assert any("goal_neighbours" in p for p in problems)
    for f in ("twin.npz", "replay_final.npz"):
        (d / f).write_text("x", encoding="utf-8")
    (d / "checkpoints.jsonl").write_text(json.dumps({"env_step": 1000, "goal_diagnostics": {
        "goal_neighbours": [], "V_by_zone": {}, "latent_zone_separation": {}}}) + "\n", encoding="utf-8")
    assert pa.check_truncated(d, cfg) == [] and pa.check_truncated(d) == []


def test_measure_smoke_on_a_finished_run(tmp_path):
    """Check 6 on a tiny finished run (read only): the goal diagnostics and one full checkpoint measurement, timed, into
    the preflight's own directory; the run's directory is not modified."""
    from test_e3_campaign_run import campaign_tiny

    run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_preflight_agent_run_agent")
    cfg = campaign_tiny()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_agent.run_one(cfg, 0, "reference", run_dir, pa.run_mini.configure_runtime(cfg))
    before = sorted((p.relative_to(run_dir).as_posix(), p.stat().st_mtime_ns) for p in run_dir.rglob("*"))
    smoke = pa.measure_smoke(cfg, run_dir, pa.run_mini.configure_runtime(cfg), tmp_path / "pre" / "measure_smoke")
    assert sorted((p.relative_to(run_dir).as_posix(), p.stat().st_mtime_ns) for p in run_dir.rglob("*")) == before
    assert smoke["seconds_full_measures"] > 0 and smoke["seconds_goal_diagnostics"] > 0 and smoke["contexts"] > 0
    assert len(smoke["goal_pair_masses"]) == 4 and set(smoke["I_s_by_zone"]) == set(pa.run_mini.e3_env.ZONE_CATEGORIES)
    assert (tmp_path / "pre" / "measure_smoke" / "measures" / "env000300" / "arrays.npz").exists()
