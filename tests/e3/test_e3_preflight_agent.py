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
