"""remeasure_checkpoints.py (docs/e3/DECISIONS.md, 2026-10-05 (23)) on a tiny closed-loop run: the replay at a
checkpoint is exactly the prefix of the last resume point's replay; each checkpoint's weights are measured on it with
the run's counts; a checkpoint after the last resume point is measured on that replay and flagged."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch

from _experiment_loader import load_experiment_module
from test_e3_closed_loop import run_agent, run_mini, tiny_config

rm = load_experiment_module("E3_worldmodel_4zones", "remeasure_checkpoints.py", "test_remeasure")


def test_prefix_and_remeasure(tmp_path):
    cfg = tiny_config()
    cfg["measure"]["final"] = False
    d = tmp_path / "r"
    d.mkdir()
    with pytest.raises(run_agent.Interrupted):
        run_agent.run_one(cfg, 0, "experimental", d, run_mini.configure_runtime(cfg), interrupt_after=100)
    early = pickle.loads(torch.load(d / "checkpoints" / "resume.pt", weights_only=False)["replay"]).export_compact()
    assert run_agent.main(["--resume", str(d), "--device", "cpu"]) == 0
    state = torch.load(d / "checkpoints" / "resume.pt", weights_only=False)
    assert state["progress"]["env_step"] == 250  # the last resume point (resume_every 50, none at the end)
    full = pickle.loads(state["replay"])
    rec100 = next(json.loads(l) for l in (d / "checkpoints.jsonl").read_text().splitlines() if json.loads(l)["env_step"] == 100)
    prefix = rm.replay_prefix(full, 100 + rec100["episodes"]).export_compact()
    assert all(np.array_equal(prefix[k], early[k]) for k in early)  # the replay at checkpoint 100, exactly
    out = tmp_path / "o"
    paths = rm.remeasure(d, [100, 300], "cpu", out)
    r100 = json.loads((paths[0] / "results.json").read_text())
    r300 = json.loads((paths[1] / "results.json").read_text())
    assert r100["replay_env_steps"] == 100 and r100["weights"] == "agent_env000100.pt" and "I_s_projected_by_category_real" in r100
    assert r300["replay_env_steps"] == 250 and "FLAG" in (paths[1] / "console.log").read_text()
    assert (paths[0] / "arrays.npz").exists()
    with pytest.raises(ValueError, match="not a checkpoint"):
        rm.remeasure(d, [150], "cpu", tmp_path / "o2")
