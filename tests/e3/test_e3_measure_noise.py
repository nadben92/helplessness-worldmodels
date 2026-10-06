"""measure_noise.py (docs/e3/DECISIONS.md, 2026-10-06 (26)) on a tiny closed-loop run: the noise mode gives one I_s by
zone per repeat and their standard deviation per context multiplier; the goal-control mode gives the four pairs into
the goal with their mass and Q gap."""

from __future__ import annotations

import json

import numpy as np

from _experiment_loader import load_experiment_module
from test_e3_closed_loop import run_agent, run_mini, tiny_config

mn = load_experiment_module("E3_worldmodel_4zones", "measure_noise.py", "test_measure_noise")


def test_noise_and_goal_control(tmp_path):
    cfg = tiny_config()
    cfg["measure"]["final"] = False
    cfg["measure"]["checkpoint_goal_kernel"] = {"draws": 4}
    d = tmp_path / "r"
    d.mkdir()
    run_agent.run_one(cfg, 0, "experimental", d, run_mini.configure_runtime(cfg))
    out = tmp_path / "o"
    assert mn.main(["--noise", str(d), "--repeats", "3", "--contexts-multipliers", "1", "2", "--device", "cpu", "--out-dir", str(out)]) == 0
    r = json.loads((out / f"noise__{d.name}.json").read_text())
    assert r["replay_env_steps"] == 250 and set(r["by_multiplier"]) == {"1", "2"}
    m1, m2 = r["by_multiplier"]["1"], r["by_multiplier"]["2"]
    assert m2["max_per_state"] == 2 * m1["max_per_state"] and len(m1["repeats"]) == 3
    assert all(x["contexts"] > 0 for x in m1["repeats"]) and m2["repeats"][0]["contexts"] >= m1["repeats"][0]["contexts"]
    assert set(m1["sd_by_zone"]) == set(run_mini.e3_env.ZONE_CATEGORIES)
    # repeats are independent draws: the same repeat index gives the same value (seeded), different indices differ
    again = mn.noise(d, 1, [1], "cpu")["by_multiplier"]["1"]["repeats"][0]["I_s_by_zone"]
    assert again == m1["repeats"][0]["I_s_by_zone"]
    assert mn.main(["--goal-control", f"{d}:200", "--draws", "50", "--device", "cpu", "--out-dir", str(out)]) == 0
    g = json.loads((out / "goal_control.json").read_text())
    assert len(g["goal_pairs"]) == 4 and all(p["tie_tolerance"] == pytest_approx(0.5 * 6 * 0.9 ** 35) for p in g["goal_pairs"])
    assert all(np.isfinite(p["gap_into_goal_minus_best_other"]) for p in g["goal_pairs"] if p["contexts"] > 0)


def pytest_approx(x):
    import pytest

    return pytest.approx(x)


def test_checkpoints_mode(tmp_path):
    """2026-10-06 (27): I_s by zone of each checkpoint's weights on the replay as of it, once per multiplier."""
    cfg = tiny_config()
    cfg["measure"]["final"] = False
    cfg["measure"]["checkpoint_goal_kernel"] = {"draws": 4}
    d = tmp_path / "r"
    d.mkdir()
    run_agent.run_one(cfg, 0, "experimental", d, run_mini.configure_runtime(cfg))
    out = tmp_path / "o"
    assert mn.main(["--noise", str(d), "--repeats", "2", "--contexts-multipliers", "1", "--checkpoint-steps", "100", "300",
                    "--checkpoint-multipliers", "2", "--device", "cpu", "--out-dir", str(out)]) == 0
    assert (out / f"noise__{d.name}.json").exists()
    c = json.loads((out / f"checkpoints__{d.name}.json").read_text())
    rows = c["by_multiplier"]["2"]["by_checkpoint"]
    assert set(rows) == {"100", "300"} and rows["100"]["replay_env_steps"] == 100 and rows["300"]["replay_env_steps"] == 250
    assert c["by_multiplier"]["2"]["max_per_state"] == 2 * c["base_max_per_state"]
