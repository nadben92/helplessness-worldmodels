"""diagnose_value_generalisation.py (docs/e3/DECISIONS.md, 2026-10-05 (24)) on a tiny closed-loop run: the distance to
the goal under P, the actions-played summary, and one point end to end."""

from __future__ import annotations

import json

import numpy as np

from _experiment_loader import load_experiment_module
from test_e3_closed_loop import run_agent, run_mini, tiny_config

vd = load_experiment_module("E3_worldmodel_4zones", "diagnose_value_generalisation.py", "test_value_diag")


def test_distance_to_goal_on_a_chain():
    # 3 states in a line, goal 0; action 0 moves left deterministically, action 1 stays
    P = np.zeros((3, 2, 3))
    P[0, :, 0] = 1
    P[1, 0, 0] = P[2, 0, 1] = 1
    P[1, 1, 1] = P[2, 1, 2] = 1
    assert vd.distance_to_goal(P, 0).tolist() == [0.0, 1.0, 2.0]


def test_actions_played_and_repeat():
    arr = {"a": np.array([1, 1, 2, 0, 3, 3, 3, 0]), "is_first": np.array([1, 0, 0, 0, 1, 0, 0, 0], bool),
           "terminal": np.array([0, 0, 0, 1, 0, 0, 0, 0], bool), "truncated": np.zeros(8, bool)}
    out = vd.actions_played(arr, 0, 10)
    assert out["n"] == 7 and out["repeat_previous_action"] == 3 / 5  # pairs within episodes: (1,1) (1,2) (3,3) (3,3) (3,0)


def test_one_point_end_to_end(tmp_path):
    cfg = tiny_config()
    cfg["measure"]["final"] = False
    cfg["measure"]["checkpoint_goal_kernel"] = {"draws": 4}
    d = tmp_path / "r"
    d.mkdir()
    run_agent.run_one(cfg, 0, "experimental", d, run_mini.configure_runtime(cfg))
    out = tmp_path / "o"
    assert vd.main(["--points", f"{d}:200", "--device", "cpu", "--out-dir", str(out)]) == 0
    r = json.loads((out / "value_diagnostics.json").read_text())[0]
    assert r["env_step"] == 200 and r["actions_played_interval"]["from"] == 100
    assert -1.0 <= r["spearman_V_vs_distance_to_goal"]["rho"] <= 1.0 or np.isnan(r["spearman_V_vs_distance_to_goal"]["rho"])
    assert 0.0 <= r["greedy_optimal_share"] <= 1.0 and len(r["goal_pair_masses_at_c"]) == 4
