"""The campaign's in-run measures (docs/e3/DECISIONS.md, 2026-10-07 (29)) on a TINY configuration: full measures at every
checkpoint after the switch (plus an extra checkpoint when the grid has too few in the window after it), the goal pairs
with their own number of draws, the agent's Q gaps next to the goal, V by zone, the zones' separation in the latent,
the twin's kernels, the final replay, light arrays; the final measures are the last checkpoint's; a resumed run gives
the same outputs as an uninterrupted one."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from _experiment_loader import load_experiment_module

closed_loop_tests = load_experiment_module("..", "tests/e3/test_e3_closed_loop.py", "test_campaign_closed_loop") \
    if False else None
run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_campaign_run_agent")
run_mini = run_agent.run_mini

SPEC = {"window_after_switch": 50, "min_in_window": 2, "extra_offsets": [25, 40]}


def campaign_tiny() -> dict:
    from test_e3_closed_loop import tiny_config

    cfg = tiny_config()
    cfg["measure"].update(final=True, light_arrays=True, checkpoint_measures=dict(SPEC),
                          checkpoint_goal_kernel={"draws": 8, "goal_draws": 16, "agent": True})
    cfg["twin"]["save_kernels"] = True
    cfg["checkpoints"].update(save_final_replay=True)
    return cfg


def test_extra_checkpoints_rule():
    spec = {"window_after_switch": 50_000, "min_in_window": 2, "extra_offsets": [25_000, 50_000]}
    grid = run_mini.e2_grid_env_steps()
    assert run_agent.extra_checkpoints(10_758, grid, 300_000, spec) == []          # 15,849, 25,119, 39,811 in the window
    assert run_agent.extra_checkpoints(41_000, grid, 300_000, spec) == [66_000]    # only 63,096: one more, 25,000 after
    assert run_agent.extra_checkpoints(260_000, grid, 300_000, spec) == [285_000]  # 300,000 is the run's end (in the grid)
    assert run_agent.extra_checkpoints(290_000, [300_000], 300_000, spec) == []     # nothing at or past the end


def _run(cfg, tmp_path, name, **kwargs):
    run_dir = tmp_path / name
    run_dir.mkdir()
    report = run_agent.run_one(cfg, 0, "experimental", run_dir, run_mini.configure_runtime(cfg), **kwargs)
    return run_dir, report


def test_campaign_measures_in_the_run(tmp_path):
    cfg = campaign_tiny()
    run_dir, report = _run(cfg, tmp_path, "run")
    records = report["checkpoints"]
    log = (run_dir / "console.log").read_text(encoding="utf-8")
    switch_env = switch_env_from_log(log)
    measured = [r["env_step"] for r in records if "measures" in r]
    steps = [r["env_step"] for r in records]
    assert measured == [e for e in steps if e > switch_env] and 300 in measured  # every checkpoint after the switch, only those
    assert sorted(steps) == steps and set(steps) >= {100, 200, 300}
    extra = sorted(set(steps) - {100, 200, 300})
    # every measured checkpoint: a complete measure directory, light arrays, its summary in the record
    for r in records:
        if "measures" not in r:
            continue
        d = run_dir / r["measures"]["dir"]
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        arr = np.load(d / "arrays.npz")
        assert res["checkpoint_env_step"] == r["env_step"] and res["seed"] == 0 and res["condition"] == "experimental"
        assert {"real_I_s", "real_I_s_projected", "real_off_support", "real_p_hat", "N"} <= set(arr.files)
        assert not set(run_agent.LIGHT_DROPPED) & set(arr.files)
        assert arr["N"].sum() == r["env_step"]
        assert set(r["measures"]["I_s_by_zone"]) == set(run_mini.e3_env.ZONE_CATEGORIES)
        assert {"I_s_projected_by_zone", "I_z_by_zone", "off_support_by_zone"} <= set(r["measures"])
    # the final measures are the last checkpoint's
    final_arrays, last_arrays = np.load(run_dir / "arrays.npz"), np.load(run_dir / "measures" / "env000300" / "arrays.npz")
    assert set(final_arrays.files) == set(last_arrays.files)
    assert all(np.array_equal(final_arrays[k], last_arrays[k], equal_nan=True) for k in final_arrays.files)
    # the goal diagnostics' campaign part, at every checkpoint
    for r in records:
        g = r["goal_diagnostics"]
        assert g["goal_draws"] == 16 and g["draws"] == 8 and len(g["goal_neighbours"]) == 4
        for nb in g["goal_neighbours"]:
            assert len(nb["Q"]) == 5 and nb["tie_tolerance"] > 0
        assert set(g["V_by_zone"]) == set(run_mini.e3_env.ZONE_CATEGORIES) and len(g["V_by_state"]) == run_mini.N_STATES
        sep = g["latent_zone_separation"]
        assert 0.0 <= sep["probe_zone"]["test_accuracy"] <= 1.0 and 0.0 <= sep["probe_controllability"]["test_accuracy"] <= 1.0
        assert sep["n_contexts"] == sum(sep["contexts_by_zone"].values())
    # the twin's kernels and the final replay
    twin = np.load(run_dir / "twin.npz")
    for e in steps:
        assert twin[f"T_hat_{e}"].shape == (run_mini.N_STATES, 5, run_mini.N_STATES) and twin[f"T_hat_{e}"].dtype == np.float32
        assert {f"I_hat_{e}", f"I_ref_{e}", f"b_{e}", f"delta_{e}"} <= set(twin.files)
    replay = np.load(run_dir / "replay_final.npz")
    assert len(replay["s"]) == 300 + records[-1]["episodes"]
    # extra checkpoints come only from the rule
    assert extra == run_agent.extra_checkpoints(switch_env, [100, 200, 300], 300, SPEC)


def switch_env_from_log(log: str) -> int:
    line = next(line for line in log.splitlines() if "SWITCH" in line)
    return int(line.split("environment step ")[1].split(")")[0])


def test_campaign_run_resumes_to_the_same_outputs(tmp_path):
    cfg = campaign_tiny()
    full_dir, _ = _run(cfg, tmp_path, "full")
    cut_dir = tmp_path / "cut"
    cut_dir.mkdir()
    with pytest.raises(run_agent.Interrupted):
        run_agent.run_one(cfg, 0, "experimental", cut_dir, run_mini.configure_runtime(cfg), interrupt_after=150)
    assert run_agent.main(["--resume", str(cut_dir), "--device", "cpu"]) == 0
    assert (cut_dir / "checkpoints.jsonl").read_text() == (full_dir / "checkpoints.jsonl").read_text()
    for name in ("counts.npz", "twin.npz", "replay_final.npz", "arrays.npz", "measures/env000300/arrays.npz"):
        a, b = np.load(full_dir / name), np.load(cut_dir / name)
        assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files), name
    wa = torch.load(full_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    wb = torch.load(cut_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    assert all(torch.equal(wa["rssm"][k], wb["rssm"][k]) for k in wa["rssm"])


def test_the_measures_do_not_change_the_training(tmp_path):
    """The full measures fork the global torch RNG and restore the model's mode: the run with them trains exactly as the
    run without them (same weights at the end, same behaviour)."""
    cfg = campaign_tiny()
    cfg["measure"]["checkpoint_measures"]["min_in_window"] = 0  # no extra checkpoint: the two runs have the same checkpoints
    plain = campaign_tiny()
    plain["measure"].pop("checkpoint_measures")
    with_dir, rep_with = _run(cfg, tmp_path, "with")
    without_dir, rep_without = _run(plain, tmp_path, "without")
    wa = torch.load(with_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    wb = torch.load(without_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    for part in ("rssm", "goal_head", "critic"):
        assert all(torch.equal(wa[part][k], wb[part][k]) for k in wa[part]), part
    common = [r["env_step"] for r in rep_without["checkpoints"]]
    by_step = {r["env_step"]: r for r in rep_with["checkpoints"]}
    for r in rep_without["checkpoints"]:
        assert by_step[r["env_step"]]["successes"] == r["successes"] and by_step[r["env_step"]]["episodes"] == r["episodes"]
    assert common == [100, 200, 300]
