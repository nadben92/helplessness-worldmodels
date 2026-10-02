"""The closed-loop GPU pilot's config (docs/e3/DECISIONS.md, 2026-10-02 (19), (20)) is the configuration retained for
E3 -- gpu_e30b_win64.yaml (16 windows x 64 steps) -- for the network, rendering, training procedure and measures, with
agent_mini.yaml's agent; it differs only by what the closed loop needs, and has no acceleration."""

from __future__ import annotations

from _experiment_loader import load_experiment_module

run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_pilot_config_run_agent")
run_mini = run_agent.run_mini
CONFIGS = run_mini.REPO_ROOT / "configs" / "e3"
# run_mini.py's schedule keys (unused by run_agent.py) and the closed loop's own measure keys
MEASURE_SCHEDULE = {"separation_every", "monitor_every", "diagnostics_every", "final", "checkpoint_goal_kernel"}


def test_pilot_is_win64_with_the_closed_loop_agent():
    pilot = run_mini.load_mini_config(CONFIGS / "gpu_pilot_closed_loop.yaml")
    win64 = run_mini.load_mini_config(CONFIGS / "gpu_e30b_win64.yaml")
    bench = run_mini.load_mini_config(CONFIGS / "agent_mini.yaml")
    assert pilot["model"] == win64["model"] and pilot["render"] == win64["render"]
    # training: win64's entirely, replay ratio included (DECISIONS 2026-10-02 (21)): B = 16, T = 64, 0.6
    assert pilot["train"] == win64["train"]
    assert (pilot["train"]["batch_size"], pilot["train"]["seq_len"], pilot["train"]["replay_ratio"]) == (16, 64, 0.6)
    assert {k: v for k, v in pilot["measure"].items() if k not in MEASURE_SCHEDULE} == \
        {k: v for k, v in win64["measure"].items() if k not in MEASURE_SCHEDULE}
    assert pilot["measure"]["final"] is True and pilot["measure"]["checkpoint_goal_kernel"] == {"draws": 256}
    assert pilot["collection"]["start_distribution"] == win64["collection"]["start_distribution"]
    # the closed loop's agent, twin and logging: agent_mini.yaml's
    for block in ("agent", "twin", "log"):
        assert pilot[block] == bench[block], block
    assert pilot["budget"] == {"total_steps": 299_744, "prefill_steps": 256}
    assert pilot["checkpoints"] == {"grid": "e2", "resume_every": 10_000}
    assert pilot["runtime"] == {"num_threads": 1, "device": "cuda", "deterministic": True}
    assert pilot["seeds"] == [0, 5] and pilot["conditions"] == ["experimental", "reference"]
    assert "compile" not in pilot["runtime"] and "stacked_seeds" not in pilot["runtime"]  # no acceleration
    steps = run_agent.checkpoint_env_steps(pilot, 300_000)
    assert steps[0] == 1000 and steps[-2:] == [251_189, 300_000] and len(steps) == 14
    acfg = run_agent.agent_config(pilot)
    assert abs(acfg.tie_tol_abs - 0.5 * 6 * 0.9 ** 35) < 1e-12 and acfg.tie_tol_rel is None
    run_mini.build_separation_switch(pilot)
