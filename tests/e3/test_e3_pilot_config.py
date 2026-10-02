"""The closed-loop GPU pilot's config (docs/e3/DECISIONS.md, 2026-10-02 (19)) is agent_mini.yaml's network, procedure
and agent, with only the budget, runtime, seeds / conditions, resume interval and checkpoint goal diagnostics changed,
and no acceleration."""

from __future__ import annotations

from _experiment_loader import load_experiment_module

run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_pilot_config_run_agent")
run_mini = run_agent.run_mini
CONFIGS = run_mini.REPO_ROOT / "configs" / "e3"


def test_pilot_is_agent_mini_except_budget_runtime_and_checkpoint_diagnostics():
    pilot, bench = run_mini.load_mini_config(CONFIGS / "gpu_pilot_closed_loop.yaml"), run_mini.load_mini_config(CONFIGS / "agent_mini.yaml")
    for block in ("model", "render", "train", "agent", "twin", "collection", "log"):
        assert pilot[block] == bench[block], block
    assert pilot["budget"] == {"total_steps": 299_744, "prefill_steps": 256}
    assert pilot["train"]["replay_ratio"] == 2
    assert {k: v for k, v in pilot["measure"].items() if k != "checkpoint_goal_kernel"} == bench["measure"]
    assert pilot["measure"]["checkpoint_goal_kernel"] == {"draws": 256}
    assert pilot["checkpoints"] == {"grid": "e2", "resume_every": 10_000}
    assert pilot["runtime"] == {"num_threads": 1, "device": "cuda", "deterministic": True}
    assert pilot["seeds"] == [0, 5] and pilot["conditions"] == ["experimental", "reference"]
    assert "compile" not in pilot["runtime"] and "stacked_seeds" not in pilot["runtime"]  # no acceleration
    steps = run_agent.checkpoint_env_steps(pilot, 300_000)
    assert steps[0] == 1000 and steps[-2:] == [251_189, 300_000] and len(steps) == 14
    acfg = run_agent.agent_config(pilot)
    assert abs(acfg.tie_tol_abs - 0.5 * 6 * 0.9 ** 35) < 1e-12 and acfg.tie_tol_rel is None
