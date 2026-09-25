"""E3 against E2's protocol, as specified by docs/e3/DECISIONS.md and
docs/e3/SPEC_STATUS.md (not by the initial prompt alone). Every E2 value
is READ from E2's own code (experiments/E2_online_4zones/sweep.py,
pilot.py, env.py), never copied here as a literal.

Parameters: start distribution, collection epsilon, truncation at H,
terminal at the goal only, reward, gamma, the budget and the 16-point
checkpoint grid, the independence of the random streams. What later
phases implement is marked xfail with the reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import chisquare

from _experiment_loader import load_experiment_module

from helplessness.agents.worldmodel.loop import collect_fixed_policy_data, non_terminal_states
from helplessness.agents.worldmodel.replay import ReplayBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]
e2_sweep = load_experiment_module("E2_online_4zones", "sweep.py", "conformance_e2_sweep")
e2_pilot = load_experiment_module("E2_online_4zones", "pilot.py", "conformance_e2_pilot")
e2_env = load_experiment_module("E2_online_4zones", "env.py", "conformance_e2_env")
run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "conformance_run_mini")

E3_CONFIGS = ("mini.yaml", "mini_long.yaml")
E3_EXPERIMENT_CONFIG = yaml.safe_load((REPO_ROOT / "experiments" / "E3_worldmodel_4zones" / "config.yaml").read_text(encoding="utf-8"))


def e3_config(name: str) -> dict:
    return run_mini.load_mini_config(REPO_ROOT / "configs" / "e3" / name)


def e3_env(seed: int = 0, k: int = 2):
    cfg = e3_config("mini.yaml")
    cfg["render"]["k"] = k
    return run_mini.build_run(cfg, seed)[2]


E2_R, E2_TERMINAL = e2_env.reward_and_terminal(e2_sweep.CFG, goal_reward=e2_sweep.GOAL_REWARD)
E2_BRIDGE_EPSILON = next(c["epsilon"] for c in e2_sweep.CONDITIONS if c["key"].endswith("bridge"))


# ---------------------------------------------------------------- start distribution


@pytest.mark.parametrize("name", E3_CONFIGS)
def test_configs_declare_uniform_non_terminal_starts(name):
    assert e3_config(name)["collection"]["start_distribution"] == "uniform_non_terminal"


def test_start_support_is_e2s_non_terminal_states():
    assert np.array_equal(non_terminal_states(e3_env()), e2_sweep.NON_TERMINAL_STATES)


def _chi2_uniform(starts: np.ndarray, support: np.ndarray) -> float:
    counts = np.array([(starts == s).sum() for s in support])
    assert counts.sum() == len(starts)  # nothing outside the support
    return chisquare(counts).pvalue


def test_run_mini_collection_starts_are_uniform_and_never_the_goal():
    env = e3_env(seed=0)
    replay = ReplayBuffer(capacity=40_000, image_shape=(24, 24, 3))
    starts: list[int] = []
    run_mini.collect_and_count(env, replay, n_steps=40_000, rng=np.random.default_rng(0), n_states=144, n_actions=5,
                               start_rng=np.random.default_rng([0, 9]), starts_out=starts)
    starts = np.asarray(starts)
    goal = int(np.flatnonzero(E2_TERMINAL)[0])
    assert len(starts) > 1000 and goal not in starts
    assert _chi2_uniform(starts, e2_sweep.NON_TERMINAL_STATES) > 1e-3
    first = replay._s[: len(replay)][replay._is_first[: len(replay)]]  # the replay's episode-start steps are these starts
    assert np.array_equal(first, starts[: len(first)])


def test_library_collection_starts_are_uniform_and_never_the_goal():
    env = e3_env(seed=5)
    replay = ReplayBuffer(capacity=30_000, image_shape=(24, 24, 3))
    collect_fixed_policy_data(env, replay, n_steps=30_000, rng=np.random.default_rng(5), start_rng=np.random.default_rng([5, 9]))
    starts = replay._s[: len(replay)][replay._is_first[: len(replay)]]
    assert len(starts) > 800 and int(np.flatnonzero(E2_TERMINAL)[0]) not in starts
    assert _chi2_uniform(starts, e2_sweep.NON_TERMINAL_STATES) > 1e-3


# ---------------------------------------------------------------- collection policy


@pytest.mark.parametrize("name", E3_CONFIGS)
def test_collection_epsilon_is_e2s_uniform_bridge_condition(name):
    """E3-0 (spec 6): epsilon = 1, E2's own epsilon=1 bridge condition."""
    assert e3_config(name)["collection"]["epsilon"] == E2_BRIDGE_EPSILON == 1.0


def test_collection_actions_are_uniform():
    env = e3_env(seed=0)
    replay = ReplayBuffer(capacity=20_000, image_shape=(24, 24, 3))
    run_mini.collect_and_count(env, replay, n_steps=20_000, rng=np.random.default_rng(0), n_states=144, n_actions=5,
                               start_rng=np.random.default_rng([0, 9]))
    assert chisquare(np.bincount(replay._a[: len(replay)], minlength=5)).pvalue > 1e-3


@pytest.mark.xfail(reason="closed-loop policy (epsilon-greedy on one-step lookahead) is Phase 5; not implemented yet", strict=True)
def test_closed_loop_epsilon_is_e2s():
    assert e3_config("mini.yaml")["agent"]["epsilon"] == e2_pilot.EPSILON


# ---------------------------------------------------------------- episode structure


def test_truncation_at_e2s_horizon_is_not_termination():
    env = e3_env(seed=0)
    assert env.max_steps == e2_sweep.H
    rng = np.random.default_rng(3)
    lengths_truncated, n_terminated = [], 0
    for _ in range(300):
        _, info = env.reset(start_state=int(rng.choice(e2_sweep.NON_TERMINAL_STATES)))
        for t in range(1, 10 * e2_sweep.H):
            _, _, terminated, truncated, info = env.step(int(rng.integers(5)))
            assert t <= e2_sweep.H
            if terminated:
                assert E2_TERMINAL[info["state"]]  # termination only at the goal
                n_terminated += 1
                break
            if truncated:
                assert not terminated
                lengths_truncated.append(t)
                break
    assert lengths_truncated and set(lengths_truncated) == {e2_sweep.H}
    assert n_terminated > 0


def test_terminal_at_the_goal_only():
    assert np.array_equal(e3_env().terminal_mask(), E2_TERMINAL)


def test_reward_matches_e2():
    """E2: goal_reward at the goal, E2's step reward (build_env's default,
    -1) everywhere else."""
    assert np.array_equal(e3_env().reward_vector(), E2_R)


# ---------------------------------------------------------------- discounting, budget, checkpoints


def test_gamma_matches_e2():
    assert E3_EXPERIMENT_CONFIG["episode"]["gamma"] == e2_sweep.GAMMA


@pytest.mark.xfail(reason="no planner / critic in E3 before Phase 5, so gamma is declared but not yet consumed", strict=True)
def test_gamma_is_used_by_the_e3_agent():
    assert "gamma" in e3_config("mini.yaml").get("agent", {})


def test_budget_and_checkpoint_grid_match_e2():
    assert E3_EXPERIMENT_CONFIG["budget"]["total_steps"] == e2_sweep.BUDGET
    assert list(E3_EXPERIMENT_CONFIG["budget"]["checkpoints"]) == list(e2_sweep.CHECKPOINTS)
    assert len(e2_sweep.CHECKPOINTS) == 16


@pytest.mark.xfail(reason="periodic checkpoints on E2's 16-point grid and --resume are Phase 4a (B5); run_mini saves none yet",
                   strict=True)
def test_runs_save_the_model_on_e2s_checkpoint_grid():
    assert hasattr(run_mini, "save_checkpoint")


# ---------------------------------------------------------------- random streams (spec 6)


def test_start_state_stream_is_e2s_spawned_stream():
    e2_rng = load_experiment_module("E2_online_4zones", "rng.py", "conformance_e2_rng")
    ours = run_mini.seed_streams(3, device=__import__("torch").device("cpu"))["start_state"]
    theirs = e2_rng.spawn_streams(3)["start_state"]
    assert np.array_equal(ours.integers(1 << 30, size=20), theirs.integers(1 << 30, size=20))


def test_collection_dynamics_and_theme_streams_are_independent():
    """Spec 6 (docs/e3/DECISIONS.md, 2026-09-25): every stream is a distinct
    child of SeedSequence(seed).spawn; none is seeded with the integer seed."""
    rng_streams = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "conformance_rng_streams")
    for seed in (0, 5):
        env = e3_env(seed=seed)
        actions = run_mini.seed_streams(seed, device=__import__("torch").device("cpu"))["collect"]
        dynamics = env._dynamics_rng
        themes = rng_streams.spawn_streams(seed)["theme"]
        states = [g.bit_generator.state["state"]["state"] for g in (actions, dynamics, themes)]
        assert len(set(states)) == 3
        assert all(g.bit_generator.state != np.random.default_rng(seed).bit_generator.state for g in (actions, dynamics, themes))


def test_first_draws_of_the_action_and_dynamics_streams_differ():
    for seed in (0, 1, 5):
        env = e3_env(seed=seed)
        actions = run_mini.seed_streams(seed, device=__import__("torch").device("cpu"))["collect"]
        assert not np.array_equal(actions.integers(1 << 30, size=16), env._dynamics_rng.integers(1 << 30, size=16))


def test_e3_streams_extend_e2s_spawned_streams():
    """The first children of E3's spawn are E2's own streams for the same seed."""
    e2_rng = load_experiment_module("E2_online_4zones", "rng.py", "conformance_e2_rng_prefix")
    rng_streams = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "conformance_rng_streams_prefix")
    assert rng_streams.STREAM_NAMES[: len(e2_rng.STREAM_NAMES)] == tuple(e2_rng.STREAM_NAMES)
    ours, theirs = rng_streams.spawn_streams(7), e2_rng.spawn_streams(7)
    for name in e2_rng.STREAM_NAMES:
        assert np.array_equal(ours[name].integers(1 << 30, size=8), theirs[name].integers(1 << 30, size=8)), name


def test_environment_dynamics_seed_is_drawn_as_in_e2():
    rng_streams = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "conformance_rng_streams_env")
    e2_rng = load_experiment_module("E2_online_4zones", "rng.py", "conformance_e2_rng_env")
    env = e3_env(seed=4)
    expected = np.random.default_rng(int(e2_rng.spawn_streams(4)["env_dynamics"].integers(2**31)))  # sweep.py's own draw
    assert np.array_equal(env._dynamics_rng.integers(1 << 30, size=8), expected.integers(1 << 30, size=8))


# ---------------------------------------------------------------- episode parameters read from the config


def test_episode_parameters_come_from_the_config_and_equal_e2s():
    import inspect

    episode = run_mini.load_episode_config()
    e2_step_reward = inspect.signature(e2_env.build_env).parameters["step_reward"].default
    assert episode["goal_reward"] == e2_sweep.GOAL_REWARD
    assert episode["step_reward"] == e2_step_reward
    assert episode["max_steps"] == e2_sweep.H
    assert episode["gamma"] == e2_sweep.GAMMA
    env = e3_env()
    assert env.max_steps == episode["max_steps"]
    R = env.reward_vector()
    assert set(R.tolist()) == {episode["goal_reward"], episode["step_reward"]}
