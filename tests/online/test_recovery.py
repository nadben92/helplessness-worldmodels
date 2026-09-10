"""Tests for online.recovery: recovery_time, the periodic out-of-loop
evaluation/mechanistic-metrics tracker, and a small end-to-end run.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.config import RecoveryConfig
from helplessness.envs.legacy.gridworld import GridWorld
from online.agent import OnlineTabularAgent
from online.loop import train_online_with_burnin
from online.recovery import RecoveryTracker, run_condition, recovery_time


def test_recovery_time_finds_first_threshold_crossing():
    rows = [
        {"t": 200, "return_mean": 0.0},
        {"t": 400, "return_mean": 50.0},
        {"t": 600, "return_mean": 95.0},
        {"t": 800, "return_mean": 92.0},
    ]
    assert recovery_time(rows, oracle_return=100.0, threshold=0.95) == 600


def test_recovery_time_none_when_never_reached():
    rows = [{"t": 200, "return_mean": 0.0}, {"t": 400, "return_mean": 10.0}]
    assert recovery_time(rows, oracle_return=100.0, threshold=0.95) is None


def test_recovery_time_is_order_independent():
    rows = [
        {"t": 600, "return_mean": 95.0},
        {"t": 200, "return_mean": 0.0},
        {"t": 400, "return_mean": 50.0},
    ]
    assert recovery_time(rows, oracle_return=100.0, threshold=0.95) == 600


def test_tracker_zero_sample_pairs_decreases_as_window_fills():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    agent.plan(np.random.default_rng(0))
    cfg = RecoveryConfig(mechanistic_window=100, eval_every=10, n_eval_episodes=5)
    non_terminal = ~env.terminal_mask()
    pairs = [(s, a) for s in range(env.n_states) if non_terminal[s] for a in range(env.n_actions)]
    tracker = RecoveryTracker(agent, cfg, eval_seed=123, non_terminal_pairs=pairs)

    for t in range(1, 11):
        tracker(t, s=0, a=0, r=-1.0, s_next=1, terminated=False, truncated=False)
    first = tracker.rows[0]["zero_sample_pairs"]

    for t in range(11, 31):
        tracker(t, s=1, a=2, r=-1.0, s_next=2, terminated=False, truncated=False)
    later = tracker.rows[-1]["zero_sample_pairs"]

    assert later <= first  # more distinct (s,a) sampled -> no more unsampled pairs
    assert first == len(pairs) - 1  # only (0,0) sampled by t=10


def test_tracker_state_entropy_is_zero_for_single_state_window():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    agent.plan(np.random.default_rng(0))
    cfg = RecoveryConfig(mechanistic_window=50, eval_every=5, n_eval_episodes=5)
    non_terminal = ~env.terminal_mask()
    pairs = [(s, a) for s in range(env.n_states) if non_terminal[s] for a in range(env.n_actions)]
    tracker = RecoveryTracker(agent, cfg, eval_seed=123, non_terminal_pairs=pairs)
    for t in range(1, 6):
        tracker(t, s=0, a=0, r=-1.0, s_next=0, terminated=False, truncated=False)
    assert tracker.rows[-1]["state_entropy"] == pytest.approx(0.0)


def test_run_condition_end_to_end(tmp_path):
    """Small-scale smoke test of the full snapshot -> recovery-condition
    path: train a tiny snapshot, save it, reload it via run_condition, and
    check the curve has the expected shape and monotone t."""
    p_train = 0.0
    train_env = GridWorld(p=p_train, seed=0, n_actions=5, step_reward=-1.0, goal_reward=100.0, trap_reward=-100.0)
    train_env.seed_dynamics(0)
    agent = OnlineTabularAgent(train_env, alpha=0.1, gamma=0.95)
    train_online_with_burnin(
        train_env, agent, 5_000, 0.1, 500, burn_in_steps=1_000,
        explore_rng=np.random.default_rng(1), tiebreak_rng=np.random.default_rng(2),
        burn_in_rng=np.random.default_rng(3), learning=True,
    )
    snapshot_file = str(tmp_path / "snap.npz")
    agent.save(snapshot_file)

    cfg = RecoveryConfig(
        n_phase2_steps=1_000, replan_every=100, eval_every=100, n_eval_episodes=20,
        mechanistic_window=200, gamma=0.95, n_actions=5, step_cost=1.0, trap_cost=100.0, goal_reward=100.0,
    )
    rows = run_condition(cfg, snapshot_file, "A", seed=0)

    assert len(rows) == 1_000 // cfg.eval_every
    ts = [row["t"] for row in rows]
    assert ts == sorted(ts)
    for row in rows:
        assert 0 <= row["zero_sample_pairs"]
        assert row["state_entropy"] >= 0.0


def test_run_condition_rescale_true_rescales_before_training(tmp_path):
    train_env = GridWorld(p=0.0, seed=0, n_actions=5, step_reward=-1.0, goal_reward=100.0, trap_reward=-100.0)
    train_env.seed_dynamics(0)
    agent = OnlineTabularAgent(train_env, alpha=0.1, gamma=0.95)
    train_online_with_burnin(
        train_env, agent, 5_000, 0.1, 500, burn_in_steps=1_000,
        explore_rng=np.random.default_rng(1), tiebreak_rng=np.random.default_rng(2),
        burn_in_rng=np.random.default_rng(3), learning=True,
    )
    snapshot_file = str(tmp_path / "snap.npz")
    agent.save(snapshot_file)
    n_before = agent.N.sum()

    # n_cap close to n_before (mild compression): this agent trains for only
    # 5000 transitions (test speed), far below the real 50k Phase-1 budget
    # rescale_counts's default tolerance was calibrated against -- see its
    # docstring. A steep compression on such a thin belief would trip that
    # tolerance for reasons unrelated to what this test checks (which
    # snapshot the rescale is applied to, not how tight the tolerance is).
    cfg = RecoveryConfig(n_phase2_steps=50, replan_every=25, eval_every=25, n_eval_episodes=5, n_cap=3_000.0)
    reloaded = OnlineTabularAgent.load(snapshot_file)
    assert reloaded.N.sum() == pytest.approx(n_before)
    run_condition(cfg, snapshot_file, "C", seed=0)  # condition C rescales

    # the snapshot on disk must be untouched by a condition's rescale (each
    # condition loads its own fresh copy -- see run_condition).
    reloaded_again = OnlineTabularAgent.load(snapshot_file)
    assert reloaded_again.N.sum() == pytest.approx(n_before)
