"""Tests for helplessness.agents.online.recovery: the environment-agnostic
pieces of the Phase-2 recovery protocol (RecoveryTracker, recovery_time).
GridWorld-specific orchestration (run_condition, run_all, the CLI) lives in
experiments/E0_offline_5x5/05_online_prototype/recovery.py instead, and is
exercised by the real recovery run already committed as
results/online/recovery_summary.csv rather than duplicated here.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.agents.online.recovery import RecoveryTracker, recovery_time
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.agents.online.agent import OnlineTabularAgent


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


def _tracker(env, agent, **overrides):
    non_terminal = ~env.terminal_mask()
    pairs = [(s, a) for s in range(env.n_states) if non_terminal[s] for a in range(env.n_actions)]
    kwargs = dict(
        eval_env_factory=lambda: GridWorld(p=0.0, seed=123, n_actions=5),
        eval_every=10, n_eval_episodes=5, eval_seed=123,
        non_terminal_pairs=pairs, mechanistic_window=100,
    )
    kwargs.update(overrides)
    return RecoveryTracker(agent, **kwargs)


def test_tracker_zero_sample_pairs_decreases_as_window_fills():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    agent.plan(np.random.default_rng(0))
    tracker = _tracker(env, agent)

    for t in range(1, 11):
        tracker(t, s=0, a=0, r=-1.0, s_next=1, terminated=False, truncated=False)
    first = tracker.rows[0]["zero_sample_pairs"]

    for t in range(11, 31):
        tracker(t, s=1, a=2, r=-1.0, s_next=2, terminated=False, truncated=False)
    later = tracker.rows[-1]["zero_sample_pairs"]

    assert later <= first  # more distinct (s,a) sampled -> no more unsampled pairs
    assert first == len(tracker.non_terminal_pairs) - 1  # only (0,0) sampled by t=10


def test_tracker_state_entropy_is_zero_for_single_state_window():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    agent.plan(np.random.default_rng(0))
    tracker = _tracker(env, agent, eval_every=5)
    for t in range(1, 6):
        tracker(t, s=0, a=0, r=-1.0, s_next=0, terminated=False, truncated=False)
    assert tracker.rows[-1]["state_entropy"] == pytest.approx(0.0)


def test_tracker_uses_eval_env_factory_not_training_env():
    """eval_env_factory is called for evaluation, never the training env
    passed to train_online -- out-of-loop by construction."""
    calls = []

    def factory():
        calls.append(1)
        return GridWorld(p=0.0, seed=123, n_actions=5)

    env = GridWorld(p=0.0, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1, gamma=0.95)
    agent.plan(np.random.default_rng(0))
    tracker = _tracker(env, agent, eval_env_factory=factory, eval_every=1)

    for t in range(1, 4):
        tracker(t, s=0, a=0, r=-1.0, s_next=0, terminated=False, truncated=False)
    assert len(calls) == 3
