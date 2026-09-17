"""Tests for experiments/E2_online_4zones/pilot_report.py's analysis
functions, on synthetic data (fast -- no dependency on an actual pilot
run having completed)."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

report_mod = load_experiment_module("E2_online_4zones", "pilot_report.py", "_e2_pilot_report_for_report_test")

N_STATES = report_mod.N_STATES
N_ACTIONS = report_mod.N_ACTIONS


def test_state_zone_covers_every_state_exactly_once():
    zones = report_mod.STATE_ZONE
    assert len(zones) == N_STATES
    assert all(z is not None for z in zones)
    assert set(zones) == set(report_mod.ZONE_CATEGORIES)


def test_max_min_per_state_by_zone_on_a_uniform_toy_array():
    N_pair = np.full((N_STATES, N_ACTIONS), 5.0)
    result = report_mod.max_min_per_state_by_zone(N_pair)
    for category, stats in result.items():
        assert stats["max_action_mean"] == 5.0
        assert stats["min_action_mean"] == 5.0


def test_max_min_per_state_by_zone_distinguishes_max_and_min_action():
    N_pair = np.zeros((N_STATES, N_ACTIONS))
    N_pair[:, 0] = 100.0  # action 0 always the most-visited
    N_pair[:, 1:] = 3.0
    result = report_mod.max_min_per_state_by_zone(N_pair)
    for category, stats in result.items():
        assert stats["max_action_mean"] == 100.0
        assert stats["min_action_mean"] == 3.0


def test_fraction_below_thresholds_basic():
    N_pair = np.zeros((N_STATES, N_ACTIONS))
    N_pair[0, 0] = 0  # below all thresholds
    N_pair[0, 1] = 5  # below 10, 100
    N_pair[0, 2] = 50  # below 100 only
    N_pair[0, 3] = 500  # below none
    N_pair[0, 4] = 0
    result = report_mod.fraction_below_thresholds(N_pair, thresholds=(1, 10, 100))
    category = report_mod.STATE_ZONE[0]
    assert result[category][1] > 0  # some pairs have N < 1 (the two zeros, at least state 0's)


def test_corridor_min_reaches_checkpoint_finds_the_right_step():
    idx = np.flatnonzero(report_mod.STATE_ZONE == "corridor")
    raw = {}
    checkpoint_steps = [10, 100, 1000]
    for t in checkpoint_steps:
        N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
        # corridor min count grows with t; other zones irrelevant here
        fill = {10: 5, 100: 99, 1000: 150}[t]
        N[idx, :, 0] = fill
        raw[f"N_checkpoint_{t}"] = N
    result = report_mod.corridor_min_reaches_checkpoint(raw, checkpoint_steps, target=100.0)
    assert result == 1000


def test_corridor_min_reaches_checkpoint_none_if_never_reached():
    idx = np.flatnonzero(report_mod.STATE_ZONE == "corridor")
    raw = {}
    checkpoint_steps = [10, 100]
    for t in checkpoint_steps:
        N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
        N[idx, :, 0] = 5
        raw[f"N_checkpoint_{t}"] = N
    result = report_mod.corridor_min_reaches_checkpoint(raw, checkpoint_steps, target=100.0)
    assert result is None


def test_episode_success_rate_over_time_buckets_and_rates():
    episode_t = np.array([1, 2, 3, 500, 501, 999])
    episode_success = np.array([True, False, True, True, True, False])
    buckets = report_mod.episode_success_rate_over_time(episode_t, episode_success, n_buckets=2)
    assert len(buckets) == 2
    assert sum(b["n_episodes"] for b in buckets) == len(episode_t)


def test_value_iteration_cost_basic_stats():
    n_iters = np.array([10, 20, 30, 40])
    stats = report_mod.value_iteration_cost(n_iters, elapsed_seconds=123.4)
    assert stats["n_replans"] == 4
    assert stats["mean_n_iter"] == 25.0
    assert stats["max_n_iter"] == 40
    assert stats["min_n_iter"] == 10
    assert stats["elapsed_seconds"] == 123.4


def test_value_iteration_cost_handles_empty():
    stats = report_mod.value_iteration_cost(np.array([], dtype=np.int64), elapsed_seconds=0.0)
    assert stats["n_replans"] == 0
    assert stats["max_n_iter"] == 0
