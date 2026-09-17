"""Tests for experiments/E2_online_4zones/trajectory.py::reconstruct_N_from_trajectory."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

traj_mod = load_experiment_module("E2_online_4zones", "trajectory.py", "_e2_trajectory_for_trajectory_test")


def test_reconstruct_n_matches_manual_counting():
    traj_t = np.array([1, 2, 3, 4, 5])
    traj_s = np.array([0, 0, 1, 0, 2])
    traj_a = np.array([0, 1, 0, 0, 1])
    traj_s_next = np.array([1, 2, 0, 1, 0])

    N = traj_mod.reconstruct_N_from_trajectory(traj_t, traj_s, traj_a, traj_s_next, up_to_t=5, n_states=3, n_actions=2)
    expected = np.zeros((3, 2, 3), dtype=np.int64)
    expected[0, 0, 1] = 2  # t=1 and t=4
    expected[0, 1, 2] = 1  # t=2
    expected[1, 0, 0] = 1  # t=3
    expected[2, 1, 0] = 1  # t=5
    assert np.array_equal(N, expected)


def test_reconstruct_n_respects_up_to_t():
    traj_t = np.array([1, 2, 3])
    traj_s = np.array([0, 0, 0])
    traj_a = np.array([0, 0, 0])
    traj_s_next = np.array([1, 1, 1])

    N_at_2 = traj_mod.reconstruct_N_from_trajectory(traj_t, traj_s, traj_a, traj_s_next, up_to_t=2, n_states=2, n_actions=1)
    assert N_at_2[0, 0, 1] == 2
    N_at_3 = traj_mod.reconstruct_N_from_trajectory(traj_t, traj_s, traj_a, traj_s_next, up_to_t=3, n_states=2, n_actions=1)
    assert N_at_3[0, 0, 1] == 3


def test_reconstruct_n_sum_equals_number_of_transitions_up_to_t():
    rng = np.random.default_rng(0)
    n = 500
    traj_t = np.arange(1, n + 1)
    traj_s = rng.integers(0, 10, size=n)
    traj_a = rng.integers(0, 4, size=n)
    traj_s_next = rng.integers(0, 10, size=n)

    for up_to_t in (10, 100, 500):
        N = traj_mod.reconstruct_N_from_trajectory(traj_t, traj_s, traj_a, traj_s_next, up_to_t=up_to_t, n_states=10, n_actions=4)
        assert int(N.sum()) == up_to_t


def test_reconstruct_n_is_order_independent():
    """The trajectory doesn't need to be sorted by t -- a mask, not a
    slice, selects which rows count."""
    traj_t = np.array([3, 1, 2])
    traj_s = np.array([0, 1, 0])
    traj_a = np.array([0, 0, 1])
    traj_s_next = np.array([1, 0, 1])
    N = traj_mod.reconstruct_N_from_trajectory(traj_t, traj_s, traj_a, traj_s_next, up_to_t=2, n_states=2, n_actions=2)
    # only t=1 (s=1,a=0,s'=0) and t=2 (s=0,a=1,s'=1) count
    expected = np.zeros((2, 2, 2), dtype=np.int64)
    expected[1, 0, 0] = 1
    expected[0, 1, 1] = 1
    assert np.array_equal(N, expected)
