"""Tests for experiments/E2_online_4zones/matched_control.py::collect_matched."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

mc_mod = load_experiment_module("E2_online_4zones", "matched_control.py", "_e2_matched_control_for_mc_test")


def _toy_tensor():
    # 3 states, 2 actions, arbitrary but fixed transition rows
    P = np.zeros((3, 2, 3))
    P[0, 0] = [0.9, 0.05, 0.05]
    P[0, 1] = [0.1, 0.8, 0.1]
    P[1, 0] = [0.2, 0.2, 0.6]
    P[1, 1] = [0.0, 1.0, 0.0]
    P[2, 0] = [0.5, 0.5, 0.0]
    P[2, 1] = [0.3, 0.3, 0.4]
    return P


def test_collect_matched_respects_per_pair_target_counts():
    P = _toy_tensor()
    N_target = np.array([[100, 0], [50, 200], [0, 10]])
    rng = np.random.default_rng(0)
    counts = mc_mod.collect_matched(P, N_target, rng)
    assert counts.sum(axis=2).tolist() == N_target.tolist()


def test_collect_matched_zero_target_draws_nothing():
    P = _toy_tensor()
    N_target = np.zeros((3, 2), dtype=int)
    rng = np.random.default_rng(0)
    counts = mc_mod.collect_matched(P, N_target, rng)
    assert counts.sum() == 0


def test_collect_matched_deterministic_row_reproduces_exactly_at_any_count():
    """A deterministic transition row must land ALL draws on the single
    reachable successor, regardless of how many draws are requested."""
    P = _toy_tensor()
    P[1, 1] = [0.0, 1.0, 0.0]  # already deterministic
    N_target = np.array([[0, 0], [0, 777], [0, 0]])
    rng = np.random.default_rng(0)
    counts = mc_mod.collect_matched(P, N_target, rng)
    assert counts[1, 1].tolist() == [0, 777, 0]


def test_collect_matched_rejects_wrong_shape():
    P = _toy_tensor()
    bad_target = np.zeros((3, 3), dtype=int)  # wrong n_actions
    import pytest
    with pytest.raises(ValueError, match="shape"):
        mc_mod.collect_matched(P, bad_target, np.random.default_rng(0))
