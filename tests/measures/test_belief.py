"""Tests for measures.belief.l1_error_per_pair/model_error: the per-pair
L1 primitive E1 aggregates by zone for its model-fidelity report, and the
(unchanged-behaviour) total-variation scalar that already used it inline."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures.belief import l1_error_per_pair, model_error

_T_hat = np.array([[[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
_T_true = np.array([[[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]], [[0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]])


def test_l1_error_per_pair_matches_hand_computation():
    expected_00 = abs(0.5 - 0.4) + abs(0.5 - 0.4) + abs(0.0 - 0.2)
    per_pair = l1_error_per_pair(_T_hat, _T_true)
    assert per_pair.shape == (2, 2)
    assert per_pair[0, 0] == pytest.approx(expected_00)
    assert per_pair[0, 1] == pytest.approx(0.0)  # identical rows


def test_model_error_is_half_the_l1_norm():
    visited = np.array([[True, True], [False, True]])
    tv_visited, tv_all = model_error(_T_hat, _T_true, visited)
    per_pair = l1_error_per_pair(_T_hat, _T_true)
    expected_tv = 0.5 * per_pair
    assert tv_all == pytest.approx(expected_tv.mean())
    assert tv_visited == pytest.approx(expected_tv[visited].mean())
