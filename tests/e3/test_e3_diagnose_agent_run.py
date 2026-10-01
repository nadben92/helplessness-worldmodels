"""The statistics of diagnose_agent_run.py (docs/e3/DECISIONS.md, 2026-10-01 (11)): AUC with ties,
Wilson interval, exact two-sided binomial test, episode split of a closed-loop replay."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

diag = load_experiment_module("E3_worldmodel_4zones", "diagnose_agent_run.py", "test_diagnose_agent_run")


def test_auc():
    assert diag.auc(np.array([0.9, 0.8]), np.array([0.1, 0.2, 0.3])) == 1.0
    assert diag.auc(np.array([0.1]), np.array([0.9])) == 0.0
    assert diag.auc(np.array([0.5]), np.array([0.5])) == 0.5  # a tie counts half
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 400), rng.normal(0, 1, 600)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert diag.auc(pos, neg) == pytest.approx(brute)


def test_wilson_and_binomial():
    lo, hi = diag.wilson(16, 496)
    assert lo < 16 / 496 < hi and lo == pytest.approx(0.0200, abs=2e-4) and hi == pytest.approx(0.0518, abs=2e-4)
    assert diag.binom_two_sided(5, 10, 0.5) == pytest.approx(1.0)
    assert diag.binom_two_sided(0, 10, 0.5) == pytest.approx(2 * 0.5 ** 10)


def test_episode_split_of_a_closed_loop_replay():
    # two episodes: a goal reached after 2 steps (3 observations), then a running one
    arr = {"s": np.array([5, 6, 0, 7, 8]), "a": np.array([1, 2, -1, 3, 4]), "r": np.array([0, -1, 50, 0, -1.0]),
           "is_first": np.array([1, 0, 0, 1, 0], bool), "terminal": np.array([0, 0, 1, 0, 0], bool),
           "truncated": np.zeros(5, bool)}
    eps = diag.episodes_from_replay(arr)
    assert [list(e["s"]) for e in eps] == [[5, 6, 0], [7, 8]] and [e["goal"] for e in eps] == [True, False]
    assert list(eps[0]["env_step"]) == [0, 1, 2] and list(eps[1]["env_step"]) == [2, 3]
