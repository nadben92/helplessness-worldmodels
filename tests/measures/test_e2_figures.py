"""Tests for experiments/E2_online_4zones/figures.py's aggregation
primitives -- the ones every F1-F7 figure reuses."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

ad = load_experiment_module("E2_online_4zones", "analysis_data.py", "_e2_ad_for_figures_test")
fg = load_experiment_module("E2_online_4zones", "figures.py", "_e2_figures_for_figures_test")


def _row(**kwargs):
    row = {
        "condition": "online", "alpha": 1.0, "epsilon": 0.1, "seed": 0, "assignment": 0,
        "checkpoint": 1000, "zone": "corridor", "zone_type": "ControllableDeterministic",
        "region": "interior", "state": 1, "I_hat": 0.0, "I_star": 0.0, "b": 0.0, "delta": 0.0,
        "v_pi_true": 0.0, "v_pi_believed": 0.0,
    }
    row.update(kwargs)
    return tuple(row[f] for f in ad.PER_STATE_DTYPE.names)


def test_per_seed_values_by_checkpoint_averages_within_seed():
    rows = [
        _row(seed=0, checkpoint=1000, b=1.0),
        _row(seed=0, checkpoint=1000, b=3.0),  # same seed, same checkpoint -- averaged
        _row(seed=1, checkpoint=1000, b=5.0),
    ]
    arr = np.array(rows, dtype=ad.PER_STATE_DTYPE)
    mask = np.ones(len(arr), dtype=bool)
    result = fg.per_seed_values_by_checkpoint(arr, mask, "b", checkpoints=(1000,))
    assert sorted(result[1000].tolist()) == [2.0, 5.0]  # seed 0 averaged to 2.0, seed 1 stays 5.0


def test_zone_condition_mask_selects_correctly():
    rows = [
        _row(zone="corridor", condition="online", alpha=1.0, epsilon=0.1),
        _row(zone="corridor", condition="matched", alpha=1.0, epsilon=0.1),
        _row(zone="controllable_noisy", condition="online", alpha=1.0, epsilon=0.1),
        _row(zone="corridor", condition="online", alpha=0.5, epsilon=0.1),
    ]
    arr = np.array(rows, dtype=ad.PER_STATE_DTYPE)
    mask = fg.zone_condition_mask(arr, "corridor", "online", 1.0, 0.1)
    assert mask.tolist() == [True, False, False, False]


def test_paired_diff_by_checkpoint_computes_intra_seed_difference():
    rows = [
        _row(seed=0, checkpoint=1000, condition="online", zone="corridor", b=5.0),
        _row(seed=0, checkpoint=1000, condition="matched", zone="corridor", b=2.0),
        _row(seed=1, checkpoint=1000, condition="online", zone="corridor", b=10.0),
        _row(seed=1, checkpoint=1000, condition="matched", zone="corridor", b=1.0),
    ]
    arr = np.array(rows, dtype=ad.PER_STATE_DTYPE)
    result = fg.paired_diff_by_checkpoint(arr, "corridor", 1.0, 0.1, "b", checkpoints=(1000,))
    assert sorted(result[1000].tolist()) == [3.0, 9.0]  # 5-2=3, 10-1=9
