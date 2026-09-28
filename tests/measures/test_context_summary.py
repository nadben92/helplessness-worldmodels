from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures.context_summary import per_state_context_summary, zone_mean_with_se


def test_per_state_summary_by_hand():
    values = np.array([1.0, 3.0, 2.0, np.nan, 5.0])
    states = np.array([0, 0, 1, 1, 2])
    mc_se = np.array([0.1, 0.1, 0.2, 0.2, 0.3])
    out = per_state_context_summary(values, states, n_states=4, mc_se=mc_se)
    assert out["n"].tolist() == [2, 1, 1, 0]
    assert out["mean"][0] == 2.0 and out["mean"][1] == 2.0 and out["mean"][2] == 5.0 and np.isnan(out["mean"][3])
    assert out["sd"][0] == pytest.approx(np.sqrt(2.0)) and np.isnan(out["sd"][1])
    assert out["se"][0] == pytest.approx(1.0)  # sqrt(2) / sqrt(2)
    assert out["se"][1] == 0.2 and out["se"][2] == 0.3  # a single context: its own MC se
    assert np.isnan(out["se"][3])


def test_zone_mean_ignores_unmeasured_states():
    mean = np.array([1.0, np.nan, 3.0, 10.0])
    se = np.array([0.3, np.nan, 0.4, 1.0])
    out = zone_mean_with_se(mean, se, [0, 1, 2])
    assert out["mean"] == 2.0 and out["n_measured"] == 2 and out["n_states"] == 3
    assert out["se"] == pytest.approx(0.5 / 2)
    empty = zone_mean_with_se(mean, se, [1])
    assert np.isnan(empty["mean"]) and empty["n_measured"] == 0
