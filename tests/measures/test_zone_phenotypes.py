"""E6's zone phenotypes (helplessness.measures.zone_phenotypes) on hand-built episodes."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures import zone_phenotypes as zp

# a 1-D toy: states 0..5, the zone = {2, 3, 4}, the goal at 0 (distance = the index)
ZONE = np.array([False, False, True, True, True, False])
DIST = np.arange(6)


def episodes(paths, actions=None, H=6):
    n = len(paths)
    states = np.full((n, H + 1), -1)
    acts = np.full((n, H), -1)
    for i, p in enumerate(paths):
        states[i, : len(p)] = p
        acts[i, : len(p) - 1] = 0 if actions is None else actions[i]
    return states, acts, np.array([len(p) - 1 for p in paths])


def test_first_exit_and_censoring():
    s, a, L = episodes([[3, 2, 1, 0], [3, 4, 3, 4, 3, 4, 3], [4, 5]])
    ex = zp.first_exit(s, L, ZONE)
    assert ex["time"].tolist() == [2, 6, 1] and ex["censored"].tolist() == [False, True, False]
    with pytest.raises(ValueError):
        zp.first_exit(*episodes([[1, 0]])[::2], ZONE)


def test_kaplan_meier_median_respects_censoring():
    # times 1, 2, 3 (events) and 4 censored: S = 3/4, 2/4 -> median 2
    km = zp.kaplan_meier(np.array([1, 2, 3, 4]), np.array([False, False, False, True]))
    assert km["survival"].tolist() == [0.75, 0.5, 0.25] and km["median"] == 2.0
    # more than half censored before any event that brings S to 0.5: no median
    assert np.isnan(zp.kaplan_meier(np.array([1, 5, 5, 5]), np.array([False, True, True, True]))["median"])


def test_directed_share_and_by_index():
    s, a, L = episodes([[4, 3, 2, 1], [4, 4, 3, 2, 1]])
    ds = zp.directed_share(s, L, ZONE, DIST)
    assert ds["n_zone_steps"].tolist() == [3, 4] and ds["n_directed"].tolist() == [3, 3]
    by = zp.directed_by_index(s, L, ZONE, DIST, 5)
    assert by["n"].tolist() == [2, 2, 2, 1, 0]
    assert by["directed"].tolist() == [1, 2, 2, 1, 0]
    assert by["share"][:4].tolist() == [0.5, 1.0, 1.0, 1.0] and np.isnan(by["share"][4])


def test_recovery_threshold_and_latency():
    theta = zp.recovery_threshold(0.1)
    assert theta == pytest.approx(0.5 * (0.4 + 0.94))
    assert zp.recovery_latency(np.array([0.9, 0.9]), np.array([30, 30]), theta, 20)["latency"] == 1.0
    r = zp.recovery_latency(np.array([0.4, 0.5, 0.8]), np.array([30, 30, 30]), 0.65, 20)
    assert r["latency"] == pytest.approx(2 + 0.5) and not r["censored"]
    r = zp.recovery_latency(np.array([0.4, 0.5, 0.9]), np.array([30, 30, 10]), 0.65, 20)  # the third index has too few episodes
    assert r["censored"] and np.isnan(r["latency"]) and r["max_index_read"] == 2


def test_stay_revisit_zone_share():
    s, a, L = episodes([[3, 3, 4, 3, 2, 1]], actions=[[4, 0, 0, 0, 0]])
    assert zp.stay_counts(s, a, L, ZONE).tolist() == [1]
    # transitions from the zone: 3->3 (revisit), 3->4, 4->3 (revisit), 3->2, 2->1
    assert zp.revisit_counts(s, L, ZONE, 4).tolist() == [2]
    assert zp.revisit_counts(s, L, ZONE, 1).tolist() == [1]  # only s_t itself: the stay
    assert zp.zone_share(s, L, ZONE).tolist() == [1.0]


def test_summarize_and_bootstrap_are_consistent():
    rng = np.random.default_rng(0)
    s, a, L = episodes([[3, 2, 1, 0], [4, 3, 2, 1], [3, 4, 5], [2, 1, 0]] * 5)
    out = zp.summarize(s, a, L, ZONE, DIST, 0.1, n_min=2)
    assert out["n_episodes"] == 20 and 0 <= out["directed_share"] <= 1
    ci = zp.bootstrap(s, a, L, ZONE, DIST, 0.1, rng, n_boot=20, strata=s[:, 0], n_min=2)
    lo, hi = ci["directed_share"]
    assert lo <= out["directed_share"] <= hi
