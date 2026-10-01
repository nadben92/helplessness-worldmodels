"""hybrid_codes_check.py (docs/e3/DECISIONS.md, 2026-09-30 (5)): the
classification of a prior draw's code against the posterior's codes, on
hand-made codes."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

hc = load_experiment_module("E3_worldmodel_4zones", "hybrid_codes_check.py", "test_hybrid_codes")


def test_codes_are_classified_against_the_posterior_s_codes():
    C = 4
    # the posterior produced two codes: state 0 -> (0, 0, 0), state 1 -> (1, 1, 1); plus (2, 3, 0) elsewhere
    observed = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 0]])
    keys = np.unique(hc.code_keys(observed, C))
    pairs = hc.pair_table(observed, C)
    successors = np.zeros((3, C), dtype=bool)  # classes produced for the reachable successors (states 0 and 1)
    successors[:, [0, 1]] = True
    draws = np.array([[0, 0, 0],    # observed
                      [0, 1, 0],    # a mix of the two successors' codes: novel, a novel pair, hybrid
                      [2, 3, 0],    # observed (another state's code): not novel
                      [3, 3, 3]])   # novel, but not made of the successors' classes: not hybrid
    flags = hc.classify_codes(draws, keys, pairs, np.broadcast_to(successors, (4, 3, C)), C)
    assert flags["novel"].tolist() == [False, True, False, True]
    assert flags["novel_pair"].tolist() == [False, True, False, True]
    assert flags["hybrid"].tolist() == [False, True, False, False]


def test_a_novel_code_can_have_every_pair_observed():
    C = 2
    observed = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])  # every pair of classes occurs; (0, 0, 0) does not
    flags = hc.classify_codes(np.array([[0, 0, 0]]), np.unique(hc.code_keys(observed, C)), hc.pair_table(observed, C),
                              np.ones((1, 3, C), dtype=bool), C)
    assert flags["novel"].tolist() == [True] and flags["novel_pair"].tolist() == [False] and flags["hybrid"].tolist() == [True]


def test_summary_shares():
    off = np.array([True, True, False, False, False])
    out = hc.summarise(off, {"novel": np.array([True, False, True, False, False])})
    assert out["off_support_share"] == 0.4 and out["n_off_support"] == 2
    assert out["novel_share_of_off_support"] == 0.5 and out["novel_share_of_draws"] == 0.4
    assert out["off_support_share_among_novel"] == 0.5 and np.isclose(out["off_support_share_among_not_novel"], 1 / 3)


def test_verdict_is_the_ratio_of_the_two_off_support_masses():
    """docs/e3/DECISIONS.md, 2026-09-30 (6): confirmed if the posterior codes' off-support mass is at most half the factorised one."""
    assert hc.verdict(0.10, 0.05)["verdict"] == "CONFIRMED" and hc.verdict(0.10, 0.05)["ratio"] == 0.5
    assert hc.verdict(0.10, 0.004)["verdict"] == "CONFIRMED"
    assert hc.verdict(0.10, 0.051)["verdict"] == "NOT CONFIRMED" and hc.verdict(0.10, 0.20)["verdict"] == "NOT CONFIRMED"
    assert hc.verdict(0.0, 0.0)["verdict"].startswith("NOT EVALUABLE") and hc.verdict(float("nan"), 0.1)["verdict"] == "NOT EVALUABLE"


def test_masses_are_means_over_pairs():
    off = np.array([True, True, True, True, False, False])
    s = np.array([0, 0, 0, 0, 1, 1])
    a = np.array([0, 0, 0, 0, 2, 3])
    mass, n_pairs = hc.mean_over_pairs(off, s, a, n_actions=5)  # pair (0, 0): 1.0; (1, 2): 0; (1, 3): 0
    assert n_pairs == 3 and np.isclose(mass, 1 / 3)
    assert np.isnan(hc.mean_over_pairs(off[:0], s[:0], a[:0], 5)[0])
