"""E6's monotone curves (helplessness.measures.monotone_fit)."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures import monotone_fit as mf


def test_pava_pools_violators_and_picks_the_direction():
    fit = mf.isotonic(np.array([0, 1, 2, 3]), np.array([1.0, 3.0, 2.0, 4.0]), increasing=True)
    assert fit["fitted"].tolist() == [1.0, 2.5, 2.5, 4.0]
    dec = mf.isotonic(np.array([0, 1, 2, 3]), np.array([4.0, 2.0, 3.0, 1.0]))
    assert not dec["increasing"] and dec["fitted"].tolist() == [4.0, 2.5, 2.5, 1.0]
    ties = mf.isotonic(np.array([0, 0, 1]), np.array([1.0, 3.0, 5.0]))
    assert ties["x"].tolist() == [0, 1] and ties["fitted"].tolist() == [2.0, 5.0]


def test_smooth_is_monotone_and_inverts():
    x = np.linspace(-2.3, 0, 12)
    y = 0.4 + 0.5 / (1 + np.exp(-4 * (x + 1)))
    fit = mf.isotonic(x, y)
    grid = np.arange(-2.3, 0.001, 0.01)
    c = mf.smooth(fit, grid)
    assert np.all(np.diff(c) >= -1e-12)
    inv = mf.invert(fit, y[3:9], grid)
    assert np.max(np.abs(inv["b_hat"] - x[3:9])) < 0.02 and not inv["outside"].any()


def test_held_out_on_a_common_curve_predicts_b():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.3, 0, 40)
    y = 0.4 + 0.25 * (x + 2.3)
    groups = np.repeat(["a", "b", "c", "d"], 10)
    grid = np.arange(-2.4, 0.1, 0.005)
    for h in mf.held_out(x, y, groups, grid):
        inside = ~h["outside_b_range"]
        assert np.max(np.abs(h["residual"][inside])) < 0.02
        assert np.max(np.abs(h["b_error"][inside])) < 0.08


def test_cluster_bootstrap_band_and_strict_range():
    rng = np.random.default_rng(1)
    x = np.tile(np.linspace(-2, 0, 8), 5)
    clusters = np.repeat(np.arange(5), 8)
    y = 0.4 + 0.25 * (x + 2) + rng.normal(0, 0.01, x.size)
    grid = np.arange(-2, 0.001, 0.01)
    s = mf.curve_summary(x, y, clusters, grid, rng, 200, 0.25, 0.02)
    assert s["increasing"] and s["spearman"] > 0.95
    ok = np.isfinite(s["band_lo"])
    assert np.all(s["band_lo"][ok] <= s["band_hi"][ok] + 1e-12)
    lo, hi = s["strict"]["invertible_range"]
    assert lo == pytest.approx(-2, abs=0.05) and hi == pytest.approx(0, abs=0.05)
    flat = mf.curve_summary(x, np.full_like(x, 0.5) + rng.normal(0, 0.01, x.size), clusters, grid, rng, 200, 0.25, 0.5)
    assert flat["strict"]["invertible_range"] == (None, None)
