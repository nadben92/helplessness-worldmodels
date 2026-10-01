"""Analytical, numpy-only tests for measures.kernel_probe (spec section
9, phase-3 point 3a): a synthetic scenario where a factorised
combination produces "cross" draws that match no real template equally
well, classified ghosts, with the ghost rate matching the constructed
analytical value exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures.kernel_probe import kernel_from_matches, kernel_probe, match_templates


def _one_hot_templates(n_states: int) -> np.ndarray:
    return np.eye(n_states, dtype=np.float64)


def test_synthetic_cross_combination_is_classified_ghost_and_matches_analytic_rate():
    """4 canonical states (standard basis vectors e0..e3). Two "real"
    draws exactly match e0 and e1. Three "cross" draws are the EXACT
    midpoint of e0 and e1 (0.5, 0.5, 0, 0) -- equidistant from e0 and e1
    (margin == 0 exactly), and clearly closer to {e0, e1} than to {e2,
    e3}, so they are genuine "confusable between two real states" cases,
    not just noise. With ANY positive margin_threshold, these 3 draws
    are ghosts by construction. Analytic ghost rate: 3/6 = 0.5 exactly.
    """
    templates = _one_hot_templates(4)
    e0, e1 = templates[0], templates[1]
    cross = 0.5 * (e0 + e1)

    decoded = np.stack([e0, e0, e1, cross, cross, cross], axis=0)
    result = kernel_probe(decoded, templates, margin_threshold=1e-6)

    assert result["ghost_fraction"] == pytest.approx(0.5, abs=1e-12)
    assert result["n_ghosts"] == 3
    assert result["n_samples"] == 6
    assert list(result["is_ghost"]) == [False, False, False, True, True, True]

    # p_hat renormalised over the 3 non-ghost draws only: 2/3 e0, 1/3 e1
    assert result["p_hat"][0] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert result["p_hat"][1] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert result["p_hat"][2] == 0.0
    assert result["p_hat"][3] == 0.0
    assert result["p_hat"].sum() == pytest.approx(1.0, abs=1e-12)  # a proper distribution, ghost mass excluded


def test_margin_zero_means_ties_are_always_ghosts_regardless_of_threshold():
    templates = _one_hot_templates(3)
    cross = 0.5 * (templates[0] + templates[1])
    decoded = np.stack([cross], axis=0)

    for threshold in (1e-9, 1e-3, 0.5):
        result = kernel_probe(decoded, templates, margin_threshold=threshold)
        assert result["is_ghost"][0]


def test_clear_matches_are_never_ghosts_for_a_small_threshold():
    templates = _one_hot_templates(5)
    decoded = templates.copy()  # exact matches to every template
    result = kernel_probe(decoded, templates, margin_threshold=1e-6)
    assert not result["is_ghost"].any()
    assert result["ghost_fraction"] == 0.0
    assert list(result["nearest_state"]) == [0, 1, 2, 3, 4]
    assert np.allclose(result["p_hat"], np.full(5, 0.2))


def test_margin_threshold_controls_the_ghost_boundary_analytically():
    """Two templates 1.0 apart (squared L2) in a toy 1-D-ish setup; a
    probe point at a KNOWN distance from each lets the ghost/non-ghost
    boundary be predicted exactly from margin_threshold."""
    templates = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])  # e.g. states A, B, C
    probe_point = np.array([[0.4, 0.0]])  # dist^2 to A = 0.16, to B = 0.36, to C large
    margin = 0.36 - 0.16  # = 0.20 exactly

    below = kernel_probe(probe_point, templates, margin_threshold=margin + 0.01)
    above = kernel_probe(probe_point, templates, margin_threshold=margin - 0.01)
    assert below["is_ghost"][0] is np.True_ or bool(below["is_ghost"][0]) is True
    assert bool(above["is_ghost"][0]) is False
    assert bool(above["nearest_state"][0] == 0)  # matches A, the closer template


def test_kernel_from_matches_reports_nan_ghost_fraction_for_zero_samples():
    result = kernel_from_matches(np.array([], dtype=int), np.array([], dtype=bool), n_states=4)
    assert np.isnan(result["ghost_fraction"])
    assert result["n_samples"] == 0
    assert np.all(result["p_hat"] == 0.0)


def test_match_templates_shapes():
    templates = _one_hot_templates(6)
    decoded = np.random.default_rng(0).random((10, 6))
    matched = match_templates(decoded, templates, margin_threshold=0.01)
    assert matched["nearest_state"].shape == (10,)
    assert matched["is_ghost"].shape == (10,)
    assert matched["margin"].shape == (10,)
    assert (matched["margin"] >= 0).all()


def test_match_templates_distances_equal_the_direct_pairwise_computation():
    """The |a|^2 + |b|^2 - 2 a.b expansion must pick the same nearest
    template and the same margins as the direct pairwise difference."""
    rng = np.random.default_rng(1)
    templates = rng.random((20, 3, 12, 12))
    decoded = templates[rng.integers(0, 20, size=15)] + 0.05 * rng.standard_normal((15, 3, 12, 12))
    matched = match_templates(decoded, templates, margin_threshold=0.0)

    direct = ((decoded.reshape(15, 1, -1) - templates.reshape(1, 20, -1)) ** 2).sum(axis=-1)
    order = np.argsort(direct, axis=1)
    assert np.array_equal(matched["nearest_state"], order[:, 0])
    expected_margin = direct[np.arange(15), order[:, 1]] - direct[np.arange(15), order[:, 0]]
    assert np.allclose(matched["margin"], expected_margin, atol=1e-9)


# ---------------------------------------------------------------------------
# C-b's sampling floor (docs/e3/DECISIONS.md, 2026-09-29)
# ---------------------------------------------------------------------------


def test_expected_l1_sampling_floor_matches_simulation():
    from helplessness.measures.kernel_probe import expected_l1_sampling_floor

    p = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
    rng = np.random.default_rng(0)
    for K in (32, 200):
        sim = np.mean([np.abs(rng.multinomial(K, p) / K - p).sum() for _ in range(20_000)])
        assert expected_l1_sampling_floor(p, K) == pytest.approx(sim, rel=0.02)
    assert expected_l1_sampling_floor(np.array([1.0, 0, 0, 0, 0]), 32) == 0.0  # a deterministic kernel has no floor
    assert expected_l1_sampling_floor(np.stack([p, p]), 32).shape == (2,)


def test_draws_for_l1_floor_is_the_smallest_sufficient_k():
    from helplessness.measures.kernel_probe import draws_for_l1_floor, expected_l1_sampling_floor

    P = np.array([[0.9, 0.025, 0.025, 0.025, 0.025], [1.0, 0.0, 0.0, 0.0, 0.0]])
    K = draws_for_l1_floor(P, 0.02)
    assert K == 1358
    assert all(expected_l1_sampling_floor(P[0], k) <= 0.02 for k in range(K, 2 * K + 1))
    assert expected_l1_sampling_floor(P[0], K - 1) > 0.02


# ---------------------------------------------------------------------------
# projection on a support and off-support mass (docs/e3/DECISIONS.md, 2026-09-30 (5))
# ---------------------------------------------------------------------------


def test_match_from_distances_is_match_templates_and_projects_on_a_support():
    from helplessness.measures.kernel_probe import match_from_distances, template_distances

    rng = np.random.default_rng(0)
    templates = rng.random((6, 3, 4, 4))
    decoded = templates[[0, 3, 5, 5]] + 0.01 * rng.standard_normal((4, 3, 4, 4))
    dists = template_distances(decoded, templates)
    full = match_from_distances(dists, 1e-3)
    ref = match_templates(decoded, templates, 1e-3)
    for key in ("nearest_state", "is_ghost", "margin"):
        assert np.array_equal(full[key], ref[key])
    assert full["nearest_state"].tolist() == [0, 3, 5, 5]
    allowed = np.zeros((4, 6), dtype=bool)
    allowed[0, [0, 1]] = True   # the true match is allowed: unchanged
    allowed[1, [1, 2]] = True   # it is not: reassigned to the nearest allowed template
    allowed[2, [5]] = True      # a single allowed template: never a ghost
    proj = match_from_distances(dists, 1e-3, allowed=allowed)  # row 3: nothing allowed
    assert proj["nearest_state"][0] == 0 and proj["nearest_state"][1] in (1, 2) and proj["nearest_state"][2] == 5
    assert proj["is_ghost"].tolist() == [False, False, False, True]


def test_off_support_mass():
    from helplessness.measures.kernel_probe import off_support_mass

    p_hat = np.zeros((3, 2, 3))
    p_hat[0, 0] = [0.7, 0.2, 0.1]
    p_hat[0, 1] = [1.0, 0.0, 0.0]
    p_hat[1] = [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    p_hat[2] = np.nan
    support = np.array([[True, True, False], [False, True, False], [True, True, True]])
    out = off_support_mass(p_hat, support)
    assert out[0].tolist() == pytest.approx([0.1, 0.0]) and out[1].tolist() == pytest.approx([0.5, 1.0])
    assert np.isnan(out[2]).all()
