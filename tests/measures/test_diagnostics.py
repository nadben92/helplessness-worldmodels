from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures.diagnostics import alpha_app, jensen_shannon_categorical, jensen_shannon_product


def test_alpha_app_zero_means_position_driven():
    assert alpha_app(I_z_swap=1.0, I_L=1.0, I_Lprime=3.0, contrast_threshold=0.1) == pytest.approx(0.0)


def test_alpha_app_one_means_appearance_driven():
    assert alpha_app(I_z_swap=3.0, I_L=1.0, I_Lprime=3.0, contrast_threshold=0.1) == pytest.approx(1.0)


def test_alpha_app_intermediate_value():
    assert alpha_app(I_z_swap=2.0, I_L=1.0, I_Lprime=3.0, contrast_threshold=0.1) == pytest.approx(0.5)


def test_alpha_app_undefined_without_contrast():
    assert alpha_app(I_z_swap=1.5, I_L=1.0, I_Lprime=1.05, contrast_threshold=0.1) is None


def test_alpha_app_defined_right_at_the_threshold_boundary_is_still_none_when_equal():
    # 0.5 and 1.0 are exactly representable in binary floating point, so
    # contrast == threshold holds exactly here (no float-rounding noise).
    assert alpha_app(I_z_swap=1.5, I_L=1.0, I_Lprime=1.5, contrast_threshold=0.5) is None


def test_jensen_shannon_categorical_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert jensen_shannon_categorical(p, p) == pytest.approx(0.0, abs=1e-9)


def test_jensen_shannon_categorical_is_symmetric():
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.1, 0.4, 0.5])
    assert jensen_shannon_categorical(p, q) == pytest.approx(jensen_shannon_categorical(q, p))


def test_jensen_shannon_categorical_disjoint_deltas_equals_one_bit():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jensen_shannon_categorical(p, q) == pytest.approx(1.0, abs=1e-6)


def test_jensen_shannon_categorical_is_nonnegative_and_bounded_by_one_bit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        js = jensen_shannon_categorical(p, q)
        assert -1e-9 <= js <= 1.0 + 1e-9


def test_jensen_shannon_product_sums_per_variable_divergences():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    # var 0: disjoint deltas -> 1 bit; var 1: identical -> 0 bit
    assert jensen_shannon_product(p, q) == pytest.approx(1.0, abs=1e-6)


def test_jensen_shannon_product_zero_for_identical_product_distributions():
    p = np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
    assert jensen_shannon_product(p, p) == pytest.approx(0.0, abs=1e-9)


def test_mean_pairwise_jensen_shannon_is_zero_when_every_item_is_identical():
    from helplessness.measures.diagnostics import mean_pairwise_jensen_shannon

    single = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert mean_pairwise_jensen_shannon(np.stack([single] * 5)) == pytest.approx(0.0, abs=1e-9)


def test_mean_pairwise_jensen_shannon_is_n_vars_bits_for_disjoint_point_masses():
    from helplessness.measures.diagnostics import mean_pairwise_jensen_shannon

    # 4 items, 2 variables, 4 classes: item i puts all mass on class i in both variables
    items = np.stack([np.stack([np.eye(4)[i], np.eye(4)[i]]) for i in range(4)])
    assert mean_pairwise_jensen_shannon(items) == pytest.approx(2.0, abs=1e-6)


def test_mean_pairwise_jensen_shannon_matches_the_pairwise_loop():
    from helplessness.measures.diagnostics import mean_pairwise_jensen_shannon

    items = np.random.default_rng(0).dirichlet(np.ones(3), size=(6, 2))
    pairs = [jensen_shannon_product(items[i], items[j]) for i in range(6) for j in range(6) if i != j]
    assert mean_pairwise_jensen_shannon(items) == pytest.approx(float(np.mean(pairs)))


def test_mean_random_pair_jensen_shannon_never_pairs_an_item_with_itself():
    """Items that are all distinct point masses: every distinct pair has
    JS = n_vars bits exactly, so the statistic must be exactly n_vars --
    any self-pair would pull it below."""
    from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon

    items = np.stack([np.stack([np.eye(6)[i], np.eye(6)[i]]) for i in range(6)])  # 6 items, 2 vars
    assert mean_random_pair_jensen_shannon(items, np.random.default_rng(0), n_pairs=500) == pytest.approx(2.0, abs=1e-6)


def test_mean_random_pair_jensen_shannon_is_zero_under_collapse():
    from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon

    single = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert mean_random_pair_jensen_shannon(np.stack([single] * 10), np.random.default_rng(0)) == pytest.approx(0.0, abs=1e-9)


def test_mean_random_pair_jensen_shannon_estimates_the_all_pairs_mean():
    from helplessness.measures.diagnostics import mean_pairwise_jensen_shannon, mean_random_pair_jensen_shannon

    items = np.random.default_rng(1).dirichlet(np.ones(4), size=(30, 3))
    exact = mean_pairwise_jensen_shannon(items)
    estimate = mean_random_pair_jensen_shannon(items, np.random.default_rng(2), n_pairs=20_000)
    assert estimate == pytest.approx(exact, rel=0.03)
