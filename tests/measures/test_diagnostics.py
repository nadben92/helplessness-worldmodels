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


def test_d1_contrast_uses_the_larger_of_the_floor_and_three_combined_se():
    from helplessness.measures.diagnostics import d1_contrast

    small_se = d1_contrast(I_L=0.5, se_L=0.01, I_Lprime=0.7, se_Lprime=0.01)  # floor 0.25 dominates
    assert small_se["threshold"] == 0.25 and not small_se["accepted"]
    assert d1_contrast(0.5, 0.01, 0.8, 0.01)["accepted"]
    big_se = d1_contrast(I_L=0.5, se_L=0.1, I_Lprime=0.9, se_Lprime=0.1)  # 3 * sqrt(0.02) = 0.424 > 0.4
    assert big_se["threshold"] == pytest.approx(3 * np.sqrt(0.02)) and not big_se["accepted"]
    assert d1_contrast(0.9, 0.01, 0.5, 0.01)["accepted"]  # sign does not matter


def test_alpha_app_with_se_hand_values_and_numeric_derivative():
    from helplessness.measures.diagnostics import alpha_app_with_se

    alpha, se = alpha_app_with_se(np.array([0.5]), np.array([0.0]), I_L=0.0, se_L=0.0, I_Lprime=2.0, se_Lprime=0.0)
    assert alpha[0] == 0.25 and se[0] == 0.0
    _, se_swap_only = alpha_app_with_se(np.array([0.5]), np.array([0.1]), 0.0, 0.0, 2.0, 0.0)
    assert se_swap_only[0] == pytest.approx(0.05)  # 0.1 / c
    # each partial derivative against a finite difference
    I_s, I_L, I_Lp, h = 0.7, 0.2, 1.4, 1e-6
    f = lambda a, b, c: (a - b) / (c - b)
    for which in range(3):
        se_args = [0.0, 0.0, 0.0]
        se_args[which] = 1.0
        _, se_one = alpha_app_with_se(np.array([I_s]), np.array([se_args[0]]), I_L, se_args[1], I_Lp, se_args[2])
        args = [I_s, I_L, I_Lp]
        up = list(args); up[which] += h
        dn = list(args); dn[which] -= h
        assert se_one[0] == pytest.approx(abs((f(*up) - f(*dn)) / (2 * h)), rel=1e-5)


def test_summarize_alpha_app_flags_anomalies_and_never_averages_them():
    from helplessness.measures.diagnostics import summarize_alpha_app

    alpha = np.array([0.2, 0.4, 1.5, -0.05, 0.3])
    se = np.array([0.01, 0.01, 0.01, 0.1, 0.01])  # 1.5 is an anomaly; -0.05 is within 3 se of 0
    out = summarize_alpha_app(alpha, se)
    assert out["n_anomalies"] == 1 and out["anomalies"][0]["index"] == 2
    kept = np.array([0.2, 0.4, -0.05, 0.3])
    assert out["mean"] == pytest.approx(kept.mean())
    half = 3 * kept.std(ddof=1) / 2.0
    assert out["ci_low"] == pytest.approx(kept.mean() - half) and out["ci_high"] == pytest.approx(kept.mean() + half)
    lone = summarize_alpha_app(np.array([0.5, 5.0]), np.array([0.01, 0.01]))
    assert lone["n_kept"] == 1 and lone["ci_low"] is None
