"""measures.rsa (docs/e3/DECISIONS.md, 2026-10-13 (31)): the generic RSA on synthetic latents."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.measures import rsa


def test_point_means_and_missing_points():
    x = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 5.0]])
    means, counts = rsa.point_means(x, np.array([0, 0, 2]), 3)
    assert counts.tolist() == [2, 0, 1]
    assert np.allclose(means[0], [2.0, 1.0]) and np.isnan(means[1]).all() and np.allclose(means[2], [5.0, 5.0])
    with pytest.raises(ValueError):
        rsa.point_means(x, np.array([0, 3, 1]), 3)


def test_l2_and_js_distances():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert np.allclose(rsa.l2_distances(x), [[0, 5], [5, 0]])
    p = np.zeros((3, 2, 2))
    p[0, :, 0] = 1.0           # both variables on class 0
    p[1, :, 1] = 1.0           # both on class 1: disjoint, 1 bit per variable
    p[2] = 0.5                 # uniform
    d = rsa.js_distances(p)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0)
    assert np.isclose(d[0, 1], 2.0)                      # V = 2 variables, each disjoint
    h_mix = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))  # point mass vs uniform: m = (0.75, 0.25) per variable
    assert np.isclose(d[0, 2], 2 * (h_mix - 0.5 * (0.0 + 1.0))) and np.isclose(d[0, 2], d[1, 2])


def test_torus_distances_and_centre():
    pos = np.array([[0, 0], [0, 11], [6, 6]])
    d = rsa.torus_distances(pos, (12, 12))
    assert d[0, 1] == 1 and d[0, 2] == 12 and d[1, 2] == 11
    assert np.isclose(rsa.torus_distances(pos, (12, 12), "euclidean")[0, 2], np.hypot(6, 6))
    c = rsa.torus_centre(np.array([[0, 11], [0, 0], [0, 1]]), (12, 12))
    assert np.allclose(c, [0, 0], atol=1e-9) or np.allclose(c, [0, 12], atol=1e-9)


def test_regression_recovers_the_generating_matrix_and_drops_constant_predictors():
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 12, size=(40, 2))
    theme = rng.integers(0, 4, size=40)
    D_pos, D_theme = rsa.torus_distances(pos, (12, 12)), rsa.label_distances(theme)
    model = 0.2 * D_pos + 3.0 * D_theme
    res = rsa.rsa_regression(model, {"position": D_pos, "theme": D_theme, "controllability": rsa.label_distances(["c"] * 40)})
    assert res["constant_predictors"] == ["controllability"]
    assert np.isclose(res["r2"], 1.0) and res["n_pairs"] == 40 * 39 // 2
    assert res["weights"]["theme"] > res["weights"]["position"] > 0
    assert res["unique_r2"]["theme"] > 0 and res["unique_r2"]["position"] > 0
    noisy = rsa.rsa_regression(D_pos + rng.normal(0, 50, D_pos.shape), {"position": D_pos})
    assert noisy["r2"] < 0.2


def test_points_without_latent_are_left_out():
    x = np.array([[0.0], [1.0], [np.nan], [3.0]])
    d = rsa.l2_distances(np.nan_to_num(x, nan=0.0))
    d[2, :] = d[:, 2] = np.nan
    d[2, 2] = 0.0
    assert rsa.valid_points(d).tolist() == [True, True, False, True]
    res = rsa.rsa_regression(d, {"lab": rsa.label_distances(["a", "a", "b", "b"])})
    assert res["n_points"] == 3 and res["n_pairs"] == 3
    coords, share = rsa.classical_mds(d)
    assert np.isnan(coords[2]).all() and np.isfinite(coords[[0, 1, 3]]).all()


def test_group_ratio_separated_and_confused():
    # two zones per group, four points per zone; controllable zones far from uncontrollable ones
    zones = np.repeat(["CD", "CN", "UD", "UN"], 4)
    groups = np.where(np.char.startswith(zones, "C"), "controllable", "uncontrollable")
    rng = np.random.default_rng(1)
    centre = {"CD": 0.0, "CN": 1.0, "UD": 20.0, "UN": 21.0}
    x = np.array([[centre[z] + rng.normal(0, 0.1)] for z in zones])
    sep = rsa.group_distance_ratio(rsa.l2_distances(x), groups, "controllable", "uncontrollable", zones)
    assert sep["ratio"] > 20 and sep["n_a"] == 8 and sep["n_b"] == 8
    x2 = np.array([[rng.normal(0, 1.0)] for _ in zones])
    conf = rsa.group_distance_ratio(rsa.l2_distances(x2), groups, "controllable", "uncontrollable", zones)
    assert 0.5 < conf["ratio"] < 2.0


def test_classical_mds_recovers_a_planar_configuration():
    pts = np.array([[0, 0], [1, 0], [0, 2], [3, 1], [2, 2]], dtype=float)
    coords, share = rsa.classical_mds(rsa.l2_distances(pts))
    assert np.allclose(rsa.l2_distances(coords), rsa.l2_distances(pts), atol=1e-8)
    assert np.isclose(share.sum(), 1.0)


def test_analyse_points_end_to_end():
    """A latent built from the zone (theme) plus noise: theme explains the model matrix, the split-half reliability is
    high, controllability (collinear with the zone here) is estimated, a constant attribute is dropped."""
    rng = np.random.default_rng(3)
    grid = (6, 6)
    pos = np.array([divmod(s, 6) for s in range(36)])
    zone = np.where(pos[:, 0] < 3, np.where(pos[:, 1] < 3, "CD", "UD"), np.where(pos[:, 1] < 3, "CN", "UN"))
    controllable = np.where(np.char.startswith(zone, "C"), "c", "u")
    centres = {"CD": 0, "UD": 1, "CN": 2, "UN": 3}
    point = np.repeat(np.arange(36), 30)
    h = np.zeros((point.size, 4))
    h[np.arange(point.size), [centres[zone[p]] for p in point]] = 3.0
    h += rng.normal(0, 0.5, h.shape)
    z = np.full((point.size, 2, 4), 0.05)
    z[np.arange(point.size), 0, [centres[zone[p]] for p in point]] = 0.85
    z += rng.uniform(0, 0.1, z.shape)
    z /= z.sum(-1, keepdims=True)
    mask = np.ones(36, dtype=bool)
    mask[0] = False  # e.g. a terminal state
    out = rsa.analyse_points({"h": h, "z": z}, point, 36, {"theme": zone, "controllability": controllable, "constant": ["x"] * 36},
                             pos, grid, rng, ratios={"c_vs_u": ("controllability", "c", "u", "theme")}, mask=mask)
    for name in ("h", "z"):
        reg = out["latents"][name]["regression"]
        assert reg["n_points"] == 35 and reg["constant_predictors"] == ["constant"]
        assert reg["unique_r2"]["theme"] > 0.2 and reg["r2"] > 0.8
        assert out["latents"][name]["split_half"]["spearman_brown"] > 0.9
        assert out["latents"][name]["ratios"]["c_vs_u"]["ratio"] > 3
        assert np.isnan(out["matrices"][f"{name}_mds"][0]).all()
    assert out["contexts_per_point"]["min"] == 30 and out["point_means"]["z"].shape == (36, 2, 4)
