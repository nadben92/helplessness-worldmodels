"""E7's direction statistics (helplessness.interventions.latent_direction)."""

from __future__ import annotations

import numpy as np

from helplessness.interventions import latent_direction as ld


def test_aligned_corrections_against_the_null():
    rng = np.random.default_rng(0)
    d = 64
    v = ld.unit(rng.standard_normal(d))
    aligned = 3.0 * v[None, :] + 0.1 * rng.standard_normal((40, d))
    a = ld.alignment(aligned, rng, n_null=50)
    assert a["cos_mean"] > 0.9 and abs(a["null_cos_mean"]) < 0.05 and a["null_cos_p95"] < 0.2
    assert a["null_pair_sd_analytic"] == 1 / 8
    iso = ld.alignment(rng.standard_normal((40, d)), rng, n_null=50)
    assert abs(iso["cos_mean"]) < 0.1


def test_leave_one_out_excludes_itself():
    d = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(ld.leave_one_out_cosines(d), [0.0, 0.0])  # each against the other only


def test_stability_and_efficacy():
    s = ld.stability(np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))
    assert s["cos_mean"] == 1.0 and s["norm_max"] == 2.0
    e = ld.efficacy(np.array([0.5, 1.0, 0.0]), np.array([1.0, 2.0, 0.01]))
    assert e["ratio_of_means"] == 1.5 / 3.01 and e["median_ratio"] == 0.5 and e["n_ratio"] == 2
