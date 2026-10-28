"""Is there a common direction among corrections of a recurrent state? (E7, docs/E7_latent_manipulation.md.) Arrays in,
arrays and scalars out: nothing here imports an environment, an agent or torch (the corrections are computed by
probing.belief_steering).

  mean_direction    v = the mean of the corrections (n, d), and its unit vector.
  alignment         cos(delta_i, v_{-i}) with v_{-i} the mean of the OTHER corrections (leave-one-out: a correction is
                    not compared with a mean that contains it), summarised; the null reference is the same statistic for
                    isotropic Gaussian vectors with the same norms (simulated), and the analytic sd of the cosine
                    between two independent isotropic directions in d dimensions, 1 / sqrt(d) (its mean is 0).
  pairwise_cosines  the cosines between solutions of one situation (restarts), for the stability of the optimum.
  efficacy          the shift of b obtained with the common direction over the shift with each situation's own optimum.
"""

from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def mean_direction(deltas: np.ndarray) -> dict[str, np.ndarray | float]:
    deltas = np.asarray(deltas, dtype=float)
    v = deltas.mean(axis=0)
    return {"v": v, "v_unit": unit(v), "v_norm": float(np.linalg.norm(v)), "mean_norm": float(np.linalg.norm(deltas, axis=1).mean())}


def leave_one_out_cosines(deltas: np.ndarray) -> np.ndarray:
    """(n,): cos(delta_i, mean of the others)."""
    deltas = np.asarray(deltas, dtype=float)
    n = deltas.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    others = (deltas.sum(axis=0)[None, :] - deltas) / (n - 1)
    return (unit(deltas) * unit(others)).sum(axis=1)


def alignment(deltas: np.ndarray, rng: np.random.Generator, n_null: int = 200) -> dict:
    """The leave-one-out cosines (mean, median, sd, share > 0) and the null: the same mean cosine for isotropic vectors
    with the same norms, n_null times (its mean and 95th percentile), and 1 / sqrt(d)."""
    deltas = np.asarray(deltas, dtype=float)
    cos = leave_one_out_cosines(deltas)
    norms = np.linalg.norm(deltas, axis=1)
    null = []
    for _ in range(n_null):
        g = unit(rng.standard_normal(deltas.shape)) * norms[:, None]
        null.append(float(np.nanmean(leave_one_out_cosines(g))))
    null = np.asarray(null)
    d = deltas.shape[1]
    return {"n": int(deltas.shape[0]), "dim": int(d), "cos_mean": float(np.nanmean(cos)), "cos_median": float(np.nanmedian(cos)),
            "cos_sd": float(np.nanstd(cos)), "cos_share_positive": float(np.nanmean(cos > 0)),
            "null_cos_mean": float(null.mean()), "null_cos_p95": float(np.quantile(null, 0.95)),
            "null_pair_sd_analytic": float(1.0 / np.sqrt(d)), "cosines": cos}


def pairwise_cosines(solutions: np.ndarray) -> np.ndarray:
    """(k, d) solutions -> the k(k-1)/2 cosines between them."""
    u = unit(np.asarray(solutions, dtype=float))
    c = u @ u.T
    return c[np.triu_indices(len(u), k=1)]


def stability(solutions: np.ndarray) -> dict:
    """Restarts of one situation (k, d): norms (mean, sd, min, max) and pairwise cosines (mean, min)."""
    norms = np.linalg.norm(solutions, axis=1)
    cos = pairwise_cosines(solutions)
    return {"k": int(len(solutions)), "norm_mean": float(norms.mean()), "norm_sd": float(norms.std()), "norm_min": float(norms.min()),
            "norm_max": float(norms.max()), "cos_mean": float(cos.mean()) if cos.size else float("nan"),
            "cos_min": float(cos.min()) if cos.size else float("nan")}


def efficacy(db_direction: np.ndarray, db_optimal: np.ndarray) -> dict:
    """The shift of b with the common direction against the shift with each situation's own optimum: the ratio of the
    means, and the median of the per-situation ratios (situations whose optimal shift is below 0.05 bit left out)."""
    db_direction, db_optimal = np.asarray(db_direction, float), np.asarray(db_optimal, float)
    ok = np.isfinite(db_direction) & np.isfinite(db_optimal)
    big = ok & (np.abs(db_optimal) >= 0.05)
    ratio_of_means = float(db_direction[ok].mean() / db_optimal[ok].mean()) if ok.any() and db_optimal[ok].mean() != 0 else float("nan")
    return {"n": int(ok.sum()), "db_direction_mean": float(db_direction[ok].mean()) if ok.any() else float("nan"),
            "db_optimal_mean": float(db_optimal[ok].mean()) if ok.any() else float("nan"), "ratio_of_means": ratio_of_means,
            "median_ratio": float(np.median(db_direction[big] / db_optimal[big])) if big.any() else float("nan"), "n_ratio": int(big.sum())}
