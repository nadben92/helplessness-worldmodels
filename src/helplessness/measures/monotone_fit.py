"""Monotone curves of a phenotype against the belief b (E6's analysis, docs/E6_phenotypes_latent.md): arrays in, arrays and
scalars out (numpy, scipy). Nothing here imports an environment, an agent or torch.

  isotonic       least-squares isotonic regression (pool-adjacent-violators), increasing or decreasing; the direction,
                 when not given, is the one with the smaller residual sum of squares.
  smooth         the isotonic fit's blocks (one knot per block: the block's mean x and fitted y) joined by a monotone
                 cubic (PCHIP), constant beyond the end knots: a smooth curve with the isotonic fit's direction.
  invert         the smooth curve's inverse on its range (a grid search: the curve is monotone, possibly flat): the b whose
                 predicted phenotype is nearest to an observed one; flagged when the observation is outside the range.
  cluster bootstrap  resample the CLUSTERS (agents) with replacement, refit, evaluate on a grid: the confidence band,
                 and the monotonicity read on it (strict_range).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Increasing isotonic fit of y (already sorted by x) with weights w."""
    means, weights, sizes = [], [], []
    for yi, wi in zip(y, w):
        means.append(float(yi))
        weights.append(float(wi))
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            m2, w2, n2 = means.pop(), weights.pop(), sizes.pop()
            m1, w1, n1 = means.pop(), weights.pop(), sizes.pop()
            means.append((m1 * w1 + m2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
            sizes.append(n1 + n2)
    return np.repeat(means, sizes)


def isotonic(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, increasing: bool | None = None) -> dict:
    """The isotonic fit at the sorted points: x_sorted, fitted, increasing (the direction used), sse. Ties in x are
    pooled first (their weighted mean), so the fit is a function of x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    ux, inv = np.unique(x, return_inverse=True)
    wy = np.bincount(inv, weights=w * y)
    ww = np.bincount(inv, weights=w)
    uy = wy / ww
    fits = {}
    for inc in ((True, False) if increasing is None else (increasing,)):
        f = _pava(uy if inc else -uy, ww)
        f = f if inc else -f
        fits[inc] = (f, float((w * (y - f[inv]) ** 2).sum()))
    inc = min(fits, key=lambda k: fits[k][1])
    return {"x": ux, "fitted": fits[inc][0], "weights": ww, "increasing": inc, "sse": fits[inc][1]}


def knots(fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """One knot per constant block of the isotonic fit: the block's weighted mean x and its value."""
    f, x, w = fit["fitted"], fit["x"], fit["weights"]
    edges = np.flatnonzero(np.diff(f) != 0) + 1
    kx, ky = [], []
    for block in np.split(np.arange(len(f)), edges):
        kx.append(float((x[block] * w[block]).sum() / w[block].sum()))
        ky.append(float(f[block[0]]))
    return np.asarray(kx), np.asarray(ky)


def smooth(fit: dict, grid: np.ndarray) -> np.ndarray:
    """The smooth monotone curve on `grid` (module docstring); a single block gives a constant."""
    kx, ky = knots(fit)
    if kx.size == 1:
        return np.full(len(grid), ky[0])
    return PchipInterpolator(kx, ky, extrapolate=False)(np.clip(grid, kx[0], kx[-1]))


def step_curve(fit: dict, grid: np.ndarray) -> np.ndarray:
    """The isotonic fit itself on `grid`: linear between the fitted points, constant beyond them."""
    return np.interp(grid, fit["x"], fit["fitted"])


def predict(fit: dict, x: np.ndarray, kind: str = "smooth") -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return smooth(fit, x) if kind == "smooth" else step_curve(fit, x)


def invert(fit: dict, y_obs: np.ndarray, grid: np.ndarray) -> dict[str, np.ndarray]:
    """b predicted from observed phenotypes: the grid point whose smooth prediction is nearest (the middle of the
    nearest points' run when the curve is flat there); outside = the observation lies outside the curve's range."""
    curve = smooth(fit, grid)
    y_obs = np.atleast_1d(np.asarray(y_obs, dtype=float))
    b_hat = np.empty(len(y_obs))
    flat = np.zeros(len(y_obs), dtype=bool)
    for i, y in enumerate(y_obs):
        d = np.abs(curve - y)
        near = np.flatnonzero(d <= d.min() + 1e-12)
        b_hat[i], flat[i] = 0.5 * (grid[near[0]] + grid[near[-1]]), near.size > 1
    lo, hi = np.nanmin(curve), np.nanmax(curve)
    return {"b_hat": b_hat, "outside": (y_obs < lo - 1e-12) | (y_obs > hi + 1e-12), "flat": flat}


def cluster_bootstrap(x: np.ndarray, y: np.ndarray, clusters: np.ndarray, grid: np.ndarray, rng: np.random.Generator,
                      n_boot: int = 1000, increasing: bool | None = None, w: np.ndarray | None = None) -> dict:
    """Clusters (agents) resampled with replacement; the direction fixed to the full fit's unless given. Returns the
    smooth curves (n_boot, len(grid)), NaN outside each resample's x range (no extrapolation is read)."""
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    if increasing is None:
        increasing = isotonic(x, y, w)["increasing"]
    ids = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in ids]
    curves = np.full((n_boot, len(grid)), np.nan)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(len(ids), size=len(ids))])
        if np.unique(x[idx]).size < 2:
            continue
        fit = isotonic(x[idx], y[idx], w[idx], increasing)
        c = smooth(fit, grid)
        c[(grid < x[idx].min()) | (grid > x[idx].max())] = np.nan
        curves[b] = c
    return {"curves": curves, "increasing": increasing}


def band(curves: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile band of bootstrap curves (NaN where no resample defines the curve)."""
    q = (1.0 - level) / 2.0
    lo, hi = np.full(curves.shape[1], np.nan), np.full(curves.shape[1], np.nan)
    for j in range(curves.shape[1]):
        v = curves[:, j][np.isfinite(curves[:, j])]
        if v.size:
            lo[j], hi[j] = np.quantile(v, q), np.quantile(v, 1 - q)
    return lo, hi


def strict_range(curves: np.ndarray, grid: np.ndarray, increasing: bool, window: float, min_delta: float, level: float = 0.95) -> dict:
    """Where the curve is strictly monotone, read on the bootstrap: at each grid point b, the change over [b, b + window]
    is in the main direction and larger than min_delta in at least `level` of the resamples defined there. Returns the
    mask, the longest run of such points as [lo, hi + window] (the range on which the curve is invertible at that
    resolution) and its share of the grid's span."""
    step = grid[1] - grid[0]
    shift = max(1, int(round(window / step)))
    diff = curves[:, shift:] - curves[:, :-shift]
    diff = diff if increasing else -diff
    defined = np.isfinite(diff)
    ok = (diff > min_delta) & defined
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = ok.sum(axis=0) / np.maximum(defined.sum(axis=0), 1)
    mask = (frac >= level) & (defined.sum(axis=0) > 0)
    best, run = (None, None), (None, 0)
    start = None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        if not m and start is not None:
            if i - start > run[1]:
                run = (start, i - start)
                best = (float(grid[start]), float(grid[i - 1] + shift * step))
            start = None
    span = grid[-1] - grid[0]
    length = 0.0 if best[0] is None else best[1] - best[0]
    return {"mask": mask, "share_strict": frac, "invertible_range": best, "invertible_share": float(length / span) if span else float("nan")}


def curve_summary(x: np.ndarray, y: np.ndarray, clusters: np.ndarray, grid: np.ndarray, rng: np.random.Generator, n_boot: int,
                  window: float, min_delta_share: float, level: float = 0.95) -> dict:
    """The main curve of a phenotype against b, as E6 and E7 read it: the isotonic fit (direction by the smaller SSE),
    its smooth version on the grid (NaN outside the observed range of b), the cluster-bootstrap band, Spearman's rho with
    its cluster-bootstrap interval, and the strict-monotonicity range (strict_range, min_delta = min_delta_share x the
    observed range of y)."""
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    fit = isotonic(x, y)
    curve = smooth(fit, grid)
    curve[(grid < x.min()) | (grid > x.max())] = np.nan
    boot = cluster_bootstrap(x, y, clusters, grid, rng, n_boot, fit["increasing"])
    lo, hi = band(boot["curves"], level)
    ids = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in ids]
    rhos = []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(len(ids), size=len(ids))])
        rhos.append(spearman(x[idx], y[idx]))
    rhos = np.asarray(rhos)
    rhos = rhos[np.isfinite(rhos)]
    q = (1.0 - level) / 2.0
    y_range = float(np.nanmax(y) - np.nanmin(y)) if y.size else 0.0
    strict = strict_range(boot["curves"], grid, fit["increasing"], window, min_delta_share * y_range, level)
    return {"fit": fit, "curve": curve, "band_lo": lo, "band_hi": hi, "increasing": fit["increasing"], "spearman": spearman(x, y),
            "spearman_ci": (float(np.quantile(rhos, q)), float(np.quantile(rhos, 1 - q))) if rhos.size else (float("nan"), float("nan")),
            "strict": strict, "n_points": int(x.size), "n_clusters": int(ids.size)}


def held_out(x: np.ndarray, y: np.ndarray, groups: np.ndarray, grid: np.ndarray) -> list[dict]:
    """For each group (a condition, or an agent): the curve fitted WITHOUT its points (isotonic, smooth), then its points'
    residuals y - f(x) and the blind test, b predicted from y by inverting that curve, against the measured b. Points
    outside the curve's range of b are predicted by the end values (constant extension) and flagged."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    out = []
    for g in np.unique(groups):
        test, train = groups == g, groups != g
        if np.unique(x[train]).size < 2:
            out.append({"group": g, "n": int(test.sum()), "skipped": "fewer than two distinct b without it"})
            continue
        fit = isotonic(x[train], y[train])
        pred = predict(fit, x[test])
        inv = invert(fit, y[test], grid)
        out.append({"group": g, "n": int(test.sum()), "increasing": fit["increasing"], "x": x[test], "y": y[test], "residual": y[test] - pred,
                    "b_hat": inv["b_hat"], "b_error": inv["b_hat"] - x[test], "outside_y_range": inv["outside"], "flat": inv["flat"],
                    "outside_b_range": (x[test] < x[train].min()) | (x[test] > x[train].max()),
                    "rmse_residual": float(np.sqrt(np.mean((y[test] - pred) ** 2))),
                    "mae_b": float(np.mean(np.abs(inv["b_hat"] - x[test])))})
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation (average ranks for ties)."""
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)
