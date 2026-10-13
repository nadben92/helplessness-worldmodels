"""Representational similarity analysis (RSA) of a latent state, generic (docs/e3/DECISIONS.md, 2026-10-13 (31)).

Arrays in, arrays / scalars out; no environment or agent import (CLAUDE.md, "the single source of truth"). The bridge
from a model to these arrays -- which contexts, which latent, which labels -- belongs to the experiment.

The pipeline, for one model at one checkpoint:
  1. measurement points: every context carries the index of its point (a state, a zone, anything); the latent of a
     point is the mean over its contexts (point_means), the deterministic part h and the categorical part z
     (probabilities, shape (..., V, C)) separately;
  2. model distance matrices between points: L2 on h (l2_distances), Jensen-Shannon on z (js_distances: the sum over
     the V categorical variables of the JS divergence between the points' mean distributions, in bits);
  3. theoretical distance matrices from the points' attributes: distance on the torus (torus_distances) and
     "different label" indicators (label_distances: same theme, same controllability, same stochasticity, ...);
  4. regression of a model matrix on the theoretical ones over the off-diagonal pairs (rsa_regression: ordinary least
     squares on z-scored vectors, standardised weights, R^2, each predictor's unique share of variance); a predictor
     constant over the pairs (e.g. controllability in an environment where every zone is controllable) is left out
     and reported as such, never silently;
  5. scalars (group_distance_ratio: the mean distance between two groups of points over the mean distance within
     groups -- the confusion between controllable and uncontrollable zones when the groups are those categories);
  6. a 2D embedding for display (classical_mds).
"""

from __future__ import annotations

import numpy as np

LOG2 = np.log(2.0)


def point_means(x: np.ndarray, point_of_context: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of x (n_contexts, ...) over the contexts of each point, and the number of contexts per point. A point
    without a context gets NaN."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.asarray(point_of_context, dtype=np.int64)
    if idx.shape[0] != x.shape[0]:
        raise ValueError(f"{x.shape[0]} contexts but {idx.shape[0]} point indices")
    if idx.size and (idx.min() < 0 or idx.max() >= n_points):
        raise ValueError("point index outside [0, n_points)")
    flat = x.reshape(x.shape[0], -1)
    sums = np.zeros((n_points, flat.shape[1]))
    np.add.at(sums, idx, flat)
    counts = np.bincount(idx, minlength=n_points)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts[:, None]
    means[counts == 0] = np.nan
    return means.reshape((n_points,) + x.shape[1:]), counts


def l2_distances(x: np.ndarray) -> np.ndarray:
    """Euclidean distances between the rows of x (n, d)."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    sq = (x ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * x @ x.T, 0.0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1) / LOG2


def js_distances(p: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (bits) between factorised categorical distributions p (n, V, C) (each of the V
    variables a distribution over C classes): the sum over the variables of the JS divergence of each variable's
    distributions. Bounded by V bits; 0 on the diagonal."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"expected (n, V, C) probabilities, got shape {p.shape}")
    n = p.shape[0]
    h = _entropy_bits(p).sum(axis=1)  # (n,)
    d = np.zeros((n, n))
    for i in range(n):
        m = 0.5 * (p[i][None] + p)  # (n, V, C)
        d[i] = _entropy_bits(m).sum(axis=1) - 0.5 * (h[i] + h)
    d = np.maximum(0.5 * (d + d.T), 0.0)
    np.fill_diagonal(d, 0.0)
    return d


def torus_distances(positions: np.ndarray, grid_shape: tuple[int, int], metric: str = "manhattan") -> np.ndarray:
    """Distances between points at (row, col) positions (n, 2), possibly fractional (a zone's centre), on a torus of
    grid_shape: per axis the shorter way round, combined as manhattan (4-neighbour moves) or euclidean."""
    pos = np.asarray(positions, dtype=np.float64)
    shape = np.asarray(grid_shape, dtype=np.float64)
    diff = np.abs(pos[:, None, :] - pos[None, :, :])
    diff = np.minimum(diff, shape - diff)
    if metric == "manhattan":
        return diff.sum(axis=-1)
    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=-1))
    raise ValueError(f"unknown metric {metric!r}")


def torus_centre(positions: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """The circular mean of (row, col) positions (k, 2) on the torus: a set of cells' centre."""
    pos = np.asarray(positions, dtype=np.float64)
    out = np.empty(2)
    for axis in range(2):
        angle = 2.0 * np.pi * pos[:, axis] / grid_shape[axis]
        mean = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
        out[axis] = (mean * grid_shape[axis] / (2.0 * np.pi)) % grid_shape[axis]
    return out


def label_distances(labels) -> np.ndarray:
    """1 where two points carry different labels, 0 where the same (the theoretical matrix of a categorical
    attribute: same theme, same controllability, ...)."""
    lab = np.asarray([str(x) for x in labels])
    return (lab[:, None] != lab[None, :]).astype(np.float64)


def valid_points(d: np.ndarray) -> np.ndarray:
    """The points of a distance matrix that have a latent: not every off-diagonal distance NaN (a point without a
    context has a NaN mean, hence a NaN row)."""
    d = np.asarray(d, dtype=np.float64)
    off = np.where(np.eye(len(d), dtype=bool), np.nan, d)
    return np.isfinite(off).any(axis=1) if len(d) > 1 else np.ones(len(d), dtype=bool)


def upper(d: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """The off-diagonal upper triangle of a square matrix as a vector (restricted to points in mask, if given)."""
    d = np.asarray(d, dtype=np.float64)
    if mask is not None:
        d = d[np.ix_(mask, mask)]
    i, j = np.triu_indices(d.shape[0], k=1)
    return d[i, j]


def _zscore(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1.0 - (resid ** 2).sum() / (y ** 2).sum())


def rsa_regression(model: np.ndarray, predictors: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict:
    """Regression of a model distance matrix on theoretical ones over the off-diagonal pairs of the points in mask
    (points with a NaN model distance are excluded first). Response and predictors are z-scored, so the weights are
    standardised and there is no intercept. Returns the weights, R^2, each predictor's unique share (R^2 lost when
    it is removed), each predictor's (Pearson) correlation with the response, the predictors left out as constant, the
    rank of the design and the number of pairs."""
    model = np.asarray(model, dtype=np.float64)
    keep = valid_points(model) if mask is None else valid_points(model) & np.asarray(mask, dtype=bool)
    y = upper(model, keep)
    if not np.isfinite(y).all():
        raise ValueError("NaN distances between points that both have a latent")
    out = {"n_points": int(keep.sum()), "n_pairs": int(y.size), "weights": {}, "unique_r2": {}, "correlation": {},
           "constant_predictors": [], "r2": float("nan"), "rank": 0}
    if y.size < 3 or y.std() == 0:
        out["undefined_reason"] = "fewer than 3 pairs" if y.size < 3 else "constant model distances"
        return out
    yz = _zscore(y)
    cols, names = [], []
    for name, d in predictors.items():
        v = upper(d, keep)
        if v.std() == 0:
            out["constant_predictors"].append(name)
            continue
        cols.append(_zscore(v))
        names.append(name)
        out["correlation"][name] = float(np.corrcoef(v, y)[0, 1])
    X = np.column_stack(cols) if cols else np.zeros((y.size, 0))
    out["rank"] = int(np.linalg.matrix_rank(X)) if cols else 0
    if cols:
        beta, *_ = np.linalg.lstsq(X, yz, rcond=None)
        out["weights"] = {nm: float(b) for nm, b in zip(names, beta)}
    out["r2"] = _r2(yz, X)
    for k, nm in enumerate(names):
        out["unique_r2"][nm] = out["r2"] - _r2(yz, np.delete(X, k, axis=1))
    if out["rank"] < len(names):
        out["collinear"] = True  # weights not unique: read the unique shares and correlations, not the weights
    return out


def group_distance_ratio(d: np.ndarray, groups, a: str, b: str, within) -> dict:
    """Mean distance between the points of group a and those of group b, over the mean distance between distinct
    points that share a `within` label (e.g. the same zone); groups and within are per-point labels. A ratio near 1:
    the two groups are no further apart than points of one zone (confusion); much above 1: separated."""
    d = np.asarray(d, dtype=np.float64)
    g = np.asarray([str(x) for x in groups])
    w = np.asarray([str(x) for x in within])
    ok = valid_points(d)
    ia, ib = np.flatnonzero((g == a) & ok), np.flatnonzero((g == b) & ok)
    between = d[np.ix_(ia, ib)]
    same = (w[:, None] == w[None, :]) & ~np.eye(len(d), dtype=bool) & ok[:, None] & ok[None, :] & np.isin(g, [a, b])[:, None]
    within_mean = float(d[same].mean()) if same.any() else float("nan")
    between_mean = float(between.mean()) if between.size else float("nan")
    return {"between": between_mean, "within": within_mean,
            "ratio": (between_mean / within_mean if within_mean > 0 else (float("inf") if between_mean > 0 else float("nan")))
            if np.isfinite(within_mean) and np.isfinite(between_mean) else float("nan"),
            "n_a": int(ia.size), "n_b": int(ib.size)}


def classical_mds(d: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Classical (Torgerson) multidimensional scaling of a distance matrix: coordinates (n, k) and the share of the
    positive eigenvalues' sum carried by each of the k axes. Points with a NaN distance get NaN coordinates."""
    d = np.asarray(d, dtype=np.float64)
    n = d.shape[0]
    ok = valid_points(d)
    coords = np.full((n, k), np.nan)
    m = int(ok.sum())
    if m < 2:
        return coords, np.full(k, np.nan)
    dd = d[np.ix_(ok, ok)]
    J = np.eye(m) - 1.0 / m
    B = -0.5 * J @ (dd ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = np.clip(vals, 0.0, None)
    kk = min(k, m)
    coords[ok, :kk] = vecs[:, :kk] * np.sqrt(pos[:kk])
    share = np.full(k, np.nan)
    share[:kk] = pos[:kk] / pos.sum() if pos.sum() > 0 else np.nan
    return coords, share


def split_half_reliability(x: np.ndarray, point_of_context: np.ndarray, n_points: int, distance, rng: np.random.Generator,
                           mask: np.ndarray | None = None) -> dict:
    """The model matrix's reliability given the finite contexts: each point's contexts split at random into two halves,
    the distance matrix of each half's means, the Pearson correlation of their off-diagonal entries (points with
    contexts in both halves, within mask) and its Spearman-Brown correction to the full number of contexts, 2r / (1 + r)
    (the ceiling of what any theoretical matrix can explain)."""
    idx = np.asarray(point_of_context, dtype=np.int64)
    half = np.zeros(idx.size, dtype=bool)
    for p in np.unique(idx):
        members = np.flatnonzero(idx == p)
        half[rng.permutation(members)[: members.size // 2]] = True
    d1 = distance(point_means(x[half], idx[half], n_points)[0])
    d2 = distance(point_means(x[~half], idx[~half], n_points)[0])
    keep = valid_points(d1) & valid_points(d2)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    v1, v2 = upper(d1, keep), upper(d2, keep)
    if v1.size < 3 or v1.std() == 0 or v2.std() == 0:
        return {"r": float("nan"), "spearman_brown": float("nan"), "n_points": int(keep.sum())}
    r = float(np.corrcoef(v1, v2)[0, 1])
    return {"r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan"), "n_points": int(keep.sum())}


DISTANCES = {"h": l2_distances, "z": js_distances}


def analyse_point_means(latents: dict[str, np.ndarray], attributes: dict[str, list], positions: np.ndarray,
                        grid_shape: tuple[int, int], ratios: dict[str, tuple[str, str, str, str]] | None = None,
                        mask: np.ndarray | None = None, predictors: list[str] | None = None) -> dict:
    """The RSA of points whose latents are given (point means): latents {"h": (n, H), "z": (n, V, C) probabilities};
    attributes {name: per-point labels} (each gives a "different label" theoretical matrix), with the torus distance
    between the positions as the "position" matrix; ratios {name: (attribute, a, b, within attribute)} for
    group_distance_ratio. Per latent: the regression on every theoretical matrix (predictors: the attributes used as
    predictors, default all -- leave out one identical to another), the ratios and a 2D MDS. Points out of mask are left
    out of the regression and the ratios (and get no MDS coordinates)."""
    names = list(attributes) if predictors is None else list(predictors)
    theory = {"position": torus_distances(positions, grid_shape), **{k: label_distances(attributes[k]) for k in names}}
    n = len(positions)
    keep = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = {"theory": list(theory), "latents": {}, "matrices": {}}
    for name, lat in latents.items():
        d = DISTANCES[name](lat)
        d_masked = d.copy()
        d_masked[~keep, :] = np.nan
        d_masked[:, ~keep] = np.nan
        np.fill_diagonal(d_masked, 0.0)
        entry = {"regression": rsa_regression(d_masked, theory), "ratios": {}}
        for rname, (attr, a, b, within) in (ratios or {}).items():
            entry["ratios"][rname] = group_distance_ratio(d_masked, attributes[attr], a, b, attributes[within])
        coords, share = classical_mds(d_masked)
        entry["mds_share"] = share.tolist()
        out["latents"][name] = entry
        out["matrices"][name] = d
        out["matrices"][f"{name}_mds"] = coords
    return out


def analyse_points(contexts: dict[str, np.ndarray], point_of_context: np.ndarray, n_points: int, attributes: dict[str, list],
                   positions: np.ndarray, grid_shape: tuple[int, int], rng: np.random.Generator,
                   ratios: dict[str, tuple[str, str, str, str]] | None = None, mask: np.ndarray | None = None,
                   predictors: list[str] | None = None) -> dict:
    """analyse_point_means on the contexts' point means, with the number of contexts per point and each latent's
    split-half reliability. contexts: {"h": (n_ctx, H), "z": (n_ctx, V, C) probabilities}."""
    means, counts = {}, None
    for name, x in contexts.items():
        means[name], counts = point_means(x, point_of_context, n_points)
    out = analyse_point_means(means, attributes, positions, grid_shape, ratios, mask, predictors)
    out["contexts_per_point"] = {"min": int(counts.min()), "median": float(np.median(counts)), "max": int(counts.max()),
                                 "points_without_context": int((counts == 0).sum())}
    for name, x in contexts.items():
        out["latents"][name]["split_half"] = split_half_reliability(x, point_of_context, n_points, DISTANCES[name], rng, mask)
    out["point_means"] = means
    return out
