"""Axes of themes between two colours (E4, docs/E4_generalization_gradient.md): the segment [T_inc, T_ctrl] and the
two circular arcs of radius D = |T_inc - T_ctrl| (Euclidean distance in RGB), 11 points each, used as the themes of
E4's zero-data probes. Pure colour geometry, numpy only: no zone, run or model here.

  segment  T_inc + t (T_ctrl - T_inc), t = 0, 0.1, ..., 1.
  arc      centred on `center`, radius D, starting at `start` (|start - center| = D): every point is at distance D from
           the centre, and the k-th is at distance d_k = k D / 10 from the start (angle theta_k = 2 arcsin(d_k / 2D),
           from 0 to 60 degrees). arc_ctrl = centred on T_ctrl from T_inc; arc_inc = centred on T_inc from T_ctrl.

The plane of an arc (spanned by u = (start - center) / D and a unit v orthogonal to u) is chosen by a fixed rule
(choose_arc_plane): among the planes v(phi) = cos(phi) e1 + sin(phi) e2 (e1, e2 a fixed orthonormal basis of the
plane orthogonal to u, built by Gram-Schmidt from the canonical axes; phi on a grid of `n_angles` values in [0, 2 pi)),
keep those whose 11 points, rounded to integers, all lie in the RGB cube [0, 255]^3; among them, take the one that
maximises the smallest CIELAB distance (deltaE76) from any point to the forbidden colours; ties -> the smallest phi.
The two arcs of a pair are chosen in a fixed order -- first the arc centred on the pair's second colour b, then the arc
centred on a -- and the second must keep its points (but its start) at least D / 10 (the axes' spacing) in RGB from
every point of the first: otherwise both arcs maximise the same margin, land in the same plane on the same side, and
end at the same colour (the apex of the equilateral triangle a, b, apex), which would merge two probes.
Rendered themes are integer RGB triples, so every axis point is rounded once (round half to even, numpy's rule).

An axis does not depend on which colour of the pair is T_inc: swapping the two colours swaps arc_ctrl and arc_inc and
reverses the segment, so both orientations of a pair use the same three sets of colours.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.appearance.cielab import srgb_to_lab

N_POINTS = 11


def rgb_distance(a, b) -> np.ndarray:
    """Euclidean distance in RGB (0-255 units) between colours (..., 3)."""
    return np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), axis=-1)


def lab(colors) -> np.ndarray:
    """CIELAB of RGB colours (..., 3) in 0-255 units."""
    return srgb_to_lab(np.asarray(colors, dtype=np.float64) / 255.0)


def lab_distances(colors, forbidden) -> np.ndarray:
    """deltaE76 between every colour (n, 3) and every forbidden colour (m, 3): (n, m)."""
    a, b = lab(np.atleast_2d(colors)), lab(np.atleast_2d(forbidden))
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def segment_axis(t_inc, t_ctrl, n: int = N_POINTS) -> np.ndarray:
    """(n, 3) float points of the segment, from T_inc (t = 0) to T_ctrl (t = 1)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    t_inc, t_ctrl = np.asarray(t_inc, dtype=np.float64), np.asarray(t_ctrl, dtype=np.float64)
    return t_inc + t * (t_ctrl - t_inc)


def orthonormal_complement(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A fixed orthonormal basis (e1, e2) of the plane orthogonal to the unit vector u: Gram-Schmidt on the canonical
    axes, in the order x, y, z, keeping the first two that are not (nearly) parallel to what is already kept."""
    basis = [u]
    for axis in np.eye(3):
        w = axis - sum(np.dot(axis, b) * b for b in basis)
        if np.linalg.norm(w) > 1e-6:
            basis.append(w / np.linalg.norm(w))
        if len(basis) == 3:
            break
    return basis[1], basis[2]


def arc_angles(n: int = N_POINTS) -> np.ndarray:
    """theta_k = 2 arcsin(d_k / 2D) with d_k = k D / (n - 1): equal steps of distance to the start, 0 to 60 degrees."""
    d_over_D = np.arange(n) / (n - 1)
    return 2.0 * np.arcsin(d_over_D / 2.0)


def arc_axis(center, start, phi: float, n: int = N_POINTS) -> np.ndarray:
    """(n, 3) float points of the arc centred on `center`, radius |start - center|, from `start`, in the plane of
    angle phi (module docstring)."""
    center, start = np.asarray(center, dtype=np.float64), np.asarray(start, dtype=np.float64)
    radius = float(np.linalg.norm(start - center))
    u = (start - center) / radius
    e1, e2 = orthonormal_complement(u)
    v = np.cos(phi) * e1 + np.sin(phi) * e2
    theta = arc_angles(n)[:, None]
    return center + radius * (np.cos(theta) * u + np.sin(theta) * v)


def to_rgb(points: np.ndarray) -> np.ndarray:
    """Integer RGB (round half to even); values outside [0, 255] are kept, so a caller can reject them."""
    return np.rint(np.asarray(points, dtype=np.float64)).astype(np.int64)


def in_cube(points_rgb: np.ndarray) -> bool:
    return bool(np.all((points_rgb >= 0) & (points_rgb <= 255)))


def choose_arc_plane(center, start, forbidden, n: int = N_POINTS, n_angles: int = 360, exclude=None,
                     min_separation: float = 0.0) -> dict | None:
    """The plane of an arc by the module's rule: in-cube planes only, then the largest smallest deltaE76 to the
    forbidden colours, ties to the smallest phi. exclude (m, 3), min_separation: planes whose points (but the start)
    come closer than min_separation (RGB) to an excluded colour are rejected. Returns {"phi", "points" (n, 3) int,
    "min_lab_distance"}, or None when no plane is left."""
    best = None
    for phi in np.arange(n_angles) * (2.0 * np.pi / n_angles):
        pts = to_rgb(arc_axis(center, start, phi, n))
        if not in_cube(pts):
            continue
        if exclude is not None and rgb_distance(pts[1:, None, :], np.asarray(exclude)[None, :, :]).min() < min_separation:
            continue
        margin = float(lab_distances(pts, forbidden).min())
        if best is None or margin > best["min_lab_distance"] + 1e-12:
            best = {"phi": float(phi), "points": pts, "min_lab_distance": margin}
    return best


def pair_axes(color_a, color_b, forbidden, n: int = N_POINTS, n_angles: int = 360) -> dict | None:
    """The three axes of a pair (color_a, color_b), independent of the orientation: "segment" from a to b,
    "arc_center_b" (centred on b, from a) and "arc_center_a" (centred on a, from b). None when an arc cannot stay
    in the cube. Each arc carries its plane angle and its smallest deltaE76 to the forbidden colours."""
    arc_b = choose_arc_plane(color_b, color_a, forbidden, n, n_angles)
    if arc_b is None:
        return None
    D = float(rgb_distance(color_a, color_b))
    arc_a = choose_arc_plane(color_a, color_b, forbidden, n, n_angles, exclude=arc_b["points"][1:], min_separation=D / (n - 1))
    if arc_a is None:
        return None
    segment = to_rgb(segment_axis(color_a, color_b, n))
    return {"segment": segment, "arc_center_b": arc_b["points"], "arc_center_a": arc_a["points"],
            "phi_center_b": arc_b["phi"], "phi_center_a": arc_a["phi"],
            "D": float(rgb_distance(color_a, color_b))}


def axes_for_orientation(axes: dict, inc: str) -> dict[str, np.ndarray]:
    """The three axes of one orientation, named by role, each from its own start as defined in the module docstring:
    inc = "a" means T_inc = color_a, T_ctrl = color_b. segment: T_inc -> T_ctrl; arc_ctrl: centred on T_ctrl, from
    T_inc; arc_inc: centred on T_inc, from T_ctrl."""
    if inc == "a":
        return {"segment": axes["segment"], "arc_ctrl": axes["arc_center_b"], "arc_inc": axes["arc_center_a"]}
    if inc == "b":
        return {"segment": axes["segment"][::-1].copy(), "arc_ctrl": axes["arc_center_a"], "arc_inc": axes["arc_center_b"]}
    raise ValueError(f"inc must be 'a' or 'b', got {inc!r}")


def axes_min_lab_distance(axes: dict, forbidden) -> float:
    """Smallest deltaE76 between any point of the three axes and any forbidden colour."""
    pts = np.concatenate([axes["segment"], axes["arc_center_b"], axes["arc_center_a"]])
    return float(lab_distances(pts, forbidden).min())
