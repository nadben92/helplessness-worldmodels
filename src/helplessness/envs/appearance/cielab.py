"""Self-contained sRGB -> CIELAB conversion (D65 white point, 2-degree
observer), numpy only -- no new dependency (see docs/e3/DECISIONS.md,
point 6). Standard formulas (IEC 61966-2-1 sRGB companding, CIE 1976
L*a*b*), tested against published reference sRGB/Lab pairs in
tests/envs/appearance/test_cielab.py.

Used by theme_sampling.py to enforce a minimum perceptual distance
between zone theme colours (spec section 2.5) -- this module has no
knowledge of zones, themes, or runs; it is pure colour math.
"""

from __future__ import annotations

import numpy as np

# sRGB -> linear-RGB -> XYZ (D65), IEC 61966-2-1 matrix.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

# Reference white: the matrix's OWN implied white point (its row sums,
# i.e. what (1,1,1) linear-RGB maps to) rather than an independently
# rounded D65 constant -- the two agree to ~1e-5 but not to float
# precision, and using the matrix's own white point is what makes
# srgb_to_lab([1,1,1]) == [100, 0, 0] exactly (see test_cielab.py), the
# self-consistency property CIELAB is defined to have.
_WHITE_XYZ = _SRGB_TO_XYZ @ np.array([1.0, 1.0, 1.0])

_DELTA = 6.0 / 29.0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA**3, np.cbrt(t), t / (3 * _DELTA**2) + 4.0 / 29.0)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) array in [0, 1] (sRGB, gamma-encoded). Returns
    (..., 3) array [L*, a*, b*], L* in [0, 100]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = _srgb_to_linear(rgb)
    xyz = linear @ _SRGB_TO_XYZ.T
    xyz_n = xyz / _WHITE_XYZ
    fx, fy, fz = _f(xyz_n[..., 0]), _f(xyz_n[..., 1]), _f(xyz_n[..., 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Euclidean distance in CIELAB space (deltaE76) -- the simplest,
    standard perceptual-distance metric on L*a*b*; sufficient for the
    "reasonably distinct hues" requirement here (no need for CIEDE2000's
    hue/chroma weighting at this coarse a granularity)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))
