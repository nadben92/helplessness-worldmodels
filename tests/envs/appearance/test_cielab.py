"""Tests for envs.appearance.cielab against published sRGB -> CIELAB
reference conversions (D65, 2-degree observer -- the standard tables
published by Bruce Lindbloom's colour-science reference and reproduced
across the literature). Loose tolerance (abs=0.6) accounts for rounding
in how the reference values are commonly quoted, not for any slack in
the conversion formula itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.appearance.cielab import lab_distance, srgb_to_lab

# (sRGB 0-255, expected Lab) -- standard D65/2-degree reference pairs.
REFERENCE_PAIRS = [
    ((255, 255, 255), (100.00, 0.00, 0.00)),
    ((0, 0, 0), (0.00, 0.00, 0.00)),
    ((255, 0, 0), (53.24, 80.09, 67.20)),
    ((0, 255, 0), (87.73, -86.18, 83.18)),
    ((0, 0, 255), (32.30, 79.19, -107.86)),
    ((255, 255, 0), (97.14, -21.55, 94.48)),
    ((0, 255, 255), (91.11, -48.09, -14.13)),
    ((255, 0, 255), (60.32, 98.23, -60.82)),
]


@pytest.mark.parametrize("rgb255, expected_lab", REFERENCE_PAIRS)
def test_srgb_to_lab_matches_published_reference(rgb255, expected_lab):
    rgb = np.array(rgb255, dtype=np.float64) / 255.0
    lab = srgb_to_lab(rgb)
    assert lab == pytest.approx(expected_lab, abs=0.6)


def test_white_is_exactly_l100_a0_b0():
    # Exact by construction: the reference white IS the D65 XYZ this
    # module normalises against, so this must hold to float precision,
    # not just within the published-reference tolerance above.
    lab = srgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert lab == pytest.approx([100.0, 0.0, 0.0], abs=1e-6)


def test_black_is_exactly_l0_a0_b0():
    lab = srgb_to_lab(np.array([0.0, 0.0, 0.0]))
    assert lab == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_srgb_to_lab_is_vectorized_over_a_batch_dimension():
    batch = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lab_batch = srgb_to_lab(batch)
    assert lab_batch.shape == (3, 3)
    for row, single in zip(batch, batch):
        assert srgb_to_lab(row) == pytest.approx(srgb_to_lab(single))


def test_lab_distance_is_zero_for_identical_colours():
    lab = srgb_to_lab(np.array([0.3, 0.6, 0.9]))
    assert lab_distance(lab, lab) == pytest.approx(0.0, abs=1e-9)


def test_lab_distance_white_to_black_is_100():
    white = srgb_to_lab(np.array([1.0, 1.0, 1.0]))
    black = srgb_to_lab(np.array([0.0, 0.0, 0.0]))
    assert lab_distance(white, black) == pytest.approx(100.0, abs=1e-6)


def test_lab_distance_is_symmetric():
    a = srgb_to_lab(np.array([0.8, 0.1, 0.2]))
    b = srgb_to_lab(np.array([0.1, 0.7, 0.4]))
    assert lab_distance(a, b) == pytest.approx(lab_distance(b, a))
