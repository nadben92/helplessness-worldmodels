"""E6's localisation measures (helplessness.measures.localisation_errors) on hand-built episodes."""

from __future__ import annotations

import numpy as np

from helplessness.measures import localisation_errors as le

# 1-D toy: states 0..5, the zone {2, 3, 4}, the goal at 0; the conveyor pushes +1 (away from the goal)
ZONE = np.array([False, False, True, True, True, False])
DIST = np.arange(6)
CONVEYOR = np.minimum(np.arange(6) + 1, 5)
NEIGH = np.array([[(s - 1) % 6, (s + 1) % 6, (s - 1) % 6, (s + 1) % 6] for s in range(6)])


def test_steps_split_by_localisation():
    states = np.array([[4, 3, 2, 1, -1]])
    length = np.array([3])
    decoded = np.array([[4.0, 4.0, 2.0, np.nan]])  # right, wrong, right
    out = le.steps_by_localisation(states, length, ZONE, DIST, decoded)
    assert out["n_localised"] == 2 and out["directed_localised"] == 1.0
    assert out["n_mislocalised"] == 1 and out["directed_mislocalised"] == 1.0


def test_errors_on_the_conveyor_cell():
    # t = 1: from 3 the agent went to 2; the conveyor would have taken it to 4; decoded 4 -> on the conveyor
    # t = 2 (2 -> 3) and t = 3 (3 -> 4) are the conveyor's own moves: errors (decoded 4, 3), left out of the conveyor share
    states = np.array([[3, 2, 3, 4, 5]])
    decoded = np.array([[3.0, 4.0, 4.0, 3.0]])
    out = le.error_nature(states, np.array([4]), ZONE, decoded, CONVEYOR, NEIGH)
    assert out["n_errors"] == 3 and out["n_eligible"] == 1 and out["share_on_conveyor"] == 1.0
    assert out["n_neighbour_test"] == 0  # conveyor[3] = 4 is not a neighbour of 2 in this toy
    assert out["chance_among_neighbours"] == 0.25 and out["share_on_neighbour"] == 2 / 3  # t = 2 and t = 3


def test_wilson_and_neighbours():
    assert le.wilson(0, 0) != le.wilson(0, 0)  # NaN
    lo, hi = le.wilson(50, 100)
    assert lo < 0.5 < hi
    n = le.torus_neighbours((12, 12))
    assert n[0].tolist() == [132, 1, 12, 11]


def test_homologous_cells_of_other_quadrants():
    zone = np.zeros(144, dtype=bool)
    zone[[13, 14]] = True                                   # (1, 1), (1, 2)
    states = np.array([[13, 14, 13, 13]])
    decoded = np.array([[13.0, 14 + 6 * 12 + 6, 13 + 6]])  # t = 1 -> (7, 8): diagonal; t = 2 -> (1, 7): right
    out = le.error_nature(states, np.array([3]), zone, decoded, np.arange(144), le.torus_neighbours((12, 12)))
    assert out["n_errors"] == 2 and out["share_on_homologous_diagonal"] == 0.5 and out["share_on_homologous_right"] == 0.5
    regions = np.where(np.arange(144) // 12 < 6, "north", "south")
    out = le.error_nature(states, np.array([3]), zone, decoded, np.arange(144), le.torus_neighbours((12, 12)), regions=regions)
    assert out["share_in_north"] == 0.5 and out["share_in_south"] == 0.5
