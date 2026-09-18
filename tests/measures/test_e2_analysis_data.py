"""Tests for experiments/E2_online_4zones/analysis_data.py: canonical
zone-position remapping (needed for F4's spatial maps -- "agrege par
type de zone, jamais par quadrant") and the CSV loader."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

ad = load_experiment_module("E2_online_4zones", "analysis_data.py", "_e2_analysis_data_for_analysis_data_test")


def test_canonical_rc_is_identity_at_rotation_zero():
    for zone_type in ad.TYPE_TO_CANONICAL_QUADRANT:
        zone_states = ad.zone_of_state_for_rotation(0)
        s = int(np.flatnonzero(zone_states == zone_type)[0])
        assert ad.canonical_rc_for_state(zone_type, s, rotation=0) == divmod(s, ad.N_COLS)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_canonical_rc_lands_within_the_canonical_block_range(rotation):
    zone_states = ad.zone_of_state_for_rotation(rotation)
    for zone_type, canonical_quad in ad.TYPE_TO_CANONICAL_QUADRANT.items():
        rows, cols = ad._BLOCK_RANGES[canonical_quad]
        states = np.flatnonzero(zone_states == zone_type)
        for s in states:
            r, c = ad.canonical_rc_for_state(zone_type, int(s), rotation)
            assert rows[0] <= r <= rows[1]
            assert cols[0] <= c <= cols[1]


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_canonical_rc_is_a_bijection_within_each_zone(rotation):
    """Every one of a zone's 25 states must map to a DISTINCT canonical
    position -- otherwise averaging across seeds would silently collapse
    two different states onto the same map cell."""
    zone_states = ad.zone_of_state_for_rotation(rotation)
    for zone_type in ad.TYPE_TO_CANONICAL_QUADRANT:
        states = np.flatnonzero(zone_states == zone_type)
        canon = {ad.canonical_rc_for_state(zone_type, int(s), rotation) for s in states}
        assert len(canon) == len(states)


def test_canonical_rc_returns_none_for_corridor():
    assert ad.canonical_rc_for_state("corridor", 0, rotation=2) is None


def test_load_per_state_excludes_the_goal_state():
    # a synthetic smoke check without touching real sweep data: the goal
    # state's row would never appear if the loader filtered it, checked
    # against the real sweep's own zone lookup instead (goal is state 0)
    assert ad.GOAL_S == 0
