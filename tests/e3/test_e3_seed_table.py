"""E3 reuses E2's seed -> quadrant-permutation table exactly (spec
decision, docs/e3/DECISIONS.md point 5), so that E3's seed k is directly
comparable to E2's seed k. This is the "frozen values" test point C
asked for: a hardcoded copy of that table, compared against what
experiments/E3_worldmodel_4zones/env.py actually re-exports (which
itself imports experiments/E2_online_4zones/env.py unchanged) -- if
E2's counterbalancing ever changed, this test would catch the drift
immediately rather than silently propagating a different table into E3.

The hardcoded values below were captured directly from
E2's rotation_for_seed_index(seed, seeds_per_rotation=5) and
quadrant_assignment_for_rotation(rotation), for seeds 0..19, and
verified separately (see docs/e3/DECISIONS.md point 5) to be a balanced
cyclic Latin square: every (dynamics type, quadrant) pair occurs in
exactly 5 of the 20 seeds.
"""

from __future__ import annotations

import pytest

from _experiment_loader import load_experiment_module

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_e3_seed_table_env")

# seed_index -> (rotation, {target_quadrant: source_quadrant})
FROZEN_SEED_TABLE = {
    0: (0, {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"}),
    1: (0, {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"}),
    2: (0, {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"}),
    3: (0, {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"}),
    4: (0, {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"}),
    5: (1, {"nw": "ne", "ne": "se", "se": "sw", "sw": "nw"}),
    6: (1, {"nw": "ne", "ne": "se", "se": "sw", "sw": "nw"}),
    7: (1, {"nw": "ne", "ne": "se", "se": "sw", "sw": "nw"}),
    8: (1, {"nw": "ne", "ne": "se", "se": "sw", "sw": "nw"}),
    9: (1, {"nw": "ne", "ne": "se", "se": "sw", "sw": "nw"}),
    10: (2, {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}),
    11: (2, {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}),
    12: (2, {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}),
    13: (2, {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}),
    14: (2, {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}),
    15: (3, {"nw": "sw", "ne": "nw", "se": "ne", "sw": "se"}),
    16: (3, {"nw": "sw", "ne": "nw", "se": "ne", "sw": "se"}),
    17: (3, {"nw": "sw", "ne": "nw", "se": "ne", "sw": "se"}),
    18: (3, {"nw": "sw", "ne": "nw", "se": "ne", "sw": "se"}),
    19: (3, {"nw": "sw", "ne": "nw", "se": "ne", "sw": "se"}),
}


@pytest.mark.parametrize("seed_index", range(20))
def test_e3_seed_table_matches_hardcoded_frozen_copy(seed_index):
    rotation = e3_env.rotation_for_seed_index(seed_index)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    expected_rotation, expected_assignment = FROZEN_SEED_TABLE[seed_index]
    assert rotation == expected_rotation
    assert assignment == expected_assignment


def test_e3_seed_table_is_a_balanced_cyclic_latin_square():
    """Every (dynamics-type-position "source", target quadrant) pair
    occurs in exactly 5 of the 20 seeds -- each zone type visits each
    geometric location the same number of times across the seed set."""
    from collections import defaultdict

    counts = defaultdict(lambda: defaultdict(int))
    for seed_index in range(20):
        rotation = e3_env.rotation_for_seed_index(seed_index)
        assignment = e3_env.quadrant_assignment_for_rotation(rotation)
        for target, source in assignment.items():
            counts[target][source] += 1

    for target in e3_env.QUADRANT_ORDER:
        for source in e3_env.QUADRANT_ORDER:
            assert counts[target][source] == 5, f"{target} <- {source}: got {counts[target][source]}, expected 5"


def test_e3_env_re_exports_the_same_quadrant_order_as_e2():
    e2_env = load_experiment_module("E2_online_4zones", "env.py", "test_e3_seed_table_e2_env")
    assert e3_env.QUADRANT_ORDER == e2_env.QUADRANT_ORDER
