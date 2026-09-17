"""Tests for experiments/E2_online_4zones/measurement.py: per_state_rows/
per_pair_rows build correctly-shaped, correctly-labeled rows from a
twin_from_counts() result."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

env_mod = load_experiment_module("E2_online_4zones", "env.py", "_e2_env_for_measurement_test")
twin_mod = load_experiment_module("E2_online_4zones", "twin.py", "_e2_twin_for_measurement_test")
measurement_mod = load_experiment_module("E2_online_4zones", "measurement.py", "_e2_measurement_for_measurement_test")

from helplessness.config import load_yaml_block  # noqa: E402
from helplessness.envs.zones import geographic_support_mask  # noqa: E402

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5


def _small_twin_result(alpha=1.0, n_per_pair=5, seed=0):
    layout = env_mod.build_layout(CFG, DIRECTION)
    ctrl_layout = env_mod.build_control_layout(CFG, DIRECTION)
    P_true = env_mod.build_env(layout, CFG, goal_reward=50.0).transition_tensor()
    P_ref = env_mod.build_env(ctrl_layout, CFG, goal_reward=50.0).transition_tensor()
    R_true, terminal = env_mod.reward_and_terminal(CFG, goal_reward=50.0)
    support = geographic_support_mask(GRID_SHAPE)

    rng = np.random.default_rng(seed)
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N[s, a] = rng.multinomial(n_per_pair, P_true[s, a])

    result = twin_mod.twin_from_counts(
        N=N, alpha=alpha, support=support, P_true=P_true, R_true=R_true, terminal=terminal,
        P_ref=P_ref, gamma=0.9, reference_rng=np.random.default_rng(seed + 1000),
    )
    return layout, result


META = {
    "exp": "E2", "env": "exp", "condition": "matched", "alpha": 1.0, "epsilon": 0.1,
    "seed": 0, "assignment": 0, "checkpoint": 100,
}


def test_per_state_rows_has_exactly_one_row_per_state():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_state_rows(result, layout, GRID_SHAPE, META)
    assert len(rows) == N_STATES
    assert sorted(r["state"] for r in rows) == list(range(N_STATES))


def test_per_state_rows_carries_meta_fields_and_matches_twin_values():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_state_rows(result, layout, GRID_SHAPE, META)
    row0 = next(r for r in rows if r["state"] == 0)
    assert row0["exp"] == "E2"
    assert row0["condition"] == "matched"
    assert row0["checkpoint"] == 100
    assert row0["I_hat"] == pytest.approx(result["I_hat"][0])
    assert row0["delta"] == pytest.approx(result["delta"][0])


def test_per_state_rows_region_is_interior_or_boundary_never_all():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_state_rows(result, layout, GRID_SHAPE, META)
    regions = {r["region"] for r in rows}
    assert regions <= {"interior", "boundary"}
    assert "all" not in regions


def test_per_state_rows_zone_type_matches_known_e1_4zone_categories():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_state_rows(result, layout, GRID_SHAPE, META)
    zones_seen = {r["zone"] for r in rows}
    expected = {"corridor", "uncontrollable_deterministic", "controllable_noisy",
                "controllable_deterministic", "uncontrollable_noisy"}
    assert zones_seen == expected


def test_per_pair_rows_has_states_times_actions_rows():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_pair_rows(result, layout, GRID_SHAPE, N_ACTIONS, META)
    assert len(rows) == N_STATES * N_ACTIONS


def test_per_pair_rows_n_matches_twin_n_target():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_pair_rows(result, layout, GRID_SHAPE, N_ACTIONS, META)
    row = next(r for r in rows if r["state"] == 3 and r["action"] == 2)
    assert row["N"] == int(result["N_target"][3, 2])


def test_per_pair_rows_is_optimal_under_p_has_at_least_one_true_per_state():
    layout, result = _small_twin_result()
    rows = measurement_mod.per_pair_rows(result, layout, GRID_SHAPE, N_ACTIONS, META)
    by_state = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r["is_optimal_under_P"])
    for s, flags in by_state.items():
        assert any(flags), f"state {s} has no action marked optimal under P"
