"""Tests for experiments/E2_online_4zones/artifact.py: write/load
roundtrip for both tables, and the E1-format adapter's compatibility
with experiments/E1_offline_4zones/artifact.py::join_exp_ctrl (spec:
"garde... la compatibilite avec join_exp_ctrl()")."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

env_mod = load_experiment_module("E2_online_4zones", "env.py", "_e2_env_for_artifact_test")
twin_mod = load_experiment_module("E2_online_4zones", "twin.py", "_e2_twin_for_artifact_test")
measurement_mod = load_experiment_module("E2_online_4zones", "measurement.py", "_e2_measurement_for_artifact_test")
artifact_mod = load_experiment_module("E2_online_4zones", "artifact.py", "_e2_artifact_for_artifact_test")
e1_artifact_mod = load_experiment_module("E1_offline_4zones", "artifact.py", "_e1_artifact_for_e2_artifact_test")

from helplessness.config import load_yaml_block  # noqa: E402
from helplessness.envs.zones import geographic_support_mask  # noqa: E402

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5


def _twin_result_for(layout, P_true, P_ref, R_true, terminal, support, seed):
    rng = np.random.default_rng(seed)
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N[s, a] = rng.multinomial(5, P_true[s, a])
    return twin_mod.twin_from_counts(
        N=N, alpha=1.0, support=support, P_true=P_true, R_true=R_true, terminal=terminal,
        P_ref=P_ref, gamma=0.9, reference_rng=np.random.default_rng(seed + 1000),
    )


def _build_exp_ctrl_rows(seed=0, checkpoint=100, assignment=0):
    layout = env_mod.build_layout(CFG, DIRECTION)
    ctrl_layout = env_mod.build_control_layout(CFG, DIRECTION)
    P_true = env_mod.build_env(layout, CFG, goal_reward=50.0).transition_tensor()
    P_ref = env_mod.build_env(ctrl_layout, CFG, goal_reward=50.0).transition_tensor()
    R_true, terminal = env_mod.reward_and_terminal(CFG, goal_reward=50.0)
    support = geographic_support_mask(GRID_SHAPE)

    result_exp = _twin_result_for(layout, P_true, P_ref, R_true, terminal, support, seed)
    result_ctrl = _twin_result_for(ctrl_layout, P_ref, P_true, R_true, terminal, support, seed + 1)

    meta_exp = {"exp": "E2", "env": "exp", "condition": "matched", "alpha": 1.0, "epsilon": 0.1,
                "seed": seed, "assignment": assignment, "checkpoint": checkpoint}
    meta_ctrl = {**meta_exp, "env": "ctrl"}

    rows_exp = measurement_mod.per_state_rows(result_exp, layout, GRID_SHAPE, meta_exp)
    rows_ctrl = measurement_mod.per_state_rows(result_ctrl, ctrl_layout, GRID_SHAPE, meta_ctrl)
    pair_rows_exp = measurement_mod.per_pair_rows(result_exp, layout, GRID_SHAPE, N_ACTIONS, meta_exp)
    pair_rows_ctrl = measurement_mod.per_pair_rows(result_ctrl, ctrl_layout, GRID_SHAPE, N_ACTIONS, meta_ctrl)
    return rows_exp + rows_ctrl, pair_rows_exp + pair_rows_ctrl


def test_per_state_write_load_roundtrip(tmp_path):
    layout = env_mod.build_layout(CFG, DIRECTION)
    P_true = env_mod.build_env(layout, CFG, goal_reward=50.0).transition_tensor()
    R_true, terminal = env_mod.reward_and_terminal(CFG, goal_reward=50.0)
    support = geographic_support_mask(GRID_SHAPE)
    result = _twin_result_for(layout, P_true, P_true, R_true, terminal, support, seed=0)
    meta = {"exp": "E2", "env": "exp", "condition": "online", "alpha": 1.0, "epsilon": 0.1,
            "seed": 3, "assignment": 1, "checkpoint": 500}
    rows = measurement_mod.per_state_rows(result, layout, GRID_SHAPE, meta)

    path = artifact_mod.write_per_state_artifact(rows, tmp_path / "per_state.csv")
    loaded = artifact_mod.load_per_state_artifact(path)

    assert len(loaded) == len(rows)
    original_by_state = {r["state"]: r for r in rows}
    for row in loaded:
        original = original_by_state[row["state"]]
        assert row["I_hat"] == pytest.approx(original["I_hat"])
        assert row["delta"] == pytest.approx(original["delta"])
        assert row["seed"] == original["seed"]
        assert row["checkpoint"] == original["checkpoint"]


def test_per_pair_write_load_roundtrip(tmp_path):
    layout = env_mod.build_layout(CFG, DIRECTION)
    P_true = env_mod.build_env(layout, CFG, goal_reward=50.0).transition_tensor()
    R_true, terminal = env_mod.reward_and_terminal(CFG, goal_reward=50.0)
    support = geographic_support_mask(GRID_SHAPE)
    result = _twin_result_for(layout, P_true, P_true, R_true, terminal, support, seed=0)
    meta = {"exp": "E2", "env": "exp", "condition": "online", "alpha": 1.0, "epsilon": 0.1,
            "seed": 3, "assignment": 1, "checkpoint": 500}
    rows = measurement_mod.per_pair_rows(result, layout, GRID_SHAPE, N_ACTIONS, meta)

    path = artifact_mod.write_per_pair_artifact(rows, tmp_path / "per_pair.csv")
    loaded = artifact_mod.load_per_pair_artifact(path)

    assert len(loaded) == len(rows)
    original_by_key = {(r["state"], r["action"]): r for r in rows}
    for row in loaded:
        original = original_by_key[(row["state"], row["action"])]
        assert row["N"] == original["N"]
        assert row["l1"] == pytest.approx(original["l1"])
        assert row["is_optimal_under_P"] == original["is_optimal_under_P"]


def test_to_e1_zone_format_is_consumable_by_e1_join_exp_ctrl():
    """The actual compatibility requirement: E1's join_exp_ctrl, imported
    UNMODIFIED, must successfully join E2's derived zone-aggregate rows."""
    rows, pair_rows = _build_exp_ctrl_rows(seed=0, checkpoint=100, assignment=0)
    e1_rows = artifact_mod.to_e1_zone_format(rows, pair_rows)

    joined = e1_artifact_mod.join_exp_ctrl(e1_rows)
    assert len(joined) > 0
    # every zone/region/all-scope combination on the exp side found its ctrl match
    all_scope_exp_rows = [r for r in e1_rows if r["env"] == "exp" and r["region"] == "all"]
    assert len(joined) >= len(set((r["zone"], r["region"]) for r in e1_rows if r["env"] == "exp"))
    for row in joined:
        assert "ctrl_delta" in row
        assert "ctrl_I_hat" in row


def test_to_e1_zone_format_region_all_has_i_hat_i_star_b_populated():
    rows, pair_rows = _build_exp_ctrl_rows()
    e1_rows = artifact_mod.to_e1_zone_format(rows, pair_rows)
    for row in e1_rows:
        is_all = row["region"] == "all"
        i_hat_is_nan = row["I_hat"] != row["I_hat"]
        assert is_all != i_hat_is_nan, f"I_hat null-ness disagrees with region=='all' at {row}"
        b_is_nan = row["b"] != row["b"]
        should_have_b = is_all and row["env"] == "exp"
        assert should_have_b != b_is_nan, f"b null-ness disagrees with (region=='all', env=='exp') at {row}"


def test_to_e1_zone_format_passes_e1s_own_region_invariant_assertion():
    """E1's load_measurement_artifact asserts this invariant internally
    (_assert_region_all_invariant) -- writing E2's derived rows through
    E1's own writer/loader roundtrip must not raise."""
    import tempfile
    from pathlib import Path

    rows, pair_rows = _build_exp_ctrl_rows()
    e1_rows = artifact_mod.to_e1_zone_format(rows, pair_rows)
    with tempfile.TemporaryDirectory() as tmp:
        path = e1_artifact_mod.write_measurement_artifact(e1_rows, Path(tmp) / "e2_as_e1.csv")
        reloaded = e1_artifact_mod.load_measurement_artifact(path)  # raises on invariant violation
        assert len(reloaded) == len(e1_rows)
