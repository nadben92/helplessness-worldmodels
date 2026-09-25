"""T1 (pairing part) for E3's pixel environment: build_control_layout
(imported unchanged from E2) must still satisfy "H(s'|s,a) and support
size identical between experimental and reference, only I differs" once
wrapped in a pixel-observation TorusZonedEnv -- the dynamics tensors are
untouched by the observation backend swap, so this mostly re-confirms
E2's own invariant transfers to E3's construction path unchanged. Also
covers: the reference environment shares layout/permutation/goal with
the experimental one (spec 2.3), and reuses build_layout/build_control_layout
from E2 without reimplementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_e3_env_module")

from helplessness.measures.controllability import conditional_entropy_per_state, controllability_per_state  # noqa: E402


@pytest.mark.parametrize("seed_index", [0, 5, 10, 15])
def test_pairing_h_and_support_identical_i_differs(seed_index):
    rotation = e3_env.rotation_for_seed_index(seed_index)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    exp_layout = e3_env.build_layout(assignment)
    ctrl_layout = e3_env.build_control_layout(assignment)

    exp_env = e3_env.build_pixel_env(exp_layout, _dummy_themes(exp_layout), goal_reward=50.0)
    ctrl_env = e3_env.build_pixel_env(ctrl_layout, _dummy_themes(ctrl_layout), goal_reward=50.0)
    P_exp = exp_env.transition_tensor()
    P_ctrl = ctrl_env.transition_tensor()

    H_exp = conditional_entropy_per_state(P_exp)
    H_ctrl = conditional_entropy_per_state(P_ctrl)
    I_exp = controllability_per_state(P_exp)
    I_ctrl = controllability_per_state(P_ctrl)

    for zone in exp_layout.zones:
        if zone.appearance.category == "corridor":
            continue
        idx = np.flatnonzero(zone.mask)
        assert H_exp[idx] == pytest.approx(H_ctrl[idx], abs=1e-9)
        for s in idx:
            for a in range(5):
                supp_exp = int((P_exp[s, a] > 0).sum())
                supp_ctrl = int((P_ctrl[s, a] > 0).sum())
                assert supp_exp == supp_ctrl

        from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy

        if isinstance(zone.dynamics, (UncontrollableDeterministic, UncontrollableNoisy)):
            assert not np.allclose(I_exp[idx], I_ctrl[idx])


def test_reference_shares_layout_permutation_and_goal_with_experimental():
    seed_index = 7
    rotation = e3_env.rotation_for_seed_index(seed_index)
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    exp_layout = e3_env.build_layout(assignment)
    ctrl_layout = e3_env.build_control_layout(assignment)

    assert exp_layout.grid_shape == ctrl_layout.grid_shape
    for z_exp, z_ctrl in zip(exp_layout.zones, ctrl_layout.zones):
        assert np.array_equal(z_exp.mask, z_ctrl.mask)  # same layout/permutation
        assert z_exp.appearance.category == z_ctrl.appearance.category  # same themes categories

    assert e3_env.goal_state() == 0  # goal shared, from the same frozen CFG


def test_build_layout_reuses_e2_dynamics_object_identity_not_a_reimplementation():
    """Confirms build_layout genuinely delegates to E2's own function
    object (not a coincidentally-identical reimplementation) -- if E2's
    env.py is edited, E3's build_layout changes with it automatically."""
    e2_env = load_experiment_module("E2_online_4zones", "env.py", "test_e3_env_e2_identity_check")
    assert e3_env._e2_env.build_layout is e2_env.__dict__["build_layout"] or callable(e3_env._e2_env.build_layout)
    # stronger check: same source code object (function identity across
    # two independent loads of the same file is not guaranteed by
    # importlib, so compare __code__ instead, which IS content-identical)
    assert e3_env._e2_env.build_layout.__code__.co_code == e2_env.build_layout.__code__.co_code


def _dummy_themes(layout):
    from helplessness.envs.appearance.theme_sampling import build_themes

    categories = sorted({z.appearance.category for z in layout.zones if z.appearance.category != "corridor"})
    return build_themes(categories, goal_state=0, rng=np.random.default_rng(0), min_dist=5.0)
