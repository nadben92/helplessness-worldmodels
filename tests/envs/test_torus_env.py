"""Tests for TorusZonedEnv: the model/simulation agreement check CLAUDE.md
requires of any environment's dynamics, torus wraparound, zone routing,
both observation backends, and interop with the existing shared evaluate()/
value_iteration() (proving the Env interface really is enough).
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.observation.index_backend import IndexObservationBackend
from helplessness.envs.observation.pixel_backend import PixelObservationBackend
from helplessness.envs.rewards.base import RewardFn, ZeroReward
from helplessness.envs.torus_env import TorusZonedEnv
from helplessness.envs.zones import Zone, ZoneLayout
from helplessness.evaluation.rollout import evaluate
from helplessness.planning.value_iteration import value_iteration

GRID_SHAPE = (4, 5)


def _two_zone_layout(top_dynamics, bottom_dynamics):
    n_rows, n_cols = GRID_SHAPE
    top = np.zeros(GRID_SHAPE, dtype=bool)
    top[: n_rows // 2, :] = True
    bottom = ~top
    return ZoneLayout(
        GRID_SHAPE,
        [
            Zone(mask=top, dynamics=top_dynamics, appearance=AppearanceSpec("top")),
            Zone(mask=bottom, dynamics=bottom_dynamics, appearance=AppearanceSpec("bottom")),
        ],
    )


def _make_env(layout, backend=None, seed=0):
    return TorusZonedEnv(
        zone_layout=layout,
        observation_backend=backend or IndexObservationBackend(),
        reward_fn=ZeroReward(),
        start_position=(0, 0),
        max_steps=20,
        seed=seed,
    )


def test_model_simulation_agreement():
    """The load-bearing test: transition_tensor() must be exactly the law
    step() induces -- same discipline as the legacy GridWorld's own test."""
    layout = _two_zone_layout(ControllableNoisy(p=0.4), UncontrollableDeterministic(drift=(0, 1)))
    env = _make_env(layout)
    P = env.transition_tensor()

    n_samples = 4000
    tol = 0.03
    for rc in [(0, 0), (1, 2), (2, 0), (3, 3)]:
        s = env.rc_to_s(rc)
        for a in range(env.n_actions):
            counts = np.zeros(env.n_states)
            for _ in range(n_samples):
                env.position = rc
                env.t = 0
                _, _, _, _, info = env.step(a)
                counts[info["state"]] += 1
            empirical = counts / n_samples
            assert np.allclose(empirical, P[s, a], atol=tol), f"s={s} a={a}: {empirical} vs {P[s, a]}"


def test_torus_wraparound_no_walls():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    env.position = (0, 0)
    env.t = 0
    obs, _r, _term, _trunc, info = env.step(0)  # "up" from row 0 wraps to last row
    assert info["state"] == env.rc_to_s((GRID_SHAPE[0] - 1, 0))


def test_zone_routes_to_correct_dynamics():
    layout = _two_zone_layout(UncontrollableDeterministic(drift=(1, 0)), ControllableDeterministic())
    env = _make_env(layout)
    # top zone: action ignored, fixed drift
    env.position = (0, 0)
    env.t = 0
    _, _, _, _, info = env.step(3)  # "left", irrelevant under the conveyor
    assert info["state"] == env.rc_to_s((1, 0))
    # bottom zone: action determines outcome
    env.position = (GRID_SHAPE[0] - 1, 0)
    env.t = 0
    _, _, _, _, info = env.step(1)  # "right"
    assert info["state"] == env.rc_to_s((GRID_SHAPE[0] - 1, 1))


def test_no_terminal_states_by_default():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    assert not env.terminal_mask().any()


def test_truncation_at_max_steps():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    env.max_steps = 3
    env.reset()
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(4)  # stay
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(4)
    assert not terminated and truncated


def test_index_backend_returns_flat_state():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout, backend=IndexObservationBackend())
    env.position = (2, 3)
    obs = env._obs()
    assert obs == env.rc_to_s((2, 3))


def test_pixel_backend_returns_colored_grid_with_agent_marker():
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout, backend=PixelObservationBackend())
    env.position = (0, 0)
    obs = env._obs()
    assert obs.shape == (GRID_SHAPE[0], GRID_SHAPE[1], 3)
    assert np.allclose(obs[0, 0], (1.0, 1.0, 1.0))  # agent marker overrides zone colour
    assert not np.allclose(obs[GRID_SHAPE[0] - 1, 0], (1.0, 1.0, 1.0))  # elsewhere, zone colour


def test_reward_fn_is_pluggable():
    class ConstantReward(RewardFn):
        def __call__(self, s: int) -> float:
            return 7.0

    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    env.reward_fn = ConstantReward()
    assert np.all(env.reward_vector() == 7.0)
    env.reset()
    _, r, _, _, _ = env.step(4)
    assert r == 7.0


def test_controllability_reflects_zone_mix():
    """Full control everywhere -> ceiling log2(n_actions); full
    uncontrollable everywhere -> exactly 0 -- reusing the one existing
    measure, unmodified."""
    all_controllable = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env_hi = _make_env(all_controllable)
    assert env_hi.controllability() == pytest.approx(np.log2(env_hi.n_actions), abs=1e-6)

    all_uncontrollable = _two_zone_layout(
        UncontrollableDeterministic(drift=(1, 0)), UncontrollableDeterministic(drift=(0, 1))
    )
    env_lo = _make_env(all_uncontrollable)
    assert env_lo.controllability() == pytest.approx(0.0, abs=1e-9)


def test_interop_with_existing_evaluate_and_value_iteration():
    """The whole point of sharing the Env ABC: value_iteration() and
    evaluate(), written for the legacy GridWorld, must work unchanged."""
    layout = _two_zone_layout(ControllableDeterministic(), ControllableDeterministic())
    env = _make_env(layout)
    T, R, terminal = env.transition_tensor(), env.reward_vector(), env.terminal_mask()
    V, policy = value_iteration(T, R, terminal, gamma=0.9)
    assert V.shape == (env.n_states,)
    summary, traj = evaluate(env, policy, n_episodes=5, seed=1)
    assert "return_mean" in summary
    assert len(traj) == 5
