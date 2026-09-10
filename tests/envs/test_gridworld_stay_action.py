"""Tests for the optional 5th "stay" action (n_actions=5), added for the
online experiment. Kept separate from test_gridworld.py so the offline
environment's own test file stays untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.legacy.gridworld import N_ACTIONS, GridWorld


def test_default_n_actions_is_unchanged():
    """No n_actions argument -> exactly the offline default (4)."""
    env = GridWorld(p=0.3, seed=0)
    assert env.n_actions == 4 == N_ACTIONS


def test_invalid_n_actions_raises():
    with pytest.raises(ValueError):
        GridWorld(n_actions=3)
    with pytest.raises(ValueError):
        GridWorld(n_actions=6)


def test_stay_action_keeps_agent_in_place_when_not_corrupted():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    env.reset()
    before = env.agent_rc
    _, r, terminated, truncated, info = env.step(4)  # 4 = stay
    assert env.agent_rc == before
    assert not info["slipped"]
    assert r == pytest.approx(env.step_reward)


def test_model_simulation_agreement_with_stay_action():
    """Same check as test_gridworld.test_model_simulation_agreement, but for
    the 5-action environment: transition_tensor() must stay the exact law
    that _sample_effective_action() induces once "stay" is enabled."""
    n_samples = 6000
    tol = 0.03

    for p in (0.0, 0.3, 0.7, 1.0):
        env = GridWorld(p=p, seed=0, n_actions=5)
        env.seed_dynamics(123)
        P = env.transition_tensor()

        non_terminal = [
            s
            for s in range(env.n_states)
            if not (env.walls[env.s_to_rc(s)] or env.goals[env.s_to_rc(s)] or env.traps[env.s_to_rc(s)])
        ]

        for s in non_terminal[:3]:
            rc = env.s_to_rc(s)
            for a in range(env.n_actions):
                counts = np.zeros(env.n_states)
                for _ in range(n_samples):
                    env.agent_rc = rc
                    env.t = 0
                    _, _, _, _, info = env.step(a)
                    counts[info["state"]] += 1
                empirical = counts / n_samples
                assert np.allclose(empirical, P[s, a], atol=tol), (
                    f"p={p} s={s} a={a}: empirical={empirical} vs tensor={P[s, a]}"
                )


def test_controllability_bounds_and_monotone_5_actions():
    env = GridWorld(p=0.0, seed=0, n_actions=5)
    assert env.controllability() == pytest.approx(1.677, abs=0.01)

    env_full = GridWorld(p=1.0, seed=0, n_actions=5)
    assert env_full.controllability() == pytest.approx(0.0, abs=1e-9)

    p_values = np.linspace(0.0, 1.0, 11)
    values = [GridWorld(p=p, seed=0, n_actions=5).controllability() for p in p_values]
    diffs = np.diff(values)
    assert np.all(diffs <= 1e-9), f"controllability non-monotone: {values}"


def test_transition_tensor_rows_sum_to_one_5_actions():
    for p in (0.0, 0.5, 1.0):
        env = GridWorld(p=p, seed=0, n_actions=5)
        P = env.transition_tensor()
        assert P.shape[1] == 5
        assert np.allclose(P.sum(axis=2), 1.0)
