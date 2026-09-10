"""Consistency tests for helplessness.envs.legacy.gridworld."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.legacy.gridworld import N_ACTIONS, GridWorld


# ----------------------------------------------------------------------
# model / simulation agreement -- the most important one: checks that
# transition_tensor() is indeed the law that _sample_effective_action() induces.
# ----------------------------------------------------------------------
def test_model_simulation_agreement():
    n_samples = 6000
    tol = 0.03  # ~4 standard deviations for n=6000 (max std ~ 0.5/sqrt(n) ~= 0.0065)

    for p in (0.0, 0.3, 0.7, 1.0):
        env = GridWorld(p=p, seed=0)
        env.seed_dynamics(123)
        P = env.transition_tensor()

        non_terminal = [
            s
            for s in range(env.n_states)
            if not (env.walls[env.s_to_rc(s)] or env.goals[env.s_to_rc(s)] or env.traps[env.s_to_rc(s)])
        ]

        for s in non_terminal[:3]:
            rc = env.s_to_rc(s)
            for a in range(N_ACTIONS):
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


# ----------------------------------------------------------------------
# controllability: bounds and monotonicity
# ----------------------------------------------------------------------
def test_controllability_bounds_and_monotone():
    env = GridWorld(p=0.0, seed=0)
    assert env.controllability() == pytest.approx(1.78, abs=0.05)

    env_full = GridWorld(p=1.0, seed=0)
    assert env_full.controllability() == pytest.approx(0.0, abs=1e-9)

    p_values = np.linspace(0.0, 1.0, 11)
    values = [GridWorld(p=p, seed=0).controllability() for p in p_values]
    diffs = np.diff(values)
    assert np.all(diffs <= 1e-9), f"controllability non-monotone: {values}"


# ----------------------------------------------------------------------
# consistency between the two trap modes
# ----------------------------------------------------------------------
def test_terminal_traps_modes_consistency():
    env_term = GridWorld(p=0.2, terminal_traps=True, seed=0)
    env_free = GridWorld(p=0.2, terminal_traps=False, seed=0)

    trap_s = next(s for s in range(env_term.n_states) if env_term.traps[env_term.s_to_rc(s)])

    # terminal_mask
    assert env_term.terminal_mask()[trap_s]
    assert not env_free.terminal_mask()[trap_s]

    # absorption in the tensor
    P_term = env_term.transition_tensor()
    P_free = env_free.transition_tensor()
    assert P_term[trap_s, :, trap_s].min() == pytest.approx(1.0)
    assert not np.allclose(P_free[trap_s, 0], np.eye(env_free.n_states)[trap_s])

    # reward_vector is unchanged by the mode
    assert np.array_equal(env_term.reward_vector(), env_free.reward_vector())


# ----------------------------------------------------------------------
# determinism of the noise stream
# ----------------------------------------------------------------------
def test_dynamics_determinism():
    actions = [0, 1, 2, 3, 0, 1, 1, 2, 3, 0]

    def run(seed):
        env = GridWorld(p=0.5, seed=0)
        env.seed_dynamics(seed)
        env.reset()
        states, slips = [], []
        for a in actions:
            _, _, term, trunc, info = env.step(a)
            states.append(info["state"])
            slips.append(info["slipped"])
            if term or trunc:
                env.reset()
        return states, slips

    states1, slips1 = run(777)
    states2, slips2 = run(777)
    assert states1 == states2
    assert slips1 == slips2

    states3, _ = run(778)
    assert states1 != states3  # checks that the test is not trivially always equal


# ----------------------------------------------------------------------
# layout validation
# ----------------------------------------------------------------------
def test_layout_no_start_raises():
    layout = "....\n.##.\n....\n"
    with pytest.raises(ValueError):
        GridWorld(layout=layout)


def test_layout_two_starts_raises():
    layout = "S...\n.##.\n...S\n"
    with pytest.raises(ValueError):
        GridWorld(layout=layout)


def test_layout_unknown_char_raises():
    layout = "S...\n.##.\n...G\n..x.\n"
    with pytest.raises(ValueError):
        GridWorld(layout=layout)


@pytest.mark.skip(
    reason=(
        "Not reachable via the public constructor: the layout stores a "
        "single character per cell (list(row)), so a cell can never be "
        "both 'S' and '#'. The self.walls[self.start_rc] guard in "
        "__init__ is therefore dead code with the current layout format. "
        "Flagged to the user rather than artificially simulated."
    )
)
def test_start_on_wall_raises():
    pass
