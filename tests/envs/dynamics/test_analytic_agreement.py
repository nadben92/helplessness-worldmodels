"""The load-bearing test for the dynamics registry: each DynamicsSpec's
analytic_controllability() (closed form) must agree with the single,
general-purpose numeric measure (measures.controllability_from_tensor)
applied to the exact tensor transition_row() produces -- the same
"ground truth must agree with the general-purpose measure" pattern
CLAUDE.md requires of GridWorld's transition_tensor()/step() pair.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.dynamics import (
    ControllableDeterministic,
    ControllableNoisy,
    UncontrollableDeterministic,
    UncontrollableNoisy,
    N_ACTIONS,
)
from helplessness.measures.controllability import controllability_from_tensor

GRID_SHAPE = (6, 7)  # deliberately non-square, no degenerate torus wraparound


def _numeric_controllability(spec, grid_shape=GRID_SHAPE, n_actions=N_ACTIONS) -> float:
    """Builds the exact (S, A, S) tensor for every cell of the torus and
    runs it through the one general-purpose measure -- no cell is special
    on a torus (no walls), but we build the full tensor anyway rather than
    assume that, exactly the discipline transition_tensor() follows."""
    n_rows, n_cols = grid_shape
    n_states = n_rows * n_cols
    P = np.zeros((n_states, n_actions, n_states))
    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            for a in range(n_actions):
                P[s, a] = spec.transition_row((r, c), a, grid_shape)
    terminal = np.zeros(n_states, dtype=bool)  # torus: nothing is absorbing
    return controllability_from_tensor(P, terminal)


@pytest.mark.parametrize(
    "spec",
    [
        ControllableDeterministic(),
        ControllableNoisy(p=0.0),
        ControllableNoisy(p=0.3),
        ControllableNoisy(p=0.7),
        ControllableNoisy(p=1.0),
        UncontrollableDeterministic(drift=(1, 0)),
        UncontrollableDeterministic(drift=(0, -1)),
        UncontrollableNoisy(),
        UncontrollableNoisy(move_probs=np.array([0.4, 0.3, 0.1, 0.1, 0.1])),
    ],
    ids=[
        "controllable_det", "controllable_noisy_p0", "controllable_noisy_p0.3",
        "controllable_noisy_p0.7", "controllable_noisy_p1", "uncontrollable_det_drift(1,0)",
        "uncontrollable_det_drift(0,-1)", "uncontrollable_noisy_uniform", "uncontrollable_noisy_skewed",
    ],
)
def test_analytic_matches_numeric(spec):
    analytic = spec.analytic_controllability(N_ACTIONS)
    numeric = _numeric_controllability(spec)
    assert analytic == pytest.approx(numeric, abs=1e-9)


def test_transition_rows_sum_to_one_and_are_nonnegative():
    specs = [
        ControllableDeterministic(),
        ControllableNoisy(p=0.4),
        UncontrollableDeterministic(drift=(1, 1)),
        UncontrollableNoisy(),
    ]
    for spec in specs:
        for a in range(N_ACTIONS):
            row = spec.transition_row((2, 3), a, GRID_SHAPE)
            assert row.min() >= 0.0
            assert row.sum() == pytest.approx(1.0)


def test_torus_wraps_at_borders():
    spec = ControllableDeterministic()
    n_rows, n_cols = GRID_SHAPE
    # "up" from row 0 must wrap to the last row, not clip/self-loop
    row = spec.transition_row((0, 3), a=0, grid_shape=GRID_SHAPE)
    expected_s = (n_rows - 1) * n_cols + 3
    assert row[expected_s] == pytest.approx(1.0)
