"""DynamicsSpec: the per-zone dynamics registry.

Four parametrized types (controllable/uncontrollable x deterministic/noisy)
live in controllable.py / uncontrollable.py. Each exposes its own
analytic_controllability() -- a closed form, no simulation -- alongside the
same transition_row() every other dynamics type implements, so that
measures.controllability.controllability_from_tensor (the one numeric
implementation, unchanged) can be run on the tensor a spec produces and
checked against the closed form. That agreement is asserted in
tests/envs/dynamics/test_analytic_agreement.py -- the same "ground truth
must agree with the general-purpose measure" pattern CLAUDE.md already
requires for GridWorld's transition_tensor()/step() pair.

All four types operate on a torus (see envs/torus_env.py): no walls, no
absorbing borders, positions wrap via modulo. ACTIONS/N_ACTIONS mirror the
legacy GridWorld's convention (0=up,1=right,2=down,3=left,4=stay) but are
defined independently here -- this package must not import the legacy
environment (see CLAUDE.md: no coupling between the frozen 5x5 variant and
the new family).
"""

from __future__ import annotations

import abc

import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))
ACTION_NAMES = ("up", "right", "down", "left", "stay")
N_ACTIONS = 5


def _wrap(rc: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    r, c = rc
    n_rows, n_cols = grid_shape
    return (r % n_rows, c % n_cols)


def _rc_to_s(rc: tuple[int, int], grid_shape: tuple[int, int]) -> int:
    return rc[0] * grid_shape[1] + rc[1]


class DynamicsSpec(abc.ABC):
    """One zone's dynamics. Stateless and immutable: a spec fully describes
    the rule, it does not hold or mutate any position."""

    @abc.abstractmethod
    def transition_row(self, rc: tuple[int, int], a: int, grid_shape: tuple[int, int]) -> np.ndarray:
        """Exact P(s' | rc, a) as a length-(n_rows*n_cols) array summing to
        1, on the torus defined by grid_shape (row-major flattening, same
        convention as the legacy GridWorld's rc_to_s)."""

    @abc.abstractmethod
    def analytic_controllability(self, n_actions: int = N_ACTIONS) -> float:
        """I(a; s' | s) in bits, in closed form, under a uniform action
        prior -- identical at every cell of a torus (no walls to break the
        symmetry), so this single number is exact for the whole zone,
        unlike the legacy GridWorld where walls forced a per-state,
        simulate-then-average computation."""
