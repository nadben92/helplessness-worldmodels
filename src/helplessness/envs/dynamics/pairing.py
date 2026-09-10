"""Automatic derivation of the paired control: renders an uncontrollable
zone's dynamics controllable, deterministically and identically every
time, so a control environment is never a second hand-written config that
could drift from the one it is meant to mirror (see CLAUDE.md: the pairing
must be a transformation of the experimental config, applied once).

Convention: the noise LEVEL is preserved, only who determines the "clean"
outcome changes -- the agent's action, instead of a fixed drift.
UncontrollableDeterministic(drift) (no noise at all) pairs to
ControllableDeterministic() (no noise at all). UncontrollableNoisy(drift, p)
pairs to ControllableNoisy(p) -- same p, because the two share the exact
same mixture structure by construction (see uncontrollable.py's
docstring): the only thing that changes is whether the (1-p) "clean" mass
goes to a fixed drift or to the chosen action. A pairing that collapsed
every uncontrollable zone to the same clean ControllableDeterministic()
regardless of how noisy it was would confound "made controllable" with
"also denoised" -- two different manipulations. Controllable zones are
left untouched -- pairing only ever changes uncontrollable ones.
"""

from __future__ import annotations

from helplessness.envs.dynamics.base import DynamicsSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy


def derive_control(spec: DynamicsSpec) -> DynamicsSpec:
    if isinstance(spec, UncontrollableDeterministic):
        return ControllableDeterministic()
    if isinstance(spec, UncontrollableNoisy):
        return ControllableNoisy(p=spec.p)
    return spec
