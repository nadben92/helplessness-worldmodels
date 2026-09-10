"""Automatic derivation of the paired control: renders an uncontrollable
zone's dynamics controllable, deterministically and identically every
time, so a control environment is never a second hand-written config that
could drift from the one it is meant to mirror (see CLAUDE.md: the pairing
must be a transformation of the experimental config, applied once).

Convention: any uncontrollable dynamics -- fixed drift or noisy, it does
not matter which -- pairs to full, clean, deterministic control
(ControllableDeterministic), never to a noise-matched controllable variant.
Rationale: "make it controllable" should give the agent a clean reference
condition to contrast against, the same role p=0 played throughout the
offline study; a noise-matched pairing would still leave a bias (the
noise) confounded with the change being tested. Controllable zones are
left untouched -- pairing only ever changes uncontrollable ones.
"""

from __future__ import annotations

from helplessness.envs.dynamics.base import DynamicsSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy


def derive_control(spec: DynamicsSpec) -> DynamicsSpec:
    if isinstance(spec, (UncontrollableDeterministic, UncontrollableNoisy)):
        return ControllableDeterministic()
    return spec
