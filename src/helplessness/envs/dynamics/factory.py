"""Builds a DynamicsSpec from a plain dict (as loaded from YAML by
helplessness.config.compose()) -- the bridge between configs/ data and the
Python objects in this package. One explicit registry, not
getattr/eval-based type lookup, so an unknown "type" fails loudly rather
than doing something clever and wrong.
"""

from __future__ import annotations

from helplessness.envs.dynamics.base import DynamicsSpec
from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy

_REGISTRY = {
    "ControllableDeterministic": ControllableDeterministic,
    "ControllableNoisy": ControllableNoisy,
    "UncontrollableDeterministic": UncontrollableDeterministic,
    "UncontrollableNoisy": UncontrollableNoisy,
}


def dynamics_from_config(cfg: dict) -> DynamicsSpec:
    cfg = dict(cfg)
    type_name = cfg.pop("type")
    if type_name not in _REGISTRY:
        raise ValueError(f"unknown dynamics type {type_name!r}, expected one of {sorted(_REGISTRY)}")
    return _REGISTRY[type_name](**cfg)
