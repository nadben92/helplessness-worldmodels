"""Tests for dynamics_from_config: the configs/ YAML -> DynamicsSpec bridge."""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.envs.dynamics.controllable import ControllableDeterministic, ControllableNoisy
from helplessness.envs.dynamics.factory import dynamics_from_config
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic, UncontrollableNoisy


def test_controllable_deterministic_from_config():
    spec = dynamics_from_config({"type": "ControllableDeterministic"})
    assert isinstance(spec, ControllableDeterministic)


def test_controllable_noisy_from_config():
    spec = dynamics_from_config({"type": "ControllableNoisy", "p": 0.4})
    assert isinstance(spec, ControllableNoisy)
    assert spec.p == 0.4


def test_uncontrollable_deterministic_from_config():
    spec = dynamics_from_config({"type": "UncontrollableDeterministic", "drift": [1, 0]})
    assert isinstance(spec, UncontrollableDeterministic)
    assert spec.drift == (1, 0)


def test_uncontrollable_noisy_from_config_with_explicit_move_probs():
    spec = dynamics_from_config({"type": "UncontrollableNoisy", "move_probs": [0.4, 0.3, 0.1, 0.1, 0.1]})
    assert isinstance(spec, UncontrollableNoisy)
    assert np.allclose(spec.move_probs, [0.4, 0.3, 0.1, 0.1, 0.1])


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown dynamics type"):
        dynamics_from_config({"type": "NotARealType"})


def test_does_not_mutate_the_input_dict():
    cfg = {"type": "ControllableNoisy", "p": 0.2}
    dynamics_from_config(cfg)
    assert cfg == {"type": "ControllableNoisy", "p": 0.2}
