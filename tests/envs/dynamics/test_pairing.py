"""Tests for derive_control: uncontrollable -> ControllableDeterministic,
controllable left untouched."""

from __future__ import annotations

from helplessness.envs.dynamics import (
    ControllableDeterministic,
    ControllableNoisy,
    UncontrollableDeterministic,
    UncontrollableNoisy,
    derive_control,
)


def test_uncontrollable_deterministic_pairs_to_controllable_deterministic():
    paired = derive_control(UncontrollableDeterministic(drift=(1, 0)))
    assert isinstance(paired, ControllableDeterministic)


def test_uncontrollable_noisy_pairs_to_controllable_deterministic():
    paired = derive_control(UncontrollableNoisy())
    assert isinstance(paired, ControllableDeterministic)


def test_controllable_specs_pass_through_unchanged():
    det = ControllableDeterministic()
    noisy = ControllableNoisy(p=0.4)
    assert derive_control(det) is det
    assert derive_control(noisy) is noisy
