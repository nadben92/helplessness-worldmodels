"""Tests for derive_control: uncontrollable -> controllable at the SAME
noise level p, controllable left untouched."""

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


def test_uncontrollable_noisy_pairs_to_controllable_noisy_at_the_same_p():
    for p in (0.0, 0.15, 0.3, 0.7, 1.0):
        paired = derive_control(UncontrollableNoisy(drift=(1, 0), p=p))
        assert isinstance(paired, ControllableNoisy)
        assert paired.p == p


def test_controllable_specs_pass_through_unchanged():
    det = ControllableDeterministic()
    noisy = ControllableNoisy(p=0.4)
    assert derive_control(det) is det
    assert derive_control(noisy) is noisy
