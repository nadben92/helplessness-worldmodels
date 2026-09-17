"""Tests for experiments/E2_online_4zones/rng.py::spawn_streams."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

rng_mod = load_experiment_module("E2_online_4zones", "rng.py", "_e2_rng_for_rng_test")


def test_spawn_streams_returns_one_generator_per_name():
    streams = rng_mod.spawn_streams(0)
    assert set(streams.keys()) == set(rng_mod.STREAM_NAMES)
    for gen in streams.values():
        assert isinstance(gen, np.random.Generator)


def test_spawn_streams_is_deterministic_given_the_same_seed():
    a = rng_mod.spawn_streams(42)
    b = rng_mod.spawn_streams(42)
    for name in rng_mod.STREAM_NAMES:
        assert a[name].random(5).tolist() == b[name].random(5).tolist()


def test_spawn_streams_are_mutually_independent_draws():
    """Different named streams must not produce identical draw sequences
    -- the exact failure mode E1's spawn_exp_ctrl_rngs bug produced when
    two Generators were seeded with the same integer value."""
    streams = rng_mod.spawn_streams(7)
    draws = {name: gen.random(20).tolist() for name, gen in streams.items()}
    names = list(draws.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert draws[names[i]] != draws[names[j]], f"{names[i]} and {names[j]} produced identical draws"


def test_spawn_streams_different_seeds_give_different_streams():
    a = rng_mod.spawn_streams(1)
    b = rng_mod.spawn_streams(2)
    for name in rng_mod.STREAM_NAMES:
        assert a[name].random(5).tolist() != b[name].random(5).tolist()
