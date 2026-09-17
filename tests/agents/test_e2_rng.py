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


def test_spawn_checkpoint_streams_deterministic_given_seed_and_checkpoint():
    a = rng_mod.spawn_checkpoint_streams(0, 1000)
    b = rng_mod.spawn_checkpoint_streams(0, 1000)
    assert a["matched_control"].random(5).tolist() == b["matched_control"].random(5).tolist()
    assert a["reference"].random(5).tolist() == b["reference"].random(5).tolist()


def test_spawn_checkpoint_streams_independent_across_checkpoints():
    a = rng_mod.spawn_checkpoint_streams(0, 1000)
    b = rng_mod.spawn_checkpoint_streams(0, 2000)
    assert a["matched_control"].random(5).tolist() != b["matched_control"].random(5).tolist()
    assert a["reference"].random(5).tolist() != b["reference"].random(5).tolist()


def test_spawn_checkpoint_streams_independent_across_seeds():
    a = rng_mod.spawn_checkpoint_streams(0, 1000)
    b = rng_mod.spawn_checkpoint_streams(1, 1000)
    assert a["matched_control"].random(5).tolist() != b["matched_control"].random(5).tolist()


def test_spawn_checkpoint_streams_result_independent_of_visiting_order():
    """The property the sweep's parallelisation depends on: computing
    checkpoint 2000's streams does not require having first computed
    checkpoint 1000's (unlike advancing one shared per-seed stream
    checkpoint-by-checkpoint would) -- checking each checkpoint in
    isolation, in either order, gives the same result."""
    forward_1000 = rng_mod.spawn_checkpoint_streams(5, 1000)["matched_control"].random(3).tolist()
    forward_2000 = rng_mod.spawn_checkpoint_streams(5, 2000)["matched_control"].random(3).tolist()

    # recompute in the opposite order, fresh
    backward_2000 = rng_mod.spawn_checkpoint_streams(5, 2000)["matched_control"].random(3).tolist()
    backward_1000 = rng_mod.spawn_checkpoint_streams(5, 1000)["matched_control"].random(3).tolist()

    assert forward_1000 == backward_1000
    assert forward_2000 == backward_2000


def test_spawn_streams_defaults_reproduce_plain_seed_entropy():
    """alpha/epsilon omitted must reproduce the exact pre-existing
    behaviour (the pilot's own usage, unaffected by this extension)."""
    a = rng_mod.spawn_streams(0)
    b = rng_mod.spawn_streams(0, alpha=None, epsilon=None)
    for name in rng_mod.STREAM_NAMES:
        assert a[name].random(5).tolist() == b[name].random(5).tolist()


def test_spawn_streams_different_conditions_same_seed_are_independent():
    a = rng_mod.spawn_streams(0, alpha=0.5, epsilon=0.1)
    b = rng_mod.spawn_streams(0, alpha=1.0, epsilon=0.1)
    c = rng_mod.spawn_streams(0, alpha=1.0, epsilon=1.0)
    for name in rng_mod.STREAM_NAMES:
        va, vb, vc = a[name].random(5).tolist(), b[name].random(5).tolist(), c[name].random(5).tolist()
        assert va != vb
        assert vb != vc
        assert va != vc


def test_spawn_streams_same_condition_deterministic():
    a = rng_mod.spawn_streams(3, alpha=5.0, epsilon=0.1)
    b = rng_mod.spawn_streams(3, alpha=5.0, epsilon=0.1)
    for name in rng_mod.STREAM_NAMES:
        assert a[name].random(5).tolist() == b[name].random(5).tolist()


def test_spawn_checkpoint_streams_different_conditions_same_seed_and_step_independent():
    a = rng_mod.spawn_checkpoint_streams(0, 1000, alpha=0.5, epsilon=0.1)
    b = rng_mod.spawn_checkpoint_streams(0, 1000, alpha=1.0, epsilon=0.1)
    assert a["matched_control"].random(5).tolist() != b["matched_control"].random(5).tolist()
    assert a["reference"].random(5).tolist() != b["reference"].random(5).tolist()


def test_spawn_checkpoint_streams_condition_defaults_reproduce_plain_entropy():
    a = rng_mod.spawn_checkpoint_streams(0, 1000)
    b = rng_mod.spawn_checkpoint_streams(0, 1000, alpha=None, epsilon=None)
    assert a["matched_control"].random(5).tolist() == b["matched_control"].random(5).tolist()


def test_spawn_checkpoint_streams_independent_from_spawn_streams():
    """The checkpoint-level matched_control/reference streams must not be
    the same object (or in lockstep with) spawn_streams(seed)'s own
    matched_control/reference -- distinct entropy root entirely."""
    main = rng_mod.spawn_streams(0)
    checkpoint = rng_mod.spawn_checkpoint_streams(0, 1000)
    assert main["matched_control"].random(5).tolist() != checkpoint["matched_control"].random(5).tolist()
    assert main["reference"].random(5).tolist() != checkpoint["reference"].random(5).tolist()
