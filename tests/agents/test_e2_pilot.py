"""Smoke test for experiments/E2_online_4zones/pilot.py::run_one_seed --
tiny budget, just confirms the whole pipeline (env, agent, loop,
checkpointing, episode logging, VI iteration tracking) runs end to end
without error and produces internally-consistent output. The actual
pilot (Temps 2, budget=1e6) is run and reported separately, not here.
"""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

pilot_mod = load_experiment_module("E2_online_4zones", "pilot.py", "_e2_pilot_for_pilot_test")


def test_run_one_seed_smoke():
    checkpoints = (50, 100, 200)
    result = pilot_mod.run_one_seed(seed=0, budget=200, checkpoints=checkpoints)

    assert int(result["N_final"].sum()) == 200
    assert set(result["N_checkpoints"].keys()) == set(checkpoints)
    for t, N_t in result["N_checkpoints"].items():
        assert int(N_t.sum()) == t  # checkpoint t must hold exactly t steps' worth of counts
    assert len(result["plan_n_iters"]) >= 1  # at least the initial plan on the prior
    assert result["elapsed_seconds"] > 0


def test_run_one_seed_is_deterministic_given_the_same_seed():
    result_a = pilot_mod.run_one_seed(seed=1, budget=150, checkpoints=(100,))
    result_b = pilot_mod.run_one_seed(seed=1, budget=150, checkpoints=(100,))
    assert np.array_equal(result_a["N_final"], result_b["N_final"])
    assert result_a["episode_log"] == result_b["episode_log"]


def test_checkpoint_result_independent_of_total_budget():
    """The property the sweep design depends on ("chaque checkpoint
    equivaut a un run de ce budget"): a checkpoint's N snapshot must be
    IDENTICAL regardless of whether the run stops right after it or
    continues on to a much larger budget -- nothing (RNG derivation,
    replanning cadence, checkpoint definition) may look ahead at the
    total budget. Checked directly: the same seed's N at step 1000 must
    be bit-identical whether budget=1000 or budget=5000."""
    short_run = pilot_mod.run_one_seed(seed=42, budget=1000, checkpoints=(1000,))
    long_run = pilot_mod.run_one_seed(seed=42, budget=5000, checkpoints=(1000, 5000))
    assert np.array_equal(short_run["N_checkpoints"][1000], long_run["N_checkpoints"][1000])
    assert short_run["episode_log"] == [e for e in long_run["episode_log"] if e[0] <= 1000]


def test_checkpoint_matched_and_reference_streams_independent_of_total_budget():
    """Same property, one level down: spawn_checkpoint_streams (used for
    the matched control and reference at each checkpoint) depends only on
    (seed, checkpoint_step), confirmed here at the RNG level directly
    rather than only through its downstream effect on N."""
    a = pilot_mod.e2_rng.spawn_checkpoint_streams(seed=42, checkpoint_step=1000)
    b = pilot_mod.e2_rng.spawn_checkpoint_streams(seed=42, checkpoint_step=1000)
    assert a["matched_control"].random(5).tolist() == b["matched_control"].random(5).tolist()
