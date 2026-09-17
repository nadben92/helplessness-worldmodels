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
