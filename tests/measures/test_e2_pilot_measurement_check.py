"""Tests for experiments/E2_online_4zones/pilot_measurement_check.py:
measure_checkpoint's form checks and jumeau_matches_fresh_agent, on
modest synthetic counts (fast) -- the actual pilot data is checked
separately (see the Temps-2 verification response), this locks in the
MECHANICS so a regression is caught even without rerunning the pilot.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

pmc_mod = load_experiment_module("E2_online_4zones", "pilot_measurement_check.py", "_e2_pmc_for_pmc_test")

N_STATES = pmc_mod.N_STATES
N_ACTIONS = pmc_mod.N_ACTIONS


def _synthetic_N(seed: int, n_per_pair: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N[s, a] = rng.multinomial(n_per_pair, pmc_mod._P_TRUE[s, a])
    return N


def test_measure_checkpoint_all_form_checks_pass_on_synthetic_counts():
    N = _synthetic_N(seed=0)
    out = pmc_mod.measure_checkpoint(N, seed=0, checkpoint_step=1000)
    assert all(out["checks"].values()), out["checks"]


def test_measure_checkpoint_is_deterministic_given_seed_and_checkpoint():
    N = _synthetic_N(seed=1)
    out_a = pmc_mod.measure_checkpoint(N, seed=5, checkpoint_step=2000)
    out_b = pmc_mod.measure_checkpoint(N, seed=5, checkpoint_step=2000)
    assert np.array_equal(out_a["matched_result"]["T_hat"], out_b["matched_result"]["T_hat"])
    assert out_a["online_result"]["b"] == pytest.approx(out_b["online_result"]["b"])


def test_measure_checkpoint_uses_same_reference_realisation_for_both_conditions():
    """spec: la meme realisation de I_ref doit servir aux deux conditions
    d'une graine -- checked directly: I_ref must be IDENTICAL between the
    online and matched results (only I_hat differs)."""
    N = _synthetic_N(seed=2)
    out = pmc_mod.measure_checkpoint(N, seed=7, checkpoint_step=3000)
    assert np.array_equal(out["online_result"]["I_ref"], out["matched_result"]["I_ref"])


def test_jumeau_matches_fresh_agent_on_synthetic_counts():
    N = _synthetic_N(seed=3, n_per_pair=50)
    result = pmc_mod.jumeau_matches_fresh_agent(N)
    assert result["t_hat_matches"]
    assert result["i_hat_matches"]
    assert result["v_matches"]
