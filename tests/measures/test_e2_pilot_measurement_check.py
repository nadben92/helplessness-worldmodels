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


def test_jumeau_matches_live_agent_when_n_log_and_n_agent_and_v_agent_all_consistent():
    """The non-circular check must PASS when N_log, N_agent, and V_agent
    are genuinely consistent (V_agent freshly converged on the same N) --
    the "everything lines up" case."""
    N = _synthetic_N(seed=4, n_per_pair=100)
    from helplessness.planning.value_iteration import value_iteration
    from helplessness.agents.tabular import TabularModelBasedAgent

    tmp_agent = TabularModelBasedAgent(N_STATES, N_ACTIONS, alpha=pmc_mod.ALPHA, alpha_support=pmc_mod._SUPPORT)
    tmp_agent.N = N
    V_agent, _ = value_iteration(tmp_agent.T_hat, pmc_mod._R_TRUE, pmc_mod._TERMINAL, gamma=pmc_mod.GAMMA)
    Q_agent = np.einsum("sap,p->sa", tmp_agent.T_hat, pmc_mod._R_TRUE + pmc_mod.GAMMA * V_agent * (~pmc_mod._TERMINAL))
    policy_agent = Q_agent.argmax(axis=1)

    result = pmc_mod.jumeau_matches_live_agent(N_log=N, N_agent=N, V_agent=V_agent, policy_agent=policy_agent)
    assert result["n_log_matches_n_agent"]
    assert result["t_hat_matches"]
    assert result["v_gap"] < 1e-6
    assert result["greedy_action_mismatches"] == []


def test_jumeau_matches_live_agent_detects_genuine_staleness():
    """The check must actually HAVE TEETH: if V_agent was computed under
    a DIFFERENT N than N_agent (standing in for staleness -- exactly what
    a mid-episode checkpoint produces in the real pilot: V computed under
    an older T_hat than the N captured a moment later), the v_gap must be
    large, not silently pass. This is precisely what the old (circular)
    jumeau_matches_fresh_agent could never detect, since it always
    recomputes both sides from the SAME N. Uses two independently-drawn
    N realisations (not a single-pair tweak, which can land on a
    non-argmax action and change nothing -- see the first version of
    this test) to guarantee a real T_hat, hence V, divergence."""
    from helplessness.planning.value_iteration import value_iteration
    from helplessness.agents.tabular import TabularModelBasedAgent

    N_agent = _synthetic_N(seed=5, n_per_pair=50)
    N_different = _synthetic_N(seed=99, n_per_pair=50)  # a wholly independent realisation

    stale_agent = TabularModelBasedAgent(N_STATES, N_ACTIONS, alpha=pmc_mod.ALPHA, alpha_support=pmc_mod._SUPPORT)
    stale_agent.N = N_different
    V_agent_stale, _ = value_iteration(stale_agent.T_hat, pmc_mod._R_TRUE, pmc_mod._TERMINAL, gamma=pmc_mod.GAMMA)
    Q_stale = np.einsum("sap,p->sa", stale_agent.T_hat, pmc_mod._R_TRUE + pmc_mod.GAMMA * V_agent_stale * (~pmc_mod._TERMINAL))
    policy_stale = Q_stale.argmax(axis=1)

    result = pmc_mod.jumeau_matches_live_agent(N_log=N_agent, N_agent=N_agent, V_agent=V_agent_stale, policy_agent=policy_stale)
    assert result["n_log_matches_n_agent"]  # N_log and N_agent still agree (both are N_agent)
    assert result["t_hat_matches"]  # both built from N_agent
    assert result["v_gap"] > 1e-3  # V_agent doesn't match N_agent's own T_hat -- must show up
