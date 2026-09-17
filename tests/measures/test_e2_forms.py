"""Structural ("formes exactes") and statistical ("formes statistiques")
form tests for E2's measurement chain, on the REAL E1 4-zone geometry
(reused via experiments/E2_online_4zones/env.py) at modest budgets --
these validate the machinery's correctness before the actual pilot
(Temps 2), not the pilot's own scientific numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from _experiment_loader import load_experiment_module

env_mod = load_experiment_module("E2_online_4zones", "env.py", "_e2_env_for_forms_test")
agent_mod = load_experiment_module("E2_online_4zones", "agent.py", "_e2_agent_for_forms_test")
loop_mod = load_experiment_module("E2_online_4zones", "loop.py", "_e2_loop_for_forms_test")
twin_mod = load_experiment_module("E2_online_4zones", "twin.py", "_e2_twin_for_forms_test")
matched_mod = load_experiment_module("E2_online_4zones", "matched_control.py", "_e2_matched_for_forms_test")
rng_mod = load_experiment_module("E2_online_4zones", "rng.py", "_e2_rng_for_forms_test")

from helplessness.config import load_yaml_block  # noqa: E402
from helplessness.envs.zones import geographic_support_mask  # noqa: E402

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GOAL_REWARD = 50.0
GAMMA = 0.9
ALPHA = 1.0

_LAYOUT = env_mod.build_layout(CFG, DIRECTION)
_CTRL_LAYOUT = env_mod.build_control_layout(CFG, DIRECTION)
_P_TRUE = env_mod.build_env(_LAYOUT, CFG, goal_reward=GOAL_REWARD).transition_tensor()
_P_REF = env_mod.build_env(_CTRL_LAYOUT, CFG, goal_reward=GOAL_REWARD).transition_tensor()
_R_TRUE, _TERMINAL = env_mod.reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
_SUPPORT = geographic_support_mask(GRID_SHAPE)
_GOAL_S = env_mod.goal_state(CFG)
_NON_TERMINAL_STATES = np.array([s for s in range(N_STATES) if s != _GOAL_S])


def _zone_state_indices(category: str) -> np.ndarray:
    zone = next(z for z in _LAYOUT.zones if z.appearance.category == category)
    return np.flatnonzero(zone.mask.ravel())


def _run_online(seed: int, n_steps: int, epsilon: float) -> "agent_mod.OnlineTabularAgentKnownReward":
    streams = rng_mod.spawn_streams(seed)
    env = env_mod.build_env(
        _LAYOUT, CFG, goal_reward=GOAL_REWARD, max_steps=36,
        seed=int(streams["env_dynamics"].integers(2**31)),
    )
    agent = agent_mod.OnlineTabularAgentKnownReward(
        n_states=N_STATES, n_actions=N_ACTIONS, R_true=_R_TRUE, terminal_true=_TERMINAL,
        support=_SUPPORT, alpha=ALPHA, gamma=GAMMA,
    )
    loop_mod.train_online_e2(
        env, agent, n_steps=n_steps, epsilon=epsilon, non_terminal_states=_NON_TERMINAL_STATES,
        explore_rng=streams["explore"], tiebreak_rng=streams["tiebreak"], start_rng=streams["start_state"],
    )
    agent.plan(streams["tiebreak"])  # final replan matching the final N (see test_e2_twin.py's own note)
    return agent


def _twin(N: np.ndarray, reference_rng: np.random.Generator, P_ref=None) -> dict:
    return twin_mod.twin_from_counts(
        N=N, alpha=ALPHA, support=_SUPPORT, P_true=_P_TRUE, R_true=_R_TRUE, terminal=_TERMINAL,
        P_ref=_P_REF if P_ref is None else P_ref, gamma=GAMMA, reference_rng=reference_rng,
    )


# ---------------------------------------------------------------------
# formes exactes
# ---------------------------------------------------------------------


def test_b_is_exactly_zero_in_corridor_and_controllable_deterministic_both_conditions():
    seed = 0
    agent = _run_online(seed, n_steps=4000, epsilon=0.1)
    streams = rng_mod.spawn_streams(seed)

    online_result = _twin(agent.N, streams["reference"])

    N_target = agent.N.sum(axis=2)
    matched_N = matched_mod.collect_matched(_P_TRUE, N_target, streams["matched_control"])
    matched_result = _twin(matched_N, streams["reference"])

    for category in ("corridor", "controllable_deterministic"):
        idx = _zone_state_indices(category)
        assert online_result["b"][idx] == pytest.approx(0.0, abs=1e-9), f"{category}: online b != 0"
        assert matched_result["b"][idx] == pytest.approx(0.0, abs=1e-9), f"{category}: matched b != 0"


def test_t_hat_online_and_matched_are_identical_in_deterministic_zones_at_equal_counts():
    """A deterministic transition row always lands 100% of its draws on
    the SAME single successor, regardless of how the draws were obtained
    (real trajectory vs fresh multinomial resample) -- so at equal N,
    T_hat must be bit-identical between online and matched in the
    corridor and controllable_deterministic zones."""
    seed = 1
    agent = _run_online(seed, n_steps=4000, epsilon=0.1)
    streams = rng_mod.spawn_streams(seed)

    N_target = agent.N.sum(axis=2)
    matched_N = matched_mod.collect_matched(_P_TRUE, N_target, streams["matched_control"])

    online_T = twin_mod._t_hat_from_counts(agent.N, ALPHA, _SUPPORT)
    matched_T = twin_mod._t_hat_from_counts(matched_N, ALPHA, _SUPPORT)

    for category in ("corridor", "controllable_deterministic"):
        idx = _zone_state_indices(category)
        assert np.array_equal(online_T[idx], matched_T[idx]), f"{category}: T_hat differs between online and matched"


def test_delta_is_nonnegative_everywhere_online_and_matched():
    seed = 2
    agent = _run_online(seed, n_steps=4000, epsilon=0.1)
    streams = rng_mod.spawn_streams(seed)

    online_result = _twin(agent.N, streams["reference"])
    assert np.all(online_result["delta"] >= -1e-8)

    N_target = agent.N.sum(axis=2)
    matched_N = matched_mod.collect_matched(_P_TRUE, N_target, streams["matched_control"])
    matched_result = _twin(matched_N, streams["reference"])
    assert np.all(matched_result["delta"] >= -1e-8)


# ---------------------------------------------------------------------
# formes statistiques
# ---------------------------------------------------------------------


def test_b_has_nonzero_variance_across_seeds_in_both_noisy_zones():
    """Sanity check against correlated generators (E1's own historical
    bug): b must show real between-seed variance in the two noisy zones,
    at a modest, fixed N per pair (generative, not a full online run --
    fast, and this is purely about RNG independence, not adaptivity)."""
    n_per_pair = 20
    b_by_zone = {"controllable_noisy": [], "uncontrollable_noisy": []}
    for seed in range(8):
        seed_seq = np.random.SeedSequence(seed)
        child_exp, child_ref = seed_seq.spawn(2)
        rng_exp, rng_ref = np.random.default_rng(child_exp), np.random.default_rng(child_ref)
        N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                N[s, a] = rng_exp.multinomial(n_per_pair, _P_TRUE[s, a])
        result = _twin(N, rng_ref)
        for category in b_by_zone:
            idx = _zone_state_indices(category)
            b_by_zone[category].append(float(np.mean(result["b"][idx])))

    for category, values in b_by_zone.items():
        assert np.var(values) > 1e-8, f"{category}: b variance across seeds is ~0 -- generators may be correlated"


def test_b_converges_toward_analytic_predictions_at_large_n():
    """Generative, large n (fast -- no real online rollout needed to test
    convergence, matching E1's own Etape B approach): b -> -log2(5) in
    the conveyor, ~-1.65 in uncontrollable_noisy, -> 0 in the two
    controllable zones. Analytic values used ONLY for this convergence
    check, per spec -- never inside the measurement chain itself."""
    n_per_pair = 3000
    rng = np.random.default_rng(0)
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int64)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            N[s, a] = rng.multinomial(n_per_pair, _P_TRUE[s, a])
    result = _twin(N, np.random.default_rng(1))

    b_conveyor = float(np.mean(result["b"][_zone_state_indices("uncontrollable_deterministic")]))
    b_noisy_unc = float(np.mean(result["b"][_zone_state_indices("uncontrollable_noisy")]))
    b_ctrl_noisy = float(np.mean(result["b"][_zone_state_indices("controllable_noisy")]))
    b_ctrl_det = float(np.mean(result["b"][_zone_state_indices("controllable_deterministic")]))

    assert b_conveyor == pytest.approx(-np.log2(5), abs=0.03)
    assert b_noisy_unc == pytest.approx(-1.65, abs=0.05)
    assert b_ctrl_noisy == pytest.approx(0.0, abs=0.05)
    assert b_ctrl_det == pytest.approx(0.0, abs=1e-9)


def test_no_distribution_difference_between_online_and_matched_at_epsilon_one():
    """spec: 'Pour epsilon=1, aucune difference de distribution entre
    online et apparie (pas de boucle, donc pas d'adaptativite).' Two-sample
    (KS) test on the zone-mean b value, across independent seeds, for
    online(epsilon=1) vs its own matched condition."""
    n_seeds = 15
    n_steps = 1500
    category = "controllable_noisy"
    idx = _zone_state_indices(category)

    b_online, b_matched = [], []
    for seed in range(100, 100 + n_seeds):
        agent = _run_online(seed, n_steps=n_steps, epsilon=1.0)
        streams = rng_mod.spawn_streams(seed)
        online_result = _twin(agent.N, streams["reference"])
        b_online.append(float(np.mean(online_result["b"][idx])))

        N_target = agent.N.sum(axis=2)
        matched_N = matched_mod.collect_matched(_P_TRUE, N_target, streams["matched_control"])
        matched_result = _twin(matched_N, streams["reference"])
        b_matched.append(float(np.mean(matched_result["b"][idx])))

    stat, p_value = ks_2samp(b_online, b_matched)
    assert p_value > 0.05, f"online vs matched b distributions differ at epsilon=1: KS p={p_value:.4f}"
