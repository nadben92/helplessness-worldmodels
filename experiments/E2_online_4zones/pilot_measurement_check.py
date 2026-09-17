"""Verification 3 (Temps-2 response): runs the FULL measurement chain
(matched control, matched reference, T_hat, I_hat, b, Delta, L1) at every
checkpoint of every pilot seed, and reports ONLY the status of the form
checks -- no interpretation of the resulting numbers.
"""

from __future__ import annotations

import numpy as np

import agent as e2_agent
import env as e2_env
import matched_control as matched_mod
import rng as rng_mod
import twin as twin_mod

from helplessness.config import load_yaml_block
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.controllability import controllability_per_state

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GOAL_REWARD = 50.0
GAMMA = 0.9
ALPHA = 1.0  # the pilot's own alpha

_LAYOUT = e2_env.build_layout(CFG, DIRECTION)
_CTRL_LAYOUT = e2_env.build_control_layout(CFG, DIRECTION)
_P_TRUE = e2_env.build_env(_LAYOUT, CFG, goal_reward=GOAL_REWARD).transition_tensor()
_P_REF = e2_env.build_env(_CTRL_LAYOUT, CFG, goal_reward=GOAL_REWARD).transition_tensor()
_R_TRUE, _TERMINAL = e2_env.reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
_SUPPORT = geographic_support_mask(GRID_SHAPE)
_GOAL_S = e2_env.goal_state(CFG)


def _zone_state_indices(category: str, exclude_goal: bool = True) -> np.ndarray:
    zone = next(z for z in _LAYOUT.zones if z.appearance.category == category)
    idx = np.flatnonzero(zone.mask.ravel())
    if exclude_goal:
        idx = idx[idx != _GOAL_S]
    return idx


def measure_checkpoint(N_online: np.ndarray, seed: int, checkpoint_step: int) -> dict:
    """Full chain for one (seed, checkpoint): online + matched, both twin
    results, form-check statuses. N_online: (S,A,S) counts at this
    checkpoint (an online agent's own N snapshot)."""
    streams = rng_mod.spawn_checkpoint_streams(seed, checkpoint_step)

    online_result = twin_mod.twin_from_counts(
        N=N_online, alpha=ALPHA, support=_SUPPORT, P_true=_P_TRUE, R_true=_R_TRUE,
        terminal=_TERMINAL, P_ref=_P_REF, gamma=GAMMA, reference_rng=streams["reference"],
    )

    N_target = N_online.sum(axis=2)
    matched_N = matched_mod.collect_matched(_P_TRUE, N_target, streams["matched_control"])
    # The SAME reference realisation is reused for online and matched
    # (spec: "la meme realisation de I_ref... est utilisee pour les deux
    # conditions d'une graine") -- recompute I_ref identically by reusing
    # the SAME reference_rng STATE. Since twin_from_counts() consumes the
    # reference_rng internally, we re-derive a fresh generator seeded
    # identically rather than reusing the (now-advanced) online one.
    streams_for_matched = rng_mod.spawn_checkpoint_streams(seed, checkpoint_step)
    matched_result = twin_mod.twin_from_counts(
        N=matched_N, alpha=ALPHA, support=_SUPPORT, P_true=_P_TRUE, R_true=_R_TRUE,
        terminal=_TERMINAL, P_ref=_P_REF, gamma=GAMMA, reference_rng=streams_for_matched["reference"],
    )

    checks = {}

    # 1) b == 0 exact in corridor and controllable_deterministic, both conditions
    for category in ("corridor", "controllable_deterministic"):
        idx = _zone_state_indices(category)
        checks[f"b_exact_zero_online_{category}"] = bool(np.allclose(online_result["b"][idx], 0.0, atol=1e-9))
        checks[f"b_exact_zero_matched_{category}"] = bool(np.allclose(matched_result["b"][idx], 0.0, atol=1e-9))

    # 2) T_hat online == T_hat matched in conveyor + controllable_deterministic
    online_T = online_result["T_hat"]
    matched_T = matched_result["T_hat"]
    for category in ("uncontrollable_deterministic", "controllable_deterministic"):
        idx = _zone_state_indices(category)
        checks[f"t_hat_online_eq_matched_{category}"] = bool(np.array_equal(online_T[idx], matched_T[idx]))

    # 3) Delta >= 0 everywhere (after clamp), both conditions
    checks["delta_nonneg_online"] = bool(np.all(online_result["delta"] >= -1e-8))
    checks["delta_nonneg_matched"] = bool(np.all(matched_result["delta"] >= -1e-8))

    return {
        "online_result": online_result,
        "matched_result": matched_result,
        "checks": checks,
    }


def jumeau_matches_fresh_agent(N_final: np.ndarray, alpha: float = ALPHA) -> dict:
    """Verification 3's 3rd bullet: jumeau(N) == the agent's own internal
    belief. Since value_iteration's converged V is warm-start-independent
    (a contraction mapping's unique fixed point -- see
    tests/planning/test_value_iteration.py::test_value_iteration_warm_start_converges_to_the_same_v_in_fewer_iterations),
    a FRESH agent planning cold on this exact final N must converge to
    the mathematically identical V as the original (warm-started) online
    run did -- checked directly against an independently constructed
    OnlineTabularAgentKnownReward, not merely against twin.py's own
    internals (twin.py already reuses that code path, see twin.py's
    _t_hat_from_counts -- this check is against a SEPARATE, freshly
    constructed agent object)."""
    fresh_agent = e2_agent.OnlineTabularAgentKnownReward(
        n_states=N_STATES, n_actions=N_ACTIONS, R_true=_R_TRUE, terminal_true=_TERMINAL,
        support=_SUPPORT, alpha=alpha, gamma=GAMMA,
    )
    fresh_agent._model.N = N_final.copy()
    fresh_agent.plan(np.random.default_rng(0))  # tiebreak_rng irrelevant to V/T_hat

    twin_result = twin_mod.twin_from_counts(
        N=N_final, alpha=alpha, support=_SUPPORT, P_true=_P_TRUE, R_true=_R_TRUE,
        terminal=_TERMINAL, P_ref=_P_REF, gamma=GAMMA, reference_rng=np.random.default_rng(0),
    )

    return {
        "t_hat_matches": bool(np.array_equal(fresh_agent.T_hat, twin_result["T_hat"])),
        "i_hat_matches": bool(np.allclose(controllability_per_state(fresh_agent.T_hat), twin_result["I_hat"])),
        "v_matches": bool(np.allclose(fresh_agent.V, twin_result["v_pi_believed"])),
    }
