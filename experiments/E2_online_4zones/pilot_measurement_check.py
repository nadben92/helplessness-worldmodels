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

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import load_yaml_block
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.controllability import controllability_per_state
from helplessness.planning.value_iteration import value_iteration

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
    """CIRCULAR, kept only as a basic sanity check on twin.py's own
    internal consistency -- both sides of this comparison are computed
    by the SAME code (a freshly constructed agent planning on N, vs
    twin.py doing the same thing internally), so this can never catch a
    real discrepancy between the twin and an agent that actually
    accumulated its counts step by step during collection. See
    jumeau_matches_live_agent() below for the real, non-circular check
    (compares against the LIVE agent's own N/V/policy as captured during
    the actual online run, never recomputed)."""
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


def jumeau_matches_live_agent(
    N_log: np.ndarray, N_agent: np.ndarray, V_agent: np.ndarray, policy_agent: np.ndarray,
    alpha: float = ALPHA, gamma: float = GAMMA, tol: float = 1e-10,
) -> dict:
    """The NON-CIRCULAR jumeau check: N_log comes from an independent
    reconstruction of the raw trajectory log (trajectory.py), never from
    the agent's own counting; N_agent/V_agent/policy_agent are captured
    DIRECTLY off the live agent object during the real online run (see
    pilot.py), never recomputed here -- two genuinely separate sources
    compared against each other, not two computations of the same thing.

    Returns 4 checks:
    - n_log_matches_n_agent: the trajectory reconstruction agrees with
      the agent's own internal counting, bit-for-bit.
    - t_hat_matches: T_hat built from N_log (via a fresh
      TabularModelBasedAgent, NOT via twin.py, to keep this comparison
      independent of twin.py's own code path too) equals T_hat built
      from N_agent the same way, bit-for-bit -- trivially implied by the
      first check given T_hat is a pure function of N, but exercised
      explicitly anyway.
    - v_gap: max|V_jumeau - V_agent|, where V_jumeau is a COLD
      value_iteration on T_hat(N_log) and V_agent is the LIVE agent's own
      V, captured as-is (may be stale relative to N_agent if the
      checkpoint falls mid-episode -- replanning happens only at episode
      boundaries, see loop.py -- not masked by a forced extra replan).
    - greedy_action_mismatches: states where policy_agent[s] is NOT among
      the jumeau's own tied-optimal actions (tolerance-based tie check on
      Q_jumeau) -- ties are excluded from the comparison, per spec.
    """
    n_log_matches_n_agent = bool(np.array_equal(N_log, N_agent))

    agent_from_log = TabularModelBasedAgent(N_STATES, N_ACTIONS, alpha=alpha, alpha_support=_SUPPORT)
    agent_from_log.N = N_log
    T_hat_jumeau = agent_from_log.T_hat

    agent_from_agent_n = TabularModelBasedAgent(N_STATES, N_ACTIONS, alpha=alpha, alpha_support=_SUPPORT)
    agent_from_agent_n.N = N_agent
    T_hat_from_agent_n = agent_from_agent_n.T_hat

    t_hat_matches = bool(np.array_equal(T_hat_jumeau, T_hat_from_agent_n))

    V_jumeau, _ = value_iteration(T_hat_jumeau, _R_TRUE, _TERMINAL, gamma=gamma, tol=tol)
    v_gap = float(np.max(np.abs(V_jumeau - V_agent)))

    Q_jumeau = np.einsum("sap,p->sa", T_hat_jumeau, _R_TRUE + gamma * V_jumeau * (~_TERMINAL))
    row_max = Q_jumeau.max(axis=1, keepdims=True)
    tied = Q_jumeau >= (row_max - 1e-6)
    mismatches = [int(s) for s in range(N_STATES) if not tied[s, policy_agent[s]]]

    return {
        "n_log_matches_n_agent": n_log_matches_n_agent,
        "t_hat_matches": t_hat_matches,
        "v_gap": v_gap,
        "greedy_action_mismatches": mismatches,
    }
