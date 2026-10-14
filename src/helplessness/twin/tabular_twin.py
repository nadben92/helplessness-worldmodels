"""The tabular twin: ONE function, N[s,a,s'] (plus alpha, support, the true
and reference kernels, R, terminal, gamma) IN, the full measurement chain
(T_hat, I_hat, I_ref, b, V_believed, V_pi_true, Delta, per-pair L1,
is_optimal_under_P) OUT. Depends on no agent internal state whatsoever
(no OnlineTabularAgentKnownReward import here) -- it reconstructs T_hat
from N via the same TabularModelBasedAgent estimator every other tabular
computation in this codebase uses (agents.tabular, not reimplemented),
which is a computational tool, not "the agent": this function would run
identically on a logged N[s,a,s'] table produced by a future world model
(E3), which is the entire point (see docs/E2_online_4zones.md).

Everything here routes through the existing shared measures/planning
primitives (measures.controllability, measures.belief, planning.value_iteration,
evaluation.tiebreak) -- no second implementation of I(a;s'|s), L1, or
value iteration.

Moved from experiments/E2_online_4zones/twin.py on 2026-10-14 (a re-export
shim stays there); body unchanged, except that collect_matched is imported
from helplessness.twin.matched_control instead of the bare sibling module.
"""

from __future__ import annotations

import numpy as np

from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.evaluation.tiebreak import uniform_tie_policy_probs
from helplessness.measures.belief import l1_error_per_pair
from helplessness.measures.controllability import controllability_per_state
from helplessness.planning.value_iteration import policy_evaluation_stochastic, value_iteration
from helplessness.twin.matched_control import collect_matched


def _t_hat_from_counts(N: np.ndarray, alpha: float, support: np.ndarray) -> np.ndarray:
    n_states, n_actions, _ = N.shape
    agent = TabularModelBasedAgent(n_states, n_actions, alpha=alpha, alpha_support=support)
    agent.N = N.copy()
    return agent.T_hat


def twin_from_counts(
    N: np.ndarray,
    alpha: float,
    support: np.ndarray,
    P_true: np.ndarray,
    R_true: np.ndarray,
    terminal: np.ndarray,
    P_ref: np.ndarray,
    gamma: float,
    reference_rng: np.random.Generator,
    tol: float = 1e-9,
) -> dict:
    """N: (S, A, S) logged counts (e.g. an online agent's final N, or any
    snapshot of it -- this function never reads anything else off "the
    agent"). Returns a dict of per-state and per-pair arrays:

    per-state: I_hat, I_ref, b, v_pi_believed (V_believed, the planner's
    own value under T_hat -- kept for parity with E1's pessimism report,
    never used for Delta), v_pi_true (V^pi_hat, evaluated under P_true),
    delta (V_star - v_pi_true, clamped per E1's convention).
    per-pair: T_hat, l1 (vs P_true), is_optimal_under_P (bool, TRUE
    kernel's own tied-optimal actions, tolerance `tol`).

    I_ref is estimated by drawing, for every (s,a) pair, EXACTLY N[s,a]
    transitions from P_ref (matched-counts reference, never the analytic
    value -- see docs/E2_online_4zones.md) via reference_rng, an
    INDEPENDENT generator from whatever produced N itself.
    """
    n_states, n_actions, _ = N.shape
    N_target = N.sum(axis=2)  # (S, A)

    T_hat = _t_hat_from_counts(N, alpha, support)
    I_hat = controllability_per_state(T_hat)
    I_star = controllability_per_state(P_true)

    N_ref = collect_matched(P_ref, N_target, reference_rng)
    T_ref = _t_hat_from_counts(N_ref, alpha, support)
    I_ref = controllability_per_state(T_ref)

    b = I_hat - I_ref

    # V_believed: the planner's OWN value under (T_hat, R_true, terminal)
    # -- kept for the pessimism report (E1's v_pi_believed), never fed
    # into Delta (which must be pi_hat evaluated under the TRUE kernel).
    V_believed, _ = value_iteration(T_hat, R_true, terminal, gamma=gamma)
    Q_hat = np.einsum("sap,p->sa", T_hat, R_true + gamma * V_believed * (~terminal))

    # E2's Delta convention (not E1's plain argmax, see
    # docs/E2_online_4zones.md): uniform mixture over Q_hat's own ties.
    policy_probs = uniform_tie_policy_probs(Q_hat, tol=tol)
    V_pi_true = policy_evaluation_stochastic(policy_probs, P_true, R_true, terminal, gamma=gamma)

    V_star, _ = value_iteration(P_true, R_true, terminal, gamma=gamma)
    delta = V_star - V_pi_true
    delta = np.where(np.abs(delta) < 1e-9, 0.0, delta)  # E1's floating-point floor clamp, unchanged

    l1 = l1_error_per_pair(T_hat, P_true)  # (S, A)

    Q_star = np.einsum("sap,p->sa", P_true, R_true + gamma * V_star * (~terminal))
    row_max = Q_star.max(axis=1, keepdims=True)
    is_optimal_under_P = Q_star >= (row_max - tol)  # (S, A) bool, ties included

    return {
        "T_hat": T_hat,
        "I_hat": I_hat,
        "I_star": I_star,
        "I_ref": I_ref,
        "b": b,
        "v_pi_believed": V_believed,
        "v_pi_true": V_pi_true,
        "v_star": V_star,
        "delta": delta,
        "l1": l1,
        "is_optimal_under_P": is_optimal_under_P,
        "N_target": N_target,
    }
