"""Value iteration on an arbitrary model -- oracle and planning.

Single source for the discount factor: every experiment and library
function that needs gamma imports GAMMA from here rather than redefining its
own copy (previously duplicated across run_offline.py and plot_sweep.py).
"""

from __future__ import annotations

import numpy as np

GAMMA = 0.95


def value_iteration(P, R, terminal, gamma: float = GAMMA, tol=1e-10, max_iter=10_000, V_init=None, return_n_iter=False):
    """P[s,a,s'], R[s'] reward on arrival. Returns (V, greedy policy), or
    (V, greedy policy, n_iter) if return_n_iter=True (additive, default
    False -- every existing 2-tuple-unpacking call site is unaffected).

    Solves the infinite-horizon discounted Bellman equation exactly, with no
    notion of max_steps -- terminal must contain only genuinely absorbing
    states (see step()), never a truncation. This is the reference that any
    agent bootstrapping on V(s') on truncated=True must reproduce, rather
    than treating truncation as a terminal state.

    V_init: optional warm start (defaults to zeros, matching every
    existing caller's behaviour exactly). Value iteration is a contraction
    mapping with a unique fixed point, so the CONVERGED V is identical
    regardless of V_init -- only the number of iterations to reach it
    changes (see tests/planning/test_value_iteration.py's regression pin).
    Needed by E2's online agent (docs/E2_online_4zones.md: "demarree
    depuis le V precedent"), where replanning from zero every time would
    make a large-budget run computationally impractical for no accuracy
    benefit.
    """
    n_states = P.shape[0]
    V = np.zeros(n_states) if V_init is None else np.array(V_init, dtype=np.float64, copy=True)
    V[terminal] = 0.0
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        Q = np.einsum("sap,p->sa", P, R + gamma * V * (~terminal))
        V_new = Q.max(axis=1)
        V_new[terminal] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = np.einsum("sap,p->sa", P, R + gamma * V * (~terminal))
    if return_n_iter:
        return V, Q.argmax(axis=1), n_iter
    return V, Q.argmax(axis=1)


def policy_evaluation(policy, P, R, terminal, gamma: float = GAMMA, tol=1e-10, max_iter=10_000):
    """V^policy under P -- the fixed-policy analogue of value_iteration(),
    with NO max anywhere: policy[s] is fixed and selects the one row of P
    used at s, never re-derived or improved. This is the only correct way
    to get "what does this specific policy actually achieve under this
    specific kernel" -- e.g. E1's Delta(s) = V*(s) - V^pi_hat(s) needs
    V^pi_hat evaluated under the TRUE kernel with pi_hat (learned from
    T_hat) held fixed, which value_iteration() cannot produce: its
    returned V is always the value of the GREEDY policy under whatever P
    it was called with, never a caller-supplied policy evaluated under a
    DIFFERENT P. Reusing value_iteration(T_hat, ...)'s own returned V as
    "V^pi_hat" would silently evaluate the policy under the wrong kernel
    (T_hat, not P) -- a distinct bug this function exists to make
    impossible by construction.

    policy[s] : int, the fixed action at each state. P[s,a,s'], R[s']
    reward on arrival. Same terminal-state convention as value_iteration():
    V[terminal] = 0, and R still contributes on arrival at a terminal
    state (only the *next* bootstrap term is masked), so the two functions
    agree exactly when policy is P's own optimal policy.
    """
    n_states = P.shape[0]
    P_pi = P[np.arange(n_states), policy]  # (S, S): the policy's own transition row per state
    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_new = P_pi @ (R + gamma * V * (~terminal))
        V_new[terminal] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V


def policy_evaluation_stochastic(policy_probs, P, R, terminal, gamma: float = GAMMA, tol=1e-10, max_iter=10_000):
    """V^policy under P, for a STOCHASTIC policy (S, A) row-stochastic --
    the mixed-action generalisation of policy_evaluation() above, needed
    wherever the greedy policy itself is a uniform mixture over tied
    actions (see evaluation.tiebreak.uniform_tie_policy_probs) rather than
    a single action per state.

    policy_probs[s] : distribution over actions at s, sums to 1.
    P[s,a,s'], R[s'] reward on arrival. Same terminal-state convention as
    policy_evaluation(): V[terminal] = 0, R still contributes on arrival
    at a terminal state. Reduces exactly to policy_evaluation() when
    policy_probs is one-hot per row (see
    tests/planning/test_value_iteration.py's regression pin), so this is
    a strict generalisation, not a second implementation of the same
    thing under a different name.
    """
    policy_probs = np.asarray(policy_probs, dtype=np.float64)
    P_pi = np.einsum("sa,sap->sp", policy_probs, P)  # (S, S): mixed transition row per state
    V = np.zeros(P.shape[0])
    for _ in range(max_iter):
        V_new = P_pi @ (R + gamma * V * (~terminal))
        V_new[terminal] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V
