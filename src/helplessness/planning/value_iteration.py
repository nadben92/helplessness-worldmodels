"""Value iteration on an arbitrary model -- oracle and planning.

Single source for the discount factor: every experiment and library
function that needs gamma imports GAMMA from here rather than redefining its
own copy (previously duplicated across run_offline.py and plot_sweep.py).
"""

from __future__ import annotations

import numpy as np

GAMMA = 0.95


def value_iteration(P, R, terminal, gamma: float = GAMMA, tol=1e-10, max_iter=10_000):
    """P[s,a,s'], R[s'] reward on arrival. Returns (V, greedy policy).

    Solves the infinite-horizon discounted Bellman equation exactly, with no
    notion of max_steps -- terminal must contain only genuinely absorbing
    states (see step()), never a truncation. This is the reference that any
    agent bootstrapping on V(s') on truncated=True must reproduce, rather
    than treating truncation as a terminal state.
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)
    for _ in range(max_iter):
        Q = np.einsum("sap,p->sa", P, R + gamma * V * (~terminal))
        V_new = Q.max(axis=1)
        V_new[terminal] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = np.einsum("sap,p->sa", P, R + gamma * V * (~terminal))
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
