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
