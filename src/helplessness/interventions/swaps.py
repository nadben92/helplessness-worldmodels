"""Component swaps between agents (Phase 3 -- aetiology), cf. roadmap in CLAUDE.md.

A model-based agent separates three pieces: the transition belief T_hat, the
reward belief R_hat, and the planner. The planner is fixed for all agents
(project invariant), so any behavioural difference must come from T_hat, from
R_hat, or both. Swapping a piece between an agent trained at low p and an
agent trained at high p, then replanning, reveals which one carries the
phenotype.

R_hat is expected to be near-perfect at any p, since the reward is
deterministic given s' (reward_vector() does not depend on p). The prior is
therefore that the phenotype lives entirely in T_hat -- this module serves to
verify that rather than assume it.

Operates on (T, R) arrays and on agent-like objects duck-typed via .T_hat /
.R_hat, never on an imported agent class: keeps this module transferable to
non-tabular agents.

How to read results/swaps.csv (written by experiments/02_swaps/run.py):
  - if T_high_R_low behaves like intact_high, and T_low_R_high like
    intact_low, the phenotype is entirely carried by T_hat.
  - if T_true_R_high recovers oracle-level behaviour, the reward beliefs are
    intact and irrelevant to the phenotype.
  - if T_high_R_true still fails, that confirms the same thing from the
    other end (the problem tracks T_hat, not R_hat).
  - any other pattern (e.g. T_true_R_high still fails, or T_high_R_true
    succeeds) means reward devaluation also contributes, and deserves its
    own analysis -- do not silently dismiss it.
"""

from __future__ import annotations

import numpy as np

from helplessness.planning.value_iteration import GAMMA, value_iteration

CONDITIONS = [
    "intact_low", "intact_high", "T_high_R_low", "T_low_R_high",
    "T_true_R_high", "T_high_R_true", "oracle",
]


def plan_from_components(
    T: np.ndarray, R: np.ndarray, terminal: np.ndarray, gamma: float = GAMMA
):
    """Value iteration on an arbitrary (T, R) pair. Returns (V, Q, policy)."""
    V, policy = value_iteration(T, R, terminal, gamma=gamma)
    Q = np.einsum("sap,p->sa", T, R + gamma * V * (~terminal))
    return V, Q, policy


def swap_matrix(agent_low, agent_high, T_true: np.ndarray, R_true: np.ndarray, terminal: np.ndarray, gamma: float = GAMMA) -> dict:
    """Builds and plans the 7 conditions of the swap table.

    terminal must come from the TEST environment (p=0), not from either
    donor agent: it is a property of the layout, not of a belief -- passing
    a donor's terminal_mask() would be correct here since the layout is
    fixed, but keeping it explicitly separate from the agent avoids any
    ambiguity if a donor were to come from a different layout.

    Returns {condition: (V, Q, policy)} for the 7 conditions in CONDITIONS.
    """
    T_low, R_low = agent_low.T_hat, agent_low.R_hat
    T_high, R_high = agent_high.T_hat, agent_high.R_hat

    components = {
        "intact_low": (T_low, R_low),
        "intact_high": (T_high, R_high),
        "T_high_R_low": (T_high, R_low),
        "T_low_R_high": (T_low, R_high),
        "T_true_R_high": (T_true, R_high),
        "T_high_R_true": (T_high, R_true),
        "oracle": (T_true, R_true),
    }
    return {
        name: plan_from_components(T, R, terminal, gamma=gamma)
        for name, (T, R) in components.items()
    }
