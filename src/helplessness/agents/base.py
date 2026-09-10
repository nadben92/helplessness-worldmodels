"""Minimal common interface for agents.

Deliberately small: TabularModelBasedAgent (offline) and OnlineTabularAgent
already differ in their observe()/plan() signatures by design (the online
agent learns terminal[s] from experience and plans against a tie-break RNG;
the offline agent takes an externally-supplied terminal mask and a gamma).
Forcing those into one shape now, before a world-model agent exists to tell
us which parts of the shape are load-bearing, would be exactly the kind of
premature abstraction CLAUDE.md's own "what not to do" section warns
against for the environment side -- the same caution applies here.

What genuinely needs to be uniform, right now, is the one thing
measures/controllability.py needs to stay backend-agnostic: a way to ask
any agent for its belief about where a given (state, action) leads,
regardless of whether that belief lives in a counts table or a world
model's transition head. That is transition_distribution(). Everything
else two agents happen to share (a .policy attribute, a way to update on a
transition) is a convention today, documented here, not yet enforced by
this ABC -- promote it once a second concrete agent (the world-model one)
shows which parts are actually common.
"""

from __future__ import annotations

import abc

import numpy as np


class Agent(abc.ABC):
    """Common interface. Subclasses provide the concrete estimator/policy.

    Convention (not yet enforced by this ABC -- see module docstring):
    every agent exposes a `.policy` attribute, an (n_states,)-shaped array
    of the greedy action per state, kept up to date by whatever the
    subclass calls its planning step.
    """

    @abc.abstractmethod
    def transition_distribution(self, s: int, a: int) -> np.ndarray:
        """This agent's belief about P(s' | s, a), as a length-n_states
        array summing to 1. The single point of contact between any agent
        (table-backed or, later, world-model-backed) and
        measures.controllability -- see materialize_transition_tensor()
        below for how the two connect.
        """


def materialize_transition_tensor(agent: Agent, n_states: int, n_actions: int) -> np.ndarray:
    """Builds a full (n_states, n_actions, n_states) tensor by calling
    agent.transition_distribution(s, a) for every (s, a) pair.

    Lives here, not in measures/, because measures/ must never import an
    agent class (see CONTRIBUTING.md: it stays array-in/array-out so it
    transfers unchanged to a neural world model's belief). This is the
    bridge on the agent side: any Agent, tabular or (later) world-model,
    can be hooked to measures.controllability.controllability_from_tensor
    via `controllability_from_tensor(materialize_transition_tensor(agent, S, A), terminal)`
    without measures/ ever knowing an agent exists.

    Whether this is affordable depends on the backend: cheap for a tabular
    agent (a table lookup per call), and the reason a world-model agent may
    instead need its own batched/sampled way of producing a comparable
    tensor for large or continuous state spaces -- a concern for whichever
    agent implements this method, not for this function.
    """
    T = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for s in range(n_states):
        for a in range(n_actions):
            T[s, a] = agent.transition_distribution(s, a)
    return T
