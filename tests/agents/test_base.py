"""Tests for the Agent ABC and its bridge to measures/ (materialize_transition_tensor):
verifies both existing concrete agents satisfy the interface, and that
routing through transition_distribution() reproduces the same
controllability numbers the direct T_hat-based path already produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from helplessness.agents.base import Agent, materialize_transition_tensor
from helplessness.agents.online.agent import OnlineTabularAgent
from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.measures.controllability import controllability_from_tensor


def test_tabular_agent_is_an_agent():
    agent = TabularModelBasedAgent(n_states=10, n_actions=4, alpha=0.1)
    assert isinstance(agent, Agent)
    assert np.allclose(agent.transition_distribution(0, 0), agent.T_hat[0, 0])


def test_online_agent_is_an_agent():
    env = GridWorld(p=0.2, seed=0, n_actions=5)
    agent = OnlineTabularAgent(env, alpha=0.1)
    assert isinstance(agent, Agent)
    assert np.allclose(agent.transition_distribution(2, 1), agent.T_hat[2, 1])


def test_materialize_transition_tensor_matches_direct_T_hat():
    env = GridWorld(p=0.3, seed=0)
    agent = TabularModelBasedAgent(env.n_states, 4, alpha=0.1)
    agent.observe(0, 0, -0.01, 1)
    agent.observe(0, 0, -0.01, 1)
    agent.observe(3, 2, -0.01, 8)

    T_bridged = materialize_transition_tensor(agent, env.n_states, 4)
    assert np.allclose(T_bridged, agent.T_hat)


def test_controllability_via_agent_bridge_matches_direct_call():
    """The point of the bridge: routing through the Agent interface must
    give the exact same I(a;s'|s) as calling controllability_from_tensor
    directly on T_hat -- measures/ never needs to know an agent exists."""
    env = GridWorld(p=0.4, seed=0)
    agent = TabularModelBasedAgent(env.n_states, 4, alpha=0.1)
    for s, a, s_next in [(0, 0, 1), (0, 1, 5), (3, 2, 8), (3, 2, 8), (10, 3, 9)]:
        agent.observe(s, a, -0.01, s_next)

    terminal = env.terminal_mask()
    direct = controllability_from_tensor(agent.T_hat, terminal)
    via_bridge = controllability_from_tensor(
        materialize_transition_tensor(agent, env.n_states, 4), terminal
    )
    assert direct == pytest.approx(via_bridge)
