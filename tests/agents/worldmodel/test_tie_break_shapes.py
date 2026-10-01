"""The two shape tests of the absolute tie tolerance (docs/e3/DECISIONS.md, 2026-10-01 (12)), through the
agent's own q_values (one-step lookahead, K prior draws) and act (epsilon-greedy, E2's tie-break):

(i)  no value information: r_hat constant across actions (p_goal = 0) and V = constant + Gaussian noise of
     standard deviation sigma on every evaluated latent, sigma from 0.005 to 0.05: ties >= 99 % and the actions
     played uniform (as E2's tabular agent without information);
(ii) information: the Q gap of a goal 20 steps away, then 36 (the horizon), between the action that brings the
     agent one step closer and the four that leave the distance unchanged (the smallest informative gap); with the
     same critic noise on top, the greedy choice follows the informative action.

The world model and the heads are stubs: h'_a carries the action a (one-hot), the prior is uniform, the critic
reads the value of the successor it is given and adds the noise from its own generator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from helplessness.agents.worldmodel import policy
from helplessness.agents.worldmodel.heads import AgentConfig, horizon_information_gap

A, C, V_VARS = 5, 4, 2
CFG = AgentConfig(epsilon=0.1, k_act=16)  # the bench's K and epsilon; the tolerance from E2's episode: 0.0751


class StubRSSM:
    cfg = SimpleNamespace(n_actions=A, n_classes=C)

    @staticmethod
    def latent_hook(h, z, context):
        return h, z

    def imagination_step(self, h, z, a_onehot, context=None, generator=None):
        prior = torch.full((h.shape[0], V_VARS, C), 1.0 / C)
        return {"h": a_onehot.clone(), "z": None, "prior_probs": prior}


class NoGoal(torch.nn.Module):
    def p_goal(self, h, z):
        return torch.zeros(h.shape[0])


class NoisyCritic(torch.nn.Module):
    """V(h', z') = value[action carried by h'] + sigma * N(0, 1), a new draw at every evaluated latent."""

    def __init__(self, values, sigma, seed):
        super().__init__()
        self.values, self.sigma = torch.as_tensor(values, dtype=torch.float32), sigma
        self.gen = torch.Generator().manual_seed(seed)

    def forward(self, h, z):
        return self.values[h.argmax(-1)] + self.sigma * torch.randn(h.shape[0], generator=self.gen)


def _play(values, sigma, n=3000, seed=0):
    rssm, goal, critic = StubRSSM(), NoGoal(), NoisyCritic(values, sigma, seed)
    explore, tiebreak = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    gen = torch.Generator().manual_seed(seed + 3)
    h, z = torch.zeros(1, 3), torch.zeros(1, V_VARS, C)
    out = [policy.act(policy.q_values(rssm, goal, critic, h, z, CFG.k_act, CFG, gen)[0], CFG, explore, tiebreak) for _ in range(n)]
    return np.array([o["action"] for o in out]), np.array([o["tied"] for o in out]), np.array([o["explored"] for o in out])


@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.02, 0.03, 0.05])
def test_without_value_information_the_policy_is_uniform(sigma):
    actions, tied, _ = _play(np.full(A, -10.0), sigma)
    assert tied.mean() >= 0.99
    counts = np.bincount(actions, minlength=A)
    chi2 = ((counts - len(actions) / A) ** 2 / (len(actions) / A)).sum()
    assert chi2 < 18.47  # chi-square, 4 degrees of freedom, p = 0.001


def _v(d, gamma=0.9, g=50.0, s=-1.0):
    return s * (1 - gamma ** (d - 1)) / (1 - gamma) + gamma ** (d - 1) * g


@pytest.mark.parametrize("distance", [20, 36])
@pytest.mark.parametrize("sigma", [0.0, 0.05])
def test_information_at_the_horizon_is_followed(distance, sigma):
    """The goal `distance` steps away: action 2 leads to a state distance - 1 from the goal, the four others leave
    the agent at `distance`. Every greedy step must pick action 2, never as a tie."""
    values = np.full(A, _v(distance))
    values[2] = _v(distance - 1)
    gap = CFG.gamma * (values[2] - values[0])
    assert gap == pytest.approx(horizon_information_gap(0.9, 50.0, -1.0, distance))
    assert gap >= 2 * CFG.tie_tol_abs - 1e-12  # the informative gap is at least twice the tolerance
    actions, tied, explored = _play(values, sigma)
    greedy = ~explored
    assert (actions[greedy] == 2).all() and not tied[greedy].any()


def test_the_tolerance_is_the_protocol_s_not_the_data_s():
    assert CFG.tie_tol_abs == pytest.approx(0.5 * 6 * 0.9 ** 35, rel=1e-12)
    assert AgentConfig(episode_horizon=20).tie_tol_abs == pytest.approx(0.5 * 6 * 0.9 ** 19)
