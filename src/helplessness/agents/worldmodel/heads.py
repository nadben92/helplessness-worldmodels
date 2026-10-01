"""The agent's heads on the world model's latent state (spec section 4,
Phase 5): the goal head g(h, z) = p_goal (4.1) and the critic V(h, z)
(4.2), both MLPs on sg([h, z]) -- no gradient of either ever reaches the
world model -- and the slow critic (an EMA copy, 4.2).

Reward and continuation are read from the goal head, never from a state
readout (spec 4.1, amended: docs/e3/SPEC_STATUS.md section 4.1):

    r = goal_reward * p_goal + step_reward * (1 - p_goal)     (50 * p - 1 * (1 - p))
    c = 1 - p_goal

E2's convention: the step that reaches the goal earns the goal reward
INSTEAD of the step cost, and the goal is absorbing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AgentConfig:
    """Spec section 4's constants (defaults = the spec's; the episode
    values gamma, goal_reward, step_reward are E2's and come from the
    experiment's episode config, never retyped in a run config)."""

    gamma: float = 0.9
    goal_reward: float = 50.0
    step_reward: float = -1.0
    epsilon: float = 0.1
    k_act: int = 16  # prior draws per action to act (4.3)
    k_imag: int = 4  # prior draws per action in imagination (4.3)
    horizon: int = 15  # imagined steps (4.4)
    lam: float = 0.95  # lambda-returns (4.4)
    slow_tau: float = 0.02  # EMA of the slow critic (4.2)
    # Tie-break tolerance (4.3, amended twice; docs/e3/DECISIONS.md 2026-10-01 (12)): ABSOLUTE, in units of value,
    # tie_fraction x the smallest Q gap that information at the episode horizon produces
    # (horizon_information_gap); computed from gamma, the rewards and episode_horizon unless tie_tol_abs is given.
    tie_fraction: float = 0.5
    episode_horizon: int = 36  # E2's H (max_steps)
    tie_tol_abs: float | None = None
    tie_tol_rel: float | None = None  # LEGACY (runs before 2026-10-01 (12)): max(tie_tol_min, tie_tol_rel * std(Q)); read-only
    tie_tol_min: float = 1e-6
    hidden_dim: int = 128

    def __post_init__(self):
        if self.tie_tol_abs is None and self.tie_tol_rel is None:
            object.__setattr__(self, "tie_tol_abs", self.tie_fraction * horizon_information_gap(
                self.gamma, self.goal_reward, self.step_reward, self.episode_horizon))
        if self.tie_tol_abs is not None and self.tie_tol_rel is not None:
            raise ValueError("AgentConfig: an absolute tie tolerance and a legacy relative one; choose one")


def horizon_information_gap(gamma: float, goal_reward: float, step_reward: float, horizon: int) -> float:
    """The smallest Q gap that information at `horizon` steps produces (docs/e3/DECISIONS.md, 2026-10-01 (12)).
    E2's episode: step_reward per step, goal_reward on the step that reaches the goal (instead of the step cost), the
    goal absorbing. A state d steps from the goal along a deterministic path has
        V(d) = step_reward (1 - gamma^(d-1)) / (1 - gamma) + gamma^(d-1) goal_reward,
    and the one-step lookahead's Q gap between an action that brings it one step closer and one that leaves the
    distance unchanged (the smallest gap an informative action makes) is
        gamma (V(d-1) - V(d)) = ((1 - gamma) goal_reward - step_reward) gamma^(d-1).
    With gamma 0.9, rewards 50 / -1 and d = 36: 6 x 0.9^35 = 0.1502."""
    return ((1.0 - gamma) * goal_reward - step_reward) * gamma ** (horizon - 1)


def _head_mlp(in_dim: int, hidden_dim: int, zero_init: bool) -> nn.Sequential:
    """Two hidden layers. zero_init: the output layer starts at zero
    (weights and bias), as DreamerV3's reward and critic heads: an untrained
    head gives the SAME output for every state, so an agent that has learnt
    nothing has no preference -- every Q-value ties and the random tie-break
    explores uniformly (as E2's tabular agent on unvisited pairs), instead
    of following the arbitrary ranking of a random initialisation
    (docs/e3/DECISIONS.md, 2026-09-30 (5))."""
    net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
    if zero_init:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def _features(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """sg([h, z]): z (..., n_vars, n_classes) flattened next to h (..., h_dim)."""
    return torch.cat([h, z.reshape(*h.shape[:-1], -1)], dim=-1).detach()


class GoalHead(nn.Module):
    """g(h, z): the LOGIT of p_goal = P(this latent state is the goal),
    an MLP on sg([h, z]) (spec 4.1). Trained by BCE against 1[r_t > 0] on
    the replay's posterior states (imagination.goal_head_loss): the target
    comes from the reward signal, never from the state label."""

    def __init__(self, h_dim: int, latent_dim: int, hidden_dim: int = 128, zero_init: bool = True):
        super().__init__()
        self.net = _head_mlp(h_dim + latent_dim, hidden_dim, zero_init)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(_features(h, z)).squeeze(-1)

    def p_goal(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(h, z))


class Critic(nn.Module):
    """V(h, z): an MLP on sg([h, z]) (spec 4.2)."""

    def __init__(self, h_dim: int, latent_dim: int, hidden_dim: int = 128, zero_init: bool = True):
        super().__init__()
        self.net = _head_mlp(h_dim + latent_dim, hidden_dim, zero_init)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(_features(h, z)).squeeze(-1)


def make_slow_critic(critic: Critic) -> Critic:
    """A frozen copy of the critic: the target network of the lambda-returns."""
    slow = copy.deepcopy(critic)
    for p in slow.parameters():
        p.requires_grad_(False)
    return slow


@torch.no_grad()
def update_slow_critic(slow: Critic, critic: Critic, tau: float) -> None:
    """slow <- (1 - tau) * slow + tau * critic (spec 4.2: EMA, tau = 0.02)."""
    for p_slow, p in zip(slow.parameters(), critic.parameters()):
        p_slow.mul_(1.0 - tau).add_(p.detach(), alpha=tau)


def reward_from_goal(p_goal: torch.Tensor, cfg: AgentConfig) -> torch.Tensor:
    """r = goal_reward * p_goal + step_reward * (1 - p_goal)."""
    return cfg.goal_reward * p_goal + cfg.step_reward * (1.0 - p_goal)


def continuation_from_goal(p_goal: torch.Tensor) -> torch.Tensor:
    """c = 1 - p_goal: the goal is the only terminal state."""
    return 1.0 - p_goal
