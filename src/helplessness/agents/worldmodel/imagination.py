"""Critic training in imagination and goal-head training (spec 4.1, 4.2,
4.4; Phase 5). Nothing here sends a gradient to the world model: every
latent state is detached before it reaches a head (heads._features), and
the rollouts run under no_grad.

Imagination (4.4): from every posterior state of a replay batch
(stop-gradient), `horizon` steps with the agent's own epsilon-greedy
policy (K_imag prior draws per action) and the RSSM's imagination_step.
With s_0 the start and s_1..s_H the imagined states, r_k = r(s_k),
c_k = c(s_k) (from the goal head) and V_slow the slow critic:

    R_k = r_{k+1} + gamma * c_{k+1} * [ (1 - lam) * V_slow(s_{k+1}) + lam * R_{k+1} ],   R_H = V_slow(s_H)
    loss = mean over starts and k < H of  w_k * (V(s_k) - sg(R_k))^2,   w_0 = c_0,  w_k = w_{k-1} * c_k

so a start that is the goal (c_0 ~ 0) does not count, and nothing counts
after an imagined arrival at the goal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.heads import (
    AgentConfig,
    Critic,
    GoalHead,
    continuation_from_goal,
    reward_from_goal,
    update_slow_critic,
)
from helplessness.agents.worldmodel.policy import epsilon_greedy, q_values
from helplessness.agents.worldmodel.rssm import RSSM


@torch.no_grad()
def imagine(rssm: RSSM, goal_head: GoalHead, critic: Critic, h0: torch.Tensor, z0: torch.Tensor, cfg: AgentConfig,
            generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Rollouts of cfg.horizon steps from the starts (B, ...). Returns h
    (H+1, B, h_dim) and z (H+1, B, n_vars, n_classes) -- index 0 is the
    start --, the actions (H, B) and the fraction of imagined choices
    whose greedy tie set held more than one action."""
    A = rssm.cfg.n_actions
    h, z = h0.detach(), z0.detach()
    hs, zs, actions, tied = [h], [z], [], []
    for _ in range(cfg.horizon):
        choice = epsilon_greedy(q_values(rssm, goal_head, critic, h, z, cfg.k_imag, cfg, generator), cfg, generator)
        out = rssm.imagination_step(h, z, F.one_hot(choice["action"], num_classes=A).to(h.dtype), generator=generator)
        h, z = out["h"], out["z"].detach()
        hs.append(h), zs.append(z), actions.append(choice["action"]), tied.append(choice["tied"])
    return {"h": torch.stack(hs), "z": torch.stack(zs), "actions": torch.stack(actions),
            "tie_fraction": torch.stack(tied).float().mean()}


def lambda_returns(r: torch.Tensor, c: torch.Tensor, v_slow: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """r, c, v_slow: (H+1, B), the reward, continuation and slow value OF
    each state s_0..s_H. Returns R (H+1, B) with R_H = V_slow(s_H) and
    R_k = r_{k+1} + gamma * c_{k+1} * ((1 - lam) * V_slow(s_{k+1}) + lam * R_{k+1})."""
    H = r.shape[0] - 1
    returns = [v_slow[H]]
    for k in range(H - 1, -1, -1):
        returns.append(r[k + 1] + gamma * c[k + 1] * ((1.0 - lam) * v_slow[k + 1] + lam * returns[-1]))
    return torch.stack(returns[::-1])


def critic_loss(critic: Critic, slow_critic: Critic, goal_head: GoalHead, rollout: dict, cfg: AgentConfig) -> dict[str, torch.Tensor]:
    """The weighted MSE of V on the lambda-returns of a rollout (spec 4.4).
    Gradient: the critic only."""
    h, z = rollout["h"], rollout["z"]
    with torch.no_grad():
        p = goal_head.p_goal(h, z)
        r, c = reward_from_goal(p, cfg), continuation_from_goal(p)
        returns = lambda_returns(r, c, slow_critic(h, z), cfg.gamma, cfg.lam)
        weights = torch.cumprod(c, dim=0)  # w_0 = c_0, w_k = w_{k-1} * c_k
    value = critic(h[:-1], z[:-1])
    loss = (weights[:-1] * (value - returns[:-1]) ** 2).mean()
    return {"critic_loss": loss, "value_mean": value.detach().mean(), "return_mean": returns[:-1].mean(),
            "weight_mean": weights[:-1].mean()}


def goal_head_loss(goal_head: GoalHead, h: torch.Tensor, z: torch.Tensor, reward: torch.Tensor) -> dict[str, torch.Tensor]:
    """BCE of g(h, z) against 1[r > 0] on the replay's posterior states
    (spec 4.1). reward: the reward received ON ARRIVING at each state (the
    closed-loop replay's convention, closed_loop.py), same leading shape as
    h. The target is derived from the reward signal, never from the label."""
    target = (reward > 0).to(h.dtype)
    logits = goal_head(h, z)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    with torch.no_grad():
        predicted = logits > 0
        n_goal = target.sum()
        recall = (predicted & (target > 0)).sum() / n_goal.clamp(min=1)
    return {"goal_loss": loss, "goal_target_rate": target.mean(), "goal_recall": recall,
            "goal_false_positive_rate": (predicted & (target == 0)).float().mean()}


def train_heads_step(
    rssm: RSSM, goal_head: GoalHead, critic: Critic, slow_critic: Critic, goal_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer, h: torch.Tensor, z: torch.Tensor, reward: torch.Tensor, cfg: AgentConfig,
    generator: torch.Generator | None = None, critic_clip_norm: float | None = 100.0,
) -> dict[str, float]:
    """One update of the goal head and of the critic from a replay batch's
    posterior states h, z ((B, T, ...), detached here) and arrival rewards
    (B, T). The critic starts from every posterior state (spec 4.4), its
    gradient norm clipped at critic_clip_norm (4.2: 100), then the slow
    critic moves by EMA. Returns detached scalars."""
    h = h.detach().reshape(-1, h.shape[-1])
    z = z.detach().reshape(-1, *z.shape[-2:])
    reward = torch.as_tensor(reward, device=h.device).reshape(-1)

    goal_optimizer.zero_grad()
    goal = goal_head_loss(goal_head, h, z, reward)
    goal["goal_loss"].backward()
    goal_optimizer.step()

    rollout = imagine(rssm, goal_head, critic, h, z, cfg, generator)
    critic_optimizer.zero_grad()
    crit = critic_loss(critic, slow_critic, goal_head, rollout, cfg)
    crit["critic_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float("inf") if critic_clip_norm is None else critic_clip_norm)
    critic_optimizer.step()
    update_slow_critic(slow_critic, critic, cfg.slow_tau)
    return {**{k: float(v.detach()) for k, v in {**goal, **crit}.items()}, "critic_grad_norm": float(grad_norm),
            "imagined_tie_fraction": float(rollout["tie_fraction"])}
