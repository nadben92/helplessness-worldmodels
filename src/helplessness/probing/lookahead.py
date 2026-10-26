"""The value side of a frozen world-model agent's choice (E6/E7's diagnostics, docs/e6/DECISIONS.md (51)-(53)): the
critic's value of a cell, the one-step Q with the transition belief replaced by the truth (oracle), and the k-step Q of
a lookahead that follows the greedy one-step action after the first action. New module (2026-10-26); it calls the
RSSM's own methods (gru_step, posterior_probs, imagination_step), the policy's own lookahead / q_values / tie_set and the
heads' own reward and continuation -- never a reimplementation. Nothing is learned; no weight is touched.

  Oracle one-step Q   Q_or(a) = mean_k [r + gamma c V(h'_a, z'_k)], h'_a = GRU(h, z, a), z'_k ~ q(z | h'_a, e(x(s'_a))),
                      s'_a the TRUE successor of the context's state; r and c TRUE (goal_reward and c = 0 on the goal,
                      step_reward and c = 1 elsewhere). Only V is the model's.
  k-step Q            Q_k(a) = sum_{i<k} gamma^i (prod_{j<=i} c_j) r_{i+1} + gamma^k (prod_{j<=k} c_j) V(s_k), the mean
                      over n_particles particles per (context, action); after the first action every particle follows
                      the GREEDY one-step action of its variant (epsilon = 0, random tie-break in the training's tie set)
                      computed with k_inner draws. Variant "model": the prior (agents.worldmodel.policy.lookahead, then
                      rssm.imagination_step), r and c from the goal head, greedy on policy.q_values; for k = 1 it is
                      q_values exactly (the same draws from the same generator). Variant "oracle": the true successors,
                      the posterior of the true image, the true reward, greedy on Q_or.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.heads import continuation_from_goal, reward_from_goal
from helplessness.agents.worldmodel.policy import lookahead, q_values, tie_set
from helplessness.agents.worldmodel.rssm import sample_categorical_index
from helplessness.probing.frozen_episodes import FrozenAgent, TestWorld, fresh_start_contexts

VARIANTS = ("model", "oracle")


def embeddings(agent: FrozenAgent, world: TestWorld) -> torch.Tensor:
    """(n_states, embed_dim): the encoder's embedding of every state image (the encoder reads the image alone)."""
    with torch.no_grad():
        return agent.rssm.encoder(world.templates_t)


def cell_values(agent: FrozenAgent, world: TestWorld, states: np.ndarray, n_samples: int, generator: torch.Generator) -> np.ndarray:
    """(len(states), n_samples): the critic's V at the first step of an episode started on each state (E6's step-1
    context, frozen_episodes.fresh_start_contexts), one posterior draw per sample."""
    rep = np.repeat(np.asarray(states), int(n_samples))
    h, z = fresh_start_contexts(agent.rssm, world, rep, generator)
    with torch.no_grad():
        v = agent.critic(h, z)
    return v.cpu().numpy().reshape(len(states), int(n_samples))


def _onehot(actions: torch.Tensor, A: int, dtype) -> torch.Tensor:
    return F.one_hot(actions, num_classes=A).to(dtype)


def _true_reward(agent: FrozenAgent, world: TestWorld, s_next: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    """r and c of arriving at s_next (E2's convention: the goal reward instead of the step cost, the goal absorbing)."""
    goal = torch.as_tensor(np.asarray(s_next) == world.goal_state, device=device)
    r = torch.where(goal, agent.cfg.goal_reward, agent.cfg.step_reward).to(torch.float32)
    return r, (~goal).to(torch.float32)


def _posterior_draws(agent: FrozenAgent, h: torch.Tensor, e: torch.Tensor, K: int, generator) -> torch.Tensor:
    """(n, K, V, C) one-hot: K draws of the posterior q(z | h, e)."""
    post = agent.rssm.posterior_probs(h, e).unsqueeze(1).expand(-1, K, -1, -1)
    return F.one_hot(sample_categorical_index(post, generator), num_classes=agent.rssm.cfg.n_classes).to(h.dtype)


@torch.no_grad()
def oracle_q(agent: FrozenAgent, world: TestWorld, embeds: torch.Tensor, h: torch.Tensor, z: torch.Tensor, states: np.ndarray, K: int,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """(n, A): the oracle one-step Q (module docstring) at the contexts (h, z) of the true states `states` (n,)."""
    rssm, cfg = agent.rssm, agent.cfg
    n, A = h.shape[0], rssm.cfg.n_actions
    out = []
    for a in range(A):
        h_a = rssm.gru_step(h, z, _onehot(torch.full((n,), a, device=h.device, dtype=torch.long), A, h.dtype))
        s_a = world.succ[np.asarray(states), a]
        z_a = _posterior_draws(agent, h_a, embeds[torch.as_tensor(s_a, device=h.device)], K, generator)
        r, c = _true_reward(agent, world, s_a, h.device)
        v = agent.critic(h_a.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1), z_a.reshape(n * K, *z_a.shape[-2:])).reshape(n, K)
        out.append((r[:, None] + cfg.gamma * c[:, None] * v).mean(dim=1))
    return torch.stack(out, dim=1)


def greedy(Q: torch.Tensor, agent: FrozenAgent, generator: torch.Generator | None = None) -> torch.Tensor:
    """(B,) the greedy action: a uniform pick among the training's tie set (policy.tie_set), epsilon = 0."""
    tied = tie_set(Q, agent.cfg)
    return torch.rand(Q.shape, generator=generator, device=Q.device).masked_fill(~tied, -1.0).argmax(dim=-1)


@torch.no_grad()
def k_step_q(agent: FrozenAgent, world: TestWorld, h: torch.Tensor, z: torch.Tensor, states: np.ndarray | None, k_max: int, variant: str,
             n_particles: int, k_inner: int, generator: torch.Generator | None = None, embeds: torch.Tensor | None = None,
             n_trace: int = 0) -> dict:
    """Q_k for k = 1..k_max at the contexts (h, z) (module docstring). states (n,): the contexts' true states (the oracle
    only). Returns Q (n, A, k_max) and, with n_trace > 0, the first n_trace particles of every (context, action) along the
    rollout: trace_h (k_max, n, A, n_trace, h_dim), trace_z (k_max, n, A, n_trace, V, C), trace_actions (k_max, n, A,
    n_trace) (the action that LED to each step: the first action at step 1), trace_states (oracle: the true states)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    rssm, cfg = agent.rssm, agent.cfg
    n, A, K = h.shape[0], rssm.cfg.n_actions, int(n_particles)
    device = h.device
    if variant == "model":  # step 1: policy.q_values' own computation, term for term
        h_next, z_next = lookahead(rssm, h, z, K, generator)
        h_k = h_next.unsqueeze(2).expand(-1, -1, K, -1)
        flat_h, flat_z = h_k.reshape(-1, h_k.shape[-1]), z_next.reshape(-1, *z_next.shape[-2:])
        flat_h, flat_z = rssm.latent_hook(flat_h, flat_z, {})
        p = agent.goal_head.p_goal(flat_h, flat_z)
        r, c = reward_from_goal(p, cfg), continuation_from_goal(p)
        flat_s = None
    else:
        if embeds is None or states is None:
            raise ValueError("the oracle needs the states and the embeddings")
        hs, zs, ss = [], [], []
        for a in range(A):
            h_a = rssm.gru_step(h, z, _onehot(torch.full((n,), a, device=device, dtype=torch.long), A, h.dtype))
            s_a = world.succ[np.asarray(states), a]
            hs.append(h_a.unsqueeze(1).expand(-1, K, -1))
            zs.append(_posterior_draws(agent, h_a, embeds[torch.as_tensor(s_a, device=device)], K, generator))
            ss.append(np.repeat(s_a[:, None], K, axis=1))
        flat_h = torch.stack(hs, dim=1).reshape(n * A * K, -1)
        flat_z = torch.stack(zs, dim=1).reshape(n * A * K, *zs[0].shape[-2:])
        flat_s = np.stack(ss, axis=1).reshape(-1)
        r, c = _true_reward(agent, world, flat_s, device)
    v = agent.critic(flat_h, flat_z)
    Q = [(r + cfg.gamma * c * v).reshape(n, A, K).mean(dim=-1)]
    G, D = r, cfg.gamma * c
    first_actions = torch.arange(A, device=device).view(1, A, 1).expand(n, A, K).reshape(-1)
    trace = {"h": [], "z": [], "a": [], "s": []}

    def record(actions):
        if n_trace > 0:
            pick = lambda x: x.reshape(n, A, K, *x.shape[1:])[:, :, :n_trace]  # noqa: E731
            trace["h"].append(pick(flat_h).cpu()), trace["z"].append(pick(flat_z).cpu()), trace["a"].append(pick(actions).cpu())
            if flat_s is not None:
                trace["s"].append(flat_s.reshape(n, A, K)[:, :, :n_trace].copy())

    record(first_actions)
    for _ in range(1, int(k_max)):
        if variant == "model":
            act = greedy(q_values(rssm, agent.goal_head, agent.critic, flat_h, flat_z, int(k_inner), cfg, generator), agent, generator)
            out = rssm.imagination_step(flat_h, flat_z, _onehot(act, A, flat_h.dtype), generator=generator)
            flat_h, flat_z = out["h"], out["z"]
            p = agent.goal_head.p_goal(flat_h, flat_z)
            r, c = reward_from_goal(p, cfg), continuation_from_goal(p)
        else:
            act = greedy(oracle_q(agent, world, embeds, flat_h, flat_z, flat_s, int(k_inner), generator), agent, generator)
            flat_h = rssm.gru_step(flat_h, flat_z, _onehot(act, A, flat_h.dtype))
            flat_s = world.succ[flat_s, act.cpu().numpy()]
            flat_z = _posterior_draws(agent, flat_h, embeds[torch.as_tensor(flat_s, device=device)], 1, generator)[:, 0]
            r, c = _true_reward(agent, world, flat_s, device)
        v = agent.critic(flat_h, flat_z)
        G = G + D * r
        D = D * cfg.gamma * c
        Q.append((G + D * v).reshape(n, A, K).mean(dim=-1))
        record(act)
    res = {"Q": torch.stack(Q, dim=-1)}
    if n_trace > 0:
        res.update({"trace_h": torch.stack(trace["h"]), "trace_z": torch.stack(trace["z"]), "trace_actions": torch.stack(trace["a"])})
        if trace["s"]:
            res["trace_states"] = np.stack(trace["s"])
    return res
