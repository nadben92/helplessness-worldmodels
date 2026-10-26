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
                      the posterior of the true image, the true reward, greedy on Q_or. Variant "mixed" (E7's
                      anti-oracle, docs/e6/DECISIONS.md (55)): from a cell where `mix` is True, each particle takes with
                      probability p the successor table `alt_succ` (oracle_q's path: posterior of that cell's image,
                      true reward) and otherwise the prior (the model variant); greedy on the expected mixed one-step Q.
                      The particle's cell is tracked: alt_succ's, or the world's successor of the action in the model
                      branch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.heads import continuation_from_goal, reward_from_goal
from helplessness.agents.worldmodel.policy import lookahead, q_values, tie_set
from helplessness.agents.worldmodel.rssm import sample_categorical_index
from helplessness.probing.frozen_episodes import FrozenAgent, TestWorld, fresh_start_contexts

VARIANTS = ("model", "oracle", "mixed")


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
             generator: torch.Generator | None = None, succ: np.ndarray | None = None, h_mode: str = "action") -> torch.Tensor:
    """(n, A): the oracle one-step Q (module docstring) at the contexts (h, z) of the true states `states` (n,). succ
    (n_states, A): the successor table the planner is shown (default the world's, the oracle; the conveyor's is E7's
    anti-oracle). h_mode: "action" (default, D4's oracle) h'_a = GRU(h, z, a); "mean" (E7's anti-oracle,
    docs/e6/DECISIONS.md (57)) the mean of the h'_a over the actions for every action, and one set of posterior draws per
    distinct successor -- the action then reaches Q only through the successor's cell (equal successors, equal Q)."""
    rssm, cfg = agent.rssm, agent.cfg
    succ = world.succ if succ is None else np.asarray(succ)
    n, A = h.shape[0], rssm.cfg.n_actions
    hs = [rssm.gru_step(h, z, _onehot(torch.full((n,), a, device=h.device, dtype=torch.long), A, h.dtype)) for a in range(A)]
    if h_mode == "mean":
        hs = [torch.stack(hs).mean(dim=0)] * A
    elif h_mode != "action":
        raise ValueError(f"h_mode {h_mode!r}")
    out, drawn = [], []
    for a in range(A):
        h_a = hs[a]
        s_a = succ[np.asarray(states), a]
        same = [zz for ss, zz in drawn if h_mode == "mean" and np.array_equal(ss, s_a)]
        z_a = same[0] if same else _posterior_draws(agent, h_a, embeds[torch.as_tensor(s_a, device=h.device)], K, generator)
        drawn.append((s_a, z_a))
        r, c = _true_reward(agent, world, s_a, h.device)
        v = agent.critic(h_a.unsqueeze(1).expand(-1, K, -1).reshape(n * K, -1), z_a.reshape(n * K, *z_a.shape[-2:])).reshape(n, K)
        out.append((r[:, None] + cfg.gamma * c[:, None] * v).mean(dim=1))
    return torch.stack(out, dim=1)


@torch.no_grad()
def mixed_policy_q(agent: FrozenAgent, world: TestWorld, embeds: torch.Tensor, h: torch.Tensor, z: torch.Tensor, states: np.ndarray,
                   p_mix: float, mix: np.ndarray, alt_succ: np.ndarray, act_generator: torch.Generator | None,
                   alt_generator: torch.Generator | None, h_mode: str = "mean") -> torch.Tensor:
    """(n, A): the policy's Q with the anti-oracle at k = 1 (docs/e6/DECISIONS.md (55)): policy.q_values (K_act draws,
    act_generator: E6's consumption), and from the states where `mix` is True, (1 - p) Q_model + p Q_alt, Q_alt = oracle_q
    on the successor table alt_succ (K_act posterior draws, alt_generator; h_mode as oracle_q's, "mean" by default ((57)):
    at p = 1 every action reaching the same cell has the same Q). p = 0 returns q_values' tensor itself."""
    q = q_values(agent.rssm, agent.goal_head, agent.critic, h, z, agent.cfg.k_act, agent.cfg, act_generator)
    on = np.asarray(mix, dtype=bool)[np.asarray(states)]
    if p_mix == 0 or not on.any():
        return q
    q_alt = oracle_q(agent, world, embeds, h, z, states, agent.cfg.k_act, alt_generator, succ=alt_succ, h_mode=h_mode)
    return torch.where(torch.as_tensor(on, device=h.device)[:, None], (1.0 - p_mix) * q + p_mix * q_alt, q)


@torch.no_grad()
def sampled_mixed_policy_q(agent: FrozenAgent, world: TestWorld, embeds: torch.Tensor, h: torch.Tensor, z: torch.Tensor, states: np.ndarray,
                           p_mix: float, mix: np.ndarray, alt_succ: np.ndarray, act_generator: torch.Generator | None,
                           alt_generator: torch.Generator | None, shared: bool = False) -> torch.Tensor:
    """(n, A): the policy's Q with the anti-oracle BY DRAWS (docs/e6/DECISIONS.md (60)). The policy's K_act prior draws per
    action (policy.lookahead with act_generator: E6's draws); from the states where `mix` is True, each draw is replaced
    with probability p by a draw of the posterior of alt_succ's cell (the common h of (57): the mean of the h'_a; true r and
    c), independently per (action, draw), or with shared=True by one Bernoulli per row for every action and draw; Q(a) = the
    mean of the K values, the expression of policy.q_values term for term. p = 0 returns q_values' tensor itself."""
    rssm, cfg = agent.rssm, agent.cfg
    on = np.asarray(mix, dtype=bool)[np.asarray(states)]
    if p_mix == 0 or not on.any():
        return q_values(rssm, agent.goal_head, agent.critic, h, z, cfg.k_act, cfg, act_generator)
    n, A, K = h.shape[0], rssm.cfg.n_actions, cfg.k_act
    h_next, z_next = lookahead(rssm, h, z, K, act_generator)
    h_bar = h_next.mean(dim=1)
    z_alt, drawn = [], []
    for a in range(A):
        s_a = np.asarray(alt_succ)[np.asarray(states), a]
        same = [zz for ss, zz in drawn if np.array_equal(ss, s_a)]
        z_a = same[0] if same else _posterior_draws(agent, h_bar, embeds[torch.as_tensor(s_a, device=h.device)], K, alt_generator)
        drawn.append((s_a, z_a))
        z_alt.append(z_a)
    z_alt = torch.stack(z_alt, dim=1)                                                       # (n, A, K, V, C)
    shape = (n, 1, 1) if shared else (n, A, K)
    use = (torch.rand(shape, generator=alt_generator, device=h.device) < p_mix).expand(n, A, K)
    use = use & torch.as_tensor(on, device=h.device)[:, None, None]
    flat_use = use.reshape(-1)
    flat_h = torch.where(use[..., None], h_bar[:, None, None, :].expand(-1, A, K, -1), h_next.unsqueeze(2).expand(-1, -1, K, -1)).reshape(n * A * K, -1)
    flat_z = torch.where(use[..., None, None], z_alt, z_next).reshape(n * A * K, *z_next.shape[-2:])
    flat_h, flat_z = rssm.latent_hook(flat_h, flat_z, {})
    p = agent.goal_head.p_goal(flat_h, flat_z)
    alt_cells = np.repeat(np.asarray(alt_succ)[np.asarray(states)][:, :, None], K, axis=2).reshape(-1)
    r_alt, c_alt = _true_reward(agent, world, alt_cells, h.device)
    r = torch.where(flat_use, r_alt, reward_from_goal(p, cfg))
    c = torch.where(flat_use, c_alt, continuation_from_goal(p))
    q = r + cfg.gamma * c * agent.critic(flat_h, flat_z)
    return q.reshape(n, A, K).mean(dim=-1)


def greedy(Q: torch.Tensor, agent: FrozenAgent, generator: torch.Generator | None = None) -> torch.Tensor:
    """(B,) the greedy action: a uniform pick among the training's tie set (policy.tie_set), epsilon = 0."""
    tied = tie_set(Q, agent.cfg)
    return torch.rand(Q.shape, generator=generator, device=Q.device).masked_fill(~tied, -1.0).argmax(dim=-1)


@torch.no_grad()
def k_step_q(agent: FrozenAgent, world: TestWorld, h: torch.Tensor, z: torch.Tensor, states: np.ndarray | None, k_max: int, variant: str,
             n_particles: int, k_inner: int, generator: torch.Generator | None = None, embeds: torch.Tensor | None = None,
             n_trace: int = 0, p_mix: float = 0.0, mix: np.ndarray | None = None, alt_succ: np.ndarray | None = None,
             h_mode: str = "mean") -> dict:
    """Q_k for k = 1..k_max at the contexts (h, z) (module docstring). states (n,): the contexts' true states (the oracle
    and the mixed variant). p_mix, mix (n_states,) bool, alt_succ (n_states, A), h_mode: the mixed variant's probability,
    cells, successor table and h in its alt branch (oracle_q's h_mode; "mean": the particle's h after an alt step is the
    mean of GRU(h, z, a) over the actions). Returns Q (n, A, k_max) and, with n_trace > 0, the first n_trace particles of every (context, action) along the
    rollout: trace_h (k_max, n, A, n_trace, h_dim), trace_z (k_max, n, A, n_trace, V, C), trace_actions (k_max, n, A,
    n_trace) (the action that LED to each step: the first action at step 1), trace_states (oracle: the true states)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {VARIANTS}")
    rssm, cfg = agent.rssm, agent.cfg
    n, A, K = h.shape[0], rssm.cfg.n_actions, int(n_particles)
    device = h.device
    first_actions = torch.arange(A, device=device).view(1, A, 1).expand(n, A, K).reshape(-1)

    def mixed_q(h_, z_, cells):  # the expected one-step Q of the mixed variant
        q = q_values(rssm, agent.goal_head, agent.critic, h_, z_, int(k_inner), cfg, generator)
        if p_mix > 0:
            q_alt = oracle_q(agent, world, embeds, h_, z_, cells, int(k_inner), generator, succ=alt_succ, h_mode=h_mode)
            on = torch.as_tensor(mix[cells], device=device)[:, None]
            q = torch.where(on, (1.0 - p_mix) * q + p_mix * q_alt, q)
        return q

    def mixed_step(h_, z_, act, cells):  # one imagined step: alt_succ's branch with probability p_mix from a mix cell
        out = rssm.imagination_step(h_, z_, _onehot(act, A, h_.dtype), generator=generator)
        a_np = act.cpu().numpy()
        alt = alt_succ[cells, a_np]
        h_alt = out["h"]
        if h_mode == "mean":
            h_alt = torch.stack([rssm.gru_step(h_, z_, _onehot(torch.full((len(cells),), b, device=device, dtype=torch.long), A, h_.dtype))
                                 for b in range(A)]).mean(dim=0)
        z_alt = _posterior_draws(agent, h_alt, embeds[torch.as_tensor(alt, device=device)], 1, generator)[:, 0]
        use = mix[cells] & (torch.rand(len(cells), generator=generator, device=device).cpu().numpy() < p_mix)
        on = torch.as_tensor(use, device=device)
        z_new = torch.where(on[:, None, None], z_alt, out["z"])
        h_new = torch.where(on[:, None], h_alt, out["h"])
        pg = agent.goal_head.p_goal(h_new, z_new)
        r_alt, c_alt = _true_reward(agent, world, alt, device)
        r_new = torch.where(on, r_alt, reward_from_goal(pg, cfg))
        c_new = torch.where(on, c_alt, continuation_from_goal(pg))
        return h_new, z_new, r_new, c_new, np.where(use, alt, world.succ[cells, a_np])

    if variant == "mixed":
        if embeds is None or states is None or mix is None or alt_succ is None:
            raise ValueError("the mixed variant needs the states, the embeddings, mix and alt_succ")
        mix, alt_succ = np.asarray(mix, dtype=bool), np.asarray(alt_succ)
        root = torch.arange(n, device=device).repeat_interleave(A * K)
        flat_h, flat_z, r, c, flat_s = mixed_step(h[root], z[root], first_actions, np.repeat(np.asarray(states), A * K))
    elif variant == "model":  # step 1: policy.q_values' own computation, term for term
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
        elif variant == "mixed":
            act = greedy(mixed_q(flat_h, flat_z, flat_s), agent, generator)
            flat_h, flat_z, r, c, flat_s = mixed_step(flat_h, flat_z, act, flat_s)
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
