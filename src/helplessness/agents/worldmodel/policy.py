"""The agent's policy (spec 4.3, Phase 5): epsilon-greedy on a one-step
lookahead, no actor.

    h'_a = GRU(h, z, a),   z'_k ~ p(. | h'_a),   k = 1..K
    Q(h, z, a) = (1/K) sum_k [ r(h'_a, z'_k) + gamma * c(h'_a, z'_k) * V(h'_a, z'_k) ]

with r and c read from the goal head (heads.py). The lookahead calls the
RSSM's own imagination_step (GRU, prior, latent_hook) -- never a second
forward pass -- and NEVER the s-space measurement bridge
(WorldModelAgent.transition_distribution): the policy has no state readout.

Action choice: with probability epsilon a uniform action; otherwise the
argmax with a RANDOM tie-break among the actions with Q >= max Q - tol,
tol ABSOLUTE in units of value: half the smallest Q gap that information at
the episode horizon produces (heads.horizon_information_gap; 0.0751 for E2's
episode). Q gaps of the order of the critic's noise are ties, so an agent
without value information plays uniformly, as E2's tabular agent does
(docs/e3/DECISIONS.md, 2026-10-01 (12); it replaces the relative tolerance
of 2026-09-30 (5), kept only to read the runs made with it). The tolerance has one implementation
(tie_tolerance); the pick has two callers' forms: real steps draw from the
numpy explore / tiebreak streams through evaluation.tiebreak
.argmax_random_tie (E2's), imagination picks a whole batch on the device
from a torch generator (epsilon_greedy) -- the same tie SET in both
(tested).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead, continuation_from_goal, reward_from_goal
from helplessness.agents.worldmodel.rssm import RSSM, sample_categorical_index
from helplessness.evaluation.tiebreak import argmax_random_tie


@torch.no_grad()
def lookahead(rssm: RSSM, h: torch.Tensor, z: torch.Tensor, K: int, generator: torch.Generator | None = None,
              context: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """For every action: h'_a (B, A, h_dim) and K prior draws z'_k
    (B, A, K, n_vars, n_classes), one-hot. rssm.imagination_step gives
    h'_a and the prior (latent_hook applied, as everywhere); the K draws
    are taken from that prior."""
    A, C = rssm.cfg.n_actions, rssm.cfg.n_classes
    batch = h.shape[0]
    h_next, z_next = [], []
    for a in range(A):
        a_onehot = F.one_hot(torch.full((batch,), a, device=h.device, dtype=torch.long), num_classes=A).to(h.dtype)
        out = rssm.imagination_step(h, z, a_onehot, context=context, generator=generator)
        prior = out["prior_probs"].unsqueeze(1).expand(-1, K, -1, -1)
        idx = sample_categorical_index(prior, generator)
        h_next.append(out["h"])
        z_next.append(F.one_hot(idx, num_classes=C).to(h.dtype))
    return torch.stack(h_next, dim=1), torch.stack(z_next, dim=1)


@torch.no_grad()
def q_values(rssm: RSSM, goal_head: GoalHead, critic: Critic, h: torch.Tensor, z: torch.Tensor, K: int, cfg: AgentConfig,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Q(h, z, a) for every action, (B, A): the mean over K prior draws of
    r + gamma * c * V at the predicted next latent state (spec 4.3)."""
    h_next, z_next = lookahead(rssm, h, z, K, generator)
    h_k = h_next.unsqueeze(2).expand(-1, -1, K, -1)
    flat_h, flat_z = h_k.reshape(-1, h_k.shape[-1]), z_next.reshape(-1, *z_next.shape[-2:])
    flat_h, flat_z = rssm.latent_hook(flat_h, flat_z, {})
    p = goal_head.p_goal(flat_h, flat_z)
    q = reward_from_goal(p, cfg) + cfg.gamma * continuation_from_goal(p) * critic(flat_h, flat_z)
    return q.reshape(h.shape[0], rssm.cfg.n_actions, K).mean(dim=-1)


def tie_tolerance(Q: torch.Tensor, cfg: AgentConfig) -> torch.Tensor:
    """(B,) tolerance of each row's tie-break: max(tie_tol_min, tie_tol_abs), the same for every row; LEGACY
    configs (tie_tol_rel set, runs before 2026-10-01 (12)): max(tie_tol_min, tie_tol_rel * std(Q(h, z, .)))."""
    if cfg.tie_tol_abs is not None:
        return torch.full(Q.shape[:-1], max(cfg.tie_tol_min, cfg.tie_tol_abs), dtype=Q.dtype, device=Q.device)
    return torch.clamp(cfg.tie_tol_rel * Q.std(dim=-1, unbiased=False), min=cfg.tie_tol_min)


def tie_set(Q: torch.Tensor, cfg: AgentConfig) -> torch.Tensor:
    """(B, A) bool: the actions within the tolerance of the row's max."""
    return Q >= (Q.max(dim=-1, keepdim=True).values - tie_tolerance(Q, cfg).unsqueeze(-1))


@torch.no_grad()
def epsilon_greedy(Q: torch.Tensor, cfg: AgentConfig, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """A batch of epsilon-greedy choices on the device (imagination): (B,)
    action, tied (more than one action in the greedy tie set) and explored
    (the uniform branch was taken). All draws from `generator`."""
    batch, A = Q.shape
    tied = tie_set(Q, cfg)
    scores = torch.rand(Q.shape, generator=generator, device=Q.device).masked_fill(~tied, -1.0)
    greedy = scores.argmax(dim=-1)
    explored = torch.rand(batch, generator=generator, device=Q.device) < cfg.epsilon
    uniform = torch.randint(0, A, (batch,), generator=generator, device=Q.device)
    return {"action": torch.where(explored, uniform, greedy), "tied": tied.sum(dim=-1) > 1, "explored": explored}


def act(Q_row: torch.Tensor, cfg: AgentConfig, explore_rng: np.random.Generator, tiebreak_rng: np.random.Generator) -> dict:
    """ONE real step's choice, from E2's separate numpy streams: explore_rng
    decides the uniform branch and draws its action; otherwise
    evaluation.tiebreak.argmax_random_tie picks among the tie set with
    tiebreak_rng (the stream is consumed only on greedy steps, as in E2).
    Q_row: (A,). Returns action (int), tied and explored (bool)."""
    tied = bool(tie_set(Q_row.unsqueeze(0), cfg)[0].sum() > 1)
    if explore_rng.random() < cfg.epsilon:
        return {"action": int(explore_rng.integers(Q_row.shape[0])), "tied": tied, "explored": True}
    tol = float(tie_tolerance(Q_row.unsqueeze(0), cfg)[0])
    action = int(argmax_random_tie(Q_row.detach().cpu().numpy()[None, :], tiebreak_rng, tol=tol)[0])
    return {"action": action, "tied": tied, "explored": False}


class LatentFilter:
    """The agent's latent state along a real episode (spec 4.3, "filtrage
    a chaque pas reel"): h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}), z_t drawn
    from the posterior given the observation; reset at each episode start.
    One environment (batch of 1). Uses RSSM.observation_step, as training."""

    def __init__(self, rssm: RSSM, device: torch.device, generator: torch.Generator | None = None):
        self.rssm, self.device, self.generator = rssm, device, generator
        self.h, self.z = rssm.initial_state(1, device)
        self.a_prev = torch.zeros(1, rssm.cfg.n_actions, device=device)

    @torch.no_grad()
    def observe(self, x: torch.Tensor, is_first: bool, estimator=None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (C, H, W) float in [0, 1] on the device. Returns (h_t, z_t), each with a batch dimension of 1."""
        out = self.rssm.observation_step(self.h, self.z, self.a_prev, x.unsqueeze(0), torch.tensor([is_first], device=self.device),
                                         estimator=estimator, generator=self.generator)
        self.h, self.z = out["h"], out["z"]
        return self.h, self.z

    def set_action(self, action: int) -> None:
        self.a_prev = F.one_hot(torch.tensor([action], device=self.device), num_classes=self.rssm.cfg.n_actions).to(self.h.dtype)
