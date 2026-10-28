"""A correction of the recurrent state h that makes a frozen world model predict a given dynamics (E7, docs/E7_latent_manipulation.md,
docs/e6/DECISIONS.md). New module (2026-10-23); torch side only (the statistics of the corrections are
interventions.latent_direction's, numpy). Only h is changed, never z nor a weight.

For a situation (h_t, z_t) at a state s and, for every action a, a TARGET successor s'_a (the test environment's
controllable dynamics, or the exposed training dynamics for the reverse direction):

  target_a   = q(z | h'_a, e(x(s'_a))),  h'_a = GRU(h_t, z_t, a)      -- the posterior of the true next image, computed
                                                                        with the UNMODIFIED h (phase 0: the posterior
                                                                        depends on h far less than on the image);
  prior_a(d) = p(z | GRU(h_t + d, z_t, a));
  L(d)       = sum_a KL(target_a || prior_a(d)) + lam * |d|^2         -- KL exact, summed over the categorical variables
                                                                        (rssm.categorical_kl), no pixel loss.

optimise_delta minimises L for a batch of independent situations with Adam (elementwise, so the situations do not
interact), from d = 0 or a given start, until max_steps or a plateau (every situation's loss changed by less than tol,
relatively, over the last `patience` steps).

The CONSTRAINED correction (docs/e6/DECISIONS.md (47)) adds what must not move, as optimise_delta's `extra` term:
  mu_pos * D_pos(d) + mu_goal * (p_goal(h + d, z) - p_goal(h, z))^2
D_pos = KL(pos(h) || pos(h + d)), pos = soft_agent_position: the agent's position read in the DECODED image, a softmax over
the cells of -(the smallest squared distance of a cell's pixels to the agent's colour) / tau -- the decoding E4's
localisation reads (its argmax is exactly probe_validity.decoded_agent_cell). p_goal: the goal head's output, through a
forward pass that keeps the gradient (the head's own forward detaches its input). Without `extra` (or with mu_pos =
mu_goal = 0 through constraint_terms(...) = None) the optimisation is the unconstrained one, step for step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.rssm import RSSM, categorical_kl


def _onehot(a: int, n: int, A: int, device, dtype) -> torch.Tensor:
    return F.one_hot(torch.full((n,), a, device=device, dtype=torch.long), num_classes=A).to(dtype)


def target_posteriors(rssm: RSSM, h: torch.Tensor, z: torch.Tensor, successors: np.ndarray, templates_t: torch.Tensor) -> torch.Tensor:
    """(n, A, V, C): for each situation and action, the posterior of the target successor's image given the unmodified
    h'_a. successors: (n, A) target states."""
    n, A = h.shape[0], rssm.cfg.n_actions
    device = h.device
    out = []
    with torch.no_grad():
        for a in range(A):
            h_next = rssm.gru_step(h, z, _onehot(a, n, A, device, h.dtype))
            e = rssm.encoder(templates_t[torch.as_tensor(successors[:, a], device=device)])
            out.append(rssm.posterior_probs(h_next, e))
    return torch.stack(out, dim=1)


def prediction_loss(rssm: RSSM, h: torch.Tensor, z: torch.Tensor, delta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(n,): sum over actions of KL(target_a || prior(GRU(h + delta, z, a))), summed over the variables."""
    n, A = h.shape[0], rssm.cfg.n_actions
    total = torch.zeros(n, device=h.device, dtype=h.dtype)
    for a in range(A):
        h_next = rssm.gru_step(h + delta, z, _onehot(a, n, A, h.device, h.dtype))
        total = total + categorical_kl(target[:, a], rssm.prior_probs(h_next))
    return total


def soft_agent_position(rssm: RSSM, h: torch.Tensor, z: torch.Tensor, agent: torch.Tensor, kpx: int, tau: float) -> torch.Tensor:
    """(n, n_cells): the soft position of the agent in decode(h, z) (module docstring); differentiable in h. agent: (3, 1, 1)
    its colour in [0, 1]; kpx: pixels per cell."""
    dec = rssm.decode(h, z).clamp(0, 1)
    d = ((dec - agent) ** 2).sum(dim=1)
    n, H, W = d.shape
    R, C = H // kpx, W // kpx
    per_cell = d.reshape(n, R, kpx, C, kpx).amin(dim=(2, 4)).reshape(n, R * C)
    return torch.softmax(-per_cell / tau, dim=-1)


def p_goal_with_grad(goal_head, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """The goal head's p_goal(h, z) without its input's stop-gradient (the same network and weights)."""
    return torch.sigmoid(goal_head.net(torch.cat([h, z.reshape(h.shape[0], -1)], dim=-1)).squeeze(-1))


def constraint_terms(rssm: RSSM, goal_head, h: torch.Tensor, z: torch.Tensor, agent: torch.Tensor, kpx: int, mu_pos: float, mu_goal: float,
                     tau: float = 0.02):
    """The constrained correction's extra term, d -> (n,) (module docstring), or None when mu_pos = mu_goal = 0."""
    if mu_pos == 0 and mu_goal == 0:
        return None
    with torch.no_grad():
        pos0 = soft_agent_position(rssm, h, z, agent, kpx, tau)
        goal0 = p_goal_with_grad(goal_head, h, z)

    def extra(delta: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        if mu_pos:
            pos = soft_agent_position(rssm, h + delta, z, agent, kpx, tau)
            out = out + mu_pos * (pos0 * (torch.log(pos0 + 1e-12) - torch.log(pos + 1e-12))).sum(dim=-1)
        if mu_goal:
            out = out + mu_goal * (p_goal_with_grad(goal_head, h + delta, z) - goal0) ** 2
        return out

    return extra


def optimise_delta(rssm: RSSM, h: torch.Tensor, z: torch.Tensor, target: torch.Tensor, lam: float, lr: float = 0.05,
                   max_steps: int = 1000, tol: float = 1e-4, patience: int = 50, init: torch.Tensor | None = None,
                   extra=None) -> dict:
    """delta (n, h_dim) minimising L (module docstring) plus extra(delta) when given (constraint_terms); the weights stay
    frozen (no parameter of rssm gets a gradient).
    Returns delta, loss (n,) and kl (n,) at the end, kl0 (n,) at delta = 0, steps (the Adam steps taken), converged
    (stopped on the plateau)."""
    h, z, target = h.detach(), z.detach(), target.detach()
    delta = (torch.zeros_like(h) if init is None else init.detach().clone()).requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)
    history: list[torch.Tensor] = []
    with torch.no_grad():
        kl0 = prediction_loss(rssm, h, z, torch.zeros_like(h), target)
    steps, converged = 0, False
    with torch.enable_grad():
        for steps in range(1, max_steps + 1):
            opt.zero_grad()
            kl = prediction_loss(rssm, h, z, delta, target)
            loss = kl + lam * (delta ** 2).sum(dim=1)
            if extra is not None:
                loss = loss + extra(delta)
            loss.sum().backward()
            opt.step()
            history.append(loss.detach())
            if len(history) > patience:
                old, new = history[-patience - 1], history[-1]
                if bool(((old - new).abs() <= tol * old.abs().clamp_min(1e-8)).all()):
                    converged = True
                    break
    with torch.no_grad():
        kl = prediction_loss(rssm, h, z, delta, target)
        loss = kl + lam * (delta ** 2).sum(dim=1)
        if extra is not None:
            loss = loss + extra(delta)
    for p in rssm.parameters():
        if p.grad is not None:
            raise RuntimeError("a weight received a gradient: the model must stay frozen")
    return {"delta": delta.detach(), "loss": loss, "kl": kl, "kl0": kl0, "steps": steps, "converged": converged}
