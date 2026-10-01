"""The goal's (terminal) frame in the world model's loss (docs/e3/DECISIONS.md, 2026-10-01 (12), d).

A synthetic episode that ends at the goal is stored as the closed-loop collector stores it (closed_loop.py: every
observation with the action taken from it and the reward of arriving at it; the goal's observation last, terminal,
action 0, never is_first), followed by the next episode's first observation. A replay window over it (the same
replay.windows the training sampler uses) is passed through the training loss: the transition
(h, z, a) -> goal frame must enter the dynamic KL with the same weight as every other step -- no terminal mask, no
reset before the goal frame -- and the action that led to the goal must be the one the GRU receives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from helplessness.agents.worldmodel.loop import WorldModelLossConfig, unroll_observation, world_model_loss
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, LatentEstimator, RSSMConfig, categorical_kl

CFG = RSSMConfig(image_size=(24, 24), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4)
GOAL = 3
PROBS = LatentEstimator("probs")  # deterministic: the comparison is exact


def _table(seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(6, 24, 24, 3), dtype=np.uint8)


def _replay(table):
    """Episode 1: states 1 -> 2 -> GOAL with actions 4 then 2; episode 2 starts at state 5."""
    buf = ReplayBuffer(capacity=10, table=table)
    buf.add(x=table[1], a=4, r=0.0, terminal=False, truncated=False, is_first=True, s=1)
    buf.add(x=table[2], a=2, r=-1.0, terminal=False, truncated=False, is_first=False, s=2)
    buf.add(x=table[GOAL], a=0, r=50.0, terminal=True, truncated=False, is_first=False, s=GOAL)  # as closed_loop.py stores it
    buf.add(x=table[5], a=1, r=0.0, terminal=False, truncated=False, is_first=True, s=5)
    return buf


def _model():
    torch.manual_seed(0)
    return RSSM(CFG)


def test_the_transition_into_the_goal_enters_the_dynamic_loss():
    table = _table()
    batch, labels = _replay(table).windows(np.array([0]), 4)  # the training sampler's window function
    assert labels.s[0].tolist() == [1, 2, GOAL, 5]
    assert batch.is_first[0].tolist() == [True, False, False, True] and batch.terminal[0].tolist() == [False, False, True, False]
    rssm, bg = _model(), torch.rand(3, 24, 24, generator=torch.Generator().manual_seed(1))
    loss_cfg = WorldModelLossConfig(k=2)
    loss = world_model_loss(rssm, batch, bg, loss_cfg, estimator=PROBS)
    # the loss's dynamic term is the plain mean of the per-step KL over the window's 4 steps, the goal step included
    x = torch.as_tensor(batch.x).float().permute(0, 1, 4, 2, 3) / 255.0
    out = unroll_observation(rssm, x, torch.as_tensor(batch.a), torch.as_tensor(batch.is_first), estimator=PROBS)
    per_step = categorical_kl(out["post_probs"][0], out["prior_probs"][0])
    assert loss["kl_dyn"].item() == pytest.approx(per_step.mean().item(), rel=1e-6)
    assert per_step[2] > 0  # the goal step has its own prior (from the transition) and its own posterior
    # the GRU step into the goal frame received the action taken at state 2 (a = 2), not the goal frame's own 0
    h_into_goal = rssm.gru_step(out["h"][:, 1], out["z"][:, 1], torch.nn.functional.one_hot(torch.tensor([2]), 5).float())
    assert torch.allclose(out["h"][:, 2], h_into_goal, atol=1e-6)


def test_the_goal_frame_moves_the_dynamic_loss_and_its_gradient():
    """Changing only the goal's image changes the dynamic KL and its gradient on the prior network: the goal step
    is not masked out of the loss."""
    rssm, bg = _model(), torch.rand(3, 24, 24, generator=torch.Generator().manual_seed(1))
    grads, values = [], []
    for seed in (0, 7):
        table = _table(0)
        table[GOAL] = np.random.default_rng(seed + 100).integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
        batch, _ = _replay(table).windows(np.array([0]), 4)
        rssm.zero_grad()
        loss = world_model_loss(rssm, batch, bg, WorldModelLossConfig(k=2), estimator=PROBS)
        loss["kl_dyn"].backward()
        values.append(loss["kl_dyn"].item())
        grads.append(torch.cat([p.grad.flatten() for p in rssm.prior_net.parameters()]).clone())
    assert values[0] != values[1] and not torch.allclose(grads[0], grads[1])


def test_a_window_that_ends_on_the_goal_keeps_it():
    """The window's last step can be the goal: the sampler's valid starts reach it (no window is cut before a
    terminal frame)."""
    buf = _replay(_table())
    starts = np.arange(len(buf) - 3 + 1)  # every valid start of a 3-step window (ReplayBuffer.sample's range)
    covered = {int(st) + t for st in starts for t in range(3)}
    assert {1, 2} <= covered  # the pre-goal frame and the goal frame (logical indices 1 and 2) are both sampled
    batch, labels = buf.windows(np.array([0]), 3)
    assert labels.s[0].tolist()[-1] == GOAL
