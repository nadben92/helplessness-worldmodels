"""unroll_observation encodes the window's images in one pass and decodes in
one pass after the recurrence (audit 7.1): the same computation as the
step-by-step version -- identical latent samples for a given generator,
and the same values up to floating-point reduction order."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.loop import unroll_observation
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig


def _stepwise(rssm, x_seq, a_seq, is_first_seq, generator):
    B, T = x_seq.shape[:2]
    a_onehot = F.one_hot(a_seq, num_classes=rssm.cfg.n_actions).float()
    h, z = rssm.initial_state(B, x_seq.device)
    a_prev = torch.zeros(B, rssm.cfg.n_actions)
    hs, zs, posts, priors, dec = [], [], [], [], []
    for t in range(T):
        out = rssm.observation_step(h, z, a_prev, x_seq[:, t], is_first_seq[:, t], generator=generator)
        h, z = out["h"], out["z"]
        hs.append(h), zs.append(z), posts.append(out["post_probs"]), priors.append(out["prior_probs"])
        dec.append(rssm.decode(h, z))
        a_prev = a_onehot[:, t]
    return {k: torch.stack(v, 1) for k, v in (("h", hs), ("z", zs), ("post_probs", posts), ("prior_probs", priors), ("decoded", dec))}


def test_batched_unroll_matches_the_step_by_step_unroll():
    torch.manual_seed(0)
    rssm = RSSM(RSSMConfig(image_size=(24, 24), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4))
    B, T = 3, 7
    x = torch.rand(B, T, 3, 24, 24)
    a = torch.randint(0, 5, (B, T))
    is_first = torch.zeros(B, T, dtype=torch.bool)
    is_first[:, 0] = True
    is_first[1, 4] = True
    with torch.no_grad():
        ref = _stepwise(rssm, x, a, is_first, torch.Generator().manual_seed(1))
        new = unroll_observation(rssm, x, a, is_first, generator=torch.Generator().manual_seed(1))
        latents_only = unroll_observation(rssm, x, a, is_first, generator=torch.Generator().manual_seed(1), decode=False)
    assert torch.equal(ref["z"], new["z"])  # same samples, same order
    for key in ("h", "post_probs", "prior_probs", "decoded"):
        assert torch.allclose(ref[key], new[key], atol=1e-5, rtol=1e-5), key
    assert "decoded" not in latents_only and torch.equal(latents_only["z"], new["z"])
