"""loop.encode_windows (docs/e3/DECISIONS.md, 2026-09-29): the measurement
probes encode a large block of windows at most max_images images at a time.
Without a cap it is exactly unroll_observation's own encoding; with one, the
embeddings agree up to the convolutions' reduction order and the latent
draws of the unroll (step by step over the whole block) are the same."""

from __future__ import annotations

import pytest
import torch

from helplessness.agents.worldmodel.loop import encode_windows, to_chw_float, unroll_observation
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig


def _setup():
    torch.manual_seed(0)
    rssm = RSSM(RSSMConfig(image_size=(24, 24), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4))
    B, T = 5, 6
    x = torch.randint(0, 256, (B, T, 24, 24, 3), dtype=torch.uint8)
    a = torch.randint(0, 5, (B, T))
    is_first = torch.zeros(B, T, dtype=torch.bool)
    is_first[:, 0] = True
    return rssm, x, a, is_first


def test_uncapped_encoding_is_the_unroll_s_own_bit_for_bit():
    rssm, x, a, is_first = _setup()
    with torch.no_grad():
        ref = unroll_observation(rssm, to_chw_float(x, torch.device("cpu")), a, is_first, generator=torch.Generator().manual_seed(1),
                                 decode=False)
        embeds = encode_windows(rssm, x, torch.device("cpu"), max_images=None)
        new = unroll_observation(rssm, None, a, is_first, generator=torch.Generator().manual_seed(1), decode=False, embeds=embeds)
        big_cap = encode_windows(rssm, x.numpy(), torch.device("cpu"), max_images=10_000)
    for key in ("h", "z", "post_probs", "prior_probs"):
        assert torch.equal(ref[key], new[key]), key
    assert torch.equal(big_cap, embeds)


@pytest.mark.parametrize("max_images", [1, 7, 29])
def test_capped_encoding_gives_the_same_embeddings_and_draws(max_images):
    rssm, x, a, is_first = _setup()
    with torch.no_grad():
        whole = encode_windows(rssm, x, torch.device("cpu"))
        blocked = encode_windows(rssm, x, torch.device("cpu"), max_images=max_images)
        ref = unroll_observation(rssm, None, a, is_first, generator=torch.Generator().manual_seed(1), decode=False, embeds=whole)
        new = unroll_observation(rssm, None, a, is_first, generator=torch.Generator().manual_seed(1), decode=False, embeds=blocked)
    assert blocked.shape == whole.shape == (5, 6, 16)
    assert torch.allclose(blocked, whole, atol=1e-5, rtol=1e-5)
    # the same draws (z = onehot + probs - sg(probs) carries a rounding residue of probs: compare the classes)
    assert torch.equal(ref["z"].argmax(-1), new["z"].argmax(-1))
    assert torch.allclose(ref["post_probs"], new["post_probs"], atol=1e-6)


def test_invalid_cap_is_refused():
    rssm, x, _, _ = _setup()
    with pytest.raises(ValueError, match="max_images"):
        encode_windows(rssm, x, torch.device("cpu"), max_images=0)
