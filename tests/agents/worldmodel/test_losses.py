from __future__ import annotations

import pytest
import torch

from helplessness.agents.worldmodel.losses import (
    dyn_rep_kl,
    patch_only_reconstruction_error,
    reconstruction_mask,
    weighted_reconstruction_loss,
)


def test_reconstruction_mask_grows_by_approximately_k_pixels():
    background = torch.zeros(1, 3, 20, 20)
    x = background.clone()
    x[0, :, 9, 9] = 1.0  # a single-pixel diff
    k = 6
    mask = reconstruction_mask(x, background, k=k)
    covered = (mask[0, 0] > 0).nonzero()
    row_span = covered[:, 0].max() - covered[:, 0].min() + 1
    col_span = covered[:, 1].max() - covered[:, 1].min() + 1
    # a single differing pixel dilated by a k-pixel structuring element
    # should cover roughly a k-ish sized neighbourhood, not the whole
    # image and not just the original single pixel.
    assert 1 < row_span <= k + 1
    assert 1 < col_span <= k + 1


def test_reconstruction_mask_is_zero_when_images_are_identical():
    img = torch.rand(2, 3, 16, 16)
    mask = reconstruction_mask(img, img, k=4)
    assert torch.all(mask == 0)


def test_weighted_reconstruction_loss_upweights_the_masked_region():
    background = torch.zeros(1, 3, 16, 16)
    x = background.clone()
    x[0, :, 6:10, 6:10] = 1.0  # the "agent" patch that defines the mask

    # Perturb exactly ONE pixel (all 3 channels) each time, by the SAME
    # ADDITIVE delta relative to x's own value there (x differs between
    # the two locations: 1.0 inside the patch, 0.0 outside it) -- so
    # only the WEIGHT differs between the two cases, not the error
    # magnitude. Inside: the patch itself (definitely under the dilated
    # mask). Outside: the far corner (definitely outside it, per the
    # dilation test above).
    delta = 0.1
    x_hat_inside = x.clone()
    x_hat_inside[0, :, 7, 7] -= delta
    x_hat_outside = x.clone()
    x_hat_outside[0, :, 0, 0] += delta

    loss_inside = weighted_reconstruction_loss(x, x_hat_inside, background, k=4, w_patch=10.0)
    loss_outside = weighted_reconstruction_loss(x, x_hat_outside, background, k=4, w_patch=10.0)
    assert loss_inside > loss_outside  # same-sized error is penalised more inside the patch mask


def test_weighted_reconstruction_loss_reduces_to_summed_squared_error_when_w_patch_is_1():
    """Aggregation pin (docs/e3/DECISIONS.md, 2026-09-24): SUM over
    channels and pixels of each image, MEAN over images only -- never a
    mean over pixels, which shrank this term by C*H*W relative to the
    KL and caused the mini run's posterior collapse."""
    x = torch.rand(2, 3, 16, 16)
    x_hat = torch.rand(2, 3, 16, 16)
    background = torch.rand(2, 3, 16, 16)
    weighted = weighted_reconstruction_loss(x, x_hat, background, k=4, w_patch=1.0)
    per_image_sum = ((x - x_hat) ** 2).sum(dim=(1, 2, 3))
    assert torch.allclose(weighted, per_image_sum.mean())
    assert not torch.allclose(weighted, ((x - x_hat) ** 2).mean())


def test_weighted_reconstruction_loss_does_not_shrink_with_image_size():
    """The same single-pixel error must cost the same whatever the
    image size -- true under sum-per-image, false under a per-pixel
    mean (which dilutes it by the pixel count)."""
    losses = []
    for side in (8, 32):
        background = torch.zeros(1, 3, side, side)
        x = background.clone()
        x_hat = x.clone()
        x_hat[0, :, 0, 0] += 0.5
        losses.append(weighted_reconstruction_loss(x, x_hat, background, k=2, w_patch=1.0).item())
    assert losses[0] == pytest.approx(losses[1])
    assert losses[0] == pytest.approx(3 * 0.5**2)


def test_dyn_rep_kl_zero_when_prior_equals_posterior():
    probs = torch.softmax(torch.randn(3, 2, 5), dim=-1)
    result = dyn_rep_kl(probs, probs)
    assert result["dyn"].item() < 1e-5
    assert result["rep"].item() < 1e-5
    assert result["total"].item() < 1e-5


def test_dyn_term_has_no_gradient_into_the_prior_only_into_the_posterior_is_detached():
    """dyn = KL(sg(q) || p): gradient must flow into p (prior), NOT into
    q (posterior) -- q is stop-gradiented in this term."""
    prior_logits = torch.randn(2, 2, 4, requires_grad=True)
    post_logits = torch.randn(2, 2, 4, requires_grad=True)
    prior = torch.softmax(prior_logits, dim=-1)
    post = torch.softmax(post_logits, dim=-1)

    result = dyn_rep_kl(prior, post)
    result["dyn"].backward(retain_graph=True)
    assert prior_logits.grad is not None and prior_logits.grad.abs().sum() > 0
    assert post_logits.grad is None or post_logits.grad.abs().sum() == 0


def test_rep_term_has_no_gradient_into_the_posterior_only_into_the_prior_is_detached():
    """rep = KL(q || sg(p)): gradient must flow into q (posterior), NOT
    into p (prior) -- p is stop-gradiented in this term."""
    prior_logits = torch.randn(2, 2, 4, requires_grad=True)
    post_logits = torch.randn(2, 2, 4, requires_grad=True)
    prior = torch.softmax(prior_logits, dim=-1)
    post = torch.softmax(post_logits, dim=-1)

    result = dyn_rep_kl(prior, post)
    result["rep"].backward()
    assert post_logits.grad is not None and post_logits.grad.abs().sum() > 0
    assert prior_logits.grad is None or prior_logits.grad.abs().sum() == 0


def test_free_bits_zero_is_a_no_op():
    prior = torch.softmax(torch.randn(4, 2, 5), dim=-1)
    post = torch.softmax(torch.randn(4, 2, 5), dim=-1)
    a = dyn_rep_kl(prior, post, free_bits=0.0, free_bits_mode="per_step")
    b = dyn_rep_kl(prior, post, free_bits=0.0, free_bits_mode="batch_mean")
    assert torch.allclose(a["total"], b["total"])


def test_free_bits_per_step_floors_every_kl_value():
    prior = torch.softmax(torch.randn(4, 2, 5) * 0.01, dim=-1)  # near-uniform -> tiny KL
    post = prior.clone()
    floor = 0.5
    result = dyn_rep_kl(prior, post, free_bits=floor, free_bits_mode="per_step")
    assert result["dyn"].item() == pytest.approx(floor, abs=1e-5)
    assert result["rep"].item() == pytest.approx(floor, abs=1e-5)


def test_patch_only_reconstruction_error_ignores_errors_outside_the_mask():
    background = torch.zeros(1, 3, 16, 16)
    x = background.clone()
    x[0, :, 6:10, 6:10] = 1.0
    x_hat = x.clone()
    x_hat[0, :, 0, 0] += 0.5  # error strictly outside the dilated mask
    err = patch_only_reconstruction_error(x, x_hat, background, k=4)
    assert err.item() == pytest.approx(0.0, abs=1e-9)


def test_patch_only_reconstruction_error_is_the_summed_squared_error_inside_the_mask():
    background = torch.zeros(1, 3, 16, 16)
    x = background.clone()
    x[0, :, 6:10, 6:10] = 1.0
    x_hat = x.clone()
    x_hat[0, :, 7, 7] -= 0.2  # one pixel, all 3 channels, inside the mask
    err = patch_only_reconstruction_error(x, x_hat, background, k=4)
    assert err.item() == pytest.approx(3 * 0.2**2, abs=1e-6)  # summed over the 3 channels, one image
