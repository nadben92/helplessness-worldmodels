"""World-model losses (spec 3.2): masked pixel reconstruction, and the
two KL terms (dyn/rep) with an optional (default-off) free-bits floor.

The reconstruction mask is computed from IMAGES ALONE (x vs. the
rendered background, both plain tensors) -- no state label ever appears
in this module, satisfying spec 3.2's "sans aucune etiquette d'etat" by
construction (there is no state argument to accidentally pass one
through).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.rssm import categorical_kl

FreeBitsMode = Literal["per_step", "batch_mean"]


def reconstruction_mask(x: torch.Tensor, background: torch.Tensor, k: int, threshold: float = 1.0 / 255.0) -> torch.Tensor:
    """x, background: (N, C, H, W) in [0, 1]. Returns M in [0, 1], the
    dilation-by-one-cell (k pixels) of the binary mask where x differs
    from the (agent-free) background -- spec 3.2: "M est la dilatation
    d'une case de l'ensemble des pixels ou x != render_background",
    computed from the image alone."""
    diff = (x - background).abs().amax(dim=1, keepdim=True)  # (N, 1, H, W)
    binary = (diff > threshold).to(x.dtype)
    pad = k // 2
    dilated = F.max_pool2d(binary, kernel_size=k, stride=1, padding=pad)
    H, W = x.shape[-2], x.shape[-1]
    return dilated[..., :H, :W]


def weighted_reconstruction_loss(
    x: torch.Tensor, x_hat: torch.Tensor, background: torch.Tensor, k: int, w_patch: float
) -> torch.Tensor:
    """Squared error weighted by w = 1 + (w_patch - 1) * M (spec 3.2),
    SUMMED over channels and pixels of each image and averaged over
    images only (x: (N, C, H, W), N = batch x time) -- the DreamerV3
    convention (log-likelihood summed over observation dimensions).
    Averaging over pixels instead shrinks this term by C*H*W (1728 at
    24x24x3) relative to the KL terms, which are summed over latent
    variables -- the cause of the Phase-3 mini run's posterior collapse
    (docs/e3/DECISIONS.md, 2026-09-24)."""
    M = reconstruction_mask(x, background, k)
    w = 1.0 + (w_patch - 1.0) * M
    return (w * (x - x_hat) ** 2).sum(dim=(1, 2, 3)).mean()


def patch_only_reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor, background: torch.Tensor, k: int) -> torch.Tensor:
    """Diagnostic-only (never part of the training loss): UNWEIGHTED
    squared error summed over the masked (agent-patch) pixels and
    channels of each image, averaged over images -- same aggregation as
    weighted_reconstruction_loss (sum per image, mean over images), so
    the two are directly comparable: recon_patch is the unweighted part
    of each image's error that sits on the patch."""
    M = reconstruction_mask(x, background, k)
    sq_err = (x - x_hat) ** 2
    return (sq_err * (M > 0)).sum(dim=(1, 2, 3)).mean()


def dyn_rep_kl(
    prior_probs: torch.Tensor,
    post_probs: torch.Tensor,
    beta_dyn: float = 0.5,
    beta_rep: float = 0.1,
    free_bits: float = 0.0,
    free_bits_mode: FreeBitsMode = "per_step",
) -> dict[str, torch.Tensor]:
    """dyn = KL(sg(q) || p), rep = KL(q || sg(p)) (spec 3.2), each exact
    (categorical_kl sums over the n_vars variables). free_bits=0 (default)
    is a no-op (KL >= 0 always, so clamping at 0 changes nothing) -- the
    parameter and its mode stay live in the signature so it can be
    re-enabled without touching call sites (spec 3.2's own requirement).
    """
    dyn = categorical_kl(post_probs.detach(), prior_probs)  # (batch,)
    rep = categorical_kl(post_probs, prior_probs.detach())  # (batch,)

    if free_bits_mode not in ("per_step", "batch_mean"):
        raise ValueError(f"free_bits_mode must be 'per_step' or 'batch_mean', got {free_bits_mode!r}")

    if free_bits_mode == "per_step":
        dyn_loss = torch.clamp(dyn, min=free_bits).mean()
        rep_loss = torch.clamp(rep, min=free_bits).mean()
    else:  # batch_mean
        dyn_loss = torch.clamp(dyn.mean(), min=free_bits)
        rep_loss = torch.clamp(rep.mean(), min=free_bits)

    total = beta_dyn * dyn_loss + beta_rep * rep_loss
    return {"total": total, "dyn": dyn_loss, "rep": rep_loss, "dyn_raw_mean": dyn.mean(), "rep_raw_mean": rep.mean()}


def mean_categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    """probs: (..., n_vars, n_classes). Entropy in NATS, summed over the
    latent variables, averaged over every leading index -- the H(q) of
    option (ii)'s penalty eta * H(q) (DECISIONS.md, 2026-09-24). Same
    summed-over-variables convention as categorical_kl."""
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=(-2, -1)).mean()
