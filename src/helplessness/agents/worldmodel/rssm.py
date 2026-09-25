"""RSSM (Recurrent State-Space Model), DreamerV3-lineage, simplified per
spec section 3.1: CNN encoder/decoder, a categorical latent with
straight-through sampling and 1% unimix, a GRU deterministic core, and a
`latent_hook` extension point (identity by default -- the causal latent
manipulation hook E6 needs, wired here but never invoked with anything
but the identity until then).

Kept deliberately explicit and readable (this codebase has no external
RL framework, see CLAUDE.md) rather than routing through a generic
"world model" abstraction -- there is exactly one RSSM implementation in
this project, this one; probe.py (Phase 3) and the agent (Phase 5) both
call ITS methods (observation_step / imagination_step / prior / decode),
never a reimplementation of the forward pass (see docs/e3/DECISIONS.md,
adjustment B).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

LatentHook = Callable[[torch.Tensor, torch.Tensor, dict], tuple[torch.Tensor, torch.Tensor]]


def _identity_hook(h: torch.Tensor, z: torch.Tensor, context: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return h, z


def resolve_device(device: str = "auto") -> torch.device:
    """Config-driven device resolution -- no hardcoded .cuda() anywhere
    (see docs/e3/DECISIONS.md, point 7's constraints)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class RSSMConfig:
    image_size: tuple[int, int] = (72, 72)
    in_channels: int = 3
    base_depth: int = 32
    embed_dim: int = 256
    h_dim: int = 256
    n_vars: int = 8
    n_classes: int = 8
    n_actions: int = 5
    unimix: float = 0.01
    # Scale applied to the default init of the prior/posterior logit
    # output layers (weights multiplied, bias zeroed): logits start near
    # 0, so both distributions start near uniform, as in DreamerV3
    # (docs/e3/DECISIONS.md, 2026-09-24 estimator audit).
    logit_init_scale: float = 0.01

    def __post_init__(self):
        h, w = self.image_size
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"image_size {self.image_size} must be divisible by 8 (three stride-2 conv layers)")

    @property
    def latent_dim(self) -> int:
        return self.n_vars * self.n_classes

    @property
    def encoded_spatial(self) -> tuple[int, int]:
        h, w = self.image_size
        return h // 8, w // 8


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of an (N, C, H, W) tensor --
    spec 3.1's "LayerNorm sur les canaux" (as opposed to BatchNorm)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class Encoder(nn.Module):
    """Three stride-2 conv layers (base_depth, base_depth*2, base_depth*4),
    ChannelLayerNorm + SiLU each, flattened and projected to embed_dim."""

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        d = cfg.base_depth
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.in_channels, d, kernel_size=4, stride=2, padding=1),
            ChannelLayerNorm(d),
            nn.SiLU(),
            nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1),
            ChannelLayerNorm(d * 2),
            nn.SiLU(),
            nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1),
            ChannelLayerNorm(d * 4),
            nn.SiLU(),
        )
        eh, ew = cfg.encoded_spatial
        self.proj = nn.Linear(d * 4 * eh * ew, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.reshape(h.shape[0], -1)
        return self.proj(h)


class Decoder(nn.Module):
    """Mirror of Encoder: projects (h, z) to a small feature map, then
    three stride-2 transposed convs back to image resolution. Outputs the
    reconstruction MEAN (linear head, no activation -- spec 3.1: "CNN
    transpose qui sort l'image moyenne"), matched against images
    normalised to [0, 1]."""

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        d = cfg.base_depth
        eh, ew = cfg.encoded_spatial
        self._eh, self._ew, self._d = eh, ew, d
        self.proj = nn.Linear(cfg.h_dim + cfg.latent_dim, d * 4 * eh * ew)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d * 4, d * 2, kernel_size=4, stride=2, padding=1),
            ChannelLayerNorm(d * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(d * 2, d, kernel_size=4, stride=2, padding=1),
            ChannelLayerNorm(d),
            nn.SiLU(),
            nn.ConvTranspose2d(d, cfg.in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, h: torch.Tensor, z_flat: torch.Tensor) -> torch.Tensor:
        x = self.proj(torch.cat([h, z_flat], dim=-1))
        x = x.reshape(-1, self._d * 4, self._eh, self._ew)
        return self.deconv(x)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def apply_unimix(probs: torch.Tensor, ratio: float) -> torch.Tensor:
    """probs: (..., n_classes). Mixes in a uniform floor -- spec 3.1's 1%
    unimix on both prior and posterior: no category's probability can
    collapse to exactly 0, which keeps KL well-defined and the categorical
    sampler exploring every category with non-zero probability."""
    n_classes = probs.shape[-1]
    uniform = torch.full_like(probs, 1.0 / n_classes)
    return (1.0 - ratio) * probs + ratio * uniform


def straight_through_sample(probs: torch.Tensor) -> torch.Tensor:
    """probs: (..., n_vars, n_classes), already unimix'd. Returns a
    one-hot sample per variable with the straight-through gradient trick
    (spec 3.1): z = onehot(sample) + probs - sg(probs) -- forward value is
    a genuine one-hot draw, backward gradient flows through probs as if z
    == probs (the softmax relaxation), giving a non-zero gradient into
    whatever produced probs (T0's own requirement)."""
    dist = torch.distributions.Categorical(probs=probs)
    sample = dist.sample()
    one_hot = F.one_hot(sample, num_classes=probs.shape[-1]).to(probs.dtype)
    return one_hot + probs - probs.detach()


def gumbel_softmax_straight_through(probs: torch.Tensor, tau: float) -> torch.Tensor:
    """probs: (..., n_vars, n_classes), already unimix'd. Forward: the
    one-hot argmax of (log probs + Gumbel noise) -- an EXACT sample from
    Categorical(probs), the same law straight_through_sample draws from.
    Backward: through softmax((log probs + Gumbel noise) / tau), a
    tempered relaxation of that same sample (Jang et al. 2017, "ST
    Gumbel-softmax"). Only the gradient estimator differs; the model's
    distribution does not."""
    logits = torch.log(probs.clamp_min(1e-20))
    u = torch.rand_like(logits).clamp(1e-10, 1.0 - 1e-10)
    y = F.softmax((logits - torch.log(-torch.log(u))) / tau, dim=-1)
    hard = F.one_hot(y.argmax(dim=-1), num_classes=probs.shape[-1]).to(y.dtype)
    return hard + y - y.detach()


@dataclass(frozen=True)
class LatentEstimator:
    """How the POSTERIOR z is produced during TRAINING (the reconstruction
    path) -- a training-procedure choice, not part of the model: the
    categorical/unimix definition, the canonical probe, the prior samples
    and every exact-density measure are unaffected (docs/e3/DECISIONS.md,
    2026-09-24, a documented deviation from DreamerV3).

    kind:
      "straight_through": one-hot sample + probs - sg(probs) (DreamerV3's).
      "gumbel_st": straight-through Gumbel-softmax at temperature tau.
      "probs": z = the posterior probabilities themselves, no sampling
          (the warm-up phase of the "warmup_probs" schedule, loop.py).
      "partial_straight_through": a fraction rho of the latent variables
          (chosen at random, independently per example) receives the
          straight-through sample, the others the probabilities
          (partial_straight_through below; option (iii), DECISIONS.md
          2026-09-24).
    """

    kind: str = "straight_through"
    tau: float = 1.0
    rho: float = 1.0

    def __post_init__(self):
        if self.kind not in ("straight_through", "gumbel_st", "probs", "partial_straight_through"):
            raise ValueError(f"unknown LatentEstimator kind {self.kind!r}")
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [0, 1], got {self.rho}")


def partial_straight_through(probs: torch.Tensor, rho: float) -> torch.Tensor:
    """probs: (..., n_vars, n_classes). For every leading index, exactly
    round(rho * n_vars) variables, drawn uniformly without replacement,
    get the straight-through sample (straight_through_sample); the others
    keep the probabilities. rho=0 is "probs", rho=1 is plain
    straight-through."""
    n_vars = probs.shape[-2]
    n_sampled = int(round(rho * n_vars))
    if n_sampled == 0:
        return probs
    if n_sampled == n_vars:
        return straight_through_sample(probs)
    ranks = torch.rand(probs.shape[:-1], device=probs.device).argsort(dim=-1).argsort(dim=-1)
    use_sample = (ranks < n_sampled).unsqueeze(-1).to(probs.dtype)  # (..., n_vars, 1)
    return use_sample * straight_through_sample(probs) + (1.0 - use_sample) * probs


def sample_posterior(probs: torch.Tensor, estimator: LatentEstimator) -> torch.Tensor:
    if estimator.kind == "straight_through":
        return straight_through_sample(probs)
    if estimator.kind == "partial_straight_through":
        return partial_straight_through(probs, estimator.rho)
    if estimator.kind == "gumbel_st":
        return gumbel_softmax_straight_through(probs, estimator.tau)
    return probs


def categorical_kl(p_probs: torch.Tensor, q_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q), exact, summed over the n_vars variables (spec 3.2:
    "KL calculee exactement, en sommant sur les variables"). p_probs,
    q_probs: (..., n_vars, n_classes). Returns (...,)."""
    per_class = p_probs * (torch.log(p_probs + eps) - torch.log(q_probs + eps))
    per_var = per_class.sum(dim=-1)  # sum over classes
    return per_var.sum(dim=-1)  # sum over variables


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        self.input_proj = _mlp(cfg.latent_dim + cfg.n_actions, cfg.h_dim, cfg.h_dim)
        self.gru = nn.GRUCell(cfg.h_dim, cfg.h_dim)
        self.prior_net = _mlp(cfg.h_dim, cfg.h_dim, cfg.latent_dim)
        self.posterior_net = _mlp(cfg.h_dim + cfg.embed_dim, cfg.h_dim, cfg.latent_dim)

        self.h0 = nn.Parameter(torch.zeros(cfg.h_dim))

        with torch.no_grad():
            for net in (self.prior_net, self.posterior_net):
                net[-1].weight.mul_(cfg.logit_init_scale)
                net[-1].bias.zero_()

        self.latent_hook: LatentHook = _identity_hook

    # ------------------------------------------------------------------
    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.h0.unsqueeze(0).expand(batch_size, -1).to(device)
        z0 = torch.zeros(batch_size, self.cfg.n_vars, self.cfg.n_classes, device=device)
        return h0, z0

    def _flat(self, z: torch.Tensor) -> torch.Tensor:
        return z.reshape(z.shape[0], -1)

    def gru_step(self, h_prev: torch.Tensor, z_prev: torch.Tensor, a_prev_onehot: torch.Tensor) -> torch.Tensor:
        """h_t = GRU(h_{t-1}, [z_{t-1}, onehot(a_{t-1})]), with an MLP
        input projection (spec 3.1: "une projection MLP en entree comme
        dans V3")."""
        x = torch.cat([self._flat(z_prev), a_prev_onehot], dim=-1)
        proj = self.input_proj(x)
        return self.gru(proj, h_prev)

    def prior_probs(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.prior_net(h).reshape(-1, self.cfg.n_vars, self.cfg.n_classes)
        return apply_unimix(F.softmax(logits, dim=-1), self.cfg.unimix)

    def posterior_probs(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        logits = self.posterior_net(torch.cat([h, e], dim=-1)).reshape(-1, self.cfg.n_vars, self.cfg.n_classes)
        return apply_unimix(F.softmax(logits, dim=-1), self.cfg.unimix)

    def decode(self, h: torch.Tensor, z: torch.Tensor, context: dict | None = None) -> torch.Tensor:
        h, z = self.latent_hook(h, z, context or {})
        return self.decoder(h, self._flat(z))

    # ------------------------------------------------------------------
    def observation_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev_onehot: torch.Tensor,
        x_t: torch.Tensor,
        is_first: torch.Tensor,
        context: dict | None = None,
        estimator: LatentEstimator | None = None,
    ) -> dict[str, torch.Tensor]:
        """One real (posterior) step: h_t = GRU(...), e_t = encode(x_t),
        z_t ~ posterior(h_t, e_t) (via `estimator`, default plain
        straight-through -- see LatentEstimator), prior evaluated at
        h_t too (for the KL terms). is_first (bool, (batch,)): where
        True, h_prev/z_prev/a_prev_onehot are replaced by the learned
        initial state / zeros before the GRU step (spec 3.1's episode-
        start convention), never mixed with the previous episode's
        state.
        """
        batch = h_prev.shape[0]
        device = h_prev.device
        h0, z0 = self.initial_state(batch, device)
        a0 = torch.zeros_like(a_prev_onehot)

        is_first = is_first.to(dtype=torch.bool)
        mask = is_first.view(-1, *([1] * (h_prev.dim() - 1)))
        h_prev = torch.where(mask, h0, h_prev)
        z_mask = is_first.view(-1, *([1] * (z_prev.dim() - 1)))
        z_prev = torch.where(z_mask, z0, z_prev)
        a_mask = is_first.view(-1, *([1] * (a_prev_onehot.dim() - 1)))
        a_prev_onehot = torch.where(a_mask, a0, a_prev_onehot)

        h_t = self.gru_step(h_prev, z_prev, a_prev_onehot)
        hook_h, _ = self.latent_hook(h_t, torch.zeros_like(z_prev), context or {})
        prior = self.prior_probs(hook_h)

        e_t = self.encoder(x_t)
        post = self.posterior_probs(h_t, e_t)
        z_t = sample_posterior(post, estimator or LatentEstimator())

        return {"h": h_t, "z": z_t, "prior_probs": prior, "post_probs": post, "embed": e_t}

    def imagination_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev_onehot: torch.Tensor,
        context: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """One imagined (prior-only) step -- no observation, no
        posterior. Used by the agent's one-step lookahead policy (spec
        4.3) and critic-in-imagination training (spec 4.4), Phase 5.
        Routes through latent_hook exactly like observation_step, so
        probe.py (Phase 3) and the agent (Phase 5) see the identical
        transformed prior for the same (h, z, a) and hook (see
        docs/e3/DECISIONS.md, adjustment B)."""
        h_t = self.gru_step(h_prev, z_prev, a_prev_onehot)
        hook_h, hook_z = self.latent_hook(h_t, z_prev, context or {})
        prior = self.prior_probs(hook_h)
        z_t = straight_through_sample(prior)
        return {"h": h_t, "z": z_t, "prior_probs": prior}
