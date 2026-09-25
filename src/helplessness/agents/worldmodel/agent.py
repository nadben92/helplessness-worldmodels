"""WorldModelAgent: wraps a frozen RSSM.

`transition_distribution(s, a)` (spec section 5.3's decode+template
bridge) is a MEASUREMENT ADAPTER ONLY -- it lets
`helplessness.agents.base.materialize_transition_tensor` /
`measures.controllability.controllability_from_tensor` run on this
agent's belief exactly as they already run on the tabular agents'
T_hat, per the ABC's own contract (see agents/base.py). The POLICY
(`act()`, spec section 4.3's one-step-anticipation epsilon-greedy
choice, Phase 5) never calls it -- pinned by
tests/agents/worldmodel/test_worldmodel_agent.py's mock test.

`act()` is a placeholder (`NotImplementedError`) here: the real policy
(K_act rollouts, the Q formula, the goal head, the critic) is Phase 5
scope. Only the measurement bridge is decided and built now (see
docs/e3/DECISIONS.md, point 2).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.base import Agent
from helplessness.agents.worldmodel.rssm import RSSM
from helplessness.measures.kernel_probe import kernel_probe

logger = logging.getLogger(__name__)


def categorical_mode(probs: torch.Tensor) -> torch.Tensor:
    """probs: (..., n_vars, n_classes). Returns the one-hot MODE per
    variable (argmax), NOT a stochastic sample -- spec 5.1's canonical
    probe explicitly wants "z0 = mode de q(...)", a deterministic value
    per state so repeated calls at the same state are comparable."""
    idx = probs.argmax(dim=-1)
    return F.one_hot(idx, num_classes=probs.shape[-1]).to(probs.dtype)


def sample_categorical_batch(probs: torch.Tensor, K: int) -> torch.Tensor:
    """probs: (n_vars, n_classes), a single product distribution.
    Returns K independent one-hot samples: (K, n_vars, n_classes)."""
    n_vars, n_classes = probs.shape
    dist = torch.distributions.Categorical(probs=probs.unsqueeze(0).expand(K, -1, -1))
    idx = dist.sample()  # (K, n_vars)
    return F.one_hot(idx, num_classes=n_classes).to(probs.dtype)


class WorldModelAgent(Agent):
    def __init__(
        self,
        rssm: RSSM,
        templates: torch.Tensor,
        K_samples: int = 32,
        margin_threshold: float = 1e-3,
        device: torch.device | None = None,
    ):
        """templates: (n_states, C, H, W) float in [0, 1] -- the run's
        144 canonical renders (render(s, layout, themes) for every s),
        reused as BOTH the canonical probe's own encoder input at each
        state (spec 5.1: x = render(s)) and the template gallery
        match_templates() matches decoded samples against (spec 5.3) --
        the two are literally the same array by construction, so only
        one is stored."""
        self.rssm = rssm
        self.templates = templates
        self.n_states = templates.shape[0]
        self.K_samples = K_samples
        self.margin_threshold = margin_threshold
        self.device = device or next(rssm.parameters()).device

    def _canonical_h_z(self, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        """h0, z0 at state s: fresh episode start (is_first=True), z0 =
        the posterior MODE (not a sample) -- calls the RSSM's own
        methods directly (encoder, gru_step, posterior_probs), never a
        reimplementation of its forward pass."""
        x = self.templates[s].unsqueeze(0).to(self.device)
        h0, z_init = self.rssm.initial_state(1, self.device)
        a0 = torch.zeros(1, self.rssm.cfg.n_actions, device=self.device)
        e = self.rssm.encoder(x)
        post_probs = self.rssm.posterior_probs(h0, e)
        z0 = categorical_mode(post_probs)
        return h0, z0

    def transition_distribution_with_diagnostics(self, s: int, a: int) -> tuple[np.ndarray, float, int]:
        """MEASUREMENT ADAPTER ONLY -- see module docstring. Decodes
        K_samples draws of z' ~ prior(h_1_a) (h_1_a from
        RSSM.imagination_step, which applies latent_hook exactly as the
        policy will -- spec 3.1, adjustment B), matches each against the
        144 canonical templates (measures.kernel_probe).

        Returns (kernel, ghost_fraction, n_valid_draws) -- a structured
        result, not a side-effect attribute. kernel is renormalised
        over the non-ghost draws only. If EVERY draw was a ghost
        (n_valid_draws == 0), kernel is NaN-filled (never a silent
        division by zero, never a zero-filled or uniform stand-in for
        "no information") and a warning is logged."""
        with torch.no_grad():
            h0, z0 = self._canonical_h_z(s)
            a_onehot = F.one_hot(torch.tensor([a], device=self.device), num_classes=self.rssm.cfg.n_actions).to(z0.dtype)
            img_out = self.rssm.imagination_step(h0, z0, a_onehot)
            h1_a, prior_probs = img_out["h"], img_out["prior_probs"]

            z_samples = sample_categorical_batch(prior_probs[0], self.K_samples)  # (K, n_vars, n_classes)
            h1_a_batch = h1_a.expand(self.K_samples, -1)
            decoded = self.rssm.decode(h1_a_batch, z_samples)  # (K, C, H, W), applies latent_hook again (spec 3.1)

            decoded_np = decoded.detach().cpu().numpy()
            templates_np = self.templates.detach().cpu().numpy()
            result = kernel_probe(decoded_np, templates_np, self.margin_threshold)

        ghost_fraction = result["ghost_fraction"]
        n_valid_draws = result["n_samples"] - result["n_ghosts"]
        if n_valid_draws == 0:
            kernel = np.full(self.n_states, np.nan)
            logger.warning(
                "WorldModelAgent.transition_distribution_with_diagnostics(s=%d, a=%d): "
                "all %d draws were classified ghosts (margin_threshold=%g) -- returning a "
                "NaN kernel rather than a silently renormalised or zero-filled one",
                s, a, result["n_samples"], self.margin_threshold,
            )
        else:
            kernel = result["p_hat"]
        return kernel, ghost_fraction, n_valid_draws

    def transition_distribution(self, s: int, a: int) -> np.ndarray:
        """Agent ABC wrapper (fixed single-ndarray-return signature) --
        see transition_distribution_with_diagnostics() for the
        ghost_fraction/n_valid_draws diagnostics this discards."""
        kernel, _, _ = self.transition_distribution_with_diagnostics(s, a)
        return kernel

    def act(self, s: int, rng: np.random.Generator) -> int:
        """Placeholder -- the real one-step-anticipation epsilon-greedy
        policy (spec 4.3: K_act rollouts, the Q formula over the goal
        head and critic) is Phase 5 scope, not decided or built here.
        MUST NEVER call transition_distribution() (see
        tests/agents/worldmodel/test_agent.py's mock test, which will
        keep passing unmodified once this is replaced in Phase 5)."""
        raise NotImplementedError("policy (spec section 4.3) is Phase 5 scope, not yet implemented")
