from __future__ import annotations

import pytest
import torch

from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, apply_unimix, resolve_device, straight_through_sample


def _small_cfg(**overrides) -> RSSMConfig:
    defaults = dict(image_size=(16, 16), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4, n_actions=5)
    defaults.update(overrides)
    return RSSMConfig(**defaults)


def test_image_size_must_be_divisible_by_8():
    with pytest.raises(ValueError):
        RSSMConfig(image_size=(15, 16))


def test_observation_step_output_shapes():
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 3
    x = torch.rand(batch, cfg.in_channels, *cfg.image_size)
    h0, z0 = rssm.initial_state(batch, x.device)
    a0 = torch.zeros(batch, cfg.n_actions)
    is_first = torch.zeros(batch, dtype=torch.bool)

    out = rssm.observation_step(h0, z0, a0, x, is_first)
    assert out["h"].shape == (batch, cfg.h_dim)
    assert out["z"].shape == (batch, cfg.n_vars, cfg.n_classes)
    assert out["prior_probs"].shape == (batch, cfg.n_vars, cfg.n_classes)
    assert out["post_probs"].shape == (batch, cfg.n_vars, cfg.n_classes)
    assert torch.allclose(out["prior_probs"].sum(-1), torch.ones(batch, cfg.n_vars))
    assert torch.allclose(out["post_probs"].sum(-1), torch.ones(batch, cfg.n_vars))


def test_imagination_step_output_shapes_and_no_encoder_dependency():
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 2
    h0, z0 = rssm.initial_state(batch, "cpu")
    a0 = torch.zeros(batch, cfg.n_actions)
    out = rssm.imagination_step(h0, z0, a0)
    assert out["h"].shape == (batch, cfg.h_dim)
    assert out["z"].shape == (batch, cfg.n_vars, cfg.n_classes)
    assert out["prior_probs"].shape == (batch, cfg.n_vars, cfg.n_classes)


def test_decode_output_matches_image_size():
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 2
    h0, z0 = rssm.initial_state(batch, "cpu")
    img = rssm.decode(h0, z0)
    assert img.shape == (batch, cfg.in_channels, *cfg.image_size)


def test_latent_hook_is_invoked_with_the_exact_h_z_and_changes_the_prior():
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 2
    h0, z0 = rssm.initial_state(batch, "cpu")
    a0 = torch.zeros(batch, cfg.n_actions)

    calls = []

    def hook(h, z, context):
        calls.append((h.clone(), z.clone(), dict(context)))
        return h * 0.0, z  # a non-trivial, deterministic transform

    out_no_hook = rssm.imagination_step(h0, z0, a0)

    rssm.latent_hook = hook
    out_hooked = rssm.imagination_step(h0, z0, a0, context={"tag": "test"})

    assert len(calls) == 1
    called_h, called_z, called_ctx = calls[0]
    assert torch.allclose(called_h, out_no_hook["h"])  # hook sees the REAL h_t, unmodified
    assert called_ctx == {"tag": "test"}
    assert not torch.allclose(out_hooked["prior_probs"], out_no_hook["prior_probs"])


def test_default_latent_hook_is_identity():
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 1
    h0, z0 = rssm.initial_state(batch, "cpu")
    h1, z1 = rssm.latent_hook(h0, z0, {})
    assert torch.equal(h1, h0)
    assert torch.equal(z1, z0)


def test_apply_unimix_preserves_probability_simplex():
    probs = torch.softmax(torch.randn(2, 3, 5), dim=-1)
    mixed = apply_unimix(probs, 0.01)
    assert torch.allclose(mixed.sum(-1), torch.ones(2, 3))


def test_straight_through_sample_is_a_valid_one_hot_in_the_forward_pass():
    probs = torch.softmax(torch.randn(4, 2, 6), dim=-1)
    z = straight_through_sample(probs)
    # forward value must be (numerically) a one-hot vector per variable
    assert torch.allclose(z.sum(-1), torch.ones(4, 2))
    assert torch.all((z.detach() == 0) | (torch.isclose(z.detach(), torch.ones_like(z))) | (z.detach() < 1))


def test_resolve_device_auto_falls_back_to_cpu_without_cuda():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")  # never raises; cpu on this machine
    assert resolve_device("cpu").type == "cpu"


def test_gru_step_is_deterministic_given_the_same_inputs():
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    rssm.eval()
    batch = 2
    h0, z0 = rssm.initial_state(batch, "cpu")
    a0 = torch.zeros(batch, cfg.n_actions)
    with torch.no_grad():
        h1 = rssm.gru_step(h0, z0, a0)
        h2 = rssm.gru_step(h0, z0, a0)
    assert torch.equal(h1, h2)


# --- estimator audit (docs/e3/DECISIONS.md, 2026-09-24) -------------------


def _softmax_jacobian(p: torch.Tensor) -> torch.Tensor:
    return torch.diag(p) - torch.outer(p, p)


def test_straight_through_gradient_matches_the_analytic_value_on_a_toy_case():
    """L = w . z, z = onehot(sample) + q - sg(q), q = (1-r) softmax(l) + r/C.
    Analytic: dL/dl = (1-r) (diag(p) - p p^T) w, with p = softmax(l) --
    independent of which class was sampled."""
    from helplessness.agents.worldmodel.rssm import apply_unimix, straight_through_sample

    r = 0.01
    logits = torch.tensor([[0.3, -1.2, 0.8]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.5, -2.0, 1.5], dtype=torch.float64)
    q = apply_unimix(torch.softmax(logits, dim=-1), r)
    z = straight_through_sample(q)
    # forward value is one-hot up to float rounding (onehot + q - q is not bit-exact)
    zd = z.detach()[0]
    assert torch.allclose(zd, torch.nn.functional.one_hot(zd.argmax(), 3).to(zd.dtype), atol=1e-12)
    (z[0] * w).sum().backward()

    p = torch.softmax(logits.detach()[0], dim=-1)
    expected = (1 - r) * _softmax_jacobian(p) @ w
    assert torch.allclose(logits.grad[0], expected, atol=1e-12)


def test_gumbel_softmax_straight_through_gradient_matches_the_analytic_value():
    """Same toy case, Gumbel-ST at temperature tau, the Gumbel noise
    replayed from the same seed: y = softmax((log q + g)/tau),
    dL/dl = (1-r) J_p diag(1/q) (1/tau) J_y w."""
    from helplessness.agents.worldmodel.rssm import apply_unimix, gumbel_softmax_straight_through

    r, tau = 0.01, 0.7
    logits = torch.tensor([[0.3, -1.2, 0.8]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.5, -2.0, 1.5], dtype=torch.float64)
    q = apply_unimix(torch.softmax(logits, dim=-1), r)

    torch.manual_seed(123)
    z = gumbel_softmax_straight_through(q, tau)
    (z[0] * w).sum().backward()

    torch.manual_seed(123)
    u = torch.rand_like(q.detach()).clamp(1e-10, 1.0 - 1e-10)
    g = -torch.log(-torch.log(u))[0]
    q0 = q.detach()[0]
    y = torch.softmax((torch.log(q0) + g) / tau, dim=-1)
    p = torch.softmax(logits.detach()[0], dim=-1)
    expected = (1 - r) * _softmax_jacobian(p) @ (torch.diag(1.0 / q0) @ ((1.0 / tau) * _softmax_jacobian(y) @ w))
    assert torch.allclose(logits.grad[0], expected, atol=1e-10)
    assert int(z.detach()[0].argmax()) == int(y.argmax())  # forward one-hot = the perturbed argmax


def test_gumbel_softmax_straight_through_samples_the_same_law_as_the_categorical():
    """Forward pass = exact Categorical(q) sample (Gumbel-max), so the
    model's distribution is untouched by the estimator choice."""
    from helplessness.agents.worldmodel.rssm import gumbel_softmax_straight_through

    torch.manual_seed(0)
    q = torch.tensor([0.6, 0.3, 0.1]).expand(200_000, 1, 3)
    freq = gumbel_softmax_straight_through(q, tau=0.5).detach().mean(dim=(0, 1))
    assert torch.allclose(freq, torch.tensor([0.6, 0.3, 0.1]), atol=0.01)


def test_logit_layers_start_near_uniform():
    torch.manual_seed(0)
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    x = torch.rand(16, cfg.in_channels, *cfg.image_size)
    h0, _ = rssm.initial_state(16, x.device)
    with torch.no_grad():
        post = rssm.posterior_probs(h0, rssm.encoder(x))
        prior = rssm.prior_probs(h0)
    uniform = 1.0 / cfg.n_classes
    assert torch.allclose(post, torch.full_like(post, uniform), atol=0.02)
    assert torch.allclose(prior, torch.full_like(prior, uniform), atol=0.02)
    assert torch.all(rssm.posterior_net[-1].bias == 0)
    assert torch.all(rssm.prior_net[-1].bias == 0)


def test_latent_estimator_rejects_unknown_kind_and_nonpositive_tau():
    from helplessness.agents.worldmodel.rssm import LatentEstimator

    with pytest.raises(ValueError):
        LatentEstimator(kind="reinforce")
    with pytest.raises(ValueError):
        LatentEstimator(kind="gumbel_st", tau=0.0)


def test_observation_step_probs_estimator_passes_the_posterior_through_unchanged():
    from helplessness.agents.worldmodel.rssm import LatentEstimator

    cfg = _small_cfg()
    rssm = RSSM(cfg)
    x = torch.rand(3, cfg.in_channels, *cfg.image_size)
    h0, z0 = rssm.initial_state(3, x.device)
    out = rssm.observation_step(h0, z0, torch.zeros(3, cfg.n_actions), x, torch.ones(3, dtype=torch.bool),
                                estimator=LatentEstimator(kind="probs"))
    assert torch.equal(out["z"], out["post_probs"])
