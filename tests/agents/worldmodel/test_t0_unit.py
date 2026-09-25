"""T0 (spec section 9, "unitaires"), each of the 6 checklist items as
its own test:

1. gradient straight-through non nul vers l'encodeur
2. KL exacte contre estimation Monte-Carlo
3. plancher de l'unimix
4. masque de perte calcule sans etiquette
5. reinitialisation par is_first dans le replay
6. truncated distinct de terminal
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from helplessness.agents.worldmodel.losses import reconstruction_mask
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, apply_unimix, categorical_kl, straight_through_sample


def _small_cfg() -> RSSMConfig:
    return RSSMConfig(image_size=(16, 16), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4, n_actions=5)


# --- 1. gradient straight-through non nul vers l'encodeur -----------------


def test_straight_through_gradient_reaches_the_encoder():
    torch.manual_seed(0)
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 2
    x = torch.rand(batch, cfg.in_channels, *cfg.image_size)
    h0, z0 = rssm.initial_state(batch, x.device)
    a0 = torch.zeros(batch, cfg.n_actions)
    is_first = torch.ones(batch, dtype=torch.bool)

    out = rssm.observation_step(h0, z0, a0, x, is_first)
    loss = rssm.decode(out["h"], out["z"]).sum()
    loss.backward()

    encoder_grads = [p.grad for p in rssm.encoder.parameters() if p.requires_grad]
    assert encoder_grads, "encoder has no trainable parameters to check"
    assert all(g is not None for g in encoder_grads)
    total_grad_norm = sum(g.abs().sum().item() for g in encoder_grads)
    assert total_grad_norm > 0.0


# --- 2. KL exacte contre estimation Monte-Carlo ----------------------------


def test_categorical_kl_matches_monte_carlo_estimate():
    torch.manual_seed(1)
    n_vars, n_classes = 2, 5
    p_logits = torch.randn(1, n_vars, n_classes) * 2
    q_logits = torch.randn(1, n_vars, n_classes) * 2
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)

    exact = categorical_kl(p, q).item()

    # Monte-Carlo: KL(p||q) = E_{x~p}[log p(x) - log q(x)], summed over
    # the n_vars independent categorical variables.
    n_samples = 200_000
    mc_total = 0.0
    for var in range(n_vars):
        dist = torch.distributions.Categorical(probs=p[0, var])
        samples = dist.sample((n_samples,))
        log_p = torch.log(p[0, var][samples])
        log_q = torch.log(q[0, var][samples])
        mc_total += (log_p - log_q).mean().item()

    assert abs(exact - mc_total) < 0.05, f"exact={exact}, mc={mc_total}"


# --- 3. plancher de l'unimix -----------------------------------------------


def test_unimix_floor_is_exactly_ratio_over_n_classes():
    n_classes = 6
    ratio = 0.01
    # A near-delta distribution (softmax of a huge-magnitude logit spike).
    logits = torch.full((1, 1, n_classes), -1e6)
    logits[0, 0, 0] = 1e6
    probs = torch.softmax(logits, dim=-1)
    mixed = apply_unimix(probs, ratio)

    expected_floor = ratio / n_classes
    assert mixed.min().item() == pytest.approx(expected_floor, abs=1e-6)
    assert (mixed >= expected_floor - 1e-6).all()


# --- 4. masque de perte calcule sans etiquette -----------------------------


def test_reconstruction_mask_signature_has_no_state_or_label_parameter():
    params = list(inspect.signature(reconstruction_mask).parameters)
    assert "s" not in params
    assert "state" not in params
    assert set(params) == {"x", "background", "k", "threshold"}


def test_reconstruction_mask_is_computed_from_images_alone():
    background = torch.zeros(1, 3, 12, 12)
    x = background.clone()
    x[0, :, 4:8, 4:8] = 1.0  # a 4x4 "agent" patch, image-only difference
    mask = reconstruction_mask(x, background, k=4)
    assert mask.shape == (1, 1, 12, 12)
    assert mask[0, 0, 4:8, 4:8].min() > 0  # covers the actual diff
    assert mask[0, 0, 0, 0] == 0  # far corner untouched (dilation is local)


# --- 5. reinitialisation par is_first dans le replay -----------------------


def test_observation_step_ignores_prev_state_entirely_when_is_first():
    torch.manual_seed(2)
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    batch = 1
    x = torch.rand(batch, cfg.in_channels, *cfg.image_size)
    is_first = torch.ones(batch, dtype=torch.bool)

    h0, z0 = rssm.initial_state(batch, x.device)
    a0 = torch.zeros(batch, cfg.n_actions)
    out_clean = rssm.observation_step(h0, z0, a0, x, is_first)

    # Arbitrary, very different "previous" state -- must be fully
    # discarded because is_first=True, giving byte-identical output.
    h_garbage = torch.randn(batch, cfg.h_dim) * 100
    z_garbage = torch.rand(batch, cfg.n_vars, cfg.n_classes)
    a_garbage = torch.rand(batch, cfg.n_actions)
    out_garbage_prev = rssm.observation_step(h_garbage, z_garbage, a_garbage, x, is_first)

    assert torch.allclose(out_clean["h"], out_garbage_prev["h"])
    assert torch.allclose(out_clean["prior_probs"], out_garbage_prev["prior_probs"])
    assert torch.allclose(out_clean["post_probs"], out_garbage_prev["post_probs"])


def test_replay_buffer_stores_is_first_exactly_as_given():
    buf = ReplayBuffer(capacity=8, image_shape=(4, 4, 3))
    flags = [True, False, False, True, False]
    for i, is_first in enumerate(flags):
        buf.add(
            x=np.zeros((4, 4, 3), dtype=np.uint8), a=i, r=0.0, terminal=False, truncated=False,
            is_first=is_first, s=i,
        )
    model_batch, _ = buf.sample(batch_size=1, seq_len=5, rng=np.random.default_rng(0))
    assert list(model_batch.is_first[0]) == flags


# --- 6. truncated distinct de terminal -------------------------------------


def test_truncated_and_terminal_are_never_conflated_in_replay():
    buf = ReplayBuffer(capacity=4, image_shape=(2, 2, 3))
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    # a step that truncates (hits max_steps) but does NOT terminate
    buf.add(x=img, a=0, r=-1.0, terminal=False, truncated=True, is_first=False, s=1)
    # a step that terminates (reaches the goal) but does NOT truncate
    buf.add(x=img, a=1, r=50.0, terminal=True, truncated=False, is_first=False, s=2)

    model_batch, _ = buf.sample(batch_size=1, seq_len=2, rng=np.random.default_rng(0))
    terminal = model_batch.terminal[0]
    truncated = model_batch.truncated[0]
    assert list(terminal) == [False, True]
    assert list(truncated) == [True, False]
    assert not np.array_equal(terminal, truncated)


def test_e3_env_can_produce_both_a_truncation_and_a_termination_distinctly():
    from _experiment_loader import load_experiment_module

    e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_t0_e3_env")
    layout = e3_env.build_layout()
    from helplessness.envs.appearance.theme_sampling import build_themes

    themes = build_themes(e3_env.ZONE_CATEGORIES, goal_state=0, rng=np.random.default_rng(0), min_dist=5.0)

    # Truncation: start far from the goal, tiny max_steps, no way to
    # reach the goal in time.
    env = e3_env.build_pixel_env(layout, themes, goal_reward=50.0, start_position=(6, 6), max_steps=1)
    _, _, terminated, truncated, _ = env.step(4)  # "stay"
    assert truncated is True
    assert terminated is False

    # Termination: start one deterministic step away from the goal.
    env2 = e3_env.build_pixel_env(layout, themes, goal_reward=50.0, start_position=(11, 0), max_steps=36)
    # corridor is ControllableDeterministic; action "up" (index 0) from
    # (11, 0) moves to (10, 0) -- not the goal yet, so instead start
    # directly adjacent and step "up" twice is unnecessary: start at
    # (1, 0) and go "up" reaches (0, 0) = the goal directly.
    env3 = e3_env.build_pixel_env(layout, themes, goal_reward=50.0, start_position=(1, 0), max_steps=36)
    _, _, terminated3, truncated3, _ = env3.step(0)  # "up"
    assert terminated3 is True
    assert truncated3 is False
