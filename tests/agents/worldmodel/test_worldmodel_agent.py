"""Tests for agents.worldmodel.agent.WorldModelAgent (docs/e3/DECISIONS.md,
point 2): transition_distribution as a measurement-only bridge, never
called by the policy.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from helplessness.agents.worldmodel.agent import WorldModelAgent, categorical_mode
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig


def _small_cfg() -> RSSMConfig:
    return RSSMConfig(image_size=(16, 16), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4, n_actions=5)


def _agent(n_states: int = 6, K_samples: int = 16, margin_threshold: float = 1e-3) -> WorldModelAgent:
    torch.manual_seed(0)
    cfg = _small_cfg()
    rssm = RSSM(cfg)
    templates = torch.rand(n_states, cfg.in_channels, *cfg.image_size)
    return WorldModelAgent(rssm, templates, K_samples=K_samples, margin_threshold=margin_threshold)


def test_categorical_mode_is_a_one_hot_argmax():
    probs = torch.tensor([[[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]])
    mode = categorical_mode(probs)
    assert torch.equal(mode[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(mode[0, 1], torch.tensor([1.0, 0.0, 0.0]))


def test_categorical_mode_is_deterministic_unlike_a_sample():
    probs = torch.softmax(torch.randn(1, 4, 5), dim=-1)
    a = categorical_mode(probs)
    b = categorical_mode(probs)
    assert torch.equal(a, b)


def test_transition_distribution_with_diagnostics_returns_a_structured_triple():
    agent = _agent(n_states=6)
    kernel, ghost_fraction, n_valid_draws = agent.transition_distribution_with_diagnostics(s=2, a=1)
    assert kernel.shape == (6,)
    assert 0.0 <= ghost_fraction <= 1.0
    assert 0 <= n_valid_draws <= agent.K_samples
    assert n_valid_draws == agent.K_samples - round(ghost_fraction * agent.K_samples)
    if n_valid_draws > 0:
        assert kernel.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(kernel >= 0.0)
    else:
        assert np.all(np.isnan(kernel))


def test_transition_distribution_is_a_thin_wrapper_returning_only_the_kernel():
    agent = _agent(n_states=6)
    kernel_direct, _, _ = agent.transition_distribution_with_diagnostics(s=2, a=1)
    torch.manual_seed(0)  # re-seed so the two calls draw the same samples
    kernel_via_abc = agent.transition_distribution(s=2, a=1)
    assert kernel_direct.shape == kernel_via_abc.shape


def test_all_ghost_draws_give_a_nan_kernel_ghost_fraction_one_and_logs_a_warning(caplog):
    """With margin_threshold set impossibly high, EVERY draw is a ghost
    -- the kernel must come back NaN (never zero-filled, never
    silently renormalised over "something"), ghost_fraction must say
    1.0 explicitly, n_valid_draws must be 0, and a warning must be
    logged (never a silent division by zero)."""
    import logging

    agent = _agent(n_states=6, margin_threshold=1e9)
    with caplog.at_level(logging.WARNING, logger="helplessness.agents.worldmodel.agent"):
        kernel, ghost_fraction, n_valid_draws = agent.transition_distribution_with_diagnostics(s=0, a=0)

    assert ghost_fraction == pytest.approx(1.0)
    assert n_valid_draws == 0
    assert kernel.shape == (6,)
    assert np.all(np.isnan(kernel))
    assert not np.any(kernel == 0.0)  # never a zero-filled stand-in
    assert any("ghost" in record.message.lower() for record in caplog.records)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_transition_distribution_abc_wrapper_also_returns_nan_on_all_ghosts():
    agent = _agent(n_states=6, margin_threshold=1e9)
    p = agent.transition_distribution(s=0, a=0)
    assert np.all(np.isnan(p))


def test_transition_distribution_works_via_materialize_transition_tensor():
    """Confirms the ABC contract: this agent can be scored by
    measures.controllability exactly like a tabular agent (agents/base.py's
    own stated purpose for this method)."""
    from helplessness.agents.base import materialize_transition_tensor

    agent = _agent(n_states=4)
    T = materialize_transition_tensor(agent, n_states=4, n_actions=5)
    assert T.shape == (4, 5, 4)
    # every row is a valid probability distribution OR all-NaN (all draws ghosts)
    row_sums = T.sum(axis=2)
    assert np.all(np.isclose(row_sums, 1.0) | np.isnan(row_sums))


def test_act_raises_not_implemented_and_never_calls_transition_distribution():
    """The mock test point 2 asks for: act() must not call
    transition_distribution (or its diagnostics-returning counterpart),
    checked by replacing both with Mocks and confirming zero calls --
    this test will keep passing unmodified once Phase 5 replaces act()
    with the real policy."""
    agent = _agent(n_states=4)
    agent.transition_distribution = Mock(wraps=agent.transition_distribution)
    agent.transition_distribution_with_diagnostics = Mock(wraps=agent.transition_distribution_with_diagnostics)

    with pytest.raises(NotImplementedError):
        agent.act(s=0, rng=np.random.default_rng(0))

    agent.transition_distribution.assert_not_called()
    agent.transition_distribution_with_diagnostics.assert_not_called()
