"""The probabilities-phase aids (docs/e3/DECISIONS.md, 2026-09-24): the
partial straight-through estimator, the posterior entropy term, and the
ProbsPhaseAid schedule that drives them during a SeparationSwitch's
probabilities phase."""

from __future__ import annotations

import math

import pytest
import torch

from helplessness.agents.worldmodel.loop import ProbsPhaseAid, SeparationSwitch
from helplessness.agents.worldmodel.losses import mean_categorical_entropy
from helplessness.agents.worldmodel.rssm import LatentEstimator, partial_straight_through, sample_posterior


def _probs(batch=6, n_vars=8, n_classes=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(batch, n_vars, n_classes, generator=g), dim=-1)


def _is_one_hot(z):
    return torch.isclose(z.max(dim=-1).values, torch.tensor(1.0)) & torch.isclose(z.sum(dim=-1), torch.tensor(1.0))


@pytest.mark.parametrize("rho, n_sampled", [(0.0, 0), (0.25, 2), (0.5, 4), (0.8, 6), (1.0, 8)])
def test_partial_straight_through_samples_exactly_round_rho_n_vars_variables_per_example(rho, n_sampled):
    probs = _probs()
    torch.manual_seed(1)
    z = partial_straight_through(probs, rho)
    sampled = _is_one_hot(z) & ~torch.isclose(z, probs).all(dim=-1)
    assert (sampled.sum(dim=-1) == n_sampled).all()
    kept = ~sampled
    assert torch.allclose(z[kept], probs[kept])


def test_partial_straight_through_picks_different_variables_across_examples():
    probs = _probs(batch=64)
    torch.manual_seed(2)
    sampled = ~torch.isclose(partial_straight_through(probs, 0.5), probs).all(dim=-1)
    assert len({tuple(row.tolist()) for row in sampled}) > 1


def test_partial_straight_through_gradient_is_the_identity_on_every_variable():
    probs = _probs().requires_grad_(True)
    torch.manual_seed(3)
    weights = torch.randn_like(probs)
    (partial_straight_through(probs, 0.5) * weights).sum().backward()
    assert torch.allclose(probs.grad, weights)


def test_sample_posterior_dispatches_partial_straight_through_and_validates_rho():
    probs = _probs()
    assert torch.equal(sample_posterior(probs, LatentEstimator("partial_straight_through", rho=0.0)), probs)
    with pytest.raises(ValueError):
        LatentEstimator("partial_straight_through", rho=1.5)


def test_mean_categorical_entropy_hand_values():
    uniform = torch.full((3, 8, 8), 1 / 8)
    assert math.isclose(float(mean_categorical_entropy(uniform)), 8 * math.log(8), rel_tol=1e-6)
    one_hot = torch.zeros(2, 8, 8)
    one_hot[..., 0] = 1.0
    assert float(mean_categorical_entropy(one_hot)) == pytest.approx(0.0, abs=1e-9)
    two = torch.tensor([[[0.5, 0.5]], [[1.0, 0.0]]])  # (2 examples, 1 var, 2 classes): ln 2 and 0 -> mean ln2 / 2
    assert math.isclose(float(mean_categorical_entropy(two)), math.log(2) / 2, rel_tol=1e-6)


def test_entropy_weight_ramps_linearly_over_the_deadline_and_stops_at_the_switch():
    aid = ProbsPhaseAid("entropy_penalty", eta_max=2.0, ramp_steps=100)
    assert aid.entropy_weight(0, switched=False) == 0.0
    assert aid.entropy_weight(25, switched=False) == pytest.approx(0.5)
    assert aid.entropy_weight(100, switched=False) == pytest.approx(2.0)
    assert aid.entropy_weight(150, switched=False) == pytest.approx(2.0)
    assert aid.entropy_weight(50, switched=True) == 0.0
    assert ProbsPhaseAid("progressive_samples", ramp_steps=100).entropy_weight(50, switched=False) == 0.0


def test_progressive_estimator_ramps_rho_then_hands_back_to_the_switch():
    switch = SeparationSwitch(max_steps=1000)
    aid = ProbsPhaseAid("progressive_samples", ramp_steps=1000)
    assert aid.estimator(0, switch) == LatentEstimator("partial_straight_through", rho=0.0)
    assert aid.estimator(250, switch) == LatentEstimator("partial_straight_through", rho=0.25)
    switch.switch_step = 300
    assert aid.estimator(400, switch) == LatentEstimator("straight_through")
    assert ProbsPhaseAid("entropy_penalty", ramp_steps=10).estimator(5, SeparationSwitch(max_steps=1000)) == LatentEstimator("probs")


def test_probs_phase_aid_validation():
    with pytest.raises(ValueError):
        ProbsPhaseAid("temperature")
    with pytest.raises(ValueError):
        ProbsPhaseAid("entropy_penalty", ramp_steps=0)
