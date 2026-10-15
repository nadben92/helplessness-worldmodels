"""E4's fixed walks (helplessness.probing.walks) and the fixed-trajectory measure (helplessness.probing.fixed_trajectory)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from _experiment_loader import load_experiment_module
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, sample_categorical_index
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.measures.kernel_probe import match_from_distances, template_distances
from helplessness.pipeline.measure import controllability_of_kernel
from helplessness.probing import walks as W
from helplessness.probing.fixed_trajectory import measure_walks, step_controllability, summarize_zone

e4_setup = load_experiment_module("E4_generalization_gradient", "e4_setup.py", "test_e4_walks_setup")
CFG = e4_setup.load_config()


@pytest.fixture(scope="module")
def env():
    layout = two_zones.build_layout(1, "reference")
    world = e4_setup.probe_world(CFG, e4_setup.configuration(CFG, "p1-a-d1"))
    return layout, world, world.env.transition_tensor()


@pytest.mark.parametrize("slot", QUADRANT_ORDER)
def test_walks_stay_legal(env, slot):
    layout, _, P = env
    regions = two_zones.region_states(layout)
    zone, corridor = set(regions[slot]), set(two_zones.corridor_states(layout))
    w = e4_setup.walks_for_slot(CFG, slot)
    succ = P.argmax(-1)
    n_pre, n_in = CFG["e4"]["walks"]["n_pre"], CFG["e4"]["walks"]["n_in"]
    assert w["states"].shape[0] == CFG["e4"]["walks"]["n_walks"]
    for i in range(w["states"].shape[0]):
        n = w["length"][i]
        states, actions, phase = w["states"][i, :n], w["actions"][i, :n], w["phase"][i, :n]
        assert two_zones.GOAL_STATE not in states                                       # never the goal
        assert set(states) <= zone | corridor                                           # never another quadrant
        assert all(states[t + 1] == succ[states[t], actions[t]] for t in range(n - 1))  # true dynamics
        assert set(states[phase == W.PHASE_PRE]) <= corridor and (phase == W.PHASE_PRE).sum() == n_pre + 1
        assert (phase == W.PHASE_ZONE).sum() == n_in and set(states[phase == W.PHASE_ZONE]) <= zone
        k = w["steps_since_entry"][i, :n]
        assert list(k[phase == W.PHASE_ZONE]) == list(range(1, n_in + 1)) and states[phase == W.PHASE_ZONE][0] == w["entry_state"][i]
        assert np.all(w["states"][i, n:] == -1)


def test_walks_reproducible_shared_and_prefix_consistent():
    a = e4_setup.walks_for_slot(CFG, "nw")
    e4_setup._walks.cache_clear()
    b = e4_setup.walks_for_slot(CFG, "nw")
    assert all(np.array_equal(a[k], b[k]) for k in a)
    small = e4_setup.walks_for_slot(CFG, "nw", 50)
    T = small["states"].shape[1]
    assert all(np.array_equal(small[k], a[k][:50, :T] if a[k].ndim == 2 else a[k][:50]) for k in small)
    other = e4_setup.walks_for_slot(CFG, "se")
    assert not np.array_equal(a["entry_state"], other["entry_state"])


def tiny_rssm(seed: int) -> RSSM:
    torch.manual_seed(seed)
    return RSSM(RSSMConfig(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)).eval()


def naive_kernel(rssm, h, z, templates_t, draws, margin, seed):
    """Every draw decoded on its own (no deduplication), numpy's template distances: the reference implementation."""
    gen = torch.Generator().manual_seed(seed)
    A, V, C = rssm.cfg.n_actions, rssm.cfg.n_vars, rssm.cfg.n_classes
    p_hat = np.full((h.shape[0], A, templates_t.shape[0]), np.nan)
    with torch.no_grad():
        h_next, prior = torch.zeros(h.shape[0], A, rssm.cfg.h_dim), torch.zeros(h.shape[0], A, V, C)
        for a in range(A):
            out = rssm.imagination_step(h, z, F.one_hot(torch.full((h.shape[0],), a), A).float(), generator=gen)
            h_next[:, a], prior[:, a] = out["h"], out["prior_probs"]
        rows = h.shape[0] * A
        idx = sample_categorical_index(prior.reshape(rows, 1, V, C).expand(rows, draws, V, C).reshape(rows * draws, V, C), gen)
        decoded = rssm.decode(h_next.reshape(rows, -1).repeat_interleave(draws, 0), F.one_hot(idx, C).float()).numpy()
    m = match_from_distances(template_distances(decoded, templates_t.numpy()), margin)
    near, ghost = m["nearest_state"].reshape(rows, draws), m["is_ghost"].reshape(rows, draws)
    for r in range(rows):
        ok = ~ghost[r]
        if ok.any():
            p_hat.reshape(rows, -1)[r] = np.bincount(near[r][ok], minlength=templates_t.shape[0]) / ok.sum()
    return controllability_of_kernel(p_hat)


def test_deduplicated_decoding_equals_decoding_every_draw(env):
    _, world, _ = env
    rssm = tiny_rssm(0)
    torch.manual_seed(1)
    h = torch.randn(7, rssm.cfg.h_dim)
    z = F.one_hot(torch.randint(0, rssm.cfg.n_classes, (7, rssm.cfg.n_vars)), rssm.cfg.n_classes).float()
    fast = step_controllability(rssm, h, z, world.templates_t, draws=6, margin_threshold=1e-3, generator=torch.Generator().manual_seed(3))
    slow = naive_kernel(rssm, h, z, world.templates_t, 6, 1e-3, 3)
    assert np.allclose(fast["I"], slow, equal_nan=True)


def test_summarize_zone_by_state_then_zone():
    values = np.array([1.0, 3.0, 2.0, 10.0, np.nan])
    states = np.array([5, 5, 6, 7, 8])
    k = np.array([1, 2, 1, 6, 1])
    s = summarize_zone(values, states, k, m=5, n_in=6)
    assert s["entry"] == pytest.approx((2.0 + 2.0) / 2) and s["n_states_entry"] == 2  # state 5: mean(1, 3); state 6: 2
    assert s["curve"][0] == pytest.approx(1.5) and s["n_states_curve"][0] == 2       # k=1: states 5 (1) and 6 (2)
    assert s["curve"][5] == pytest.approx(10.0) and np.isnan(s["curve"][3])


def test_measure_identical_for_the_same_model_and_seed(env):
    _, world, P = env
    rssm = tiny_rssm(2)
    w = e4_setup.walks_for_slot(CFG, "nw", 4)
    args = (world.templates_np, world.templates_t, w, P.argmax(-1), 4, 1e-3, 123, 5, 25)
    one, two = measure_walks(rssm, *args), measure_walks(tiny_rssm(2), *args)
    assert np.array_equal(one["I"], two["I"], equal_nan=True) and one["entry"] == pytest.approx(two["entry"], nan_ok=True)
    assert (one["steps_since_entry"] >= 1).all() and one["I"].shape == (4 * 25,)
