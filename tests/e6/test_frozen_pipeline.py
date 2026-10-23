"""E6 and E7's torch side and shared set-up on a small untrained model (no trained weights needed): the episodes
(probing.frozen_episodes), the corrections of h (probing.belief_steering), the context walks (probing.context_walks)
and pipeline.frozen_setup -- reproducibility, the shared random draws of a pair, alpha = 0's identity, frozen weights."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import yaml

from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.probing.belief_steering import optimise_delta, prediction_loss, target_posteriors
from helplessness.probing.context_walks import SIDES, context_walks
from helplessness.probing.fixed_trajectory import fixed_trajectory_context
from helplessness.probing.frozen_episodes import explore_sequences, fresh_start_contexts, load_frozen_agent, run_episodes

CFG = fs.load_layered_config(REPO_ROOT / "configs" / "e6" / "e6.yaml", ("e6", "e7"))
E6 = {**fs.e6_protocol(CFG), "n_reps": 2, "draws": 4, "fresh_samples": 2, "phenotypes": {"k_revisit": 4, "n_min": 2, "n_boot": 5}}
DEV = torch.device("cpu")


def make_agent(path, seed):
    torch.manual_seed(seed)
    rssm = RSSM(RSSMConfig(base_depth=4, embed_dim=16, h_dim=16))
    acfg = AgentConfig(hidden_dim=16)
    goal, critic = GoalHead(16, 64, 16, zero_init=False), Critic(16, 64, 16, zero_init=False)
    torch.save({"rssm": rssm.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg), "goal_head": goal.state_dict(),
                "critic": critic.state_dict(), "agent_config": dataclasses.asdict(acfg)}, path)
    return load_frozen_agent(path, DEV)


@pytest.fixture(scope="module")
def world():
    conf = {**setup.configuration(CFG, "p1-b-d1"), "seed": 2}
    images = fs.test_images(conf, CFG["e6"]["images"])
    _, tw = fs.build_test_world(CFG, conf, images[0]["slot_colors"], DEV)
    return conf, images, tw


@pytest.fixture(scope="module")
def agents(tmp_path_factory):
    d = tmp_path_factory.mktemp("agents")
    return make_agent(d / "a.pt", 0), make_agent(d / "b.pt", 1)


def test_zone_and_images(world):
    conf, images, _ = world
    assert [im["name"] for im in images] == ["training", "position_only", "theme_only", "neutral"]
    assert [im["slot"] for im in images] == ["nw", "nw", "ne", "ne"]
    assert images[0]["is_training"] and images[3]["is_training"] and not images[1]["is_training"]
    assert images[2]["slot_colors"]["ne"] == conf["t_inc"]
    inner = fs.interior_states("nw")
    assert sorted(divmod(int(s), 12) for s in inner) == [(r, c) for r in (2, 3, 4) for c in (2, 3, 4)]
    assert fs.starts_for("nw", 2).tolist() == inner.tolist() * 2


def test_layered_config_refuses_overrides(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"base": "configs/e4/e4.yaml", "e6": {}, "train": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        fs.load_layered_config(bad, ("e6", "e7"))
    e7 = fs.load_layered_config(REPO_ROOT / "configs" / "e7" / "e7.yaml", ("e6", "e7"))
    assert e7["_bases"] == ["configs/e4/e4.yaml", "configs/e6/e6.yaml"] and e7["e6"] == CFG["e6"]


def test_episodes_follow_the_dynamics_and_are_reproducible(world, agents):
    _, images, tw = world
    a = agents[0]
    runs = [fs.run_model(a, tw, images[0], E6, 0.1) for _ in range(2)]
    for key in runs[0]:
        assert np.array_equal(runs[0][key], runs[1][key], equal_nan=True), key
    ep = runs[0]
    for i in range(len(ep["length"])):
        n = ep["length"][i]
        s, act = ep["states"][i, : n + 1], ep["actions"][i, :n]
        assert all(s[t + 1] == tw.succ[s[t], act[t]] for t in range(n))
        assert ep["terminated"][i] == (s[-1] == two_zones.GOAL_STATE)
        assert ep["terminated"][i] or n == E6["horizon"]
    assert np.isfinite(ep["I"][:, 0]).all()


def test_a_pair_shares_starts_and_epsilon_draws(world, agents):
    _, images, tw = world
    e = [fs.run_model(m, tw, images[0], E6, 0.3) for m in agents]
    assert np.array_equal(e[0]["states"][:, 0], e[1]["states"][:, 0])
    T = min(e[0]["length"].min(), e[1]["length"].min())
    assert np.array_equal(e[0]["explored"][:, :T], e[1]["explored"][:, :T])
    both = (e[0]["explored"][:, :T] == 1)
    assert np.array_equal(e[0]["actions"][:, :T][both], e[1]["actions"][:, :T][both])


def test_alpha_zero_hook_is_the_identity_and_weights_stay_frozen(world, agents):
    _, images, tw = world
    a = agents[0]
    before = {k: v.clone() for k, v in a.rssm.state_dict().items()}
    plain = fs.run_model(a, tw, images[0], E6, 0.1)
    v = np.random.default_rng(0).standard_normal(16)
    hooked = fs.run_model(a, tw, images[0], E6, 0.1, hook=lambda t, h, z, s, inz, rows: (h + 0.0 * torch.as_tensor(v, dtype=h.dtype), {}))
    for key in plain:
        assert np.array_equal(plain[key], hooked[key], equal_nan=True), key
    moved = fs.run_model(a, tw, images[0], E6, 0.1, hook=lambda t, h, z, s, inz, rows: (h + 3.0 * torch.as_tensor(v, dtype=h.dtype), {}))
    assert not np.array_equal(plain["I"], moved["I"], equal_nan=True)
    assert all(torch.equal(before[k], v_) for k, v_ in a.rssm.state_dict().items())


def test_fresh_start_is_the_first_step(world, agents):
    _, _, tw = world
    rssm = agents[0].rssm
    h, _ = fresh_start_contexts(rssm, tw, np.array([30, 40]), torch.Generator().manual_seed(0))
    h0, z0 = rssm.initial_state(2, DEV)
    with torch.no_grad():
        assert torch.allclose(h, rssm.gru_step(h0, z0, torch.zeros(2, 5)))


def test_rows_of_a_pair(world, agents):
    _, images, tw = world
    ep = {r: fs.run_model(m, tw, images[0], E6, 0.1) for r, m in zip(("experimental", "reference"), agents)}
    rf = fs.reference_fresh(agents[1], tw, images[0], E6)
    assert np.isnan(rf[two_zones.GOAL_STATE]) and np.isfinite(np.delete(rf, two_zones.GOAL_STATE)).all()
    b = fs.belief_row(ep["experimental"], ep["reference"], rf, images[0])
    assert b["b_step1"] == pytest.approx(b["I_step1"] - b["I_ref_step1"])
    assert fs.belief_row(ep["reference"], ep["reference"], rf, images[0])["b_step1"] == 0.0
    row = fs.phenotype_row(ep["experimental"], images[0], 0.1, E6)
    assert row["n_episodes"] == 18 and row["directed_share_lo"] <= row["directed_share_hi"]  # 5 resamples: an interval, not coverage


def test_correction_lowers_the_loss_and_never_touches_the_weights(world, agents):
    _, _, tw = world
    rssm = agents[0].rssm
    states = fs.zone_states("nw")[:6]
    h, z = fresh_start_contexts(rssm, tw, states, torch.Generator().manual_seed(0))
    target = target_posteriors(rssm, h, z, tw.succ[states], tw.templates_t)
    before = {k: v.clone() for k, v in rssm.state_dict().items()}
    out = optimise_delta(rssm, h, z, target, lam=0.01, max_steps=200)
    assert (out["kl"] < out["kl0"]).all()
    assert torch.allclose(prediction_loss(rssm, h, z, torch.zeros_like(h), target), out["kl0"])
    assert all(torch.equal(before[k], v) for k, v in rssm.state_dict().items())
    assert all(p.grad is None for p in rssm.parameters())


def test_context_walks_reach_their_targets(world):
    _, _, tw = world
    layout = two_zones.build_layout(1, "reference")
    zone, corridor = fs.zone_states("nw"), two_zones.corridor_states(layout)
    w = context_walks(tw.succ, zone, corridor, fs.goal_distances(), zone[:3], 8, np.random.default_rng(0), 12)
    assert set(w["side"]) == set(SIDES) and set(w["kind"]) == {"random", "goal"}
    for i in range(len(w["length"])):
        n = w["length"][i]
        s, a = w["states"][i, :n], w["actions"][i, :n]
        assert s[0] in set(corridor) and s[-1] == w["target"][i] and two_zones.GOAL_STATE not in s
        assert all(s[t + 1] == tw.succ[s[t], a[t]] for t in range(n - 1))
        assert set(s) <= set(corridor) | set(zone)
    images_np = setup.probe_world(CFG, {**setup.configuration(CFG, "p1-b-d1"), "seed": 2}).templates_np
    ctx = fixed_trajectory_context(RSSM(RSSMConfig(base_depth=4, embed_dim=16, h_dim=16)), images_np, w)
    assert ctx["h"].shape[:2] == w["states"].shape


def test_curve_points_selection():
    rows = [{"agent": "a", "condition": "c", "image": "i", "role": "exposed", "valid": True, "b_step1": -2.0, "recovery_latency": float("nan"),
             "recovery_censored": True, "recovery_max_index_read": 7, "loc_template_step1": 1.0, "loc_template_zone": 0.6},
            {"agent": "a", "condition": "c", "image": "i", "role": "reference", "valid": True, "b_step1": 0.0, "recovery_latency": 1.0,
             "recovery_censored": False, "loc_template_step1": 1.0, "loc_template_zone": 1.0},
            {"agent": "a", "condition": "c", "image": "j", "role": "exposed", "valid": "False", "b_step1": -1.0, "recovery_latency": 2.0,
             "recovery_censored": False, "loc_template_step1": 1.0, "loc_template_zone": 1.0}]
    p = fs.curve_points(rows, "recovery_latency", include_references=True)
    assert p["b"].tolist() == [-2.0, 0.0] and p["y"].tolist() == [8.0, 1.0] and p["censored"].tolist() == [True, False]
    assert fs.curve_points(rows, "recovery_latency", include_references=False)["b"].tolist() == [-2.0]


def test_explore_sequences_do_not_depend_on_the_batch():
    a = explore_sequences([1, 2, 3], 5)
    b = explore_sequences([1, 2, 3], 9)
    assert [np.random.default_rng(s).random() for s in a] == [np.random.default_rng(s).random() for s in b[:5]]


def test_run_episodes_measure_modes(world, agents):
    _, images, tw = world
    zone = np.zeros(144, bool)
    zone[fs.zone_states("nw")] = True
    starts = fs.starts_for("nw", 1)
    seqs = explore_sequences([0], len(starts))
    full = run_episodes(agents[0], tw, starts, 36, 0.1, seqs, 1, 2, zone, draws=4, measure="all")
    first = run_episodes(agents[0], tw, starts, 36, 0.1, seqs, 1, 2, zone, draws=4, measure="first")
    assert np.array_equal(full["states"], first["states"]) and np.array_equal(full["I"][:, 0], first["I"][:, 0])
    assert np.isnan(first["I"][:, 1:]).all()
