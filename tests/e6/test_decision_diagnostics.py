"""E6/E7's diagnostics of the value and of the decision rule (docs/e6/DECISIONS.md (52)) on a small untrained model:
measures.decision_rule, probing.lookahead (k = 1 is the policy's Q, the oracle reads the true successors and rewards)
and run_episodes' test-time options (the defaults are E6's, tau -> 0 is the argmax), weights untouched."""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest
import torch

from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead
from helplessness.agents.worldmodel.policy import q_values
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.measures import decision_rule as dr
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup
from helplessness.probing.frozen_episodes import fresh_start_contexts, load_frozen_agent
from helplessness.probing.lookahead import cell_values, embeddings, k_step_q, oracle_q

CFG = fs.load_layered_config(REPO_ROOT / "configs" / "e6" / "e6.yaml", ("e6", "e7"))
E6 = {**fs.e6_protocol(CFG), "n_reps": 1, "draws": 4, "fresh_samples": 2, "horizon": 8}
DEV = torch.device("cpu")


def weights_hash(agent) -> str:
    h = hashlib.sha256()
    for module in (agent.rssm, agent.goal_head, agent.critic):
        for k, v in module.state_dict().items():
            h.update(k.encode()), h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def world():
    conf = {**setup.configuration(CFG, "p1-b-d1"), "seed": 2}
    images = fs.test_images(conf, CFG["e6"]["images"])
    _, tw = fs.build_test_world(CFG, conf, images[0]["slot_colors"], DEV)
    return images, tw


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    torch.manual_seed(0)
    rssm = RSSM(RSSMConfig(base_depth=4, embed_dim=16, h_dim=16))
    acfg = AgentConfig(hidden_dim=16)
    goal, critic = GoalHead(16, 64, 16, zero_init=False), Critic(16, 64, 16, zero_init=False)
    path = tmp_path_factory.mktemp("agent") / "a.pt"
    torch.save({"rssm": rssm.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg), "goal_head": goal.state_dict(),
                "critic": critic.state_dict(), "agent_config": dataclasses.asdict(acfg)}, path)
    return load_frozen_agent(path, DEV)


def test_decision_rule_measures():
    succ = np.array([[1, 2, 0], [1, 1, 2], [2, 2, 2]])
    dist = np.array([2, 1, 0])
    assert dr.directed_action_mask(succ, dist).tolist() == [[True, True, False], [False, False, True], [False, False, False]]
    Q = np.array([[1.0, 0.95, 0.2], [0.0, 3.0, 2.0]])
    assert np.allclose(dr.best_second_gap(Q), [0.05, 1.0])
    directed = np.array([[False, True, True], [True, False, True]])
    assert np.allclose(dr.greedy_directed_share(Q, directed, 0.1), [0.5, 0.0])
    assert dr.argmax_directed(Q, directed).tolist() == [False, False]
    p = dr.softmax_probs(Q, 1.0)
    assert np.allclose(p.sum(axis=1), 1.0) and np.argmax(p[1]) == 1
    assert dr.softmax_choice(np.array([0.0, 5.0, 1.0]), 1e-6, 0.999) == 1 and dr.softmax_choice(np.zeros(4), 1.0, 0.6) == 2
    b = dr.binned(np.array([-0.1, -0.2, -0.3, np.nan]), np.array([1.0, 3.0, 5.0, 7.0]), np.array([-0.5, -0.25, 0.0]))
    assert b["n"].tolist() == [1, 2] and b["median"].tolist() == [5.0, 2.0]


def test_k1_model_is_the_policy_q(world, agent):
    _, tw = world
    states = np.array([26, 27, 40, 50])
    h, z = fresh_start_contexts(agent.rssm, tw, states, torch.Generator().manual_seed(3))
    Q_policy = q_values(agent.rssm, agent.goal_head, agent.critic, h, z, agent.cfg.k_act, agent.cfg, torch.Generator().manual_seed(9))
    out = k_step_q(agent, tw, h, z, states, 3, "model", agent.cfg.k_act, 4, torch.Generator().manual_seed(9), n_trace=2)
    assert torch.equal(out["Q"][..., 0], Q_policy)
    assert out["Q"].shape == (4, 5, 3) and out["trace_h"].shape == (3, 4, 5, 2, 16)


def test_oracle_reads_the_truth(world, agent):
    _, tw = world
    emb = embeddings(agent, tw)
    near = np.array([s for s in range(two_zones.N_STATES) if s != tw.goal_state and (tw.succ[s] == tw.goal_state).any()][:3])
    h, z = fresh_start_contexts(agent.rssm, tw, near, torch.Generator().manual_seed(1))
    Q = oracle_q(agent, tw, emb, h, z, near, 4, torch.Generator().manual_seed(2))
    to_goal = tw.succ[near] == tw.goal_state
    assert torch.allclose(Q[torch.as_tensor(to_goal)], torch.tensor(agent.cfg.goal_reward))  # c = 0 at the goal: r only
    out = k_step_q(agent, tw, h, z, near, 4, "oracle", 4, 4, torch.Generator().manual_seed(2), embeds=emb, n_trace=1)
    assert torch.allclose(out["Q"][..., 0], Q)  # the oracle's k = 1 is the oracle one-step Q (the same draws)
    assert torch.allclose(out["Q"][torch.as_tensor(to_goal)], torch.tensor(agent.cfg.goal_reward))  # nothing counts after the goal
    s = out["trace_states"]  # the particles follow the true dynamics
    for i in range(1, s.shape[0]):
        assert np.array_equal(s[i], tw.succ[s[i - 1], out["trace_actions"][i].numpy()])


def test_run_episodes_defaults_and_tau(world, agent):
    images, tw = world
    before = weights_hash(agent)
    plain = fs.run_model(agent, tw, images[0], E6, 0.1)
    explicit = fs.run_model(agent, tw, images[0], E6, 0.1, q_fn=lambda h, z, s, g: q_values(agent.rssm, agent.goal_head, agent.critic, h, z,
                                                                                               agent.cfg.k_act, agent.cfg, g))
    for key in plain:
        assert np.array_equal(plain[key], explicit[key], equal_nan=True), key
    cold = fs.run_model(agent, tw, images[0], E6, 0.1, tau=1e-9)
    steps = cold["actions"] >= 0
    assert np.array_equal(cold["actions"][steps], np.argmax(cold["Q"], axis=-1)[steps]) and not np.nansum(cold["explored"])
    assert np.array_equal(cold["states"][:, 0], plain["states"][:, 0])
    cell_values(agent, tw, np.array([0, 1]), 2, torch.Generator().manual_seed(0))
    assert weights_hash(agent) == before


def test_kernel_option_and_anti_oracle(world, agent):
    from helplessness.pipeline.measure import controllability_of_kernel
    from helplessness.probing.fixed_trajectory import step_controllability
    from helplessness.probing.lookahead import mixed_policy_q

    images, tw = world
    conf = {**setup.configuration(CFG, "p1-b-d1"), "seed": 2}
    conveyor = setup.successors(setup.world(CFG, conf, "experimental"))[:, 4]
    alt = np.repeat(conveyor[:, None], 5, axis=1)
    zone = np.zeros(two_zones.N_STATES, dtype=bool)
    zone[fs.zone_states(images[0]["slot"])] = True
    states = fs.interior_states(images[0]["slot"])[:4]
    h, z = fresh_start_contexts(agent.rssm, tw, states, torch.Generator().manual_seed(3))
    plain = step_controllability(agent.rssm, h, z, tw.templates_t, 4, 1e-3, generator=torch.Generator().manual_seed(1))
    kern = step_controllability(agent.rssm, h, z, tw.templates_t, 4, 1e-3, generator=torch.Generator().manual_seed(1), return_kernel=True)
    assert "p_hat" not in plain and np.array_equal(kern["I"], plain["I"], equal_nan=True)
    assert np.array_equal(controllability_of_kernel(kern["p_hat"]), kern["I"], equal_nan=True)
    emb = embeddings(agent, tw)
    q0 = mixed_policy_q(agent, tw, emb, h, z, states, 0.0, zone, alt, torch.Generator().manual_seed(9), torch.Generator().manual_seed(5))
    assert torch.equal(q0, q_values(agent.rssm, agent.goal_head, agent.critic, h, z, agent.cfg.k_act, agent.cfg, torch.Generator().manual_seed(9)))
    q1 = mixed_policy_q(agent, tw, emb, h, z, states, 1.0, zone, alt, torch.Generator().manual_seed(9), torch.Generator().manual_seed(5))
    assert torch.allclose(q1, oracle_q(agent, tw, emb, h, z, states, agent.cfg.k_act, torch.Generator().manual_seed(5), succ=alt, h_mode="mean"))
    assert torch.equal(q1, q1[:, :1].expand_as(q1))  # (57): one cell for every action, one h -- equal Q
    leak = mixed_policy_q(agent, tw, emb, h, z, states, 1.0, zone, alt, torch.Generator().manual_seed(9), torch.Generator().manual_seed(5), h_mode="action")
    assert not torch.equal(leak, leak[:, :1].expand_as(leak))  # the action reaches V through h'_a
    for p, follows in ((0.0, tw.succ), (1.0, alt)):
        out = k_step_q(agent, tw, h, z, states, 3, "mixed", 4, 2, torch.Generator().manual_seed(2), embeds=emb, n_trace=2, p_mix=p, mix=zone,
                       alt_succ=alt)
        assert torch.isfinite(out["Q"]).all()
        s, act = out["trace_states"], out["trace_actions"].numpy()
        assert np.array_equal(s[0], follows[np.repeat(states, 5)[:, None].reshape(4, 5)[..., None].repeat(2, -1), act[0]])
        for i in range(1, s.shape[0]):  # p = 1: the conveyor from a zone cell, the world outside; p = 0: the world
            expect = np.where(zone[s[i - 1]] & (p == 1.0), alt[s[i - 1], act[i]], tw.succ[s[i - 1], act[i]])
            assert np.array_equal(s[i], expect)
    # p = 0 at k = 1 through the episodes: E6 exactly
    plain_ep = fs.run_model(agent, tw, images[0], E6, 0.1)
    mixed_ep = fs.run_model(agent, tw, images[0], E6, 0.1, q_fn=lambda h_, z_, s_, g: mixed_policy_q(agent, tw, emb, h_, z_, s_, 0.0, zone, alt, g, None))
    for key in plain_ep:
        assert np.array_equal(plain_ep[key], mixed_ep[key], equal_nan=True), key


def test_action_kernel_and_anti_oracle_by_draws(world, agent):
    from helplessness.probing.lookahead import mixed_policy_q, sampled_mixed_policy_q

    images, tw = world
    plain = fs.run_model(agent, tw, images[0], E6, 0.1)
    rich = fs.run_model(agent, tw, images[0], E6, 0.1, action_kernel=True)
    for key in plain:  # the option adds arrays, never changes one
        assert np.array_equal(plain[key], rich[key], equal_nan=True), key
    played = np.take_along_axis(rich["mass_true_all"], np.clip(rich["actions"], 0, None)[..., None], axis=2)[..., 0]
    measured = np.isfinite(rich["mass_played"])
    assert np.allclose(played[measured], rich["mass_played"][measured])
    assert set(np.unique(rich["argmax_true"][np.isfinite(rich["argmax_true"])])) <= {0.0, 1.0}
    conf = {**setup.configuration(CFG, "p1-b-d1"), "seed": 2}
    alt = np.repeat(setup.successors(setup.world(CFG, conf, "experimental"))[:, 4][:, None], 5, axis=1)
    zone = np.zeros(two_zones.N_STATES, dtype=bool)
    zone[fs.zone_states(images[0]["slot"])] = True
    states = fs.interior_states(images[0]["slot"])[:4]
    h, z = fresh_start_contexts(agent.rssm, tw, states, torch.Generator().manual_seed(3))
    emb = embeddings(agent, tw)
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    q_pol = q_values(agent.rssm, agent.goal_head, agent.critic, h, z, agent.cfg.k_act, agent.cfg, g(9))
    assert torch.equal(sampled_mixed_policy_q(agent, tw, emb, h, z, states, 0.0, zone, alt, g(9), g(5)), q_pol)
    q1 = sampled_mixed_policy_q(agent, tw, emb, h, z, states, 1.0, zone, alt, g(9), g(5))
    assert torch.equal(q1, q1[:, :1].expand_as(q1))  # every draw on the conveyor's cell, one h: equal Q
    assert torch.allclose(q1, mixed_policy_q(agent, tw, emb, h, z, states, 1.0, zone, alt, g(9), g(5)))  # p = 1: the expectation's
    qs = sampled_mixed_policy_q(agent, tw, emb, h, z, states, 0.5, zone, alt, g(9), g(5), shared=True)
    for i in range(len(states)):  # one Bernoulli per row: the model's row or a constant row
        assert torch.allclose(qs[i], q_pol[i]) or torch.equal(qs[i], qs[i, :1].expand_as(qs[i]))


def test_channel_q(world, agent):
    from helplessness.probing.lookahead import channel_q

    _, tw = world
    states = np.array([26, 27, 40, 50])
    h, z = fresh_start_contexts(agent.rssm, tw, states, torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(9)
    clone = torch.Generator()
    clone.set_state(g.get_state())
    ch = channel_q(agent, h, z, clone, torch.Generator().manual_seed(4))
    q = q_values(agent.rssm, agent.goal_head, agent.critic, h, z, agent.cfg.k_act, agent.cfg, g)
    assert torch.equal(ch["Q_normal"], q)  # (a): the policy's Q, bit for bit, from a clone of its generator
    assert torch.equal(clone.get_state(), g.get_state())  # the clone consumed exactly what q_values consumes
    for key in ("Q_hbar", "Q_zbar", "V_zbar"):
        assert ch[key].shape == q.shape and torch.isfinite(ch[key]).all()
    clone.set_state(g.get_state())  # the option adds a key, changes nothing else
    g2 = torch.Generator()
    g2.set_state(torch.Generator().manual_seed(9).get_state())
    plain = channel_q(agent, h, z, g2, torch.Generator().manual_seed(4))
    g3 = torch.Generator().manual_seed(9)
    rich = channel_q(agent, h, z, g3, torch.Generator().manual_seed(4), mean_zbar=True)
    assert all(torch.equal(plain[k], rich[k]) for k in plain) and rich["V_zbar_mean"].shape == q.shape
