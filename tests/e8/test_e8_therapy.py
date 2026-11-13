"""E8's therapy runs on E5's mini configuration (CPU; docs/e8/DECISIONS.md (4)-(8), the brief's section 7): a start file
from a short mini training, then the therapy -- checkpoint 0 equal to the start file; the resets (empty replay, new
optimisers, critic and slow critic loaded, the post-switch schedule from step 0 that never switches again); a therapy
at learning rate 0 leaves the weights the measures read untouched; the therapy environments (the baseline's is E5's
reference environment; a shuffle therapy on nw draws only the 3 assignments with T_inc in nw, uniformly, with the
reference's dynamics; "all" equals E5's f = 1 draws); an interrupted therapy resumed equals the uninterrupted one; the
library's new options leave E5's records unchanged by default."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from _experiment_loader import load_experiment_module
from helplessness.agents.worldmodel.loop import SeparationSwitch
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import ClosedLoopHooks, FreshStart, run_one
from helplessness.pipeline.episode_sources import (ASSIGNMENTS, TreatedAssignments, ZoneAssignmentShuffle, assignment_world,
                                                   canonical_assignment, treated_assignments)
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import make_hooks as e4_hooks
from helplessness.streams import therapy as therapy_streams

e8_setup = load_experiment_module("E8_therapy_relapse", "e8_setup.py", "test_e8_setup")
e8_therapy = load_experiment_module("E8_therapy_relapse", "e8_therapy.py", "test_e8_therapy_hooks")
run_e5 = load_experiment_module("E5_belief_training", "run_e5.py", "test_e8_run_e5")
MINI = REPO_ROOT / "configs" / "e8" / "e8_mini.yaml"
SEED = 2
WEIGHT_KEYS = ("rssm", "goal_head", "critic")


def mini_cfg() -> dict:
    cfg = e8_setup.load_config(MINI)
    two_zones_setup.apply_e5_procedure(cfg)
    cfg["e4"]["run_configuration"] = "p1-b-d1"
    return cfg


def conf_of(cfg: dict) -> dict:
    return {**two_zones_setup.configuration(cfg, "p1-b-d1"), "seed": SEED}


@pytest.fixture(scope="module")
def start_file(tmp_path_factory):
    """A switched mini model: E4's hooks (records off), 120 steps, its agent_env000120.pt."""
    root = tmp_path_factory.mktemp("start")
    cfg = mini_cfg()
    cfg["checkpoints"]["grid"] = [120]
    conf = conf_of(cfg)
    base = e4_hooks(cfg, conf)
    hooks = ClosedLoopHooks(episode=base.episode, build_world=base.build_world, checkpoint_record=lambda s: {},
                            finalize=lambda r, w, d: {})
    run_one(cfg, SEED, "experimental", root / "start", configure_runtime(cfg), hooks=hooks, max_env_steps=120)
    path = root / "start" / "checkpoints" / "agent_env000120.pt"
    assert torch.load(path, weights_only=False)["switch_step"] is not None, "the start model must have switched"
    return path


def therapy(start, out, therapy_name="full", cfg=None, **kwargs):
    cfg = mini_cfg() if cfg is None else cfg
    cfg["checkpoints"]["grid"] = e8_setup.checkpoint_grid(cfg)
    conf = conf_of(cfg)
    hooks = e8_therapy.make_hooks(cfg, conf, therapy_name, e8_setup.treated_slot({"kind": "x"}, conf, therapy_name), SEED)
    fresh = FreshStart(weights=start, stream_root=therapy_streams.therapy_root(SEED))
    run_one(cfg, SEED, "experimental", out, configure_runtime(cfg), hooks=hooks, total_env_steps=int(cfg["e8"]["max_env_steps"]),
            save_resumable_at=tuple(cfg["checkpoints"]["grid"]), fresh_start=fresh, **kwargs)
    return [json.loads(line) for line in (out / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]


def weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same_state(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def baseline_run(start_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("therapy") / "base"
    return out, therapy(start_file, out)


def test_checkpoint_zero_is_the_start_file(start_file, baseline_run):
    out, records = baseline_run
    assert [r["env_step"] for r in records] == [0, 50, 100, 200]
    zero, start = weights(out / "checkpoints" / "agent_env000000.pt"), weights(start_file)
    for key in (*WEIGHT_KEYS, "slow_critic"):
        assert same_state(zero[key], start[key]), key
    assert records[0]["stored_steps"] == 0 and records[0]["grad_step"] == 0  # the replay starts empty
    later = weights(out / "checkpoints" / "agent_env000200.pt")
    assert not same_state(later["rssm"], start["rssm"])  # the therapy trains


def test_resets_and_schedule(start_file, baseline_run):
    out, records = baseline_run
    state0 = torch.load(out / "checkpoints" / "resumable_env000000.pt", weights_only=False)
    for key in ("optimizer", "goal_optimizer", "critic_optimizer"):  # new optimisers: no moment yet
        assert state0[key]["state"] == {}, key
    assert len(pickle_loads(state0["replay"])) == 0
    assert state0["switch"]["switch_step"] == 0 and state0["progress"]["rep_warmup"] == 0
    assert all(r["switch_step"] == 0 for r in records)  # never switches again
    meta = yaml.safe_load((out / "meta.yaml").read_text(encoding="utf-8"))
    start = weights(start_file)
    assert meta["fresh_start"]["origin_switch_step"] == start["switch_step"] and meta["fresh_start"]["origin_env_step"] == 120
    # prefill: no gradient step during the first prefill_steps environment steps, then replay_ratio per step
    cfg = mini_cfg()
    prefill, ratio = cfg["budget"]["prefill_steps"], cfg["train"]["replay_ratio"]
    for r in records[1:]:
        assert r["grad_step"] == int(max(0, r["env_step"] - prefill) * ratio)


def test_the_switch_cannot_fire_again():
    from helplessness.agents.worldmodel.loop import phase_learning_rate, rep_beta

    sw = SeparationSwitch(max_steps=1000, threshold_bits=3.0, consecutive=2, eval_every=5, min_steps=10, switch_step=0)
    for step in range(0, 400, 5):
        sw.update(step, 10.0)  # far above the threshold
        sw.check_deadline(step)
    assert sw.switch_step == 0 and sw.switched
    assert phase_learning_rate(0, 0, 1e-3, 1e-4) == 1e-4 and rep_beta(0, 0, 0, 0.1) == 0.1


def test_learning_rate_zero_leaves_the_measured_weights(start_file, tmp_path):
    cfg = mini_cfg()
    cfg["train"]["lr"] = cfg["train"]["lr_after_switch"] = 0.0
    cfg["agent"]["goal_lr"] = cfg["agent"]["critic_lr"] = 0.0
    cfg["e8"]["max_env_steps"], cfg["e8"]["checkpoints"] = 100, [0, 100]
    therapy(start_file, tmp_path / "lr0", cfg=cfg)
    start, end = weights(start_file), weights(tmp_path / "lr0" / "checkpoints" / "agent_env000100.pt")
    for key in WEIGHT_KEYS:  # what the frozen measures read (the slow critic is the critic's EMA target: it moves)
        assert same_state(start[key], end[key]), key


def test_baseline_world_is_e5s_reference_environment():
    cfg = mini_cfg()
    conf = conf_of(cfg)
    e8_world = e8_therapy.make_hooks(cfg, conf, "full", "nw", SEED).build_world(cfg, SEED, "experimental")
    e5_conf = run_e5.e5_setup.configuration(run_e5.e5_setup.load_config(REPO_ROOT / "configs" / "e5" / "e5_mini.yaml"), "p1-b-d1", "A")
    e5_world = run_e5.baseline_hooks(cfg, e5_conf).build_world(cfg, SEED, "reference")
    assert np.array_equal(e8_world.templates_np, e5_world.templates_np)
    assert np.array_equal(e8_world.env.transition_tensor(), e5_world.env.transition_tensor())
    assert np.array_equal(e8_world.background_t.numpy(), e5_world.background_t.numpy())


def _worlds(cfg, conf):
    return [assignment_world(k, conf, "reference", cfg["render"], cfg["episode"], SEED) for k in range(len(ASSIGNMENTS))]


def test_treated_assignments_draws():
    cfg = mini_cfg()
    conf = conf_of(cfg)
    worlds = _worlds(cfg, conf)
    canonical = canonical_assignment(conf["diagonal"])
    assert treated_assignments("nw") == (0, 1, 2) and treated_assignments("all") == tuple(range(12))
    src = TreatedAssignments(worlds, canonical, "nw", therapy_streams.assignment_generator(8008))
    draws = np.array([src.next_episode(i)[1] // two_zones.N_STATES for i in range(3000)])
    assert set(draws) == {0, 1, 2}
    counts = np.bincount(draws, minlength=3)[:3]
    assert np.all(np.abs(counts / 3000 - 1 / 3) < 0.03), counts  # uniform
    assert all(ASSIGNMENTS[k][0] == "nw" for k in draws)
    # the reference's dynamics, as E5's shuffle reference: every assignment world equals E5's reference assignment world
    e5_worlds = run_e5.assignment_worlds(worlds[canonical], conf, "reference", cfg, SEED)
    for k in range(12):
        e5_w = e5_worlds[k]
        assert np.array_equal(worlds[k].env.transition_tensor(), e5_w.env.transition_tensor())
        assert np.array_equal(worlds[k].templates_np, e5_w.templates_np)
    # "all" draws as E5's f = 1 shuffle on the same generator
    a = TreatedAssignments(worlds, canonical, "all", np.random.default_rng(5))
    b = ZoneAssignmentShuffle(worlds, canonical, 1.0, np.random.default_rng(5))
    assert [a.next_episode(i)[1] for i in range(500)] == [b.next_episode(i)[1] for i in range(500)]


def test_shuffle_therapy_run(start_file, tmp_path):
    records = therapy(start_file, tmp_path / "sh", therapy_name="nw")
    log = np.load(tmp_path / "sh" / "steps.npz")
    assert set(np.unique(log["assignment"])) <= {0, 1, 2}  # only the treated assignments are shown
    ud = np.array([two_zones.region_states(two_zones.build_layout(1, "reference"))["nw"]])
    assert np.array_equal(log["treated"], np.isin(log["state"], ud))
    assert records[-1]["e8"]["visits"]["treated"] == int(log["treated"].sum())
    assert sum(records[-1]["e8"]["episodes_by_assignment"][3:]) == 0


def test_resume_equals_uninterrupted(start_file, baseline_run, tmp_path):
    from helplessness.pipeline.agent_run import Interrupted

    out = tmp_path / "interrupted"
    with pytest.raises(Interrupted):
        therapy(start_file, out, interrupt_after=60)
    cfg = e8_setup.load_config(out / "config.yaml")
    conf = conf_of(cfg)
    hooks = e8_therapy.make_hooks(cfg, conf, "full", "nw", SEED)
    run_one(cfg, SEED, "experimental", out, configure_runtime(cfg), resume=True, hooks=hooks, total_env_steps=int(cfg["e8"]["max_env_steps"]),
            save_resumable_at=tuple(cfg["checkpoints"]["grid"]), fresh_start=FreshStart(weights=start_file, stream_root=therapy_streams.therapy_root(SEED)))
    ref_dir, _ = baseline_run
    for step in (0, 50, 100, 200):
        a, b = weights(out / "checkpoints" / f"agent_env{step:06d}.pt"), weights(ref_dir / "checkpoints" / f"agent_env{step:06d}.pt")
        for key in (*WEIGHT_KEYS, "slow_critic"):
            assert same_state(a[key], b[key]), (step, key)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("e8",)} | {"visits": r["e8"]["visits"], "health": r["e8"]["health"]}  # noqa: E731
    ra = [strip(json.loads(x)) for x in (out / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    rb = [strip(json.loads(x)) for x in (ref_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ra == rb


def test_walks_by_assignment_options_default(start_file):
    """slots_of measures a subset with the same values; step1 only adds a key; the default output is unchanged."""
    from helplessness.pipeline.assignment_records import walks_by_assignment
    from helplessness.probing.frozen_episodes import load_frozen_agent

    cfg = mini_cfg()
    conf = conf_of(cfg)
    model = load_frozen_agent(start_file, torch.device("cpu"))
    pw = {k: two_zones_setup.probe_world(cfg, conf, run_e5.assignment_colors(k, conf["diagonal"], conf["t_inc"], conf["t_ctrl"]))
          for k in (0, 1)}
    canonical = canonical_assignment(conf["diagonal"])
    full = walks_by_assignment(model.rssm, cfg, conf, pw, canonical, 3, "checkpoint")
    part = walks_by_assignment(model.rssm, cfg, conf, pw, canonical, 3, "checkpoint", slots_of={0: ("nw",), 1: ("nw",)}, step1=True)
    for k in (0, 1):
        got = dict(part[k]["slots"]["nw"])
        assert set(got.pop("step1")) == {"mass_step1", "ghost_step1", "I_step1", "n_states_step1"}
        assert got == full[k]["slots"]["nw"]
        assert part[k]["I_step1_by_type"]["ud"] == full[k]["I_step1_by_type"]["ud"]


def pickle_loads(blob):
    import pickle

    return pickle.loads(blob)
