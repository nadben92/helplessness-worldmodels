"""Phase 5, closed loop (docs/e3/DECISIONS.md, 2026-09-30 (5)), on a TINY
configuration: the collector's replay convention (the goal's observation is
stored, with the reward of reaching it), the counts of the tabular twin, the
runner's outputs, and determinism. The switch is forced early so the heads
train too."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from _experiment_loader import load_experiment_module
from helplessness.agents.worldmodel.closed_loop import ClosedLoopCollector, stored_steps_bound
from helplessness.agents.worldmodel.heads import Critic, GoalHead
from helplessness.agents.worldmodel.replay import ModelBatch, ReplayBuffer

run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_closed_loop_run_agent")
run_mini = run_agent.run_mini
CONFIGS = run_mini.REPO_ROOT / "configs" / "e3"


def tiny_config() -> dict:
    cfg = copy.deepcopy(run_mini.load_mini_config(CONFIGS / "agent_mini.yaml"))
    cfg["render"]["k"] = 2
    cfg["model"].update(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)
    cfg["budget"].update(total_steps=260, prefill_steps=40)
    cfg["train"].update(batch_size=4, seq_len=8, replay_ratio=1)
    cfg["train"]["latent_estimator"].update(switch_js_threshold=-1.0, switch_consecutive=2, switch_eval_every=5,
                                            switch_min_steps=10, switch_max_steps=60, collapse_js_threshold=float("-inf"))
    cfg["agent"].update(k_act=4, k_imag=2, horizon=3, hidden_dim=16)
    cfg["checkpoints"] = {"grid": [100, 200], "resume_every": 50}
    cfg["measure"].update(iz_mc_samples_training=50, iz_mc_samples_final=100, kernel_k_s=2, kernel_k_s_training=2,
                          context_dispersion_n_samples=10, z_only_seq_len=16, final=False)
    cfg["measure"]["real_context"].update(window_len=16, burn_in=8, max_per_state=3, kernel_draws=4, kernel_draws_training=2)
    cfg["measure"]["is_per_context"] = {"states_per_zone": 1, "contexts": 2, "draws": 8}
    cfg["log"].update(monitor_every=100, behaviour_window=20)
    cfg["runtime"].update(num_threads=2, device="cpu")
    return cfg


def test_the_bench_config_is_mini_long_s_network_and_procedure():
    bench, ref = run_mini.load_mini_config(CONFIGS / "agent_mini.yaml"), run_mini.load_mini_config(CONFIGS / "mini_long.yaml")
    for block in ("model", "render", "train"):
        assert bench[block] == ref[block], block
    assert bench["budget"] == {"total_steps": ref["budget"]["total_steps"], "prefill_steps": ref["budget"]["prefill_steps"]}
    acfg = run_agent.agent_config(bench)
    episode = run_mini.load_episode_config()
    assert (acfg.gamma, acfg.goal_reward, acfg.step_reward) == (episode["gamma"], episode["goal_reward"], episode["step_reward"])
    assert (acfg.epsilon, acfg.k_act, acfg.k_imag, acfg.horizon, acfg.lam, acfg.slow_tau) == (0.1, 16, 4, 15, 0.95, 0.02)  # the spec's
    assert bench["agent"]["critic_lr"] == 3e-5 and bench["agent"]["critic_clip_norm"] == 100.0
    # the E3 measures: mini_long's estimators and settings, taken at the end of the run; resume points every 2,000 steps
    assert bench["measure"]["final"] is True
    assert {k: v for k, v in bench["measure"].items() if k != "final"} == ref["measure"]
    assert bench["checkpoints"] == {"grid": "e2", "resume_every": 2000}


def _collector(cfg, seed=0, n_steps=400):
    world = run_mini.build_world(cfg, seed, "experimental")
    acfg = run_agent.agent_config(cfg)
    torch.manual_seed(0)
    rssm = run_mini.build_rssm(cfg)
    goal, critic = GoalHead(16, 12, 16), Critic(16, 12, 16)
    replay = ReplayBuffer.for_run(budget=stored_steps_bound(n_steps), prefill_steps=0, table=world.templates_np)
    streams = run_agent.e3_rng.spawn_streams(seed)
    collector = ClosedLoopCollector(world.env, replay, rssm, goal, critic, acfg, 144, streams["start_state"], streams["explore"],
                                    streams["tiebreak"], torch.Generator().manual_seed(0), torch.device("cpu"))
    transitions = [collector.step() for _ in range(n_steps)]
    return world, collector, replay, transitions


def test_replay_holds_observations_with_their_arrival_reward_and_the_goal_itself():
    cfg = tiny_config()
    world, collector, replay, transitions = _collector(cfg)
    stats = collector.stats
    arr = replay.export_compact()
    goal, episode = run_mini.e3_env.goal_state(), run_mini.load_episode_config()
    assert stats.env_steps == 400 and collector.N.sum() == 400  # one count per real transition
    assert len(replay) == 400 + stats.episodes <= stored_steps_bound(400)  # one more observation per finished episode
    assert stats.episodes >= 400 // episode["max_steps"] and stats.successes >= 1
    # an episode's first observation: no arrival reward; the goal is stored, terminal, with the goal reward
    assert np.all(arr["r"][arr["is_first"]] == 0.0)
    terminal = arr["terminal"]
    assert terminal.sum() == stats.successes and np.all(arr["s"][terminal] == goal) and np.all(arr["r"][terminal] == episode["goal_reward"])
    assert np.all(arr["s"][~terminal] != goal)  # the goal is never anything but a terminal observation
    assert np.all((arr["r"] > 0) == terminal)   # the goal head's target 1[r > 0] marks exactly the goal's observations
    assert arr["truncated"].sum() == stats.episodes - stats.successes and not np.any(terminal & arr["truncated"])
    # the step after an episode's last observation starts a new episode
    last = np.flatnonzero(terminal | arr["truncated"])
    assert np.all(arr["is_first"][last[last + 1 < len(replay)] + 1])
    # the counts are the environment's own transitions
    N = np.zeros_like(collector.N)
    for t in transitions:
        N[t["s"], t["a"], t["s_next"]] += 1
    assert np.array_equal(N, collector.N)
    assert sum(t["terminated"] for t in transitions) == stats.successes
    # the model batch carries no label
    batch, labels = replay.sample(2, 8, np.random.default_rng(0))
    assert isinstance(batch, ModelBatch) and not hasattr(batch, "s") and labels.s.shape == (2, 8)


def test_policy_streams_are_separate_and_the_collection_is_deterministic():
    cfg = tiny_config()
    _, c1, r1, t1 = _collector(cfg, n_steps=150)
    _, c2, r2, t2 = _collector(cfg, n_steps=150)
    assert [t["a"] for t in t1] == [t["a"] for t in t2] and np.array_equal(c1.N, c2.N)
    assert all(np.array_equal(v, r2.export_compact()[k]) for k, v in r1.export_compact().items())
    explored = np.mean([t["explored"] for t in t1])
    assert 0.02 < explored < 0.25  # epsilon = 0.1


def test_closed_loop_run_outputs_and_twin(tmp_path):
    cfg = tiny_config()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = run_agent.run_one(cfg, 0, "experimental", run_dir, run_mini.configure_runtime(cfg))
    assert report["env_steps"] == 300 and report["grad_steps"] == 260 and report["switch_step"] is not None
    assert [c["env_step"] for c in report["checkpoints"]] == [100, 200, 300]
    for name in ("config.yaml", "meta.yaml", "console.log", "metrics.csv", "checkpoints.jsonl", "results.json", "counts.npz",
                 "checkpoints/agent_final.pt"):
        assert (run_dir / name).is_file(), name
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["source_sha256"] and meta["agent"]["gamma"] == 0.9
    last = report["checkpoints"][-1]
    assert set(last["twin_by_zone"]) == set(run_mini.e3_env.ZONE_CATEGORIES)
    for z in last["twin_by_zone"].values():
        assert set(z) >= {"I_hat", "I_ref", "b", "delta", "l1", "visit_share"} and z["delta"] >= -1e-9
    assert sum(z["visit_share"] for z in last["twin_by_zone"].values()) <= 1.0 + 1e-9  # the corridor's states are in no zone
    assert {"goal_loss", "critic_loss", "imagined_tie_fraction", "recon"} <= set(last["train"])  # the heads trained after the switch
    counts = np.load(run_dir / "counts.npz")
    assert counts["N_300"].sum() == 300 and counts["N_100"].sum() == 100
    state = torch.load(run_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    assert {"rssm", "goal_head", "critic", "slow_critic"} <= set(state)
    # the weights at every checkpoint (2026-10-01 (12)); the last one is the final state
    saved = [torch.load(run_dir / "checkpoints" / f"agent_env{e:06d}.pt", weights_only=False) for e in (100, 200, 300)]
    assert [w["env_step"] for w in saved] == [100, 200, 300]
    assert all(torch.equal(saved[-1]["rssm"][k], state["rssm"][k]) for k in state["rssm"])
    assert not all(torch.equal(saved[0]["rssm"][k], state["rssm"][k]) for k in state["rssm"])
    rows = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 3 * 4
    assert [json.loads(line)["env_step"] for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()] == [100, 200, 300]
    # the world-model agent's own Delta under P (spec 5.7): real contexts (main) and canonical probe, >= 0 by construction
    for reading in ("real_contexts", "canonical"):
        d = last["agent_delta"][reading]
        assert set(d["delta_by_zone"]) == set(run_mini.e3_env.ZONE_CATEGORIES)
        assert all(v >= -1e-9 for v in d["delta_by_zone"].values()) and d["delta_mean"] >= -1e-9
    assert last["agent_delta"]["canonical"]["n_states_read"] == run_mini.N_STATES
    assert 0 < last["agent_delta"]["real_contexts"]["n_states_read"] <= run_mini.N_STATES
    assert rows[0].split(",")[-2:] == ["agent_delta_real", "agent_delta_canonical"]
    assert not (run_dir / "arrays.npz").exists()  # measure.final false here


def _run(cfg, tmp_path, name, condition="experimental", **kwargs):
    run_dir = tmp_path / name
    run_dir.mkdir()
    report = run_agent.run_one(cfg, 0, condition, run_dir, run_mini.configure_runtime(cfg), **kwargs)
    return run_dir, report


def test_resume_gives_the_same_run_as_an_uninterrupted_one(tmp_path):
    """docs/e3/DECISIONS.md, 2026-09-30 (8): an interruption after a resume point, then --resume in the same
    directory, gives the same checkpoints, counts and weights as the run that was never interrupted (CPU)."""
    cfg = tiny_config()
    full_dir, _ = _run(cfg, tmp_path, "full")
    cut_dir = tmp_path / "cut"
    cut_dir.mkdir()
    with pytest.raises(run_agent.Interrupted):
        run_agent.run_one(cfg, 0, "experimental", cut_dir, run_mini.configure_runtime(cfg), interrupt_after=150)
    assert yaml.safe_load((cut_dir / "meta.yaml").read_text(encoding="utf-8"))["status"] == "running"
    assert [json.loads(line)["env_step"] for line in (cut_dir / "checkpoints.jsonl").read_text().splitlines()] == [100]
    assert run_agent.main(["--resume", str(cut_dir), "--device", "cpu"]) == 0
    resumed = json.loads((cut_dir / "results.json").read_text(encoding="utf-8"))
    reference = json.loads((full_dir / "results.json").read_text(encoding="utf-8"))
    assert resumed["checkpoints"] == reference["checkpoints"]
    assert (cut_dir / "checkpoints.jsonl").read_text() == (full_dir / "checkpoints.jsonl").read_text()
    assert (cut_dir / "metrics.csv").read_text() == (full_dir / "metrics.csv").read_text()
    for key in ("env_steps", "grad_steps", "switch_step"):
        assert resumed[key] == reference[key], key
    a, b = np.load(full_dir / "counts.npz"), np.load(cut_dir / "counts.npz")
    assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    wa = torch.load(full_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    wb = torch.load(cut_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    for part in ("rssm", "goal_head", "critic", "slow_critic"):
        assert all(torch.equal(wa[part][k], wb[part][k]) for k in wa[part]), part
    meta = yaml.safe_load((cut_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["resumed_at"]


def test_final_measures_in_closed_loop_pair_in_aggregate_b(tmp_path):
    """measure.final: run_mini's final measures on the closed-loop replay, written as run_mini writes them, so
    aggregate_b.py pairs the environment run with its reference (b on I_s and I_s projected, bias control)."""
    cfg = tiny_config()
    cfg["measure"]["final"] = True
    cfg["checkpoints"] = {"grid": [], "resume_every": None}
    dirs = {}
    for cond in ("experimental", "reference"):  # named as main() names them, so aggregate_b reads the config name
        dirs[cond] = run_mini.make_seed_run_dir(tmp_path, "agent-tiny", 0, cond)
        run_agent.run_one(cfg, 0, cond, dirs[cond], run_mini.configure_runtime(cfg), max_env_steps=200)
    for run_dir in dirs.values():
        arrays = np.load(run_dir / "arrays.npz")
        assert {"real_I_s", "real_I_s_projected", "real_off_support", "real_p_hat", "N"} <= set(arrays.files)
        assert arrays["N"].sum() == 200
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        assert {"I_s_by_category_real", "I_s_projected_by_category_real", "z_only_probe", "h_only_probe"} <= set(results)
        assert "diagnostics_final" not in results and results["truncated_by_max_env_steps"] is True
    aggregate_b = load_experiment_module("E3_worldmodel_4zones", "aggregate_b.py", "test_closed_loop_aggregate_b")
    agg = aggregate_b.aggregate([aggregate_b.load_run(d) for d in dirs.values()])
    assert len(agg["pairs"]) == 1 and not agg["unpaired"]
    control = agg["pairs"][0]["bias_control"]
    assert set(control) == set(run_mini.e3_env.ZONE_CATEGORIES)
    for cat, v in control.items():  # both versions of b present (2026-10-02 (16): reported as measured, no verdict)
        assert np.isfinite(v["b_I_s"]) and np.isfinite(v["b_I_s_projected"]) and "verdict" not in v, cat
    flagged = aggregate_b.aggregate([aggregate_b.load_run(d) for d in dirs.values()], aggregate_b.load_e30_control())
    assert all("aberrations" in v for v in flagged["pairs"][0]["bias_control"].values())


def test_a_run_without_separation_stops_on_the_guard(tmp_path):
    cfg = tiny_config()
    cfg["train"]["latent_estimator"].update(switch_js_threshold=1e9, switch_max_steps=30)  # can never switch
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(run_agent.SeparationNotReachedError):
        run_agent.run_one(cfg, 0, "experimental", run_dir, run_mini.configure_runtime(cfg))
    assert yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))["status"] == "failed"


def test_checkpoint_goal_diagnostics(tmp_path):
    """docs/e3/DECISIONS.md, 2026-10-02 (19): with measure.checkpoint_goal_kernel, every checkpoint records the goal
    arrivals in the replay, the prior's mass on the goal for the four pairs into it and the median mass on the right
    successor of the pairs visited 2-8 times; without it (agent_mini.yaml), nothing changes."""
    cfg = tiny_config()
    cfg["measure"]["checkpoint_goal_kernel"] = {"draws": 8}
    cfg["checkpoints"] = {"grid": [150], "resume_every": None}
    run_dir, report = _run(cfg, tmp_path, "goal", max_env_steps=200)
    for rec in report["checkpoints"]:
        g = rec["goal_diagnostics"]
        assert g["goal_arrivals_in_replay"] == rec["successes"]  # every success stores the goal's observation
        assert len(g["goal_pairs"]) == 4 and all(p["s"] != run_mini.e3_env.goal_state() for p in g["goal_pairs"])
        assert all(p["mass_on_goal"] is None or 0.0 <= p["mass_on_goal"] <= 1.0 for p in g["goal_pairs"])
        assert g["draws"] == 8 and (g["pairs_n2_8"] == 0 or 0.0 <= g["median_mass_right_successor_n2_8"] <= 1.0)
    plain = tiny_config()
    plain["checkpoints"] = {"grid": [150], "resume_every": None}
    _, rep2 = _run(plain, tmp_path, "plain", max_env_steps=200)
    assert all("goal_diagnostics" not in r for r in rep2["checkpoints"])
    assert [r["twin_by_zone"] for r in rep2["checkpoints"]] == [r["twin_by_zone"] for r in report["checkpoints"]]  # own stream
