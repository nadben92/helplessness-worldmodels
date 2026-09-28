"""The real-context main measure, checkpoints, resume and recomputation
(docs/e3/DECISIONS.md, 2026-09-28), on a TINY configuration:

- select_real_contexts: burn-in, is_first exclusion, cap per state, draws
  from the given stream only;
- the one-step KL of a context is the training loss's KL at the next step;
- a run interrupted after a resume point and continued with resume=True
  gives bit-identical losses and final measures to an uninterrupted run
  (CPU, same threads);
- recompute_measures.py reproduces the run's final measures from the run
  directory alone, and recomputes every saved checkpoint;
- the reference condition: same permutation and themes, controllable
  counterparts of the uncontrollable zones, independent streams.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from _experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_real_context_run_mini")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "test_real_context_probe")
recompute = load_experiment_module("E3_worldmodel_4zones", "recompute_measures.py", "test_real_context_recompute")


def tiny_config() -> dict:
    cfg = copy.deepcopy(run_mini.load_mini_config())
    cfg["render"]["k"] = 2
    cfg["model"].update(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)
    cfg["budget"].update(total_steps=152, prefill_steps=8)
    cfg["train"].update(batch_size=4, seq_len=8, replay_ratio=1)
    cfg["train"]["latent_estimator"].update(switch_js_threshold=-1.0, switch_consecutive=2, switch_eval_every=5,
                                            switch_min_steps=10, switch_max_steps=60, collapse_js_threshold=float("-inf"))
    cfg["measure"].update(iz_mc_samples_training=100, iz_mc_samples_final=200, kernel_k_s=2, separation_every=25,
                          diagnostics_every=None, monitor_every=50, z_only_seq_len=16)
    cfg["measure"]["real_context"].update(window_len=16, burn_in=8, max_per_state=3, kernel_draws=4)
    cfg["checkpoints"] = {"grid": [40, 100], "resume_every": 30}
    cfg["runtime"].update(num_threads=2, device="cpu")
    return cfg


def _run(tmp_path, name, cfg, **kwargs):
    device = run_mini.configure_runtime(cfg)
    run_dir = tmp_path / name
    run_dir.mkdir()
    return run_mini.run_one_seed(cfg, seed=0, run_dir=run_dir, device=device, **kwargs), run_dir


def test_select_real_contexts_rules():
    s_all = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # 13 steps, window_len 4 -> 3 windows (12 + 1 extra step)
    is_first = np.zeros(13, dtype=bool)
    is_first[[3, 6]] = True
    sel = e3_probe.select_real_contexts(s_all, is_first, window_len=4, burn_in=2, max_per_state=10, n_states=3,
                                        rng=np.random.default_rng(0))
    # eligible: t in {2, 3} of each window -> positions 2, 3, 6, 7, 10, 11, minus is_first 3 and 6
    assert sorted(sel["positions"].tolist()) == [2, 7, 10, 11]
    assert np.array_equal(sel["states"], s_all[sel["positions"]]) and np.all(sel["t"] >= 2)
    assert sel["n_eligible"].tolist() == [0, 2, 2]  # states 2, 1, 1, 2
    capped = e3_probe.select_real_contexts(s_all, is_first, 4, 2, max_per_state=1, n_states=3, rng=np.random.default_rng(0))
    assert np.bincount(capped["states"], minlength=3).tolist() == [0, 1, 1]
    again = e3_probe.select_real_contexts(s_all, is_first, 4, 2, max_per_state=1, n_states=3, rng=np.random.default_rng(0))
    assert np.array_equal(capped["positions"], again["positions"])


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real_context")
    return _run(tmp, "full", tiny_config())


def test_one_step_kl_is_the_training_losses_pairing(trained):
    """kl_next of a context = KL(q_{t+1} || p_{t+1}) of the same window's
    unroll, from the same generator: recomputed here independently."""
    _, run_dir = trained
    cfg = tiny_config()
    device = torch.device("cpu")
    world, replay, _, _ = run_mini.load_run_data(run_dir, cfg, 0, "experimental")
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(torch.load(run_dir / "checkpoints" / "weights_final.pt", weights_only=False)["rssm"])
    ctx = e3_probe.real_context_probe(rssm, replay, 16, 8, 3, 144, np.random.default_rng(0), torch.Generator().manual_seed(0))
    assert ctx["positions"].size > 0 and np.all(ctx["t"] >= 8)
    windows = np.unique(ctx["window"])
    model_batch, _ = replay.windows(windows * 16, 17)
    from helplessness.agents.worldmodel.loop import to_chw_float, unroll_observation

    with torch.no_grad():
        out = unroll_observation(rssm, to_chw_float(model_batch.x, device), torch.as_tensor(model_batch.a),
                                 torch.as_tensor(model_batch.is_first), generator=torch.Generator().manual_seed(0), decode=False)
    for i in range(ctx["positions"].size):
        w, t = np.searchsorted(windows, ctx["window"][i]), ctx["t"][i]
        expected = float(run_mini.categorical_kl(out["post_probs"][w, t + 1], out["prior_probs"][w, t + 1])) / np.log(2.0)
        if model_batch.is_first[w, t + 1]:
            assert np.isnan(ctx["kl_next_bits"][i])
        else:
            assert ctx["kl_next_bits"][i] == pytest.approx(expected, rel=1e-5, abs=1e-6)
    assert np.all(ctx["prior_probs"].sum(axis=-1) == pytest.approx(1.0))


def test_run_outputs_checkpoints_and_real_context_record(trained):
    report, run_dir = trained
    for name in ("world.json", "replay.npz", "checkpoints/weights_step00000040.pt", "checkpoints/weights_step00000100.pt",
                 "checkpoints/weights_final.pt", "checkpoints/resume.pt", "checkpoints/latest.json"):
        assert (run_dir / name).is_file(), name
    replay = np.load(run_dir / "replay.npz")
    assert replay["s"].dtype == np.int16 and len(replay["s"]) == 160 and replay["N"].sum() == 160
    # N rebuilt from the compact replay's own (s, a, s_next): the counts of the twin are recoverable
    N = np.zeros((144, 5, 144), dtype=np.int64)
    np.add.at(N, (replay["s"].astype(int), replay["a"].astype(int), replay["s_next"].astype(int)), 1)
    assert np.array_equal(N, replay["N"])
    rc = report["real_context"]
    assert rc["n_contexts_total"] == int(report["n_contexts"].sum()) and rc["n_states_measured"] + len(rc["not_measured_states"]) == 144
    assert np.all(np.isnan(report["I_z_real"][report["n_contexts"] == 0]))
    steps = [json.loads(line)["step"] for line in (run_dir / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert steps == [40, 100, 160]  # at the checkpoints of the grid, plus the final point
    assert "Real contexts: coverage" in (run_dir / "REPORT.md").read_text(encoding="utf-8")


def test_resume_after_an_interruption_is_bit_identical(trained, tmp_path, monkeypatch):
    full, _ = trained
    original = run_mini.train_world_model_step
    calls = {"n": 0}

    def crashing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 75:  # after the resume point at step 60, before the checkpoint at 100
            raise RuntimeError("simulated preemption")
        return original(*args, **kwargs)

    monkeypatch.setattr(run_mini, "train_world_model_step", crashing)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        _run(tmp_path, "interrupted", tiny_config())
    monkeypatch.setattr(run_mini, "train_world_model_step", original)
    run_dir = tmp_path / "interrupted"
    assert json.loads((run_dir / "checkpoints" / "latest.json").read_text())["step"] == 60
    resumed = run_mini.run_one_seed(tiny_config(), 0, run_dir, torch.device("cpu"), resume=True)
    for key in (*run_mini.LOSS_TERMS, "grad_norm", "lr", "beta_rep", "step"):
        assert resumed["history"][key] == full["history"][key], key
    assert resumed["training_procedure"]["switch_step"] == full["training_procedure"]["switch_step"]
    for key in ("I_z_real", "I_z", "n_contexts"):
        assert np.array_equal(resumed[key], full[key], equal_nan=True), key
    assert [d["step"] for d in resumed["diagnostics"]] == [d["step"] for d in full["diagnostics"]]


def test_recompute_reproduces_the_final_measures_and_every_checkpoint(trained):
    _, run_dir = trained
    outs = recompute.recompute(run_dir, ["final", "all"], "cpu")
    same = json.loads((outs[0] / "comparison_with_run.json").read_text(encoding="utf-8"))
    assert same and all(same.values()), same
    assert (outs[0] / "REPORT.md").is_file() and (outs[0] / "SUMMARY.md").is_file()
    labels = sorted(o.name.split("__")[0] for o in outs)
    assert labels == ["final", "step00000040", "step00000100"]  # "final" requested twice, computed once


def test_reference_condition_shares_only_permutation_and_themes():
    cfg = tiny_config()
    exp = run_mini.build_world(cfg, 5, "experimental")
    ref = run_mini.build_world(cfg, 5, "reference")
    assert exp.themes == ref.themes and np.array_equal(exp.templates_np, ref.templates_np)
    assert exp.env_seed != ref.env_seed
    assert exp.zone_kinds == {z: z for z in run_mini.e3_env.ZONE_CATEGORIES}
    assert ref.zone_kinds == {"uncontrollable_deterministic": "controllable_deterministic", "controllable_noisy": "controllable_noisy",
                              "uncontrollable_noisy": "controllable_noisy", "controllable_deterministic": "controllable_deterministic"}
    s_exp = run_mini.seed_streams(5, torch.device("cpu"), "experimental")
    s_ref = run_mini.seed_streams(5, torch.device("cpu"), "reference")
    for name in ("collect", "start_state", "replay", "switch_pairs"):
        assert s_exp[name].integers(2**62) != s_ref[name].integers(2**62), name
    assert s_exp["network_init_seed"] != s_ref["network_init_seed"]
    assert run_mini.make_seed_run_dir.__defaults__ == ("experimental",)
    assert run_mini.PAIR_ROLE["reference"] == "control"
