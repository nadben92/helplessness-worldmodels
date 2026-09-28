"""run_mini.run_one_seed end to end on a TINY configuration (k=2, 3x4
latent, 128 gradient steps), through the real entry point:

- separate random streams (docs/e3/DECISIONS.md, 2026-09-25): the same
  seed measured often and rarely gives bit-identical training losses
  (CPU, same number of threads);
- the run directory: config.yaml, meta.yaml (environment, TF32 flags,
  status), diagnostics.jsonl appended per point, results.json,
  arrays.npz, REPORT.md + artifacts, the switch margin;
- --max-steps truncation and the CLI.

Structure only: 128 steps cannot separate 144 states, so the switch
threshold is scaled to a value that is always met (as in
test_e3_mini_pipeline.py) and the collapse guard is disabled.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch
import yaml

from _experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_run_outputs_run_mini")


def tiny_config(separation_every: int, diagnostics_every: int | None) -> dict:
    cfg = copy.deepcopy(run_mini.load_mini_config())
    cfg["render"]["k"] = 2
    cfg["model"].update(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)
    cfg["budget"].update(total_steps=120, prefill_steps=8)
    cfg["train"].update(batch_size=4, seq_len=8, replay_ratio=1)
    cfg["train"]["latent_estimator"].update(switch_js_threshold=-1.0, switch_consecutive=2, switch_eval_every=5,
                                            switch_min_steps=10, switch_max_steps=60, collapse_js_threshold=float("-inf"))
    cfg["measure"].update(iz_mc_samples_training=200, iz_mc_samples_final=300, kernel_k_s=2, context_dispersion_n_samples=10, separation_every=separation_every,
                          diagnostics_every=diagnostics_every, monitor_every=50)
    cfg["measure"]["real_context"].update(window_len=16, burn_in=8, max_per_state=5, kernel_draws=4)
    cfg["runtime"].update(num_threads=2, device="cpu")
    return cfg


def _run(tmp_path, name: str, cfg: dict, max_steps: int | None = None):
    device = run_mini.configure_runtime(cfg)
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "artifacts").mkdir()
    return run_mini.run_one_seed(cfg, seed=0, run_dir=run_dir, device=device, max_steps=max_steps), run_dir


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runs")
    often = _run(tmp, "often", tiny_config(separation_every=10, diagnostics_every=20))
    rarely = _run(tmp, "rarely", tiny_config(separation_every=50, diagnostics_every=None))
    return often, rarely


def test_measurement_frequency_never_changes_the_training_trajectory(two_runs):
    (often, _), (rarely, _) = two_runs
    assert len(often["diagnostics"]) > len(rarely["diagnostics"])  # the two runs really measured differently
    assert len(often["separation_curve"]) > len(rarely["separation_curve"])
    for key in (*run_mini.LOSS_TERMS, "grad_norm", "lr", "beta_rep"):
        assert often["history"][key] == rarely["history"][key], key  # bit-identical, not approximately equal
    assert often["training_procedure"]["switch_step"] == rarely["training_procedure"]["switch_step"] == 16


def test_run_directory_contents(two_runs):
    (report, run_dir), _ = two_runs
    for name in ("config.yaml", "meta.yaml", "console.log", "diagnostics.jsonl", "results.json", "arrays.npz", "REPORT.md"):
        assert (run_dir / name).is_file(), name
    for name in ("loss_curves.png", "separation_curves.png", "iz_by_zone.png", "training_diagnostics_seed00.png",
                 "collapse_grid_seed00.png"):
        assert (run_dir / "artifacts" / name).is_file(), name

    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta["device"] == "cpu" and meta["num_threads"] == 2
    assert meta["allow_tf32_matmul"] is False and meta["allow_tf32_cudnn"] is False
    for key in ("python_version", "torch_version", "cuda_version", "cudnn_version", "gpu_name", "git_commit", "git_dirty",
                "deterministic_algorithms", "cudnn_benchmark"):
        assert key in meta, key
    assert meta["switch_margin"] == 60 - 16

    lines = (run_dir / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["step"] for r in records] == [0, 20, 40, 60, 80, 100, 120, 128]  # every 20 steps, plus the final point
    for key in ("kl_by_zone", "l1_vs_p_by_zone", "l1_vs_b_by_zone", "l1_b_vs_p_by_zone", "I_z_by_zone", "frequent_outcome",
                "posterior_entropy_canonical", "acc_replay_hard", "acc_canonical_hard", "acc_canonical_mode"):
        assert key in records[-1], key

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["training_procedure"]["switch_margin"] == 44
    assert set(results["accuracies"]) == set(run_mini.ACC_LABELS)
    arrays = np.load(run_dir / "arrays.npz")
    assert arrays["N"].sum() == 128 and arrays["p_hat"].shape == (144, 5, 144)
    assert yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["measure"]["diagnostics_every"] == 20

    text = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "margin (deadline - switch step): 44 steps" in text
    for label in run_mini.ACC_LABELS.values():
        assert label in text


def test_d1_entries_follow_the_acceptance_rule(two_runs):
    """Real contexts (main D1): a zone with no measured state is "not measured"; an accepted pair covers
    the MEASURED states of the own zone. Canonical D1 (secondary): every state of the zone (25)."""
    (report, _), _ = two_runs
    measured = {cat: sum(1 for s in range(144) if report["zone_of_state"][s] == cat and report["n_contexts"][s] > 0)
                for cat in report["zone_categories"]}
    for d1, n_of in ((report["d1"], lambda cat: measured[cat]), (report["d1_canonical"], lambda cat: 25)):
        for d in d1:
            if d.get("not_measured"):
                assert not d["accepted"] and n_of(d["own_category"]) * n_of(d["target_category"]) == 0
                continue
            assert d["threshold"] == pytest.approx(max(0.25, 3 * d["combined_se"]))
            assert d["accepted"] == (abs(d["contrast"]) > d["threshold"])
            if d["accepted"]:
                for kind in ("local", "global"):
                    assert d[kind]["n_kept"] + d[kind]["n_anomalies"] == d[kind]["n_states"] == n_of(d["own_category"])
            else:
                assert "local" not in d and "global" not in d


def test_max_steps_truncates_and_is_recorded(tmp_path):
    report, run_dir = _run(tmp_path, "short", tiny_config(separation_every=50, diagnostics_every=None), max_steps=30)
    assert report["training_procedure"]["n_grad_steps"] == 30 and report["truncated_by_max_steps"]
    assert "Truncated by --max-steps" in (run_dir / "REPORT.md").read_text(encoding="utf-8")


def test_cli_parses_every_option(tmp_path):
    args = run_mini.parse_args(["--config", "configs/e3/mini.yaml", "--device", "cpu", "--out-dir", str(tmp_path),
                                "--seeds", "0", "5", "--max-steps", "100"])
    assert (args.device, args.seeds, args.max_steps, args.out_dir) == ("cpu", [0, 5], 100, tmp_path)
    with pytest.raises(SystemExit):
        run_mini.parse_args(["--device", "tpu"])


def test_switch_deadline_must_be_absolute():
    cfg = tiny_config(separation_every=50, diagnostics_every=None)
    cfg["train"]["latent_estimator"]["switch_max_steps"] = None
    with pytest.raises(ValueError, match="absolute"):
        run_mini.build_separation_switch(cfg)


def test_shape_verdicts_rules():
    def record(step, kl, l1, iz_conv, iz_ctrl, se=0.01):
        cats = run_mini.e3_env.ZONE_CATEGORIES
        return {"step": step, "kl_by_zone": {z: kl for z in cats}, "l1_vs_p_by_zone": {z: l1 for z in cats},
                "l1_b_vs_p_by_zone": {z: 0.1 for z in cats},
                "I_z_by_zone": {z: {"mean": {"uncontrollable_deterministic": iz_conv, "controllable_deterministic": iz_ctrl}.get(z, 0.5),
                                    "mc_se": se} for z in cats}}

    rows = run_mini.shape_verdicts([record(0, 20.0, 2.0, 1.0, 1.0), record(2000, 5.0, 1.5, 0.9, 1.1),
                                    record(4000, 0.5, 1.0, 0.4, 1.6)], switch_step=1500)
    verdicts = {r["check"].split(" (")[0].split(" =")[0]: r["verdict"] for r in rows}
    assert all(r["first_step"] == 2000 and r["last_step"] == 4000 for r in rows)
    assert {r["verdict"] for r in rows} == {"conforme"}, verdicts
    noisy = run_mini.shape_verdicts([record(2000, 5.0, 1.5, 0.9, 1.1, se=0.5), record(4000, 2.0, 1.6, 0.8, 1.2, se=0.5)], 1500)
    by_check = {r["check"]: r["verdict"] for r in noisy}
    assert by_check["KL decreasing to < 1 bit (controllable_deterministic)"] == "non conforme"  # 2 bits is not < 1
    assert by_check["I_z decreasing (uncontrollable_deterministic)"] == "non concluant"  # within 3 combined se
    assert run_mini.shape_verdicts([record(2000, 5.0, 1.5, 0.9, 1.1)], 1500)[0]["verdict"].startswith("non évaluable")
    assert torch.get_num_threads() >= 1


def test_episode_starts_come_from_the_dedicated_stream_whatever_the_measurement_frequency(two_runs):
    """Uniform starts (docs/e3/DECISIONS.md, 2026-09-25): the sequence of
    episode starts is the same measured often or rarely, and it is exactly
    the sequence of uniform draws of E2's start_state stream for the seed
    over the non-terminal states -- no other stream touches it."""
    (_, dir_often), (_, dir_rarely) = two_runs
    starts_often = np.load(dir_often / "arrays.npz")["episode_starts"]
    starts_rarely = np.load(dir_rarely / "arrays.npz")["episode_starts"]
    assert len(starts_often) > 1 and np.array_equal(starts_often, starts_rarely)

    e2_rng = load_experiment_module("E2_online_4zones", "rng.py", "test_run_outputs_e2_rng")
    stream = e2_rng.spawn_streams(0)["start_state"]
    support = np.array([s for s in range(144) if s != run_mini.e3_env.goal_state()])
    expected = [int(support[stream.integers(len(support))]) for _ in range(len(starts_often))]
    assert starts_often.tolist() == expected

    meta = yaml.safe_load((dir_often / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["start_distribution"] == "uniform_non_terminal" and meta["collection_epsilon"] == 1.0
    # spec 10: the permutation, the themes and the block class are in meta.yaml
    assert meta["rotation"] == run_mini.e3_env.rotation_for_seed_index(0) and meta["block_class"] == "A"
    assert meta["quadrant_assignment"] == {str(k): str(v) for k, v in
                                           run_mini.e3_env.quadrant_assignment_for_rotation(meta["rotation"]).items()}
    themes = run_mini.build_run(tiny_config(10, None), 0)[1]
    assert meta["themes"]["zone_colors"] == {c: list(rgb) for c, rgb in themes.zone_colors.items()}
    assert meta["themes"]["agent_marker_color"] == list(themes.agent_marker_color)
    assert meta["episode"] == run_mini.load_episode_config()
