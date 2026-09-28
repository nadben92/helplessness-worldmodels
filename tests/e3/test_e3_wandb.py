"""Optional W&B tracking (experiments/E3_worldmodel_4zones/wandb_logger.py,
docs/e3/DECISIONS.md 2026-09-28): it never changes the training trajectory
(bit-identical losses with and without it, offline mode), never stops a run
(a failing wandb only warns), is off by default without WANDB_API_KEY, and
its artifact holds the run directory without resume.pt."""

from __future__ import annotations

import copy
import sys
import types

import pytest
import torch
import yaml

from _experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_wandb_run_mini")
wandb_logger = load_experiment_module("E3_worldmodel_4zones", "wandb_logger.py", "test_wandb_logger")


def tiny_config() -> dict:
    cfg = copy.deepcopy(run_mini.load_mini_config())
    cfg["render"]["k"] = 2
    cfg["model"].update(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)
    cfg["budget"].update(total_steps=88, prefill_steps=8)
    cfg["train"].update(batch_size=4, seq_len=8, replay_ratio=1)
    cfg["train"]["latent_estimator"].update(switch_js_threshold=-1.0, switch_consecutive=2, switch_eval_every=5,
                                            switch_min_steps=10, switch_max_steps=60, collapse_js_threshold=float("-inf"))
    cfg["measure"].update(iz_mc_samples_training=50, iz_mc_samples_final=50, kernel_k_s=2, separation_every=40,
                          diagnostics_every=None, monitor_every=20, z_only_seq_len=16)
    cfg["measure"]["real_context"].update(window_len=16, burn_in=8, max_per_state=2, kernel_draws=2)
    cfg["checkpoints"] = {"grid": [40], "resume_every": None}
    cfg["runtime"].update(num_threads=2, device="cpu")
    return cfg


def _run(tmp_path, name, **kwargs):
    cfg = tiny_config()
    device = run_mini.configure_runtime(cfg)
    run_dir = tmp_path / name
    run_dir.mkdir()
    return run_mini.run_one_seed(cfg, 0, run_dir, device, **kwargs), run_dir


def test_off_by_default_without_a_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert not wandb_logger.wandb_requested(tiny_config())
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert wandb_logger.wandb_requested(tiny_config())
    assert wandb_logger.wandb_requested({"logging": {"wandb": {"enabled": False}}}, force=True)


def test_losses_bit_identical_with_and_without_wandb_offline(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    without, _ = _run(tmp_path, "without")
    with_wandb, run_dir = _run(tmp_path, "with", wandb=True)
    for key in (*run_mini.LOSS_TERMS, "grad_norm", "lr", "beta_rep", "step"):
        assert with_wandb["history"][key] == without["history"][key], key
    assert with_wandb["training_procedure"]["switch_step"] == without["training_procedure"]["switch_step"]
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and meta.get("wandb_run_id")
    assert list((run_dir / "wandb").glob("offline-run-*")), "no offline W&B run written"


def _fake_wandb(fail_at: str | None, added: list):
    class Run:
        id, url, summary = "fake", None, {}

        def define_metric(self, *a, **k):
            pass

        def log(self, data):
            if fail_at == "log":
                raise ConnectionError("network down")

        def save(self, *a, **k):
            pass

        def log_artifact(self, art):
            pass

        def finish(self):
            pass

    class Artifact:
        def __init__(self, **k):
            pass

        def add_file(self, path, name):
            added.append(name)

    def init(**kwargs):
        if fail_at == "init":
            raise ConnectionError("cannot reach api.wandb.ai")
        return Run()

    return types.SimpleNamespace(init=init, Artifact=Artifact, Image=lambda p: p)


@pytest.mark.parametrize("fail_at", ["init", "log"])
def test_a_failing_wandb_only_warns(tmp_path, monkeypatch, fail_at):
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(fail_at, []))
    report, run_dir = _run(tmp_path, f"fail_{fail_at}", wandb=True)
    assert yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))["status"] == "done"
    console = (run_dir / "console.log").read_text(encoding="utf-8")
    assert "[wandb] WARNING" in console and "the run itself continues normally" in console
    assert report["training_procedure"]["n_grad_steps"] == 96


def test_artifact_holds_the_run_directory_without_resume_pt(tmp_path, monkeypatch):
    added: list[str] = []
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(None, added))
    _run(tmp_path, "artifact", wandb=True, wandb_artifact=True)
    assert "checkpoints/weights_final.pt" in added and "checkpoints/weights_step00000040.pt" in added
    assert "checkpoints/resume.pt" not in added
    for name in ("config.yaml", "meta.yaml", "replay.npz", "world.json", "results.json", "arrays.npz", "REPORT.md"):
        assert name in added, name
    assert not any(name.startswith("wandb/") for name in added)
    assert torch.get_num_threads() >= 1
