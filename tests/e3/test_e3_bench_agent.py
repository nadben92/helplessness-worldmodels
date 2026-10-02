"""bench_agent.py (docs/e3/DECISIONS.md, 2026-10-02 (19)) on a tiny configuration, CPU: every variant is timed or
reported (compile skipped off Linux / CUDA), the stacked variants' losses match the plain step, and the loop mode
reports environment steps per second. Informative only: nothing here enters a verdict."""

from __future__ import annotations

import copy

import torch

from _experiment_loader import load_experiment_module

bench = load_experiment_module("E3_worldmodel_4zones", "bench_agent.py", "test_bench_agent")
run_mini = bench.run_mini


def _tiny():
    cfg = copy.deepcopy(run_mini.load_mini_config(run_mini.REPO_ROOT / "configs" / "e3" / "gpu_pilot_closed_loop.yaml"))
    cfg["render"]["k"] = 2
    cfg["model"].update(n_vars=3, n_classes=4, h_dim=16, embed_dim=16, base_depth=4)
    cfg["train"].update(batch_size=4, seq_len=8)
    cfg["agent"].update(k_act=4, k_imag=2, horizon=3, hidden_dim=16)
    cfg["runtime"].update(device="cpu", num_threads=2)
    return cfg


def test_bench_reports_every_variant():
    out = bench.bench(_tiny(), torch.device("cpu"), n=3, budget_minutes=30.0, platform="win32")
    v = out["variants"]
    for name in ("world_model", "heads", "env", "stacked_1", "stacked_2", "stacked_4", "stacked_8"):
        assert v[name]["seed_steps_per_s"] > 0, name
    for n in (1, 2, 4, 8):
        assert v[f"stacked_{n}"]["max_rel_loss_diff"] < 1e-3  # the stacked step reproduces the plain step's losses
    assert not out["compile_gate"]["passed"] and "Linux / CUDA" in out["compile_gate"]["error"]
    assert all("skipped" in v[k] for k in ("compiled", "stacked_1_compiled", "stacked_8_compiled"))
    assert out["one_process_seconds_per_env_step"] > 0


def test_time_budget_skips_the_rest():
    out = bench.bench(_tiny(), torch.device("cpu"), n=2, budget_minutes=0.0, platform="win32")
    assert all("skipped" in v for k, v in out["variants"].items() if k != "world_model") or all("skipped" in v for v in out["variants"].values())


def test_loop_mode():
    out = bench.loop(_tiny(), torch.device("cpu"), seconds=1.0)
    assert out["env_steps_per_s"] > 0
