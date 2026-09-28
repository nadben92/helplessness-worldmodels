"""The first GPU run's configs (docs/e3/DECISIONS.md, 2026-09-28): gpu_e30
uses the spec's architecture with an 8x8 latent; the bridge run is
mini_long.yaml except the budget, the device/runtime and the output
schedule -- same model block, same procedure."""

from __future__ import annotations

import yaml

from _experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_gpu_configs_run_mini")
CONFIGS = run_mini.REPO_ROOT / "configs" / "e3"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_gpu_e30_uses_the_spec_architecture_with_an_8x8_latent():
    cfg = _load("gpu_e30.yaml")
    assert cfg["model"] == {"n_vars": 8, "n_classes": 8, "h_dim": 256, "embed_dim": 256, "base_depth": 32, "unimix": 0.01}
    assert (cfg["train"]["batch_size"], cfg["train"]["seq_len"]) == (16, 64)
    assert cfg["seeds"] == [0, 5] and cfg["conditions"] == ["experimental", "reference"]


def test_bridge_differs_from_mini_long_only_by_budget_runtime_and_outputs():
    bridge, ref, gpu = _load("gpu_e30_bridge.yaml"), _load("mini_long.yaml"), _load("gpu_e30.yaml")
    assert bridge["model"] == ref["model"]
    for block in ("render", "collection", "conditions", "seeds"):
        assert bridge[block] == ref[block], block
    train_ref = {k: v for k, v in ref["train"].items() if k != "replay_ratio"}
    assert {k: v for k, v in bridge["train"].items() if k != "replay_ratio"} == train_ref
    # the budget: the same data and gradient steps as gpu_e30
    assert bridge["budget"]["total_steps"] == gpu["budget"]["total_steps"]
    assert bridge["budget"]["prefill_steps"] == gpu["budget"]["prefill_steps"]
    assert bridge["train"]["replay_ratio"] == gpu["train"]["replay_ratio"]
    # measures: same estimators and settings; only the schedule and the z-only window length differ
    schedule = {"separation_every", "monitor_every", "diagnostics_every", "z_only_seq_len"}
    assert {k: v for k, v in bridge["measure"].items() if k not in schedule} == \
        {k: v for k, v in ref["measure"].items() if k not in schedule}
    assert bridge["runtime"]["device"] == "cuda" and bridge["runtime"]["deterministic"] is True
    run_mini.build_separation_switch(bridge)
