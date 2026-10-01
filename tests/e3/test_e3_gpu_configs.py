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
    # measures: same estimators and settings; only the schedule, the z-only window length and the
    # encoder block of the probes (memory only, docs/e3/DECISIONS.md 2026-09-29) differ
    schedule = {"separation_every", "monitor_every", "diagnostics_every", "z_only_seq_len", "max_images_per_forward"}
    assert {k: v for k, v in bridge["measure"].items() if k not in schedule} == \
        {k: v for k, v in ref["measure"].items() if k not in schedule}
    assert bridge["runtime"]["device"] == "cuda" and bridge["runtime"]["deterministic"] is True
    run_mini.build_separation_switch(bridge)


def test_gpu_e30b_is_the_bridge_with_1024_images_per_step_in_two_shapes():
    """docs/e3/DECISIONS.md, 2026-09-30: gpu_e30b_win64 (16 x 64) and
    gpu_e30b_win16 (64 x 16) differ from the bridge only by the batch shape,
    the seeds and conditions (0 and 5, experimental and reference) and one
    CPU thread per process."""
    bridge = _load("gpu_e30_bridge.yaml")
    for name, shape in (("gpu_e30b_win64.yaml", (16, 64)), ("gpu_e30b_win16.yaml", (64, 16))):
        cfg = _load(name)
        assert (cfg["train"]["batch_size"], cfg["train"]["seq_len"]) == shape
        assert cfg["train"]["batch_size"] * cfg["train"]["seq_len"] == 1024
        assert cfg["seeds"] == [0, 5] and cfg["conditions"] == ["experimental", "reference"]
        assert cfg["runtime"]["num_threads"] == 1
        for block in ("model", "render", "budget", "collection", "measure", "checkpoints", "logging"):
            assert cfg[block] == bridge[block], (name, block)
        assert {k: v for k, v in cfg["train"].items() if k not in ("batch_size", "seq_len")} == \
            {k: v for k, v in bridge["train"].items() if k not in ("batch_size", "seq_len")}
        assert {k: v for k, v in cfg["runtime"].items() if k != "num_threads"} == \
            {k: v for k, v in bridge["runtime"].items() if k != "num_threads"}
        run_mini.build_separation_switch(cfg)


def test_every_e3_config_measures_at_the_corrected_k_with_a_reduced_training_k():
    for name in ("mini.yaml", "mini_long.yaml", "gpu_e30.yaml", "gpu_e30_bridge.yaml", "gpu_e30b_win64.yaml", "gpu_e30b_win16.yaml"):
        m = _load(name)["measure"]
        assert m["kernel_k_s"] == m["real_context"]["kernel_draws"] == 1358, name
        assert m["kernel_k_s_training"] == m["real_context"]["kernel_draws_training"] == 256, name
        assert m["is_bootstrap"] == 200, name
        assert m["is_per_context"] == {"states_per_zone": 4, "contexts": 10, "draws": 512}, name


def test_scale_free_switch_threshold_reproduces_the_8x8_behaviour_exactly():
    """docs/e3/DECISIONS.md, 2026-09-30 (7): 0.375 per variable x 8 = 3 bits, bit for bit; existing configs unchanged."""
    est = {"switch_js_threshold_per_var": 0.375, "collapse_js_threshold_per_var": 0.125}
    assert run_mini.switch_thresholds_bits(est, 8) == (3.0, 1.0)
    assert run_mini.switch_thresholds_bits(est, 4) == (1.5, 0.5)
    for name in ("mini.yaml", "mini_long.yaml", "gpu_e30.yaml", "gpu_e30_bridge.yaml", "gpu_e30b_win64.yaml", "gpu_e30b_win16.yaml",
                 "agent_mini.yaml"):
        cfg = _load(name)
        legacy = cfg["train"]["latent_estimator"]
        switch, _ = run_mini.build_separation_switch(cfg)
        assert (switch.threshold_bits, switch.collapse_threshold_bits) == (legacy["switch_js_threshold"], legacy["collapse_js_threshold"])
        scale_free = {**cfg, "train": {**cfg["train"], "latent_estimator": {
            **{k: v for k, v in legacy.items() if k not in ("switch_js_threshold", "collapse_js_threshold")},
            "switch_js_threshold_per_var": legacy["switch_js_threshold"] / cfg["model"]["n_vars"],
            "collapse_js_threshold_per_var": legacy["collapse_js_threshold"] / cfg["model"]["n_vars"]}}}
        s2, _ = run_mini.build_separation_switch(scale_free)
        assert (s2.threshold_bits, s2.collapse_threshold_bits) == (switch.threshold_bits, switch.collapse_threshold_bits), name
    import pytest
    with pytest.raises(ValueError, match="choose one"):
        run_mini.switch_thresholds_bits({"switch_js_threshold": 3.0, "switch_js_threshold_per_var": 0.375,
                                         "collapse_js_threshold": 1.0}, 8)
