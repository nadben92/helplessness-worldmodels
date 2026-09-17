"""Tests for experiments/E2_online_4zones/episode_analysis.py, against a
real (tiny) sweep run -- not synthetic data, since the whole point is
reading the sweep's own per-run summary.json/arrays.npz files correctly.
"""

from __future__ import annotations

from _experiment_loader import load_experiment_module

sweep_mod = load_experiment_module("E2_online_4zones", "sweep.py", "_e2_sweep_for_episode_analysis_test")
ea_mod = load_experiment_module("E2_online_4zones", "episode_analysis.py", "_e2_episode_analysis_for_episode_analysis_test")

CONDITION = {"alpha": 1.0, "epsilon": 0.1, "key": "ea_test_condition"}


def test_condition_keys_present_and_analyze_condition_on_a_real_tiny_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(sweep_mod, "EXPERIMENT_ID", "ea_test")
    monkeypatch.setattr(sweep_mod, "BUDGET", 5_000)
    monkeypatch.setattr(sweep_mod, "CHECKPOINTS", (2_500, 5_000))
    monkeypatch.setattr(ea_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(ea_mod, "EXPERIMENT_ID", "ea_test")

    sweep_mod.run_and_save(CONDITION, 0)
    sweep_mod.run_and_save(CONDITION, 1)

    keys = ea_mod.condition_keys_present()
    assert keys == ["ea_test_condition"]

    report = ea_mod.analyze_condition("ea_test_condition")
    assert set(report.keys()) <= {
        "corridor", "uncontrollable_deterministic", "controllable_noisy",
        "controllable_deterministic", "uncontrollable_noisy",
    }
    total_episodes = sum(stats["n_episodes"] for stats in report.values())
    assert total_episodes > 0
    for zone, stats in report.items():
        assert stats["median_length"] > 0
        assert stats["p90_length"] >= stats["median_length"]
        assert stats["p99_length"] >= stats["p90_length"]
        assert 0.0 <= stats["truncation_rate"] <= 1.0


def test_condition_keys_present_empty_when_no_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(ea_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(ea_mod, "EXPERIMENT_ID", "nonexistent")
    assert ea_mod.condition_keys_present() == []


def test_zone_of_state_for_assignment_covers_every_state_exactly_once():
    zones = ea_mod.zone_of_state_for_assignment(0)
    assert len(zones) == ea_mod.N_STATES
    assert all(z is not None for z in zones)


def test_zone_of_state_for_assignment_differs_across_rotations():
    """A sanity check that the per-run rotation is actually being used --
    not silently defaulting to rotation 0 for every run regardless of
    what summary.json says."""
    zones_r0 = ea_mod.zone_of_state_for_assignment(0)
    zones_r1 = ea_mod.zone_of_state_for_assignment(1)
    assert not (zones_r0 == zones_r1).all()
