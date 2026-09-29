"""aggregate_b.py (docs/e3/DECISIONS.md, 2026-09-30) on synthetic run
directories: pairing by (config, seed), b = experimental - reference by
zone for I_s (main) and I_z (secondary), class means, calibration flags."""

from __future__ import annotations

import json

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

agg = load_experiment_module("E3_worldmodel_4zones", "aggregate_b.py", "test_aggregate_b")
CATS = agg.CATEGORIES


def _write_run(root, config, seed, condition, i_s, i_z, acc=0.995, z_only=0.995, l1_wm=0.02):
    role = "experimental" if condition == "experimental" else "control"
    d = root / f"grid12x12__4zones__pair-{role}__worldmodel-{config}__seed{seed:02d}__20260930T000000Z"
    d.mkdir()
    zone_of_state = [CATS[s % 4] for s in range(144)]
    results = {"seed": seed, "condition": condition, "zone_of_state": zone_of_state,
               "accuracies": {"acc_replay_hard": acc}, "z_only_probe": {"test_accuracy": z_only},
               "real_context": {"kernel_draws": 1358, "l1_floor_by_zone": {c: 0.0 for c in CATS},
                                "I_s_by_zone": {c: {"mean": i_s[k], "se": 0.01} for k, c in enumerate(CATS)},
                                "I_z_by_zone": {c: {"mean": i_z[k], "se": 0.02} for k, c in enumerate(CATS)}}}
    (d / "results.json").write_text(json.dumps(results), encoding="utf-8")
    P = np.zeros((144, 5, 144))
    P[np.arange(144), :, np.arange(144)] = 1.0
    p_hat = P.copy()
    p_hat[:, :, 0] += l1_wm / 2  # L1 error of l1_wm on every row (s != 0)
    p_hat[np.arange(144), :, np.arange(144)] -= l1_wm / 2
    np.savez(d / "arrays.npz", N=np.full((144, 5, 144), 1.0), P_true=P, T_hat_B=P, real_p_hat=p_hat)
    return d


def test_pairs_b_class_means_and_calibration(tmp_path):
    dirs = [
        _write_run(tmp_path, "gpu_e30b_win16", 0, "experimental", [0.1, 1.7, 0.1, 2.3], [0.3, 2.0, 0.3, 2.3]),
        _write_run(tmp_path, "gpu_e30b_win16", 0, "reference", [2.3, 1.7, 1.7, 2.3], [2.3, 2.1, 1.9, 2.3]),
        _write_run(tmp_path, "gpu_e30b_win16", 5, "experimental", [0.2, 1.6, 0.0, 2.2], [0.4, 2.0, 0.2, 2.2], l1_wm=0.5),
        _write_run(tmp_path, "gpu_e30b_win16", 5, "reference", [2.2, 1.6, 1.6, 2.2], [2.2, 2.0, 1.9, 2.2]),
        _write_run(tmp_path, "gpu_e30b_win64", 0, "experimental", [0.1, 1.7, 0.1, 2.3], [0.3, 2.0, 0.3, 2.3]),
    ]
    result = agg.aggregate([agg.load_run(d) for d in dirs])
    assert [(p["config"], p["seed"], p["zone_class"]) for p in result["pairs"]] == [("gpu_e30b_win16", 0, "A"), ("gpu_e30b_win16", 5, "B")]
    assert result["unpaired"] == [str(dirs[4])]
    b = result["pairs"][0]["I_s"]
    assert b[CATS[0]]["b"] == pytest.approx(0.1 - 2.3) and b[CATS[1]]["b"] == pytest.approx(0.0)
    assert b[CATS[0]]["se"] == pytest.approx(np.hypot(0.01, 0.01))
    assert result["pairs"][0]["I_z"][CATS[2]]["b"] == pytest.approx(0.3 - 1.9)
    classes = result["class_means"]
    assert set(classes) == {"gpu_e30b_win16/A", "gpu_e30b_win16/B"}
    assert classes["gpu_e30b_win16/A"]["all_calibrated"] and not classes["gpu_e30b_win16/B"]["all_calibrated"]
    cal = result["pairs"][1]["calibration"]["experimental"]
    assert not cal["c_b"][CATS[1]]["ok"] and cal["c_a"]["ok"] and cal["c_c"]["ok"]
    text = agg.markdown(result)
    assert "I_s (MAIN)" in text and "I_z (secondary)" in text and "gpu_e30b_win16/A" in text


def test_a_duplicate_run_is_refused(tmp_path):
    a = _write_run(tmp_path, "gpu_e30b_win16", 0, "experimental", [0] * 4, [0] * 4)
    (tmp_path / "copy").mkdir()
    b = _write_run(tmp_path / "copy", "gpu_e30b_win16", 0, "experimental", [0] * 4, [0] * 4)
    with pytest.raises(ValueError, match="two experimental runs"):
        agg.aggregate([agg.load_run(a), agg.load_run(b)])
