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
    # off-support mass from the saved kernel (l1_wm / 2 of each row is on state 0, outside the support of most states)
    off = result["pairs"][1]["calibration"]["experimental"]["off_support_mass"]
    assert all(0.2 < v <= 0.25 for v in off.values()) and "off-support mass by zone" in text
    assert "## I_s projected" not in text  # these runs have no projected I_s: its sections are left out


def test_a_duplicate_run_is_refused(tmp_path):
    a = _write_run(tmp_path, "gpu_e30b_win16", 0, "experimental", [0] * 4, [0] * 4)
    (tmp_path / "copy").mkdir()
    b = _write_run(tmp_path / "copy", "gpu_e30b_win16", 0, "experimental", [0] * 4, [0] * 4)
    with pytest.raises(ValueError, match="two experimental runs"):
        agg.aggregate([agg.load_run(a), agg.load_run(b)])


def _pair(tmp_path, name, i_s_env, i_s_ref, proj_env=None, proj_ref=None, off_env=None):
    """One environment / reference pair; off_env: {zone index: extra off-support mass of the environment run}."""
    (tmp_path / name).mkdir()
    e = agg.load_run(_write_run(tmp_path / name, "cfg", 0, "experimental", i_s_env, i_s_env, l1_wm=0.02))
    r = agg.load_run(_write_run(tmp_path / name, "cfg", 0, "reference", i_s_ref, i_s_ref, l1_wm=0.02))
    for run, proj in ((e, proj_env), (r, proj_ref)):
        if proj is not None:
            run["I_s_projected"] = {c: {"mean": proj[k], "se": 0.01} for k, c in enumerate(CATS)}
    for k, extra in (off_env or {}).items():
        e["calibration"]["off_support_mass"][CATS[k]] += extra
    return [e, r]


def test_bias_control_reading_rule_by_zone_type(tmp_path):
    """docs/e3/DECISIONS.md, 2026-09-30 (9): controllable zones (CN, CD) need the two versions of b to agree and the
    off-support mass to be the same in both environments; uncontrollable zones (UN, UD) are read against the E3-0
    floor, provided the environment-side off-support mass stays within the floor's interval."""
    assert [agg.is_controllable(c) for c in CATS] == [False, True, False, True]
    env, ref = [0.1, 1.7, 0.1, 2.3], [2.3, 1.7, 1.7, 2.3]
    proj_env, proj_ref = [0.05, 1.6, 0.05, 2.3], [2.3, 1.6, 1.6, 2.3]
    runs = _pair(tmp_path, "a", env, ref, proj_env, proj_ref)
    partial = agg.aggregate(_pair(tmp_path, "p", env, ref), 0.1, 0.02)["pairs"][0]["bias_control"]
    assert partial[CATS[1]]["verdict"].startswith("partial: off-support ok") and partial[CATS[0]]["verdict"] == "floor unavailable"
    full = agg.aggregate(runs, 0.1, 0.02)["pairs"][0]["bias_control"]
    assert full[CATS[1]]["b_I_s"] == pytest.approx(0.0) and full[CATS[1]]["verdict"] == "interpretable"
    assert full[CATS[0]]["b_I_s"] == pytest.approx(-2.2) and full[CATS[0]]["verdict"] == "floor unavailable"
    # thresholds: twice the sd of the tested quantity, over the E3-0 pairs, controllable zones only
    floor_pairs = agg.aggregate(runs, 0.1, 0.02)["pairs"]
    th = agg.thresholds_from_pairs(floor_pairs * 2)  # two identical pairs: sd 0
    assert th["tau_b"][CATS[1]] == 0.0 and th["tau_off_floor"][CATS[0]] == 0.0 and CATS[0] not in th["tau_b"]
    other = agg.aggregate(_pair(tmp_path, "o", [0.1, 1.9, 0.1, 2.3], ref, [0.05, 1.6, 0.05, 2.3], proj_ref), 0.1, 0.02)["pairs"]
    th2 = agg.thresholds_from_pairs(floor_pairs + other)  # b(I_s) - b(projected) in CN: -0.1 + 0.1 = 0 and 0.1 + 0.3... see below
    diffs = [p["bias_control"][CATS[1]]["b_difference"] for p in floor_pairs + other]
    assert th2["tau_b"][CATS[1]] == pytest.approx(2 * np.std(diffs, ddof=1))
    floor = agg.floor_from_pairs(floor_pairs, {CATS[0]: 0.01, CATS[2]: 0.01})
    assert floor["zones"][CATS[0]]["b_I_s"] == pytest.approx(-2.2)
    # a closed-loop pair whose conveyor b matches the floor, off-support mass inside the interval
    ok = agg.aggregate(_pair(tmp_path, "c", [0.2, 1.7, 0.2, 2.3], ref, [0.1, 1.6, 0.1, 2.3], proj_ref), 0.1, 0.02, floor)["pairs"][0]["bias_control"]
    assert ok[CATS[0]]["verdict"] == "read against the floor" and ok[CATS[0]]["b_minus_floor"] == pytest.approx(0.1)
    # environment-side off-support mass outside the interval: that zone is diagnostic only, the others unaffected
    out = agg.aggregate(_pair(tmp_path, "d", [0.2, 1.7, 0.2, 2.3], ref, [0.1, 1.6, 0.1, 2.3], proj_ref, off_env={2: 0.2}),
                        0.1, 0.02, floor)["pairs"][0]["bias_control"]
    assert out[CATS[2]]["verdict"].startswith("diagnostic only") and out[CATS[0]]["verdict"] == "read against the floor"
    # a controllable zone whose off-support mass differs between the environments
    leak = agg.aggregate(_pair(tmp_path, "e", env, ref, proj_env, proj_ref, off_env={1: 0.1}), 0.1, 0.02, floor)["pairs"][0]["bias_control"]
    assert leak[CATS[1]]["verdict"] == "NOT interpretable" and leak[CATS[3]]["verdict"] == "interpretable"
    # thresholds are per zone
    runs = _pair(tmp_path, "f", env, ref, [0.05, 1.5, 0.05, 2.3], proj_ref)  # in CN the projected b is -0.1 while b(I_s) is 0
    per_zone = agg.aggregate(runs, {CATS[1]: 0.01, CATS[3]: 0.5}, {CATS[1]: 0.5, CATS[3]: 0.5}, floor)["pairs"][0]["bias_control"]
    assert per_zone[CATS[1]]["verdict"] == "NOT interpretable"  # |b(I_s) - b(projected)| = 0.1 > 0.01 in that zone
    assert per_zone[CATS[3]]["verdict"] == "interpretable"
    text = agg.markdown(agg.aggregate(runs, th["tau_b"], th["tau_off"], floor))
    assert "## Bias control" in text and "read against the floor" in text
