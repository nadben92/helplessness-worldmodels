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


def _pair(tmp_path, name, i_s_env, i_s_ref, proj_env=None, proj_ref=None, off_env=None, seed=0):
    """One environment / reference pair; off_env: {zone index: extra off-support mass of the environment run}."""
    (tmp_path / name).mkdir()
    e = agg.load_run(_write_run(tmp_path / name, "cfg", seed, "experimental", i_s_env, i_s_env, l1_wm=0.02))
    r = agg.load_run(_write_run(tmp_path / name, "cfg", seed, "reference", i_s_ref, i_s_ref, l1_wm=0.02))
    for run, proj in ((e, proj_env), (r, proj_ref)):
        if proj is not None:
            run["I_s_projected"] = {c: {"mean": proj[k], "se": 0.01} for k, c in enumerate(CATS)}
    for k, extra in (off_env or {}).items():
        e["calibration"]["off_support_mass"][CATS[k]] += extra
    return [e, r]


ENV, REF = [0.1, 1.7, 0.1, 2.3], [2.3, 1.7, 1.7, 2.3]
PROJ_ENV, PROJ_REF = [0.05, 1.6, 0.05, 2.3], [2.3, 1.6, 1.6, 2.3]


def test_e30_ranges_only_flag_aberrations(tmp_path):
    """docs/e3/DECISIONS.md, 2026-10-02 (16), 2: E3-0's ranges flag a closed-loop value outside them; nothing is
    excluded and no verdict is given from them."""
    assert [agg.is_controllable(c) for c in CATS] == [False, True, False, True]
    control = agg.load_e30_control()  # the versioned file of (16), values of (15)
    assert set(control["tau_b"]) == set(control["tau_off"]) == {CATS[1], CATS[3]} and set(control["off_support_interval"]) == {CATS[0], CATS[2]}
    assert control["tau_b"][CATS[1]] == pytest.approx(0.0200, abs=1e-4) and control["off_support_interval"][CATS[2]][0] == pytest.approx(0.1207, abs=1e-4)
    toy = {"tau_b": {CATS[1]: 0.05, CATS[3]: 0.05}, "tau_off": {CATS[1]: 0.02, CATS[3]: 0.02},
           "off_support_interval": {CATS[0]: [0.0, 0.1], CATS[2]: [0.0, 0.1]}}
    clean = agg.aggregate(_pair(tmp_path, "a", ENV, REF, [0.1, 1.68, 0.1, 2.3], [2.3, 1.7, 1.7, 2.3]), toy)["pairs"][0]["bias_control"]
    assert all(v["aberrations"] == [] for v in clean.values()) and not any("verdict" in v for v in clean.values())
    out = agg.aggregate(_pair(tmp_path, "b", ENV, REF, PROJ_ENV, [2.3, 1.7, 1.7, 2.3], off_env={1: 0.1, 2: 0.2}), toy)["pairs"][0]["bias_control"]
    assert len(out[CATS[1]]["aberrations"]) == 2  # b(I_s) - b(projected) = 0.1 > 0.05, and the off-support difference
    assert out[CATS[2]]["aberrations"][0].startswith("off-support (env)") and out[CATS[0]]["aberrations"] == []
    assert out[CATS[3]]["aberrations"] == []
    text = agg.markdown(agg.aggregate(_pair(tmp_path, "c", ENV, REF, PROJ_ENV, PROJ_REF), toy))
    assert "aberration flags" in text and "## Campaign rule" not in text
    bare = agg.aggregate(_pair(tmp_path, "d", ENV, REF))["pairs"][0]["bias_control"]
    assert "aberrations" not in bare[CATS[1]]


def _campaign(tmp_path, diffs_cn, off_cn, seeds, prefix="s"):
    runs = []
    for k, (seed, d, o) in enumerate(zip(seeds, diffs_cn, off_cn)):
        proj_env = [0.05, 1.7 - d, 0.05, 2.3]  # b(I_s) - b(projected) in CN = d
        runs += _pair(tmp_path, f"{prefix}{k}", ENV, REF, proj_env, [2.3, 1.7, 1.6, 2.3], off_env={1: o}, seed=seed)
    return runs


def test_campaign_rule_with_unset_parameters_is_descriptive(tmp_path):
    """(16), 3: N, FRACTION and REFERENCE EFFECT are fixed by the user before the campaign; unset -> not applied."""
    runs = _campaign(tmp_path, [0.01, -0.01, 0.005, -0.005], [0.0, 0.001, -0.001, 0.0], [0, 1, 5, 6])
    cr = agg.aggregate(runs, None, {"n_seeds": None, "fraction": None, "reference_effect": None})["campaign_rule"]
    assert cr["status"].startswith("not applied: n_seeds, fraction, reference_effect not fixed")
    assert set(cr["zones"]) == {CATS[1], CATS[3]} and "verdict" not in cr["zones"][CATS[1]]
    assert cr["zones"][CATS[1]]["b_difference"]["mean"] == pytest.approx(0.0, abs=1e-12)
    pilot = agg.aggregate(runs[:4], None, {"n_seeds": 2, "fraction": 0.5, "reference_effect": 0.02})["campaign_rule"]
    assert pilot["status"] == "not applied: two seeds (pilot); descriptive only"


def test_campaign_rule_confidence_intervals(tmp_path):
    from scipy.stats import t

    diffs = [0.004, -0.002, 0.003, -0.001]
    runs = _campaign(tmp_path, diffs, [0.001, -0.001, 0.0, 0.0005], [0, 1, 5, 6])
    cr = agg.aggregate(runs, None, {"n_seeds": 4, "fraction": 0.5, "reference_effect": -0.02})["campaign_rule"]
    z = cr["zones"][CATS[1]]["b_difference"]
    half = t.ppf(0.975, 3) * np.std(diffs, ddof=1) / 2
    assert z["mean"] == pytest.approx(np.mean(diffs)) and z["ci95"] == pytest.approx([np.mean(diffs) - half, np.mean(diffs) + half])
    assert cr["status"] == "applied" and cr["bound"] == pytest.approx(0.01)
    assert cr["zones"][CATS[1]]["verdict"] == "interpretable"  # both intervals within +- 0.01
    wide = _campaign(tmp_path, [0.03, -0.02, 0.025, -0.01], [0.001, -0.001, 0.0, 0.0005], [0, 1, 5, 6], prefix="w")
    assert agg.aggregate(wide, None, {"n_seeds": 4, "fraction": 0.5, "reference_effect": -0.02})["campaign_rule"]["zones"][CATS[1]]["verdict"] == "NOT interpretable"
    with pytest.raises(ValueError, match="classes A and B in equal numbers"):
        agg.aggregate(_campaign(tmp_path, diffs, [0, 0, 0, 0], [0, 1, 2, 5], prefix="u"), None,
                      {"n_seeds": 4, "fraction": 0.5, "reference_effect": 0.02})
    with pytest.raises(ValueError, match="expected N = 6"):
        agg.aggregate(runs, None, {"n_seeds": 6, "fraction": 0.5, "reference_effect": 0.02})


def test_thresholds_and_floor_of_e30_pairs(tmp_path):
    """The functions that produced E3-0's ranges in (15) (tau_gpu_e30b.py), unchanged."""
    floor_pairs = agg.aggregate(_pair(tmp_path, "a", ENV, REF, PROJ_ENV, PROJ_REF))["pairs"]
    th = agg.thresholds_from_pairs(floor_pairs * 2)  # two identical pairs: sd 0
    assert th["tau_b"][CATS[1]] == 0.0 and th["tau_off_floor"][CATS[0]] == 0.0 and CATS[0] not in th["tau_b"]
    other = agg.aggregate(_pair(tmp_path, "o", [0.1, 1.9, 0.1, 2.3], REF, PROJ_ENV, PROJ_REF))["pairs"]
    diffs = [p["bias_control"][CATS[1]]["b_difference"] for p in floor_pairs + other]
    assert agg.thresholds_from_pairs(floor_pairs + other)["tau_b"][CATS[1]] == pytest.approx(2 * np.std(diffs, ddof=1))
    floor = agg.floor_from_pairs(floor_pairs, {CATS[0]: 0.01, CATS[2]: 0.01})
    assert floor["zones"][CATS[0]]["b_I_s"] == pytest.approx(-2.2)
