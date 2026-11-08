"""Wave 1's additions to the analysis (docs/e8/EXPECTED_FORMS.md section 4): form H on synthetic run records, TW4 of
therapy_counts.py on a synthetic replay (equal counts pass, a changed count fails), the device comparison, and P2 marked
not decidable on the baselines alone."""

from __future__ import annotations

import json

import numpy as np
import yaml

from _experiment_loader import load_experiment_module
from helplessness.twin.replay_counts import counts_at, replay_transitions

analyze = load_experiment_module("E8_therapy_relapse", "analyze_e8.py", "test_e8_w1_analyze")
therapy_counts = load_experiment_module("E8_therapy_relapse", "therapy_counts.py", "test_e8_w1_therapy_counts")


def _run(tmp_path, name, role, records, status="done"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump({"e8": {"max_env_steps": 10000, "run": {"model": name, "role": role}}}), encoding="utf-8")
    (d / "meta.yaml").write_text(yaml.safe_dump({"status": status, "env_steps": 10000}), encoding="utf-8")
    (d / "checkpoints.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return d


def _rec(t, success, masses, stat=5.0, switch=0):
    health = None if t == 0 else {"mass_true_by_region": masses, "I_s_by_region": {k: 2.3 for k in masses}}
    return {"env_step": t, "switch_step": switch, "success_rate": None if t == 0 else success,
            "switch_pairs": {"evals": [[t, stat, 1024, 1]]}, "e8": {"health": health}}


def test_form_H(tmp_path):
    ok = {"corridor": 0.97, "nw": 0.95, "ne": 0.99, "se": 0.99, "sw": 0.99}
    low_nw = {**ok, "nw": 0.3}
    _run(tmp_path, "m_exp", "experimental", [_rec(0, None, ok), _rec(500, 1.0, {**low_nw, "corridor": 0.88}), _rec(10000, 1.0, low_nw)])
    _run(tmp_path, "m_ref", "reference", [_rec(0, None, ok), _rec(5000, 1.0, ok), _rec(10000, 1.0, ok)])
    h = analyze.form_H([str(tmp_path / "m_*")])
    assert h["held"]
    exp = h["by_model"]["m_exp"]
    assert exp["transient_before_5000"] == [{"checkpoint": 500, "what": ["mass corridor 0.880"]}]  # the exposed nw: no verdict
    assert [x["checkpoint"] for x in exp["exposed_nw_recovery"]] == [500, 10000]
    _run(tmp_path, "m_bad", "reference", [_rec(0, None, ok), _rec(10000, 0.8, low_nw, stat=0.5, switch=7)])
    bad = analyze.form_H([str(tmp_path / "m_bad")])["by_model"]["m_bad"]
    assert not bad["held"] and not bad["guard_above_collapse"] and not bad["switch_step_0_everywhere"]
    assert bad["failures"][0]["what"] == ["success 0.800", "mass nw 0.300"]


def test_tw4_equal_and_different():
    # two episodes on 3 states, the second still running (pending transition completed by P)
    s = np.array([0, 1, 2, 0, 1]); a = np.array([1, 1, 0, 1, 1])
    terminal = np.array([0, 0, 1, 0, 0], bool); truncated = np.zeros(5, bool); is_first = np.array([1, 0, 0, 1, 0], bool)
    P = np.zeros((3, 2, 3)); P[0, :, 1] = 1; P[1, :, 2] = 1; P[2, :, 2] = 1
    t = replay_transitions(s, a, terminal, truncated, is_first, P=P)
    counts = {f"N_{k}": counts_at(t, k, 3, 2) for k in (0, 2, 3)}
    run = {"P": P, "t": t, "records": [{"env_step": k} for k in (0, 2, 3)], "counts": _Npz(counts)}
    assert therapy_counts.tw4(run)["ok"]
    counts["N_3"] = counts["N_3"].copy(); counts["N_3"][0, 1, 1] += 1
    assert not therapy_counts.tw4({**run, "counts": _Npz(counts)})["equal_by_checkpoint"][3]


class _Npz(dict):
    @property
    def files(self):
        return list(self)


def test_compare_devices_and_p2_decidability():
    row = {"pair": "base_A", "checkpoint": 0, "role": "exposed", "image": "training", **{f: 0.5 for f in analyze.COMPARE_FIELDS}}
    other = {**row, "directed_share": 0.52}
    cmp = analyze.compare_devices([row], [other])
    assert np.isclose(cmp[0]["d_directed_share"], 0.02) and cmp[0]["d_b_walks"] == 0.0
    times = {p: {"T": 150000, "t_half_C3": 5000, "t_half_D": 10000} for p in analyze.BASELINES}
    assert analyze.p2(times)["lags"] == {"base_A": 5000.0, "base_B": 5000.0}


def test_p2_undefined_lag_and_b_not_decidable():
    times = {p: {"T": 150000, "t_half_C3": 2000, "t_half_D": 5000} for p in (*analyze.BASELINES, "shuffle_A", "shuffle_E")}
    times["shuffle_F"] = {"T": 250000, "t_half_C3": "nothing to recover", "t_half_D": 5000}
    out = analyze.p2(times)
    assert out["lags"]["shuffle_F"] == "nothing to recover" and out["a_positive"] == 4 and out["a_held"] is True
    assert out["b_held"] == "not decidable" and out["held"] is True and out["held_on"] == "(a) only"


def test_p2_censoring_gives_bounds():
    """EXPECTED_FORMS section 5 (corrected): D censored -> lower bound (positive), C3 censored -> negative upper bound,
    both censored -> not decidable for the agent; (a) and (b) decided on the intervals."""
    T = 250000
    times = {"base_A": {"T": 150000, "t_half_C3": 2000, "t_half_D": 1000}, "base_B": {"T": 150000, "t_half_C3": 2000, "t_half_D": 2000},
             "shuffle_A": {"T": T, "t_half_C3": 10000, "t_half_D": f"censored at {T}"},
             "shuffle_E": {"T": T, "t_half_C3": 5000, "t_half_D": 15000},
             "shuffle_F": {"T": T, "t_half_C3": f"censored at {T}", "t_half_D": 7500}}
    out = analyze.p2(times)
    assert out["lags"]["shuffle_A"] == ">= 240000 (D censored)" and out["lags"]["shuffle_F"] == "<= -242500 (C3 censored)"
    assert out["lags"]["shuffle_E"] == 10000.0 and out["a_positive"] == 2 and out["a_held"] is False  # 2 positive, none undecided
    assert out["b_held"] is True and out["held"] is False and out["held_on"] == "(a) and (b)"
    times["shuffle_A"] = {"T": T, "t_half_C3": f"censored at {T}", "t_half_D": f"censored at {T}"}
    out = analyze.p2(times)
    assert out["lags"]["shuffle_A"] == "not decidable (both censored)" and out["b_held"] == "not decidable"
    times.update({p: {"T": T, "t_half_C3": 1000, "t_half_D": 2000} for p in ("base_A", "base_B", "shuffle_F")})
    times["shuffle_E"] = {"T": T, "t_half_C3": 1000, "t_half_D": 5000}
    out = analyze.p2(times)  # 4 positive, A undecided: (a) holds whatever A is; (b) hangs on A (E's 4,000 > the light 1,000)
    assert out["a_positive"] == 4 and out["a_held"] is True and out["held"] is True and out["held_on"] == "(a) only"


def _p4_rows(image, treated, b0, bT, d_exp0, d_expT, d_ref=1.0):
    rows = []
    for t, b, d in ((0, b0, d_exp0), (250000, bT, d_expT)):
        common = {"pair": "shuffle_A", "image": image, "checkpoint": t, "treated": treated, "walks_valid": True}
        rows += [{**common, "role": "exposed", "b_walks": b, "directed_share": d}, {**common, "role": "reference", "directed_share": d_ref}]
    return rows


def test_p4_filters():
    rows = (_p4_rows("t1", True, -2.0, -0.2, 0.5, 0.9) + _p4_rows("t2", True, -0.3, -0.1, 0.5, 0.9)  # t2: b(0) > -0.5, f_b out
            + _p4_rows("u1", False, -2.0, -1.0, 0.5, 0.75) + _p4_rows("u2", False, -0.8, -0.8, 0.5, 0.5))  # u2: b(0) > -1.0, out of both
    out = analyze.p4(rows, "shuffle_A")
    assert out["b"]["n_treated"] == 1 and out["b"]["n_untreated"] == 1 and np.isclose(out["b"]["TI"], 0.5 / 0.9)
    assert out["D"]["n_treated"] == 2 and out["D"]["n_untreated"] == 1 and np.isclose(out["D"]["TI"], 0.5 / 0.8)
    assert out["verdict"] == "transfer"


def test_form_H_shuffle_record(tmp_path):
    ok = {"corridor": 0.97, "nw": 0.95, "ne": 0.99, "se": 0.99, "sw": 0.99}
    rec = _rec(10000, 1.0, ok)
    rec["e8"]["health"] = {"0": {"mass_true_by_region": {**ok, "nw": 0.2}, "I_s_by_region": {k: 2.3 for k in ok}},
                           "1": {"mass_true_by_region": {**ok, "se": 0.5}, "I_s_by_region": {k: 2.3 for k in ok}}}
    rec["e8"]["treated_slot"] = "nw"
    _run(tmp_path, "s_exp", "experimental", [_rec(0, None, ok), rec])
    h = analyze.form_H([str(tmp_path / "s_exp")])["by_model"]["s_exp"]
    assert h["failures"] == [{"checkpoint": 10000, "what": ["mass assignment 1: se 0.500"]}]
    assert [x["where"] for x in h["exposed_nw_recovery"]] == ["assignment 0: nw", "assignment 1: nw"]


def test_form_H_unmeasured_region_is_reported_not_failed(tmp_path):
    ok = {"corridor": 0.97, "nw": 0.95, "ne": 0.99, "se": 0.99, "sw": 0.99}
    _run(tmp_path, "u_ref", "reference", [_rec(0, None, ok), _rec(10000, 1.0, {**ok, "sw": None})])
    h = analyze.form_H([str(tmp_path / "u_ref")])["by_model"]["u_ref"]
    assert h["held"] and h["regions_not_measured"] == [{"checkpoint": 10000, "where": "sw"}]
