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
