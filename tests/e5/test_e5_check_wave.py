"""check_wave.py (docs/e5/DECISIONS.md (31)): the learning check between two waves, one line per run and per pair, from
light files only; on synthetic runs: a healthy pair (switched, no collapse, mass, E4's criterion met) and a run that
collapsed after its switch. And the single probe batch after wave 3."""

from __future__ import annotations

import json
import math

from _experiment_loader import load_experiment_module
from helplessness.envs.layouts import two_zones

cw = load_experiment_module("E5_belief_training", "check_wave.py", "e5_check_wave_under_test")
batches = load_experiment_module("E5_belief_training", "batches.py", "e5_batches_check_wave")
submit = load_experiment_module("E5_belief_training", "submit_kubeflow.py", "e5_submit_check_wave")


def run_dir(root, role: str, i_ud: float, stats: list[float], mass: float = 0.99, status: str = "done"):
    d = root / f"grid12x12__2zones-p1-b-d1-sh100__pair-{'experimental' if role == 'exp' else 'control'}__worldmodel-agent-e5__seed102__20261030T000000Z"
    d.mkdir()
    (d / "meta.yaml").write_text(f"status: {status}\nseed: 102\n", encoding="utf-8")
    records = []
    for step, js_block in ((100000, stats[:10]), (300000, stats[10:])):
        records.append({"env_step": step, "grad_step": step // 2, "switch_step": 6301, "success_rate": 0.9, "episode_length": 12.0,
                        "batch_pair_js": js_block[-1],
                        "switch_pairs": {"evals": [[6300 + 50 * (i + (0 if step == 100000 else 10)), v, 1024, 10] for i, v in enumerate(js_block)]},
                        "e4": {"real_context": {"mass_true_by_region": {q: mass for q in ("corridor", "nw", "ne", "se", "sw")}},
                               "walks": {q: {"I_curve": [i_ud if q == "nw" else math.log2(5)]} for q in ("nw", "ne", "se", "sw")}},
                        "e5": {"estimator_switch": {"grad_step": 6301, "env_step": 10760, "rule": "late_threshold"}}})
    (d / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return d


def test_check_wave_flags(tmp_path):
    assert cw.ROLES == two_zones.roles(1)
    exp = run_dir(tmp_path, "exp", 0.0, [3.0] * 20)
    ref = run_dir(tmp_path, "ref", math.log2(5), [3.0] * 20)
    runs = [cw.read_run(d) for d in (exp, ref)]
    checks = [cw.check_run(r) for r in runs]
    assert all(c["attention"] == [] for c in checks)
    assert checks[0]["switch"] == {"grad_step": 6301, "env_step": 10760, "rule": "late_threshold"}
    pair = cw.check_pair(runs[0], runs[1])
    assert pair["converged_at"] == [100000, 300000] and pair["attention"] == []
    assert abs(pair["b_step1_by_slot"]["nw"] + math.log2(5)) < 1e-12


def test_check_wave_sees_a_collapse_after_the_switch(tmp_path):
    bad = cw.read_run(run_dir(tmp_path, "exp", 0.0, [3.0] * 12 + [0.0] * 8, mass=0.5))
    c = cw.check_run(bad)
    assert c["collapse"]["longest_below_1bit"] == 8
    assert any("consecutive evaluations below" in f for f in c["attention"]) and any("quadrant mass" in f for f in c["attention"])


def test_single_probe_batch_after_wave_3():
    jobs = dict(submit.commands("probes", "PY", "e5_probes_all", "p1-b-d1", checkpoint=200000, e4_results="/scratch/e4/results"))
    assert sorted(jobs) == sorted(f"probes_{label}_{tag}_k200000" for label, tag in
                                  (("A", "base"), ("B", "base"), ("C", "base"), ("A", "sh100"), ("B", "sh100"), ("C", "sh100"), ("D", "sh100")))
    assert [cmd for name, cmd in jobs.items() if "--e4-results" in cmd] == [jobs["probes_A_base_k200000"]]
    assert all(cmd[cmd.index("--checkpoint") + 1] == "200000" for cmd in jobs.values())
    assert batches.batch_runs("e5_probes_all") == {}


def test_lock_in_and_non_finite_alerts(tmp_path):
    """docs/e5/DECISIONS.md (32): a run that never switched with its statistic below 0.01 bit from gradient step 6,100 on is
    flagged LOCK-IN; a non-finite computed loss is flagged; the heads' losses ABSENT before the switch are not."""
    d = tmp_path / "grid12x12__2zones-p1-b-d1-sh100__pair-experimental__worldmodel-agent-e5__seed102__20261030T000000Z"
    d.mkdir()
    (d / "meta.yaml").write_text("status: running\nseed: 102\n", encoding="utf-8")
    mass = {q: 0.1 for q in ("corridor", "nw", "ne", "se", "sw")}
    walks = {q: {"I_curve": [0.5]} for q in ("nw", "ne", "se", "sw")}
    recs = [{"env_step": 10000, "grad_step": 5846, "switch_step": None, "batch_pair_js": 0.002, "train": {"recon_patch": 2.0},
             "switch_pairs": {"evals": [[s, 0.002] for s in range(0, 5900, 50)]}, "e4": {"real_context": {"mass_true_by_region": mass}, "walks": walks}},
            {"env_step": 25119, "grad_step": 14917, "switch_step": None, "batch_pair_js": 0.001, "train": {"recon_patch": float("nan")},
             "switch_pairs": {"evals": [[s, 0.001] for s in range(5900, 14950, 50)]}, "e4": {"real_context": {"mass_true_by_region": mass}, "walks": walks}}]
    (d / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    c = cw.check_run(cw.read_run(d))
    assert c["lock_in"] is not None and any(f.startswith("LOCK-IN") for f in c["attention"])
    assert c["non_finite"] == ["25119: recon_patch"]
    healthy = cw.read_run(run_dir(tmp_path, "ref", 2.3, [3.0] * 20))
    assert cw.check_run(healthy)["lock_in"] is None and cw.check_run(healthy)["non_finite"] == []
