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


def test_single_probe_batch_after_wave_2():
    """docs/e5/DECISIONS.md (32): one probe batch after wave 2, at K = 300,000: baselines A (E4's pair) and B; f = 1 A, B and
    the replacements E and F against B's reference (seed B's own pair is left out at submission: its run locked in)."""
    jobs = dict(submit.commands("probes", "PY", "e5_probes_all", "p1-b-d1", checkpoint=batches.PROBES_CHECKPOINT,
                                e4_results="/scratch/e4/results"))
    assert sorted(jobs) == sorted(f"probes_{label}_{tag}_k300000" for label, tag in
                                  (("A", "base"), ("B", "base"), ("A", "sh100"), ("B", "sh100"), ("E", "sh100"), ("F", "sh100")))
    assert [cmd for name, cmd in jobs.items() if "--e4-results" in cmd] == [jobs["probes_A_base_k300000"]]
    assert all(cmd[cmd.index("--checkpoint") + 1] == "300000" for cmd in jobs.values())
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



def healthy_run(root, seed: int, tag: str, role: str, *, status="done", switch=True, locked=False, weights=True):
    name = (f"grid12x12__2zones-p1-b-d1-{tag}__pair-{'experimental' if role == 'exp' else 'control'}__worldmodel-agent-e5__"
            f"seed{seed:02d}__20261030T000000Z")
    d = root / "results" / "E5_belief_training" / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "meta.yaml").write_text(f"status: {status}" + chr(10) + f"seed: {seed}" + chr(10), encoding="utf-8")
    mass = {q: 0.99 for q in ("corridor", "nw", "ne", "se", "sw")}
    walks = {q: {"I_curve": [2.3]} for q in ("nw", "ne", "se", "sw")}
    stat = 0.001 if locked else 3.0
    rec = {"env_step": 300000, "grad_step": 179846, "switch_step": 6301 if switch else None, "batch_pair_js": stat, "train": {},
           "switch_pairs": {"evals": [[s, stat] for s in range(0, 9000, 50)]},
           "e4": {"real_context": {"mass_true_by_region": mass}, "walks": walks},
           "e5": {"estimator_switch": {"grad_step": 6301, "env_step": 10760, "rule": "late_threshold"} if switch else {}}}
    (d / "checkpoints.jsonl").write_text(json.dumps(rec) + chr(10), encoding="utf-8")
    if weights:
        (d / "checkpoints" / "agent_env300000.pt").write_bytes(b"w")
    return d


def test_probe_plan_keeps_healthy_pairs_and_pairs_the_replacements_with_b(tmp_path):
    """docs/e5/DECISIONS.md (32): at submission, the probes keep the healthy pairs only; seed B's own f = 1 pair is left out
    (its experimental run locked in); E is measured against B's reference; F, failed, is left out; seed A's baseline needs
    E4's two runs at the checkpoint."""
    import shutil

    from helplessness.config import REPO_ROOT

    for rel in ("configs/e5/e5.yaml", "configs/e4/e4.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    for role in ("exp", "ref"):
        healthy_run(tmp_path, 2, "sh100", role)
        healthy_run(tmp_path, 102, "base", role)
    healthy_run(tmp_path, 102, "sh100", "exp", status="failed", switch=False, locked=True, weights=False)
    healthy_run(tmp_path, 102, "sh100", "ref")
    healthy_run(tmp_path, 402, "sh100", "exp")
    healthy_run(tmp_path, 502, "sh100", "exp", status="failed", switch=False, locked=True, weights=False)
    e4 = tmp_path / "e4results" / "E4_generalization_gradient"
    for role in ("experimental", "control"):
        ck = e4 / f"grid12x12__2zones-p1-b-d1__pair-{role}__worldmodel-agent-e4__seed02__20261016T144543Z" / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "agent_env300000.pt").write_bytes(b"w")
    keep, excluded = submit.probe_plan(tmp_path, "e5_probes_all", "p1-b-d1", 300000, str(tmp_path / "e4results"))
    assert keep == {"probes_A_base_k300000", "probes_B_base_k300000", "probes_A_sh100_k300000", "probes_E_sh100_k300000"}
    assert "LOCK-IN" in excluded["probes_B_sh100_k300000"] and "experimental failed" in excluded["probes_F_sh100_k300000"]
    assert cw.reference_seed(tmp_path, "sh100", 402) == 102 and cw.reference_seed(tmp_path, "sh100", 2) == 2


def test_cross_seed_reference_only_when_asked():
    """measure_probes.py pairs an experimental run with another seed's reference only when asked
    (--reference-seed-label: E and F against B's reference, docs/e5/DECISIONS.md (32))."""
    import pytest

    mp = load_experiment_module("E5_belief_training", "measure_probes.py", "e5_measure_probes_cross_seed")
    exp = {"configuration": "p1-b-d1", "seed": 402, "condition": "experimental", "training": "shuffle", "fraction": 1.0}
    ref = {"configuration": "p1-b-d1", "seed": 102, "condition": "reference", "training": "shuffle", "fraction": 1.0}
    with pytest.raises(SystemExit):
        mp.check_pair(exp, ref)
    mp.check_pair(exp, ref, cross_seed=True)


def test_unmeasured_values_are_flagged_not_fatal(tmp_path):
    """A quadrant with no measured state at an early checkpoint is written null (a NaN, pipeline.runtime._jsonable): seed
    B's f = 1 reference at 1,000 and 1,585 (nw), the replacement F at 1,000-3,981 -- check_wave.py crashed on it
    (2026-10-26). It is now flagged (NOT MEASURED, criterion not computable there), never fatal."""
    exp = run_dir(tmp_path, "exp", 0.0, [3.0] * 20)
    ref = run_dir(tmp_path, "ref", math.log2(5), [3.0] * 20)
    recs = [json.loads(x) for x in (ref / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    recs[0]["e4"]["real_context"]["mass_true_by_region"]["nw"] = None
    (ref / "checkpoints.jsonl").write_text("".join(json.dumps(r) + chr(10) for r in recs), encoding="utf-8")
    runs = [cw.read_run(d) for d in (exp, ref)]
    c = cw.check_run(runs[1])
    assert c["not_measured"] == ["100000: mass nw"] and any(f.startswith("NOT MEASURED") for f in c["attention"])
    pair = cw.check_pair(runs[0], runs[1])
    assert pair["converged_at"] == [300000]
    assert any("not computable" in f for f in pair["attention"])



def test_a_stopped_continuation_is_read_paired_and_probed(tmp_path):
    """docs/e5/DECISIONS.md (33): the user stops the continuations' pod once every pair meets E4's criterion. A stopped run
    keeps status running and may end on a truncated record line (a stop while appending); its weights are written
    atomically (whole or absent). check_wave.py reads it (the truncated line ignored), keeps it in its pairs and gives the
    checkpoints where EVERY pair converged; the probe plan keeps it when its weights at K exist."""
    import shutil

    from helplessness.config import REPO_ROOT

    for rel in ("configs/e5/e5.yaml", "configs/e4/e4.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    exp = healthy_run(tmp_path, 402, "sh100", "exp", status="running")
    ref = healthy_run(tmp_path, 102, "sh100", "ref", status="running")
    with open(exp / "checkpoints.jsonl", "a", encoding="utf-8") as f:
        f.write('{"env_step": 325000, "grad_st')  # a stop while appending
    run = cw.read_run(exp)
    assert [r["env_step"] for r in run["records"]] == [300000]
    c = cw.check_run(run)
    assert not any("status" in f for f in c["attention"])
    pair = cw.check_pair(run, cw.read_run(ref))
    assert pair["last_common_env_step"] == 300000 and "b_step1_by_slot" in pair  # paired, read up to its last whole record
    keep, excluded = submit.probe_plan(tmp_path, "e5_probes_all", "p1-b-d1", 300000, str(tmp_path / "no_e4"))
    assert "probes_E_sh100_k300000" in keep, excluded



def test_each_continuation_batch_has_its_own_synthesis(tmp_path, capsys):
    """docs/e5/DECISIONS.md (33): check_wave.py --batch e5_extend_a gives the synthesis line of seed A's pair alone,
    --batch e5_extend_b that of its two pairs (E and F against seed B's reference, which is in the batch)."""
    import shutil

    from helplessness.config import REPO_ROOT

    for rel in ("configs/e5/e5.yaml", "configs/e4/e4.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    for seed, role in ((2, "exp"), (2, "ref"), (102, "ref"), (402, "exp"), (502, "exp")):
        healthy_run(tmp_path, seed, "sh100", role, status="running")
    cw.main(["--batch", "e5_extend_a", "--repo-dir", str(tmp_path)])
    out_a = capsys.readouterr().out
    cw.main(["--batch", "e5_extend_b", "--repo-dir", str(tmp_path)])
    out_b = capsys.readouterr().out
    assert "ALL 1 PAIRS MEET E4's CRITERION AT" in out_a and "sh100 seed 2:" in out_a
    assert "ALL 2 PAIRS MEET E4's CRITERION AT" in out_b and "sh100 seed 402:" in out_b and "sh100 seed 502:" in out_b


def test_continuation_carries_the_switch_of_its_original(tmp_path):
    """docs/e5/DECISIONS.md (34): a continuation (run_e5.py --extend) branched from a run that switched carries that switch;
    before its first checkpoint (no record yet) check_wave reads it from the original (found by name next to it, from
    meta.yaml's branched_from, a cluster path): neither flagged "never switched" nor excluded from its pair. A continuation
    that failed at once is still excluded, as failed, not as never switched."""
    from helplessness.config import REPO_ROOT

    originals = {role: run_dir(tmp_path, role, 0.0 if role == "exp" else math.log2(5), [3.0] * 20) for role in ("exp", "ref")}

    def continuation(role: str, status: str):
        orig = originals[role]
        d = tmp_path / orig.name.replace("20261030T000000Z", "20261031T000000Z")
        d.mkdir()
        (d / "meta.yaml").write_text(f"status: {status}\nseed: 102\nbranched_from: /scratch/helplessness-worldmodels-e5/results/"
                                     f"E5_belief_training/{orig.name}/checkpoints/resume.pt\n", encoding="utf-8")
        (d / "checkpoints.jsonl").write_text("", encoding="utf-8")
        return d

    exp, ref = continuation("exp", "running"), continuation("ref", "running")
    c = cw.check_run(cw.read_run(exp))
    assert c["switch"] == {"grad_step": 6301, "env_step": 10760, "rule": "late_threshold", "from": originals["exp"].name}
    assert c["attention"] == [] and c["lock_in"] is None
    out = tmp_path / "check.json"
    assert cw.main(["--runs", str(exp), str(ref), "--repo-dir", str(REPO_ROOT), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["excluded_pairs"] == [] and len(report["pairs"]) == 1

    (exp / "meta.yaml").write_text((exp / "meta.yaml").read_text(encoding="utf-8").replace("status: running", "status: failed"),
                                   encoding="utf-8")
    c = cw.check_run(cw.read_run(exp))
    assert c["switch"]["grad_step"] == 6301 and not any("never switched" in f for f in c["attention"])
    assert cw.main(["--runs", str(exp), str(ref), "--repo-dir", str(REPO_ROOT), "--out", str(out)]) == 0
    assert [e["reason"] for e in json.loads(out.read_text(encoding="utf-8"))["excluded_pairs"]] == ["experimental failed"]

    lost = tmp_path / "elsewhere"
    lost.mkdir()
    import shutil

    shutil.move(str(ref), lost / ref.name)  # its original not next to it: the switch unknown, flagged, not excluded
    c = cw.check_run(cw.read_run(lost / ref.name))
    assert c["switch"]["grad_step"] is None and any("switch unknown" in f for f in c["attention"])
    assert not any("never switched" in f for f in c["attention"])
