"""docs/e5/DECISIONS.md (36): the preliminary probes of E and F at 475,000 compared with baseline B at its checkpoints
matched to their level of learning (matched_checkpoint.py), and the baselines probed at 300,000 and at those checkpoints
in one pod (e5_probes_baselines, --checkpoint k [k ...]); the single-checkpoint submissions are unchanged."""

from __future__ import annotations

import json
import shutil

import pytest

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT

batches = load_experiment_module("E5_belief_training", "batches.py", "e5_batches_matched")
submit = load_experiment_module("E5_belief_training", "submit_kubeflow.py", "e5_submit_matched")
mc = load_experiment_module("E5_belief_training", "matched_checkpoint.py", "e5_matched_checkpoint_under_test")

Q = ("nw", "ne", "se", "sw")


def repo(tmp_path):
    for rel in ("configs/e5/e5.yaml", "configs/e4/e4.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    return tmp_path


def run(root, seed: int, tag: str, role: str, masses: dict[int, float | dict], weights=(), stamp="20261030T000000Z",
        i_step1: dict | None = None):
    """A run with one record per step: the same mass in every quadrant (or a dict by quadrant; None = not measured)."""
    name = (f"grid12x12__2zones-p1-b-d1-{tag}__pair-{'experimental' if role == 'exp' else 'control'}__worldmodel-agent-e5__"
            f"seed{seed:02d}__{stamp}")
    d = root / "results" / "E5_belief_training" / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "meta.yaml").write_text(f"status: running\nseed: {seed}\n", encoding="utf-8")
    recs = []
    for step, m in masses.items():
        by_q = m if isinstance(m, dict) else {q: m for q in Q}
        walks = {q: {"I_curve": [(i_step1 or {}).get(q, 2.3)]} for q in Q}
        recs.append({"env_step": step, "e4": {"real_context": {"mass_true_by_region": {**by_q, "corridor": 0.99}}, "walks": walks},
                     "e5": {"estimator_switch": {"grad_step": 6101, "env_step": 12000, "rule": "threshold"}}})
    (d / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    for k in weights:
        (d / "checkpoints" / f"agent_env{k:06d}.pt").write_bytes(b"w")
    return d


def test_single_checkpoint_submissions_unchanged():
    """One --checkpoint value gives the jobs it gave before (33)'s sets and E4's baseline A."""
    assert sorted(dict(submit.commands("probes", "PY", "e5_probes_b", "p1-b-d1", checkpoint=475000))) == \
        ["probes_E_sh100_k475000", "probes_F_sh100_k475000"]
    assert sorted(dict(submit.commands("probes", "PY", "e5_probes_a", "p1-b-d1", checkpoint=550000, e4_results="/e4"))) == \
        ["probes_A_base_k300000", "probes_A_sh100_k550000", "probes_B_base_k300000"]
    assert batches.probe_specs("e5_probes_b", 475000) == batches.probe_specs("e5_probes_b", [475000])


def test_baselines_at_300000_and_baseline_b_at_its_matched_checkpoints():
    """e5_probes_baselines: A and B at PROBES_CHECKPOINT, B also at each checkpoint given; no duplicate job (a matched
    checkpoint equal to another, or to 300,000)."""
    jobs = dict(submit.commands("probes", "PY", "e5_probes_baselines", "p1-b-d1", checkpoint=[39811, 63096, 39811, 300000],
                                e4_results="/e4"))
    assert list(jobs) == ["probes_A_base_k300000", "probes_B_base_k300000", "probes_B_base_k039811", "probes_B_base_k063096"]
    for name, cmd in jobs.items():
        assert cmd[cmd.index("--checkpoint") + 1] == name[-6:].lstrip("0")
        assert cmd[cmd.index("--training") + 1] == "baseline" and "--reference-seed-label" not in cmd
    assert jobs["probes_A_base_k300000"][jobs["probes_A_base_k300000"].index("--e4-results") + 1] == "/e4"
    with pytest.raises(ValueError):  # seed A's baseline is E4's pair
        submit.commands("probes", "PY", "e5_probes_baselines", "p1-b-d1", checkpoint=[39811])


def test_several_checkpoints_in_one_dry_run(tmp_path, capsys):
    """--checkpoint takes several values: one pod, the plan checked per job (a checkpoint without weights LEFT OUT)."""
    root = repo(tmp_path)
    for role in ("exp", "ref"):
        run(root, 102, "base", role, {300000: 0.99}, weights=(300000, 39811))
    e4 = root / "e4results" / "E4_generalization_gradient"
    for role in ("experimental", "control"):
        ck = e4 / f"grid12x12__2zones-p1-b-d1__pair-{role}__worldmodel-agent-e4__seed02__20261016T144543Z" / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "agent_env300000.pt").write_bytes(b"w")
    assert submit.main(["--batch", "e5_probes_baselines", "--phase", "probes", "--checkpoint", "39811", "63096",
                        "--e4-results", str(root / "e4results"), "--repo-dir", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "LEFT OUT probes_B_base_k063096" in out
    plan = json.loads(out[out.index("{"):])
    assert [name for name, _ in plan["jobs"]] == ["probes_A_base_k300000", "probes_B_base_k300000", "probes_B_base_k039811"]


def test_probe_plan_leaves_out_a_matched_checkpoint_without_weights(tmp_path):
    root = repo(tmp_path)
    for role in ("exp", "ref"):
        run(root, 102, "base", role, {300000: 0.99}, weights=(300000, 39811))
    e4 = root / "e4results" / "E4_generalization_gradient"
    for role in ("experimental", "control"):
        ck = e4 / f"grid12x12__2zones-p1-b-d1__pair-{role}__worldmodel-agent-e4__seed02__20261016T144543Z" / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "agent_env300000.pt").write_bytes(b"w")
    keep, excluded = submit.probe_plan(root, "e5_probes_baselines", "p1-b-d1", [39811, 63096], str(root / "e4results"))
    assert keep == {"probes_A_base_k300000", "probes_B_base_k300000", "probes_B_base_k039811"}
    assert "agent_env063096.pt" in excluded["probes_B_base_k063096"]


def test_matched_step_is_the_first_checkpoint_reaching_the_level():
    """The level is the smallest of both runs' quadrant masses; a checkpoint with a quadrant not measured does not count;
    a later dip does not move the match back."""
    exp = [{"env_step": s, "e4": {"real_context": {"mass_true_by_region": m}}} for s, m in
           ((10000, {"nw": None, "ne": 0.9, "se": 0.9, "sw": 0.9}), (25119, {q: 0.86 for q in Q}), (39811, {q: 0.88 for q in Q}),
            (63096, {q: 0.95 for q in Q}), (100000, {q: 0.87 for q in Q}))]
    ref = [{"env_step": s, "e4": {"real_context": {"mass_true_by_region": {q: 0.99 for q in Q}}}} for s in (10000, 25119, 39811, 63096, 100000)]
    k, trajectory = mc.matched_step(exp, ref, 0.868)
    assert k == 39811 and trajectory == [(10000, None), (25119, 0.86), (39811, 0.88)]
    assert mc.matched_step(exp, ref, 0.99) == (None, [(10000, None), (25119, 0.86), (39811, 0.88), (63096, 0.95), (100000, 0.87)])


def test_matched_checkpoint_of_e_and_f_against_baseline_b(tmp_path, capsys):
    """e5_extend_b at 475,000: E (level 0.868 = its own smallest mass) and F (0.887) against B's reference; baseline B's
    matched checkpoints, the declared level (E4's criterion not met, the failing sub-criteria) and the submission line."""
    root = repo(tmp_path)
    ud = {"nw": 2.3 - 2.098, "ne": 2.3, "se": 2.3, "sw": 2.3}
    run(root, 402, "sh100", "exp", {450000: 0.867, 475000: {"nw": 0.868, "ne": 0.9, "se": 0.95, "sw": 0.9}}, i_step1=ud)
    run(root, 502, "sh100", "exp", {475000: 0.887}, i_step1=ud)
    run(root, 102, "sh100", "ref", {450000: 0.932, 475000: 0.939})
    run(root, 102, "sh100", "ref", {300000: 0.5}, stamp="20261022T000000Z")  # the original: not the newest
    grid = {10000: 0.5, 15849: 0.7, 25119: 0.86, 39811: 0.88, 63096: 0.95, 100000: 0.99}
    for role in ("exp", "ref"):
        run(root, 102, "base", role, grid, weights=(39811, 63096))
    out = tmp_path / "matched.json"
    mc.main(["--batch", "e5_extend_b", "--at", "475000", "--repo-dir", str(root), "--out", str(out)])
    text = capsys.readouterr().out
    res = {r["pair"]: r for r in json.loads(out.read_text(encoding="utf-8"))}
    e, f = res["E vs reference B"], res["F vs reference B"]
    assert (e["level"], e["matched_checkpoint"]) == (0.868, 39811) and (f["level"], f["matched_checkpoint"]) == (0.887, 63096)
    assert not e["declared"]["converged"]
    assert any(x.startswith("mass experimental 0.868") for x in e["declared"]["failing"])
    assert any(x.startswith("mass reference 0.939") for x in e["declared"]["failing"])
    assert any(x.startswith("|b_UD + log2 5| = 0.224") for x in e["declared"]["failing"])
    assert e["weights"] == {"experimental": True, "reference": True} and e["before_switch"] == []
    assert "--checkpoint 39811 63096" in text and "--batch e5_probes_baselines" in text


def test_no_match_is_an_error(tmp_path):
    root = repo(tmp_path)
    run(root, 402, "sh100", "exp", {475000: 0.95})
    run(root, 502, "sh100", "exp", {475000: 0.95})
    run(root, 102, "sh100", "ref", {475000: 0.95})
    for role in ("exp", "ref"):
        run(root, 102, "base", role, {10000: 0.5, 100000: 0.9})
    with pytest.raises(SystemExit, match="never reaches"):
        mc.main(["--batch", "e5_extend_b", "--at", "475000", "--repo-dir", str(root)])
    with pytest.raises(SystemExit, match="no checkpoint record at 500000"):
        mc.main(["--batch", "e5_extend_b", "--at", "500000", "--repo-dir", str(root)])
