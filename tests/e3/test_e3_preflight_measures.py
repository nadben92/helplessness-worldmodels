"""preflight_measures.py (docs/e3/DECISIONS.md, 2026-10-01 (10)) without a GPU: the CPU quota is a
problem unless requested = prescribed = usable, the code hash must be RUN_GPU.md's, and each source
run must be the job's (name, meta.yaml, window) with the expected weights."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from _experiment_loader import load_experiment_module

pm = load_experiment_module("E3_worldmodel_4zones", "preflight_measures.py", "test_preflight_measures")
NAME = "grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z"


def _run(root, name=NAME, seed=5, condition="reference", seq_len=64, status="done", weights=b"weights"):
    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    for f in ("world.json", "replay.npz"):
        (d / f).write_bytes(b"x")
    (d / "meta.yaml").write_text(yaml.safe_dump({"seed": seed, "condition": condition, "status": status}), encoding="utf-8")
    (d / "config.yaml").write_text(yaml.safe_dump({"train": {"seq_len": seq_len, "batch_size": 1024 // seq_len}}), encoding="utf-8")
    (d / "checkpoints" / "weights_final.pt").write_bytes(weights)
    return d, hashlib.sha256(weights).hexdigest()


def test_run_identity_and_weights(tmp_path):
    d, sha = _run(tmp_path / "ok")
    assert pm.check_run("win64_seed5_reference", d, sha) == []
    assert "SHA-256" in pm.check_run("win64_seed5_reference", d, "0" * 64)[0]
    assert "directory name" in pm.check_run("win64_seed5_experimental", d, sha)[0]  # a control run under an experimental job
    d2, sha2 = _run(tmp_path / "meta", seed=0)
    assert any("meta.yaml says seed 0" in p for p in pm.check_run("win64_seed5_reference", d2, sha2))
    d3, sha3 = _run(tmp_path / "win", seq_len=16)
    assert any("seq_len 16" in p for p in pm.check_run("win64_seed5_reference", d3, sha3))
    d4, sha4 = _run(tmp_path / "missing")
    (d4 / "replay.npz").unlink()
    assert "lacks ['replay.npz']" in pm.check_run("win64_seed5_reference", d4, sha4)[0]
    assert "does not exist" in pm.check_run("win64_seed5_reference", tmp_path / "nowhere" / NAME, sha)[0]
    with pytest.raises(ValueError):
        pm.parse_job("seed5_reference")


def test_cpu_quota_is_explicit():
    assert pm.cpu_problems(5, 5, {"usable": 5}) == []
    assert len(pm.cpu_problems(3, 5, {"usable": 3})) == 2  # requested and usable both wrong
    assert "usable" in pm.cpu_problems(5, 5, {"usable": 3})[0]
    assert "requested None" in pm.cpu_problems(None, 5, {"usable": 5})[0]


def test_code_hash_is_run_gpu_md_s():
    stated = pm.stated_hash()
    assert stated is not None and stated == pm.source_tree_hash(pm.run_mini.REPO_ROOT)["source_sha256"]


def test_decision(tmp_path, monkeypatch):
    assert pm.decide([], 0.5, 4.0) == "launch"
    assert pm.decide([], 4.5, 4.0) == "stop" and pm.decide(["x"], 0.5, 4.0) == "stop" and pm.decide([], None, 4.0) == "stop"
    d, sha = _run(tmp_path / "src")
    monkeypatch.setitem(pm.cluster.MEASURE_BATCHES, "gpu_e30b_measures",
                        {"source_root": str(tmp_path / "src"), "runs": {"win64_seed5_reference": (NAME, sha)}})
    out = tmp_path / "pre"
    # 5 CPUs requested for a batch prescribed 3 (2026-10-01 (18)): a problem, so stop -- whatever the rest
    assert pm.main(["--batch", "gpu_e30b_measures", "--cpus", "5", "--device", "cpu", "--out-dir", str(out), "--skip-smoke"]) == 1
    verdict = yaml.safe_load((out / "preflight_verdict.json").read_text(encoding="utf-8"))
    assert verdict["decision"] == "stop" and any("CPUs requested 5" in p for p in verdict["problems"])


def test_check_run_of_a_closed_loop_run(tmp_path):
    """Kind "checkpoints" (2026-10-05 (23)): resume point, counts, records and every requested checkpoint's weights."""
    name = "grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z"
    d = tmp_path / name
    (d / "checkpoints").mkdir(parents=True)
    for f in ("checkpoints.jsonl", "counts.npz", "checkpoints/resume.pt", "checkpoints/agent_env063096.pt", "checkpoints/agent_final.pt"):
        (d / f).write_bytes(b"x")
    (d / "meta.yaml").write_text(yaml.safe_dump({"seed": 5, "condition": "reference", "status": "done", "env_steps": 300000}), encoding="utf-8")
    (d / "config.yaml").write_text(yaml.safe_dump({"train": {"seq_len": 64}}), encoding="utf-8")
    assert pm.check_run("win64_seed5_reference", d, None, "checkpoints", [63096, 300000]) == []  # 300,000: agent_final.pt
    assert "agent_env100000.pt" in pm.check_run("win64_seed5_reference", d, None, "checkpoints", [63096, 100000])[0]
    assert "directory name" in pm.check_run("win64_seed0_reference", d, None, "checkpoints", [63096])[0]


def test_smoke_line_separates_the_smoke_from_the_projection():
    """2026-10-07 (28): gpu_pilot_noise's log printed "smoke: 6692 s", the projection per job, for a 197 s measurement."""
    line = pm.smoke_line(196.8, 6692.0, None)
    assert "smoke: 197 s" in line and "projected per job: 6692 s" in line
