"""make_handoff.py (docs/e3/DECISIONS.md, 2026-09-30 (3); RUN_GPU.md step 5)
on a synthetic batch: one light archive per run without any .pt, the logs,
handoff.md with each run's status, and a refusal to overwrite."""

from __future__ import annotations

import json
import tarfile

import pytest
import yaml

from _experiment_loader import load_experiment_module

handoff = load_experiment_module("E3_worldmodel_4zones", "make_handoff.py", "test_make_handoff")


def _job(config, seed, condition, exit_code, started="2026-10-01T08:00:05+00:00"):
    return {"command": ["python", "-u", "run_mini.py", "--config", f"configs/e3/{config}.yaml", "--device", "cuda", "--seeds", str(seed),
                        "--conditions", condition, "--wandb-artifact"], "exit_code": exit_code, "started_at": started,
            "finished_at": "2026-10-01T18:00:00+00:00"}


def _run(root, name, meta, finished):
    d = root / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "weights_final.pt").write_bytes(b"w" * 100)
    (d / "failure").mkdir()
    (d / "failure" / "checkpoint.pt").write_bytes(b"w")
    (d / "meta.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    (d / "replay.npz").write_bytes(b"r")
    if finished:
        (d / "results.json").write_text("{}", encoding="utf-8")
    return d


def test_handoff_archives_and_report(tmp_path):
    results = tmp_path / "results"
    _run(results, "grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260930T120000Z",
         {"status": "done"}, True)  # an older attempt, before the job: not this batch's
    done = _run(results, "grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20261001T080006Z",
                {"status": "done", "switch_step": 4201}, True)
    failed = _run(results, "grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win16__seed05__20261001T080007Z",
                  {"status": "failed", "steps_done": 25000, "switch_step": None, "error": "separation not reached\nmore"}, False)
    train = tmp_path / "logs" / "train-20261001T080000Z"
    train.mkdir(parents=True)
    (train / "status.json").write_text(json.dumps({
        "win64_seed0_experimental": _job("gpu_e30b_win64", 0, "experimental", 0),
        "win16_seed5_reference": _job("gpu_e30b_win16", 5, "reference", 1),
        "win16_seed0_reference": {**_job("gpu_e30b_win16", 0, "reference", None), "not_started": True},
    }), encoding="utf-8")
    (train / "win64_seed0_experimental.log").write_text("ok", encoding="utf-8")
    pre = tmp_path / "logs" / "preflight-20261001T070000Z"
    (pre / "runs").mkdir(parents=True)
    (pre / "runs" / "preflight_verdict.json").write_text(json.dumps(
        {"decision": "launch", "parallel": 8, "projected_batch_hours": 11.2, "problems": [], "warnings": []}), encoding="utf-8")
    (pre / "runs" / "weights.pt").write_bytes(b"w")
    out = tmp_path / "handoff" / "gpu_e30b"

    assert handoff.main(["--train-dir", str(train), "--preflight-dir", str(pre), "--out", str(out), "--results-root", str(results)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted([f"{done.name}.tgz", f"{failed.name}.tgz", "logs.tgz", "handoff.md", "handoff.json"])
    for archive in out.glob("*.tgz"):
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        assert names and not any(n.endswith(".pt") or "/checkpoints" in n for n in names), names
    text = (out / "handoff.md").read_text(encoding="utf-8")
    assert "decision **launch**" in text and "| win16_seed0_reference | - | not started |" in text
    assert "| win64_seed0_experimental | 0 | done | - | 4201 | yes | - |" in text
    assert "| win16_seed5_reference | 1 | failed | 25000 | none | no | separation not reached |" in text
    with pytest.raises(SystemExit):
        handoff.main(["--train-dir", str(train), "--out", str(out), "--results-root", str(results)])


def test_handoff_without_training(tmp_path):
    """The preflight said stop: the handoff holds the preflight only."""
    pre = tmp_path / "preflight-20261001T070000Z"
    (pre / "runs").mkdir(parents=True)
    (pre / "runs" / "preflight_verdict.json").write_text(json.dumps(
        {"decision": "stop", "parallel": 4, "projected_batch_hours": 25.0, "problems": [], "warnings": []}), encoding="utf-8")
    out = tmp_path / "handoff"
    assert handoff.main(["--preflight-dir", str(pre), "--out", str(out), "--results-root", str(tmp_path)]) == 0
    text = (out / "handoff.md").read_text(encoding="utf-8")
    assert "decision **stop**" in text and "Training not launched" in text and (out / "logs.tgz").is_file()
