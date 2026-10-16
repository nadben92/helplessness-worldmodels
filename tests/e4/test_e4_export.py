"""export_batch.py: one archive per batch (light files, final weights, probes, supervisor logs, manifest) and its check;
plus the two walk lengths of the probes (the grid's walks are the full walks cut after n_in_grid steps in the zone)."""

from __future__ import annotations

import json
import tarfile

import numpy as np

from _experiment_loader import load_experiment_module

export = load_experiment_module("E4_generalization_gradient", "export_batch.py", "test_e4_export_batch")
e4_setup = load_experiment_module("E4_generalization_gradient", "e4_setup.py", "test_e4_export_setup")


def make_repo(root):
    base = root / "results" / "E4_generalization_gradient"
    for role in ("experimental", "control"):
        run = base / f"grid12x12__2zones-p1-a-d1__pair-{role}__worldmodel-agent-e4__seed01__20261016T000000Z"
        (run / "checkpoints").mkdir(parents=True)
        for name in ("config.yaml", "results.json", "metrics.csv", "checkpoints.jsonl", "console.log"):
            (run / name).write_text("x\n", encoding="utf-8")
        (run / "meta.yaml").write_text("status: done\nseed: 1\n", encoding="utf-8")
        for step in (100000, 150000, 300000):
            (run / "checkpoints" / f"agent_env{step:06d}.pt").write_bytes(bytes([step % 251]) * 50)
        (run / "checkpoints" / "resume.pt").write_bytes(b"r" * 10)
        (run / "replay_final.npz").write_bytes(b"z" * 10)
    probes = base / "probes__p1-a-d1__env300000__20261016T010000Z"
    probes.mkdir()
    for name in ("images.csv", "probes.csv", "status.json"):
        (probes / name).write_text("y\n", encoding="utf-8")
    sup = root / "logs" / "kubeflow" / "e4-train-e4_wave1-20261016T000000000000Z"
    sup.mkdir(parents=True)
    (sup / "status.json").write_text("{}", encoding="utf-8")
    (sup / "p1-a-d1_experimental.log").write_text("log", encoding="utf-8")
    return root


def test_create_then_verify(tmp_path, capsys):
    repo = make_repo(tmp_path / "repo")
    archive = tmp_path / "out" / "e4_wave1.tar.gz"
    assert export.main(["create", "--batch", "e4_wave1", "--repo-dir", str(repo), "--out", str(archive),
                        "--weights-checkpoints", "150000"]) == 0
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "MANIFEST.json" in names
    assert sum(n.endswith("agent_env300000.pt") for n in names) == 2 and sum(n.endswith("agent_env150000.pt") for n in names) == 2
    assert not any(n.endswith(("agent_env100000.pt", "resume.pt", "replay_final.npz")) for n in names)
    assert any(n.endswith("probes.csv") for n in names) and any(n.endswith("p1-a-d1_experimental.log") for n in names)
    out = tmp_path / "extracted"
    with tarfile.open(archive) as tar:
        tar.extractall(out)
    capsys.readouterr()
    assert export.main(["verify", str(out)]) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["runs"]["p1-a-d1_experimental"]["status"] == "done" and manifest["runs"]["p1-b-d1_experimental"]["status"] == "not found"
    victim = next(out.rglob("metrics.csv"))
    victim.write_text("changed\n", encoding="utf-8")
    assert export.main(["verify", str(out)]) == 1


def test_grid_walks_are_the_full_walks_cut():
    cfg = e4_setup.load_config()
    full, grid = e4_setup.walks_for_slot(cfg, "nw"), e4_setup.walks_for_slot(cfg, "nw", n_in=cfg["e4"]["walks"]["n_in_grid"])
    assert grid["steps_since_entry"].max() == cfg["e4"]["walks"]["n_in_grid"]
    for i in range(full["states"].shape[0]):
        n = grid["length"][i]
        assert np.array_equal(grid["states"][i, :n], full["states"][i, :n]) and (grid["steps_since_entry"][i] > 0).sum() == 10
        assert grid["actions"][i, n - 1] == -1
