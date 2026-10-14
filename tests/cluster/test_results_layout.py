"""helplessness.cluster.handoff.write_results_layout: the tree of a batch's orphan results branch
(docs/RESULTS_CONVENTION.md) -- light files copied, heavy files only listed with their SHA-256."""

from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from helplessness.cluster.handoff import write_results_layout


def _batch(tmp_path):
    run = tmp_path / "results_root" / "grid12x12__2zones__pair-experimental__worldmodel-x__seed00__20261014T000000Z"
    (run / "checkpoints").mkdir(parents=True)
    for name, text in (("config.yaml", "a: 1\n"), ("meta.yaml", "status: done\n"), ("results.json", "{}"),
                       ("REPORT.md", "# r\n"), ("console.log", "line 1\nline 2\n"), ("notes.txt", "not copied")):
        (run / name).write_bytes(text.encode("utf-8"))
    (run / "arrays.npz").write_bytes(b"\x00" * 64)
    (run / "checkpoints" / "agent_final.pt").write_bytes(b"\x01" * 32)
    handoff = tmp_path / "handoff" / "gpu_test"
    handoff.mkdir(parents=True)
    (handoff / "handoff.json").write_text(json.dumps({"runs": [{"job": "j"}], "archives": ["logs.tgz"]}), encoding="utf-8")
    (handoff / "handoff.md").write_text("# handoff\n", encoding="utf-8")
    (handoff / "logs.tgz").write_bytes(b"tgz")
    supervisor = tmp_path / "train-x"
    supervisor.mkdir()
    (supervisor / "status.json").write_text('{"j": {"exit_code": 0}}', encoding="utf-8")
    return run, handoff, supervisor


def test_layout_tree_and_heavy_files(tmp_path):
    run, handoff, supervisor = _batch(tmp_path)
    dest = write_results_layout(tmp_path / "layout", "E4_generalization_gradient", "gpu_test", handoff, supervisor, [run])
    assert dest == tmp_path / "layout" / "results" / "E4_generalization_gradient" / "gpu_test"
    assert sorted(p.name for p in dest.iterdir()) == sorted(["handoff.json", "handoff.md", "logs.tgz", "status.json", run.name])
    assert sorted(p.name for p in (dest / run.name).iterdir()) == sorted(
        ["config.yaml", "meta.yaml", "results.json", "REPORT.md", "console.log.gz"])
    assert gzip.decompress((dest / run.name / "console.log.gz").read_bytes()) == b"line 1\nline 2\n"
    data = json.loads((dest / "handoff.json").read_text(encoding="utf-8"))
    assert data["runs"] == [{"job": "j"}]
    heavy = {entry["path"]: entry for entry in data["heavy_files"]}
    assert set(heavy) == {str((run / "arrays.npz").resolve()), str((run / "checkpoints" / "agent_final.pt").resolve())}
    entry = heavy[str((run / "arrays.npz").resolve())]
    assert entry["sha256"] == hashlib.sha256(b"\x00" * 64).hexdigest() and entry["bytes"] == 64
    assert not any(p.suffix in (".pt", ".npz") for p in dest.rglob("*"))


def test_layout_is_deterministic_and_refuses_an_existing_destination(tmp_path):
    run, handoff, supervisor = _batch(tmp_path)
    a = write_results_layout(tmp_path / "a", "E", "b", handoff, supervisor, [run])
    b = write_results_layout(tmp_path / "b", "E", "b", handoff, supervisor, [run])
    files_a = {p.relative_to(a): p.read_bytes() for p in a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(b): p.read_bytes() for p in b.rglob("*") if p.is_file()}
    assert files_a == files_b
    with pytest.raises(FileExistsError):
        write_results_layout(tmp_path / "a", "E", "b", handoff, supervisor, [run])
