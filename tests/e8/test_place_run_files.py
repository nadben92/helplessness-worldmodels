"""place_run_files.py (docs/e8/DECISIONS.md (22)): each file goes to the run directory named in its file name, copied (the
original kept) with its SHA-256 checked; a missing run directory, an unknown file name or a different file already in place
stops everything before any copy; a second call with the files in place copies nothing."""

from __future__ import annotations

import pytest

from _experiment_loader import load_experiment_module

place = load_experiment_module("E8_therapy_relapse", "place_run_files.py", "test_e8_place_run_files")

RUN = "grid12x12__2zones-p1-b-d1-base-thfull__pair-control__worldmodel-agent-e8__seed02__20261106T151313Z"


def _layout(tmp_path):
    src, root = tmp_path / "src", tmp_path / "batch"
    src.mkdir()
    run_dir = root / "results" / "E8_therapy_relapse" / RUN
    run_dir.mkdir(parents=True)
    (run_dir / "meta.yaml").write_text("status: done\n", encoding="utf-8")
    (src / f"{RUN}__replay_final.npz").write_bytes(b"replay")
    (src / f"{RUN}__counts.npz").write_bytes(b"counts")
    return src, root, run_dir


def test_copies_into_the_named_run_and_keeps_the_originals(tmp_path):
    src, root, run_dir = _layout(tmp_path)
    assert place.main(["--src", str(src), "--batch-root", str(root)]) == 0
    assert (run_dir / "replay_final.npz").read_bytes() == b"replay"
    assert (run_dir / "counts.npz").read_bytes() == b"counts"
    assert (src / f"{RUN}__counts.npz").is_file() and (src / f"{RUN}__replay_final.npz").is_file()
    assert all(e["in_place"] for e in place.plan(src, root))  # a second call copies nothing


def test_dry_run_copies_nothing(tmp_path):
    src, root, run_dir = _layout(tmp_path)
    assert place.main(["--src", str(src), "--batch-root", str(root), "--dry-run"]) == 0
    assert not (run_dir / "counts.npz").exists()


def test_refusals_copy_nothing(tmp_path):
    src, root, run_dir = _layout(tmp_path)
    (run_dir / "counts.npz").write_bytes(b"other")
    with pytest.raises(ValueError, match="another content"):
        place.main(["--src", str(src), "--batch-root", str(root)])
    assert not (run_dir / "replay_final.npz").exists()
    (run_dir / "counts.npz").unlink()
    (src / "grid12x12__missing-run__counts.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="no run directory"):
        place.plan(src, root)
    (src / "grid12x12__missing-run__counts.npz").unlink()
    (src / f"{RUN}__steps.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown file name"):
        place.plan(src, root)
