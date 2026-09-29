"""Tests for helplessness.runs: the results-naming convention for E1+."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import helplessness.runs as runs_module
from helplessness.runs import (
    UnknownCodeVersionError,
    code_provenance,
    make_run_dir,
    make_run_name,
    require_code_provenance,
    source_tree_hash,
    write_checkpoint_pointer,
    write_run_meta,
)


def test_make_run_name_format():
    name = make_run_name(
        grid_shape=(16, 16), zone_preset="4zones-std", pair_role="experimental",
        agent_type="tabular", seed=3, timestamp="20260910T153000Z",
    )
    assert name == "grid16x16__4zones-std__pair-experimental__tabular__seed03__20260910T153000Z"


def test_make_run_name_rejects_invalid_pair_role():
    with pytest.raises(ValueError, match="pair_role"):
        make_run_name((8, 8), "preset", "escapable", "tabular", 0)


def test_make_run_dir_creates_structure(tmp_path):
    run_dir = make_run_dir(tmp_path, "E1_offline_4zones", "grid8x8__preset__pair-control__tabular__seed00__ts")
    assert run_dir.is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "artifacts").is_dir()


def test_make_run_dir_refuses_to_overwrite_an_existing_run(tmp_path):
    make_run_dir(tmp_path, "E1", "run-a")
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "E1", "run-a")


def test_write_run_meta_first_write_sets_seed_and_commit(tmp_path):
    run_dir = make_run_dir(tmp_path, "E1", "run-a")
    write_run_meta(run_dir, seed=7, status="running")
    with open(run_dir / "meta.yaml") as f:
        meta = yaml.safe_load(f)
    assert meta["seed"] == 7
    assert meta["status"] == "running"
    assert "git_commit" in meta
    assert "started_at" in meta


def test_write_run_meta_second_write_preserves_started_at(tmp_path):
    run_dir = make_run_dir(tmp_path, "E1", "run-a")
    write_run_meta(run_dir, seed=7, status="running")
    with open(run_dir / "meta.yaml") as f:
        first = yaml.safe_load(f)

    write_run_meta(run_dir, seed=7, status="done")
    with open(run_dir / "meta.yaml") as f:
        second = yaml.safe_load(f)

    assert second["status"] == "done"
    assert second["started_at"] == first["started_at"]


def test_write_checkpoint_pointer(tmp_path):
    run_dir = make_run_dir(tmp_path, "E3", "run-a")
    write_checkpoint_pointer(run_dir, "step_000100000.pt", step=100_000)
    with open(run_dir / "checkpoints" / "latest.json") as f:
        pointer = json.load(f)
    assert pointer == {"filename": "step_000100000.pt", "step": 100_000}


# ---------------------------------------------------------------------------
# code provenance (docs/e3/DECISIONS.md, 2026-09-29): a git commit, or a
# content hash of the source files when git is unavailable -- never neither
# ---------------------------------------------------------------------------


def _tree(root):
    (root / "src" / "pkg" / "__pycache__").mkdir(parents=True)
    (root / "src" / "pkg" / "mod.py").write_bytes(b"x = 1\n")
    (root / "src" / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00cache")
    (root / "configs").mkdir()
    (root / "configs" / "a.yaml").write_bytes(b"a: 1\n")
    (root / "pyproject.toml").write_bytes(b"[project]\n")
    (root / "results").mkdir()
    (root / "results" / "big.npz").write_bytes(b"data")


def test_source_tree_hash_covers_the_code_only_and_ignores_line_endings(tmp_path):
    _tree(tmp_path)
    first = source_tree_hash(tmp_path)
    assert first["source_n_files"] == 3  # mod.py, a.yaml, pyproject.toml; not the cache, not results/
    assert source_tree_hash(tmp_path) == first
    (tmp_path / "src" / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"other")
    (tmp_path / "results" / "big.npz").write_bytes(b"other")
    assert source_tree_hash(tmp_path) == first
    (tmp_path / "src" / "pkg" / "mod.py").write_bytes(b"x = 1\r\n")  # a Windows checkout of the same file
    assert source_tree_hash(tmp_path) == first
    (tmp_path / "src" / "pkg" / "mod.py").write_bytes(b"x = 2\n")
    assert source_tree_hash(tmp_path)["source_sha256"] != first["source_sha256"]
    (tmp_path / "src" / "pkg" / "mod.py").write_bytes(b"x = 1\n")
    (tmp_path / "src" / "pkg" / "renamed.py").write_bytes((tmp_path / "src" / "pkg" / "mod.py").read_bytes())
    (tmp_path / "src" / "pkg" / "mod.py").unlink()
    assert source_tree_hash(tmp_path)["source_sha256"] != first["source_sha256"]  # paths count too


def test_source_tree_hash_of_an_empty_scope_is_none(tmp_path):
    assert source_tree_hash(tmp_path) is None


def test_provenance_without_git_falls_back_to_the_content_hash(tmp_path, monkeypatch):
    _tree(tmp_path)

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(runs_module.subprocess, "run", no_git)
    prov = code_provenance(tmp_path)
    assert prov["git_commit"] == "unknown" and prov["git_dirty"] is None
    assert prov["source_sha256"] == source_tree_hash(tmp_path)["source_sha256"]
    require_code_provenance(prov)  # the hash is enough


def test_provenance_outside_a_checkout_is_not_reported_clean(tmp_path):
    _tree(tmp_path)
    prov = code_provenance(tmp_path)  # tmp_path is not a git checkout (git may or may not be installed)
    assert prov["git_commit"] == "unknown" and prov["git_dirty"] is None
    assert prov["source_sha256"]


def test_a_run_without_commit_nor_hash_is_refused():
    with pytest.raises(UnknownCodeVersionError):
        require_code_provenance({"git_commit": "unknown", "source_sha256": None})
    require_code_provenance({"git_commit": "abc123", "source_sha256": None})


def test_the_repository_s_own_provenance_has_both():
    root = Path(__file__).resolve().parents[1]
    prov = code_provenance(root)
    assert prov["source_sha256"] and prov["source_n_files"] > 100
