"""Tests for helplessness.runs: the results-naming convention for E1+."""

from __future__ import annotations

import json

import pytest
import yaml

from helplessness.runs import make_run_dir, make_run_name, write_checkpoint_pointer, write_run_meta


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
