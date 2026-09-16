"""Tests for experiments/E1_offline_4zones/artifact.py: the region=='all'
invariant is asserted on load (not assumed), the writer/loader round-trip,
and join_exp_ctrl builds the exp-vs-ctrl comparison E2/E3 are meant to
reuse rather than reimplement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "E1_offline_4zones"))
from artifact import (  # noqa: E402
    join_exp_ctrl,
    load_measurement_artifact,
    write_measurement_artifact,
)


def _base_row(**overrides):
    row = {
        "exp": "E1", "env": "exp", "zone": "z", "zone_type": "ControllableDeterministic",
        "n_budget": 100, "seed": 0, "region": "all",
        "I_hat": 2.0, "I_star": 2.0, "b": 0.0, "delta": 0.0, "l1": 0.0,
        "v_pi_true": 10.0, "v_pi_believed": 10.0,
    }
    row.update(overrides)
    return row


def _ctrl_row(**overrides):
    row = _base_row(env="ctrl", b=float("nan"))
    row.update(overrides)
    return row


def test_round_trip_preserves_values(tmp_path):
    rows = [_base_row(), _ctrl_row()]
    path = write_measurement_artifact(rows, tmp_path / "m.csv")
    loaded = load_measurement_artifact(path)
    assert len(loaded) == 2
    assert loaded[0]["I_hat"] == 2.0
    assert loaded[0]["b"] == 0.0


def test_invariant_rejects_i_hat_populated_outside_all(tmp_path):
    bad = _base_row(region="interior")  # I_hat/I_star/l1 must be NaN here
    path = write_measurement_artifact([bad], tmp_path / "m.csv")
    with pytest.raises(ValueError, match="I_hat"):
        load_measurement_artifact(path)


def test_invariant_rejects_i_hat_missing_at_all(tmp_path):
    bad = _base_row(I_hat=float("nan"))
    path = write_measurement_artifact([bad], tmp_path / "m.csv")
    with pytest.raises(ValueError, match="I_hat"):
        load_measurement_artifact(path)


def test_invariant_rejects_b_populated_on_ctrl_row(tmp_path):
    bad = _ctrl_row(b=-1.0)
    path = write_measurement_artifact([bad], tmp_path / "m.csv")
    with pytest.raises(ValueError, match="b"):
        load_measurement_artifact(path)


def test_invariant_rejects_b_missing_on_exp_all_row(tmp_path):
    bad = _base_row(b=float("nan"))
    path = write_measurement_artifact([bad], tmp_path / "m.csv")
    with pytest.raises(ValueError, match="b"):
        load_measurement_artifact(path)


def test_valid_interior_and_boundary_rows_pass(tmp_path):
    rows = [
        _base_row(region="all"),
        _ctrl_row(region="all"),
        _base_row(region="interior", I_hat=float("nan"), I_star=float("nan"), l1=float("nan"), b=float("nan")),
        _base_row(region="boundary", I_hat=float("nan"), I_star=float("nan"), l1=float("nan"), b=float("nan")),
    ]
    path = write_measurement_artifact(rows, tmp_path / "m.csv")
    loaded = load_measurement_artifact(path)
    assert len(loaded) == 4


def test_join_exp_ctrl_merges_matching_pairs(tmp_path):
    rows = [
        _base_row(delta=1.0),
        _ctrl_row(delta=2.0),
    ]
    joined = join_exp_ctrl(rows)
    assert len(joined) == 1
    assert joined[0]["delta"] == 1.0
    assert joined[0]["ctrl_delta"] == 2.0


def test_join_exp_ctrl_skips_unmatched_exp_rows():
    rows = [_base_row(seed=1)]  # no ctrl counterpart
    assert join_exp_ctrl(rows) == []
