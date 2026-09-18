"""Tests for experiments/E2_online_4zones/synthesis_table.py, on a
small synthetic per_state array."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

ad = load_experiment_module("E2_online_4zones", "analysis_data.py", "_e2_ad_for_synth_test")
st = load_experiment_module("E2_online_4zones", "synthesis_table.py", "_e2_synth_for_synth_test")


def _row(**kwargs):
    row = {
        "condition": "online", "alpha": 1.0, "epsilon": 0.1, "seed": 0, "assignment": 0,
        "checkpoint": ad.FINAL_CHECKPOINT, "zone": "corridor", "zone_type": "ControllableDeterministic",
        "region": "interior", "state": 1, "I_hat": 0.0, "I_star": 0.0, "b": 0.0, "delta": 0.0,
        "v_pi_true": 0.0, "v_pi_believed": 0.0,
    }
    row.update(kwargs)
    return tuple(row[f] for f in ad.PER_STATE_DTYPE.names)


def test_build_synthesis_table_computes_paired_diff_correctly(tmp_path, monkeypatch):
    monkeypatch.setattr(st.fg, "FIGURES_DIR", tmp_path)
    rows = []
    for alpha in (0.5, 1.0, 5.0):
        for zone in ad.ZONE_CATEGORIES:
            for seed in range(3):
                rows.append(_row(alpha=alpha, zone=zone, seed=seed, condition="online", b=1.0, delta=0.2))
                rows.append(_row(alpha=alpha, zone=zone, seed=seed, condition="matched", b=0.4, delta=0.2))
    arr = np.array(rows, dtype=ad.PER_STATE_DTYPE)

    table_rows, exterior_rows = st.build_synthesis_table(arr)
    row = next(r for r in table_rows if r["alpha"] == 1.0 and r["zone"] == "corridor")
    assert row["b_online_mean"] == pytest.approx(1.0)
    assert row["b_matched_mean"] == pytest.approx(0.4)
    assert row["b_diff_mean"] == pytest.approx(0.6)
    assert row["n_seeds"] == 3

    ext_row = next(r for r in exterior_rows if r["alpha"] == 1.0 and r["condition"] == "online")
    assert ext_row["delta_exterior_mean"] == pytest.approx(0.2)
