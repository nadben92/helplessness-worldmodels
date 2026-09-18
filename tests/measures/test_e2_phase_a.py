"""Tests for experiments/E2_online_4zones/phase_a.py, on small synthetic
structured arrays (matching analysis_data.py's dtypes) -- fast, no
dependency on the real sweep having run."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

ad = load_experiment_module("E2_online_4zones", "analysis_data.py", "_e2_ad_for_phase_a_test")
phase_a = load_experiment_module("E2_online_4zones", "phase_a.py", "_e2_phase_a_for_phase_a_test")


def _make_per_state_rows(specs):
    """specs: list of dicts with the PER_STATE_DTYPE fields (missing
    fields default sensibly)."""
    rows = []
    for spec in specs:
        row = {
            "condition": "online", "alpha": 1.0, "epsilon": 0.1, "seed": 0, "assignment": 0,
            "checkpoint": 1000, "zone": "corridor", "zone_type": "ControllableDeterministic",
            "region": "interior", "state": 1, "I_hat": 0.0, "I_star": 0.0, "b": 0.0, "delta": 0.0,
            "v_pi_true": 0.0, "v_pi_believed": 0.0,
        }
        row.update(spec)
        rows.append(tuple(row[f] for f in ad.PER_STATE_DTYPE.names))
    return np.array(rows, dtype=ad.PER_STATE_DTYPE)


def test_b_exact_zero_status_detects_a_violation():
    rows = _make_per_state_rows([
        {"zone": "corridor", "condition": "online", "b": 0.0},
        {"zone": "corridor", "condition": "online", "b": 1e-12},  # within tolerance
        {"zone": "corridor", "condition": "matched", "b": 0.0},
        {"zone": "controllable_deterministic", "condition": "online", "b": 0.0},
        {"zone": "controllable_deterministic", "condition": "matched", "b": 0.05},  # a real violation
    ])
    result = phase_a.b_exact_zero_status(rows)
    assert result[("corridor", "online")]["status"] == "observed"
    assert result[("controllable_deterministic", "matched")]["status"] == "non observe"


def test_delta_nonneg_status_detects_a_violation():
    rows = _make_per_state_rows([
        {"condition": "online", "delta": 0.5},
        {"condition": "online", "delta": -0.5},  # violation
        {"condition": "matched", "delta": 0.1},
    ])
    results = dict(phase_a.delta_nonneg_status(rows))
    assert results["online"]["status"] == "non observe"
    assert results["online"]["n_violations"] == 1
    assert results["matched"]["status"] == "observed"


def test_b_variance_status_detects_zero_variance():
    # all seeds give the SAME b -- variance should be 0, status "non observe"
    specs = []
    for seed in range(3):
        specs.append({
            "zone": "controllable_noisy", "condition": "online", "alpha": 1.0, "epsilon": 0.1,
            "checkpoint": ad.FINAL_CHECKPOINT, "seed": seed, "b": 0.01,
        })
    rows = _make_per_state_rows(specs)
    result = phase_a.b_variance_status(rows)
    key = (1.0, 0.1, "controllable_noisy")
    assert result[key]["status"] == "non observe"
    assert result[key]["variance"] == 0.0


def test_n_start_leq_n_status_on_synthetic_arrays(tmp_path):
    import json

    run_dir = tmp_path / "run00"
    run_dir.mkdir()
    N_STATES, N_ACTIONS = ad.N_STATES, ad.N_ACTIONS
    N = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.int32)
    N[0, 0, 1] = 10
    N_start_ok = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    N_start_ok[0, 0] = 5  # <= 10, fine
    np.savez_compressed(
        run_dir / "arrays.npz", checkpoint_steps=np.array([1000]),
        N_checkpoint_1000=N, N_start_checkpoint_1000=N_start_ok,
    )
    result = phase_a.n_start_leq_n_status([run_dir])
    assert result["status"] == "observed"

    # now a violation: N_start > N
    N_start_bad = N_start_ok.copy()
    N_start_bad[0, 0] = 50
    np.savez_compressed(
        run_dir / "arrays.npz", checkpoint_steps=np.array([1000]),
        N_checkpoint_1000=N, N_start_checkpoint_1000=N_start_bad,
    )
    result_bad = phase_a.n_start_leq_n_status([run_dir])
    assert result_bad["status"] == "non observe"
    assert result_bad["n_violations"] == 1
