"""diagnose_value_generalisation.py (docs/e3/DECISIONS.md, 2026-10-05 (24)) on a tiny closed-loop run: the distance to
the goal under P, the actions-played summary, and one point end to end."""

from __future__ import annotations

import json

import numpy as np

from _experiment_loader import load_experiment_module
from test_e3_closed_loop import run_agent, run_mini, tiny_config

vd = load_experiment_module("E3_worldmodel_4zones", "diagnose_value_generalisation.py", "test_value_diag")


def test_distance_to_goal_on_a_chain():
    # 3 states in a line, goal 0; action 0 moves left deterministically, action 1 stays
    P = np.zeros((3, 2, 3))
    P[0, :, 0] = 1
    P[1, 0, 0] = P[2, 0, 1] = 1
    P[1, 1, 1] = P[2, 1, 2] = 1
    assert vd.distance_to_goal(P, 0).tolist() == [0.0, 1.0, 2.0]


def test_spearman_constant_critic_is_explicitly_undefined():
    result = vd.spearman_with_ci(np.ones(4), np.arange(4), np.random.default_rng(0))
    assert result["undefined_reason"] == "constant critic values"
    assert result["n_states"] == 4
    assert np.isnan(result["rho"]) and all(np.isnan(bound) for bound in result["ci95"])
    assert vd.run_mini._jsonable(result)["ci95"] == [None, None]


def test_spearman_constant_goal_distances_are_undefined():
    result = vd.spearman_with_ci(np.arange(4), np.ones(4), np.random.default_rng(0))
    assert result["undefined_reason"] == "constant goal distances"
    assert np.isnan(result["rho"]) and all(np.isnan(bound) for bound in result["ci95"])


def test_spearman_without_valid_bootstrap_correlations(monkeypatch):
    class ConstantResamples:
        def integers(self, low, high, size):
            return np.zeros(size, dtype=int)

    monkeypatch.setattr(vd, "N_BOOT", 3)
    result = vd.spearman_with_ci(np.arange(4), -np.arange(4), ConstantResamples())
    assert result["rho"] == -1.0
    assert result["undefined_reason"] == "no valid bootstrap correlations"
    assert all(np.isnan(bound) for bound in result["ci95"])


def test_spearman_nonconstant_inputs_keep_finite_interval(monkeypatch):
    monkeypatch.setattr(vd, "N_BOOT", 40)
    result = vd.spearman_with_ci(np.arange(6), -np.arange(6), np.random.default_rng(0))
    assert result["rho"] == -1.0
    assert np.allclose(result["ci95"], [-1.0, -1.0])
    assert "undefined_reason" not in result


def test_actions_played_and_repeat():
    arr = {"a": np.array([1, 1, 2, 0, 3, 3, 3, 0]), "is_first": np.array([1, 0, 0, 0, 1, 0, 0, 0], bool),
           "terminal": np.array([0, 0, 0, 1, 0, 0, 0, 0], bool), "truncated": np.zeros(8, bool)}
    out = vd.actions_played(arr, 0, 10)
    assert out["n"] == 7 and out["repeat_previous_action"] == 3 / 5  # pairs within episodes: (1,1) (1,2) (3,3) (3,3) (3,0)


def test_one_point_end_to_end(tmp_path):
    cfg = tiny_config()
    cfg["measure"]["final"] = False
    cfg["measure"]["checkpoint_goal_kernel"] = {"draws": 4}
    d = tmp_path / "r"
    d.mkdir()
    run_agent.run_one(cfg, 0, "experimental", d, run_mini.configure_runtime(cfg))
    out = tmp_path / "o"
    assert vd.main(["--points", f"{d}:200", "--device", "cpu", "--out-dir", str(out)]) == 0
    r = json.loads((out / "value_diagnostics.json").read_text())[0]
    assert r["env_step"] == 200 and r["actions_played_interval"]["from"] == 100
    assert -1.0 <= r["spearman_V_vs_distance_to_goal"]["rho"] <= 1.0 or np.isnan(r["spearman_V_vs_distance_to_goal"]["rho"])
    assert 0.0 <= r["greedy_optimal_share"] <= 1.0 and len(r["goal_pair_masses_at_c"]) == 4


def _original_spearman_with_ci(x, y, rng):
    """The version that ran in gpu_pilot_remeasure's first attempt (commit ff01253), kept to check the fix of 90c5465."""
    from scipy.stats import spearmanr

    rho = float(spearmanr(x, y).correlation)
    boots = []
    for _ in range(vd.N_BOOT):
        i = rng.integers(0, len(x), len(x))
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"rho": rho, "ci95": [float(lo), float(hi)], "n_states": int(len(x))}


def test_fix_gives_the_original_values_wherever_the_original_did_not_fail(monkeypatch):
    """docs/e3/DECISIONS.md, 2026-10-06 (25): on non-constant inputs -- including small samples with ties, where some
    bootstrap resamples are constant and the original dropped them as NaN -- the fixed function returns exactly the
    original's values and draws the same random numbers."""
    import warnings

    monkeypatch.setattr(vd, "N_BOOT", 300)
    cases = [(np.random.default_rng(1).normal(size=40), np.random.default_rng(2).integers(0, 9, 40).astype(float)),
             (np.array([0.0, 0.0, 0.0, 1.0, 1.0]), np.array([4.0, 3.0, 2.0, 1.0, 0.0])),  # resamples often constant
             (np.arange(30.0), -np.arange(30.0) ** 2)]
    for x, y in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = _original_spearman_with_ci(x, y, np.random.default_rng(7))
            b = vd.spearman_with_ci(x, y, np.random.default_rng(7))
        assert a == b


def test_a_checkpoint_before_the_switch_has_a_constant_critic(tmp_path):
    """Before the separation switch the agent's heads are not trained and their output layers start at zero: V is 0
    everywhere, the correlation is undefined (reported, not a failure) and every greedy choice is a tie."""
    cfg = tiny_config()
    cfg["measure"]["final"] = False
    cfg["measure"]["checkpoint_goal_kernel"] = {"draws": 4}
    cfg["train"]["latent_estimator"].update(switch_js_threshold=1e9, switch_max_steps=100_000)  # no switch in this run
    d = tmp_path / "r"
    d.mkdir()
    report = run_agent.run_one(cfg, 0, "experimental", d, run_mini.configure_runtime(cfg))
    assert report["switch_step"] is None
    r = vd.diagnose_point(d, 200, "cpu")
    assert r["spearman_V_vs_distance_to_goal"]["undefined_reason"] == "constant critic values"
    assert r["V_by_state_quantiles"] == [0.0, 0.0, 0.0] and r["greedy_tie_fraction_by_state"] == 1.0
    assert r["greedy_optimal_share"] == r["uniform_optimal_share"]
