"""campaign_analysis.py (docs/e3/DECISIONS.md, 2026-10-07 (29)): the secondary analyses on synthetic arrays, and the whole
script on a tiny environment / reference pair run with the campaign's measures."""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "test_campaign_analysis")
run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_campaign_analysis_run_agent")
run_mini = run_agent.run_mini
CN, CD = "controllable_noisy", "controllable_deterministic"


def test_visits_of_the_least_visited_action_and_b_by_visit_class():
    N = np.zeros((3, 5, 3), dtype=int)
    N[0, :, 1] = [5, 12, 40, 7, 9]
    N[1, :, 2] = 50
    N[2, :, 0] = [0, 1, 1, 1, 1]
    assert ca.min_action_visits(N).tolist() == [5, 50, 0]
    zone = np.array([CN, CN, CD])
    I_env, n_env = np.array([1.0, 2.0, 3.0]), np.array([5, 50, 400])
    I_ref, n_ref = np.array([0.5, 1.0, np.nan]), np.array([20, 60, 400])
    b = ca.b_by_visit_class(I_env, n_env, I_ref, n_ref, zone)
    assert b[CN]["0-9"] == {"b": b[CN]["0-9"]["b"], "n_states_env": 1, "n_states_ref": 0} and np.isnan(b[CN]["0-9"]["b"])
    assert b[CN]["30-99"] == {"b": 1.0, "n_states_env": 1, "n_states_ref": 1}   # 2.0 (env, n 50) - 1.0 (ref, n 60)
    assert b[CN]["10-29"]["n_states_ref"] == 1 and b[CN]["10-29"]["n_states_env"] == 0
    assert b[CD]["300-999"]["n_states_ref"] == 0  # a NaN state is left out


def test_zone_separation_on_archived_posteriors(tmp_path):
    """The fallback for runs measured without the in-run record (the pilot): z only, from the measure's arrays."""
    rng = np.random.default_rng(0)
    zones = [CN] * 4 + [CD] * 4 + ["uncontrollable_noisy"] * 4 + ["uncontrollable_deterministic"] * 4 + ["corridor"]
    states = np.repeat(np.arange(17), 20)
    post = rng.random((states.size, 8, 8)) * 0.1
    for s in range(17):  # one class per zone category on the first variable: perfectly separable
        post[states == s, 0, list(dict.fromkeys(zones)).index(zones[s])] += 1.0
    np.savez(tmp_path / "a.npz", real_post_probs=post, real_context_states=states)
    sep = ca.separation_from_arrays(np.load(tmp_path / "a.npz"), zones)
    assert sep["n_contexts"] == 16 * 20 and sep["probe_zone"]["test_accuracy"] == 1.0 and sep["latent"].startswith("z only")
    np.savez(tmp_path / "n.npz", real_post_probs=rng.random((states.size, 8, 8)) * 0.1, real_context_states=states)  # no signal
    assert sep["centroid_distance_ratio"] > 2 * ca.separation_from_arrays(np.load(tmp_path / "n.npz"), zones)["centroid_distance_ratio"]
    np.savez(tmp_path / "b.npz", real_p_hat=np.zeros(1))
    assert ca.separation_from_arrays(np.load(tmp_path / "b.npz"), zones) is None


def test_campaign_analysis_on_a_tiny_pair(tmp_path):
    from test_e3_campaign_run import campaign_tiny

    cfg = campaign_tiny()
    dirs = []
    for cond in ("experimental", "reference"):
        d = run_mini.make_seed_run_dir(tmp_path, "agent-tiny", 0, cond)
        run_agent.run_one(cfg, 0, cond, d, run_mini.configure_runtime(cfg))
        dirs.append(d)
    res = ca.analyse(dirs, e30_control=ca.aggregate_b.load_e30_control())
    rule = res["main"]["aggregate_b"]["campaign_rule"]
    assert rule["status"].startswith("not applied") and len(res["main"]["aggregate_b"]["pairs"]) == 1
    (pair,) = res["secondary"]
    assert pair["checkpoints"] and pair["checkpoints"][-1]["env_step"] == 300
    for c in pair["checkpoints"]:
        assert set(c["b_I_s"]) == set(ca.CATEGORIES) and set(c["alpha"]) == set(ca.ZONES) == set(c["b_I_s_all"])  # corridor: (31)
        assert res["propagation"] is not None and "over_seeds" in res
        for cat in ca.CATEGORIES:
            d = c["difference_of_differences"][cat]
            assert np.isclose(d["did"], d["wm_minus_twin_env"] - d["wm_minus_twin_ref"], equal_nan=True)
        assert c["latent_zone_separation"]["experimental"]["latent"] == "h and z (recorded in the run)"
        assert set(c["agent"]["reference"]) == {"V_by_zone", "goal_neighbours"}
    text = ca.markdown(res)
    assert "Campaign rule" in text and "b by visit class" in text and "Zones' separation" in text
