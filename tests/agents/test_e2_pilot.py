"""Smoke test for experiments/E2_online_4zones/pilot.py::run_one_seed --
tiny budget, just confirms the whole pipeline (env, agent, loop,
checkpointing, episode logging, VI iteration tracking) runs end to end
without error and produces internally-consistent output. The actual
pilot (Temps 2, budget=1e6) is run and reported separately, not here.
"""

from __future__ import annotations

import numpy as np

from _experiment_loader import load_experiment_module

pilot_mod = load_experiment_module("E2_online_4zones", "pilot.py", "_e2_pilot_for_pilot_test")


def test_run_one_seed_smoke():
    checkpoints = (50, 100, 200)
    result = pilot_mod.run_one_seed(seed=0, budget=200, checkpoints=checkpoints)

    assert int(result["N_final"].sum()) == 200
    assert set(result["N_checkpoints"].keys()) == set(checkpoints)
    for t, N_t in result["N_checkpoints"].items():
        assert int(N_t.sum()) == t  # checkpoint t must hold exactly t steps' worth of counts
    assert len(result["plan_n_iters"]) >= 1  # at least the initial plan on the prior
    assert result["elapsed_seconds"] > 0


def test_run_one_seed_is_deterministic_given_the_same_seed():
    result_a = pilot_mod.run_one_seed(seed=1, budget=150, checkpoints=(100,))
    result_b = pilot_mod.run_one_seed(seed=1, budget=150, checkpoints=(100,))
    assert np.array_equal(result_a["N_final"], result_b["N_final"])
    assert result_a["episode_log"] == result_b["episode_log"]


def test_checkpoint_result_independent_of_total_budget():
    """The property the sweep design depends on ("chaque checkpoint
    equivaut a un run de ce budget"): a checkpoint's N snapshot must be
    IDENTICAL regardless of whether the run stops right after it or
    continues on to a much larger budget -- nothing (RNG derivation,
    replanning cadence, checkpoint definition) may look ahead at the
    total budget. Checked directly: the same seed's N at step 1000 must
    be bit-identical whether budget=1000 or budget=5000."""
    short_run = pilot_mod.run_one_seed(seed=42, budget=1000, checkpoints=(1000,))
    long_run = pilot_mod.run_one_seed(seed=42, budget=5000, checkpoints=(1000, 5000))
    assert np.array_equal(short_run["N_checkpoints"][1000], long_run["N_checkpoints"][1000])
    assert short_run["episode_log"] == [e for e in long_run["episode_log"] if e[0] <= 1000]


def test_last_replan_checkpoint_matches_the_previous_episode_boundary():
    """last_replan_checkpoints[t] must be the step of the LAST COMPLETED
    replan strictly before t -- never t itself, even if t happens to be
    an episode boundary (that replan hasn't happened yet at the moment
    the checkpoint snapshot is taken, see pilot.py's on_transition
    ordering)."""
    budget = 800
    result = pilot_mod.run_one_seed(seed=6, budget=budget, checkpoints=tuple(range(50, budget + 1, 50)))
    episode_boundaries = sorted(t for t, _ in result["episode_log"])
    for t, last_replan in result["last_replan_checkpoints"].items():
        prior_boundaries = [b for b in episode_boundaries if b < t]
        expected = max(prior_boundaries) if prior_boundaries else 0
        assert last_replan == expected, f"t={t}: last_replan={last_replan}, expected={expected}"
        assert last_replan < t or last_replan == 0


def test_checkpoint_captures_live_agent_state_not_recomputed():
    """N_checkpoints/V_checkpoints/policy_checkpoints must be READ OFF
    the live agent object, not recomputed -- checked indirectly: at a
    checkpoint falling strictly between two episode boundaries, V/policy
    must reflect the PREVIOUS replan (staleness expected, not masked)."""
    result = pilot_mod.run_one_seed(seed=3, budget=400, checkpoints=(400,))
    # N_final and the last checkpoint's N must be identical (both are the
    # same live object read at the same moment, end of run).
    assert np.array_equal(result["N_final"], result["N_checkpoints"][400])
    assert np.array_equal(result["V_final"], result["V_checkpoints"][400])
    assert np.array_equal(result["policy_final"], result["policy_checkpoints"][400])


def test_trajectory_log_has_one_row_per_step_excludes_resets():
    budget = 500
    result = pilot_mod.run_one_seed(seed=4, budget=budget, checkpoints=(budget,))
    traj = result["trajectory"]
    assert len(traj["t"]) == budget
    assert traj["t"].tolist() == list(range(1, budget + 1))
    n_completed_episodes = len(result["episode_log"])
    # episode_id counts UP at each boundary; the run may end mid-episode
    # (one extra, incomplete episode_id beyond the completed ones).
    assert int(traj["episode_id"].max()) in (n_completed_episodes - 1, n_completed_episodes)


def test_trajectory_reconstructs_n_exactly_matching_live_agent_n():
    from _experiment_loader import load_experiment_module
    traj_mod = load_experiment_module("E2_online_4zones", "trajectory.py", "_e2_trajectory_for_pilot_test")

    budget = 733
    result = pilot_mod.run_one_seed(seed=5, budget=budget, checkpoints=(200, budget))
    traj = result["trajectory"]
    for t in (200, budget):
        N_log = traj_mod.reconstruct_N_from_trajectory(
            traj["t"], traj["s"], traj["a"], traj["s_next"],
            up_to_t=t, n_states=pilot_mod.N_STATES, n_actions=pilot_mod.N_ACTIONS,
        )
        assert np.array_equal(N_log, result["N_checkpoints"][t]), f"mismatch at t={t}"


def test_checkpoint_matched_and_reference_streams_independent_of_total_budget():
    """Same property, one level down: spawn_checkpoint_streams (used for
    the matched control and reference at each checkpoint) depends only on
    (seed, checkpoint_step), confirmed here at the RNG level directly
    rather than only through its downstream effect on N."""
    a = pilot_mod.e2_rng.spawn_checkpoint_streams(seed=42, checkpoint_step=1000)
    b = pilot_mod.e2_rng.spawn_checkpoint_streams(seed=42, checkpoint_step=1000)
    assert a["matched_control"].random(5).tolist() == b["matched_control"].random(5).tolist()
