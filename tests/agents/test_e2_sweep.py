"""Tests for experiments/E2_online_4zones/sweep.py: atomic, resumable
per-job writes.

A real OS-level interruption (kill -9 mid-write) was tried directly
against a ProcessPoolExecutor worker and found unreliable to time
precisely from outside: on Windows, a worker process can keep running
and complete its assigned job even after its parent process is killed
(ProcessPoolExecutor workers are not tied to the parent's lifetime the
way a naive external observer might expect), and matching the right PID
via a command-line grep is itself fragile (a kill command's own command
line can match its own search pattern). A controlled in-process
interruption (monkeypatching a write step to raise partway through) is
both more precise and fully deterministic, and directly verifies the
actual guarantee that matters: the final directory NEVER appears until
every file in it is fully written.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

sweep_mod = load_experiment_module("E2_online_4zones", "sweep.py", "_e2_sweep_for_sweep_test")


@pytest.fixture
def isolated_sweep(tmp_path, monkeypatch):
    """Points sweep.py's RESULTS_ROOT/EXPERIMENT_ID at a throwaway
    directory and shrinks the job to a tiny budget, so each test runs in
    a few seconds and never touches real sweep output."""
    monkeypatch.setattr(sweep_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(sweep_mod, "EXPERIMENT_ID", "sweep_test")
    monkeypatch.setattr(sweep_mod, "BUDGET", 3000)
    monkeypatch.setattr(sweep_mod, "CHECKPOINTS", (1500, 3000))
    return tmp_path


CONDITION = {"alpha": 1.0, "epsilon": 0.1, "key": "unit_test_condition"}


def test_episode_level_arrays_are_aligned_and_exclude_the_trailing_incomplete_episode(isolated_sweep):
    """episode_lengths/episode_start_state/episode_truncated must have
    the same length (one entry per COMPLETED episode) -- a run that ends
    mid-episode has one more entry in the internal episode_start_states
    list (the started-but-never-finished final episode) than in
    episode_log, which must be dropped, not misaligned against lengths."""
    result = sweep_mod.run_condition_seed(CONDITION, 0, budget=3000, checkpoints=(1500, 3000))
    arrays = result["arrays"]
    n = result["summary"]["n_episodes"]
    assert len(arrays["episode_lengths"]) == n
    assert len(arrays["episode_start_state"]) == n
    assert len(arrays["episode_truncated"]) == n
    assert arrays["episode_lengths"].sum() <= 3000  # completed episodes only, budget=3000


def test_full_n_table_saved_per_checkpoint(isolated_sweep):
    """The full (S, A, S) counts table must be persisted at every
    checkpoint -- not just the aggregated per_pair N -- so T_hat/the
    jumeau/a different alpha can be recomputed later without
    regenerating the run."""
    result = sweep_mod.run_condition_seed(CONDITION, 0, budget=3000, checkpoints=(1500, 3000))
    arrays = result["arrays"]
    assert arrays["checkpoint_steps"].tolist() == [1500, 3000]
    for t in (1500, 3000):
        N_t = arrays[f"N_checkpoint_{t}"]
        assert N_t.shape == (sweep_mod.N_STATES, sweep_mod.N_ACTIONS, sweep_mod.N_STATES)
        assert int(N_t.sum()) == t


def test_n_start_is_a_subset_of_n_at_every_checkpoint(isolated_sweep):
    """N_start[s,a] (departure-only counts) must never exceed N[s,a]
    (all counts) for any pair, at any checkpoint -- departures are a
    subset of total visits by construction."""
    result = sweep_mod.run_condition_seed(CONDITION, 0, budget=3000, checkpoints=(1500, 3000))
    arrays = result["arrays"]
    for t in (1500, 3000):
        N_pair = arrays[f"N_checkpoint_{t}"].sum(axis=2)
        N_start = arrays[f"N_start_checkpoint_{t}"]
        assert N_start.shape == (sweep_mod.N_STATES, sweep_mod.N_ACTIONS)
        assert (N_start <= N_pair).all()
        assert N_start.sum() > 0  # some departures must have occurred by t=1500


def test_full_n_table_and_n_start_survive_write_and_load(isolated_sweep):
    run_name, _, status = sweep_mod.run_and_save(CONDITION, 0)
    assert status == "done"
    final_dir = sweep_mod._final_dir(run_name)
    with sweep_mod.np.load(final_dir / "arrays.npz") as loaded:
        checkpoint_steps = loaded["checkpoint_steps"].tolist()
        assert len(checkpoint_steps) > 0
        for t in checkpoint_steps:
            assert f"N_checkpoint_{t}" in loaded
            assert f"N_start_checkpoint_{t}" in loaded
            N_pair = loaded[f"N_checkpoint_{t}"].sum(axis=2)
            assert (loaded[f"N_start_checkpoint_{t}"] <= N_pair).all()


def test_log_trajectory_does_not_perturb_the_run_and_reconstructs_n_checkpoints(isolated_sweep):
    """log_trajectory=True (used only for on-demand regeneration, see
    README.md, never by the real sweep) must not change the run's own
    RNG consumption or results -- the SAME (condition, seed_index) run,
    with vs without trajectory logging, must produce bit-identical
    N_checkpoints. Reconstructing N from the logged trajectory must
    match those N_checkpoints exactly."""
    from _experiment_loader import load_experiment_module

    traj_mod = load_experiment_module("E2_online_4zones", "trajectory.py", "_e2_trajectory_for_sweep_test")

    result_plain = sweep_mod.run_condition_seed(CONDITION, 0, budget=3000, checkpoints=(1500, 3000))
    result_logged = sweep_mod.run_condition_seed(CONDITION, 0, budget=3000, checkpoints=(1500, 3000), log_trajectory=True)

    for t in (1500, 3000):
        assert np.array_equal(
            result_plain["arrays"][f"N_checkpoint_{t}"], result_logged["arrays"][f"N_checkpoint_{t}"]
        )

    traj = result_logged["trajectory"]
    for t in (1500, 3000):
        N_log = traj_mod.reconstruct_N_from_trajectory(
            traj["t"], traj["s"], traj["a"], traj["s_next"], up_to_t=t,
            n_states=sweep_mod.N_STATES, n_actions=sweep_mod.N_ACTIONS,
        )
        assert np.array_equal(N_log.astype(np.int32), result_plain["arrays"][f"N_checkpoint_{t}"])


def test_track_agent_state_does_not_perturb_the_run(isolated_sweep):
    """track_agent_state=True (used only for the jumeau sample-verification
    rerun, see analysis.py -- never by the real sweep) must not change
    RNG consumption or results: N_checkpoints must be bit-identical with
    vs without it, same principle as log_trajectory's own test."""
    result_plain = sweep_mod.run_condition_seed(CONDITION, 0, budget=3000, checkpoints=(1500, 3000))
    result_tracked = sweep_mod.run_condition_seed(
        CONDITION, 0, budget=3000, checkpoints=(1500, 3000), track_agent_state=True,
    )
    for t in (1500, 3000):
        assert np.array_equal(
            result_plain["arrays"][f"N_checkpoint_{t}"], result_tracked["arrays"][f"N_checkpoint_{t}"]
        )
    agent_state = result_tracked["agent_state"]
    assert set(agent_state["V_checkpoints"].keys()) == {1500, 3000}
    assert set(agent_state["policy_checkpoints"].keys()) == {1500, 3000}
    for t in (1500, 3000):
        assert agent_state["last_replan_checkpoints"][t] < t
        assert agent_state["V_checkpoints"][t].shape == (sweep_mod.N_STATES,)
        assert agent_state["policy_checkpoints"][t].shape == (sweep_mod.N_STATES,)


def test_run_and_save_creates_the_final_directory_on_success(isolated_sweep):
    run_name, elapsed, status = sweep_mod.run_and_save(CONDITION, 0)
    assert status == "done"
    assert elapsed is not None
    final_dir = sweep_mod._final_dir(run_name)
    assert final_dir.exists()
    assert (final_dir / "per_state.csv").exists()
    assert (final_dir / "per_pair.csv").exists()
    assert (final_dir / "summary.json").exists()
    assert (final_dir / "meta.yaml").exists()
    # no leftover temp directory once the job succeeds
    assert list(sweep_mod._final_dir(run_name).parent.glob(".tmp_*")) == []


def test_run_and_save_skips_when_final_directory_already_exists(isolated_sweep):
    run_name, _, status1 = sweep_mod.run_and_save(CONDITION, 1)
    assert status1 == "done"
    run_name2, elapsed2, status2 = sweep_mod.run_and_save(CONDITION, 1)
    assert run_name == run_name2
    assert status2 == "skipped"
    assert elapsed2 is None


def test_interrupted_write_never_produces_a_final_directory(isolated_sweep, monkeypatch):
    """The core atomicity guarantee: an exception raised PARTWAY through
    the write sequence (after per_state.csv is written, before
    per_pair.csv) must leave the final directory absent -- only a temp
    directory with partial content, which a resumed run must not
    mistake for "done"."""
    real_write_per_pair = sweep_mod.artifact_mod.write_per_pair_artifact

    def failing_write_per_pair(*args, **kwargs):
        raise RuntimeError("simulated interruption mid-write")

    monkeypatch.setattr(sweep_mod.artifact_mod, "write_per_pair_artifact", failing_write_per_pair)

    run_name = sweep_mod._run_name(CONDITION, 2)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        sweep_mod.run_and_save(CONDITION, 2)

    final_dir = sweep_mod._final_dir(run_name)
    assert not final_dir.exists(), "final directory must not exist after an interrupted write"

    results_dir = sweep_mod.RESULTS_ROOT / sweep_mod.EXPERIMENT_ID
    tmp_dirs = list(results_dir.glob(f".tmp_{run_name}_*"))
    assert len(tmp_dirs) == 1, "exactly one leftover temp directory expected"
    assert (tmp_dirs[0] / "per_state.csv").exists()  # written before the simulated failure
    assert not (tmp_dirs[0] / "per_pair.csv").exists()  # never reached


def test_resume_after_interruption_completes_and_cleans_up_stale_temp_dir(isolated_sweep, monkeypatch):
    """After an interrupted attempt (leaving a stale .tmp_ directory,
    final directory absent), calling main() again must: (1) clean up the
    stale temp directory, (2) NOT skip this job (final dir absent), (3)
    successfully complete it."""
    real_write_per_pair = sweep_mod.artifact_mod.write_per_pair_artifact

    def failing_write_per_pair(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(sweep_mod.artifact_mod, "write_per_pair_artifact", failing_write_per_pair)
    with pytest.raises(RuntimeError):
        sweep_mod.run_and_save(CONDITION, 3)
    monkeypatch.setattr(sweep_mod.artifact_mod, "write_per_pair_artifact", real_write_per_pair)

    run_name = sweep_mod._run_name(CONDITION, 3)
    results_dir = sweep_mod.RESULTS_ROOT / sweep_mod.EXPERIMENT_ID
    assert len(list(results_dir.glob(f".tmp_{run_name}_*"))) == 1  # stale temp dir present

    sweep_mod._cleanup_stale_temp_dirs()
    assert list(results_dir.glob(".tmp_*")) == []

    run_name2, elapsed2, status2 = sweep_mod.run_and_save(CONDITION, 3)
    assert status2 == "done"
    assert sweep_mod._final_dir(run_name2).exists()


def test_resumed_run_produces_identical_output_to_an_uninterrupted_run(isolated_sweep, tmp_path, monkeypatch):
    """The end-to-end reproducibility requirement: a job computed after
    being interrupted once must produce BYTE-IDENTICAL output to the
    same job computed in one uninterrupted call (same seed => same RNG
    streams => same result, per rng.py's design)."""
    # Reference: uninterrupted, in a separate isolated root.
    ref_root = tmp_path / "reference"
    ref_root.mkdir()
    monkeypatch.setattr(sweep_mod, "RESULTS_ROOT", ref_root)
    ref_name, _, ref_status = sweep_mod.run_and_save(CONDITION, 4)
    assert ref_status == "done"
    ref_dir = sweep_mod._final_dir(ref_name)
    ref_per_state = (ref_dir / "per_state.csv").read_text()
    ref_per_pair = (ref_dir / "per_pair.csv").read_text()

    # "Interrupted" (in the isolated_sweep root): fails once, then resumes.
    monkeypatch.setattr(sweep_mod, "RESULTS_ROOT", isolated_sweep)
    real_write_per_pair = sweep_mod.artifact_mod.write_per_pair_artifact

    def failing_once(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(sweep_mod.artifact_mod, "write_per_pair_artifact", failing_once)
    with pytest.raises(RuntimeError):
        sweep_mod.run_and_save(CONDITION, 4)
    monkeypatch.setattr(sweep_mod.artifact_mod, "write_per_pair_artifact", real_write_per_pair)
    sweep_mod._cleanup_stale_temp_dirs()

    resumed_name, _, resumed_status = sweep_mod.run_and_save(CONDITION, 4)
    assert resumed_status == "done"
    resumed_dir = sweep_mod._final_dir(resumed_name)

    assert (resumed_dir / "per_state.csv").read_text() == ref_per_state
    assert (resumed_dir / "per_pair.csv").read_text() == ref_per_pair


def test_main_skips_already_done_jobs_and_reports_remaining_count(isolated_sweep, monkeypatch, capsys):
    """Exercises main()'s real job-filtering/reporting logic. Substitutes
    ThreadPoolExecutor for ProcessPoolExecutor here ONLY: this test
    module is loaded dynamically by file path (see
    tests/_experiment_loader.py), under a synthetic module name that
    isn't importable by a spawned child process -- ProcessPoolExecutor
    would fail to pickle run_and_save for cross-process submission in
    THIS test harness specifically (the real sweep.py, run as
    `python sweep.py`, has no such issue: it's a normally-imported
    top-level module there). ThreadPoolExecutor implements the identical
    submit()/as_completed()/result() interface without cross-process
    pickling, so it exercises the exact same main() code path (job
    filtering, print statements, result collection) that matters here."""
    import concurrent.futures

    monkeypatch.setattr(sweep_mod, "CONDITIONS", (CONDITION,))
    monkeypatch.setattr(sweep_mod, "N_SEEDS", 2)
    monkeypatch.setattr(sweep_mod.concurrent.futures, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor)

    sweep_mod.main(max_workers=1)
    out1 = capsys.readouterr().out
    assert "0 already done" in out1
    assert "2 to run" in out1

    sweep_mod.main(max_workers=1)
    out2 = capsys.readouterr().out
    assert "2 already done" in out2
    assert "0 to run" in out2
