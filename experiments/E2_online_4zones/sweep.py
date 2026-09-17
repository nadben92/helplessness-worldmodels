"""Temps 3 -- full sweep. 4 conditions x 20 seeds = 80 independent runs,
parallelised across worker processes (concurrent.futures.ProcessPoolExecutor).

Conditions: alpha in {0.5, 1, 5} at epsilon=0.1 (online + matched), plus
alpha=1 at epsilon=1.0 (online + matched, the bridge to E1). Seeds:
20 per condition, quadrant assignment counterbalanced as a 4-rotation x
5-seed Latin square (env.py::rotation_for_seed_index).

Every RNG stream is derived from (seed_index, alpha, epsilon[, checkpoint])
via rng.py's condition-aware spawn functions -- no two of the 80 runs, and
no two checkpoints within a run, can correlate, and the result of any one
(condition, seed) job is independent of which others ran alongside it or
in what order (needed for parallelisation and for a lone rerun to
reproduce the sweep's own artifacts exactly).

At every checkpoint, BOTH the matched control (drawn from P_true, same
N(t) as the online run) and the matched reference (drawn from P_ref, the
SAME realisation reused for online and matched, per spec) are recomputed
from N(t) -- never accumulated across checkpoints.

Writes results/E2_online_4zones_sweep/<condition_key>_seed<NN>/
(per_state.csv, per_pair.csv, summary.json, meta.yaml).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

import agent as e2_agent
import artifact as artifact_mod
import env as e2_env
import loop as e2_loop
import matched_control as matched_mod
import measurement as measurement_mod
import rng as rng_mod
import twin as twin_mod

from helplessness.config import load_yaml_block, REPO_ROOT
from helplessness.envs.zones import geographic_support_mask
from helplessness.runs import write_run_meta

CFG = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE = tuple(CFG["grid_shape"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
N_ACTIONS = 5
GOAL_S = e2_env.goal_state(CFG)
NON_TERMINAL_STATES = np.array([s for s in range(N_STATES) if s != GOAL_S])
GOAL_REWARD = 50.0
GAMMA = 0.9
H = 36
BUDGET = 1_000_000

# Same 16 log-spaced checkpoints as the pilot (Temps 2, validated).
CHECKPOINTS = tuple(sorted(set(int(round(x)) for x in np.logspace(3, 6, 16))))

CONDITIONS = (
    {"alpha": 0.5, "epsilon": 0.1, "key": "a0.5_e0.1"},
    {"alpha": 1.0, "epsilon": 0.1, "key": "a1_e0.1"},
    {"alpha": 5.0, "epsilon": 0.1, "key": "a5_e0.1"},
    {"alpha": 1.0, "epsilon": 1.0, "key": "a1_e1.0_bridge"},
)
N_SEEDS = 20
SEEDS_PER_ROTATION = 5

RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_ID = "E2_online_4zones_sweep"


def run_condition_seed(
    condition: dict, seed_index: int, budget: int = BUDGET, checkpoints: tuple[int, ...] = CHECKPOINTS,
    log_trajectory: bool = False,
) -> dict:
    """One (condition, seed) job: online training + full measurement
    chain (online + matched) at every checkpoint. Self-contained -- takes
    only (condition, seed_index), builds everything else from scratch, so
    it can run in any process, in any order, and a lone rerun reproduces
    the exact same artifacts as inside the parallel pool.

    log_trajectory=True (off by default -- the real sweep never sets it,
    see README.md's "regenerating a full trajectory" section) additionally
    builds and returns the full step-level (t, s, a, r, s_next, terminated,
    truncated) record, exactly like pilot.py's own trajectory logging.
    Kept in THIS function (not a separate duplicated setup) so the
    trajectory-logged run is built from the identical env/agent/RNG-stream
    construction the real sweep run used -- no risk of the two drifting
    apart. Since every RNG stream is a pure function of
    (seed_index, alpha, epsilon[, checkpoint]) (rng.py), calling this with
    log_trajectory=True for a (condition, seed_index) already in the
    sweep reproduces that exact run's trajectory bit-for-bit.
    """
    alpha, epsilon, cond_key = condition["alpha"], condition["epsilon"], condition["key"]
    rotation = e2_env.rotation_for_seed_index(seed_index, SEEDS_PER_ROTATION)
    assignment = e2_env.quadrant_assignment_for_rotation(rotation)

    layout = e2_env.build_layout(CFG, DIRECTION, quadrant_assignment=assignment)
    ctrl_layout = e2_env.build_control_layout(CFG, DIRECTION, quadrant_assignment=assignment)
    P_true = e2_env.build_env(layout, CFG, goal_reward=GOAL_REWARD).transition_tensor()
    P_ref = e2_env.build_env(ctrl_layout, CFG, goal_reward=GOAL_REWARD).transition_tensor()
    R_true, terminal = e2_env.reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
    support = geographic_support_mask(GRID_SHAPE)

    streams = rng_mod.spawn_streams(seed_index, alpha=alpha, epsilon=epsilon)
    env = e2_env.build_env(
        layout, CFG, goal_reward=GOAL_REWARD, max_steps=H,
        seed=int(streams["env_dynamics"].integers(2**31)),
    )
    ag = e2_agent.OnlineTabularAgentKnownReward(
        n_states=N_STATES, n_actions=N_ACTIONS, R_true=R_true, terminal_true=terminal,
        support=support, alpha=alpha, gamma=GAMMA,
    )

    checkpoint_set = set(checkpoints)
    N_checkpoints: dict[int, np.ndarray] = {}
    N_start_checkpoints: dict[int, np.ndarray] = {}  # departure-only counts, snapshotted at each checkpoint
    episode_log: list[tuple[int, bool, bool]] = []  # (t, terminated, truncated)
    episode_start_states: list[int] = []  # one entry per episode, in order -- NOT a full step-level trajectory
    plan_n_iters: list[int] = []
    departure_counts = np.zeros(N_STATES, dtype=np.int64)
    departure_pair_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    is_departure = [True]  # the very first transition of the run is a departure

    traj = None
    if log_trajectory:
        traj = {
            "t": np.zeros(budget, dtype=np.int32),
            "episode_id": np.zeros(budget, dtype=np.int32),
            "s": np.zeros(budget, dtype=np.uint16),
            "a": np.zeros(budget, dtype=np.uint8),
            "r": np.zeros(budget, dtype=np.float32),
            "s_next": np.zeros(budget, dtype=np.uint16),
            "terminated": np.zeros(budget, dtype=bool),
            "truncated": np.zeros(budget, dtype=bool),
        }
        traj_episode_counter = [0]

    real_plan = ag.plan

    def counting_plan(tiebreak_rng):
        real_plan(tiebreak_rng)
        plan_n_iters.append(ag.last_plan_n_iter)

    ag.plan = counting_plan

    def on_transition(t, s, a, r, s_next, terminated, truncated):
        if is_departure[0]:
            departure_counts[s] += 1
            departure_pair_counts[s, a] += 1
            episode_start_states.append(s)
        is_departure[0] = terminated or truncated
        if terminated or truncated:
            episode_log.append((t, bool(terminated), bool(truncated)))
        if traj is not None:
            idx = t - 1
            traj["t"][idx] = t
            traj["episode_id"][idx] = traj_episode_counter[0]
            traj["s"][idx] = s
            traj["a"][idx] = a
            traj["r"][idx] = r
            traj["s_next"][idx] = s_next
            traj["terminated"][idx] = terminated
            traj["truncated"][idx] = truncated
            if terminated or truncated:
                traj_episode_counter[0] += 1
        if t in checkpoint_set:
            # Full (S, A, S) table, captured directly off the live agent
            # -- never recomputed -- so downstream analysis (T_hat, the
            # jumeau, the twin pipeline) can be rerun on this exact data
            # without regenerating the run. N_start is the SAME shape but
            # restricted to (s, a) pairs where s was an episode's
            # DEPARTURE state -- checks that the visitation floor comes
            # from departures, not exploitation traffic through a state.
            N_checkpoints[t] = ag.N.copy()
            N_start_checkpoints[t] = departure_pair_counts.copy()

    t0 = time.time()
    e2_loop.train_online_e2(
        env, ag, n_steps=budget, epsilon=epsilon, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=streams["explore"], tiebreak_rng=streams["tiebreak"], start_rng=streams["start_state"],
        on_transition=on_transition,
    )
    elapsed = time.time() - t0

    per_state_rows: list[dict] = []
    per_pair_rows: list[dict] = []
    for t in sorted(N_checkpoints):
        N_online = N_checkpoints[t]
        N_target = N_online.sum(axis=2)

        cp_streams_online = rng_mod.spawn_checkpoint_streams(seed_index, t, alpha=alpha, epsilon=epsilon)
        online_result = twin_mod.twin_from_counts(
            N=N_online, alpha=alpha, support=support, P_true=P_true, R_true=R_true, terminal=terminal,
            P_ref=P_ref, gamma=GAMMA, reference_rng=cp_streams_online["reference"],
        )
        matched_N = matched_mod.collect_matched(P_true, N_target, cp_streams_online["matched_control"])

        # Same reference realisation for online and matched (spec): a
        # FRESH spawn_checkpoint_streams call with identical arguments
        # reproduces the identical "reference" stream state, since
        # spawn_checkpoint_streams is a pure function of its arguments.
        cp_streams_matched = rng_mod.spawn_checkpoint_streams(seed_index, t, alpha=alpha, epsilon=epsilon)
        matched_result = twin_mod.twin_from_counts(
            N=matched_N, alpha=alpha, support=support, P_true=P_true, R_true=R_true, terminal=terminal,
            P_ref=P_ref, gamma=GAMMA, reference_rng=cp_streams_matched["reference"],
        )

        meta_online = {
            "exp": "E2", "env": "exp", "condition": "online", "alpha": alpha, "epsilon": epsilon,
            "seed": seed_index, "assignment": rotation, "checkpoint": t,
        }
        meta_matched = {**meta_online, "condition": "matched"}
        per_state_rows += measurement_mod.per_state_rows(online_result, layout, GRID_SHAPE, meta_online)
        per_state_rows += measurement_mod.per_state_rows(matched_result, layout, GRID_SHAPE, meta_matched)
        per_pair_rows += measurement_mod.per_pair_rows(online_result, layout, GRID_SHAPE, N_ACTIONS, meta_online)
        per_pair_rows += measurement_mod.per_pair_rows(matched_result, layout, GRID_SHAPE, N_ACTIONS, meta_matched)

    boundaries = [t for t, _, _ in episode_log]
    lengths = np.diff([0] + boundaries).tolist() if boundaries else []
    n_episodes = len(episode_log)
    n_truncated = sum(1 for _, _, truncated in episode_log if truncated)
    final_N_pair = N_checkpoints[max(N_checkpoints)].sum(axis=2) if N_checkpoints else np.zeros((N_STATES, N_ACTIONS))
    with np.errstate(divide="ignore", invalid="ignore"):
        departure_fraction = np.where(final_N_pair > 0, departure_pair_counts / np.maximum(final_N_pair, 1), np.nan)

    summary = {
        "condition_key": cond_key, "alpha": alpha, "epsilon": epsilon, "seed": seed_index,
        "assignment": rotation, "elapsed_seconds": elapsed,
        "n_episodes": n_episodes, "n_truncated": n_truncated,
        "truncation_rate": (n_truncated / n_episodes) if n_episodes else float("nan"),
        "mean_episode_length": float(np.mean(lengths)) if lengths else float("nan"),
        "n_replans": len(plan_n_iters),
        "plan_n_iter_mean": float(np.mean(plan_n_iters)) if plan_n_iters else float("nan"),
        "plan_n_iter_max": int(max(plan_n_iters)) if plan_n_iters else 0,
    }
    # episode_start_states has one entry per episode STARTED, including a
    # trailing incomplete one if the run ends mid-episode -- truncate to
    # n_episodes (episodes actually COMPLETED, with a real length) so it
    # aligns 1:1 with episode_lengths/episode_truncated_flags.
    episode_truncated_flags = np.array([tr for _, _, tr in episode_log], dtype=bool)
    checkpoint_steps_sorted = sorted(N_checkpoints)
    arrays = {
        "departure_counts": departure_counts,
        "departure_pair_fraction": departure_fraction,
        "episode_lengths": np.array(lengths, dtype=np.int64),
        "episode_start_state": np.array(episode_start_states[:n_episodes], dtype=np.int64),
        "episode_truncated": episode_truncated_flags,
        "checkpoint_steps": np.array(checkpoint_steps_sorted, dtype=np.int64),
        # Full (S, A, S) counts table per checkpoint, int32 (a single
        # deterministic pair's count can exceed uint16's 65535 ceiling at
        # budget=1e6 -- int32 is comfortably safe) -- lets any downstream
        # analysis (T_hat, the jumeau, a different alpha's twin) be rerun
        # on this exact data without regenerating the run. N_start is the
        # (S, A) subset of those same counts restricted to transitions
        # that were an episode's DEPARTURE step, snapshotted the same way.
        **{f"N_checkpoint_{t}": N_checkpoints[t].astype(np.int32) for t in checkpoint_steps_sorted},
        **{f"N_start_checkpoint_{t}": N_start_checkpoints[t] for t in checkpoint_steps_sorted},
    }
    result = {"per_state_rows": per_state_rows, "per_pair_rows": per_pair_rows, "summary": summary, "arrays": arrays}
    if traj is not None:
        result["trajectory"] = traj
    return result


def _run_name(condition: dict, seed_index: int) -> str:
    return f"{condition['key']}_seed{seed_index:02d}"


def _final_dir(run_name: str) -> Path:
    return RESULTS_ROOT / EXPERIMENT_ID / run_name


def run_and_save(condition: dict, seed_index: int) -> tuple[str, float | None, str]:
    """Self-contained, resumable unit of work: skips outright if its
    final directory already exists (a previous sweep invocation
    completed it), otherwise computes into a uniquely-named temp
    directory and commits it to its final name with ONE atomic
    os.replace() call. A process killed at any point before that call
    (including mid-write of any individual file) leaves only the temp
    directory behind -- the final directory name never exists until
    every file in it is already fully written, so re-running the same
    sweep command can never mistake a partial run for a completed one,
    and always resumes from exactly where it left off. Returns
    (run_name, elapsed_seconds_or_None, "done"|"skipped")."""
    run_name = _run_name(condition, seed_index)
    final_dir = _final_dir(run_name)
    if final_dir.exists():
        return run_name, None, "skipped"

    results_dir = RESULTS_ROOT / EXPERIMENT_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / f".tmp_{run_name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=False)

    # BUDGET/CHECKPOINTS passed explicitly (read as module globals HERE,
    # at call time) rather than relying on run_condition_seed's own
    # default parameter values, which are bound once at function
    # DEFINITION time -- a module-level override (e.g. a test driver
    # monkeypatching sweep.BUDGET for a small, fast resumability check)
    # would silently have no effect on an already-bound default.
    result = run_condition_seed(condition, seed_index, budget=BUDGET, checkpoints=CHECKPOINTS)

    artifact_mod.write_per_state_artifact(result["per_state_rows"], tmp_dir / "per_state.csv")
    artifact_mod.write_per_pair_artifact(result["per_pair_rows"], tmp_dir / "per_pair.csv")
    np.savez_compressed(tmp_dir / "arrays.npz", **result["arrays"])
    with open(tmp_dir / "summary.json", "w") as f:
        json.dump(result["summary"], f, indent=2)
    # result["summary"] already carries "seed" (see run_condition_seed) --
    # pass status separately, everything else via the summary dict only.
    write_run_meta(tmp_dir, status="done", **result["summary"])

    # os.replace (not os.rename): atomic on both POSIX and Windows, and
    # succeeds even if final_dir were somehow already created by another
    # process in the meantime (harmless -- both results are byte-identical,
    # this job's RNG streams depend only on (condition, seed_index)).
    os.replace(tmp_dir, final_dir)

    return run_name, result["summary"]["elapsed_seconds"], "done"


def all_jobs() -> list[tuple[dict, int]]:
    return [(cond, s) for cond in CONDITIONS for s in range(N_SEEDS)]


def _cleanup_stale_temp_dirs() -> None:
    """Removes .tmp_* leftovers from a previous interrupted invocation.
    Always safe to discard: the corresponding final directory does not
    exist (checked separately before any job is skipped), so that job
    will simply be recomputed from scratch -- never a partial result."""
    results_dir = RESULTS_ROOT / EXPERIMENT_ID
    if not results_dir.exists():
        return
    for stale in results_dir.glob(".tmp_*"):
        shutil.rmtree(stale, ignore_errors=True)


def main(max_workers: int = 8) -> None:
    _cleanup_stale_temp_dirs()
    jobs = all_jobs()
    already_done = [(cond, s) for cond, s in jobs if _final_dir(_run_name(cond, s)).exists()]
    to_run = [(cond, s) for cond, s in jobs if not _final_dir(_run_name(cond, s)).exists()]

    print(f"{len(jobs)} total jobs, {len(already_done)} already done (skipped), {len(to_run)} to run.")
    print(f"Parallel workers: {max_workers} (of {os.cpu_count()} logical CPUs available).")
    for cond, s in already_done:
        print(f"[skip] {_run_name(cond, s)} (already done)")

    t0 = time.time()
    n_remaining = len(to_run)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_and_save, cond, s): (cond["key"], s) for cond, s in to_run}
        for fut in concurrent.futures.as_completed(futures):
            key, s = futures[fut]
            try:
                run_name, elapsed, status = fut.result()
                n_remaining -= 1
                print(f"[{status}] {run_name} elapsed={elapsed:.1f}s remaining={n_remaining}")
            except Exception as exc:
                n_remaining -= 1
                print(f"[FAILED] {key} seed={s}: {exc!r} remaining={n_remaining}")
    print(f"Sweep pass finished in {time.time() - t0:.1f}s "
          f"({len(already_done)} skipped, {len(to_run)} run this invocation).")


if __name__ == "__main__":
    main()
