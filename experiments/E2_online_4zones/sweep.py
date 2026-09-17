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
import time
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
from helplessness.runs import make_run_dir, write_run_meta

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


def run_condition_seed(condition: dict, seed_index: int, budget: int = BUDGET, checkpoints: tuple[int, ...] = CHECKPOINTS) -> dict:
    """One (condition, seed) job: online training + full measurement
    chain (online + matched) at every checkpoint. Self-contained -- takes
    only (condition, seed_index), builds everything else from scratch, so
    it can run in any process, in any order, and a lone rerun reproduces
    the exact same artifacts as inside the parallel pool."""
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
    episode_log: list[tuple[int, bool, bool]] = []  # (t, terminated, truncated)
    plan_n_iters: list[int] = []
    departure_counts = np.zeros(N_STATES, dtype=np.int64)
    departure_pair_counts = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    is_departure = [True]  # the very first transition of the run is a departure

    real_plan = ag.plan

    def counting_plan(tiebreak_rng):
        real_plan(tiebreak_rng)
        plan_n_iters.append(ag.last_plan_n_iter)

    ag.plan = counting_plan

    def on_transition(t, s, a, r, s_next, terminated, truncated):
        if is_departure[0]:
            departure_counts[s] += 1
            departure_pair_counts[s, a] += 1
        is_departure[0] = terminated or truncated
        if terminated or truncated:
            episode_log.append((t, bool(terminated), bool(truncated)))
        if t in checkpoint_set:
            N_checkpoints[t] = ag.N.copy()

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
    arrays = {
        "departure_counts": departure_counts,
        "departure_pair_fraction": departure_fraction,
        "episode_lengths": np.array(lengths, dtype=np.int64),
    }
    return {"per_state_rows": per_state_rows, "per_pair_rows": per_pair_rows, "summary": summary, "arrays": arrays}


def run_and_save(condition: dict, seed_index: int) -> tuple[str, float]:
    result = run_condition_seed(condition, seed_index)
    run_name = f"{condition['key']}_seed{seed_index:02d}"
    run_dir = make_run_dir(RESULTS_ROOT, EXPERIMENT_ID, run_name)

    artifact_mod.write_per_state_artifact(result["per_state_rows"], run_dir / "per_state.csv")
    artifact_mod.write_per_pair_artifact(result["per_pair_rows"], run_dir / "per_pair.csv")
    np.savez_compressed(run_dir / "arrays.npz", **result["arrays"])
    with open(run_dir / "summary.json", "w") as f:
        json.dump(result["summary"], f, indent=2)
    write_run_meta(run_dir, seed=seed_index, status="done", **result["summary"])

    return run_name, result["summary"]["elapsed_seconds"]


def all_jobs() -> list[tuple[dict, int]]:
    return [(cond, s) for cond in CONDITIONS for s in range(N_SEEDS)]


def main(max_workers: int = 8) -> None:
    jobs = all_jobs()
    print(f"Launching {len(jobs)} jobs across {max_workers} worker processes...")
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_and_save, cond, s): (cond["key"], s) for cond, s in jobs}
        n_done = 0
        for fut in concurrent.futures.as_completed(futures):
            key, s = futures[fut]
            try:
                run_name, elapsed = fut.result()
                n_done += 1
                print(f"[{n_done}/{len(jobs)}] done: {run_name} ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"[FAILED] {key} seed={s}: {exc!r}")
    print(f"Sweep finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
