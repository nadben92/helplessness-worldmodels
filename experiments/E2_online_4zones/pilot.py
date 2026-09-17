"""Temps 2 -- pilot run: 3 seeds, alpha=1, epsilon=0.1, H=36, budget=1e6
steps/seed, logarithmically-spaced checkpoints. Identity quadrant
assignment (counterbalancing is a Temps-3/full-sweep concern, not
exercised here). Reports facts only (N distribution, episode outcomes,
value-iteration cost) -- no scientific interpretation, per the task's
own working rules.

Writes results/E2_online_4zones/pilot_seed<NN>/ (config.yaml, meta.yaml,
metrics -- via helplessness.runs, same convention as E1 onward) plus a
raw .npz per seed holding every checkpointed N[s,a,s'], the full episode
log, and the per-replan value-iteration iteration counts.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import agent as e2_agent
import env as e2_env
import loop as e2_loop
import rng as e2_rng

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

ALPHA = 1.0
EPSILON = 0.1
H = 36
GAMMA = 0.9
GOAL_REWARD = 50.0
BUDGET = 1_000_000
SEEDS = (0, 1, 2)

# ~16 points, log-spaced, 1e3 .. 1e6 -- a first pass, per Temps 2's own
# instruction to propose (not finalise) a checkpoint plan.
CHECKPOINTS = tuple(sorted(set(int(round(x)) for x in np.logspace(3, 6, 16))))

RESULTS_ROOT = REPO_ROOT / "results"
EXPERIMENT_ID = "E2_online_4zones"


def run_one_seed(seed: int, budget: int = BUDGET, checkpoints: tuple[int, ...] = CHECKPOINTS) -> dict:
    layout = e2_env.build_layout(CFG, DIRECTION)  # identity assignment
    R_true, terminal = e2_env.reward_and_terminal(CFG, goal_reward=GOAL_REWARD)
    support = geographic_support_mask(GRID_SHAPE)

    streams = e2_rng.spawn_streams(seed)
    env = e2_env.build_env(
        layout, CFG, goal_reward=GOAL_REWARD, max_steps=H,
        seed=int(streams["env_dynamics"].integers(2**31)),
    )
    agent = e2_agent.OnlineTabularAgentKnownReward(
        n_states=N_STATES, n_actions=N_ACTIONS, R_true=R_true, terminal_true=terminal,
        support=support, alpha=ALPHA, gamma=GAMMA,
    )

    checkpoint_set = set(checkpoints)
    N_checkpoints: dict[int, np.ndarray] = {}
    episode_log: list[tuple[int, bool]] = []  # (t_at_episode_end, success)
    plan_n_iters: list[int] = []

    real_plan = agent.plan

    def counting_plan(tiebreak_rng):
        real_plan(tiebreak_rng)
        plan_n_iters.append(agent.last_plan_n_iter)

    agent.plan = counting_plan

    def on_transition(t, s, a, r, s_next, terminated, truncated):
        if terminated or truncated:
            episode_log.append((t, bool(terminated)))
        if t in checkpoint_set:
            N_checkpoints[t] = agent.N.copy()

    t0 = time.time()
    e2_loop.train_online_e2(
        env, agent, n_steps=budget, epsilon=EPSILON, non_terminal_states=NON_TERMINAL_STATES,
        explore_rng=streams["explore"], tiebreak_rng=streams["tiebreak"], start_rng=streams["start_state"],
        on_transition=on_transition,
    )
    elapsed = time.time() - t0
    agent.plan(streams["tiebreak"])  # final replan matching the final N (see docs/E2_online_4zones.md)

    return {
        "seed": seed,
        "N_final": agent.N.copy(),
        "N_checkpoints": N_checkpoints,
        "episode_log": episode_log,
        "plan_n_iters": np.array(plan_n_iters, dtype=np.int64),
        "elapsed_seconds": elapsed,
        "layout": layout,
    }


def save_seed_results(result: dict, run_dir: Path) -> None:
    checkpoints_sorted = sorted(result["N_checkpoints"].keys())
    np.savez(
        run_dir / "raw.npz",
        N_final=result["N_final"],
        checkpoint_steps=np.array(checkpoints_sorted),
        **{f"N_checkpoint_{t}": result["N_checkpoints"][t] for t in checkpoints_sorted},
        episode_t=np.array([t for t, _ in result["episode_log"]]),
        episode_success=np.array([s for _, s in result["episode_log"]]),
        plan_n_iters=result["plan_n_iters"],
        elapsed_seconds=np.array(result["elapsed_seconds"]),
    )
    write_run_meta(
        run_dir, seed=result["seed"], status="done",
        alpha=ALPHA, epsilon=EPSILON, H=H, gamma=GAMMA, budget=BUDGET,
        elapsed_seconds=result["elapsed_seconds"],
    )


def main() -> None:
    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        result = run_one_seed(seed)
        run_dir = make_run_dir(RESULTS_ROOT, EXPERIMENT_ID, f"pilot_seed{seed:02d}")
        save_seed_results(result, run_dir)
        print(f"seed {seed}: elapsed={result['elapsed_seconds']:.1f}s, "
              f"n_episodes={len(result['episode_log'])}, "
              f"n_replans={len(result['plan_n_iters'])} -> {run_dir}")


if __name__ == "__main__":
    main()
