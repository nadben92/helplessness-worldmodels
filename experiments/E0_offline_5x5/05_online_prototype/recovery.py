"""Phase-2 recovery protocol on the E0 (5x5) environment -- see README.md in
this directory: PROTOTYPE, not a numbered paper experiment (see train.py's
docstring). Loads a Phase-1 snapshot (train.py), runs each of the four
recovery conditions (A/B/C/D = counts intact/rescaled x epsilon 0.1/1.0)
for n_phase2_steps transitions at p=0, learning on, with periodic
out-of-loop evaluation.

The env-agnostic mechanics (RecoveryTracker, recovery_time) live in
helplessness.agents.online.recovery -- this file supplies the GridWorld-
specific glue (make_env, snapshot loading, the CLI, CSV output) on top.

No burn-in here (see helplessness.agents.online.loop.train_online_with_burnin's
docstring): the exploration deadlock that the Phase-1 burn-in works around
is, in Phase 2, part of what is being measured -- an agent whose own policy
never revisits the states that would disconfirm its belief is exactly the
phenomenon under study (see CLAUDE.md, "Recovery" and Maier & Seligman's
relearning criterion).

Evaluation is out of loop by construction: RecoveryTracker's periodic check
runs on a fresh GridWorld instance with a fixed eval seed, calling the
generic evaluate() (learning-free, doesn't touch the training agent's
counts/policy beyond reading agent.policy) -- never the training env.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from helplessness.agents.online.agent import OnlineTabularAgent
from helplessness.agents.online.loop import train_online
from helplessness.agents.online.recovery import RecoveryTracker, recovery_time
from helplessness.config import RecoveryConfig, dump_config, load_config, resolve_path
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.evaluation.rollout import evaluate
from helplessness.planning.value_iteration import value_iteration
from train import git_commit_hash, snapshot_path

CONDITIONS = {
    "A": {"rescale": False, "epsilon": 0.1},
    "B": {"rescale": False, "epsilon": 1.0},
    "C": {"rescale": True, "epsilon": 0.1},
    "D": {"rescale": True, "epsilon": 1.0},
}

CURVE_FIELDNAMES = [
    "group", "budget", "seed", "condition", "t",
    "return_mean", "success_rate", "zero_sample_pairs", "state_entropy",
]
SUMMARY_FIELDNAMES = [
    "group", "budget", "seed", "condition", "recovery_time",
    "final_return_mean", "oracle_return_mean", "git_commit",
]


def make_env(cfg: RecoveryConfig, p: float, seed: int) -> GridWorld:
    return GridWorld(
        p=p, step_reward=-cfg.step_cost, goal_reward=cfg.goal_reward,
        trap_reward=-cfg.trap_cost, n_actions=cfg.n_actions, seed=seed,
    )


def compute_oracle_return(cfg: RecoveryConfig, n_episodes: int = 500, seed: int = 999_000) -> float:
    env = make_env(cfg, 0.0, seed)
    T, R, terminal = env.transition_tensor(), env.reward_vector(), env.terminal_mask()
    _V, pi_star = value_iteration(T, R, terminal, gamma=cfg.gamma)
    summary, _ = evaluate(env, pi_star, n_episodes=n_episodes, seed=seed + 1)
    return summary["return_mean"]


def run_condition(cfg: RecoveryConfig, snapshot_file: str, condition_name: str, seed: int) -> list[dict]:
    cond = CONDITIONS[condition_name]
    agent = OnlineTabularAgent.load(snapshot_file)
    if cond["rescale"]:
        agent.rescale_counts(cfg.n_cap)

    env = make_env(cfg, 0.0, seed)
    env.seed_dynamics(seed)
    non_terminal = ~env.terminal_mask()
    non_terminal_pairs = [(s, a) for s in range(env.n_states) if non_terminal[s] for a in range(env.n_actions)]

    eval_seed = seed + 9_000_000
    tracker = RecoveryTracker(
        agent,
        eval_env_factory=lambda: make_env(cfg, 0.0, eval_seed),
        eval_every=cfg.eval_every,
        n_eval_episodes=cfg.n_eval_episodes,
        eval_seed=eval_seed,
        non_terminal_pairs=non_terminal_pairs,
        mechanistic_window=cfg.mechanistic_window,
    )
    explore_rng = np.random.default_rng(seed + 6_000_000)
    tiebreak_rng = np.random.default_rng(seed + 7_000_000)
    train_online(
        env, agent, cfg.n_phase2_steps, cond["epsilon"], cfg.replan_every,
        explore_rng, tiebreak_rng, learning=True, on_transition=tracker,
    )
    return tracker.rows


def run_all(cfg: RecoveryConfig, groups: list, seeds: list[int], budgets: list[int]) -> tuple[list[dict], list[dict]]:
    """groups: p_train floats plus the literal string "naive". Returns
    (curve_rows, summary_rows)."""
    agents_dir = resolve_path(cfg.agents_dir)
    oracle_return = compute_oracle_return(cfg)
    commit = git_commit_hash()

    curve_rows, summary_rows = [], []
    for group in groups:
        for budget in ([None] if group == "naive" else budgets):
            for seed in seeds:
                # snapshot_path ignores its budget arg for group="naive" (no
                # budget dependence for that group -- see its docstring).
                path = snapshot_path(agents_dir, budget, group, seed)
                for condition_name in CONDITIONS:
                    rows = run_condition(cfg, path, condition_name, seed)
                    for row in rows:
                        row.update({"group": group, "budget": budget, "seed": seed, "condition": condition_name})
                        curve_rows.append(row)
                    rt = recovery_time(rows, oracle_return)
                    final_return = rows[-1]["return_mean"] if rows else float("nan")
                    summary_rows.append({
                        "group": group, "budget": budget, "seed": seed, "condition": condition_name,
                        "recovery_time": rt, "final_return_mean": final_return,
                        "oracle_return_mean": oracle_return, "git_commit": commit,
                    })
                    print(
                        f"group={group} budget={budget} seed={seed} cond={condition_name}: "
                        f"recovery_time={rt} final_return={final_return:.2f} (oracle {oracle_return:.2f})"
                    )
    return curve_rows, summary_rows


def write_csv(rows: list[dict], fieldnames: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online phase-2 recovery protocol.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--p-train", type=float, nargs="*", default=None, help="override cfg.p_train_values")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="override cfg.seeds")
    parser.add_argument("--budgets", type=int, nargs="*", default=None, help="override cfg.phase1_budgets")
    parser.add_argument("--no-naive", action="store_true", help="skip the naive group")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(RecoveryConfig, args.config)
    if args.p_train is not None:
        cfg.p_train_values = args.p_train
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.budgets is not None:
        cfg.phase1_budgets = args.budgets
    if args.out is not None:
        cfg.out = args.out

    groups = list(cfg.p_train_values) + ([] if args.no_naive else ["naive"])
    curve_rows, summary_rows = run_all(cfg, groups, cfg.seeds, cfg.phase1_budgets)

    out_path = resolve_path(cfg.out)
    summary_path = out_path.replace(".csv", "_summary.csv")
    write_csv(curve_rows, CURVE_FIELDNAMES, out_path)
    write_csv(summary_rows, SUMMARY_FIELDNAMES, summary_path)
    dump_config(cfg, out_path + ".config.yaml")
    print(f"wrote {len(curve_rows)} curve rows to {out_path}")
    print(f"wrote {len(summary_rows)} summary rows to {summary_path}")
