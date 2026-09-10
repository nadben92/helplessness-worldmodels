"""Entry point for Phase-1 online training on the E0 (5x5) environment --
see README.md in this directory: this is a PROTOTYPE, not a numbered paper
experiment. It exercises helplessness.agents.online (agent, loop, recovery)
against the legacy GridWorld while E1-E2's actual environment doesn't exist
yet; the library code it calls is what carries over.

Trains a single OnlineTabularAgent, serialises it as a snapshot (so
Phase 2 / recovery.py never retrains Phase 1 -- see CONTRIBUTING.md,
"results and configs"), evaluates it greedily against the oracle, and writes
one CSV row plus a config.yaml recording exactly what produced it (git
commit hash, seed, resolved config) -- run.py in the offline experiments
follow the same convention.

Run from this directory: `python train.py --p-train 0.0 --seed 1`
(or with --config for a full sweep config; --naive for the naive group).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess

import numpy as np

from helplessness.agents.online.agent import OnlineTabularAgent
from helplessness.agents.online.loop import train_online_with_burnin
from helplessness.config import OnlineTrainConfig, dump_config, load_config, resolve_path
from helplessness.envs.legacy.gridworld import GridWorld
from helplessness.evaluation.rollout import evaluate
from helplessness.planning.value_iteration import value_iteration


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), text=True
        ).strip()
    except Exception:
        return "unknown"


def make_env(cfg: OnlineTrainConfig, p: float, seed: int) -> GridWorld:
    return GridWorld(
        p=p,
        step_reward=-cfg.step_cost,
        goal_reward=cfg.goal_reward,
        trap_reward=-cfg.trap_cost,
        n_actions=cfg.n_actions,
        seed=seed,
    )


def snapshot_path(agents_dir: str, budget: int, p_train: float | str, seed: int) -> str:
    """p_train="naive" names the naive (phase-1-skipped) snapshot -- it has
    no budget dependence (see CLAUDE.md: naive gets no extra phase-2 budget
    to compensate, and never runs phase 1 at all), so it is not filed under
    a budget<N> subdirectory the way every real p_train snapshot is."""
    if p_train == "naive":
        return os.path.join(agents_dir, f"naive_seed{seed}.npz")
    return os.path.join(agents_dir, f"budget{budget}", f"p{p_train}_seed{seed}.npz")


def train_one(cfg: OnlineTrainConfig) -> dict:
    """Trains a single agent at cfg.p_train for cfg.n_steps transitions,
    saves its snapshot, then evaluates the resulting greedy policy at p=0
    (learning disabled, fresh env instance) against the oracle for the same
    cost structure and action set. Returns a metrics row."""
    train_env = make_env(cfg, cfg.p_train, cfg.seed)
    train_env.seed_dynamics(cfg.seed)
    agent = OnlineTabularAgent(
        train_env, alpha=cfg.alpha, gamma=cfg.gamma,
        use_eta_update=cfg.use_eta_update, eta=cfg.eta,
    )

    explore_rng = np.random.default_rng(cfg.seed + 1_000_000)
    tiebreak_rng = np.random.default_rng(cfg.seed + 2_000_000)
    burn_in_rng = np.random.default_rng(cfg.seed + 5_000_000)
    train_online_with_burnin(
        train_env, agent, cfg.n_steps, cfg.epsilon, cfg.replan_every, cfg.burn_in_steps,
        explore_rng, tiebreak_rng, burn_in_rng, learning=True,
    )

    agents_dir = resolve_path(cfg.agents_dir)
    os.makedirs(os.path.join(agents_dir, f"budget{cfg.n_steps}"), exist_ok=True)
    agent.save(snapshot_path(agents_dir, cfg.n_steps, cfg.p_train, cfg.seed))

    eval_env = make_env(cfg, 0.0, cfg.seed + 3_000_000)
    summary, _traj = evaluate(eval_env, agent.policy, n_episodes=500, seed=cfg.seed + 4_000_000)

    T_true = eval_env.transition_tensor()
    R_true = eval_env.reward_vector()
    terminal_true = eval_env.terminal_mask()
    _V_star, pi_star = value_iteration(T_true, R_true, terminal_true, gamma=cfg.gamma)
    oracle_summary, _ = evaluate(eval_env, pi_star, n_episodes=500, seed=cfg.seed + 4_000_000)

    return {
        "p_train": cfg.p_train,
        "seed": cfg.seed,
        "n_steps": cfg.n_steps,
        "return_mean": summary["return_mean"],
        "success_rate": summary["success_rate"],
        "truncation_rate": summary["truncation_rate"],
        "oracle_return_mean": oracle_summary["return_mean"],
        "oracle_success_rate": oracle_summary["success_rate"],
    }


def save_naive_snapshot(cfg: OnlineTrainConfig) -> str:
    """Saves the naive group's snapshot: an OnlineTabularAgent constructed
    but never trained (phase 1 skipped entirely -- see CLAUDE.md). Still
    goes through save()/load() like every other snapshot, so recovery.py
    does not special-case it beyond the path (see snapshot_path)."""
    env = make_env(cfg, 0.0, cfg.seed)
    agent = OnlineTabularAgent(env, alpha=cfg.alpha, gamma=cfg.gamma)
    agents_dir = resolve_path(cfg.agents_dir)
    os.makedirs(agents_dir, exist_ok=True)
    path = snapshot_path(agents_dir, cfg.n_steps, "naive", cfg.seed)
    agent.save(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online phase-1 training, single agent.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--p-train", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--naive", action="store_true", help="save a naive (untrained) snapshot instead of training")
    args = parser.parse_args()

    cfg = load_config(OnlineTrainConfig, args.config)
    if args.p_train is not None:
        cfg.p_train = args.p_train
    if args.seed is not None:
        cfg.seed = args.seed

    if args.naive:
        path = save_naive_snapshot(cfg)
        print(f"wrote naive snapshot: {path}")
        raise SystemExit(0)

    row = train_one(cfg)

    out_path = resolve_path(cfg.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()) + ["git_commit"])
        if write_header:
            writer.writeheader()
        writer.writerow({**row, "git_commit": git_commit_hash()})
    dump_config(cfg, out_path + ".config.yaml")

    print(f"p_train={cfg.p_train} seed={cfg.seed} n_steps={cfg.n_steps}")
    print(f"  return_mean      = {row['return_mean']:.4f}  (oracle {row['oracle_return_mean']:.4f})")
    print(f"  success_rate     = {row['success_rate']:.4f}  (oracle {row['oracle_success_rate']:.4f})")
    print(f"  truncation_rate  = {row['truncation_rate']:.4f}")
    print(f"wrote {out_path}")
