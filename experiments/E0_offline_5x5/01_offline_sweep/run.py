"""Offline pipeline (Protocol A): single-agent validation, then full sweep.

Protocol A = offline collection of a fixed budget of transitions with a fixed
collection policy, training of T_hat/R_hat, then evaluation with learning
disabled and the planner frozen. The online protocol (learning during
evaluation) is not implemented here -- see roadmap in CLAUDE.md.

"Reachable" convention used throughout (frac_visited, policy_agreement_reachable,
distinct_states_normalized): the set of non-terminal states (walls included).
A wall can never be the current state (the agent cannot enter it); a terminal
state (goal, trap in terminal mode) can also never be the state *acted from*,
since the episode ends as soon as it is reached. This convention generalises
to any layout, unlike a fixed denominator.

sweep() trains and saves each agent (raw counts, via agent.save()) in
<agents_dir>/budget<budget>/p<p>_seed<seed>.npz. This is what reanalyze.py
and experiments/02_swaps/run.py subsequently load without retraining.

Every run writes its resolved config next to the CSV it produced
(<out>.config.yaml) -- see src/helplessness/config.py.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from helplessness.agents.collect import collect, uniform_policy
from helplessness.agents.tabular import TabularModelBasedAgent
from helplessness.config import OfflineSweepConfig, dump_config, load_config, resolve_path
from helplessness.envs.legacy.gridworld import N_ACTIONS, GridWorld
from helplessness.evaluation.offline_support import (
    OFFLINE_METRICS_FIELDNAMES,
    agent_metrics_row,
    agent_path,
    compute_oracle,
    shared_context,
    write_sweep_readme,
)
from helplessness.measures.belief import coverage, model_error


# ----------------------------------------------------------------------
def validate(budgets=(1_000, 5_000, 20_000, 50_000), seed: int = 0, alpha: float = 0.1) -> None:
    """Single agent, p=0, a single seed. Checks the convergence of
    T_hat/V_learned towards the oracle as the collection budget grows.

    The budgets are nested draws (the same seed is reused at every budget):
    the dataset of one budget is a prefix of the next budget's, which
    isolates the effect of data size without adding independent sampling
    noise between the rows of the table.
    """
    from helplessness.planning.value_iteration import GAMMA

    env = GridWorld(p=0.0, seed=seed)
    T_true, R_true, terminal, V_star, Q_star, pi_star = compute_oracle(env)
    reachable = ~terminal

    print("policy agree: agent/oracle agreement restricted to reachable (non-terminal) states -- see module docstring.\n")
    print(f"{'budget':<10}{'||V-V*||inf':<14}{'TV(visited)':<14}{'coverage':<12}{'policy agree':<14}")

    last_agent = None
    last_data = None
    for budget in budgets:
        env.seed_dynamics(seed)
        rng = np.random.default_rng(seed)
        data = collect(env, uniform_policy, budget, rng)

        agent = TabularModelBasedAgent(env.n_states, N_ACTIONS, alpha=alpha)
        for s, a, r, s_next, _terminated in data:
            agent.observe(s, a, r, s_next)
        agent.plan(terminal, gamma=GAMMA)

        v_dist = float(np.max(np.abs(agent.V - V_star)))
        tv_visited, _tv_all = model_error(agent.T_hat, T_true, agent.visited_mask)
        cov = coverage(agent.visited_mask, terminal)
        agreement = float(np.mean(agent.policy[reachable] == pi_star[reachable]))

        print(f"{budget:<10}{v_dist:<14.4f}{tv_visited:<14.4f}{cov:<12.3f}{agreement:<14.3f}")
        last_agent, last_data = agent, data

    # ------------------------------------------------------------------
    # is the residual disagreement at the largest budget a value tie in Q*,
    # or a genuine error residual?
    # ------------------------------------------------------------------
    disagreeing = np.where(reachable & (last_agent.policy != pi_star))[0]
    print(f"\nstates in disagreement at budget={budgets[-1]}: {len(disagreeing)}")
    for s in disagreeing:
        q_hat = last_agent.Q[s]
        q_star = Q_star[s]
        top2_star = np.sort(q_star)[::-1][:2]
        gap = top2_star[0] - top2_star[1]
        verdict = "tie (arbitrary tie-break)" if gap < 1e-3 else f"REAL GAP ({gap:.4f})"
        print(f"  s={s:<3} Q_learned={np.round(q_hat, 4)}  Q*={np.round(q_star, 4)}  -> {verdict}")

    # ------------------------------------------------------------------
    # sensitivity to the Dirichlet prior: smaller alpha -> lower TV at the
    # same budget, since the prior biases towards uniform and only fades
    # asymptotically. Quick check, same data, different alphas.
    # ------------------------------------------------------------------
    agent_small_alpha = TabularModelBasedAgent(env.n_states, N_ACTIONS, alpha=0.01)
    for s, a, r, s_next, _terminated in last_data:
        agent_small_alpha.observe(s, a, r, s_next)
    tv_default, _ = model_error(last_agent.T_hat, T_true, last_agent.visited_mask)
    tv_small_alpha, _ = model_error(agent_small_alpha.T_hat, T_true, agent_small_alpha.visited_mask)
    print(
        f"\nprior sensitivity at budget={budgets[-1]}: "
        f"TV(visited) alpha=0.1 -> {tv_default:.4f}, alpha=0.01 -> {tv_small_alpha:.4f} "
        f"({'alpha=0.01 lower, as expected' if tv_small_alpha < tv_default else 'UNEXPECTED REVERSAL -- investigate'})"
    )


# ----------------------------------------------------------------------
def sweep(cfg: OfflineSweepConfig) -> None:
    """Sweeps p and seed at a fixed transition budget (identical in every
    condition). Trains and saves each agent (agent.save(), raw counts) in
    <agents_dir>/budget<budget>/, then writes one metrics row per agent to
    cfg.out, plus a metadata file <out>.README.txt and <out>.config.yaml."""
    from helplessness.planning.value_iteration import GAMMA

    out_path = resolve_path(cfg.out)
    agents_dir = resolve_path(cfg.agents_dir)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.path.join(agents_dir, f"budget{cfg.budget}"), exist_ok=True)

    ref_env = GridWorld(p=0.0, seed=0)
    terminal, R_true, oracle_return, trap_states, grid_shape, n_reachable = shared_context(ref_env)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OFFLINE_METRICS_FIELDNAMES)
        writer.writeheader()

        for p in cfg.p_values:
            T_true_p = GridWorld(p=p, seed=0).transition_tensor()
            for seed in cfg.seeds:
                # environment-dynamics RNG and collection RNG: derived from
                # the same seed but kept distinct (separate-RNGs invariant).
                train_env = GridWorld(p=p, seed=seed)
                train_env.seed_dynamics(seed)
                collect_rng = np.random.default_rng(seed + 1_000_000)

                data = collect(train_env, uniform_policy, cfg.budget, collect_rng)

                agent = TabularModelBasedAgent(train_env.n_states, N_ACTIONS, alpha=cfg.alpha)
                for s, a, r, s_next, _terminated in data:
                    agent.observe(s, a, r, s_next)
                agent.plan(terminal, gamma=GAMMA)
                agent.save(agent_path(agents_dir, cfg.budget, p, seed))

                row = agent_metrics_row(
                    agent, p, seed, cfg.budget, cfg.alpha, terminal, R_true, T_true_p,
                    oracle_return, trap_states, grid_shape, n_reachable,
                )
                writer.writerow(row)

    write_sweep_readme(out_path, cfg.budget, cfg.alpha, cfg.p_values, cfg.seeds)
    dump_config(cfg, out_path + ".config.yaml")
    print(
        f"wrote {len(cfg.p_values) * len(cfg.seeds)} rows to {out_path}, "
        f"agents saved in {agents_dir}/budget{cfg.budget}/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline pipeline (Protocol A).")
    parser.add_argument("--validate", action="store_true", help="run only the single-agent validation")
    parser.add_argument("--config", type=str, default=None, help="path to a config.yaml (defaults if omitted)")
    parser.add_argument("--budget", type=int, default=None, help="override cfg.budget")
    parser.add_argument("--out", type=str, default=None, help="override cfg.out")
    parser.add_argument("--agents-dir", type=str, default=None, help="override cfg.agents_dir")
    args = parser.parse_args()

    if args.validate:
        validate()
    else:
        cfg = load_config(OfflineSweepConfig, args.config)
        if args.budget is not None:
            cfg.budget = args.budget
        if args.out is not None:
            cfg.out = args.out
        elif args.config is None:
            cfg.out = f"results/offline_sweep_{cfg.budget}.csv"
        if args.agents_dir is not None:
            cfg.agents_dir = args.agents_dir
        sweep(cfg)
