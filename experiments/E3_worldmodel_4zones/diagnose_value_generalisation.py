"""Read-only diagnostic of where the information that led the closed-loop agent's success came from, at checkpoints
where the prior put no mass on the goal (docs/e3/DECISIONS.md, 2026-10-05 (23), (24); rule of reading fixed in (24)).

For each point (run directory, checkpoint environment step c), with the weights saved at c and the replay as it was at
c (the prefix of the resume point's replay, remeasure_checkpoints.replay_prefix):
  - V: the critic on the posterior latents of real contexts (the run's real_context settings), averaged by state;
    Spearman correlation (with a 95 % bootstrap interval over states) between V and the distance to the goal under P --
    the expected number of steps to reach it under the optimal policy (stochastic shortest path, goal absorbing);
  - the share of greedy actions that are optimal under P (the agent's greedy policy read on those contexts,
    probe.agent_greedy_policy, its tie set counted uniformly), against the share a uniform choice would get;
  - the actions played over the interval ending at c (from the previous checkpoint): their distribution (chi-square
    against uniform) and the rate at which an action repeats the previous one within an episode.

Usage: python experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py --points <run dir>:<env step> ... --out-dir DIR [--device cuda]
Output: <out dir>/value_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.heads import Critic, GoalHead
from helplessness.experiment_loader import load_experiment_module

remeasure = load_experiment_module("E3_worldmodel_4zones", "remeasure_checkpoints.py", "value_diag_remeasure")
run_agent, run_mini = remeasure.run_agent, remeasure.run_mini
e3_probe, e3_env = run_agent.e3_probe, run_mini.e3_env
SEED = 20261006
N_BOOT = 2000


def distance_to_goal(P: np.ndarray, goal: int, tol: float = 1e-9) -> np.ndarray:
    """Expected number of steps to the goal under the optimal policy: T(goal) = 0, T(s) = 1 + min_a sum P(s, a, s') T(s')."""
    T = np.zeros(P.shape[0])
    for _ in range(100_000):
        new = 1.0 + (P @ T).min(axis=1)
        new[goal] = 0.0
        if np.max(np.abs(new - T)) < tol:
            return new
        T = new
    raise RuntimeError("distance to the goal did not converge")


def spearman_with_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    from scipy.stats import spearmanr

    rho = float(spearmanr(x, y).correlation)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"rho": rho, "ci95": [float(lo), float(hi)], "n_states": int(len(x))}


def actions_played(arr: dict, lo: int, hi: int) -> dict:
    """Actions stored over environment steps (lo, hi]: distribution, chi-square against uniform, repeat rate within episodes."""
    from scipy.stats import chisquare

    env_step = np.cumsum(~arr["is_first"])  # environment steps done when each observation was seen
    acting = ~arr["terminal"] & ~arr["truncated"]  # an observation an action was taken from
    sel = np.flatnonzero(acting & (env_step >= lo) & (env_step < hi))
    a = arr["a"][sel].astype(int)
    counts = np.bincount(a, minlength=5)
    prev = sel[(sel - 1 >= 0)]
    prev = prev[~arr["is_first"][prev]]
    repeat = float(np.mean(arr["a"][prev] == arr["a"][prev - 1])) if len(prev) else float("nan")
    return {"n": int(len(a)), "distribution": (counts / max(len(a), 1)).round(4).tolist(),
            "chi2_p_uniform": float(chisquare(counts).pvalue) if len(a) else float("nan"), "repeat_previous_action": repeat}


def diagnose_point(run_dir: Path, c: int, device_arg: str | None) -> dict:
    import yaml

    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition = int(meta["seed"]), meta["condition"]
    if device_arg is not None:
        cfg["runtime"]["device"] = device_arg
    device = run_mini.configure_runtime(cfg)
    records = {r["env_step"]: r for r in (json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines())}
    if c not in records:
        raise ValueError(f"{c} is not a checkpoint of {run_dir.name}")
    previous = max([e for e in records if e < c], default=0)
    resume = torch.load(run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    replay = remeasure.replay_prefix(pickle.loads(resume["replay"]), c + int(records[c]["episodes"]))
    arr = replay.export_compact()
    if device.type == "cuda":
        replay.to_device(device)
    world = run_mini.build_world(cfg, seed, condition)
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    acfg = run_agent.agent_config(cfg)
    state = torch.load(remeasure.weights_path(run_dir, c, int(meta["env_steps"])), map_location="cpu", weights_only=False)
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    goal_head = GoalHead(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
    goal_head.load_state_dict(state["goal_head"])
    critic = Critic(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
    critic.load_state_dict(state["critic"])
    for m in (rssm, goal_head, critic):
        m.to(device).eval()
    rng = np.random.default_rng(SEED)
    gen = torch.Generator(device=device).manual_seed(SEED)
    rc = run_mini.real_context_settings(cfg)
    with torch.no_grad():
        ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                          n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
        v_ctx = torch.cat([critic(ctx["h"][i:i + 4096], ctx["z"][i:i + 4096]) for i in range(0, ctx["h"].shape[0], 4096)]).cpu().numpy()
        greedy = e3_probe.agent_greedy_policy(rssm, goal_head, critic, ctx["h"], ctx["z"], ctx["states"], run_mini.N_STATES, acfg, gen)
    P = world.env.transition_tensor()
    goal = e3_env.goal_state()
    states = np.asarray(ctx["states"])
    counts = np.bincount(states, minlength=run_mini.N_STATES)
    V = np.full(run_mini.N_STATES, np.nan)
    V[counts > 0] = np.bincount(states, weights=v_ctx, minlength=run_mini.N_STATES)[counts > 0] / counts[counts > 0]
    dist = distance_to_goal(P, goal)
    keep = (counts > 0) & (np.arange(run_mini.N_STATES) != goal)
    from helplessness.planning.value_iteration import value_iteration

    R, terminal = e3_env.reward_and_terminal(acfg.goal_reward, acfg.step_reward)
    V_star, _ = value_iteration(P, R, terminal, gamma=acfg.gamma)
    Q_star = (P * (R[None, None, :] + acfg.gamma * np.where(terminal, 0.0, V_star)[None, None, :])).sum(-1)
    optimal = Q_star >= Q_star.max(-1, keepdims=True) - 1e-9
    probs = greedy["policy_probs"]
    share = float((probs[keep] * optimal[keep]).sum(-1).mean())
    chance = float(optimal[keep].mean(-1).mean())
    masses = [p["mass_on_goal"] for p in records[c]["goal_diagnostics"]["goal_pairs"]]
    return {"run": run_dir.name, "env_step": c, "condition": condition, "seed": seed, "weights": state.get("env_step"),
            "replay_observations": len(arr["s"]), "goal_pair_masses_at_c": masses,
            "spearman_V_vs_distance_to_goal": spearman_with_ci(V[keep], dist[keep], np.random.default_rng(SEED + 1)),
            "V_by_state_quantiles": np.nanquantile(V[keep], [0.05, 0.5, 0.95]).tolist(),
            "greedy_optimal_share": share, "uniform_optimal_share": chance, "greedy_tie_fraction_by_state": float((probs[keep].max(-1) < 1.0).mean()),
            "actions_played_interval": {"from": previous, "to": c, **actions_played(arr, previous, c)},
            "success_interval": {"successes": records[c]["successes"] - (records[previous]["successes"] if previous else 0),
                                 "episodes": records[c]["episodes"] - (records[previous]["episodes"] if previous else 0)}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--points", nargs="+", required=True, metavar="RUN_DIR:ENV_STEP")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"{args.out_dir} already holds files")
    results = []
    for point in args.points:
        run_dir, step = point.rsplit(":", 1)
        results.append(diagnose_point(Path(run_dir), int(step), args.device))
        print(json.dumps(run_mini._jsonable(results[-1])), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "value_diagnostics.json").write_text(json.dumps(run_mini._jsonable(results), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
