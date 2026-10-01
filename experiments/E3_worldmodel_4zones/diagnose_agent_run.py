"""Read-only diagnostics of a finished closed-loop run (run_agent.py), for the
agent_mini bench run recorded as failed (docs/e3/DECISIONS.md, 2026-10-01 (11)).
Nothing of the run is modified or re-run: the script reads its directory
(config.yaml, checkpoints.jsonl, counts.npz, checkpoints/resume.pt -- models
and replay at the last resume point -- and checkpoints/agent_final.pt), and
writes only into --out.

  a. goal frames: goal observations in the replay over environment steps, the
     positives a training batch holds (simulated window draws on the replay as
     it was), the positives the goal head saw in total, cumulative successes;
  b. goal head, independent of any threshold: p_goal on positive and negative
     frames (AUC, means, quantiles), mean p_goal over all frames against the
     true frequency of positives, on the replay (training data) and on fresh
     held-out episodes (uniform policy, own seeds), for each saved weight set;
  c. policy: Q of the five actions on latent states (replay and held-out), the
     gap max - second against the tie tolerance, which action wins (constant,
     state-dependent, context-dependent), the parts of Q (r_hat and gamma c V)
     that make the gap, and the actions played in the replay;
  d. success pooled over the episodes after the switch, Wilson interval,
     against the uniform policy's exact rate;
  e. condition B: the twin's L1 to P by zone against the counts.

Only two weight sets exist: the resume point (environment step 18,000) and the
end (20,000); run_agent.py saves no weights at the checkpoints.

Usage: python experiments/E3_worldmodel_4zones/diagnose_agent_run.py <run dir> [--out DIR] [--uniform-success 0.0459]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.closed_loop import non_terminal_states
from helplessness.agents.worldmodel.heads import Critic, GoalHead, continuation_from_goal, reward_from_goal
from helplessness.agents.worldmodel.loop import unroll_observation
from helplessness.agents.worldmodel.policy import lookahead, tie_tolerance
from helplessness.agents.worldmodel.rssm import LatentEstimator
from helplessness.experiment_loader import load_experiment_module

run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "diagnose_run_agent")
run_mini = run_agent.run_mini
e3_env = run_mini.e3_env

DIAG_SEED = 20261001  # the diagnostics' own streams (held-out episodes, window draws, latent draws)
N_HELD_OUT_EPISODES = 4000
N_Q_FRAMES = 3000
N_BATCH_DRAWS = 4000
TRUE_MDP: dict = {}  # P of the run's environment, set by main (for the comparison with the true MDP only)


# ---------------------------------------------------------------- data

def load(run_dir: Path) -> dict:
    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    import yaml

    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    world = run_mini.build_world(cfg, int(meta["seed"]), meta["condition"])
    acfg = run_agent.agent_config(cfg)
    resume = torch.load(run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    final = torch.load(run_dir / "checkpoints" / "agent_final.pt", map_location="cpu", weights_only=False)
    replay = pickle.loads(resume["replay"])
    records = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    return {"cfg": cfg, "meta": meta, "world": world, "acfg": acfg, "replay": replay.export_compact(), "records": records,
            "weights": {resume["progress"]["env_step"]: resume, int(final["env_step"]): final},
            "counts": dict(np.load(run_dir / "counts.npz"))}


def models(cfg: dict, acfg, state: dict):
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    goal = GoalHead(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
    goal.load_state_dict(state["goal_head"])
    critic = Critic(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
    critic.load_state_dict(state["critic"])
    for m in (rssm, goal, critic):
        m.eval()
    return rssm, goal, critic


def episodes_from_replay(arr: dict) -> list[dict]:
    """Split the stored observations into episodes: s (states seen, first to last), a (action taken at each but the
    last), r (arrival reward of each), env_step of each observation (environment steps done when it was seen)."""
    first = np.flatnonzero(arr["is_first"])
    bounds = list(first) + [len(arr["s"])]
    env_step = np.cumsum(~arr["is_first"])  # every non-first observation is one environment step
    out = []
    for i in range(len(first)):
        lo, hi = bounds[i], bounds[i + 1]
        out.append({"s": arr["s"][lo:hi].astype(int), "a": arr["a"][lo:hi].astype(int), "r": arr["r"][lo:hi].astype(float),
                    "env_step": env_step[lo:hi], "goal": bool(arr["terminal"][lo:hi].any()), "truncated": bool(arr["truncated"][lo:hi].any())})
    return out


def held_out_episodes(world, acfg, n: int, rng: np.random.Generator, max_steps: int) -> list[dict]:
    """Fresh episodes under the uniform policy (uniform non-terminal starts, H = max_steps), own dynamics seed."""
    env = world.env
    env.reset(seed=int(rng.integers(2**31)))
    starts = non_terminal_states(env)
    out = []
    for _ in range(n):
        _, info = env.reset(start_state=int(rng.choice(starts)))
        s, a, r = [info["state"]], [], [0.0]
        for _t in range(max_steps):
            act = int(rng.integers(env.n_actions))
            _, rew, term, trunc, info = env.step(act)
            a.append(act)
            s.append(info["state"])
            r.append(float(rew))
            if term:
                break
        a.append(0)
        out.append({"s": np.array(s), "a": np.array(a), "r": np.array(r), "goal": bool(r[-1] > 0)})
    return out


@torch.no_grad()
def posterior_states(rssm, world, episodes: list[dict], generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """The agent's posterior latent state at every observation of every episode, as its filter computes it (history
    from the episode start, straight-through posterior draws -- the estimator after the switch). Returns h, z, the
    state label and the arrival reward of every frame (flattened in episode order)."""
    L = max(len(e["s"]) for e in episodes)
    hs, zs, ss, rs = [], [], [], []
    for lo in range(0, len(episodes), 512):
        chunk = episodes[lo:lo + 512]
        S = np.zeros((len(chunk), L), dtype=int)
        A = np.zeros((len(chunk), L), dtype=int)
        first = np.zeros((len(chunk), L), dtype=bool)
        first[:, 0] = True
        for i, e in enumerate(chunk):
            S[i, :len(e["s"])] = e["s"]
            A[i, :len(e["a"])] = e["a"]
        x = world.templates_t[torch.as_tensor(S)]
        out = unroll_observation(rssm, x, torch.as_tensor(A), torch.as_tensor(first), estimator=LatentEstimator("straight_through"),
                                 generator=generator, decode=False)
        for i, e in enumerate(chunk):
            n = len(e["s"])
            hs.append(out["h"][i, :n])
            zs.append(out["z"][i, :n])
            ss.append(e["s"])
            rs.append(e["r"])
    return torch.cat(hs), torch.cat(zs), np.concatenate(ss), np.concatenate(rs)


# ---------------------------------------------------------------- statistics

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(p on a positive > p on a negative), ties count half (Mann-Whitney)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def quantiles(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(x) else [float("nan")] * 5
    return {"n": int(len(x)), "mean": float(np.mean(x)) if len(x) else float("nan"), **{f"q{int(k * 100):02d}": float(v) for k, v in
                                                                                          zip((0.05, 0.25, 0.5, 0.75, 0.95), q)}}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_two_sided(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p-value (sum of the probabilities no larger than P(k))."""
    from math import comb

    probs = np.array([comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)])
    return float(min(1.0, probs[probs <= probs[k] * (1 + 1e-9)].sum()))


# ---------------------------------------------------------------- the five diagnostics

def diag_a(d: dict) -> dict:
    arr, cfg = d["replay"], d["cfg"]
    B, T = cfg["train"]["batch_size"], cfg["train"]["seq_len"]
    env_step = np.cumsum(~arr["is_first"])
    goal = arr["terminal"].astype(bool)
    switch_grad = d["records"][-1]["switch_step"]
    # gradient steps start after the prefill, replay_ratio per environment step
    switch_env = cfg["budget"]["prefill_steps"] + switch_grad / cfg["train"]["replay_ratio"]
    rng = np.random.default_rng(DIAG_SEED)
    grid = sorted({int(math.ceil(switch_env)), *range(3000, int(env_step[-1]) + 1, 1000), int(env_step[-1])})
    rows = []
    for e in grid:
        n = int(np.searchsorted(env_step, e, side="right"))  # stored observations by environment step e
        starts = rng.integers(0, n - T + 1, size=(N_BATCH_DRAWS, B))
        win = starts[..., None] + np.arange(T)
        per_batch = goal[:n][win].sum(axis=(1, 2))
        rows.append({"env_step": e, "stored": n, "goal_frames": int(goal[:n].sum()), "goal_fraction": float(goal[:n].mean()),
                     "positives_per_batch_mean": float(per_batch.mean()), "P_batch_without_positive": float((per_batch == 0).mean()),
                     "positives_per_batch_q95": float(np.quantile(per_batch, 0.95))})
    # positives the goal head saw up to the resume point: replay_ratio batches per environment step after the switch,
    # each interval between grid points counted at its end point's mean (an upper bound: the replay only grows)
    seen = sum((b["env_step"] - a["env_step"]) * cfg["train"]["replay_ratio"] * b["positives_per_batch_mean"] for a, b in zip(rows, rows[1:]))
    cum = [{"env_step": r["env_step"], "episodes": r["episodes"], "successes": r["successes"]} for r in d["records"]]
    return {"batch_frames": B * T, "rows": rows, "positives_seen_by_goal_head_upper_bound": seen, "switch_grad_step": switch_grad,
            "switch_env_step": switch_env,
            "cumulative": cum, "goal_frames_total_at_resume_point": int(goal.sum()), "stored_at_resume_point": int(len(goal))}


@torch.no_grad()
def diag_b_c(d: dict) -> dict:
    cfg, acfg, world = d["cfg"], d["acfg"], d["world"]
    rng = np.random.default_rng(DIAG_SEED + 1)
    max_steps = run_mini.load_episode_config()["max_steps"]
    held = held_out_episodes(world, acfg, N_HELD_OUT_EPISODES, rng, max_steps)
    replay_eps = [e for e in episodes_from_replay(d["replay"])]
    goal_s = e3_env.goal_state()
    out = {"held_out_episodes": len(held), "held_out_goal_episodes": int(sum(e["goal"] for e in held)), "by_weights": {}}
    for env_step, state in sorted(d["weights"].items()):
        rssm, goal_head, critic = models(cfg, acfg, state)
        res = {}
        for name, eps in (("replay (training data, up to the resume point)", replay_eps), ("held-out uniform-policy episodes", held)):
            gen = torch.Generator().manual_seed(DIAG_SEED + 2)
            h, z, s, r = posterior_states(rssm, world, eps, gen)
            p = goal_head.p_goal(h, z).numpy()
            pos = r > 0
            res[name] = {"positives": quantiles(p[pos]), "negatives": quantiles(p[~pos]), "auc": auc(p[pos], p[~pos]),
                         "mean_p_all_frames": float(p.mean()), "true_positive_frequency": float(pos.mean()),
                         "negatives_on_goal_state": int(((s == goal_s) & ~pos).sum())}
            if name.startswith("held-out"):
                res["policy (held-out frames)"] = diag_c(rssm, goal_head, critic, acfg, h, z, s, r, np.random.default_rng(DIAG_SEED + 3))
        out["by_weights"][env_step] = res
    out["replay_actions"] = actions_played(d)
    return out


@torch.no_grad()
def diag_c(rssm, goal_head, critic, acfg, h, z, s, r, rng) -> dict:
    keep = np.flatnonzero(r <= 0)  # non-goal frames: where the agent acts
    idx = rng.choice(keep, size=min(N_Q_FRAMES, len(keep)), replace=False)
    gen = torch.Generator().manual_seed(DIAG_SEED + 4)
    h_next, z_next = lookahead(rssm, h[idx], z[idx], acfg.k_act, gen)
    K = acfg.k_act
    hk = h_next.unsqueeze(2).expand(-1, -1, K, -1)
    fh, fz = hk.reshape(-1, hk.shape[-1]), z_next.reshape(-1, *z_next.shape[-2:])
    p = goal_head.p_goal(fh, fz)
    v = critic(fh, fz)
    r_hat = reward_from_goal(p, acfg).reshape(len(idx), -1, K).mean(-1)
    cont = (acfg.gamma * continuation_from_goal(p) * v).reshape(len(idx), -1, K).mean(-1)
    Q = r_hat + cont
    tol = tie_tolerance(Q, acfg)
    srt = Q.sort(dim=-1, descending=True).values
    gap = (srt[:, 0] - srt[:, 1])
    tied = (Q >= Q.max(-1, keepdim=True).values - tol[:, None]).sum(-1) > 1
    arg = Q.argmax(-1).numpy()
    states = s[idx]
    # context dependence: for states read several times, does the argmax change between contexts?
    per_state = {}
    for st, a in zip(states, arg):
        per_state.setdefault(int(st), []).append(int(a))
    multi = {k: v for k, v in per_state.items() if len(v) >= 3}
    consistency = float(np.mean([np.bincount(v, minlength=5).max() / len(v) for v in multi.values()])) if multi else float("nan")
    modal = {k: int(np.bincount(v, minlength=5).argmax()) for k, v in multi.items()}
    # against the true MDP: is the argmax an optimal action under P, and does V follow V*?
    from helplessness.planning.value_iteration import value_iteration

    P = TRUE_MDP["P"]
    R, terminal = e3_env.reward_and_terminal(acfg.goal_reward, acfg.step_reward)
    V_star, _ = value_iteration(P, R, terminal, gamma=acfg.gamma)
    Q_star = (P * (R[None, None, :] + acfg.gamma * np.where(terminal, 0.0, V_star)[None, None, :])).sum(-1)
    optimal = Q_star >= Q_star.max(-1, keepdims=True) - 1e-9
    n_opt = optimal[states].sum(-1)
    goal_s = e3_env.goal_state()
    pre_goal = P[states, :, goal_s].max(-1) > 0  # frames one step from the goal
    to_goal = P[states, :, goal_s].argmax(-1)
    v_now = critic(h[idx], z[idx]).numpy()
    from scipy.stats import spearmanr

    near = {}
    if pre_goal.any():
        sel = np.flatnonzero(pre_goal)
        pg = p.reshape(len(idx), -1, K).mean(-1).numpy()
        near = {"n_frames": int(len(sel)),
                "p_goal_next_for_the_action_into_the_goal": quantiles(pg[sel, to_goal[sel]]),
                "p_goal_next_other_actions": quantiles(np.delete(pg[sel], to_goal[sel], axis=1).ravel() if len(sel) == 1 else
                                                       pg[sel][np.arange(5)[None, :] != to_goal[sel][:, None]]),
                "argmax_is_the_action_into_the_goal": float((arg[sel] == to_goal[sel]).mean()),
                "true_P_into_goal_for_that_action": quantiles(P[states[sel], to_goal[sel], goal_s])}
    vs_true = {"argmax_optimal_under_P": float(optimal[states, arg].mean()),
               "chance_level_uniform": float((n_opt / 5.0).mean()),
               "spearman_V_critic_vs_V_star": float(spearmanr(v_now, V_star[states]).correlation),
               "V_critic": quantiles(v_now), "V_star_of_these_states": quantiles(V_star[states]),
               "one_step_from_goal": near}
    return {"n_frames": int(len(idx)), "tie_fraction": float(tied.float().mean()), "against_true_MDP": vs_true,
            "gap_max_second": quantiles(gap.numpy()), "tolerance": quantiles(tol.numpy()),
            "gap_over_tolerance": quantiles((gap / tol).numpy()),
            "argmax_distribution": np.bincount(arg, minlength=5).tolist(),
            "states_with_3plus_reads": len(multi), "within_state_argmax_consistency": consistency,
            "distinct_modal_actions_over_states": int(len(set(modal.values()))),
            "modal_action_counts_over_states": np.bincount(list(modal.values()), minlength=5).tolist() if modal else [],
            "r_hat": {"mean": float(r_hat.mean()), "spread_over_actions": quantiles((r_hat.max(-1).values - r_hat.min(-1).values).numpy())},
            "gamma_c_V": {"mean": float(cont.mean()), "spread_over_actions": quantiles((cont.max(-1).values - cont.min(-1).values).numpy())},
            "p_goal_next": quantiles(p.numpy()), "V_next": quantiles(v.numpy())}


def actions_played(d: dict) -> dict:
    """Actions stored in the replay after the switch (greedy and exploring steps together): their distribution, how
    often an action repeats the previous one, and how often the state does not change (bumping into a wall / staying)."""
    arr = d["replay"]
    env_step = np.cumsum(~arr["is_first"])
    switch_env = next(r["env_step"] for r in d["records"] if r["switch_step"] is not None)
    out = {}
    for lo, hi in ((0, switch_env), (switch_env, 10000), (10000, int(env_step[-1]) + 1)):
        sel = np.flatnonzero((env_step >= lo) & (env_step < hi) & ~arr["terminal"] & ~arr["truncated"])
        sel = sel[sel + 1 < len(arr["s"])]
        sel = sel[~arr["is_first"][sel + 1]]  # the next observation is this episode's
        a, s, s_next = arr["a"][sel].astype(int), arr["s"][sel], arr["s"][sel + 1]
        prev_same = sel[sel - 1 >= 0]
        prev_same = prev_same[~arr["is_first"][prev_same]]
        out[f"env steps [{lo}, {hi})"] = {"n": int(len(sel)), "action_distribution": (np.bincount(a, minlength=5) / max(len(a), 1)).round(3).tolist(),
                                          "repeat_previous_action": float((arr["a"][prev_same] == arr["a"][prev_same - 1]).mean()) if len(prev_same) else float("nan"),
                                          "state_unchanged": float((s == s_next).mean())}
    return out


def diag_d(d: dict, p0: float) -> dict:
    recs = d["records"]
    first_after = next(r for r in recs if r["switch_step"] is not None)
    last = recs[-1]
    k, n = last["successes"] - first_after["successes"], last["episodes"] - first_after["episodes"]
    lo, hi = wilson(k, n)
    eps = episodes_from_replay(d["replay"])
    switch_env = first_after["env_step"]
    post = [e for e in eps if e["env_step"][0] >= switch_env and (e["goal"] or e["truncated"])]
    k2, n2 = sum(e["goal"] for e in post), len(post)
    lo2, hi2 = wilson(k2, n2)
    return {"from_records": {"from_env_step": first_after["env_step"], "to_env_step": last["env_step"], "successes": k, "episodes": n,
                             "rate": k / n if n else float("nan"), "wilson95": [lo, hi], "binomial_p_vs_uniform": binom_two_sided(k, n, p0)},
            "from_replay": {"from_env_step": switch_env, "to_env_step": int(np.cumsum(~d["replay"]["is_first"])[-1]), "successes": k2,
                            "episodes": n2, "rate": k2 / n2 if n2 else float("nan"), "wilson95": [lo2, hi2],
                            "binomial_p_vs_uniform": binom_two_sided(k2, n2, p0)},
            "uniform_policy_rate": p0}


def diag_e(d: dict) -> dict:
    out = {}
    for r in d["records"]:
        N = d["counts"][f"N_{r['env_step']}"]
        out[r["env_step"]] = {cat: {"l1": z["l1"], "visit_share": z["visit_share"]} for cat, z in r["twin_by_zone"].items()}
        out[r["env_step"]]["total_counts"] = int(N.sum())
    return out


def report(res: dict) -> str:
    return "```json\n" + json.dumps(run_mini._jsonable(res), indent=1) + "\n```\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: reports/diagnostics_<run dir name>/")
    parser.add_argument("--uniform-success", type=float, default=0.0459, help="the uniform policy's exact success rate (DECISIONS (7))")
    args = parser.parse_args(argv)
    out = args.out or run_mini.REPO_ROOT / "reports" / f"diagnostics_{args.run_dir.name}"
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already holds files")
    torch.set_num_threads(4)
    d = load(args.run_dir)
    TRUE_MDP["P"] = d["world"].env.transition_tensor()
    res = {"run": args.run_dir.name, "weights_available_at_env_steps": sorted(d["weights"]), "a": diag_a(d)}
    res["b_c"] = diag_b_c(d)
    res["d"] = diag_d(d, args.uniform_success)
    res["e"] = diag_e(d)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(run_mini._jsonable(res), indent=1), encoding="utf-8")
    print(json.dumps(run_mini._jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
