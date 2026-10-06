"""Measurement noise of I_s due to the choice of real contexts, and the goal control at REF 0 / 15,849
(docs/e3/DECISIONS.md, 2026-10-06 (26); read-only, nothing of the run is modified).

--noise: on a closed-loop run's final weights (agent_final.pt) and the replay of its last resume point
(checkpoints/resume.pt), I_s by zone measured --repeats times with independent draws of real contexts and of the
kernel's prior draws (seeds SEED + repeat), for each --contexts-multipliers m: real_context.max_per_state x m, the
kernel keeping its K draws per (s, a) spread over the state's contexts. I_s is the main measure's
(probe.real_context_kernel -> run_mini.controllability_of_kernel -> zone mean over the measured states). Reported per
multiplier: I_s by zone for every repeat, their standard deviation, and the seconds of each measurement.

--goal-control RUN_DIR:ENV_STEP: with the weights at that checkpoint and the replay as of it, the mass of the prior on
the goal for each pair into it with --draws draws (probe.real_context_kernel_pairs), and at each state next to the
goal the Q gap (the agent's greedy Q, probe.agent_greedy_policy, averaged over the state's contexts) between the action
into the goal and the best other action, against the tie-break tolerance.

Usage:
    python experiments/E3_worldmodel_4zones/measure_noise.py --noise <run dir> [--repeats 8] [--contexts-multipliers 1 2 4] --out-dir DIR
    python experiments/E3_worldmodel_4zones/measure_noise.py --goal-control <run dir>:15849 [--draws 10000] --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.heads import Critic, GoalHead
from helplessness.experiment_loader import load_experiment_module

remeasure = load_experiment_module("E3_worldmodel_4zones", "remeasure_checkpoints.py", "noise_remeasure")
run_agent, run_mini = remeasure.run_agent, remeasure.run_mini
e3_probe, e3_env = run_agent.e3_probe, run_mini.e3_env
SEED = 20261007


def _load(run_dir: Path, device_arg: str | None):
    import yaml

    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    if device_arg is not None:
        cfg["runtime"]["device"] = device_arg
    device = run_mini.configure_runtime(cfg)
    world = run_mini.build_world(cfg, int(meta["seed"]), meta["condition"])
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    resume = torch.load(run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    return cfg, meta, device, world, resume


def _models(cfg, state, device):
    acfg = run_agent.agent_config(cfg)
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    goal = GoalHead(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
    goal.load_state_dict(state["goal_head"])
    critic = Critic(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim)
    critic.load_state_dict(state["critic"])
    for m in (rssm, goal, critic):
        m.to(device).eval()
    return acfg, rssm, goal, critic


def is_by_zone_once(rssm, world, replay, cfg, max_per_state: int, rng, gen) -> dict:
    rc = run_mini.real_context_settings(cfg)
    draws, _ = run_mini.kernel_draws(cfg, final=True)
    with torch.no_grad():
        ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=max_per_state,
                                          n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
        kernel = e3_probe.real_context_kernel(rssm, ctx, world.templates_t, draws=draws, margin_threshold=cfg["measure"]["kernel_margin_threshold"],
                                              generator=gen)
    I_s = run_mini.controllability_of_kernel(kernel["p_hat"])
    out = {}
    for cat in e3_env.ZONE_CATEGORIES:
        vals = I_s[run_mini.zone_states(world.layout, cat)]
        out[cat] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    return {"I_s_by_zone": out, "contexts": int(ctx["positions"].size)}


def noise(run_dir: Path, repeats: int, multipliers: list[int], device_arg: str | None) -> dict:
    cfg, meta, device, world, resume = _load(run_dir, device_arg)
    replay = pickle.loads(resume["replay"])
    if device.type == "cuda":
        replay.to_device(device)
    state = torch.load(run_dir / "checkpoints" / "agent_final.pt", map_location="cpu", weights_only=False)
    _, rssm, _, _ = _models(cfg, state, device)
    base = run_mini.real_context_settings(cfg)["max_per_state"]
    out = {"run": run_dir.name, "weights": "agent_final.pt", "replay_env_steps": int(resume["progress"]["env_step"]),
           "base_max_per_state": base, "repeats": repeats, "by_multiplier": {}, "reproducibility": run_mini.environment_metadata(device)}
    for m in multipliers:
        reps = []
        for r in range(repeats):
            t0 = time.perf_counter()
            one = is_by_zone_once(rssm, world, replay, cfg, base * m, np.random.default_rng(SEED + r),
                                  torch.Generator(device=device).manual_seed(SEED + r))
            reps.append({**one, "seconds": time.perf_counter() - t0})
            print(f"[noise] {run_dir.name[:60]} x{m} repeat {r}: {json.dumps(one)} {reps[-1]['seconds']:.0f}s", flush=True)
        cats = list(e3_env.ZONE_CATEGORIES)
        out["by_multiplier"][str(m)] = {
            "max_per_state": base * m, "repeats": reps,
            "sd_by_zone": {c: float(np.std([x["I_s_by_zone"][c] for x in reps], ddof=1)) for c in cats},
            "mean_by_zone": {c: float(np.mean([x["I_s_by_zone"][c] for x in reps])) for c in cats},
            "seconds_mean": float(np.mean([x["seconds"] for x in reps]))}
    return out


def goal_control(run_dir: Path, c: int, draws: int, device_arg: str | None) -> dict:
    cfg, meta, device, world, resume = _load(run_dir, device_arg)
    records = {r["env_step"]: r for r in (json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines())}
    replay = remeasure.replay_prefix(pickle.loads(resume["replay"]), c + int(records[c]["episodes"]))
    if device.type == "cuda":
        replay.to_device(device)
    state = torch.load(remeasure.weights_path(run_dir, c, int(meta["env_steps"])), map_location="cpu", weights_only=False)
    acfg, rssm, goal, critic = _models(cfg, state, device)
    rng, gen = np.random.default_rng(SEED), torch.Generator(device=device).manual_seed(SEED)
    rc = run_mini.real_context_settings(cfg)
    P, g = world.env.transition_tensor(), e3_env.goal_state()
    s_into, a_into = np.nonzero(P[:, :, g] > 0)
    keep = s_into != g
    pairs = list(zip(s_into[keep].tolist(), a_into[keep].tolist()))
    with torch.no_grad():
        ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                          n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
        kernel = e3_probe.real_context_kernel_pairs(rssm, ctx, world.templates_t, pairs, draws=draws,
                                                    margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=gen)
        greedy = e3_probe.agent_greedy_policy(rssm, goal, critic, ctx["h"], ctx["z"], ctx["states"], run_mini.N_STATES, acfg, gen)
    tol = float(acfg.tie_tol_abs)
    rows = []
    for s, a in pairs:
        q = greedy["Q"][s]
        other = np.delete(q, a)
        rows.append({"s": s, "a_into_goal": a, "mass_on_goal": float(kernel[(s, a)]["p_hat"][g]) if (s, a) in kernel else None,
                     "draws": draws, "contexts": int(np.sum(np.asarray(ctx["states"]) == s)),
                     "Q": q.tolist(), "gap_into_goal_minus_best_other": float(q[a] - np.nanmax(other)), "tie_tolerance": tol,
                     "gap_over_tolerance": float((q[a] - np.nanmax(other)) / tol)})
    return {"run": run_dir.name, "env_step": c, "goal_pairs": rows, "reproducibility": run_mini.environment_metadata(device)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise", type=Path, default=None, metavar="RUN_DIR")
    parser.add_argument("--goal-control", default=None, metavar="RUN_DIR:ENV_STEP")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--contexts-multipliers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.noise is None) == (args.goal_control is None):
        parser.error("give exactly one of --noise and --goal-control")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.noise is not None:
        res, name = noise(args.noise, args.repeats, args.contexts_multipliers, args.device), f"noise__{args.noise.name}.json"
    else:
        run_dir, step = args.goal_control.rsplit(":", 1)
        res, name = goal_control(Path(run_dir), int(step), args.draws, args.device), "goal_control.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / name).write_text(json.dumps(run_mini._jsonable(res), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
