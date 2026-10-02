"""Gradients against data (docs/e3/DECISIONS.md, 2026-10-02 (13), (14)): the
world model of a closed-loop run, from its resume point, trained further on
the FROZEN replay -- the run's loss, learning rate and schedule, the separation
switch's collapse guard as in the run, no new data, no agent heads (they never
reach the world model) -- and its s-space kernel measured at a few points.

Criterion of (13), decided at the last point only (N = 20,000): median mass of
the prior on the right successor over the deterministic pairs off the goal
with 2 <= n <= 8 (n = the visits at the resume point): >= 0.5 -> limited by the
gradients; a rise < 0.1 over the start -> limited by the data; otherwise mixed.
Reported at every point (descriptive): that median and its number of pairs, by
zone the noisy pairs' L1 to P and their mass on the four minor successors, the
four goal pairs' mass on the goal.

The run directory is only read. Minibatches and latent draws come from this
script's own streams; the kernels use the measurement seeds of the baseline
(diagnose_world_model_goal.SEED), so only the weights differ between points.

Usage: python experiments/E3_worldmodel_4zones/gradients_vs_data.py <run dir> [--steps 20000] [--measure-at 5000 10000 20000] [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.loop import (
    CollapseAfterSwitchError,
    WorldModelLossConfig,
    phase_learning_rate,
    rep_beta,
    set_learning_rate,
    train_world_model_step,
)
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon

wmg = load_experiment_module("E3_worldmodel_4zones", "diagnose_world_model_goal.py", "gvd_wmg")
diag, run_mini, e3_env = wmg.diag, wmg.run_mini, wmg.e3_env

TRAIN_SEED = 20261003  # this diagnostic's minibatch / latent / switch-pair streams


def summarise(p_hat: np.ndarray, ghost: np.ndarray, N: np.ndarray, P: np.ndarray, world) -> dict:
    goal = e3_env.goal_state()
    s, a, succ = wmg.deterministic_pairs(P, goal)
    n = N[s, a].sum(-1)
    mass = p_hat[s, a, succ]
    band = (n >= 2) & (n <= 8) & np.isfinite(mass)
    zones = {}
    for cat in e3_env.ZONE_CATEGORIES:
        st = run_mini.zone_states(world.layout, cat)
        noisy = [(x, y) for x in st for y in range(P.shape[1]) if x != goal and P[x, y].max() < 1 - 1e-12]
        if not noisy:
            continue
        xs, ys = np.array(noisy).T
        main = P[xs, ys].argmax(-1)
        minor_true = P[xs, ys] > 0
        minor_true[np.arange(len(xs)), main] = False
        l1 = np.abs(p_hat[xs, ys] - P[xs, ys]).sum(-1)
        minor_mass = (p_hat[xs, ys] * minor_true).sum(-1)
        zones[cat] = {"noisy_pairs": int(len(xs)), "visits_median": float(np.median(N[xs, ys].sum(-1))),
                      "l1_median": float(np.nanmedian(l1)), "l1_mean": float(np.nanmean(l1)),
                      "minor_successors_mass_median": float(np.nanmedian(minor_mass)),
                      "minor_successors_mass_mean": float(np.nanmean(minor_mass)), "minor_successors_mass_true": float(
                          (P[xs, ys] * minor_true).sum(-1).mean())}
    gs, ga = wmg.goal_pairs(P, goal)
    return {"median_mass_right_successor_n2_8": float(np.median(mass[band])), "pairs_n2_8": int(band.sum()),
            "noisy_by_zone": zones, "goal_pairs_mass_on_goal": [float(p_hat[x, y, goal]) for x, y in zip(gs, ga)],
            "ghost_fraction_median": float(np.nanmedian(ghost))}


def verdict(start: float, end: float) -> str:
    if end >= 0.5:
        return "limited by the gradients"
    if end - start < 0.1:
        return "limited by the data"
    return "mixed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--measure-at", type=int, nargs="+", default=[5_000, 10_000, 20_000])
    parser.add_argument("--baseline", type=Path, default=None, help="kernel_18000.npz of diagnose_world_model_goal.py --ranks")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    out = args.out or run_mini.REPO_ROOT / "reports" / f"gradients_vs_data_{args.run_dir.name}"
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already holds files")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    log = run_mini.RunLog(out / "console.log", "[gradients-vs-data]")

    d = diag.load(args.run_dir)
    cfg, acfg, world = d["cfg"], d["acfg"], d["world"]
    P = world.env.transition_tensor()
    resume = torch.load(args.run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    replay = pickle.loads(resume["replay"])
    N = resume["collector"]["N"]
    rssm, _, _ = diag.models(cfg, acfg, resume)
    tr = cfg["train"]
    optimizer = torch.optim.Adam(rssm.parameters(), lr=tr["lr"], eps=tr["adam_eps"])
    optimizer.load_state_dict(resume["optimizer"])
    switch, n_pairs = run_mini.build_separation_switch(cfg)
    for name, value in resume["switch"].items():
        setattr(switch, name, value)
    loss_cfg = WorldModelLossConfig(k=cfg["render"]["k"], w_patch=tr["w_patch"], beta_dyn=tr["beta_dyn"], beta_rep=tr["beta_rep"],
                                    free_bits=tr["free_bits"], free_bits_mode=tr["free_bits_mode"])
    progress = resume["progress"]
    grad0, rep_warmup = int(progress["grad_step"]), progress["rep_warmup"] or 0
    rng = np.random.default_rng(TRAIN_SEED)
    pair_rng = np.random.default_rng(TRAIN_SEED + 1)
    gen = torch.Generator().manual_seed(TRAIN_SEED + 2)

    baseline = np.load(args.baseline or run_mini.REPO_ROOT / "reports" / f"diagnostics_wm_goal_ranks_{args.run_dir.name}" / "kernel_18000.npz")
    if not np.array_equal(baseline["N"], N):
        raise ValueError("the baseline kernel was not measured on this resume point")
    points = {0: summarise(baseline["p_hat"], baseline["ghost_fraction"], N, P, world)}
    log(f"start: gradient step {grad0} (switch {switch.switch_step}, rep warm-up {rep_warmup}), replay {len(replay)} observations; "
        f"baseline median n=2-8 {points[0]['median_mass_right_successor_n2_8']:.3f} ({points[0]['pairs_n2_8']} pairs)")
    measure_at = sorted(set(args.measure_at) | {args.steps})
    t0, collapse = time.perf_counter(), None
    rssm.train()
    for i in range(1, args.steps + 1):
        g = grad0 + i - 1
        set_learning_rate(optimizer, phase_learning_rate(g, switch.switch_step, tr["lr"], tr["lr_after_switch"]))
        beta = rep_beta(g, switch.switch_step, rep_warmup, loss_cfg.beta_rep)
        batch, _ = replay.sample(tr["batch_size"], tr["seq_len"], rng)
        eval_now = switch.is_eval_step(g)
        res = train_world_model_step(rssm, optimizer, batch, world.background_t, dataclasses.replace(loss_cfg, beta_rep=beta),
                                     clip_norm=tr["clip_norm"], estimator=switch.estimator(), return_posteriors=eval_now, generator=gen)
        if eval_now:
            try:
                switch.update(g, mean_random_pair_jensen_shannon(res["post_probs"], pair_rng, n_pairs))
            except CollapseAfterSwitchError as err:
                collapse = f"step {i}: {err}"
                log(f"COLLAPSE GUARD fired at +{i}: {err}")
                break
        if i % 1000 == 0:
            log(f"+{i:6d} | recon {res['recon']:.4f} | kl_dyn {res['kl_dyn']:.3f} | {time.perf_counter() - t0:.0f}s")
        if i in measure_at:
            rssm.eval()
            _, kernel = wmg.kernel_at_resume_point(rssm, world, replay, cfg, np.random.default_rng(wmg.SEED), torch.Generator().manual_seed(wmg.SEED))
            rssm.train()
            points[i] = summarise(kernel["p_hat"], kernel["ghost_fraction"], N, P, world)
            np.savez_compressed(out / f"kernel_plus{i:06d}.npz", p_hat=kernel["p_hat"], ghost_fraction=kernel["ghost_fraction"])
            torch.save({"rssm": rssm.state_dict(), "step": grad0 + i}, out / f"rssm_plus{i:06d}.pt")
            log(f"measured at +{i}: median n=2-8 {points[i]['median_mass_right_successor_n2_8']:.3f}")
    result = {"run": args.run_dir.name, "start_gradient_step": grad0, "steps_done": i if collapse else args.steps, "collapse": collapse,
              "seconds": time.perf_counter() - t0, "points": points}
    if args.steps in points:
        result["criterion"] = {"start": points[0]["median_mass_right_successor_n2_8"], "end": points[args.steps]["median_mass_right_successor_n2_8"],
                               "verdict": verdict(points[0]["median_mass_right_successor_n2_8"], points[args.steps]["median_mass_right_successor_n2_8"])}
    (out / "result.json").write_text(json.dumps(run_mini._jsonable(result), indent=1), encoding="utf-8")
    log(json.dumps(run_mini._jsonable(result.get("criterion", {}))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
