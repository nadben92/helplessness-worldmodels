"""Read-only diagnostics of the world model of a finished closed-loop run around
the goal (docs/e3/DECISIONS.md, 2026-10-01 (12), reading rule fixed before these
numbers were computed). Nothing of the run is modified or re-run; the two
existing weight sets are used (resume point, environment step 18,000, with its
replay and counts; end, 20,000, with the run's final measures arrays.npz).

  a. mass of the prior on the right successor (s-space kernel in real contexts)
     against the visits n of the pair, for every deterministic pair with s and
     s* off the goal; the four pairs into the goal marked; the reading rule;
  b. the four pairs into the goal: mass on the goal at both weight sets, and
     p_goal of the goal head on latents drawn from the PRIOR at their contexts;
  c. dynamic KL (posterior vs prior) per frame, goal frames vs the others, at
     both weight sets;
  e. do the templates cover the goal's appearance: the observation is the
     template by construction (ReplayBuffer table mode raises otherwise); margin
     of the goal's template to its nearest other template against the ghost
     margin; decoded posterior of the goal frames and decoded prior draws of the
     four goal pairs matched to the templates (goal / other / ghost).
(d. -- the terminal frame in the world-model loss -- is a test:
tests/e3/test_e3_terminal_frame_loss.py.)

Usage: python experiments/E3_worldmodel_4zones/diagnose_world_model_goal.py <run dir> [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.loop import unroll_observation
from helplessness.agents.worldmodel.rssm import LatentEstimator, categorical_kl, sample_categorical_index
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import match_templates, template_distances

diag = load_experiment_module("E3_worldmodel_4zones", "diagnose_agent_run.py", "diagnose_wm_agent_run")
run_mini, e3_env, e3_probe = diag.run_mini, diag.e3_env, diag.run_agent.e3_probe

SEED = 20261002  # this script's own streams
N_PRIOR_DRAWS_GOAL = 512  # prior draws per context for p_goal on prior latents


def deterministic_pairs(P: np.ndarray, goal: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(s, a, s*) of every pair with a single successor of probability 1, s and s* off the goal."""
    s, a = np.nonzero(P.max(-1) >= 1.0 - 1e-12)
    succ = P[s, a].argmax(-1)
    keep = (s != goal) & (succ != goal)
    return s[keep], a[keep], succ[keep]


def goal_pairs(P: np.ndarray, goal: int) -> tuple[np.ndarray, np.ndarray]:
    s, a = np.nonzero(P[:, :, goal] > 0)
    keep = s != goal
    return s[keep], a[keep]


def kernel_at_resume_point(rssm, world, replay, cfg, rng, gen) -> tuple[dict, dict]:
    """The s-space kernel in real contexts with the run's own settings (window, burn-in, contexts per state, K = the
    final kernel_draws, ghost margin), on the replay of the resume point."""
    rc = run_mini.real_context_settings(cfg)
    draws, _ = run_mini.kernel_draws(cfg, final=True)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
    kernel = e3_probe.real_context_kernel(rssm, ctx, world.templates_t, draws=draws, margin_threshold=cfg["measure"]["kernel_margin_threshold"],
                                          generator=gen)
    return ctx, kernel


def rule(p_hat, ghost, N, P, goal) -> dict:
    s, a, succ = deterministic_pairs(P, goal)
    n = N[s, a].sum(-1)
    mass = p_hat[s, a, succ]
    raw = mass * (1.0 - ghost[s, a])
    ok = np.isfinite(mass)
    by_n = {}
    for k in sorted(set(n[ok].tolist())):
        sel = ok & (n == k)
        by_n[int(k)] = {"pairs": int(sel.sum()), "median": float(np.median(mass[sel])),
                        "q25": float(np.quantile(mass[sel], 0.25)), "q75": float(np.quantile(mass[sel], 0.75)),
                        "median_raw_with_ghosts": float(np.median(raw[sel]))}
    band = ok & (n >= 2) & (n <= 8)
    gs, ga = goal_pairs(P, goal)
    goal_mass = p_hat[gs, ga, goal]
    med = float(np.median(mass[band])) if band.any() else float("nan")
    med_goal = float(np.median(goal_mass)) if np.isfinite(goal_mass).any() else float("nan")
    if med >= 0.9 and med_goal < 0.1:
        verdict = "specific to the goal"
    elif med <= 0.5:
        verdict = "general insufficiency in rarely visited contexts"
    else:
        verdict = "undetermined, no action"
    unmeasured = int((~ok).sum())
    return {"verdict": verdict, "median_mass_right_successor_n2_8": med, "pairs_n2_8": int(band.sum()),
            "median_mass_on_goal_of_goal_pairs": med_goal, "by_n": by_n, "deterministic_pairs": int(len(s)),
            "unmeasured_pairs_no_context": unmeasured,
            "goal_pairs": [{"s": int(x), "a": int(y), "n": int(N[x, y].sum()), "mass_on_goal": float(p_hat[x, y, goal]),
                            "ghost_fraction": float(ghost[x, y]), "argmax": int(np.nanargmax(p_hat[x, y])) if np.isfinite(p_hat[x, y]).any() else None,
                            "argmax_mass": float(np.nanmax(p_hat[x, y])) if np.isfinite(p_hat[x, y]).any() else float("nan")}
                           for x, y in zip(gs, ga)]}


@torch.no_grad()
def p_goal_on_prior_latents(rssm, goal_head, ctx, P, goal, gen) -> list[dict]:
    """For each pair into the goal, at every real context of its state: N_PRIOR_DRAWS_GOAL draws z' from the prior
    p(. | h_{t+1,a}) and p_goal(h_{t+1,a}, z')."""
    gs, ga = goal_pairs(P, goal)
    out = []
    for s, a in zip(gs, ga):
        c = np.flatnonzero(ctx["states"] == s)
        if c.size == 0:
            out.append({"s": int(s), "a": int(a), "contexts": 0})
            continue
        ct = torch.as_tensor(c)
        probs = ctx["prior_probs_t"][ct, a].unsqueeze(1).expand(-1, N_PRIOR_DRAWS_GOAL, -1, -1)
        z = F.one_hot(sample_categorical_index(probs, gen), num_classes=probs.shape[-1]).to(probs.dtype)
        h = ctx["h_next"][ct, a].unsqueeze(1).expand(-1, N_PRIOR_DRAWS_GOAL, -1)
        p = goal_head.p_goal(h.reshape(-1, h.shape[-1]), z.reshape(-1, *z.shape[-2:])).numpy()
        out.append({"s": int(s), "a": int(a), "contexts": int(c.size), "draws": int(p.size), "mean_p_goal": float(p.mean()),
                    "fraction_p_goal_above_0.5": float((p > 0.5).mean()), "max_p_goal": float(p.max())})
    return out


@torch.no_grad()
def dynamic_kl(rssm, world, episodes, gen) -> dict:
    """KL(posterior || prior) in bits per frame (the dynamic term's quantity, before any weighting / free bits),
    goal frames vs the others; the first frame of an episode (no transition into it) is left out."""
    goal = e3_env.goal_state()
    L = max(len(e["s"]) for e in episodes)
    kl_goal, kl_other, kl_pre = [], [], []
    for lo in range(0, len(episodes), 512):
        chunk = episodes[lo:lo + 512]
        S = np.zeros((len(chunk), L), dtype=int)
        A = np.zeros((len(chunk), L), dtype=int)
        first = np.zeros((len(chunk), L), dtype=bool)
        first[:, 0] = True
        for i, e in enumerate(chunk):
            S[i, :len(e["s"])] = e["s"]
            A[i, :len(e["a"])] = e["a"]
        out = unroll_observation(rssm, world.templates_t[torch.as_tensor(S)], torch.as_tensor(A), torch.as_tensor(first),
                                 estimator=LatentEstimator("straight_through"), generator=gen, decode=False)
        kl = (categorical_kl(out["post_probs"], out["prior_probs"]) / math.log(2)).numpy()
        for i, e in enumerate(chunk):
            n = len(e["s"])
            for t in range(1, n):
                (kl_goal if e["s"][t] == goal else kl_other).append(kl[i, t])
    q = diag.quantiles
    return {"goal_frames": q(np.array(kl_goal)), "other_frames": q(np.array(kl_other))}


@torch.no_grad()
def goal_appearance(rssm, world, cfg, ctx, episodes, gen) -> dict:
    goal = e3_env.goal_state()
    T = world.templates_t.numpy()
    d = template_distances(T[goal:goal + 1], T)[0]
    d_other = np.delete(d, goal)
    margin = cfg["measure"]["kernel_margin_threshold"]
    all_d = template_distances(T, T)
    np.fill_diagonal(all_d, np.inf)
    nearest_other = all_d.min(-1)
    # decoded posterior of the goal frames
    gen_eps = [e for e in episodes if e["goal"]]
    h, z, s, r = diag.posterior_states(rssm, world, gen_eps, gen)
    sel = torch.as_tensor(np.flatnonzero(s == goal))
    dec = rssm.decode(h[sel], z[sel]).numpy() if len(sel) else np.zeros((0, *T.shape[1:]))
    m = match_templates(dec, T, margin) if len(sel) else {"nearest_state": np.array([]), "is_ghost": np.array([])}
    # decoded prior draws of the four pairs into the goal
    gs, ga = goal_pairs(world.env.transition_tensor(), goal)
    prior = {}
    for s0, a0 in zip(gs, ga):
        c = np.flatnonzero(ctx["states"] == s0)
        if c.size == 0:
            continue
        ct = torch.as_tensor(np.repeat(c, max(1, 256 // c.size)))
        probs = ctx["prior_probs_t"][ct, a0]
        zz = F.one_hot(sample_categorical_index(probs, gen), num_classes=probs.shape[-1]).to(probs.dtype)
        dd = rssm.decode(ctx["h_next"][ct, a0], zz).numpy()
        mm = match_templates(dd, T, margin)
        dist_goal = template_distances(dd, T[goal:goal + 1])[:, 0]
        prior[f"{int(s0)},{int(a0)}"] = {"draws": int(len(ct)), "matched_goal": float(((mm["nearest_state"] == goal) & ~mm["is_ghost"]).mean()),
                                         "ghost": float(mm["is_ghost"].mean()),
                                         "nearest_is_goal_but_ghost": float(((mm["nearest_state"] == goal) & mm["is_ghost"]).mean()),
                                         "median_distance_to_goal_template": float(np.median(dist_goal))}
    return {"observation_is_template_by_construction": "ReplayBuffer table mode raises ObservationTableMismatch otherwise",
            "ghost_margin": margin, "goal_template_nearest_other_distance": float(d_other.min()),
            "goal_template_nearest_other_state": int(np.argsort(d)[1]),
            "all_templates_nearest_other_distance": diag.quantiles(nearest_other),
            "posterior_goal_frames": {"n": int(len(sel)), "matched_goal": float(((m["nearest_state"] == goal) & ~m["is_ghost"]).mean()) if len(sel) else float("nan"),
                                      "ghost": float(m["is_ghost"].mean()) if len(sel) else float("nan")},
            "prior_draws_of_goal_pairs": prior}


def percentile_ranks(p_hat, N, P, goal) -> list[dict]:
    """For each pair into the goal: the percentile rank of its mass on its right successor (the goal) among the
    deterministic pairs off the goal with the same number of visits n (ties count half). Descriptive only
    (docs/e3/DECISIONS.md, 2026-10-02 (13)); it does not enter the reading rule."""
    s, a, succ = deterministic_pairs(P, goal)
    n_all = N[s, a].sum(-1)
    mass = p_hat[s, a, succ]
    out = []
    for gs, ga in zip(*goal_pairs(P, goal)):
        n, m = int(N[gs, ga].sum()), float(p_hat[gs, ga, goal])
        peers = mass[(n_all == n) & np.isfinite(mass)]
        rank = float(((peers < m).sum() + 0.5 * (peers == m).sum()) / peers.size * 100) if peers.size else float("nan")
        out.append({"s": int(gs), "a": int(ga), "n": n, "mass_on_goal": m, "peers_same_n": int(peers.size), "percentile_rank": rank,
                    "peers_median": float(np.median(peers)) if peers.size else float("nan"),
                    "peers_with_mass_below_0.01": int((peers < 0.01).sum())})
    return out


def ranks_main(run_dir: Path, out: Path) -> dict:
    """The percentile ranks at both weight sets: 18,000 recomputes the kernel exactly as main() did (same weights,
    replay, settings and seeds) and saves it (kernel_18000.npz: the baseline of the gradients-vs-data diagnostic);
    20,000 uses the final measures (arrays.npz)."""
    d = diag.load(run_dir)
    cfg, acfg, world = d["cfg"], d["acfg"], d["world"]
    P, goal = world.env.transition_tensor(), e3_env.goal_state()
    resume = torch.load(run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    replay = pickle.loads(resume["replay"])
    rssm, _, _ = diag.models(cfg, acfg, resume)
    _, kernel = kernel_at_resume_point(rssm, world, replay, cfg, np.random.default_rng(SEED), torch.Generator().manual_seed(SEED))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "kernel_18000.npz", p_hat=kernel["p_hat"], ghost_fraction=kernel["ghost_fraction"], N=resume["collector"]["N"])
    arrays = np.load(run_dir / "arrays.npz")
    return {"18000": percentile_ranks(kernel["p_hat"], resume["collector"]["N"], P, goal),
            "20000": percentile_ranks(arrays["real_p_hat"], arrays["N"], P, goal),
            "check_rule_median_18000": rule(kernel["p_hat"], kernel["ghost_fraction"], resume["collector"]["N"], P, goal)["median_mass_right_successor_n2_8"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ranks", action="store_true", help="only the goal pairs' percentile ranks (and the 18,000 kernel saved)")
    args = parser.parse_args(argv)
    if args.ranks:
        out = args.out or run_mini.REPO_ROOT / "reports" / f"diagnostics_wm_goal_ranks_{args.run_dir.name}"
        if out.exists() and any(out.iterdir()):
            parser.error(f"{out} already holds files")
        torch.set_num_threads(4)
        res = ranks_main(args.run_dir, out)
        (out / "ranks.json").write_text(json.dumps(run_mini._jsonable(res), indent=1), encoding="utf-8")
        print(json.dumps(run_mini._jsonable(res), indent=1))
        return 0
    out = args.out or run_mini.REPO_ROOT / "reports" / f"diagnostics_wm_goal_{args.run_dir.name}"
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} already holds files")
    torch.set_num_threads(4)
    d = diag.load(args.run_dir)
    cfg, acfg, world = d["cfg"], d["acfg"], d["world"]
    P, goal = world.env.transition_tensor(), e3_env.goal_state()
    resume = torch.load(args.run_dir / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    replay = pickle.loads(resume["replay"])
    N_resume = resume["collector"]["N"]
    arrays = np.load(args.run_dir / "arrays.npz")
    episodes = diag.episodes_from_replay(d["replay"])
    res = {"run": args.run_dir.name, "goal_state": int(goal)}

    ctxs = {}
    for env_step, state in sorted(d["weights"].items()):
        rssm, goal_head, _ = diag.models(cfg, acfg, state)
        gen = torch.Generator().manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        ctx, kernel = kernel_at_resume_point(rssm, world, replay, cfg, rng, gen)
        ctxs[env_step] = ctx
        entry = {"kernel_on_replay_at_18000": rule(kernel["p_hat"], kernel["ghost_fraction"], N_resume, P, goal)}
        if env_step == max(d["weights"]):
            entry["kernel_final_measures_replay_at_20000"] = rule(arrays["real_p_hat"], arrays["real_ghost_fraction"], arrays["N"], P, goal)
        entry["p_goal_on_prior_latents"] = p_goal_on_prior_latents(rssm, goal_head, ctx, P, goal, torch.Generator().manual_seed(SEED + 1))
        entry["dynamic_kl_bits"] = dynamic_kl(rssm, world, episodes, torch.Generator().manual_seed(SEED + 2))
        entry["goal_appearance"] = goal_appearance(rssm, world, cfg, ctx, episodes, torch.Generator().manual_seed(SEED + 3))
        res[str(env_step)] = entry
        print(f"[diagnose_wm] weights at {env_step} done", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(run_mini._jsonable(res), indent=1), encoding="utf-8")
    print(json.dumps(run_mini._jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
