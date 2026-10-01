"""Hybrid-codes test (docs/e3/DECISIONS.md, 2026-09-30 (5); a diagnostic on
existing weights, no training). In the noisy zones the world model's
s-space kernel puts 4 to 5 % of its mass on states that are not reachable
successors. Hypothesis: the prior is a PRODUCT of n_vars independent
categoricals, so when the possible successors differ in several latent
variables, a draw can mix the variables of different successors into a code
the posterior never produces, and that code decodes to another state.

For (s, a) pairs of the noisy zones (dynamics type, in this run's
condition), from real contexts of s:

  factorised      z' ~ prod_i p_i(. | h_{t+1,a}) (the prior as it is),
                  decoded and matched to the templates: the share decoded
                  outside the 5 reachable successors of s;
  joint mode      z' = the mode of every variable of the same prior;
  posterior       the codes the model actually produced after (s, a) in the
                  replay (z_{t+1} at the steps with s_t = s, a_t = a),
                  decoded with their own h_{t+1}: a non-factorised draw of
                  the same transition.

Each factorised draw's code is classified against the posterior codes of
the whole replay (probe.replay_posterior_codes):

  novel           the full code (all n_vars classes together) was never
                  produced by the posterior;
  novel pair      some pair of variables takes two classes never produced
                  together by the posterior;
  hybrid          novel, and every variable's class is one the posterior
                  produced for a reachable successor of s (a mix of the
                  successors' own codes).

CRITERION (docs/e3/DECISIONS.md, 2026-09-30 (6): fixed after the dry run on
the bridge and BEFORE any result on the three target runs): the hypothesis
is confirmed for a run if the off-support mass of the posterior codes
actually observed after (s, a) is at most half of the off-support mass of
the factorised prior draws (noisy zones pooled). Both masses are means over
the (s, a) pairs, each pair counting once. The joint mode's off-support
mass, and the classification of the factorised draws' codes (novel, novel
pair, hybrid: the criterion of entry (5), kept as a description), are
reported next to it.

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/hybrid_codes_check.py <run dir> [--weights <weights_final.pt>]
        [--contexts 10] [--draws 64] [--device cpu]
<run dir> needs config.yaml, meta.yaml, world.json and replay.npz (the light archive has them); the weights
default to <run dir>/checkpoints/weights_final.pt.
Output: <run dir>/analysis/hybrid_codes__<UTC>/ with results.json and SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from helplessness.agents.worldmodel.agent import categorical_mode
from helplessness.agents.worldmodel.rssm import sample_categorical_index
from helplessness.envs.zones import geographic_support_mask
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import match_templates

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "hybrid_codes_run_mini")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "hybrid_codes_probe")

SLOT_HYBRID_CODES = 13  # measurement slot of this diagnostic (others: run_mini 1-4, 6-9, 11, 12; action_in_latent 10)
CRITERION = 0.5  # confirmed if off-support(posterior after (s, a)) <= CRITERION * off-support(factorised prior)


def verdict(factorised: float, posterior: float) -> dict:
    """The test's verdict from the two off-support masses (noisy zones pooled)."""
    if not (np.isfinite(factorised) and np.isfinite(posterior)):
        return {"verdict": "NOT EVALUABLE", "ratio": float("nan")}
    if factorised <= 0.0:
        return {"verdict": "NOT EVALUABLE (no off-support mass under the factorised prior)", "ratio": float("nan")}
    ratio = posterior / factorised
    return {"verdict": "CONFIRMED" if ratio <= CRITERION else "NOT CONFIRMED", "ratio": float(ratio)}


def mean_over_pairs(off: np.ndarray, s: np.ndarray, a: np.ndarray, n_actions: int) -> tuple[float, int]:
    """The mean, over the (s, a) pairs present, of each pair's off-support share: every pair counts once,
    however many draws (or replay transitions) it has. Returns (mean, number of pairs)."""
    if off.size == 0:
        return float("nan"), 0
    keys = s.astype(np.int64) * n_actions + a
    uniq, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=off.astype(np.float64), minlength=uniq.size)
    counts = np.bincount(inverse, minlength=uniq.size)
    return float((sums / counts).mean()), int(uniq.size)


def code_keys(z_index: np.ndarray, n_classes: int) -> np.ndarray:
    """One integer per code: (..., n_vars) class indices -> (...,) base-n_classes number."""
    weights = n_classes ** np.arange(z_index.shape[-1], dtype=np.int64)
    return (z_index.astype(np.int64) * weights).sum(axis=-1)


def pair_table(z_index: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_vars, n_vars, n_classes, n_classes) bool: classes of two variables produced together in some code."""
    n_vars = z_index.shape[-1]
    table = np.zeros((n_vars, n_vars, n_classes, n_classes), dtype=bool)
    for i in range(n_vars):
        for j in range(n_vars):
            table[i, j, z_index[:, i], z_index[:, j]] = True
    return table


def classify_codes(draws: np.ndarray, observed_keys: np.ndarray, pairs: np.ndarray, successor_classes: np.ndarray,
                   n_classes: int) -> dict[str, np.ndarray]:
    """draws: (n, n_vars) class indices. observed_keys: the sorted unique keys of the posterior's codes.
    pairs: pair_table of the posterior's codes. successor_classes: (n, n_vars, n_classes) bool, for each draw
    the classes the posterior produced, per variable, for the reachable successors of the draw's state.
    Returns novel, novel_pair, hybrid (n,) bool."""
    n, n_vars = draws.shape
    novel = ~np.isin(code_keys(draws, n_classes), observed_keys)
    i, j = np.triu_indices(n_vars, k=1)
    novel_pair = ~pairs[i[None, :], j[None, :], draws[:, i], draws[:, j]].all(axis=1)
    from_successors = successor_classes[np.arange(n)[:, None], np.arange(n_vars)[None, :], draws].all(axis=1)
    return {"novel": novel, "novel_pair": novel_pair, "hybrid": novel & from_successors}


def summarise(off: np.ndarray, flags: dict[str, np.ndarray]) -> dict:
    """Shares over non-ghost factorised draws: off-support, and how the off-support draws split."""
    n, n_off = int(off.size), int(off.sum())
    share = lambda mask: float(mask.mean()) if mask.size else float("nan")
    out = {"n_draws": n, "n_off_support": n_off, "off_support_share": share(off)}
    for name, flag in flags.items():
        out[f"{name}_share_of_draws"] = share(flag)
        out[f"{name}_share_of_off_support"] = share(flag[off])
        out[f"off_support_share_among_{name}"] = share(off[flag])
        out[f"off_support_share_among_not_{name}"] = share(off[~flag])
    return out


def check(rssm, world, replay, cfg: dict, rng: np.random.Generator, generator: torch.Generator, n_contexts: int, draws: int,
          chunk: int = 1024) -> dict:
    device = next(rssm.parameters()).device
    V, C, A = rssm.cfg.n_vars, rssm.cfg.n_classes, rssm.cfg.n_actions
    n_states = run_mini.N_STATES
    support = geographic_support_mask(run_mini.GRID_SHAPE)
    templates_np = world.templates_t.detach().cpu().numpy()
    margin = cfg["measure"]["kernel_margin_threshold"]
    max_images = run_mini.max_images_per_forward(cfg)

    # 1. the posterior's codes on the whole replay (windows of the training length, as in training)
    seq_len = cfg["train"]["seq_len"]
    codes = e3_probe.replay_posterior_codes(rssm, replay.iter_chronological_windows(seq_len, chunk=256), generator, max_images)
    z_all = codes["z_index"].reshape(-1, V)
    s_all = codes["s"].reshape(-1)
    observed_keys = np.unique(code_keys(z_all, C))
    pairs = pair_table(z_all, C)
    classes_of_state = np.zeros((n_states, V, C), dtype=bool)  # classes the posterior produced for each state, per variable
    classes_of_state[s_all[:, None], np.arange(V)[None, :], z_all] = True
    successor_classes = np.stack([classes_of_state[support[s]].any(axis=0) for s in range(n_states)])  # (S, V, C)

    # 2. factorised prior draws from real contexts of the noisy zones' states
    rc = run_mini.real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=n_states, rng=rng, generator=generator, max_images=max_images)
    noisy = [cat for cat in run_mini.e3_env.ZONE_CATEGORIES if "noisy" in world.zone_kinds[cat]]
    zone_of = np.array([run_mini.zone_of_state(world.layout, s) for s in range(n_states)])
    chosen = []
    for s in np.flatnonzero(np.isin(zone_of, noisy)):
        idx = np.flatnonzero(ctx["states"] == s)
        if idx.size:
            chosen.append(np.sort(rng.choice(idx, size=min(n_contexts, idx.size), replace=False)))
    chosen = np.concatenate(chosen)
    row_ctx = np.repeat(chosen, A * draws)
    row_a = np.tile(np.repeat(np.arange(A), draws), chosen.size)
    row_s = ctx["states"][row_ctx]
    z_draw = np.empty((row_ctx.size, V), dtype=np.int64)
    nearest, ghost = np.empty(row_ctx.size, dtype=np.int64), np.empty(row_ctx.size, dtype=bool)
    with torch.no_grad():
        for first in range(0, row_ctx.size, chunk):
            sl = slice(first, min(first + chunk, row_ctx.size))
            c, a = torch.as_tensor(row_ctx[sl], device=device), torch.as_tensor(row_a[sl], device=device)
            probs = ctx["prior_probs_t"][c, a]
            idx = sample_categorical_index(probs, generator)
            decoded = rssm.decode(ctx["h_next"][c, a], F.one_hot(idx, num_classes=C).to(probs.dtype)).cpu().numpy()
            matched = match_templates(decoded, templates_np, margin)
            z_draw[sl], nearest[sl], ghost[sl] = idx.cpu().numpy(), matched["nearest_state"], matched["is_ghost"]
        # joint mode of the same priors: one decode per (context, action)
        mode_ctx, mode_a = np.repeat(chosen, A), np.tile(np.arange(A), chosen.size)
        mode_nearest, mode_ghost = np.empty(mode_ctx.size, dtype=np.int64), np.empty(mode_ctx.size, dtype=bool)
        for first in range(0, mode_ctx.size, chunk):
            sl = slice(first, min(first + chunk, mode_ctx.size))
            c, a = torch.as_tensor(mode_ctx[sl], device=device), torch.as_tensor(mode_a[sl], device=device)
            decoded = rssm.decode(ctx["h_next"][c, a], categorical_mode(ctx["prior_probs_t"][c, a])).cpu().numpy()
            matched = match_templates(decoded, templates_np, margin)
            mode_nearest[sl], mode_ghost[sl] = matched["nearest_state"], matched["is_ghost"]
        # posterior codes actually produced after (s, a): z_{t+1}, h_{t+1} at the replay steps with s_t in the noisy zones
        s_w, first_w = codes["s"], codes["is_first"]
        w_idx, t_idx = np.nonzero(np.isin(zone_of[s_w[:, :-1]], noisy) & ~first_w[:, 1:])
        post_s, post_a = s_w[w_idx, t_idx], codes["a"][w_idx, t_idx]
        post_nearest, post_ghost = np.empty(w_idx.size, dtype=np.int64), np.empty(w_idx.size, dtype=bool)
        for first in range(0, w_idx.size, chunk):
            sl = slice(first, min(first + chunk, w_idx.size))
            w, t = torch.as_tensor(w_idx[sl], device=device), torch.as_tensor(t_idx[sl] + 1, device=device)
            z = F.one_hot(torch.as_tensor(codes["z_index"][w_idx[sl], t_idx[sl] + 1], device=device), num_classes=C).to(codes["h"].dtype)
            decoded = rssm.decode(codes["h"][w, t], z).cpu().numpy()
            matched = match_templates(decoded, templates_np, margin)
            post_nearest[sl], post_ghost[sl] = matched["nearest_state"], matched["is_ghost"]

    off = ~support[row_s, nearest]
    flags = classify_codes(z_draw, observed_keys, pairs, successor_classes[row_s], C)
    keep = ~ghost
    by_zone = {}
    for cat in [*noisy, "all noisy zones"]:
        m = keep & (np.isin(zone_of[row_s], noisy) if cat == "all noisy zones" else zone_of[row_s] == cat)
        mm = ~mode_ghost & (np.ones_like(mode_ghost) if cat == "all noisy zones" else zone_of[ctx["states"][mode_ctx]] == cat)
        pm = ~post_ghost & (np.ones_like(post_ghost) if cat == "all noisy zones" else zone_of[post_s] == cat)
        mode_s, mode_off, post_off = ctx["states"][mode_ctx], ~support[ctx["states"][mode_ctx], mode_nearest], ~support[post_s, post_nearest]
        fact_mass, fact_pairs = mean_over_pairs(off[m], row_s[m], row_a[m], A)
        mode_mass, _ = mean_over_pairs(mode_off[mm], mode_s[mm], mode_a[mm], A)
        post_mass, post_pairs = mean_over_pairs(post_off[pm], post_s[pm], post_a[pm], A)
        by_zone[cat] = {
            "dynamics": world.zone_kinds.get(cat, "-"),
            "factorised": {**summarise(off[m], {k: v[m] for k, v in flags.items()}), "off_support_mass": fact_mass, "n_pairs": fact_pairs},
            "joint_mode": {"n": int(mm.sum()), "off_support_mass": mode_mass},
            "posterior_after_s_a": {"n": int(pm.sum()), "off_support_mass": post_mass, "n_pairs": post_pairs},
            **verdict(fact_mass, post_mass),
        }
    pooled = by_zone["all noisy zones"]
    return {"by_zone": by_zone, "criterion": {"statistic": "off-support mass of the posterior codes observed after (s, a) / off-support mass of the "
                                                           "factorised prior draws (noisy zones pooled, means over (s, a) pairs)",
                                              "threshold": CRITERION, "value": pooled["ratio"], "verdict": pooled["verdict"],
                                              "factorised": pooled["factorised"]["off_support_mass"],
                                              "posterior": pooled["posterior_after_s_a"]["off_support_mass"],
                                              "joint_mode": pooled["joint_mode"]["off_support_mass"]},
            "n_posterior_codes": int(z_all.shape[0]), "n_distinct_posterior_codes": int(observed_keys.size),
            "n_contexts": int(chosen.size), "draws_per_context_action": draws,
            "ghost_fraction": {"factorised": float(ghost.mean()), "joint_mode": float(mode_ghost.mean()), "posterior": float(post_ghost.mean())}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="default: <run dir>/checkpoints/weights_final.pt")
    parser.add_argument("--contexts", type=int, default=10, help="real contexts per state of the noisy zones")
    parser.add_argument("--draws", type=int, default=64, help="factorised prior draws per context and action")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    weights = args.weights or run_dir / "checkpoints" / "weights_final.pt"
    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition = int(meta["seed"]), meta.get("condition", "experimental")
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = run_mini.configure_runtime(cfg)
    world, replay, _, _ = run_mini.load_run_data(run_dir, cfg, seed, condition)
    world.templates_t = world.templates_t.to(device)
    if device.type == "cuda":
        replay.to_device(device)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    rssm.to(device).eval()
    rng, gen = run_mini.measure_streams(seed, SLOT_HYBRID_CODES, run_mini.seed_streams(seed, device, condition))
    result = check(rssm, world, replay, cfg, rng, gen, args.contexts, args.draws)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / "analysis" / f"hybrid_codes__{stamp}"
    out_dir.mkdir(parents=True)
    record = {"run_dir": str(run_dir), "weights": str(weights), "step": int(state["step"]), "seed": seed, "condition": condition,
              "n_vars": rssm.cfg.n_vars, "n_classes": rssm.cfg.n_classes, **result}
    (out_dir / "results.json").write_text(json.dumps(run_mini._jsonable(record), indent=1), encoding="utf-8")
    crit = result["criterion"]
    lines = [f"# Hybrid-codes test -- {run_dir.name} (gradient step {record['step']}, {condition})", "",
             f"{result['n_contexts']} real contexts of the noisy zones' states x {rssm.cfg.n_actions} actions x {args.draws} factorised prior "
             f"draws; posterior codes of the replay: {result['n_posterior_codes']} ({result['n_distinct_posterior_codes']} distinct, of "
             f"{rssm.cfg.n_classes}^{rssm.cfg.n_vars} possible). Ghost draws are left out.", "",
             f"**Criterion (fixed in advance): {crit['statistic']} <= {crit['threshold']:g}. Factorised {crit['factorised']:.4f}, "
             f"posterior {crit['posterior']:.4f} (joint mode {crit['joint_mode']:.4f}): ratio {crit['value']:.3f} -> {crit['verdict']}.**", "",
             "| zone | dynamics | off-support mass: factorised ((s, a) pairs) | joint mode | posterior after (s, a) ((s, a) pairs) | ratio posterior / "
             "factorised | novel codes among the draws | off-support draws with a novel code | ... a novel pair | ... a hybrid code | "
             "off-support share among novel / observed codes |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cat, z in result["by_zone"].items():
        f = z["factorised"]
        lines.append(f"| {cat} | {z['dynamics']} | {f['off_support_mass']:.4f} ({f['n_pairs']}) | "
                     f"{z['joint_mode']['off_support_mass']:.4f} | {z['posterior_after_s_a']['off_support_mass']:.4f} "
                     f"({z['posterior_after_s_a']['n_pairs']}) | {z['ratio']:.3f} | "
                     f"{f['novel_share_of_draws']:.3f} | {f['novel_share_of_off_support']:.3f} | {f['novel_pair_share_of_off_support']:.3f} | "
                     f"{f['hybrid_share_of_off_support']:.3f} | {f['off_support_share_among_novel']:.4f} / {f['off_support_share_among_not_novel']:.4f} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
