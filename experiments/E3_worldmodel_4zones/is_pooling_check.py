"""I_s per context vs pooled over a state's contexts (docs/e3/DECISIONS.md,
2026-09-30). The main measure I_s pools the decoded draws of all the
contexts of a state into one kernel (K draws per (s, a), the i-th from the
(i mod n)-th context) before computing I(a ; s' | s): per-context kernels
with a plug-in bias <= 0.01 bits would need >= 512 draws in EACH context,
50 times the cost. Pooling measures the controllability of the mixture of
the contexts' kernels, which is at most the mean of their controllabilities
(mutual information is convex in the channel). This script measures the gap
on a trained run: for n_states_per_zone states per zone and n_contexts of
their real contexts, `draws` prior draws per context and action, decoded:

  per-context   mean over the contexts of I_s(context's own kernel);
  pooled        I_s of the kernel pooling the same contexts' draws;
  gap           per-context - pooled (>= 0 up to sampling);

with the bootstrap bias of each (measures.controllability).

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/is_pooling_check.py <run dir> [--checkpoint final] [--draws 512]
        [--states-per-zone 4] [--contexts 10] [--device cpu]
Output: <run dir>/analysis/is_pooling__<checkpoint>__<UTC>/ with results.json and SUMMARY.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.controllability import controllability_bootstrap, controllability_per_state

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "is_pooling_run_mini")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "is_pooling_probe")

SLOT_IS_POOLING = 12  # measurement slot of this diagnostic (run_mini: 1-4, 6-9, 11; action_in_latent: 10)


def kernels_from_samples(samples: dict, n_ctx: int, n_actions: int, n_states: int) -> np.ndarray:
    """(n_ctx, n_actions, n_states) kernels from the non-ghost draws of each context and action."""
    keep = ~samples["is_ghost"]
    counts = np.zeros((n_ctx, n_actions, n_states))
    np.add.at(counts, (samples["context"][keep], samples["a"][keep], samples["s_next"][keep]), 1.0)
    return counts / counts.sum(axis=-1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--draws", type=int, default=512, help="prior draws per context and action")
    parser.add_argument("--states-per-zone", type=int, default=4)
    parser.add_argument("--contexts", type=int, default=10, help="contexts per state")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
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
    label = "final" if args.checkpoint == "final" else f"step{int(args.checkpoint):08d}"
    state = torch.load(run_dir / "checkpoints" / ("weights_final.pt" if label == "final" else f"weights_{label}.pt"),
                       map_location="cpu", weights_only=False)
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    rssm.to(device).eval()

    streams = run_mini.seed_streams(seed, device, condition)
    rng, gen = run_mini.measure_streams(seed, SLOT_IS_POOLING, streams)
    rc = run_mini.real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
    chosen_states, chosen_ctx = [], []
    for cat in run_mini.e3_env.ZONE_CATEGORIES:
        candidates = [s for s in run_mini.zone_states(world.layout, cat) if (ctx["states"] == s).sum() >= args.contexts]
        for s in sorted(rng.choice(candidates, size=min(args.states_per_zone, len(candidates)), replace=False)):
            chosen_states.append(int(s))
            chosen_ctx.append(np.sort(rng.choice(np.flatnonzero(ctx["states"] == s), size=args.contexts, replace=False)))
    idx = np.concatenate(chosen_ctx)
    idx_t = torch.as_tensor(idx, device=device)
    sub = {"positions": ctx["positions"][idx], "states": ctx["states"][idx], "prior_probs_t": ctx["prior_probs_t"][idx_t],
           "h_next": ctx["h_next"][idx_t]}
    samples = e3_probe.real_context_action_samples(rssm, sub, world.templates_t, draws=args.draws,
                                                   margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=gen)
    A = rssm.cfg.n_actions
    per_ctx = kernels_from_samples(samples, idx.size, A, run_mini.N_STATES)  # (n_sub, A, S)
    zone_of = {s: run_mini.zone_of_state(world.layout, s) for s in chosen_states}
    rows = []
    for k, s in enumerate(chosen_states):
        block = slice(k * args.contexts, (k + 1) * args.contexts)
        kernels = per_ctx[block]
        valid = np.isfinite(kernels).all(axis=(1, 2))
        I_ctx = controllability_per_state(kernels[valid])
        bias_ctx = controllability_bootstrap(kernels[valid], args.draws, 200, rng)["bias"]
        pooled = kernels[valid].mean(axis=0, keepdims=True)  # equal draws per context: the pooled kernel
        I_pool = float(controllability_per_state(pooled)[0])
        bias_pool = float(controllability_bootstrap(pooled, args.draws * int(valid.sum()), 200, rng)["bias"][0])
        rows.append({"s": s, "zone": zone_of[s], "n_contexts": int(valid.sum()), "per_context_mean": float(I_ctx.mean()),
                     "per_context_sd": float(I_ctx.std(ddof=1)), "per_context_bias": float(np.mean(bias_ctx)),
                     "pooled": I_pool, "pooled_bias": bias_pool, "gap": float(I_ctx.mean() - I_pool)})
    by_zone = {}
    for cat in run_mini.e3_env.ZONE_CATEGORIES:
        zr = [r for r in rows if r["zone"] == cat]
        by_zone[cat] = {key: float(np.mean([r[key] for r in zr])) for key in
                        ("per_context_mean", "per_context_bias", "pooled", "pooled_bias", "gap")} | {"n_states": len(zr)}
        gaps = np.array([r["gap"] for r in zr])
        by_zone[cat]["gap_se_over_states"] = float(gaps.std(ddof=1) / np.sqrt(gaps.size)) if gaps.size > 1 else float("nan")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / "analysis" / f"is_pooling__{label}__{stamp}"
    out_dir.mkdir(parents=True)
    record = {"run_dir": str(run_dir), "checkpoint": label, "step": int(state["step"]), "draws_per_context_action": args.draws,
              "contexts_per_state": args.contexts, "ghost_fraction": float(samples["is_ghost"].mean()),
              "zone_kinds": world.zone_kinds, "by_zone": by_zone, "states": rows}
    (out_dir / "results.json").write_text(json.dumps(run_mini._jsonable(record), indent=1), encoding="utf-8")
    lines = [f"# I_s per context vs pooled -- {run_dir.name}, {label}", "",
             f"{len(chosen_states)} states ({args.states_per_zone} per zone), {args.contexts} real contexts each, {args.draws} prior draws "
             f"per context and action; ghost fraction {record['ghost_fraction']:.4f}. Bits; bias = parametric-bootstrap plug-in bias.", "",
             "| zone | dynamics | per-context mean I_s (bias) | pooled I_s (bias) | gap per-context - pooled [se over states] |",
             "|---|---|---|---|---|"]
    for cat, z in by_zone.items():
        lines.append(f"| {cat} | {world.zone_kinds[cat]} | {z['per_context_mean']:.4f} ({z['per_context_bias']:.4f}) | "
                     f"{z['pooled']:.4f} ({z['pooled_bias']:.4f}) | {z['gap']:+.4f} [{z['gap_se_over_states']:.4f}] |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
