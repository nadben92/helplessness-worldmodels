"""Does the predicted latent code z' carry the action beyond the arrival
state? (docs/e3/DECISIONS.md, 2026-09-29 -- a diagnostic requested after
the first GPU batch; it does not change the main measure.)

I_z measures how much the predicted code z' depends on the action. If the
code of one and the same arrival state s' also encodes the action that led
there (through h_{t+1,a}), I_z is inflated without any change in the belief
about the state. Test: from the real contexts of a trained run, draw z' from
the one-step prior for every action (probe.real_context_action_samples),
decode each draw to its arrival state s', and, at fixed s' (per zone of the
source state s), train a linear probe z' -> a on 80% of the draws and score
it on the rest. Reported per zone, averaged over the s' groups weighted by
their test draws:

  chance            1 / n_actions (20%);
  majority | s'     the most frequent action of the group (training split);
  (s, s') only      the most frequent action of each (s, s') pair: what the
                    source and arrival states alone tell about a -- in a
                    controllable zone, a lot; in an uncontrollable one,
                    nothing (the arrival does not depend on a);
  z' | s'           the linear probe on z' (one-hot, n_vars x n_classes);
  z' + s | s'       the linear probe on z' and the one-hot source state.

z' carries action information beyond the states when "z' + s | s'" exceeds
"(s, s') only". Ghost draws are left out.

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/action_in_latent.py <run dir> [--checkpoint final|<step>] [--draws 4] [--device cpu]
Output: <run dir>/analysis/action_in_latent__<checkpoint>__<UTC>/ with results.json and SUMMARY.md.
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

from helplessness.experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "action_in_latent_run_mini")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "action_in_latent_probe")

SLOT_ACTION_IN_LATENT = 10  # measurement slot of this diagnostic (run_mini's own slots: 1-4, 6-9)
MIN_GROUP = 50  # s' groups with fewer (non-ghost) draws are left out
TEST_FRAC = 0.2


def linear_probe_accuracy(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, n_classes: int,
                          seed: int, epochs: int = 300, lr: float = 0.05) -> float:
    """Softmax regression (full batch, Adam), as the z-only probe; test accuracy."""
    g = torch.Generator().manual_seed(seed)
    xt, yt = torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long)
    w = (0.01 * torch.randn(xt.shape[1], n_classes, generator=g)).requires_grad_()
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(xt @ w + b, yt).backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.as_tensor(x_test, dtype=torch.float32) @ w + b).argmax(-1).numpy()
    return float((pred == y_test).mean())


def majority_accuracy(keys_train: np.ndarray, y_train: np.ndarray, keys_test: np.ndarray, y_test: np.ndarray, n_classes: int) -> float:
    """Predict, for each key, the most frequent class of that key in the training split
    (the group's overall majority for a key never seen in training)."""
    fallback = np.bincount(y_train, minlength=n_classes).argmax()
    table = {k: np.bincount(y_train[keys_train == k], minlength=n_classes).argmax() for k in np.unique(keys_train)}
    pred = np.array([table.get(k, fallback) for k in keys_test])
    return float((pred == y_test).mean())


def analyse(samples: dict, zone_of_state: list[str], categories: list[str], n_vars: int, n_classes: int, n_actions: int,
            rng: np.random.Generator) -> dict:
    keep = ~samples["is_ghost"]
    s, a, s_next, zi = samples["s"][keep], samples["a"][keep], samples["s_next"][keep], samples["z_index"][keep]
    z_onehot = np.zeros((zi.shape[0], n_vars * n_classes), dtype=np.float32)
    z_onehot[np.arange(zi.shape[0])[:, None], np.arange(n_vars)[None, :] * n_classes + zi] = 1.0
    zone = np.array(zone_of_state)[s]
    out = {}
    for cat in categories:
        in_zone = np.flatnonzero(zone == cat)
        rows, n_test_total = [], 0
        for sp in np.unique(s_next[in_zone]):
            idx = in_zone[s_next[in_zone] == sp]
            if idx.size < MIN_GROUP:
                continue
            idx = rng.permutation(idx)
            n_test = max(1, int(round(TEST_FRAC * idx.size)))
            te, tr = idx[:n_test], idx[n_test:]
            src = np.unique(s[idx])
            s_onehot = (s[:, None] == src[None, :]).astype(np.float32)
            seed = int(rng.integers(2**31))
            rows.append({
                "s_next": int(sp), "n": int(idx.size), "n_test": int(n_test), "n_sources": int(src.size),
                "action_counts": np.bincount(a[idx], minlength=n_actions).tolist(),
                "majority_given_s_next": majority_accuracy(np.zeros(tr.size), a[tr], np.zeros(te.size), a[te], n_actions),
                "states_only": majority_accuracy(s[tr], a[tr], s[te], a[te], n_actions),
                "z_given_s_next": linear_probe_accuracy(z_onehot[tr], a[tr], z_onehot[te], a[te], n_actions, seed),
                "z_and_s_given_s_next": linear_probe_accuracy(np.hstack([z_onehot[tr], s_onehot[tr]]), a[tr],
                                                              np.hstack([z_onehot[te], s_onehot[te]]), a[te], n_actions, seed),
            })
            n_test_total += n_test
        weights = np.array([r["n_test"] for r in rows], dtype=float)

        def wmean(key: str) -> float:
            return float(np.average([r[key] for r in rows], weights=weights)) if rows else float("nan")

        out[cat] = {"n_groups": len(rows), "n_draws": int(sum(r["n"] for r in rows)), "n_test": n_test_total,
                    "n_draws_in_zone": int(in_zone.size), "chance": 1.0 / n_actions,
                    **{key: wmean(key) for key in ("majority_given_s_next", "states_only", "z_given_s_next", "z_and_s_given_s_next")},
                    "groups": rows}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="final", help="final (default) or a gradient step")
    parser.add_argument("--draws", type=int, default=4, help="prior draws per context and action")
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
    weights = run_dir / "checkpoints" / ("weights_final.pt" if label == "final" else f"weights_{label}.pt")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    rssm.to(device).eval()

    streams = run_mini.seed_streams(seed, device, condition)
    rng, gen = run_mini.measure_streams(seed, SLOT_ACTION_IN_LATENT, streams)
    rc = run_mini.real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
    samples = e3_probe.real_context_action_samples(rssm, ctx, world.templates_t, draws=args.draws,
                                                   margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=gen)
    cats = list(run_mini.e3_env.ZONE_CATEGORIES)
    zone_of_state = [run_mini.zone_of_state(world.layout, s) for s in range(run_mini.N_STATES)]
    result = analyse(samples, zone_of_state, cats, rssm.cfg.n_vars, rssm.cfg.n_classes, rssm.cfg.n_actions, rng)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = run_dir / "analysis" / f"action_in_latent__{label}__{stamp}"
    out_dir.mkdir(parents=True)
    record = {"run_dir": str(run_dir), "checkpoint": label, "step": int(state["step"]), "seed": seed, "condition": condition,
              "draws_per_context_action": args.draws, "n_contexts": int(ctx["positions"].size),
              "ghost_fraction": float(samples["is_ghost"].mean()), "min_group": MIN_GROUP, "test_frac": TEST_FRAC,
              "zone_kinds": world.zone_kinds, "by_zone": result}
    (out_dir / "results.json").write_text(json.dumps(run_mini._jsonable(record), indent=1), encoding="utf-8")
    lines = [f"# Action in the predicted latent -- {run_dir.name}, {label} (gradient step {record['step']})", "",
             f"{record['n_contexts']} real contexts x {rssm.cfg.n_actions} actions x {args.draws} prior draws; ghost fraction "
             f"{record['ghost_fraction']:.4f}. Test accuracy of predicting a, at fixed decoded arrival state s' (groups >= {MIN_GROUP} "
             "draws, weighted by their test draws).", "",
             "| source zone | dynamics | s' groups | test draws | chance | majority given s' | (s, s') only | z' given s' | z' + s given s' |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cat in cats:
        r = result[cat]
        lines.append(f"| {cat} | {world.zone_kinds[cat]} | {r['n_groups']} | {r['n_test']} | {r['chance']:.3f} | "
                     f"{r['majority_given_s_next']:.3f} | {r['states_only']:.3f} | {r['z_given_s_next']:.3f} | {r['z_and_s_given_s_next']:.3f} |")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
