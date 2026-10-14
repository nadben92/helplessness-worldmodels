"""The RSA of a run's latent at its checkpoints (docs/e3/DECISIONS.md, 2026-10-13 (31), F), off the cluster, on the CPU,
from the handed-off weights (checkpoints/agent_env<NNNNNN>.pt) and replay (replay_final.npz). E3's bridge to the
generic measures.rsa: which contexts, which latent, which attributes; the analysis itself is measures.rsa's.

At checkpoint c: the replay as of c (its first `stored_steps` observations, checkpoints.jsonl; a replay that stops
earlier -- the pilot's, at its last resume point -- is used whole and FLAGGED), the weights at c; real contexts
(probe.real_context_probe: windows unrolled with their true history, burn-in, up to MAX_PER_STATE per state) drawn from
the run's own measurement stream, slot SLOT_RSA, keyed by the checkpoint; per context, h_t and the posterior
probabilities q(z_t | h_t, x_t). Points: every state but the goal (terminal), and every zone (the mean of its states'
means). Attributes of a point, from the run's own environment: its theme (the background colour of its zone), its
controllability and stochasticity (from the true kernel), its position on the torus. Scalars: the regression's
weights, R^2 and unique shares; the confusion ratios (mean distance between controllable and uncontrollable zones' states
over the mean distance between states of one zone; all four zones, CD vs UD, CN vs UN); split-half reliability.

Usage: python experiments/E3_worldmodel_4zones/rsa_latents.py <run dir> [--checkpoints 25119 300000] [--out DIR]
           [--max-per-state 200] [--device cpu]
Outputs: <out>/<run name>/env<NNNNNN>.json (scalars) and .npz (point means, distance matrices, MDS coordinates).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures import rsa

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "rsa_latents_run_mini")
e3_rng = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "rsa_latents_rng")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "rsa_latents_probe")
remeasure = load_experiment_module("E3_worldmodel_4zones", "remeasure_checkpoints.py", "rsa_latents_remeasure")

SLOT_RSA = 16  # measurement slot of the RSA contexts (run_mini 1-4, 6-9, 11, 12; action_in_latent 10; hybrid codes 13;
#                 agent Delta 14; goal diagnostics 15)
MAX_PER_STATE = 200  # the measures' protocol x4 (DECISIONS (28)); the cost is the windows' unroll, not the number of contexts
GRID_SHAPE = run_mini.GRID_SHAPE
CONTROLLABLE, UNCONTROLLABLE = "controllable", "uncontrollable"
# the theoretical matrices besides the position; the zone is not one (in E3 it is the theme: identical matrices), only the
# within-group label of the ratios
PREDICTORS = ("theme", "controllability", "stochasticity")
# the confusion ratio of (31) F: the four quadrant zones grouped by APPEARANCE (their category's name), so that the reference
# contrasts the same zones (appearance only); the corridor belongs to neither group. Corrected 2026-10-14 (33): the first
# run grouped by the run's own controllability (corridor included; empty group in the reference).
RATIOS = {"controllable_vs_uncontrollable": ("appearance_group", CONTROLLABLE, UNCONTROLLABLE, "zone"),
          "CD_vs_UD": ("zone", "controllable_deterministic", "uncontrollable_deterministic", "zone"),
          "CN_vs_UN": ("zone", "controllable_noisy", "uncontrollable_noisy", "zone")}


def point_attributes(world, P: np.ndarray) -> dict:
    """Per state: zone (appearance category), theme (the zone's background colour), controllability and stochasticity
    from the true kernel P (I(a; s' | s) > 0; some action without a single successor), position."""
    run_ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "rsa_latents_ca")
    I_true = run_ca.controllability_of_kernel(P)
    zones = [run_mini.zone_of_state(world.layout, s) for s in range(run_mini.N_STATES)]
    return {"zone": zones,
            "theme": ["rgb" + ",".join(map(str, world.themes.color_for_category(z))) for z in zones],
            "controllability": [CONTROLLABLE if i > 1e-9 else UNCONTROLLABLE for i in I_true],
            "stochasticity": ["noisy" if (P[s].max(-1) < 1.0 - 1e-12).any() else "deterministic" for s in range(run_mini.N_STATES)],
            "appearance_group": [UNCONTROLLABLE if z.startswith("uncontrollable") else CONTROLLABLE if z.startswith("controllable") else z
                                 for z in zones],
            "position": np.array([divmod(s, GRID_SHAPE[1]) for s in range(run_mini.N_STATES)], dtype=float)}


def predictors_for(attrs: dict, keep: np.ndarray) -> tuple[list[str], list[str]]:
    """The predictors of PREDICTORS that vary over the kept states, and those removed because constant -- in a reference
    run every zone is controllable: its controllability predictor is removed explicitly (DECISIONS (32)), not left to the
    regression's own guard."""
    used, removed = [], []
    for k in PREDICTORS:
        (used if len(set(np.asarray(attrs[k])[keep])) > 1 else removed).append(k)
    return used, removed


def zone_level(state_out: dict, attrs: dict, keep: np.ndarray, predictors: list[str]) -> dict:
    """The zones as points: each latent the mean of its states' means (the goal left out), the position the zone's
    centre on the torus, the attributes those of its states (constant within a zone, checked)."""
    zones = sorted(set(np.asarray(attrs["zone"])[keep]))
    zarr = np.asarray(attrs["zone"])
    latents = {}
    for name, means in state_out["point_means"].items():
        latents[name] = np.stack([np.nanmean(means[keep & (zarr == z)], axis=0) for z in zones])
    za = {}
    for key in ("zone", "theme", "controllability", "stochasticity"):
        vals = []
        for z in zones:
            v = set(np.asarray(attrs[key])[keep & (zarr == z)])
            if len(v) != 1:
                raise ValueError(f"attribute {key} not constant within zone {z}: {v}")
            vals.append(v.pop())
        za[key] = vals
    pos = np.stack([rsa.torus_centre(attrs["position"][keep & (zarr == z)], GRID_SHAPE) for z in zones])
    out = rsa.analyse_point_means(latents, {k: za[k] for k in ("theme", "controllability", "stochasticity", "zone")}, pos, GRID_SHAPE,
                                  ratios=None, predictors=predictors)
    out["points"] = zones
    out["note"] = f"{len(zones)} points, {len(zones) * (len(zones) - 1) // 2} pairs: the regression is near saturation, reported, not interpreted"
    return out


def contexts_at(rssm, world, replay, cfg: dict, seed: int, condition: str, env_step: int, max_per_state: int, device) -> dict:
    seq = e3_rng.spawn_sequences(seed, condition)["measure"].spawn(e3_rng.N_MEASURE_SLOTS)[SLOT_RSA]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step)))
    rng, gen = np.random.default_rng(child), e3_rng.torch_generator(child, device)
    rc = run_mini.real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=max_per_state,
                                      n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
    return {"h": ctx["h"].cpu().numpy(), "z": np.asarray(ctx["post_probs"]), "states": np.asarray(ctx["states"]), "rng": rng}


def rsa_of_contexts(ctx: dict, world, P: np.ndarray) -> dict:
    attrs = point_attributes(world, P)
    keep = np.ones(run_mini.N_STATES, dtype=bool)
    keep[run_mini.e3_env.goal_state()] = False
    used, removed = predictors_for(attrs, keep)
    state = rsa.analyse_points({"h": ctx["h"], "z": ctx["z"]}, ctx["states"], run_mini.N_STATES,
                               {k: attrs[k] for k in ("theme", "controllability", "stochasticity", "zone", "appearance_group")}, attrs["position"],
                               GRID_SHAPE, ctx["rng"], ratios=RATIOS, mask=keep, predictors=used)
    state["removed_predictors"] = {k: "constant over the run's states" for k in removed}
    zones = zone_level(state, attrs, keep, used)
    zones["removed_predictors"] = state["removed_predictors"]
    return {"states": state, "zones": zones, "attributes": {k: v for k, v in attrs.items() if k != "position"}}


def load_checkpoint(run_dir: Path, c: int, cfg: dict, device) -> tuple:
    """The RSSM at checkpoint c, the world and the replay as of c (flag: the replay stops earlier)."""
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    seed, condition = int(results["seed"]), results["condition"]
    records = {r["env_step"]: r for r in (json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    last = max(records)
    state = torch.load(remeasure.weights_path(run_dir, c, last), map_location="cpu", weights_only=False)
    rssm = run_mini.build_rssm(cfg)
    rssm.load_state_dict(state["rssm"])
    rssm.to(device).eval()
    world = run_mini.build_world(cfg, seed, condition)
    data = np.load(run_dir / "replay_final.npz")
    full = ReplayBuffer.from_compact({k: data[k] for k in ("s", "a", "r", "terminal", "truncated", "is_first")}, world.templates_np)
    stored = int(records[c]["stored_steps"])
    flagged = stored > len(full)
    replay = full if flagged else remeasure.replay_prefix(full, stored)
    return rssm, world, replay, seed, condition, {"stored_steps": stored, "replay_observations": len(replay), "flagged_short_replay": flagged}


def rsa_run(run_dir: Path, checkpoints: list[int] | None, out_dir: Path, max_per_state: int = MAX_PER_STATE, device_arg: str = "cpu") -> list[dict]:
    run_dir = Path(run_dir)
    cfg = run_mini.load_mini_config(run_dir / "config.yaml")
    cfg["runtime"]["device"] = device_arg
    device = run_mini.configure_runtime(cfg)
    if checkpoints is None:
        checkpoints = sorted(int(d.name[3:]) for d in (run_dir / "measures").glob("env*"))
    out = Path(out_dir) / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for c in checkpoints:
        t0 = time.perf_counter()
        rssm, world, replay, seed, condition, prov = load_checkpoint(run_dir, c, cfg, device)
        t_load = time.perf_counter() - t0
        ctx = contexts_at(rssm, world, replay, cfg, seed, condition, c, max_per_state, device)
        t_ctx = time.perf_counter() - t0 - t_load
        res = rsa_of_contexts(ctx, world, world.env.transition_tensor())
        seconds = {"load": t_load, "contexts": t_ctx, "rsa": time.perf_counter() - t0 - t_load - t_ctx}
        summary = {"run": run_dir.name, "seed": seed, "condition": condition, "env_step": c, "max_per_state": max_per_state,
                   "n_contexts": int(ctx["states"].size), "slot": SLOT_RSA, **prov, "seconds": seconds,
                   "states": {k: v for k, v in res["states"].items() if k not in ("matrices", "point_means")},
                   "zones": {k: v for k, v in res["zones"].items() if k not in ("matrices",)}, "attributes": res["attributes"]}
        (out / f"env{c:06d}.json").write_text(json.dumps(run_mini._jsonable(summary), indent=1), encoding="utf-8")
        np.savez_compressed(out / f"env{c:06d}.npz", **{f"state_{k}": v for k, v in res["states"]["matrices"].items()},
                            **{f"state_mean_{k}": v for k, v in res["states"]["point_means"].items()},
                            **{f"zone_{k}": v for k, v in res["zones"]["matrices"].items()})
        summaries.append(summary)
        print(f"{run_dir.name} env {c}: {ctx['states'].size} contexts; load {t_load:.0f} s, contexts {t_ctx:.0f} s, rsa {seconds['rsa']:.0f} s", flush=True)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None, help="default: every measured checkpoint")
    parser.add_argument("--out", type=Path, default=Path("results/E3_worldmodel_4zones/campaign/rsa"))
    parser.add_argument("--max-per-state", type=int, default=MAX_PER_STATE)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args(argv)
    rsa_run(args.run_dir, args.checkpoints, args.out, args.max_per_state, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
