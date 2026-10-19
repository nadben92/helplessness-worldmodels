"""The RSA of an E5 run's latent at a checkpoint (docs/E5_belief_training.md, "measures", 3): E3's analysis
(measures.rsa: the regression of the latent's distance matrix on theoretical ones, each predictor's unique share of
variance, h and z), E5's bridge -- which contexts, which points, which attributes.

Points: replay table rows, i.e. (assignment, state) pairs, the goal left out. A run with one appearance (baseline,
immunisation) has the canonical assignment only: its 143 states, as E3 (theme and position confounded: a slot's colour is
fixed). A shuffle run has up to 12 x 143 points: the same position under different themes and controllabilities, so the
three are no longer confounded -- the output reports the correlations between the theoretical matrices
(predictor_correlations, vif) to check it, for the canonical points alone and for all of them.
Contexts: E3's real contexts (probe.real_context_probe on pipeline.assignment_records.AssignmentReplayView), up to
--max-per-state per point, from the run's own measurement stream (slot SLOT_RSA, keyed by the checkpoint); per context,
h_t and the posterior probabilities q(z_t | h_t, x_t). Attributes of a point, from that assignment's world in the run's
own dynamics: theme (the colour of the state's zone), controllability (I(a;s'|s) > 0 in the true kernel), position (the
torus). After an immunisation's switch, the run's dynamics are the experimental ones.

Usage: python experiments/E5_belief_training/rsa_e5.py <run dir> --checkpoint <env step> [--assignments all|canonical]
           [--max-per-state 200] [--device cpu] --out <dir>
Outputs: <out>/<run name>/env<NNNNNN>.json (scalars) and .npz (point means, distance matrices, MDS coordinates).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.envs.layouts import two_zones
from helplessness.measures import rsa
from helplessness.measures.controllability import controllability_per_state
from helplessness.pipeline import two_zones as two_zones_world
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.assignment_records import AssignmentReplayView
from helplessness.pipeline.episode_sources import ASSIGNMENTS, ZoneAssignmentShuffle, assignment_world, canonical_assignment
from helplessness.pipeline.measure import real_context_settings
from helplessness.pipeline.runtime import _jsonable, configure_runtime, max_images_per_forward
from helplessness.probing import probe as e3_probe
from helplessness.probing.theme_probe_measure import load_rssm
from helplessness.streams import worldmodel as wm_streams
from helplessness.streams.assignment import assignment_generator

SLOT_RSA = 24  # measurement slot (E3 1-16, E4 20-21, E5 22-23)
N = two_zones.N_STATES
PREDICTORS = ("theme", "controllability")


def load(run_dir: Path, step: int, device):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    meta = yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8"))
    seed, condition = int(meta["seed"]), meta["condition"]
    spec = cfg.get("e5", {}).get("run") or {"training": "baseline", "configuration": cfg["e4"]["run_configuration"]}  # an E4 run
    conf = {**two_zones_setup.configuration(cfg, spec["configuration"]), "seed": seed}
    records = {int(r["env_step"]): r for r in (json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    canonical = canonical_assignment(conf["diagonal"])
    world = two_zones_setup.world(cfg, conf, condition)
    data = dict(np.load(run_dir / "replay_final.npz"))
    stored = int(records[step]["stored_steps"])
    data = {k: v[:stored] for k, v in data.items()}
    if spec["training"] == "shuffle":
        worlds = {k: world if k == canonical else assignment_world(k, conf, condition, cfg["render"], cfg["episode"], seed)
                  for k in range(len(ASSIGNMENTS))}
        src = ZoneAssignmentShuffle([worlds[k] for k in range(len(ASSIGNMENTS))], canonical, float(spec["fraction"]), assignment_generator(seed))
        replay = ReplayBuffer.from_compact(data, src.table, src.background_table, src.background_of_row)
    else:
        if spec["training"] == "immunisation" and records[step].get("e5", {}).get("dynamics_switch_step") is not None:
            world = two_zones_world.build_world(conf["training_colors"], conf["agent_color"], conf["diagonal"], "experimental",
                                                cfg["render"], cfg["episode"], seed, env_condition="reference")
        worlds = {canonical: world}
        replay = _OneAppearance(ReplayBuffer.from_compact(data, world.templates_np), canonical)
    rssm = load_rssm(run_dir / "checkpoints" / f"agent_env{step:06d}.pt", device)
    return cfg, conf, seed, condition, rssm, worlds, replay, canonical, stored


class _OneAppearance:
    """A one-appearance replay seen with the canonical assignment's rows (k * N + s)."""

    def __init__(self, replay, k: int):
        self.replay, self.offset = replay, k * N

    def obs_rows_and_flags(self):
        rows, first = self.replay.obs_rows_and_flags()
        return rows + self.offset, first

    def windows(self, starts, seq_len):
        return self.replay.windows(starts, seq_len)


def attributes(worlds: dict, rows: np.ndarray) -> dict:
    theme, ctrl, pos = [], [], []
    cache = {k: controllability_per_state(w.env.transition_tensor()) for k, w in worlds.items()}
    for row in rows:
        k, s = divmod(int(row), N)
        w = worlds[k]
        zone = two_zones.region_of_state(w.layout, s)
        color = w.themes.corridor_color if zone == "corridor" else w.themes.zone_colors[zone]
        theme.append("rgb" + ",".join(str(int(v)) for v in color))
        ctrl.append("controllable" if cache[k][s] > 1e-9 else "uncontrollable")
        pos.append(divmod(s, two_zones.GRID_SHAPE[1]))
    return {"theme": theme, "controllability": ctrl, "position": np.array(pos, dtype=float)}


def analyse(ctx: dict, worlds: dict, rows_used: np.ndarray, rng) -> dict:
    """rows_used: the table rows that are points (sorted); ctx states are table rows."""
    point_of = {int(r): i for i, r in enumerate(rows_used)}
    keep_ctx = np.array([int(s) in point_of for s in ctx["states"]])
    point_of_context = np.array([point_of[int(s)] for s in ctx["states"][keep_ctx]])
    attrs = attributes(worlds, rows_used)
    used = [p for p in PREDICTORS if len(set(attrs[p])) > 1]
    res = rsa.analyse_points({"h": ctx["h"][keep_ctx], "z": ctx["z"][keep_ctx]}, point_of_context, len(rows_used),
                             {p: attrs[p] for p in PREDICTORS}, attrs["position"], two_zones.GRID_SHAPE, rng, predictors=used)
    res["removed_predictors"] = [p for p in PREDICTORS if p not in used]
    res["n_points"] = int(len(rows_used))
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", type=int, required=True, help="environment step of the weights")
    parser.add_argument("--assignments", choices=("all", "canonical"), default="all")
    parser.add_argument("--max-per-state", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    cfg0 = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    cfg0["runtime"]["device"] = args.device
    device = configure_runtime(cfg0)
    cfg, conf, seed, condition, rssm, worlds, replay, canonical, stored = load(args.run, args.checkpoint, device)
    keep_k = sorted(worlds) if args.assignments == "all" else [canonical]
    seq = wm_streams.spawn_sequences(seed, condition)["measure"].spawn(wm_streams.N_MEASURE_SLOTS)[SLOT_RSA]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(args.checkpoint)))
    rng, gen = np.random.default_rng(child), wm_streams.torch_generator(child, device)
    rc = real_context_settings(cfg)
    view = AssignmentReplayView(replay, keep_k)
    ctx = e3_probe.real_context_probe(rssm, view, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=args.max_per_state,
                                      n_states=view.n_rows, rng=rng, generator=gen, max_images=max_images_per_forward(cfg))
    ctx = {"h": ctx["h"].cpu().numpy(), "z": np.asarray(ctx["post_probs"]), "states": np.asarray(ctx["states"])}
    rows = np.unique(ctx["states"])
    rows = rows[rows % N != two_zones.GOAL_STATE]
    out = {"all_points": analyse(ctx, {k: worlds[k] for k in keep_k}, rows, rng)}
    canonical_rows = rows[rows // N == canonical]
    if len(keep_k) > 1 and canonical_rows.size > 2:
        out["canonical_points"] = analyse(ctx, {canonical: worlds[canonical]}, canonical_rows, rng)
    summary = {"run": args.run.name, "seed": seed, "condition": condition, "env_step": args.checkpoint, "stored_steps": stored,
               "assignments": keep_k, "max_per_state": args.max_per_state, "n_contexts": int(ctx["states"].size), "slot": SLOT_RSA,
               "seconds": time.perf_counter() - t0,
               **{name: {k: v for k, v in res.items() if k not in ("matrices", "point_means")} for name, res in out.items()}}
    d = args.out / args.run.name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"env{args.checkpoint:06d}.json").write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    np.savez_compressed(d / f"env{args.checkpoint:06d}.npz", rows=rows,
                        **{f"{name}_{k}": v for name, res in out.items() for k, v in {**res["matrices"],
                                                                                      **{f"mean_{m}": x for m, x in res["point_means"].items()}}.items()})
    reg = out["all_points"]["latents"]
    print(f"{args.run.name} env {args.checkpoint}: {len(rows)} points, {ctx['states'].size} contexts; unique R2 h "
          f"{ {k: round(v, 3) for k, v in reg['h']['regression']['unique_r2'].items()} } z "
          f"{ {k: round(v, 3) for k, v in reg['z']['regression']['unique_r2'].items()} }", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
