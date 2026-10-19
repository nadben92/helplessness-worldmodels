"""The cost of E5's checkpoint measures by assignment (IMPLEMENTATION_REPORT.md, "cost"): E4's two checkpoint measures
(real contexts, fixed walks; e4.checkpoint's settings) for the canonical assignment alone -- what an E4 checkpoint costs
-- and for k assignments, with the model of the given config (E4's architecture, untrained: its prior is diffuse, so the
decoding cost is an upper bound, as E4's own timing) and a replay of --steps steps collected under shuffle (f = 1)
with uniform random actions (no model: the cost of the measures depends on the replay's size, not on who collected it).

Usage: python experiments/E5_belief_training/time_assignment_records.py [--config configs/e5/e5.yaml] [--steps 20000]
           [--assignments 1 2 12] [--device cpu] [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

import e5_setup
from helplessness.agents.worldmodel.closed_loop import stored_steps_bound
from helplessness.agents.worldmodel.loop import draw_start_state, non_terminal_states
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.assignment_records import real_context_by_assignment, walks_by_assignment
from helplessness.pipeline.checkpoints import build_rssm
from helplessness.pipeline.episode_sources import ASSIGNMENTS, ZoneAssignmentShuffle, assignment_colors, assignment_world, canonical_assignment
from helplessness.pipeline.runtime import configure_runtime
from helplessness.streams.assignment import assignment_generator


def collect(source, n_steps: int, rng: np.random.Generator) -> ReplayBuffer:
    replay = ReplayBuffer.for_run(stored_steps_bound(n_steps), 0, table=source.table, background_table=source.background_table,
                                  background_of_row=source.background_of_row)
    done = 0
    while done < n_steps:
        env, offset = source.next_episode(done)
        obs, info = env.reset(start_state=draw_start_state(non_terminal_states(env), rng))
        s, r_in, first = info["state"], 0.0, True
        while True:
            a = int(rng.integers(5))
            replay.add(obs, a, r_in, False, False, first, s, obs_index=offset + s)
            obs, r, term, trunc, info = env.step(a)
            done += 1
            if term or trunc or done >= n_steps:
                replay.add(obs, 0, r, bool(term), bool(trunc and not term), False, info["state"], obs_index=offset + info["state"])
                break
            s, r_in, first = info["state"], float(r), False
    return replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(e5_setup.DEFAULT_CONFIG))
    parser.add_argument("--configuration", default="p1-a-d1")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--assignments", type=int, nargs="+", default=[1, 2, 12])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    cfg = e5_setup.load_config(args.config)
    cfg["runtime"].update(device=args.device, num_threads=args.threads)
    device = configure_runtime(cfg)
    conf = e5_setup.configuration(cfg, args.configuration, "A")
    canonical = canonical_assignment(conf["diagonal"])
    world = two_zones_setup.world(cfg, conf, "experimental")
    worlds = {k: world if k == canonical else assignment_world(k, conf, "experimental", cfg["render"], cfg["episode"], conf["seed"])
              for k in range(len(ASSIGNMENTS))}
    for w in worlds.values():
        w.templates_t = w.templates_t.to(device)
    probe_worlds = {k: two_zones_setup.probe_world(cfg, conf, assignment_colors(k, conf["diagonal"], conf["t_inc"], conf["t_ctrl"]))
                    for k in range(len(ASSIGNMENTS))}
    source = ZoneAssignmentShuffle([worlds[k] for k in range(len(ASSIGNMENTS))], canonical, 1.0, assignment_generator(conf["seed"]))
    t0 = time.perf_counter()
    replay = collect(source, args.steps, np.random.default_rng(0))
    torch.manual_seed(0)
    rssm = build_rssm(cfg).to(device).eval()
    ck = cfg["e4"]["checkpoint"]
    print(f"replay of {len(replay)} stored steps in {time.perf_counter() - t0:.0f} s", flush=True)
    out = {"steps": args.steps, "stored": len(replay), "device": str(device), "threads": torch.get_num_threads(), "settings": ck, "timings": {}}
    others = [k for k in range(len(ASSIGNMENTS)) if k != canonical]
    for n in args.assignments:
        ks = [canonical] + others[:n - 1]
        with torch.no_grad():
            t0 = time.perf_counter()
            real_context_by_assignment(rssm, replay, {k: worlds[k] for k in ks}, cfg, conf["seed"], "experimental", 1, ck["real_context_max_per_state"],
                                       ck["real_context_draws"], e5_setup.SLOT_ASSIGNMENT_REAL_CONTEXT)
            t1 = time.perf_counter()
            walks_by_assignment(rssm, cfg, conf, {k: probe_worlds[k] for k in ks}, canonical, ck["n_walks"], "checkpoint")
            t2 = time.perf_counter()
        out["timings"][n] = {"real_context_s": t1 - t0, "walks_s": t2 - t1, "total_s": t2 - t0}
        print(f"{n} assignment(s): real contexts {t1 - t0:.0f} s, walks {t2 - t1:.0f} s, total {t2 - t0:.0f} s", flush=True)
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
