"""E8's therapy runs through the library's closed loop (pipeline.agent_run.run_one, ClosedLoopHooks; docs/e8/DECISIONS.md
(4)-(9)): the therapy world, its episode source and the checkpoint record. Imported by run_e8.py as a sibling module.

  world            the configuration's canonical layout with the REFERENCE's dynamics, for the exposed model and its
                   reference alike (two_zones_setup.world(cfg, conf, "reference"): the E5 reference run's environment);
  episode source   baseline: none (E4/E5's single-world run), or episode_sources.SingleWorld with StartInZone when
                   start_in_zone_frac > 0; shuffle: episode_sources.TreatedAssignments over E5's 12 assignment worlds in
                   the reference condition, the treated assignments only, the assignment stream streams.therapy's;
  record           at every checkpoint: the visits of the treated zone and of every quadrant so far (from the replay,
                   which holds every step of the therapy), steps.npz (the per-step log), the episodes by assignment
                   (shuffle) and the health record -- E4's real_context_record on the therapy world (baseline) or E5's
                   real_context_by_assignment on the treated assignments (shuffle), skipped at checkpoint 0 (empty replay).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import ClosedLoopHooks
from helplessness.pipeline.assignment_records import real_context_by_assignment
from helplessness.pipeline.episode_sources import (ASSIGNMENTS, SingleWorld, StartInZone, TreatedAssignments, assignment_world,
                                                   canonical_assignment, treated_assignments)
from helplessness.pipeline.two_zones_records import real_context_record
from helplessness.streams import therapy as therapy_streams

N_STATES = two_zones.N_STATES
SLOT_ASSIGNMENT_REAL_CONTEXT = 22  # E5's slot of real_context_by_assignment (e5_setup.SLOT_ASSIGNMENT_REAL_CONTEXT)
_LAYOUT = two_zones.build_layout(1, "reference")
QUADRANT_OF_STATE = np.array([QUADRANT_ORDER.index(r) if r in QUADRANT_ORDER else -1
                              for r in (two_zones.region_of_state(_LAYOUT, s) for s in range(N_STATES))], dtype=np.int8)


def step_log(rows: np.ndarray, is_first: np.ndarray, therapy: str, treated: str | None) -> dict[str, np.ndarray]:
    """The per-step log from the replay's table rows: state, assignment (shuffle; -1 for a baseline), quadrant (index in
    QUADRANT_ORDER, -1 in the corridor), treated (the step is in the treated zone: the fixed slot, or the UD slot of the
    step's assignment), episode index."""
    rows = np.asarray(rows, dtype=np.int64)
    state = rows % N_STATES
    quadrant = QUADRANT_OF_STATE[state]
    if therapy == "full":
        assignment = np.full(rows.shape, -1, dtype=np.int8)
        treated_slot = np.full(rows.shape, QUADRANT_ORDER.index(treated), dtype=np.int8)
    else:
        assignment = (rows // N_STATES).astype(np.int8)
        ud = np.array([QUADRANT_ORDER.index(ASSIGNMENTS[k][0]) for k in range(len(ASSIGNMENTS))], dtype=np.int8)
        treated_slot = ud[assignment]
    return {"state": state.astype(np.int16), "assignment": assignment, "quadrant": quadrant, "treated": quadrant == treated_slot,
            "episode": (np.cumsum(np.asarray(is_first, dtype=np.int64)) - 1).astype(np.int32), "is_first": np.asarray(is_first, dtype=bool)}


def visits(log: dict[str, np.ndarray]) -> dict:
    q = log["quadrant"]
    return {"stored_steps": int(q.size), "treated": int(log["treated"].sum()), "corridor": int((q < 0).sum()),
            "by_quadrant": {name: int((q == i).sum()) for i, name in enumerate(QUADRANT_ORDER)}}


def make_hooks(cfg: dict, conf: dict, therapy: str, treated: str | None, seed: int) -> ClosedLoopHooks:
    """The hooks of a therapy run: `therapy` "full" (baseline) or a quadrant / "all" (shuffle); `treated` the fixed
    treated slot (None for "all"); seed the model's (start_in_zone_frac's generator)."""
    e8 = cfg["e8"]
    frac = float(e8.get("start_in_zone_frac", 0.0))
    health = e8["health"]
    canonical = canonical_assignment(conf["diagonal"])
    worlds: dict[int, object] = {}

    def build_world(cfg_, seed_, condition, themes=None):
        return two_zones_setup.world(cfg_, conf, "reference")

    def episode_source(world, cfg_, seed_, condition):
        if therapy == "full":
            if frac <= 0:
                return None
            zone = two_zones.region_states(_LAYOUT)[treated]
            return SingleWorld(world, StartInZone(world.env, zone, frac, therapy_streams.start_in_zone_generator(seed_)))
        worlds.clear()
        device = world.templates_t.device
        for k in range(len(ASSIGNMENTS)):
            w = world if k == canonical else assignment_world(k, conf, "reference", cfg_["render"], cfg_["episode"], seed_)
            w.templates_t = w.templates_t.to(device)
            worlds[k] = w
        sz = (frac, therapy_streams.start_in_zone_generator(seed_)) if frac > 0 else None
        return TreatedAssignments([worlds[k] for k in range(len(ASSIGNMENTS))], canonical, therapy,
                                  therapy_streams.assignment_generator(int(e8["seed"])), start_in_zone=sz)

    def checkpoint_record(state: dict) -> dict:
        rssm, device, env_step, replay = state["rssm"], state["device"], state["env_step"], state["replay"]
        t0 = time.perf_counter()
        if len(replay):
            rows, first = replay.obs_rows_and_flags()
        else:
            rows, first = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
        log = step_log(rows, first, therapy, treated)
        np.savez_compressed(state["run_dir"] / "steps.npz", **log)
        rec = {"therapy": therapy, "treated_slot": treated, "visits": visits(log), "health": None}
        source = state["episode_source"]
        if isinstance(source, TreatedAssignments):
            rec["episodes_by_assignment"] = np.bincount(np.asarray(source.episode_assignment, dtype=np.int64),
                                                        minlength=len(ASSIGNMENTS)).tolist()
            np.savez_compressed(state["run_dir"] / "assignments.npz", **source.log_arrays())
        if env_step > 0:
            was_training = rssm.training
            rssm.eval()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
                if therapy == "full":
                    rec["health"] = real_context_record(rssm, state["world"], replay, cfg, state["seed"], state["condition"], env_step, device,
                                                        int(health["real_context_max_per_state"]), int(health["real_context_draws"]),
                                                        two_zones_setup.SLOT_CHECKPOINT_REAL_CONTEXT)
                else:
                    ks = treated_assignments(therapy)
                    rc = real_context_by_assignment(rssm, replay, {k: worlds[k] for k in ks}, cfg, state["seed"], state["condition"], env_step,
                                                    int(health["real_context_max_per_state"]), int(health["real_context_draws"]),
                                                    SLOT_ASSIGNMENT_REAL_CONTEXT)
                    rec["health"] = {str(k): v for k, v in rc.items()}
            rssm.train(was_training)
        rec["seconds"] = time.perf_counter() - t0
        state["log"](f"E8 checkpoint {env_step}: treated-zone visits {rec['visits']['treated']} of {rec['visits']['stored_steps']} "
                     f"stored steps ({rec['seconds']:.0f} s)")
        return {"e8": rec}

    def finalize(records: list, world, run_dir: Path) -> dict:
        with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["env_step", "grad_step", "treated_visits", "stored_steps", "success_rate", "episode_length", "region",
                             "mass_true", "I_s_real_context"])
            for r in records:
                e8_rec = r["e8"]
                h = e8_rec["health"]
                regions = {}
                if h is not None and therapy == "full":
                    regions = {q: (h["mass_true_by_region"][q], h["I_s_by_region"][q]) for q in h["mass_true_by_region"]}
                for region, (mass, i_s) in (regions or {"": (None, None)}).items():
                    writer.writerow([r["env_step"], r["grad_step"], e8_rec["visits"]["treated"], e8_rec["visits"]["stored_steps"],
                                     r["success_rate"], r["episode_length"], region, mass, i_s])
        return {"configuration": conf["id"], "therapy": therapy, "treated_slot": treated, "roles": conf["roles"]}

    return ClosedLoopHooks(episode=cfg["episode"], build_world=build_world, checkpoint_record=checkpoint_record, finalize=finalize,
                           episode_source=None if (therapy == "full" and frac <= 0) else episode_source)
