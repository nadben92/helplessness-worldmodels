"""Steps 2-4 of the robustness diagnostic (PREDICTIONS.md, section 3): one frozen model, one image set, the swap probes'
walks (two_zones_setup.walks_for_slot, the first --n-walks, n_in = n_in_grid), on the CPU.

Per (image, zone of interest), with the probe image P and, where the set gives one, its origin image O:
  a1   the model filters the walk on P (probing.fixed_trajectory.fixed_trajectory_context, the posterior draws seeded
       as the probes' diagnostic: measurement_seed(..., "probe")); decode(h, z) -> the agent's cell
       (probe_validity.decoded_agent_cell, at the agent's brown) against the true state. The job's localisation.
  o1   the same on O (its own seed slot): the origin's localisation.
  a2   h_t of O's filtering, z_t ~ posterior(h_t^O, e(P_t)), decode(h_t^O, z_t): the encoder alone, the history clean.
  b    from a1's context at frame j - 1 and the walk's action, one prior step (rssm.imagination_step, its sample)
       decoded; correct if the agent's cell is the true state at j. b_n / b_ok: where a1 is correct at j - 1;
       b_all_ok: unconditional. Zone frames only (j has a predecessor in the walk).
  tv_a2 / tv_a1  per variable of z, the total variation between posterior(h_t^O, e(P_t)) (a2) or the posterior of a1's
       own filtering (a1) and posterior(h_t^O, e(O_t)).
  sharp / margin per variable, the largest posterior probability of a1's filtering and (top1 - top2).
  a1x / a2x  with an agent colour other than the brown (4c): the cell nearest to that colour.

Summed per walk and part (corridor: steps since entry 0; zone: >= 1; step1: = 1) -> arrays (n_walks, 3, N_FIELDS), one
per (image, zone), in <out-dir>/<model>__<set>__w<NNN>__<UTC>/walk_sums.npz, with images.json and meta.json. The
training set also writes every frame's most probable class per variable, its state and walk (frames.npz: latent MI).

Sets:
  arc_ctrl   the 33 arc_ctrl images of the swap set (3 swaps x 11 t), origin = the swap's base image (t = 0); for the
             baselines, also the se swap's base image against the canonical training image (entry "se_t0_canonical").
  step4      stimuli.step4_images (4a, 4b, 4c) and the canonical training image itself, origin = the canonical image.
  training   the 12 training assignments (no origin, no prior).

Usage, from the repo root:
  python experiments/E5_belief_training/analysis_robustness/mechanism.py --model F --set arc_ctrl --runs-root <root> [...]
      --probes-root <dir> [...] --out-dir <dir outside the repo> [--n-walks 100] [--threads 6] [--images 0 1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common
import stimuli
from helplessness.agents.worldmodel.loop import encode_windows
from helplessness.agents.worldmodel.rssm import LatentEstimator, sample_posterior
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup
from helplessness.probing import walks as walks_mod
from helplessness.probing.fixed_trajectory import fixed_trajectory_context
from helplessness.probing.probe_validity import decoded_agent_cell
from helplessness.probing.theme_probe_measure import load_rssm

SETS = ("arc_ctrl", "step4", "training")
PARTS = ("corridor", "zone", "step1")
SCALARS = ("n", "a1", "o1", "a2", "b_n", "b_ok", "b_all_n", "b_all_ok", "a1x", "a2x")
N_VARS = 8
VECTORS = ("tv_a1", "tv_a2", "sharp", "margin")
N_FIELDS = len(SCALARS) + N_VARS * len(VECTORS)
CANONICAL_INDEX = stimuli.IMAGE_BASE + 100
TRAINING_BASE = stimuli.IMAGE_BASE + 200
PURPOSE = {"origin": 101, "a2": 102, "prior": 103}  # this script's seed slots (two_zones_setup's purposes stop at 3)


def field_index() -> dict[str, slice | int]:
    out: dict[str, slice | int] = {name: i for i, name in enumerate(SCALARS)}
    for j, name in enumerate(VECTORS):
        start = len(SCALARS) + j * N_VARS
        out[name] = slice(start, start + N_VARS)
    return out


def seed(cfg: dict, conf: dict, image_index: int, slot: str, purpose: str) -> int:
    seq = np.random.SeedSequence([int(cfg["e4"]["walks"]["seed"]), int(conf["seed"]), int(image_index), QUADRANT_ORDER.index(slot),
                                  PURPOSE[purpose]])
    return int(seq.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def generator(value: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(value)
    return gen


def image_set(name: str, cfg: dict, conf: dict, model_id: str) -> list[dict]:
    """Entries {"key", "index", "colors" (slot_colors, corridor_color, agent_color), "origin" (None or {"index", colours}),
    "agent_x" (None or RGB), "info"}."""
    canon = {"index": CANONICAL_INDEX, "slot_colors": dict(conf["training_colors"]), "corridor_color": None, "agent_color": None}
    entries = []
    if name == "arc_ctrl":
        _, _, swaps = common.setup(cfg["_config_path"])
        base = {}
        for im in swaps:
            for p in im["probes"]:
                if p["axis"] == "arc_ctrl" and p["k"] == 0:
                    base[im["swap"]] = im
        for im in swaps:
            for p in im["probes"]:
                if p["axis"] != "arc_ctrl":
                    continue
                o = base[im["swap"]]
                entries.append({"key": f"i{im['index']}", "index": im["index"],
                                "colors": {"slot_colors": im["slot_colors"], "corridor_color": None, "agent_color": None},
                                "origin": {"index": o["index"], "slot_colors": o["slot_colors"], "corridor_color": None, "agent_color": None},
                                "agent_x": None, "info": {"swap": im["swap"], "axis": "arc_ctrl", "k": p["k"], "t": p["t"]}})
        if model_id.startswith("base"):
            o = base["se"]
            entries.append({"key": f"i{o['index']}_canonical", "index": o["index"],
                            "colors": {"slot_colors": o["slot_colors"], "corridor_color": None, "agent_color": None},
                            "origin": canon, "agent_x": None, "info": {"swap": "se", "axis": "se_t0_canonical", "k": 0, "t": 0.0}})
    elif name == "step4":
        entries.append({"key": f"i{CANONICAL_INDEX}", "index": CANONICAL_INDEX,
                        "colors": {k: v for k, v in canon.items() if k != "index"}, "origin": canon, "agent_x": None,
                        "info": {"family": "canonical", "name": "canonical training image"}})
        for im in stimuli.step4_images(conf):
            agent = None if im["family"] != "4c" else im["agent_color"]
            entries.append({"key": f"i{im['index']}", "index": im["index"],
                            "colors": {"slot_colors": im["slot_colors"], "corridor_color": im["corridor_color"], "agent_color": agent},
                            "origin": canon, "agent_x": agent, "info": {"family": im["family"], "name": im["name"], "min_de": im["min_de"]}})
    elif name == "training":
        for i, (ud, cd) in enumerate(permutations(QUADRANT_ORDER, 2)):
            colors = two_zones.training_colors(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], ud_slot=ud, cd_slot=cd)
            entries.append({"key": f"i{TRAINING_BASE + i}", "index": TRAINING_BASE + i,
                            "colors": {"slot_colors": colors, "corridor_color": None, "agent_color": None}, "origin": None,
                            "agent_x": None, "info": {"assignment": i, "ud": ud, "cd": cd}})
    return entries


def render_set(cfg: dict, conf: dict, colors: dict) -> np.ndarray:
    kw = {"corridor_color": colors["corridor_color"]} if colors["corridor_color"] is not None else {}
    return stimuli.images_for(cfg, conf, colors["slot_colors"], agent_color=colors["agent_color"], **kw)


def posterior(model, h: torch.Tensor, e: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    return torch.cat([model.posterior_probs(h[i:i + batch], e[i:i + batch]) for i in range(0, h.shape[0], batch)])


def cells(model, h: torch.Tensor, z: torch.Tensor, agent: torch.Tensor, kpx: int, batch: int = 1024) -> np.ndarray:
    return np.concatenate([decoded_agent_cell(model.decode(h[i:i + batch], z[i:i + batch]).clamp(0, 1), agent, kpx)
                           for i in range(0, h.shape[0], batch)])


def filtered(model, images: np.ndarray, walks: dict, valid: np.ndarray, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """(h, z, e) at the valid frames, in walk-major order."""
    ctx = fixed_trajectory_context(model, images, walks, generator=gen)
    e = encode_windows(model, images[np.clip(walks["states"], 0, None)], torch.device("cpu"))
    m = torch.as_tensor(valid)
    return ctx["h"][m], ctx["z"][m], e[m]


def measure_slot(model, cfg: dict, conf: dict, entry: dict, probe_imgs: np.ndarray, origin_cache: dict, slot: str,
                 walks: dict, brown: torch.Tensor, keep_frames: bool) -> tuple[np.ndarray, dict | None]:
    kpx = probe_imgs.shape[2] // two_zones.GRID_SHAPE[1]
    valid = walks["phase"] != walks_mod.PHASE_PAD
    states = walks["states"][valid].astype(np.int64)
    k = walks["steps_since_entry"][valid]
    walk = np.nonzero(valid)[0]
    actions = walks["actions"][valid]
    n_walks = walks["states"].shape[0]
    fi = field_index()
    out = np.zeros((n_walks, len(PARTS), N_FIELDS))
    with torch.no_grad():
        h, z, e = filtered(model, probe_imgs, walks, valid,
                           generator(two_zones_setup.measurement_seed(cfg, conf, entry["index"], slot, "probe")))
        post = posterior(model, h, e)
        a1 = cells(model, h, z, brown, kpx) == states
        top2 = post.topk(2, dim=-1).values.numpy()
        per = {"a1": a1, "sharp": top2[..., 0], "margin": top2[..., 0] - top2[..., 1]}
        if entry["agent_x"] is not None:
            agent_x = torch.tensor(entry["agent_x"], dtype=torch.float32)[:, None, None] / 255.0
            per["a1x"] = cells(model, h, z, agent_x, kpx) == states
        if entry["origin"] is not None:
            o = entry["origin"]
            key = (o["index"], slot)
            if key not in origin_cache:
                o_imgs = render_set(cfg, conf, o)
                ho, zo, eo = filtered(model, o_imgs, walks, valid, generator(seed(cfg, conf, o["index"], slot, "origin")))
                origin_cache[key] = (ho, posterior(model, ho, eo), cells(model, ho, zo, brown, kpx) == states)
            ho, post_o, o1 = origin_cache[key]
            post_a2 = posterior(model, ho, e)
            z_a2 = sample_posterior(post_a2, LatentEstimator(), generator(seed(cfg, conf, entry["index"], slot, "a2")))
            per["o1"] = o1
            per["a2"] = cells(model, ho, z_a2, brown, kpx) == states
            if entry["agent_x"] is not None:
                per["a2x"] = cells(model, ho, z_a2, agent_x, kpx) == states
            per["tv_a2"] = 0.5 * (post_a2 - post_o).abs().sum(-1).numpy()
            per["tv_a1"] = 0.5 * (post - post_o).abs().sum(-1).numpy()
        if entry["origin"] is not None or keep_frames is False:
            j = np.flatnonzero(k >= 1)  # zone frames; their predecessor is the walk's previous frame
            prev = j - 1
            assert (walk[prev] == walk[j]).all()
            a_onehot = F.one_hot(torch.as_tensor(actions[prev].astype(np.int64)), num_classes=model.cfg.n_actions).to(h.dtype)
            step = model.imagination_step(h[prev], z[prev], a_onehot, generator=generator(seed(cfg, conf, entry["index"], slot, "prior")))
            b_ok = np.zeros(len(states), dtype=bool)
            b_ok[j] = cells(model, step["h"], step["z"], brown, kpx) == states[j]
            has = np.zeros(len(states), dtype=bool)
            has[j] = True
            cond = np.zeros(len(states), dtype=bool)
            cond[j] = a1[prev]
            per.update({"b_all_n": has, "b_all_ok": b_ok & has, "b_n": cond, "b_ok": b_ok & cond})
    per["n"] = np.ones(len(states), dtype=bool)
    for p, sel in enumerate((k == 0, k >= 1, k == 1)):
        for name, values in per.items():
            values = np.asarray(values, dtype=np.float64)
            target = out[:, p, fi[name]]
            np.add.at(target, walk[sel], values[sel])
            out[:, p, fi[name]] = target
    frames = None
    if keep_frames:
        frames = {"cls": post.argmax(-1).numpy().astype(np.int8), "state": states.astype(np.int16), "walk": walk.astype(np.int16),
                  "k": k.astype(np.int16)}
    return out, frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--set", choices=SETS, required=True)
    parser.add_argument("--runs-root", type=Path, nargs="+", required=True)
    parser.add_argument("--probes-root", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-walks", type=int, default=100)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--images", type=int, nargs="+", default=None, help="positions in the set (default: all)")
    args = parser.parse_args(argv)
    model_spec = next(m for m in common.load_models() if m["id"] == args.model)
    weights = common.weights_of(model_spec, args.runs_root)
    pdir = common.probes_dir_of(model_spec, args.probes_root)
    meta_probe = common.probe_meta(pdir)
    sha = common.sha256(weights)
    if sha != meta_probe["weights_sha256"][model_spec["probes"]["model"]]:
        raise SystemExit(f"{args.model}: the local weights are not the probed ones")
    config = weights.parent.parent / "config.yaml"
    cfg, conf, _ = common.setup(str(config))
    cfg = {**cfg, "_config_path": str(config)}
    conf = {**conf, "seed": int(meta_probe["seed"])}  # the pair's measurement seed, as the probes
    torch.set_num_threads(args.threads)
    model = load_rssm(weights, torch.device("cpu"))
    brown = torch.tensor(conf["agent_color"], dtype=torch.float32)[:, None, None] / 255.0
    entries = image_set(args.set, cfg, conf, args.model)
    if args.images is not None:
        entries = [entries[i] for i in args.images]
    n_in = cfg["e4"]["walks"].get("n_in_grid", cfg["e4"]["walks"]["n_in"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"{args.model}__{args.set}__w{args.n_walks:03d}__{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    sums, frames, origin_cache = {}, {}, {}
    t0 = time.perf_counter()
    for entry in entries:
        probe_imgs = render_set(cfg, conf, entry["colors"])
        t1 = time.perf_counter()
        for slot in QUADRANT_ORDER:
            walks = two_zones_setup.walks_for_slot(cfg, slot, args.n_walks, n_in)
            arr, fr = measure_slot(model, cfg, conf, entry, probe_imgs, origin_cache, slot, walks, brown, args.set == "training")
            sums[f"{entry['key']}_{slot}"] = arr
            if fr is not None:
                for name, v in fr.items():
                    frames[f"{entry['key']}_{slot}_{name}"] = v
        print(f"[{args.model} {args.set}] {entry['key']} {entry['info']}: {time.perf_counter() - t1:.0f} s", flush=True)
    seconds = time.perf_counter() - t0
    np.savez_compressed(out / "walk_sums.npz", **sums)
    if frames:
        np.savez_compressed(out / "frames.npz", **frames)
    (out / "images.json").write_text(json.dumps([{k: v for k, v in e.items()} for e in entries], indent=1, default=list), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"model": args.model, "set": args.set, "weights": str(weights), "weights_sha256": sha,
                                               "pair_seed": conf["seed"], "n_walks": args.n_walks, "n_in": n_in, "threads": args.threads,
                                               "torch": torch.__version__, "seconds": seconds, "parts": PARTS, "fields": field_index(),
                                               "scalars": SCALARS, "vectors": VECTORS}, indent=1, default=lambda s: [s.start, s.stop]),
                                   encoding="utf-8")
    print(f"[{args.model} {args.set}] {len(entries)} images in {seconds:.0f} s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
