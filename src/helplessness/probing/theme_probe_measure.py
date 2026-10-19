"""The zero-data probes of a two-zone configuration at one checkpoint (E4's measure, docs/E4_generalization_gradient.md):
every (probe image, zone of interest), the fixed walks into that slot filtered by the exposed and the reference model
frozen at the checkpoint, I_s at every step in the zone (probing.fixed_trajectory.measure_walks), b = I_experimental -
I_reference. Moved verbatim from experiments/E4_generalization_gradient/measure_probes.py on 2026-10-19 so that E5
measures its models with the same code without importing E4; measure_probes.py keeps its command line and imports these.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.envs.appearance.theme_axes import rgb_distance
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup as e4_setup
from helplessness.pipeline.runtime import max_images_per_forward
from helplessness.probing.fixed_trajectory import measure_walks


def load_rssm(path: Path, device: torch.device) -> RSSM:
    state = torch.load(path, map_location="cpu", weights_only=False)
    rssm = RSSM(RSSMConfig(**state["rssm_config"]))
    rssm.load_state_dict(state["rssm"])
    return rssm.to(device).eval()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_extreme(image: dict) -> bool:
    """The training image, theme only (T_inc at the neutral slot) and position only."""
    return image["is_training"] or any((p["axis"], p["k"], p["position"]) in (("segment", 0, "neutral"), ("position_only", 0, "ud"))
                                       for p in image["probes"])


def measure(cfg: dict, conf: dict, models: dict, images: list[dict], device, log=lambda s: print(s, flush=True), save_steps: bool = False,
            n_walks: int | None = None) -> dict:
    """Every (image, zone) of `images` on both models. Returns rows (images.csv), curves and the optional steps."""
    w, draws = cfg["e4"]["walks"], cfg["e4"]["probe"]["draws"]
    margin = cfg["measure"]["kernel_margin_threshold"]
    role_of = {slot: role for role, slot in conf["roles"].items()}
    rows, curves_exp, curves_ref, n_states_curve, steps = [], [], [], [], {}
    for image in images:
        world = e4_setup.probe_world(cfg, conf, image["slot_colors"])
        succ = e4_setup.successors(world)
        n_in = w["n_in"] if is_extreme(image) else w.get("n_in_grid", w["n_in"])
        for slot in image["zones"]:
            walks = e4_setup.walks_for_slot(cfg, slot, n_walks, n_in)
            seed = e4_setup.measurement_seed(cfg, conf, image["index"], slot, "probe")
            t0 = time.perf_counter()
            res = {name: measure_walks(model, world.templates_np, world.templates_t, walks, succ, draws, margin, seed, w["m"], n_in,
                                       chunk=w["chunk"], max_images=max_images_per_forward(cfg))
                   for name, model in models.items()}
            seconds = time.perf_counter() - t0
            e, r = res["experimental"], res["reference"]
            rows.append({"image": image["index"], "is_training": image["is_training"], "extreme": is_extreme(image), "zone": slot,
                         "role": role_of[slot], "n_in": n_in, "n_walks": int(walks["states"].shape[0]),
                         **{f"color_{q}": "-".join(map(str, image["slot_colors"][q])) for q in QUADRANT_ORDER},
                         "I_entry_exp": e["entry"], "I_entry_ref": r["entry"], "b_entry": e["entry"] - r["entry"],
                         "n_states_entry_exp": e["n_states_entry"], "n_states_entry_ref": r["n_states_entry"],
                         "ghost_exp": float(np.nanmean(e["ghost_fraction"])), "ghost_ref": float(np.nanmean(r["ghost_fraction"])),
                         "mass_true_exp": float(np.nanmean(e["mass_true"])), "mass_true_ref": float(np.nanmean(r["mass_true"])),
                         "decodes_exp": e["n_unique_decodes"], "decodes_ref": r["n_unique_decodes"], "seconds": seconds,
                         "n_probes": len(image["probes"])})
            pad = w["n_in"] - n_in  # every curve on 1..n_in (NaN beyond a grid image's n_in_grid)
            curves_exp.append(np.concatenate([e["curve"], np.full(pad, np.nan)]))
            curves_ref.append(np.concatenate([r["curve"], np.full(pad, np.nan)]))
            n_states_curve.append(np.concatenate([np.minimum(e["n_states_curve"], r["n_states_curve"]), np.zeros(pad, dtype=np.int64)]))
            if save_steps:
                key = f"i{image['index']:03d}_{slot}"
                for field in ("states", "steps_since_entry", "walk"):  # the same walks for both models
                    steps[f"{key}_{field}"] = np.asarray(res["experimental"][field], dtype=np.int16)
                for name, res_m in res.items():
                    steps[f"{key}_{name}_I"] = np.asarray(res_m["I"], dtype=np.float32)
            log(f"image {image['index']:3d} zone {slot} ({role_of[slot]}): b_entry {rows[-1]['b_entry']:+.3f} "
                f"(I exp {e['entry']:.3f}, ref {r['entry']:.3f}; {seconds:.0f} s)")
    return {"rows": rows, "curve_exp": np.array(curves_exp), "curve_ref": np.array(curves_ref),
            "n_states_curve": np.array(n_states_curve), "steps": steps}


def probe_rows(conf: dict, images: list[dict], rows: list[dict]) -> list[dict]:
    by_key = {(r["image"], r["zone"]): r for r in rows}
    out = []
    for image in images:
        for p in image["probes"]:
            r = by_key.get((image["index"], p["slot"]))
            if r is None:
                continue
            color = image["slot_colors"][p["slot"]]
            out.append({"axis": p["axis"], "k": p["k"], "t": p["t"], "position": p["position"], "slot": p["slot"], "image": image["index"],
                        "color": "-".join(map(str, color)), "dist_to_inc": float(rgb_distance(color, conf["t_inc"])),
                        "dist_to_ctrl": float(rgb_distance(color, conf["t_ctrl"])), "b_entry": r["b_entry"],
                        "I_entry_exp": r["I_entry_exp"], "I_entry_ref": r["I_entry_ref"]})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
