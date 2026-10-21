"""E4's final analysis of the probe grid (docs/E4_generalization_gradient.md), local, CPU. Reads measure_probes.py
outputs (one directory per configuration, 300 walks) and the runs' final weights from the extracted export archives;
writes CSV tables and figures into --out (a new directory). Sub-commands:

  validity  needs the weights: for every (image, zone) of a configuration, the REFERENCE model's prior at the FIRST step
            in the zone (the fixed walks cut after that step, the first --n-walks walks, its own measurement seed): the
            mass on the true successor, the ghost fraction and I, by state then by zone. measure_probes.py keeps the
            mass and the ghosts only as means over every step in the zone; the validity criterion needs them at step 1.
  diagnose  needs the weights: for chosen images and zones, along the fixed walks (both models): the reconstruction of
            every observed frame (the posterior decoded) against the true frame (mean squared error, in [0, 1] pixel
            units) and the agent's decoded position (the state whose image is nearest to the reconstruction, as the
            kernel's matching does -- biased towards the states inside a quadrant whose colour the reconstruction does
            not render) against the true state; the agent's cell located directly in the reconstruction (the cell
            holding the pixel nearest to the agent's colour) against the true cell; the mean colour the reconstruction
            gives the zone of interest (the agent's cell left out) against the probe's colour and the training image's
            colour of that slot; the reconstruction error outside the zone of interest. Accuracies and toric grid
            distances, in the corridor and in the zone of interest (and at its first step).
  report    no weights: from the probe directories (and the validity tables), the coherence of the cluster's extreme
            probes with local measurements, the validity of every probe (reference mass at step 1 >= --mass-min and
            |I_ref - log2 5| <= --i-tol, I_ref from the cluster's steps), the gradients of b at step 1 (by axis and
            position, per configuration and averaged over the configurations where the probe is valid), the key
            comparisons, the recovery curves, and the standard error of b at step 1 against the number of walks.

b at step 1 is the main measure (first encounter, over the zone's border states); steps 2..n_in are the recovery.
Standard errors: bootstrap over the walks (pilot_analysis.b_resampled, the same walks for both models).

Usage, from the repo root:
  python experiments/E4_generalization_gradient/final_analysis.py validity --export <dir> --configuration p2-a-d1 --out <dir>
  python experiments/E4_generalization_gradient/final_analysis.py diagnose --export <dir> --configuration p2-a-d1 \\
      --images 0 2 92 --out <dir>
  python experiments/E4_generalization_gradient/final_analysis.py report --export <dir> <dir> --validity <csv> ... \\
      [--local-probes <dir> ...] --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import e4_setup
from helplessness.envs.appearance.theme_axes import rgb_distance
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.probing import walks as walks_mod
from helplessness.probing.fixed_trajectory import fixed_trajectory_context, measure_walks, template_distances_torch
from helplessness.probing.theme_probes import probe_images
from helplessness.pipeline.runtime import max_images_per_forward
from measure_probes import load_rssm
from pilot_analysis import b_resampled, export_dirs, find_in_export, probe_measures

LOG2_5 = float(np.log2(5))
POSITIONS = ("ud", "cd", "neutral")
AXES = ("segment", "arc_ctrl", "arc_inc")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def by_state_then_zone(values: np.ndarray, states: np.ndarray) -> float:
    ok = np.isfinite(values)
    if not ok.any():
        return float("nan")
    uniq, inv = np.unique(states[ok], return_inverse=True)
    return float((np.bincount(inv, weights=values[ok]) / np.bincount(inv)).mean())


def _models(export: list[Path], config_id: str, which: tuple[str, ...]) -> dict:
    out = {}
    for root in export:
        try:
            for name in which:
                run = find_in_export(root, config_id, name)
                out[name] = load_rssm(run / "checkpoints" / "agent_env300000.pt", torch.device("cpu"))
            return out
        except SystemExit:
            continue
    raise SystemExit(f"no runs of {config_id} in {export}")


# ---------------------------------------------------------------------------------------------------------- validity

def cmd_validity(args) -> None:
    cfg = e4_setup.load_config(args.config)
    conf = e4_setup.configuration(cfg, args.configuration)
    models = _models(args.export, args.configuration, tuple(args.models))
    images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    draws, margin = cfg["e4"]["probe"]["draws"], cfg["measure"]["kernel_margin_threshold"]
    role_of = {slot: role for role, slot in conf["roles"].items()}
    rows = []
    for image in images:
        world = e4_setup.probe_world(cfg, conf, image["slot_colors"])
        succ = e4_setup.successors(world)
        for slot in image["zones"]:
            walks = e4_setup.walks_for_slot(cfg, slot, args.n_walks, n_in=1)
            seed = e4_setup.measurement_seed(cfg, conf, image["index"], slot, "probe")  # the probe's own seed
            t0 = time.perf_counter()
            row = {"configuration": args.configuration, "image": image["index"], "zone": slot, "role": role_of[slot],
                   **{f"color_{q}": "-".join(map(str, image["slot_colors"][q])) for q in QUADRANT_ORDER}, "n_walks": args.n_walks}
            for name, model in models.items():
                res = measure_walks(model, world.templates_np, world.templates_t, walks, succ, draws, margin, seed, 1, 1,
                                    chunk=cfg["e4"]["walks"]["chunk"], max_images=max_images_per_forward(cfg))
                tag = "ref" if name == "reference" else "exp"
                for field, key in (("mass_true", "mass"), ("ghost_fraction", "ghost"), ("I", "I")):
                    values = np.asarray(res[field], dtype=np.float64)
                    if values.ndim == 2:  # per action: the mean over the actions, as the training diagnostic does
                        values = values.mean(axis=1)
                    row[f"{key}_{tag}_step1"] = by_state_then_zone(values, res["states"])
                row[f"n_states_{tag}_step1"] = int(np.unique(res["states"]).size)
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            print(f"{args.configuration} image {image['index']:3d} {slot}: mass_ref {row['mass_ref_step1']:.3f} ghost_ref "
                  f"{row['ghost_ref_step1']:.3f} I_ref {row['I_ref_step1']:.3f} ({row['seconds']:.0f} s)", flush=True)
    _write_csv(args.out / f"validity_{args.configuration}.csv", rows)


# ---------------------------------------------------------------------------------------------------------- diagnose

def toric_cells(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    R, C = two_zones.GRID_SHAPE
    ra, ca = np.divmod(a, C)
    rb, cb = np.divmod(b, C)
    dr, dc = np.abs(ra - rb), np.abs(ca - cb)
    return np.minimum(dr, R - dr) + np.minimum(dc, C - dc)


def cmd_diagnose(args) -> None:
    cfg = e4_setup.load_config(args.config)
    conf = e4_setup.configuration(cfg, args.configuration)
    models = _models(args.export, args.configuration, ("experimental", "reference"))
    images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    rows = []
    for idx in args.images:
        image = images[idx]
        world = e4_setup.probe_world(cfg, conf, image["slot_colors"])
        tf = world.templates_t.reshape(world.templates_t.shape[0], -1).double()
        tsq = (tf ** 2).sum(1)
        R, C = two_zones.GRID_SHAPE
        kpx = world.templates_t.shape[-1] // C
        agent = torch.tensor(conf["agent_color"], dtype=torch.float32)[:, None, None] / 255.0
        for slot in image["zones"]:
            cells = np.zeros((R, C), dtype=bool)
            for st in two_zones.region_states(world.layout)[slot]:
                cells[divmod(int(st), C)] = True
            zone_px = torch.as_tensor(np.kron(cells, np.ones((kpx, kpx), dtype=bool)))
            probe_rgb = np.array(image["slot_colors"][slot], dtype=float)
            train_rgb = np.array(conf["training_colors"][slot], dtype=float)
            walks = e4_setup.walks_for_slot(cfg, slot, args.n_walks, n_in=args.n_in)
            valid = walks["phase"] != walks_mod.PHASE_PAD
            states = walks["states"][valid]
            k = walks["steps_since_entry"][valid]
            for name, model in models.items():
                gen = torch.Generator().manual_seed(e4_setup.measurement_seed(cfg, conf, idx, slot, "probe"))
                with torch.no_grad():
                    ctx = fixed_trajectory_context(model, world.templates_np, walks, generator=gen)
                    h, z = ctx["h"][torch.as_tensor(valid)], ctx["z"][torch.as_tensor(valid)]
                    nearest, mse, mse_out, agent_cell, zone_rgb = [], [], [], [], []
                    for i in range(0, h.shape[0], 1024):
                        dec = model.decode(h[i:i + 1024], z[i:i + 1024])
                        nearest.append(template_distances_torch(dec, tf, tsq).argmin(axis=1))
                        dec = dec.clamp(0, 1)
                        truth = world.templates_t[states[i:i + 1024]]
                        err = ((dec - truth) ** 2).mean(dim=1)
                        mse.append(err.mean(dim=(1, 2)).numpy())
                        mse_out.append(err[:, ~zone_px].mean(dim=1).numpy())
                        d_agent = ((dec - agent) ** 2).sum(dim=1)  # (n, H, W)
                        per_cell = d_agent.reshape(-1, R, kpx, C, kpx).amin(dim=(2, 4)).reshape(-1, R * C)
                        agent_cell.append(per_cell.argmin(dim=1).numpy())
                        n = dec.shape[0]
                        own = torch.zeros(n, R * C, dtype=torch.bool)
                        own[torch.arange(n), torch.as_tensor(states[i:i + n].astype(np.int64))] = True
                        own_px = own.reshape(n, R, 1, C, 1).expand(n, R, kpx, C, kpx).reshape(n, R * kpx, C * kpx)
                        mask = zone_px[None] & ~own_px
                        rgb = (dec * mask[:, None]).sum(dim=(2, 3)) / mask.sum(dim=(1, 2))[:, None]
                        zone_rgb.append((rgb * 255.0).numpy())
                nearest, mse, mse_out = np.concatenate(nearest), np.concatenate(mse), np.concatenate(mse_out)
                agent_cell, zone_rgb = np.concatenate(agent_cell), np.concatenate(zone_rgb)
                dist = toric_cells(nearest, states)
                dist_agent = toric_cells(agent_cell, states)
                for part, sel in (("corridor", k == 0), ("zone", k >= 1), ("zone_step1", k == 1)):
                    rows.append({"configuration": args.configuration, "image": idx, "is_training": image["is_training"],
                                 "probes": ";".join(f"{p['axis']}:{p['k']}:{p['position']}" for p in image["probes"]) or "training",
                                 "zone": slot, "model": name, "part": part, "n_frames": int(sel.sum()),
                                 "recon_mse": float(mse[sel].mean()), "recon_mse_outside_zone": float(mse_out[sel].mean()),
                                 "nearest_template_accuracy": float((nearest[sel] == states[sel]).mean()),
                                 "nearest_template_error_cells": float(dist[sel].mean()),
                                 "agent_cell_accuracy": float((agent_cell[sel] == states[sel]).mean()),
                                 "agent_cell_error_cells": float(dist_agent[sel].mean()),
                                 "zone_rgb_reconstructed": "-".join(f"{v:.0f}" for v in zone_rgb[sel].mean(axis=0)),
                                 "zone_rgb_dist_to_probe": float(np.linalg.norm(zone_rgb[sel].mean(axis=0) - probe_rgb)),
                                 "zone_rgb_dist_to_training": float(np.linalg.norm(zone_rgb[sel].mean(axis=0) - train_rgb)),
                                 **{f"color_{q}": "-".join(map(str, image["slot_colors"][q])) for q in QUADRANT_ORDER}})
                r = rows[-2]
                print(f"{args.configuration} image {idx} {slot} {name}: zone recon {r['recon_mse']:.5f} agent cell acc "
                      f"{r['agent_cell_accuracy']:.3f} nearest template acc {r['nearest_template_accuracy']:.3f} zone rgb "
                      f"{r['zone_rgb_reconstructed']}", flush=True)
    _write_csv(args.out / f"diagnose_{args.configuration}.csv", rows)


# ------------------------------------------------------------------------------------------------------------ report

def load_probe_dir(d: Path) -> dict:
    meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
    images, probes = probe_measures(d)
    steps = dict(np.load(d / "steps.npz")) if (d / "steps.npz").is_file() else None
    return {"dir": d, "meta": meta, "configuration": meta["configuration"], "images": images, "probes": probes, "steps": steps}


def n_walks_of(p: dict) -> int:
    return int(p["meta"].get("walks", {}).get("n_walks") or p["meta"].get("n_walks") or p["images"][0]["n_walks"])


def step1_se(p: dict, image: int, slot: str, rng: np.random.Generator, n_boot: int, n: int | None = None) -> float:
    if p["steps"] is None:
        return float("nan")
    key = f"i{int(image):03d}_{slot}"
    n_total = int(p["steps"][f"{key}_walk"].max()) + 1
    n = n_total if n is None else n
    draws = [b_resampled(p["steps"], key, np.bincount(rng.integers(0, n_total, n), minlength=n_total), 1, 1) for _ in range(n_boot)]
    return float(np.nanstd(draws, ddof=1))


def extreme_rows(p: dict) -> dict:
    """(probe, zone role) -> the images.csv row, for the training image (every zone), theme only and position only."""
    out = {}
    for r in p["images"]:
        if r["is_training"] == "True":
            out[("training", r["role"])] = r
    for q in p["probes"]:
        if q["axis"] == "segment" and int(q["k"]) == 0 and q["position"] == "neutral":
            out[("theme_only", "neutral")] = next(r for r in p["images"] if r["image"] == q["image"] and r["zone"] == q["slot"])
        if q["axis"] == "position_only":
            out[("position_only", "ud")] = next(r for r in p["images"] if r["image"] == q["image"] and r["zone"] == q["slot"])
    return out


def cmd_report(args) -> None:
    plt = _plt()
    cfg = e4_setup.load_config(args.config)
    rng = np.random.default_rng(0)
    tables, figures = args.out / "tables", args.out / "figures"
    tables.mkdir()
    figures.mkdir()
    dirs = [d for root in args.export for d in export_dirs(root)[1]]
    cluster = {}
    for d in dirs:
        p = load_probe_dir(d)
        if n_walks_of(p) != args.n_walks:
            print(f"skipped {d.name}: {n_walks_of(p)} walks", flush=True)
            continue
        if p["configuration"] in cluster:
            raise SystemExit(f"two probe directories with {args.n_walks} walks for {p['configuration']}")
        cluster[p["configuration"]] = p
    configs = sorted(cluster)
    confs = {c: e4_setup.configuration(cfg, c) for c in configs}
    used = [{"configuration": c, "probes_dir": str(cluster[c]["dir"]), "n_walks": n_walks_of(cluster[c]),
             "checkpoint": cluster[c]["meta"].get("checkpoint_env_step"), "n_image_zones": len(cluster[c]["images"])} for c in configs]
    _write_csv(tables / "probes_dirs_used.csv", used)

    # 1. coherence of the cluster's extreme probes with the local CPU measurements
    local = {}
    for d in args.local_probes:
        p = load_probe_dir(d)
        local[p["configuration"]] = p
    coherence = []
    for c in configs:
        if c not in local:
            continue
        ce, le = extreme_rows(cluster[c]), extreme_rows(local[c])
        for key in sorted(ce):
            if key not in le:
                continue
            a, b = ce[key], le[key]
            se_a = step1_se(cluster[c], a["image"], a["zone"], rng, args.bootstrap)
            se_b = step1_se(local[c], b["image"], b["zone"], rng, args.bootstrap)
            diff = a["b_step1"] - b["b_step1"]
            coherence.append({"configuration": c, "probe": key[0], "zone_role": key[1], "b_step1_cluster": a["b_step1"], "se_cluster": se_a,
                              "b_step1_local": b["b_step1"], "se_local": se_b, "difference": diff,
                              "z": diff / np.sqrt(se_a ** 2 + se_b ** 2) if se_a > 0 and se_b > 0 else float("nan")})
    _write_csv(tables / "coherence_cluster_vs_local.csv", coherence)

    # 2. validity of every (image, zone)
    vrows = {}
    for path in args.validity:
        for r in _read_csv(path):
            vrows[(r["configuration"], r["image"], r["zone"])] = r
    pair_rows = []
    for c in configs:
        conf = confs[c]
        p = cluster[c]
        for r in p["images"]:
            v = vrows.get((c, r["image"], r["zone"]))
            key = f"i{int(r['image']):03d}_{r['zone']}"
            I_ref1 = by_state_then_zone(p["steps"][f"{key}_reference_I"].astype(np.float64)[p["steps"][f"{key}_steps_since_entry"] == 1],
                                        p["steps"][f"{key}_states"][p["steps"][f"{key}_steps_since_entry"] == 1])
            I_exp1 = by_state_then_zone(p["steps"][f"{key}_experimental_I"].astype(np.float64)[p["steps"][f"{key}_steps_since_entry"] == 1],
                                        p["steps"][f"{key}_states"][p["steps"][f"{key}_steps_since_entry"] == 1])
            mass = float(v["mass_ref_step1"]) if v else float("nan")
            ghost = float(v["ghost_ref_step1"]) if v else float("nan")
            valid = bool(mass >= args.mass_min and abs(I_ref1 - LOG2_5) <= args.i_tol)
            colors = {q: tuple(int(x) for x in r[f"color_{q}"].split("-")) for q in QUADRANT_ORDER}
            train_slot = conf["training_colors"][r["zone"]]
            pair_rows.append({"configuration": c, "image": int(r["image"]), "zone": r["zone"], "role": r["role"],
                              "is_training": r["is_training"] == "True", "color_zone": r[f"color_{r['zone']}"],
                              "training_color_zone": "-".join(map(str, train_slot)),
                              "dist_to_training_color_of_zone": float(rgb_distance(colors[r["zone"]], train_slot)),
                              "dist_to_agent_color": float(rgb_distance(colors[r["zone"]], conf["agent_color"])),
                              **{f"color_{q}": r[f"color_{q}"] for q in QUADRANT_ORDER},
                              "n_slots_T_inc": sum(colors[q] == conf["t_inc"] for q in QUADRANT_ORDER),
                              "n_slots_T_ctrl": sum(colors[q] == conf["t_ctrl"] for q in QUADRANT_ORDER),
                              "I_ref_step1": I_ref1, "I_exp_step1": I_exp1, "b_step1": r["b_step1"],
                              "mass_ref_step1": mass, "ghost_ref_step1": ghost,
                              "I_ref_step1_local": float(v["I_ref_step1"]) if v else float("nan"),
                              "mass_ref_mean_all_steps": float(r["mass_true_ref"]), "ghost_ref_mean_all_steps": float(r["ghost_ref"]),
                              "valid": valid, "se_b_step1": step1_se(p, r["image"], r["zone"], rng, args.bootstrap),
                              "b_curve": [float(x) for x in r["b_curve"]]})
    pair_of = {(r["configuration"], r["image"], r["zone"]): r for r in pair_rows}
    _write_csv(tables / "validity_image_zone.csv", [{k: v for k, v in r.items() if k != "b_curve"} for r in pair_rows])

    # every probe with its normalised abscissae, b at step 1, its SE and its validity
    probe_rows = []
    for c in configs:
        conf = confs[c]
        D = float(rgb_distance(conf["t_inc"], conf["t_ctrl"]))
        for q in cluster[c]["probes"]:
            pr = pair_of[(c, int(q["image"]), q["slot"])]
            probe_rows.append({"configuration": c, "pair": conf["pair"], "inc": conf["inc"], "seed": conf["seed"],
                               "T_inc": "-".join(map(str, conf["t_inc"])), "T_ctrl": "-".join(map(str, conf["t_ctrl"])), "D": D,
                               "axis": q["axis"], "k": int(q["k"]), "position": q["position"], "slot": q["slot"], "image": int(q["image"]),
                               "color": q["color"], "dist_to_inc_over_D": float(q["dist_to_inc"]) / D,
                               "dist_to_ctrl_over_D": float(q["dist_to_ctrl"]) / D,
                               "x": (float(q["dist_to_ctrl"]) if q["axis"] == "arc_inc" else float(q["dist_to_inc"])) / D,
                               "b_step1": pr["b_step1"], "se_b_step1": pr["se_b_step1"], "b_entry_mean_m": float(q["b_entry"]),
                               "dist_to_training_color_of_zone_over_D": pr["dist_to_training_color_of_zone"] / D,
                               "I_ref_step1": pr["I_ref_step1"], "mass_ref_step1": pr["mass_ref_step1"],
                               "ghost_ref_step1": pr["ghost_ref_step1"], "valid": pr["valid"]})
    _write_csv(tables / "probes_b_step1.csv", probe_rows)
    invalid = [r for r in probe_rows if not r["valid"]]
    counts = []
    for c in configs:
        for axis in (*AXES, "position_only"):
            for position in POSITIONS:
                sel = [r for r in probe_rows if r["configuration"] == c and r["axis"] == axis and r["position"] == position]
                if sel:
                    counts.append({"configuration": c, "axis": axis, "position": position, "probes": len(sel),
                                   "invalid": sum(not r["valid"] for r in sel),
                                   "invalid_k": " ".join(str(r["k"]) for r in sel if not r["valid"])})
    _write_csv(tables / "invalid_counts.csv", counts)
    bins = [0.0, 1.0, 30.0, 60.0, 90.0, 120.0, 150.0, 200.0, 250.0, 450.0]
    by_dist = []
    for c in configs:
        for lo, hi in zip(bins, bins[1:]):
            sel = [r for r in pair_rows if r["configuration"] == c and lo <= r["dist_to_training_color_of_zone"] < hi]
            if sel:
                by_dist.append({"configuration": c, "dist_to_training_color_of_zone_from": lo, "to": hi, "image_zones": len(sel),
                                "valid": sum(r["valid"] for r in sel), "median_mass_ref_step1": float(np.median([r["mass_ref_step1"] for r in sel])),
                                "median_I_ref_step1": float(np.median([r["I_ref_step1"] for r in sel]))})
    _write_csv(tables / "validity_by_distance_to_training_color.csv", by_dist)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for c in configs:
        sel = [r for r in pair_rows if r["configuration"] == c]
        axs[0].scatter([r["dist_to_training_color_of_zone"] for r in sel], [r["mass_ref_step1"] for r in sel], s=12, label=c)
        axs[1].scatter([r["dist_to_training_color_of_zone"] for r in sel], [r["I_ref_step1"] for r in sel], s=12, label=c)
    axs[0].axhline(args.mass_min, color="k", ls="--")
    axs[1].axhline(LOG2_5 - args.i_tol, color="k", ls="--")
    axs[0].set_ylabel("reference: mass on the true successor at step 1")
    axs[1].set_ylabel("reference: I at step 1 (bits)")
    for ax in axs:
        ax.set_xlabel("RGB distance of the probed colour to the zone's training colour")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "validity_vs_distance_to_training_color.png", dpi=120)
    plt.close(fig)
    _write_csv(tables / "invalid_probes.csv", invalid)
    # the invalid map
    fig, axes_ = plt.subplots(1, len(configs), figsize=(4.2 * len(configs), 4.2), squeeze=False)
    for ax, c in zip(axes_[0], configs):
        rows_, labels = [], []
        for axis in AXES:
            for position in POSITIONS:
                line = np.full(11, np.nan)
                for r in probe_rows:
                    if r["configuration"] == c and r["axis"] == axis and r["position"] == position:
                        line[r["k"]] = 0.0 if r["valid"] else 1.0
                rows_.append(line)
                labels.append(f"{axis} / {position}")
        po = [r for r in probe_rows if r["configuration"] == c and r["axis"] == "position_only"]
        line = np.full(11, np.nan)
        if po:
            line[0] = 0.0 if po[0]["valid"] else 1.0
        rows_.append(line)
        labels.append("position only / ud")
        ax.imshow(np.array(rows_), cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes_[0][0] else [""] * len(labels), fontsize=7)
        ax.set_xticks(range(11))
        ax.set_xlabel("k (point on the axis)")
        ax.set_title(f"{c}: invalid probes (red)\nT_inc {confs[c]['t_inc']}", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "invalid_map.png", dpi=130)
    plt.close(fig)

    # 3. gradients: per configuration and averaged over the configurations where the probe is valid
    mean_rows = []
    for axis in AXES:
        for position in POSITIONS:
            for k in range(11):
                sel = [r for r in probe_rows if r["axis"] == axis and r["position"] == position and r["k"] == k]
                ok = [r for r in sel if r["valid"]]
                vals = np.array([r["b_step1"] for r in ok])
                mean_rows.append({"axis": axis, "position": position, "k": k, "x": float(np.mean([r["x"] for r in sel])),
                                  "n_configs_valid": len(ok), "configs_valid": " ".join(r["configuration"] for r in ok),
                                  "b_step1_mean": float(vals.mean()) if ok else float("nan"),
                                  "b_step1_sd_between_configs": float(vals.std(ddof=1)) if len(ok) > 1 else float("nan"),
                                  "b_step1_min": float(vals.min()) if ok else float("nan"),
                                  "b_step1_max": float(vals.max()) if ok else float("nan"),
                                  "n_negative": int((vals < 0).sum()), **{f"b_{r['configuration']}": r["b_step1"] for r in sel},
                                  **{f"valid_{r['configuration']}": r["valid"] for r in sel}})
    _write_csv(tables / "gradients_mean.csv", mean_rows)
    colors_cfg = dict(zip(configs, ("tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown")))
    xlabels = {"segment": "distance to T_inc / D", "arc_ctrl": "distance to T_inc / D (distance to T_ctrl = D)",
               "arc_inc": "distance to T_ctrl / D (distance to T_inc = D)"}
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    for i, axis in enumerate(AXES):
        for j, position in enumerate(POSITIONS):
            ax = axs[i][j]
            for c in configs:
                sel = sorted((r for r in probe_rows if r["configuration"] == c and r["axis"] == axis and r["position"] == position),
                             key=lambda r: r["k"])
                xs = np.array([r["x"] for r in sel])
                ys = np.array([r["b_step1"] if r["valid"] else np.nan for r in sel])
                es = np.array([r["se_b_step1"] if r["valid"] else np.nan for r in sel])
                ax.errorbar(xs, ys, yerr=es, color=colors_cfg[c], marker=".", lw=0.8, capsize=2, alpha=0.8, label=c)
                bad = [r for r in sel if not r["valid"]]
                ax.plot([r["x"] for r in bad], [r["b_step1"] for r in bad], "x", color=colors_cfg[c], alpha=0.5)
            m = sorted((r for r in mean_rows if r["axis"] == axis and r["position"] == position), key=lambda r: r["k"])
            ax.plot([r["x"] for r in m], [r["b_step1_mean"] for r in m], color="k", lw=2, label="mean (valid)")
            ax.axhline(0, color="grey", lw=0.5)
            ax.axhline(-LOG2_5, color="k", ls=":", lw=0.8)
            ax.set_title(f"{axis} -- position {position}", fontsize=9)
            ax.set_xlabel(xlabels[axis], fontsize=8)
            if j == 0:
                ax.set_ylabel("b at step 1 (bits)")
    axs[0][0].legend(fontsize=7)
    fig.suptitle("b at the first step in the zone; x = invalid probe (excluded from the mean); dotted: -log2 5", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / "gradients_by_axis_position.png", dpi=120)
    plt.close(fig)
    for axis in AXES:  # one figure per axis: mean with the spread between configurations, by position
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for position, col in zip(POSITIONS, ("tab:red", "tab:blue", "tab:gray")):
            m = sorted((r for r in mean_rows if r["axis"] == axis and r["position"] == position), key=lambda r: r["k"])
            x = np.array([r["x"] for r in m])
            y = np.array([r["b_step1_mean"] for r in m])
            lo, hi = np.array([r["b_step1_min"] for r in m]), np.array([r["b_step1_max"] for r in m])
            ax.plot(x, y, marker="o", color=col, label=f"position {position}")
            ax.fill_between(x, lo, hi, color=col, alpha=0.15)
        po = [r for r in probe_rows if r["axis"] == "position_only" and r["valid"]]
        if po:
            ax.errorbar([1.02] * len(po), [r["b_step1"] for r in po], fmt="k*", label="position only (valid configs)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axhline(-LOG2_5, color="k", ls=":", lw=0.8)
        ax.set_xlabel(xlabels[axis])
        ax.set_ylabel("b at step 1 (bits), mean over valid configs; band = min-max")
        ax.set_title(f"{axis}: mean over configurations")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"gradient_mean_{axis}.png", dpi=120)
        plt.close(fig)

    # 4. comparisons
    pos_rows = []
    for axis in AXES:
        for k in range(11):
            for c in configs:
                get = {r["position"]: r for r in probe_rows if r["configuration"] == c and r["axis"] == axis and r["k"] == k}
                if set(get) != set(POSITIONS):
                    continue
                pos_rows.append({"axis": axis, "k": k, "x": get["ud"]["x"], "configuration": c, "color": get["ud"]["color"],
                                 **{f"b_{pos}": get[pos]["b_step1"] if get[pos]["valid"] else float("nan") for pos in POSITIONS},
                                 **{f"se_{pos}": get[pos]["se_b_step1"] for pos in POSITIONS},
                                 "ud_minus_cd": (get["ud"]["b_step1"] - get["cd"]["b_step1"]) if get["ud"]["valid"] and get["cd"]["valid"] else float("nan"),
                                 "ud_minus_neutral": (get["ud"]["b_step1"] - get["neutral"]["b_step1"]) if get["ud"]["valid"] and get["neutral"]["valid"] else float("nan")})
    _write_csv(tables / "comparison_position.csv", pos_rows)
    seg_arc = []
    for position in POSITIONS:
        for k in range(11):
            for c in configs:
                s = next((r for r in probe_rows if r["configuration"] == c and r["axis"] == "segment" and r["position"] == position and r["k"] == k), None)
                a = next((r for r in probe_rows if r["configuration"] == c and r["axis"] == "arc_ctrl" and r["position"] == position and r["k"] == k), None)
                if s is None or a is None:
                    continue
                both = s["valid"] and a["valid"]
                seg_arc.append({"position": position, "k": k, "dist_to_inc_over_D_segment": s["dist_to_inc_over_D"],
                                "dist_to_inc_over_D_arc": a["dist_to_inc_over_D"], "dist_to_ctrl_over_D_segment": s["dist_to_ctrl_over_D"],
                                "configuration": c, "b_segment": s["b_step1"] if s["valid"] else float("nan"),
                                "b_arc_ctrl": a["b_step1"] if a["valid"] else float("nan"),
                                "segment_minus_arc": s["b_step1"] - a["b_step1"] if both else float("nan"),
                                "se_difference": float(np.hypot(s["se_b_step1"], a["se_b_step1"]))})
    _write_csv(tables / "comparison_segment_vs_arc_ctrl.csv", seg_arc)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, position in zip(axs, POSITIONS):
        for name, axis, ls in (("segment", "segment", "-"), ("arc centred on T_ctrl", "arc_ctrl", "--")):
            m = sorted((r for r in mean_rows if r["axis"] == axis and r["position"] == position), key=lambda r: r["k"])
            ax.plot([r["x"] for r in m], [r["b_step1_mean"] for r in m], ls=ls, color="k", marker="o", label=f"{name} (mean)")
            for c in configs:
                sel = sorted((r for r in probe_rows if r["configuration"] == c and r["axis"] == axis and r["position"] == position),
                             key=lambda r: r["k"])
                ax.plot([r["x"] for r in sel], [r["b_step1"] if r["valid"] else np.nan for r in sel], ls=ls, color=colors_cfg[c], lw=0.8, alpha=0.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"position {position}: segment vs arc centred on T_ctrl", fontsize=9)
        ax.set_xlabel("distance to T_inc / D")
    axs[0].set_ylabel("b at step 1 (bits)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "segment_vs_arc_ctrl.png", dpi=120)
    plt.close(fig)
    theme_only = []
    for c in configs:
        r = next(r for r in probe_rows if r["configuration"] == c and r["axis"] == "segment" and r["k"] == 0 and r["position"] == "neutral")
        po = next(r for r in probe_rows if r["configuration"] == c and r["axis"] == "position_only")
        theme_only.append({"configuration": c, "T_inc": r["T_inc"], "T_ctrl": r["T_ctrl"], "seed": r["seed"],
                           "theme_only_b_step1": r["b_step1"], "theme_only_se": r["se_b_step1"], "theme_only_valid": r["valid"],
                           "theme_only_mass_ref": r["mass_ref_step1"], "theme_only_I_ref": r["I_ref_step1"],
                           "position_only_b_step1": po["b_step1"], "position_only_se": po["se_b_step1"], "position_only_valid": po["valid"],
                           "position_only_mass_ref": po["mass_ref_step1"], "position_only_I_ref": po["I_ref_step1"]})
    _write_csv(tables / "theme_only_position_only.csv", theme_only)

    # 5. recovery
    rec_rows, summary = [], []
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (probe, role) in zip(axs, (("training", "ud"), ("position_only", "ud"), ("theme_only", "neutral"))):
        for c in configs:
            row = extreme_rows(cluster[c])[(probe, role)]
            pr = pair_of[(c, int(row["image"]), row["zone"])]
            curve = np.asarray(row["b_curve"], dtype=float)
            rec_rows.append({"configuration": c, "probe": probe, "zone_role": role, "valid": pr["valid"],
                             **{f"b_k{i + 1}": v for i, v in enumerate(curve)}})
            ax.plot(np.arange(1, curve.size + 1), curve, marker=".", ls="-" if pr["valid"] else ":", color=colors_cfg[c],
                    label=f"{c}{'' if pr['valid'] else ' (invalid)'}")
        ax.axhline(-LOG2_5, color="k", ls=":", lw=0.8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{probe} ({role}): b by step since entry", fontsize=9)
        ax.set_xlabel("steps since entry")
    axs[0].set_ylabel("b (bits)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "recovery_extremes.png", dpi=120)
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, position in zip(axs, POSITIONS):
        for k, col in zip((0, 3, 5, 7, 10), ("tab:red", "tab:orange", "tab:olive", "tab:cyan", "tab:blue")):
            curves = []
            for c in configs:
                r = next(r for r in probe_rows if r["configuration"] == c and r["axis"] == "segment" and r["position"] == position and r["k"] == k)
                if r["valid"]:
                    curves.append(np.asarray(pair_of[(c, r["image"], r["slot"])]["b_curve"], dtype=float)[:10])
            if curves:
                ax.plot(np.arange(1, 11), np.nanmean(curves, axis=0), marker=".", color=col, label=f"segment k={k} ({len(curves)} configs)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"position {position}: recovery on the segment (mean over valid configs)", fontsize=9)
        ax.set_xlabel("steps since entry")
    axs[0].set_ylabel("b (bits)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "recovery_grid_segment.png", dpi=120)
    plt.close(fig)
    for c in configs:
        for position in POSITIONS:
            gaps, firsts, plateaus = [], [], []
            for r in probe_rows:
                if r["configuration"] != c or r["position"] != position or r["axis"] not in AXES or not r["valid"]:
                    continue
                curve = np.asarray(pair_of[(c, r["image"], r["slot"])]["b_curve"], dtype=float)[:10]
                plateau = float(np.nanmean(curve[5:10]))
                firsts.append(curve[0]); plateaus.append(plateau); gaps.append(plateau - curve[0])
            gaps, firsts = np.array(gaps), np.array(firsts)
            strong = firsts < -0.2
            summary.append({"configuration": c, "position": position, "n_valid_probes": len(gaps),
                            "mean_b_step1": float(np.mean(firsts)) if len(gaps) else float("nan"),
                            "mean_plateau_steps6_10": float(np.mean(plateaus)) if len(gaps) else float("nan"),
                            "mean_gap_plateau_minus_step1": float(np.mean(gaps)) if len(gaps) else float("nan"),
                            "n_probes_b_step1_below_-0.2": int(strong.sum()),
                            "mean_gap_where_b_step1_below_-0.2": float(np.mean(gaps[strong])) if strong.any() else float("nan")})
    _write_csv(tables / "recovery_curves_extremes.csv", rec_rows)
    _write_csv(tables / "recovery_summary_by_position.csv", summary)

    # 6. sufficiency: SE of b at step 1 against the number of walks, over the whole grid
    suff = []
    for c in configs:
        p = cluster[c]
        for r in p["images"]:
            pr = pair_of[(c, int(r["image"]), r["zone"])]
            for n in args.suff_walks:
                suff.append({"configuration": c, "image": int(r["image"]), "zone": r["zone"], "valid": pr["valid"], "n_walks": n,
                             "se_b_step1": step1_se(p, r["image"], r["zone"], rng, args.suff_bootstrap, n)})
    _write_csv(tables / "sufficiency_rows.csv", suff)
    ssum = []
    for n in args.suff_walks:
        for scope in ("valid", "all"):
            se = np.array([r["se_b_step1"] for r in suff if r["n_walks"] == n and (scope == "all" or r["valid"])])
            ssum.append({"n_walks": n, "scope": scope, "pairs": int(se.size), "median_se": float(np.median(se)),
                         "p90_se": float(np.quantile(se, 0.9)), "max_se": float(se.max())})
    _write_csv(tables / "sufficiency_summary.csv", ssum)
    fig, ax = plt.subplots(figsize=(6, 4))
    for stat, ls in (("median_se", "-"), ("p90_se", "--"), ("max_se", ":")):
        s = [r for r in ssum if r["scope"] == "valid"]
        ax.plot([r["n_walks"] for r in s], [r[stat] for r in s], ls=ls, marker="o", color="k", label=stat)
    ax.set_xlabel("walks")
    ax.set_ylabel("SE of b at step 1 (bits), valid probes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "sufficiency.png", dpi=120)
    plt.close(fig)
    print(json.dumps({"configs": configs, "n_probes": len(probe_rows), "n_invalid": len(invalid), "coherence_rows": len(coherence)}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validity", "diagnose", "report"):
        p = sub.add_parser(name)
        p.add_argument("--export", type=Path, nargs="+", required=True, help="extracted export_batch.py archives")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--config", type=Path, default=e4_setup.DEFAULT_CONFIG)
        p.add_argument("--threads", type=int, default=None)
    v = sub.choices["validity"]
    v.add_argument("--configuration", required=True)
    v.add_argument("--n-walks", type=int, default=100)
    v.add_argument("--models", nargs="+", default=["reference"], choices=("reference", "experimental"))
    d = sub.choices["diagnose"]
    d.add_argument("--configuration", required=True)
    d.add_argument("--images", type=int, nargs="+", required=True)
    d.add_argument("--n-walks", type=int, default=100)
    d.add_argument("--n-in", type=int, default=10)
    r = sub.choices["report"]
    r.add_argument("--validity", type=Path, nargs="+", required=True)
    r.add_argument("--local-probes", type=Path, nargs="*", default=[])
    r.add_argument("--n-walks", type=int, default=300, help="only the probe directories measured with this many walks")
    r.add_argument("--mass-min", type=float, default=0.90)
    r.add_argument("--i-tol", type=float, default=0.15)
    r.add_argument("--bootstrap", type=int, default=200)
    r.add_argument("--suff-walks", type=int, nargs="+", default=[25, 50, 100, 150, 200, 300])
    r.add_argument("--suff-bootstrap", type=int, default=100)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=args.command != "report")
    {"validity": cmd_validity, "diagnose": cmd_diagnose, "report": cmd_report}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
