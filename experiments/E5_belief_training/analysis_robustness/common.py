"""The robustness diagnostic's shared plumbing (README.md): the model table (models.yaml), where each model's weights and
cluster swap-probe measurement are, and the swap images with the localisation the cluster measured on them.

Nothing here measures: the images are helplessness.probing.swap_probes', the localisation is the cluster's
reconstruction diagnostic (helplessness.probing.probe_validity.reconstruction_diagnostic, read from reconstruction.csv).
The localisation of a (model, image, zone) is agent_cell_accuracy of part zone_step1 (the first step of the walks in
that zone, the agent's cell decoded from the posterior), as measure_probes.swap_rows reads it.
"""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from helplessness.pipeline import two_zones_setup
from helplessness.probing import swap_probes

HERE = Path(__file__).resolve().parent
LOCALISATION_MIN = 0.90  # measure_probes.LOCALISATION_MIN (analysis_450k/LOCALISATION_CRITERION.md)
AXES = ("arc_ctrl", "segment", "arc_inc")
PARTS = ("corridor", "zone", "zone_step1")


def load_models(path: Path = HERE / "models.yaml") -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["models"]


def find_one(roots: list[Path], pattern: str, need: str | None = None) -> Path | None:
    """The newest directory matching pattern under the first root that has one (with the file `need` inside, if given)."""
    for root in roots:
        hits = sorted(p for p in root.glob(pattern) if p.is_dir() and (need is None or (p / need).exists()))
        if hits:
            return hits[-1]
    return None


def weights_of(model: dict, runs_roots: list[Path]) -> Path | None:
    need = f"checkpoints/agent_env{model['checkpoint']:06d}.pt"
    run = find_one(runs_roots, model["run"], need)
    return None if run is None else run / need


def probes_dir_of(model: dict, probes_roots: list[Path]) -> Path | None:
    return find_one(probes_roots, model["probes"]["dir"], "reconstruction.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_meta(probes_dir: Path) -> dict:
    return yaml.safe_load((probes_dir / "meta.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def setup(config_path: str, configuration: str = "p1-b-d1") -> tuple[dict, dict, list[dict]]:
    """(cfg, conf, swap images) of a run's config, as measure_probes builds them (conf's seed set by the caller)."""
    cfg = two_zones_setup.load_config(config_path)
    conf = two_zones_setup.configuration(cfg, configuration)
    return cfg, conf, swap_probes.swap_probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])


def probe_points(images: list[dict]) -> list[dict]:
    """One entry per (swap, axis, k): the image index, t, the slot m, near_duplicate."""
    return [{"image": im["index"], "swap": im["swap"], "axis": p["axis"], "k": p["k"], "t": p["t"], "near_duplicate": p["near_duplicate"]}
            for im in images for p in im["probes"]]


def read_reconstruction(probes_dir: Path, model_name: str) -> dict[tuple[int, str, str], dict]:
    """(image, zone, part) -> the row of reconstruction.csv for one model of the pair."""
    out = {}
    with open(probes_dir / "reconstruction.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == model_name:
                out[(int(r["image"]), r["zone"], r["part"])] = r
    return out


def zone_class(zone: str, swap: str) -> str:
    """Where the zone of interest is relative to the recoloured slot m: "moved" (the zone is m) or "other"."""
    return "moved" if zone == swap else "other"


def localisation_table(images: list[dict], recon: dict, part: str = "zone_step1") -> list[dict]:
    """One row per (swap, axis, k, zone): the localisation (agent_cell_accuracy) and n_frames of `part`."""
    rows = []
    zones = images[0]["zones"]
    for p in probe_points(images):
        for zone in zones:
            r = recon.get((p["image"], zone, part))
            if r is None:
                continue
            rows.append({**p, "zone": zone, "where": zone_class(zone, p["swap"]), "loc": float(r["agent_cell_accuracy"]),
                         "n_frames": int(r["n_frames"])})
    return rows


def curve(rows: list[dict], axis: str, where: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(t, localisation): by t, the mean over swaps of the mean over the zones (all zones, or where = moved / other)."""
    ts = sorted({r["t"] for r in rows if r["axis"] == axis})
    out = []
    for t in ts:
        by_swap = {}
        for r in rows:
            if r["axis"] == axis and r["t"] == t and (where is None or r["where"] == where):
                by_swap.setdefault(r["swap"], []).append(r["loc"])
        out.append(np.mean([np.mean(v) for v in by_swap.values()]))
    return np.array(ts), np.array(out)


def scores(t: np.ndarray, loc: np.ndarray, threshold: float = LOCALISATION_MIN) -> dict:
    """first_t_below: the first t whose localisation is < threshold (NaN: never; reported as > 1); auc: the area under the
    curve over t in [0, 1] (trapezoids; 1 = localised everywhere)."""
    below = np.flatnonzero(loc < threshold)
    return {"first_t_below": float(t[below[0]]) if below.size else float("nan"), "auc": float(np.trapezoid(loc, t))}
