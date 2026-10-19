"""E5's shared set-up (orchestration only; everything reusable lives in the library): the config (E4's, unchanged, plus
the e5 block), the seeds A and B of a configuration, the training conditions and their run names, finding a run.
Imported by E5's scripts as a sibling module (`import e5_setup`).

A config of configs/e5/ holds `base` (an E4 config, read whole and unchanged) and the `e5` block, nothing else: every
training parameter of E5 is E4's (docs/e5/DECISIONS.md, (1)). A run's resolved config (its config.yaml) is the merged
dict and is read as it is.

Trainings (docs/E5_belief_training.md):
  baseline      E4's training (seed B: a new run; seed A: E4's own runs, reused), experimental or reference;
  shuffle       the zone assignment redrawn at each episode with probability `fraction` (pipeline.episode_sources),
                experimental or reference (the same draws, by episode index);
  immunisation  the reference's dynamics until the first episode starting at switch_step, the experimental dynamics
                for post_steps more; one run, on the reference's streams (it equals the reference of its seed until the
                switch); its b is taken against the baseline reference of its seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from e5_names import (  # noqa: F401  (re-exported: the names live in the standard-library module e5_names.py)
    AGENT_TYPE,
    EXPERIMENT_ID,
    TRAININGS,
    find_e4_run,
    find_run,
    pair_role,
    run_glob,
    training_tag,
    zone_preset,
)
from helplessness.config import REPO_ROOT
from helplessness.pipeline import two_zones_setup

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e5" / "e5.yaml"
SEED_LABELS = ("A", "B")
# measurement slots of the run's streams (E3 1-16, E4 20-21)
SLOT_ASSIGNMENT_REAL_CONTEXT = 22
SLOT_ASSIGNMENT_FINAL_REAL_CONTEXT = 23
SLOT_DECODABILITY = 25  # (the RSA's is 24)


def load_config(path: str | Path | None = None) -> dict:
    """An E5 config (base + e5) merged, or a run's resolved config.yaml as it is."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" not in raw:
        return raw
    extra = sorted(set(raw) - {"base", "e5"})
    if extra:
        raise ValueError(f"{path}: an E5 config holds `base` and `e5` only (E4's parameters unchanged), got {extra}")
    cfg = two_zones_setup.load_config(REPO_ROOT / raw["base"])
    cfg["e5"] = {**raw["e5"], "base": raw["base"]}
    return cfg


def configuration(cfg: dict, config_id: str, seed_label: str) -> dict:
    """E4's configuration (pipeline.two_zones_setup.configuration) with the seed of the label: A = E4's own seed of the
    configuration, B = A + e5.seed_b_offset. e4_seed keeps A."""
    conf = two_zones_setup.configuration(cfg, config_id)
    if seed_label not in SEED_LABELS:
        raise ValueError(f"seed label must be one of {SEED_LABELS}, got {seed_label!r}")
    seed = conf["seed"] if seed_label == "A" else conf["seed"] + int(cfg["e5"]["seed_b_offset"])
    return {**conf, "e4_seed": conf["seed"], "seed": seed, "seed_label": seed_label}


def records(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(run_dir) / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
