"""E4's shared set-up (orchestration only; everything reusable lives in the library): reading the config and the
colours, resolving a training configuration, building its training and probe worlds, the fixed walks of each slot
and the measurement seeds. Imported by E4's scripts as a sibling module (`import e4_setup`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from helplessness.config import REPO_ROOT
from helplessness.envs.appearance import theme_axes
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.envs.layouts import four_zones_pixel, two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones as two_zones_world
from helplessness.probing import walks as walks_mod

EXPERIMENT_ID = "E4_generalization_gradient"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e4" / "e4.yaml"
ZONE_PRESET = "2zones"
AGENT_TYPE = "worldmodel-agent-e4"
# Measurement slots of the E3 streams (helplessness.streams.worldmodel.measure_streams; E3 uses 1-15)
SLOT_CHECKPOINT_REAL_CONTEXT = 20
SLOT_FINAL_REAL_CONTEXT = 21
PURPOSES = {"checkpoint": 0, "final": 1, "probe": 2, "forgetting": 3}


def load_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path is not None else DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_themes(cfg: dict) -> dict:
    with open(REPO_ROOT / cfg["e4"]["themes"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def configuration(cfg: dict, config_id: str) -> dict:
    """A training configuration: its pair, orientation, diagonal and seed, T_inc / T_ctrl, the axes of its
    orientation, the slots' roles and the training image's colours."""
    spec = cfg["e4"]["configurations"][config_id]
    themes = load_themes(cfg)
    pair = themes["pairs"][spec["pair"]]
    axes = theme_axes.axes_for_orientation({k: np.array(pair[k], dtype=np.int64) for k in ("segment", "arc_center_b", "arc_center_a")},
                                           spec["inc"])
    t_inc, t_ctrl = (pair["a"], pair["b"]) if spec["inc"] == "a" else (pair["b"], pair["a"])
    t_inc, t_ctrl = tuple(int(v) for v in t_inc), tuple(int(v) for v in t_ctrl)
    return {"id": config_id, **spec, "t_inc": t_inc, "t_ctrl": t_ctrl, "D": float(pair["D"]), "axes": axes,
            "roles": two_zones.roles(spec["diagonal"]), "agent_color": tuple(themes["agent_marker_color"]),
            "training_colors": two_zones.training_colors(spec["diagonal"], t_inc, t_ctrl)}


def world(cfg: dict, conf: dict, condition: str, slot_colors: dict | None = None):
    """The pixel world of a configuration: the given dynamics condition, the training image or slot_colors."""
    return two_zones_world.build_world(slot_colors or conf["training_colors"], conf["agent_color"], conf["diagonal"], condition,
                                       cfg["render"], cfg["episode"], conf["seed"])


def probe_world(cfg: dict, conf: dict, slot_colors: dict | None = None):
    """A probe environment: the reference's dynamics (everything controllable) and an image (default: training)."""
    return world(cfg, conf, "reference", slot_colors)


@lru_cache(maxsize=None)
def _walks(n_pre: int, n_in: int, n_walks: int, seed: int, slot: str) -> dict:
    layout = two_zones.build_layout(1, "reference")  # every probe environment has these dynamics, for both diagonals
    themes = two_zones.themes({q: CORRIDOR_COLOR for q in QUADRANT_ORDER}, (0, 0, 0))
    P = four_zones_pixel.build_pixel_env(layout, themes, goal_reward=50.0).transition_tensor()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), QUADRANT_ORDER.index(slot)]))
    return walks_mod.generate_walks(P, two_zones.corridor_states(layout), two_zones.region_states(layout)[slot], n_walks, n_pre, n_in, rng)


def walks_for_slot(cfg: dict, slot: str, n_walks: int | None = None, n_in: int | None = None) -> dict:
    """The fixed walks into a slot (probing.walks), the same for every image and both models. A smaller n_walks gives
    the first n_walks of the full set (one generator, walk after walk); a smaller n_in cuts every walk after its n_in-th
    state in the zone (the same walks, shorter: e4.walks.n_in_grid for the full grid, e4.walks.n_in for the extreme
    probes and the evolution in the zone)."""
    w = cfg["e4"]["walks"]
    full = _walks(w["n_pre"], w["n_in"], w["n_walks"], w["seed"], slot)
    n = w["n_walks"] if n_walks is None else n_walks
    if n > w["n_walks"]:
        raise ValueError(f"n_walks {n} > the full set's {w['n_walks']}")
    out = {k: v[:n].copy() for k, v in full.items()}
    if n_in is not None and n_in < w["n_in"]:
        for i in range(n):
            end = int(np.flatnonzero(out["steps_since_entry"][i] == n_in)[0]) + 1
            out["actions"][i, end - 1] = -1
            for key, fill in (("states", -1), ("actions", -1), ("phase", walks_mod.PHASE_PAD), ("steps_since_start", -1),
                              ("steps_since_entry", 0)):
                out[key][i, end:] = fill
            out["length"][i] = end
    T = int(out["length"].max())
    return {k: (v[:, :T] if v.ndim == 2 else v) for k, v in out.items()}


def measurement_seed(cfg: dict, conf: dict, image_index: int, slot: str, purpose: str) -> int:
    """The torch seed of one (image, slot) measurement, the same for the experimental and reference models of a
    configuration (it never depends on the condition)."""
    seq = np.random.SeedSequence([int(cfg["e4"]["walks"]["seed"]), int(conf["seed"]), int(image_index), QUADRANT_ORDER.index(slot),
                                  PURPOSES[purpose]])
    return int(seq.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def successors(world_) -> np.ndarray:
    return world_.env.transition_tensor().argmax(axis=-1)
