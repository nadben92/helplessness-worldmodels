"""The shared set-up of E6 and E7 (docs/E6_phenotypes_latent.md, docs/E7_latent_manipulation.md, docs/e6/DECISIONS.md):
the list of agents (a configuration file, not code), the test images (E4's probe grid, by name), the zone of interest
and its start states, the episode seeds, E4's validity criterion and the measurement of one (agent, image) -- the
episodes of the exposed model and of its reference, the belief at step 1, the phenotypes. E7 measures its interventions
with the same function (measure_pair with hooks), so that alpha = 0 reproduces E6 exactly. New module (2026-10-23).

Agents file (YAML):
    checkpoint: 300000                 # default weights step (agent_env<N>.pt) of every run
    agents:
      - id: p1-b-d1_A_base             # unique; an agent is an exposed model and its reference
        condition: baseline            # free label (baseline, f1, immunisation, ...): the analysis groups by it
        configuration: p1-b-d1         # E4's configuration id (colours, roles, the probe grid)
        seed: 2                        # the exposed run's seed (E4/E5's probe seeds)
        experimental: {run: "<glob of the run directory under runs_root>"}    # or {weights: "<.pt under runs_root>"}
        reference:    {run: "...", checkpoint: 300000}                       # checkpoint per model optional
A run glob resolves to the newest matching directory (sorted names end with their UTC stamp); paths are relative to
the runs root given on the command line (no absolute path in a config).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.envs.layouts import four_zones_pixel, two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.measures import zone_phenotypes as zp
from helplessness.pipeline import two_zones_setup as setup
from helplessness.probing import probe_validity
from helplessness.probing.frozen_episodes import (FrozenAgent, TestWorld, explore_sequences, fresh_start_controllability,
                                                  run_episodes)
from helplessness.probing.theme_probe_measure import measure as probe_measure
from helplessness.probing.theme_probes import probe_images
from helplessness.probing.walks import border_states as walk_borders

SLOT_INDEX = {"nw": 0, "ne": 1, "se": 2, "sw": 3}
# E6's streams: purposes under SeedSequence([e6.seed, image index, slot index, purpose])
PURPOSE = {"explore": 0, "act": 1, "measure": 2, "fresh": 3, "bootstrap": 4}


# ---------------------------------------------------------------------------------------------------------- config
def load_layered_config(path: Path, allowed: tuple[str, ...]) -> dict:
    """A config holding `base` (another config, loaded the same way, relative to the repo root) and new blocks among
    `allowed` only: the base's blocks are never overridden (E6 reads E4's world, measure and probes unchanged; E7 reads
    E6's protocol unchanged). The merged dict records the chain under `_bases`."""
    from helplessness.config import REPO_ROOT

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "base" not in raw:
        return {**raw, "_bases": []}
    extra = sorted(set(raw) - {"base", *allowed})
    if extra:
        raise ValueError(f"{path}: only `base` and {list(allowed)} are allowed, got {extra}")
    base = load_layered_config(REPO_ROOT / raw["base"], allowed)
    clash = sorted((set(raw) - {"base"}) & (set(base) - {"_bases"}))
    if clash:
        raise ValueError(f"{path}: {clash} already defined by {raw['base']}")
    return {**base, **{k: v for k, v in raw.items() if k != "base"}, "_bases": [*base["_bases"], raw["base"]]}


def e6_protocol(cfg: dict) -> dict:
    """The e6 block with what it takes from the base: the kernel's margin; the horizon checked against the episode's."""
    e6 = dict(cfg["e6"])
    if int(e6["horizon"]) != int(cfg["episode"]["max_steps"]):
        raise ValueError(f"e6.horizon {e6['horizon']} != episode.max_steps {cfg['episode']['max_steps']}")
    e6["margin"] = float(cfg["measure"]["kernel_margin_threshold"])
    return e6


# ---------------------------------------------------------------------------------------------------------- agents
def resolve_weights(spec: dict, runs_root: Path, default_checkpoint: int) -> Path:
    """A model's weights: {"weights": path} or {"run": glob, "checkpoint": N} (default_checkpoint when absent)."""
    runs_root = Path(runs_root)
    if "weights" in spec:
        path = runs_root / spec["weights"]
    else:
        found = sorted(d for d in runs_root.glob(spec["run"]) if d.is_dir())
        if not found:
            raise FileNotFoundError(f"no run matches {spec['run']!r} under {runs_root}")
        path = found[-1] / "checkpoints" / f"agent_env{int(spec.get('checkpoint', default_checkpoint)):06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_agents(path: Path, runs_root: Path, only: list[str] | None = None) -> list[dict]:
    """The agents of an agents file, their weights resolved. only: a subset of ids (in the file's order)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    default = int(raw.get("checkpoint", 300000))
    out, seen = [], set()
    for a in raw["agents"]:
        missing = {"id", "condition", "configuration", "seed", "experimental", "reference"} - set(a)
        if missing:
            raise ValueError(f"{path}: agent {a.get('id')!r} lacks {sorted(missing)}")
        if a["id"] in seen:
            raise ValueError(f"{path}: duplicate agent id {a['id']!r}")
        seen.add(a["id"])
        if only is not None and a["id"] not in only:
            continue
        out.append({**a, "seed": int(a["seed"]), "weights": {role: resolve_weights(a[role], runs_root, default)
                                                             for role in ("experimental", "reference")}})
    if only is not None and len(out) != len(only):
        raise ValueError(f"unknown agent ids: {sorted(set(only) - {a['id'] for a in out})}")
    return out


# ---------------------------------------------------------------------------------------------------------- points
def curve_points(rows: list[dict], phenotype: str, include_references: bool, valid_only: bool = True) -> dict[str, np.ndarray]:
    """The points of a phenotype's curve from E6's rows (one per agent x image x role): b_step1, the phenotype, the
    agent (the bootstrap's cluster), the condition, the localisation at step 1, the image and role. Kept: valid (E4's
    criterion, the agent's reference) and, unless include_references, the exposed role only. A censored recovery
    latency (never reaching theta) enters at the first index it could not read (its max_index_read + 1), flagged
    (docs/e6/DECISIONS.md (35)); any other NaN phenotype is left out."""
    keep = []
    for r in rows:
        if valid_only and not _truthy(r.get("valid")):
            continue
        if r["role"] == "reference" and not include_references:
            continue
        y = float(r[phenotype])
        censored = False
        if phenotype == "recovery_latency" and _truthy(r.get("recovery_censored")):
            y, censored = float(r.get("recovery_max_index_read", np.nan)) + 1.0, True
        if not np.isfinite(y) or not np.isfinite(float(r["b_step1"])):
            continue
        keep.append((float(r["b_step1"]), y, r["agent"], r["condition"], float(r.get("loc_template_step1", np.nan)),
                     float(r.get("loc_template_zone", np.nan)), r["image"], r["role"], censored))
    cols = list(zip(*keep)) if keep else [[]] * 9
    names = ("b", "y", "agent", "condition", "loc_step1", "loc_zone", "image", "role", "censored")
    return {n: np.asarray(c) for n, c in zip(names, cols)}


def _truthy(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true") or (isinstance(v, (int, float, np.bool_)) and bool(v))


# ---------------------------------------------------------------------------------------------------------- images
def test_images(conf: dict, specs: list[dict]) -> list[dict]:
    """The test images of a configuration from their specs: {"name", "axis", "k", "position"} -- axis "training" (the
    training image, zone of interest = the slot of `position`), "position_only", or one of E4's axes (segment, arc_ctrl,
    arc_inc) with its point k: E4's probe image changing the slot of `position` (theme_probes). Each: name, index (E4's
    image index), slot (the zone of interest), role, slot_colors, is_training."""
    images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    out = []
    for spec in specs:
        slot = conf["roles"][spec["position"]]
        if spec["axis"] == "training":
            image = next(im for im in images if im["is_training"])
        else:
            key = (spec["axis"], int(spec.get("k", 0)), spec["position"])
            match = [im for im in images if any((p["axis"], p["k"], p["position"]) == key for p in im["probes"])]
            if len(match) != 1:
                raise ValueError(f"no single probe image for {key}")
            image = match[0]
        out.append({"name": spec["name"], "index": image["index"], "slot": slot, "role": spec["position"],
                    "slot_colors": image["slot_colors"], "is_training": image["is_training"], "image": image})
    return out


def zone_states(slot: str) -> np.ndarray:
    """The 25 states of a quadrant slot (the zone of interest)."""
    return two_zones.region_states(two_zones.build_layout(1, "reference"))[slot]


def interior_states(slot: str) -> np.ndarray:
    """The 9 states of the slot's central 3 x 3 square (every neighbour in the slot)."""
    zone = zone_states(slot)
    rows, cols = np.divmod(zone, two_zones.GRID_SHAPE[1])
    keep = (rows > rows.min()) & (rows < rows.max()) & (cols > cols.min()) & (cols < cols.max())
    return zone[keep]


def goal_distances() -> np.ndarray:
    return np.array([two_zones.toric_distance_to_goal(s) for s in range(two_zones.N_STATES)])


def build_test_world(cfg: dict, conf: dict, slot_colors: dict, device: torch.device):
    """The probe environment (reference dynamics, the image) as E4's pipeline world and as frozen_episodes.TestWorld."""
    world = setup.probe_world(cfg, conf, slot_colors)
    succ = setup.successors(world)
    return world, TestWorld(world.templates_t, succ, two_zones.GOAL_STATE, conf["agent_color"], cfg["render"]["k"], device)


def seeds(e6_seed: int, image: dict) -> dict:
    """The seeds of one (image, zone): the same for every agent and both models of a pair (common random numbers)."""
    def seq(purpose):
        return np.random.SeedSequence([int(e6_seed), int(image["index"]), SLOT_INDEX[image["slot"]], PURPOSE[purpose]])

    def torch_seed(purpose):
        return int(seq(purpose).generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))

    return {"explore_entropy": [int(e6_seed), int(image["index"]), SLOT_INDEX[image["slot"]], PURPOSE["explore"]],
            "act": torch_seed("act"), "measure": torch_seed("measure"), "fresh": torch_seed("fresh"),
            "bootstrap": seq("bootstrap")}


# ---------------------------------------------------------------------------------------------------------- validity
def probe_validity_pair(cfg: dict, conf: dict, rssms: dict, image: dict, n_walks: int, device) -> dict:
    """E4's probe of (image, zone) on both models (the same walks and seeds as E4/E5's probes: theme_probe_measure.measure)
    and E4's validity criterion, read from the reference at the first step in the zone. Returns b_step1_probe (exposed
    minus reference, by state then zone over the border states), the step-1 summaries of both models, valid."""
    one = {**image["image"], "zones": [image["slot"]]}
    keep = {}

    def on_pair(_image, _slot, res, _world, _walks):
        keep.update({name: probe_validity.step1_summary(r) for name, r in res.items()})
        return {}

    probe_measure(cfg, conf, rssms, [one], device, log=lambda s: None, n_walks=n_walks, on_pair=on_pair)
    e, r = keep["experimental"], keep["reference"]
    return {"b_step1_probe": e["I_step1"] - r["I_step1"], "I_exp_step1_probe": e["I_step1"], "I_ref_step1_probe": r["I_step1"],
            "mass_exp_step1_probe": e["mass_step1"], "mass_ref_step1_probe": r["mass_step1"], "ghost_ref_step1_probe": r["ghost_step1"],
            "n_border_states": r["n_states_step1"], "valid": probe_validity.is_valid(r["mass_step1"], r["I_step1"])}


# ---------------------------------------------------------------------------------------------------------- episodes
def starts_for(slot: str, n_reps: int) -> np.ndarray:
    """Episode i starts at interior state i mod 9 (n_reps rounds of the nine)."""
    return np.tile(interior_states(slot), int(n_reps))


def by_state_then_zone(values: np.ndarray, states: np.ndarray) -> float:
    return probe_validity.by_state_then_zone(values, states)


def run_model(agent: FrozenAgent, tw: TestWorld, image: dict, e6: dict, epsilon: float, hook=None, tau: float | None = None,
              q_fn=None, action_kernel: bool = False) -> dict:
    """The episodes of one model on one (image, zone): E6's protocol (module docstring of probing.frozen_episodes). tau,
    q_fn: the test-time decision rule's options of run_episodes (None: E6's); action_kernel: run_episodes' per-action
    prediction arrays (off: E6's)."""
    s = seeds(e6["seed"], image)
    starts = starts_for(image["slot"], e6["n_reps"])
    zone = np.zeros(two_zones.N_STATES, dtype=bool)
    zone[zone_states(image["slot"])] = True
    return run_episodes(agent, tw, starts, int(e6["horizon"]), epsilon, explore_sequences(s["explore_entropy"], len(starts)),
                        s["act"], s["measure"], zone, draws=int(e6["draws"]), margin=float(e6["margin"]), measure=e6["measure"], hook=hook,
                        tau=tau, q_fn=q_fn, action_kernel=action_kernel)


def reference_fresh(reference: FrozenAgent, tw: TestWorld, image: dict, e6: dict) -> np.ndarray:
    """The reference's I_s at the first step of an episode started at every non-goal state ((n_states,), NaN at the goal):
    what b_local subtracts (docs/e6/DECISIONS.md (37))."""
    states = np.array([st for st in range(two_zones.N_STATES) if st != two_zones.GOAL_STATE])
    out = np.full(two_zones.N_STATES, np.nan)
    out[states] = fresh_start_controllability(reference, tw, states, int(e6["fresh_samples"]), int(e6["draws"]), float(e6["margin"]),
                                              seeds(e6["seed"], image)["fresh"])
    return out


def border_states(slot: str) -> np.ndarray:
    """The zone's states with a corridor neighbour (E4's entry states: probing.walks.border_states), reference dynamics."""
    layout = two_zones.build_layout(1, "reference")
    themes = two_zones.themes({q: CORRIDOR_COLOR for q in QUADRANT_ORDER}, (0, 0, 0))
    succ = four_zones_pixel.build_pixel_env(layout, themes, goal_reward=50.0).transition_tensor().argmax(-1)
    return walk_borders(succ, zone_states(slot), two_zones.corridor_states(layout))


def fresh_split(exposed: FrozenAgent, tw: TestWorld, image: dict, e6: dict, ref_fresh: np.ndarray) -> dict:
    """b at the first step of an episode started from the initial state (no context), at the zone's centre and on its
    border (docs/e6/DECISIONS.md (46)): the exposed model's fresh-start I at the zone's states (the same function, seed
    and draws as reference_fresh) minus the reference's, by state then zone. With E4's probe (border, after a corridor
    walk), it splits E6's gap into a place effect (border vs centre, both fresh) and a history effect (probe vs fresh,
    both on the border)."""
    zone = zone_states(image["slot"])
    I_exp = np.full(two_zones.N_STATES, np.nan)
    I_exp[zone] = fresh_start_controllability(exposed, tw, zone, int(e6["fresh_samples"]), int(e6["draws"]), float(e6["margin"]),
                                              seeds(e6["seed"], image)["fresh"])
    out = {}
    for name, states in (("center", interior_states(image["slot"])), ("border", border_states(image["slot"]))):
        out[f"I_exp_fresh_{name}"] = float(np.nanmean(I_exp[states]))
        out[f"I_ref_fresh_{name}"] = float(np.nanmean(ref_fresh[states]))
        out[f"b_fresh_{name}"] = out[f"I_exp_fresh_{name}"] - out[f"I_ref_fresh_{name}"]
        out[f"n_fresh_{name}"] = int(len(states))
    return out


def phenotype_row(ep: dict, image: dict, epsilon: float, e6: dict, with_bootstrap: bool = True) -> dict:
    """The phenotypes of one model's episodes (measures.zone_phenotypes) and their bootstrap intervals (episodes
    resampled within each start state)."""
    zone = np.zeros(two_zones.N_STATES, dtype=bool)
    zone[zone_states(image["slot"])] = True
    ph = e6["phenotypes"]
    kw = {"k_revisit": int(ph["k_revisit"]), "n_min": int(ph["n_min"])}
    summary = zp.summarize(ep["states"], ep["actions"], ep["length"], zone, goal_distances(), epsilon, **kw)
    row = {k: v for k, v in summary.items() if k not in ("directed_by_index", "km")}
    if with_bootstrap:
        rng = np.random.default_rng(seeds(e6["seed"], image)["bootstrap"])
        ci = zp.bootstrap(ep["states"], ep["actions"], ep["length"], zone, goal_distances(), epsilon, rng, n_boot=int(ph["n_boot"]),
                          strata=ep["states"][:, 0], **kw)
        for name in zp.PHENOTYPES:
            row[f"{name}_lo"], row[f"{name}_hi"] = ci[name]
            row[f"{name}_nan_share"] = ci[f"{name}_nan_share"]
    row["success_rate"] = float(ep["terminated"].mean())
    row["by_index_n"] = summary["directed_by_index"]["n"].tolist()
    row["by_index_share"] = summary["directed_by_index"]["share"].tolist()
    return row


def belief_row(ep: dict, ep_ref: dict, ref_fresh: np.ndarray, image: dict) -> dict:
    """b at step 1 (the model's I at t = 0 minus the reference's, by state then zone over the nine starts), the
    localisation at step 1 and in the zone, and the step-1 mass and ghosts."""
    starts = ep["states"][:, 0]
    zone = np.zeros(two_zones.N_STATES, dtype=bool)
    zone[zone_states(image["slot"])] = True
    inz = zp.in_zone_steps(ep["states"], ep["length"], zone)
    I_model = by_state_then_zone(ep["I"][:, 0], starts)
    I_ref = by_state_then_zone(ep_ref["I"][:, 0], ep_ref["states"][:, 0])
    return {"I_step1": I_model, "I_ref_step1": I_ref, "b_step1": I_model - I_ref,
            "b_step1_fresh_ref": I_model - by_state_then_zone(ref_fresh[starts], starts),
            "loc_template_step1": float(np.nanmean(ep["loc_template"][:, 0])), "loc_agent_step1": float(np.nanmean(ep["loc_agent"][:, 0])),
            "loc_template_zone": float(np.nanmean(ep["loc_template"][inz])), "loc_agent_zone": float(np.nanmean(ep["loc_agent"][inz])),
            "mass_played_step1": float(np.nanmean(ep["mass_played"][:, 0])), "ghost_step1": float(np.nanmean(ep["ghost"][:, 0])),
            "p_goal_step1": float(np.nanmean(ep["p_goal"][:, 0])), "q_gap_step1": float(np.nanmean(ep["q_gap"][:, 0])),
            "tied_step1": float(np.nanmean(ep["tied"][:, 0]))}


def b_local(ep: dict, ref_fresh: np.ndarray) -> np.ndarray:
    """(n, H): the model's I at each step minus the reference's fresh-start I at the same state (NaN where unmeasured)."""
    s = np.clip(ep["states"][:, :-1], 0, None)
    return ep["I"] - ref_fresh[s]


def b_local_by_index(ep: dict, ref_fresh: np.ndarray, image: dict, max_index: int) -> dict:
    """The companion curve: b_local at the j-th step of the first stay in the zone (j = 1..), its mean and the episodes
    behind it (the recovery of the belief through h)."""
    zone = np.zeros(two_zones.N_STATES, dtype=bool)
    zone[zone_states(image["slot"])] = True
    ex = zp.first_exit(ep["states"], ep["length"], zone)
    bl = b_local(ep, ref_fresh)
    mean, n = np.full(max_index, np.nan), np.zeros(max_index, dtype=np.int64)
    for j in range(max_index):
        rows = ex["time"] > j
        v = bl[rows, j]
        v = v[np.isfinite(v)]
        n[j] = v.size
        if v.size:
            mean[j] = v.mean()
    return {"mean": mean, "n": n}
