"""E8's shared set-up (orchestration only; everything reusable lives in the library): the config (E5's, unchanged, plus
the e8 block), the start models and their checks, the therapy of a model, the run names. Imported by E8's scripts as a
sibling module (`import e8_setup`).

A config of configs/e8/ holds `base` (an E5 config: E4's blocks + e5, read whole and unchanged) and the `e8` block,
nothing else (docs/e8/DECISIONS.md (1)). A run's resolved config (its config.yaml) is the merged dict and is read as it
is.

Therapies (docs/E8_therapy_relapse.md, DECISIONS (6)):
  full      a baseline model, in the reference environment of its configuration (canonical layout, original themes);
  q         a shuffle model, on the 3 assignments whose UD slot (T_inc) is the quadrant q (nw, ne, se, sw; "all": the
            12), the reference's dynamics.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup
from helplessness.probing.theme_probe_measure import sha256

EXPERIMENT_ID = "E8_therapy_relapse"
AGENT_TYPE = "worldmodel-agent-e8"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e8" / "e8.yaml"
THERAPIES = ("full", *QUADRANT_ORDER, "all")
PAIR_ROLE = {"experimental": "experimental", "reference": "control"}


def load_config(path: str | Path | None = None) -> dict:
    """An E8 config (base + e8) merged, or a run's resolved config.yaml as it is."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "base" not in raw:
        return raw
    extra = sorted(set(raw) - {"base", "e8"})
    if extra:
        raise ValueError(f"{path}: an E8 config holds `base` and `e8` only (E5's parameters unchanged), got {extra}")
    e5 = yaml.safe_load((REPO_ROOT / raw["base"]).read_text(encoding="utf-8"))
    if set(e5) != {"base", "e5"}:
        raise ValueError(f"{raw['base']}: not an E5 config (base + e5)")
    cfg = two_zones_setup.load_config(REPO_ROOT / e5["base"])
    cfg["e5"] = {**e5["e5"], "base": e5["base"]}
    cfg["e8"] = {**raw["e8"], "base": raw["base"]}
    return cfg


def load_models(path: str | Path) -> dict:
    """{"models": id -> spec, "pairs": agent -> {exposed, reference}} of an agents file (configs/e8/agents.yaml)."""
    raw = yaml.safe_load((REPO_ROOT / path if not Path(path).is_absolute() else Path(path)).read_text(encoding="utf-8"))
    for mid, m in raw["models"].items():
        missing = {"kind", "role", "configuration", "seed", "run", "checkpoint", "sha256"} - set(m)
        if missing:
            raise ValueError(f"{path}: model {mid!r} lacks {sorted(missing)}")
        if m["kind"] not in ("baseline", "shuffle") or m["role"] not in PAIR_ROLE:
            raise ValueError(f"{path}: model {mid!r}: kind {m['kind']!r}, role {m['role']!r}")
    for agent, p in raw["pairs"].items():
        if set(p) != {"exposed", "reference"} or not all(p[r] in raw["models"] for r in p):
            raise ValueError(f"{path}: pair {agent!r} must name an exposed and a reference model of the file")
    return raw


def start_weights(model: dict, runs_root: Path, check: bool = True) -> Path:
    """The start file of a model under the runs root: <run>/checkpoints/agent_env<checkpoint>.pt, its SHA-256 checked
    (a missing file or another hash stops everything: DECISIONS (2), the brief's section 11)."""
    path = Path(runs_root) / model["run"] / "checkpoints" / f"agent_env{int(model['checkpoint']):06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"start file not found: {path}")
    if check and sha256(path) != model["sha256"]:
        raise ValueError(f"{path}: SHA-256 {sha256(path)} differs from the agents file's {model['sha256']}")
    return path


def therapy_of(model: dict, treated_quadrant: str | None) -> str:
    """A baseline model's therapy is "full" (no quadrant); a shuffle model's is its treated quadrant (or "all")."""
    if model["kind"] == "baseline":
        if treated_quadrant not in (None, "full"):
            raise ValueError("a baseline model's therapy is the full exposure (no --treated-quadrant)")
        return "full"
    if treated_quadrant not in (*QUADRANT_ORDER, "all"):
        raise ValueError(f"a shuffle model needs --treated-quadrant in {QUADRANT_ORDER} or 'all', got {treated_quadrant!r}")
    return treated_quadrant


def zone_preset(model: dict, therapy: str) -> str:
    """2zones-<configuration>-<base|sh100>-th<full|q>, e.g. 2zones-p1-b-d1-sh100-thnw."""
    return f"2zones-{model['configuration']}-{'base' if model['kind'] == 'baseline' else 'sh100'}-th{therapy}"


def run_glob(model: dict, therapy: str) -> str:
    return f"grid{two_zones.GRID_SHAPE[0]}x{two_zones.GRID_SHAPE[1]}__{zone_preset(model, therapy)}__pair-{PAIR_ROLE[model['role']]}__{AGENT_TYPE}__seed{int(model['seed']):02d}__*"


def checkpoint_grid(cfg: dict) -> list[int]:
    """The therapy's checkpoints cut at its budget, the budget itself included."""
    n = int(cfg["e8"]["max_env_steps"])
    return sorted({int(c) for c in cfg["e8"]["checkpoints"] if int(c) <= n} | {n})


def treated_slot(model: dict, conf: dict, therapy: str) -> str | None:
    """The treated zone's slot when it is fixed: the configuration's UD slot for a baseline, q for a shuffle therapy on
    q; None for "all" (the episode's UD slot)."""
    if therapy == "full":
        return conf["roles"]["ud"]
    return None if therapy == "all" else therapy
