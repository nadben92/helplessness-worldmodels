"""E4's shared set-up. Moved to the library on 2026-10-19 (helplessness.pipeline.two_zones_setup, verbatim) so that E5
uses it without importing E4; this module re-exports every name, with E4's own constants and load_config's default
path (configs/e4/e4.yaml). Imported by E4's scripts as a sibling module (`import e4_setup`).
"""

from __future__ import annotations

from pathlib import Path

from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones  # noqa: F401  (e4_setup.two_zones, read by E4's scripts and tests)
from helplessness.pipeline import two_zones_setup as _setup
from helplessness.pipeline.two_zones_setup import (  # noqa: F401
    PURPOSES,
    SLOT_CHECKPOINT_REAL_CONTEXT,
    SLOT_FINAL_REAL_CONTEXT,
    _walks,
    configuration,
    load_themes,
    measurement_seed,
    probe_world,
    successors,
    walks_for_slot,
    world,
)

EXPERIMENT_ID = "E4_generalization_gradient"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "e4" / "e4.yaml"
ZONE_PRESET = "2zones"
AGENT_TYPE = "worldmodel-agent-e4"


def load_config(path: str | Path | None = None) -> dict:
    return _setup.load_config(path if path is not None else DEFAULT_CONFIG)
