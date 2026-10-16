"""The training is unchanged by the 2026-10-16 adjustments (optional push, export, probes phase, diagnostics only at some
checkpoints, walk lengths): a fixed-seed mini training (p1-a-d1, both conditions, 300 environment steps, checkpoints
100 / 200 / 300) gives the weights recorded before those changes, tensor for tensor (tests/e4/data/
training_identity_reference.json, SHA-256 of every weight). Bit-exact only on the platform and library versions that
recorded the reference; skipped elsewhere."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy
import pytest
import torch

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline.agent_run import run_one
from helplessness.pipeline.runtime import configure_runtime

REFERENCE = json.loads((Path(__file__).parent / "data" / "training_identity_reference.json").read_text(encoding="utf-8"))
run_e4 = load_experiment_module("E4_generalization_gradient", "run_e4.py", "test_e4_identity_run")

pytestmark = pytest.mark.skipif(
    (platform.system(), platform.python_version(), torch.__version__, numpy.__version__)
    != (REFERENCE["platform"], REFERENCE["python"], REFERENCE["torch"], REFERENCE["numpy"]),
    reason="bit-exact reference recorded on another platform / library versions")


def digest(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for module in ("rssm", "goal_head", "critic"):
        for key in sorted(state[module]):
            h.update(key.encode())
            h.update(state[module][key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_mini_training_weights_unchanged(tmp_path):
    for condition in ("experimental", "reference"):
        cfg = run_e4.e4_setup.load_config(REPO_ROOT / "configs" / "e4" / "e4_mini.yaml")
        cfg["budget"]["total_steps"] = 260
        cfg["checkpoints"]["grid"] = [100, 200, 300]
        cfg["e4"]["run_configuration"] = "p1-a-d1"
        conf = run_e4.e4_setup.configuration(cfg, "p1-a-d1")
        run_dir = tmp_path / condition
        run_dir.mkdir()
        run_one(cfg, conf["seed"], condition, run_dir, configure_runtime(cfg), hooks=run_e4.make_hooks(cfg, conf))
        for step in (100, 200, 300):
            assert digest(run_dir / "checkpoints" / f"agent_env{step:06d}.pt") == REFERENCE["weights_sha256"][f"{condition}/{step}"], (condition, step)
