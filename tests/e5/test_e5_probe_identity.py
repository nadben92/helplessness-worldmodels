"""The probes E5 runs are E4's original code (user, 2026-10-19): on the same weights, the library's probe measure
(probing.theme_probe_measure, with pipeline.two_zones_setup) gives exactly the I of E4's ORIGINAL measure_probes.py and
e4_setup.py, read from git at the commit before their move (ORIGINAL), on a few images of the grid (the training image,
grid images on the three positions, position only), for both models of the pair. The weights: an E4 checkpoint (E4's
mini training, the reference of p1-a-d1, 300 steps) and an untrained model (its diffuse prior gives non-zero I
everywhere, so the comparison is not trivially 0 = 0). Skipped when git or the commit is not available."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch

from helplessness.config import REPO_ROOT
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import run_one
from helplessness.pipeline.checkpoints import build_rssm
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import make_hooks
from helplessness.probing import theme_probe_measure
from helplessness.probing.theme_probes import probe_images

ORIGINAL = "3657b33"  # E4 before the move of 2026-10-19
E4_DIR = "experiments/E4_generalization_gradient"
IMAGES = (0, 5, 17, 40, 63, 92)
N_WALKS = 6


def git_show(path: str) -> str:
    try:
        return subprocess.run(["git", "show", f"{ORIGINAL}:{path}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as err:
        pytest.skip(f"E4's original code not available from git ({err})")


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    """E4's original e4_setup and measure_probes, imported from a temporary directory (sys.modules restored after)."""
    d = tmp_path_factory.mktemp("e4_original")
    for name in ("e4_setup.py", "measure_probes.py"):
        (d / name).write_text(git_show(f"{E4_DIR}/{name}"), encoding="utf-8")
    saved = {k: sys.modules.get(k) for k in ("e4_setup",)}
    try:
        spec = importlib.util.spec_from_file_location("e4_setup", d / "e4_setup.py")
        setup = importlib.util.module_from_spec(spec)
        sys.modules["e4_setup"] = setup
        spec.loader.exec_module(setup)
        spec = importlib.util.spec_from_file_location("e4_original_measure_probes", d / "measure_probes.py")
        mp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mp)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert "def measure(" in (d / "measure_probes.py").read_text(encoding="utf-8")  # really the original body
    return setup, mp


def mini_cfg() -> dict:
    cfg = two_zones_setup.load_config(REPO_ROOT / "configs" / "e4" / "e4_mini.yaml")
    cfg["budget"]["total_steps"] = 260
    cfg["checkpoints"]["grid"] = [100, 200, 300]
    cfg["e4"]["run_configuration"] = "p1-a-d1"
    return cfg


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    cfg = mini_cfg()
    conf = two_zones_setup.configuration(cfg, "p1-a-d1")
    run_dir = tmp_path_factory.mktemp("e4_mini_reference")
    hooks = make_hooks(cfg, conf)
    hooks.checkpoint_record = lambda state: {}
    hooks.finalize = lambda records, world, run_dir: {}
    run_one(copy.deepcopy(cfg), conf["seed"], "reference", run_dir, configure_runtime(cfg), hooks=hooks)
    device = torch.device("cpu")
    trained = theme_probe_measure.load_rssm(run_dir / "checkpoints" / "agent_env000300.pt", device)
    torch.manual_seed(12345)
    untrained = build_rssm(cfg).to(device).eval()
    return {"experimental": untrained, "reference": trained}


def test_library_probes_equal_e4_original_code(original, models):
    setup, mp = original
    cfg = mini_cfg()
    device = configure_runtime(cfg)
    conf_lib = two_zones_setup.configuration(cfg, "p1-a-d1")
    conf_orig = setup.configuration(setup.load_config(REPO_ROOT / "configs" / "e4" / "e4_mini.yaml"), "p1-a-d1")
    assert {k: v for k, v in conf_lib.items() if k != "axes"} == {k: v for k, v in conf_orig.items() if k != "axes"}
    images_all = probe_images(conf_lib["diagonal"], conf_lib["t_inc"], conf_lib["t_ctrl"], conf_lib["axes"])
    images = [images_all[i] for i in IMAGES]
    lib = theme_probe_measure.measure(cfg, conf_lib, models, images, device, log=lambda s: None, save_steps=True, n_walks=N_WALKS)
    orig = mp.measure(cfg, conf_orig, models, images, device, log=lambda s: None, save_steps=True, n_walks=N_WALKS)
    for key in ("curve_exp", "curve_ref", "n_states_curve"):
        assert np.array_equal(lib[key], orig[key], equal_nan=True), key
    assert lib["steps"].keys() == orig["steps"].keys()
    for key in lib["steps"]:
        assert np.array_equal(lib["steps"][key], orig["steps"][key], equal_nan=True), key
    for a, b in zip(lib["rows"], orig["rows"]):
        assert {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k != "seconds"}
    # not trivially 0 = 0: the untrained model's I is positive somewhere
    assert np.nanmax(lib["curve_exp"]) > 0.1
