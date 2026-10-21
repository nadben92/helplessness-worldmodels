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

ORIGINAL = "f8b2dc5"  # the last e4 commit merged into e5 (E4 closed, 2026-10-21); its measure code is the pre-move original
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


def test_e5_validity_hook_leaves_the_measure_unchanged_and_flags_each_probe(models):
    """E5's probes (measure_probes.py's validity hook, helplessness.probing.probe_validity): the measure with the hook is
    the measure without it; I at step 1 of the reference is its curve at step 1; `valid` is E4's criterion on the
    reference; the reconstruction diagnostic covers every colour the slot takes in training."""
    from _experiment_loader import load_experiment_module
    from helplessness.probing import probe_validity

    e5_probes = load_experiment_module("E5_belief_training", "measure_probes.py", "test_e5_measure_probes_validity")
    cfg = mini_cfg()
    device = configure_runtime(cfg)
    conf = two_zones_setup.configuration(cfg, "p1-a-d1")
    images_all = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    images = [images_all[i] for i in (0, 92)]
    plain = theme_probe_measure.measure(cfg, conf, models, images, device, log=lambda s: None, n_walks=N_WALKS)
    diagnostics: list[dict] = []
    hook = e5_probes.validity_hook({**conf, "cfg": cfg}, models, 3, diagnostics)
    hooked = theme_probe_measure.measure(cfg, conf, models, images, device, log=lambda s: None, n_walks=N_WALKS, on_pair=hook)
    for key in ("curve_exp", "curve_ref"):
        assert np.array_equal(plain[key], hooked[key], equal_nan=True)
    for i, (a, b) in enumerate(zip(plain["rows"], hooked["rows"])):
        assert {k: v for k, v in a.items() if k != "seconds"} == {k: v for k, v in b.items() if k in a and k != "seconds"}
        assert np.isclose(b["I_ref_step1"], hooked["curve_ref"][i, 0]) and np.isclose(b["I_exp_step1"], hooked["curve_exp"][i, 0])
        assert b["valid"] == probe_validity.is_valid(b["mass_ref_step1"], b["I_ref_step1"])
        assert 0.0 <= b["mass_ref_step1"] <= 1.0 and 0.0 <= b["ghost_ref_step1"] <= 1.0
    assert len(diagnostics) == len(hooked["rows"]) * 2 * len(probe_validity.PARTS)
    assert {"zone_rgb_dist_to_canonical_training", "zone_rgb_dist_to_T_inc", "zone_rgb_dist_to_T_ctrl", "zone_rgb_dist_to_corridor",
            "zone_rgb_dist_to_probe", "zone_rgb_nearest", "agent_cell_accuracy"} <= set(diagnostics[0])
    summary = e5_probes.validity_summary([{"axis": "segment", "position": "ud", "valid": True}, {"axis": "segment", "position": "ud", "valid": False}])
    assert summary["share_valid"] == 0.5 and summary["by_axis_position"]["segment/ud"] == {"probes": 2, "valid": 1}
