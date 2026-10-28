"""E6/E7 on all agents (docs/e6/DECISIONS.md (63)): the configs differ from E6's and E7's by the test images only, every
agent's images resolve, and run_e7_d7's options default to D7 (plan B's overlay kept, the diagnostics' image)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline import frozen_setup as fs
from helplessness.pipeline import two_zones_setup as setup

rdiag = load_experiment_module("E7_latent_manipulation", "run_e7_diagnostics.py", "e7_diagnostics_under_all_test")
d7 = load_experiment_module("E7_latent_manipulation", "run_e7_d7.py", "e7_d7_under_all_test")


def raw(*parts):
    return yaml.safe_load(REPO_ROOT.joinpath(*parts).read_text(encoding="utf-8"))


def test_all_configs_change_only_the_images():
    e6, e6_all = raw("configs", "e6", "e6.yaml"), raw("configs", "e6", "e6_all.yaml")
    assert e6_all["base"] == e6["base"]
    assert {k: v for k, v in e6_all["e6"].items() if k != "images"} == {k: v for k, v in e6["e6"].items() if k != "images"}
    e7, e7_all = raw("configs", "e7", "e7.yaml"), raw("configs", "e7", "e7_all.yaml")
    assert e7_all["base"] == "configs/e6/e6_all.yaml" and e7_all["e7"] == e7["e7"]
    cfg = fs.load_layered_config(REPO_ROOT / "configs" / "e7" / "e7_all.yaml", ("e6", "e7"))
    assert cfg["_bases"] == ["configs/e4/e4.yaml", "configs/e6/e6_all.yaml"]


def test_every_agent_has_its_images():
    cfg = fs.load_layered_config(REPO_ROOT / "configs" / "e6" / "e6_all.yaml", ("e6",))
    names = raw("configs", "e6", "images_by_agent_all.yaml")
    agents = raw("configs", "e6", "agents_e5.yaml")["agents"]
    assert sorted(names) == sorted(a["id"] for a in agents)
    specs = {s["name"]: s for s in cfg["e6"]["images"]}
    assert len(specs) == len(cfg["e6"]["images"])
    for a in agents:
        assert names[a["id"]][0] == "training" and len(set(names[a["id"]])) == len(names[a["id"]])
        conf = {**setup.configuration(cfg, a["configuration"]), "seed": a["seed"]}
        images = fs.test_images(conf, [specs[n] for n in names[a["id"]]])  # each spec names one probe image
        assert images[0]["is_training"] and not any(im["is_training"] for im in images[1:])


def test_induction_items_default_and_without_plan_b(tmp_path):
    st = SimpleNamespace(e7={"induction": {"alphas": [0.0, 0.5, 1.0]}})
    np.savez(tmp_path / "directions.npz", **{"reverse__v_hat": np.ones(3), "reverse__random": -np.ones(3)})
    full = rdiag.induction_items(st, tmp_path)
    assert [it["mode"] for it in full] == ["none", "reverse_none", "reverse", "reverse", "reverse_random", "reverse_random"]
    plain = rdiag.induction_items(st, None, plan_b=False)  # directions never read
    assert plain == full[:2]


def test_d7_options_are_natural_only():
    with pytest.raises(SystemExit):
        d7.main(["--phase", "corrected", "--agents", "x", "--runs-root", "y", "--image", "training"])
    with pytest.raises(SystemExit):
        d7.main(["--phase", "corrected", "--agents", "x", "--runs-root", "y", "--no-overlay"])
