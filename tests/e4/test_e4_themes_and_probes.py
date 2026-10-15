"""E4's colours (configs/e4/themes.yaml, helplessness.envs.appearance.theme_axes) and probe set
(helplessness.probing.theme_probes): every property the design requires, checked numerically."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.envs.appearance import theme_axes as ta
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR, GOAL_COLOR
from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.observation.allocentric_pixel_backend import render
from helplessness.probing.theme_probes import probe_images

e4_setup = load_experiment_module("E4_generalization_gradient", "e4_setup.py", "test_e4_themes_setup")
choose_themes = load_experiment_module("E4_generalization_gradient", "choose_themes.py", "test_e4_choose_themes")

THEMES = yaml.safe_load((REPO_ROOT / "configs" / "e4" / "themes.yaml").read_text(encoding="utf-8"))
CFG = e4_setup.load_config()
PAIRS = list(THEMES["pairs"].values())
MIN_DIST = THEMES["min_dist_cielab"]
ROUND = np.sqrt(3) * 0.5  # the largest RGB displacement of rounding a point to integers


@pytest.mark.parametrize("pair", PAIRS, ids=list(THEMES["pairs"]))
def test_axes_geometry(pair):
    a, b, D = np.array(pair["a"]), np.array(pair["b"]), pair["D"]
    assert D == pytest.approx(ta.rgb_distance(a, b))
    seg, arc_b, arc_a = (np.array(pair[k]) for k in ("segment", "arc_center_b", "arc_center_a"))
    assert seg.shape == arc_b.shape == arc_a.shape == (11, 3)
    assert np.array_equal(seg[0], a) and np.array_equal(seg[-1], b)
    for k in range(11):  # segment: t = k / 10
        assert ta.rgb_distance(seg[k], a + k / 10 * (b - a)) <= ROUND
    for arc, center, start in ((arc_b, b, a), (arc_a, a, b)):
        assert np.array_equal(arc[0], start)
        assert np.all(np.abs(ta.rgb_distance(arc, center) - D) <= ROUND)                       # constant distance to the centre
        assert np.all(np.abs(ta.rgb_distance(arc, start) - np.arange(11) * D / 10) <= ROUND)   # distance to the start 0 .. D
    # the arcs end at different colours, D/10 apart at least (the plane rule), and both orientations share the axes
    assert ta.rgb_distance(arc_a[1:, None, :], arc_b[None, :, :]).min() >= D / 10 - 2 * ROUND
    o_a, o_b = (ta.axes_for_orientation({"segment": seg, "arc_center_b": arc_b, "arc_center_a": arc_a}, inc) for inc in ("a", "b"))
    assert np.array_equal(o_a["segment"], o_b["segment"][::-1]) and np.array_equal(o_a["arc_ctrl"], o_b["arc_inc"])


@pytest.mark.parametrize("pair", PAIRS, ids=list(THEMES["pairs"]))
def test_axes_stay_in_the_cube_and_away_from_agent_goal_corridor(pair):
    pts = np.concatenate([np.array(pair[k]) for k in ("segment", "arc_center_b", "arc_center_a")])
    assert ta.in_cube(pts)
    for name, color in (("agent", THEMES["agent_marker_color"]), ("goal", GOAL_COLOR), ("corridor", CORRIDOR_COLOR)):
        assert ta.lab_distances(pts, [color]).min() >= MIN_DIST, name
        assert pair["min_cielab_to"][name] == pytest.approx(ta.lab_distances(pts, [color]).min())


def test_pairs_are_distinct_and_the_file_is_reproducible():
    p1, p2 = PAIRS
    assert ta.lab_distances(np.array([p2["a"], p2["b"]]), np.array([p1["a"], p1["b"]])).min() >= MIN_DIST
    assert THEMES["pair2_vs_pair1"]["angle_deg"] >= THEMES["min_angle_deg"]
    assert THEMES["agent_marker_color"] == list(THEMES["agent_marker_color"])
    solved = choose_themes.solve()
    assert solved["agent_marker_color"] == THEMES["agent_marker_color"]
    for name in THEMES["pairs"]:
        for key in ("a", "b", "segment", "arc_center_b", "arc_center_a"):
            assert solved["pairs"][name][key] == THEMES["pairs"][name][key]


@pytest.mark.parametrize("config_id", list(CFG["e4"]["configurations"]))
def test_93_distinct_probe_images(config_id):
    conf = e4_setup.configuration(CFG, config_id)
    images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    assert len(images) == 93
    assert sum(len(im["probes"]) for im in images) == 100
    assert images[0]["is_training"] and images[0]["slot_colors"] == conf["training_colors"]
    assert images[0]["zones"] == list(QUADRANT_ORDER)
    assert sum(len(im["zones"]) for im in images) == 96
    # every probe kept, with its zone of interest; the training image stands for the four identity probes
    training_probes = {(p["axis"], p["k"], p["position"]) for p in images[0]["probes"]}
    assert training_probes == {("segment", 0, "ud"), ("arc_ctrl", 0, "ud"), ("segment", 10, "cd"), ("arc_inc", 0, "cd")}
    for im in images[1:]:
        assert len(im["zones"]) == 1 and all(p["slot"] == im["zones"][0] for p in im["probes"])
    # the images themselves (rendered) are pairwise distinct
    layout = two_zones.build_layout(conf["diagonal"], "reference")
    rendered = {render(0, layout, two_zones.themes(im["slot_colors"], conf["agent_color"])).tobytes() for im in images}
    assert len(rendered) == 93


def test_probe_images_change_one_slot_only():
    conf = e4_setup.configuration(CFG, "p1-a-d1")
    images = probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])
    for im in images[1:]:
        changed = [q for q in im["slot_colors"] if im["slot_colors"][q] != conf["training_colors"][q]]
        assert changed == im["zones"]
    assert conf["roles"]["neutral"] not in (conf["roles"]["ud"], conf["roles"]["cd"])
