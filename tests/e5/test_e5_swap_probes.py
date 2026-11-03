"""E5's theme-swap probe set (docs/e5/DECISIONS.md (38)): the base images are the shuffle's training assignments 4, 11
and 6, the gradient changes the swapped slot only, the near-duplicate flags, the zone roles, the batch e5_probes_swap
(the shuffle pairs at --checkpoint, the baselines at 100,000 and 300,000, the probe set's arguments) and measure_probes.py's
swap rows. The default probe set is checked bit for bit against the previous code outside the tests (identity on real
weights, reported in (38)'s implementation note)."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.assignment_records import ASSIGNMENT_IMAGE_BASE
from helplessness.pipeline.episode_sources import ASSIGNMENTS, assignment_colors, assignment_index
from helplessness.probing import swap_probes
from helplessness.probing.theme_probes import probe_images

batches = load_experiment_module("E5_belief_training", "batches.py", "e5_batches_swap_under_test")
submit = load_experiment_module("E5_belief_training", "submit_kubeflow.py", "test_e5_submit_swap")
e5_setup = load_experiment_module("E5_belief_training", "e5_setup.py", "test_e5_swap_setup")
measure_probes = load_experiment_module("E5_belief_training", "measure_probes.py", "test_e5_measure_probes_swap")


@pytest.fixture(scope="module")
def conf():
    cfg = e5_setup.load_config(e5_setup.DEFAULT_CONFIG)
    return two_zones_setup.configuration(cfg, batches.CONFIGURATION)


@pytest.fixture(scope="module")
def images(conf):
    return swap_probes.swap_probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])


def base_of(images, m):
    return next(im for im in images if im["swap"] == m and any(p["axis"] == "segment" and p["k"] == 0 for p in im["probes"]))


def test_base_images_are_training_assignments(conf, images):
    """p1-b-d1 (UD nw, CD se): the swaps to ne, sw and se are assignments 4 (ne, se), 11 (sw, se) and 6 (se, nw), and each
    base image is the exchange of T_inc with the slot's colour in the canonical image."""
    assert swap_probes.swap_targets(conf["diagonal"]) == ("ne", "se", "sw")
    expected = {"ne": 4, "sw": 11, "se": 6}
    for m, k in expected.items():
        base = base_of(images, m)
        assert assignment_index(base["ud_slot"], base["cd_slot"]) == k
        assert base["slot_colors"] == assignment_colors(k, conf["diagonal"], conf["t_inc"], conf["t_ctrl"])
        swapped = dict(conf["training_colors"])
        swapped["nw"], swapped[m] = swapped[m], swapped["nw"]
        assert base["slot_colors"] == swapped
        assert ASSIGNMENTS[k][0] == m


def test_gradient_changes_the_swapped_slot_only(conf, images):
    assert len(images) == 93  # 3 swaps x 31 colours
    keys = {tuple(im["slot_colors"][q] for q in QUADRANT_ORDER) for im in images}
    assert len(keys) == 93
    e4 = {tuple(im["slot_colors"][q] for q in QUADRANT_ORDER) for im in probe_images(conf["diagonal"], conf["t_inc"], conf["t_ctrl"], conf["axes"])}
    # one image only is also an E4 probe image: the swap to se at T_ctrl (segment t = 1, arc_inc t = 0) puts T_ctrl at nw
    # and se, E4's T_ctrl at the old UD slot (measured here with its own seed: a cross-check, (38)'s implementation note)
    t_ctrl = tuple(conf["t_ctrl"])
    assert keys & e4 == {(t_ctrl, (128, 128, 128), t_ctrl, (128, 128, 128))}
    for im in images:
        base = base_of(images, im["swap"])
        assert all(im["slot_colors"][q] == base["slot_colors"][q] for q in QUADRANT_ORDER if q != im["swap"])
        assert im["zones"] == list(QUADRANT_ORDER)
        assert im["is_training"] is False
        assert im["index"] >= swap_probes.SWAP_IMAGE_BASE > ASSIGNMENT_IMAGE_BASE + len(ASSIGNMENTS)
        for p in im["probes"]:
            assert im["slot_colors"][im["swap"]] == tuple(int(v) for v in conf["axes"][p["axis"]][p["k"]])
            assert p["t"] == p["k"] / 10
    per_swap = {m: sum(len(im["probes"]) for im in images if im["swap"] == m) for m in ("ne", "se", "sw")}
    assert per_swap == {"ne": 33, "se": 33, "sw": 33}
    assert [im["index"] for im in images] == list(range(swap_probes.SWAP_IMAGE_BASE, swap_probes.SWAP_IMAGE_BASE + 93))


def test_arc_inc_is_at_distance_D_from_t_inc(conf):
    """docs/e5/DECISIONS.md (38), amendment 1: arc_inc starts at T_ctrl and stays at D from T_inc."""
    d = swap_probes.rgb_distance(conf["axes"]["arc_inc"], conf["t_inc"]) / conf["D"]
    assert np.allclose(d, 1.0, atol=0.01)
    assert tuple(conf["axes"]["arc_inc"][0]) == tuple(conf["t_ctrl"])
    assert tuple(conf["axes"]["arc_ctrl"][0]) == tuple(conf["axes"]["segment"][0]) == tuple(conf["t_inc"])


def test_near_duplicates(images):
    """(38), amendment 2: segment t >= 0.8 and arc_inc t <= 0.2 (near T_ctrl, always in another slot), in every swap."""
    flagged = {(im["swap"], p["axis"], p["k"]) for im in images for p in im["probes"] if p["near_duplicate"]}
    expected = {(m, "segment", k) for m in ("ne", "se", "sw") for k in (8, 9, 10)} | {(m, "arc_inc", k) for m in ("ne", "se", "sw") for k in (0, 1, 2)}
    assert flagged == expected


def test_zone_roles(images):
    for m, roles in {"ne": {"nw": "neutral", "ne": "ud", "se": "cd", "sw": "neutral"},
                     "se": {"nw": "cd", "ne": "neutral", "se": "ud", "sw": "neutral"},
                     "sw": {"nw": "neutral", "ne": "neutral", "se": "cd", "sw": "ud"}}.items():
        im = base_of(images, m)
        assert {q: swap_probes.zone_role(im, q) for q in QUADRANT_ORDER} == roles
        assert im["old_ud_slot"] == "nw"


def test_batch_e5_probes_swap():
    assert batches.batch_runs("e5_probes_swap") == {}
    specs = batches.probe_specs("e5_probes_swap", [650000])
    assert specs == [("A", "shuffle", 1.0, 650000), ("E", "shuffle", 1.0, 650000), ("F", "shuffle", 1.0, 650000),
                     ("A", "baseline", None, 100000), ("A", "baseline", None, 300000),
                     ("B", "baseline", None, 100000), ("B", "baseline", None, 300000)]
    with pytest.raises(ValueError):  # baseline A is E4's own pair
        batches.probe_args("e5_probes_swap", batches.CONFIGURATION, [650000])
    jobs = dict(batches.probe_args("e5_probes_swap", batches.CONFIGURATION, [650000], "/e4"))
    assert list(jobs) == ["swap_A_sh100_k650000", "swap_E_sh100_k650000", "swap_F_sh100_k650000", "swap_A_base_k100000",
                          "swap_A_base_k300000", "swap_B_base_k100000", "swap_B_base_k300000"]
    for name, args in jobs.items():
        assert args[-4:] == ["--probe-set", "swap", "--diagnose-walks", "150"]
    assert jobs["swap_E_sh100_k650000"][jobs["swap_E_sh100_k650000"].index("--reference-seed-label") + 1] == "B"
    assert "--e4-results" in jobs["swap_A_base_k100000"] and "--e4-results" not in jobs["swap_B_base_k300000"]
    cmds = dict(submit.commands("probes", "py", "e5_probes_swap", batches.CONFIGURATION, checkpoint=[650000], e4_results="/e4"))
    assert list(cmds) == list(jobs)
    assert all("--save-steps" in c and c[c.index("--probe-set") + 1] == "swap" for c in cmds.values())


def test_other_probe_batches_unchanged():
    """The probe set's arguments and the fixed checkpoints belong to e5_probes_swap only."""
    for batch in ("e5_probes_a", "e5_probes_b", "e5_probes_baselines", "e5_probes_all"):
        for name, args in batches.probe_args(batch, batches.CONFIGURATION, [475000], "/e4"):
            assert name.startswith("probes_") and "--probe-set" not in args and "--diagnose-walks" not in args
    assert batches.probe_specs("e5_probes_baselines", [63096]) == [("A", "baseline", None, 300000), ("B", "baseline", None, 300000),
                                                                    ("B", "baseline", None, 63096)]


def test_swap_rows(conf, images):
    """measure_probes.swap_rows: one row per (probe, zone), b at step 1 from the curves, kept = valid and localised."""
    im = base_of(images, "ne")  # stands for segment k = 0 and arc_ctrl k = 0
    rows, diag = [], []
    for j, q in enumerate(QUADRANT_ORDER):
        rows.append({"image": im["index"], "zone": q, "b_entry": 0.0, "mass_ref_step1": 0.95, "I_ref_step1": 2.3, "valid": q != "se",
                     **{f"color_{s}": "-".join(map(str, im["slot_colors"][s])) for s in QUADRANT_ORDER}})
        diag += [{"image": im["index"], "zone": q, "model": "experimental", "part": "zone_step1", "agent_cell_accuracy": 0.95 if q != "sw" else 0.5},
                 {"image": im["index"], "zone": q, "model": "reference", "part": "zone_step1", "agent_cell_accuracy": 1.0}]
    curve_exp = np.array([[0.2 + j, 1.0] for j in range(4)])
    curve_ref = np.array([[2.3, 2.3]] * 4)
    out = measure_probes.swap_rows(conf, [im], rows, curve_exp, curve_ref, diag)
    assert len(out) == 8 and {r["axis"] for r in out} == {"segment", "arc_ctrl"}
    by_zone = {r["zone"]: r for r in out if r["axis"] == "segment"}
    assert by_zone["nw"]["b_step1"] == pytest.approx(0.2 - 2.3) and by_zone["nw"]["is_old_ud"] and by_zone["nw"]["zone_role"] == "neutral"
    assert by_zone["ne"]["zone_role"] == "ud" and by_zone["ne"]["dist_to_inc"] == 0.0
    assert [by_zone[q]["kept"] for q in QUADRANT_ORDER] == [True, True, False, False]  # se invalid, sw not localised
    summary = measure_probes.swap_summary(out, 150)
    assert (summary["rows"], summary["valid"], summary["kept"]) == (8, 6, 4)
    assert summary["by_swap_axis_role"]["ne/segment/neutral"] == {"rows": 2, "valid": 2, "kept": 1}


def test_export_collects_the_swap_measurements(tmp_path):
    """export_batch.py: e5_probes_swap exports the probes-swap__ directories with swap_probes.csv (validity, localisation,
    kept); the other probe batches still export probes__ directories only, with their files."""
    import re
    import shutil

    from helplessness.config import REPO_ROOT

    export = load_experiment_module("E5_belief_training", "export_batch.py", "test_e5_export_swap")
    e5_cfg = REPO_ROOT / batches.CONFIG
    base = re.search(r"^base:\s*(\S+)", e5_cfg.read_text(encoding="utf-8"), re.M).group(1)
    for rel in (batches.CONFIG, base):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    root = tmp_path / "results" / "E5_belief_training"
    for prefix, names in (("probes-swap", export.SWAP_PROBE_FILES), ("probes", export.PROBE_FILES)):
        d = root / f"{prefix}__p1-b-d1__sh100__seed402__k650000__20261102T000000Z"
        d.mkdir(parents=True)
        for name in set(names) | {"reconstruction.csv"}:
            (d / name).write_text("x", encoding="utf-8")
    swap, _ = export.collect(tmp_path, "e5_probes_swap", "p1-b-d1", [], False)
    assert {p.parent.name.split("__")[0] for p in swap} == {"probes-swap"}
    assert {p.name for p in swap} == set(export.SWAP_PROBE_FILES)
    other, _ = export.collect(tmp_path, "e5_probes_b", "p1-b-d1", [], False)
    assert {p.parent.name.split("__")[0] for p in other} == {"probes"}
    assert {p.name for p in other} == set(export.PROBE_FILES)


def test_reading_rule_plateau():
    """analyse_swap.verdict, (38): the plateau is the mean of the 3 furthest counted points (t >= 0.5); colour (> -0.5),
    arrangement (<= -1.5), both; an agent needs Delta(0) about -2 on the segment and arc_ctrl, not on arc_inc."""
    analyse = load_experiment_module("E5_belief_training", "analysis_swap/analyse_swap.py", "test_e5_analyse_swap")
    ts = [k / 10 for k in range(11)]
    linear = {t: (-2.0 * (1 - t), t >= 0.8) for t in ts}
    v = analyse.verdict(linear, "segment")
    assert v["verdict"] == "colour" and v["plateau_t"] == [0.8, 0.9, 1.0] and v["plateau"] == pytest.approx(-0.2)
    assert v["plateau_without_near_duplicates_t"] == [0.5, 0.6, 0.7] and v["plateau_without_near_duplicates"] == pytest.approx(-0.8)
    assert analyse.verdict({t: (-2.0, False) for t in ts}, "arc_ctrl")["verdict"] == "arrangement"
    assert analyse.verdict({t: (-2.0 + t, False) for t in ts}, "arc_ctrl")["verdict"] == "both"
    assert analyse.verdict({t: (-1.0, False) for t in ts}, "segment")["verdict"] == "not counted"  # Delta(0) not about -2
    few = {0.0: (-2.0, False), 0.5: (0.0, False), 0.6: (0.0, False)}
    assert analyse.verdict(few, "segment")["verdict"] == "not counted"  # 2 points with t >= 0.5
    assert analyse.verdict({0.6: (-0.1, False), 0.8: (0.0, False), 1.0: (-0.2, False)}, "arc_inc")["verdict"] == "colour"
    assert analyse.conclusion({"a": {"counted": True, "verdict": "colour"}, "b": {"counted": True, "verdict": "colour"},
                               "c": {"counted": False, "verdict": "not counted"}}) == "colour"
    assert analyse.conclusion({"a": {"counted": True, "verdict": "colour"}, "b": {"counted": False, "verdict": "not counted"}}) == "not readable"
    assert analyse.witness({0.0: (0.1, False), 0.9: (-0.7, False)}) == {"points": {"0.0": 0.1, "0.9": -0.7}, "reading": "departs",
                                                                        "departs_at": [0.9]}


def test_segment_secondary_and_shared_image(tmp_path):
    """(38): the segment's verdict also without its near-duplicate points; the shared image's control matches the E4-set
    row by its colours (T_ctrl at nw and se) and takes b at step 1 from curves.npz."""
    import yaml

    analyse = load_experiment_module("E5_belief_training", "analysis_swap/analyse_swap.py", "test_e5_analyse_swap_2")
    assert analyse.PRIMARY_AXES == ("arc_ctrl", "arc_inc") and analyse.SECONDARY_AXES == ("segment",)
    ts = [k / 10 for k in range(11)]
    v = analyse.verdict({t: (-2.0 * (1 - t), t >= 0.8) for t in ts}, "segment")
    assert (v["verdict"], v["verdict_without_near_duplicates"]) == ("colour", "both")
    t_ctrl, grey, t_inc = "227-17-15", "128-128-128", "22-0-244"
    d = tmp_path / "probes__p1-b-d1__base__seed02__k300000__x"
    d.mkdir()
    (d / "meta.yaml").write_text(yaml.safe_dump({"training": "baseline", "seed": 2, "checkpoint_k": 300000}), encoding="utf-8")
    images = [{"zone": "nw", "color_nw": t_inc, "color_ne": grey, "color_se": t_ctrl, "color_sw": grey, "valid": "True", "n_walks": "150"},
              {"zone": "nw", "color_nw": t_ctrl, "color_ne": grey, "color_se": t_ctrl, "color_sw": grey, "valid": "True", "n_walks": "150"}]
    analyse.write_csv(d / "images.csv", images)
    np.savez(d / "curves.npz", curve_exp=np.array([[0.1, 0.0], [0.05, 0.0]]), curve_ref=np.array([[2.3, 0.0], [2.3, 0.0]]))
    rows = [{"label": "baseline_A_k300000", "swap": "se", "axis": "segment", "k": "10", "zone": "nw", "zone_color": t_ctrl,
             "b_step1": "-2.2", "valid": "True", "kept": "True"}]
    out = analyse.shared_image(rows, [d])
    assert len(out) == 1 and out[0]["b_step1_e4_set"] == pytest.approx(0.05 - 2.3)
    assert out[0]["difference"] == pytest.approx(-2.2 - (0.05 - 2.3))


def test_batch_e5_probes_swap_A_refB(monkeypatch):
    """(39): one job, seed A's shuffle pair at --checkpoint against B's reference, its output named __refB__; every other
    batch keeps its reference and its names (no --name-reference)."""
    assert batches.batch_runs("e5_probes_swap_A_refB") == {}
    assert batches.REFERENCE_SEED_OF == {("E", "shuffle"): "B", ("F", "shuffle"): "B"}
    jobs = dict(batches.probe_args("e5_probes_swap_A_refB", batches.CONFIGURATION, [650000]))
    assert list(jobs) == ["swap_A_sh100_k650000"]
    args = jobs["swap_A_sh100_k650000"]
    assert args[args.index("--seed-label") + 1] == "A" and args[args.index("--reference-seed-label") + 1] == "B"
    assert "--name-reference" in args and args[-4:] == ["--probe-set", "swap", "--diagnose-walks", "150"]
    for batch in ("e5_probes_swap", "e5_probes_a", "e5_probes_b", "e5_probes_baselines"):
        for name, a in batches.probe_args(batch, batches.CONFIGURATION, [650000], "/e4"):
            assert "--name-reference" not in a
            if name.startswith(("swap_A_sh", "probes_A_sh")):
                assert "--reference-seed-label" not in a
    cmds = dict(submit.commands("probes", "py", "e5_probes_swap_A_refB", batches.CONFIGURATION, checkpoint=[650000]))
    assert list(cmds) == ["swap_A_sh100_k650000"] and "--name-reference" in cmds["swap_A_sh100_k650000"]
    # the output's name: unchanged without a reference label, __refB__ after the seed with it
    assert measure_probes.output_name(True, "p1-b-d1", "sh100", 2, 650000, "S") == "probes-swap__p1-b-d1__sh100__seed02__k650000__S"
    assert measure_probes.output_name(False, "p1-b-d1", "sh100", 402, 650000, "S") == "probes__p1-b-d1__sh100__seed402__k650000__S"
    assert (measure_probes.output_name(True, "p1-b-d1", "sh100", 2, 650000, "S", "B")
            == "probes-swap__p1-b-d1__sh100__seed02__refB__k650000__S")
    # the reference chosen: seed A's experimental run, seed B's (102) f = 1 reference
    calls = []
    monkeypatch.setattr(measure_probes.e5_setup, "find_run", lambda root, cid, training, condition, seed, fraction=None, need=None:
                        calls.append((training, condition, seed)) or f"{condition}{seed}")
    ns = type("Args", (), {"config": measure_probes.e5_setup.DEFAULT_CONFIG, "configuration": batches.CONFIGURATION, "seed_label": "A",
                           "reference_seed_label": "B", "find_runs": "results", "training": "shuffle", "fraction": 1.0,
                           "checkpoint": 650000, "e4_results": None})()
    assert measure_probes.find_pair(ns) == ("experimental2", "reference102")
    assert calls == [("shuffle", "experimental", 2), ("shuffle", "reference", 102)]
    with pytest.raises(SystemExit):  # --name-reference without a reference label
        measure_probes.main(["--checkpoint", "1", "--experimental-run", "x", "--reference-run", "y", "--name-reference"])


def test_export_and_analysis_of_the_refB_measurement(tmp_path):
    """(39): the export takes the __refB__ directory in its own batch only; analyse_swap labels it shuffle_A-refB_k650000,
    and the measurement of A against its own reference leaves the conclusions (reported, listed as superseded)."""
    import re
    import shutil

    from helplessness.config import REPO_ROOT

    export = load_experiment_module("E5_belief_training", "export_batch.py", "test_e5_export_swap_refB")
    analyse = load_experiment_module("E5_belief_training", "analysis_swap/analyse_swap.py", "test_e5_analyse_swap_refB")
    e5_cfg = REPO_ROOT / batches.CONFIG
    base = re.search(r"^base:\s*(\S+)", e5_cfg.read_text(encoding="utf-8"), re.M).group(1)
    for rel in (batches.CONFIG, base):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    root = tmp_path / "results" / "E5_belief_training"
    for name in ("probes-swap__p1-b-d1__sh100__seed02__k650000__X", "probes-swap__p1-b-d1__sh100__seed02__refB__k650000__Y"):
        (root / name).mkdir(parents=True)
        (root / name / "swap_probes.csv").write_text("x", encoding="utf-8")
    refb, _ = export.collect(tmp_path, "e5_probes_swap_A_refB", "p1-b-d1", [], False)
    assert {p.parent.name for p in refb} == {"probes-swap__p1-b-d1__sh100__seed02__refB__k650000__Y"}
    own, _ = export.collect(tmp_path, "e5_probes_swap", "p1-b-d1", [], False)
    assert {p.parent.name for p in own} == {"probes-swap__p1-b-d1__sh100__seed02__k650000__X"}
    meta = {"training": "shuffle", "seed": 2, "checkpoint_k": 650000}
    assert analyse.label_of(meta) == "shuffle_A_k650000"
    assert analyse.label_of({**meta, "reference_seed_label": "B"}) == "shuffle_A-refB_k650000"
    labels = ["shuffle_A-refB_k650000", "shuffle_A_k650000", "shuffle_E_k650000", "shuffle_F_k650000"]
    assert analyse.superseded(labels) == ["shuffle_A_k650000"] and analyse.superseded(labels[1:]) == []
    readings = {"shuffle_A_k650000": {"counted": True, "verdict": "both"}, "shuffle_A-refB_k650000": {"counted": False, "verdict": "not counted"},
                "shuffle_F_k650000": {"counted": True, "verdict": "both"}}
    assert analyse.conclusion(readings) == "both"
    assert analyse.conclusion(readings, exclude=("shuffle_A_k650000",)) == "not readable"
