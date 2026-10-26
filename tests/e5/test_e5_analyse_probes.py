"""analyse_probes.py (docs/e5/DECISIONS.md (31)): E5's probes aggregated across a variable number of seeds per condition
(here 2 for the baseline, 4 for shuffle f = 1), invalid probes left out (never imputed), the newest measurement of a
(condition, seed, k) kept; on synthetic measure_probes.py outputs."""

from __future__ import annotations

import csv
import json

import numpy as np
import yaml

from _experiment_loader import load_experiment_module

ap = load_experiment_module("E5_belief_training", "analyse_probes.py", "e5_analyse_probes_under_test")

PROBES = [("segment", 0, "neutral"), ("position_only", 0, "ud"), ("segment", 5, "ud")]


def fake(root, training: str, fraction, seed: int, k: int, b: list[float], valid: list[bool], stamp: str):
    tag = "base" if training == "baseline" else f"sh{int(fraction * 100):03d}"
    d = root / "E5_belief_training" / f"probes__p1-b-d1__{tag}__seed{seed:02d}__k{k:06d}__{stamp}"
    d.mkdir(parents=True)
    (d / "meta.yaml").write_text(yaml.safe_dump({"training": training, "fraction": fraction, "seed": seed, "checkpoint_k": k}),
                                 encoding="utf-8")
    (d / "status.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    images = [{"image": 0, "zone": "nw", "role": "ud", "is_training": True}] + \
             [{"image": i + 1, "zone": "ne", "role": "x", "is_training": False} for i in range(len(PROBES))]
    with open(d / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(images[0]))
        w.writeheader()
        w.writerows(images)
    with open(d / "probes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "k", "position", "image", "slot", "valid"])
        w.writeheader()
        for i, (axis, kk, pos) in enumerate(PROBES):
            w.writerow({"axis": axis, "k": kk, "position": pos, "image": i + 1, "slot": "ne", "valid": valid[i]})
    exp = np.array([[-2.0]] + [[x] for x in b])
    np.savez_compressed(d / "curves.npz", curve_exp=np.concatenate([exp, np.zeros((len(exp), 2))], 1),
                        curve_ref=np.zeros((len(exp), 3)), n_states=np.zeros((len(exp), 3)))
    return d


def test_aggregation_over_a_variable_number_of_seeds(tmp_path):
    for seed in (102, 202):
        fake(tmp_path, "baseline", None, seed, 100000, [-0.1, -2.0, -1.0], [True, True, seed == 102], "20261030T000000Z")
    for i, seed in enumerate((2, 102, 202, 302)):
        fake(tmp_path, "shuffle", 1.0, seed, 100000, [-1.0 - i, 0.0, -0.5], [True, True, True], "20261030T000000Z")
    fake(tmp_path, "shuffle", 1.0, 2, 100000, [-9.0, -9.0, -9.0], [True, True, True], "20261029T000000Z")  # older: dropped
    out = tmp_path / "out"
    ap.main(["--root", str(tmp_path), "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == {"baseline": [[102, 100000], [202, 100000]], "shuffle_f1": [[2, 100000], [102, 100000], [202, 100000], [302, 100000]]}
    with open(out / "probes_aggregated.csv", newline="", encoding="utf-8") as f:
        rows = {(r["condition"], r["axis"], int(r["k"]), r["position"]): r for r in csv.DictReader(f)}
    theme = rows[("shuffle_f1", "segment", 0, "neutral")]
    assert int(theme["n_seeds"]) == 4 and int(theme["n_valid"]) == 4 and np.isclose(float(theme["b_step1_mean"]), -2.5)
    assert np.isclose(float(theme["b_step1_sd"]), np.std([-1, -2, -3, -4], ddof=1))
    partial = rows[("baseline", "segment", 5, "ud")]  # valid for seed 102 only
    assert (int(partial["n_seeds"]), int(partial["n_valid"]), partial["seeds_valid"]) == (2, 1, "102")
    assert np.isclose(float(partial["b_step1_mean"]), -1.0) and partial["b_step1_sd"] == "nan"
    with open(out / "extremes.csv", newline="", encoding="utf-8") as f:
        extremes = list(csv.DictReader(f))
    assert {e["probe"] for e in extremes} == {"training_image_ud", "theme_only", "position_only"}
    assert all(float(e["b_step1"]) == -2.0 for e in extremes if e["probe"] == "training_image_ud")



def test_two_probe_sets_merge_each_pair_at_its_checkpoint(tmp_path):
    """docs/e5/DECISIONS.md (33): the baselines at 300,000 and each shuffle pair at its own K (K_a for seed A, K_b for E
    and F) merge into one table, grouped by condition; a pair measured at two checkpoints is refused without --pair-k."""
    import pytest

    fake(tmp_path, "baseline", None, 2, 300000, [-0.1, -2.0, -1.0], [True] * 3, "20261028T000000Z")
    fake(tmp_path, "baseline", None, 102, 300000, [-0.2, -2.1, -1.1], [True] * 3, "20261028T000000Z")
    fake(tmp_path, "shuffle", 1.0, 2, 550000, [-1.0, -0.1, -0.5], [True] * 3, "20261028T000000Z")   # K_a
    fake(tmp_path, "shuffle", 1.0, 402, 600000, [-1.4, -0.2, -0.5], [True] * 3, "20261030T000000Z")  # K_b
    fake(tmp_path, "shuffle", 1.0, 502, 600000, [-1.6, -0.3, -0.5], [True] * 3, "20261030T000000Z")
    out = tmp_path / "out"
    ap.main(["--root", str(tmp_path), "--out", str(out)])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == {"baseline": [[2, 300000], [102, 300000]], "shuffle_f1": [[2, 550000], [402, 600000], [502, 600000]]}
    with open(out / "probes_aggregated.csv", newline="", encoding="utf-8") as f:
        rows = {(r["condition"], r["axis"], int(r["k"]), r["position"]): r for r in csv.DictReader(f)}
    theme = rows[("shuffle_f1", "segment", 0, "neutral")]
    assert int(theme["n_seeds"]) == 3 and np.isclose(float(theme["b_step1_mean"]), np.mean([-1.0, -1.4, -1.6]))
    fake(tmp_path, "shuffle", 1.0, 2, 600000, [-1.2, -0.1, -0.5], [True] * 3, "20261031T000000Z")  # seed A again, another k
    with pytest.raises(SystemExit):
        ap.main(["--root", str(tmp_path), "--out", str(tmp_path / "out2")])
    ap.main(["--root", str(tmp_path), "--out", str(tmp_path / "out3"), "--pair-k", "shuffle_f1:2=550000"])
