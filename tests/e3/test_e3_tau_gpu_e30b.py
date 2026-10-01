"""tau_gpu_e30b.py (docs/e3/DECISIONS.md, 2026-10-01 (10)) on synthetic archives laid out as the
measurement batch and gpu_e30b's handoff: the batch check, the consistency comparison with the
original measurements, the thresholds by the fixed definitions and the verdict, in one command."""

from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

tau = load_experiment_module("E3_worldmodel_4zones", "tau_gpu_e30b.py", "test_tau_gpu_e30b")
CATS = tau.agg.CATEGORIES
RUNS = tau.cluster.MEASURE_BATCHES[tau.BATCH]["runs"]
# I_s and I_s projected by zone (UD, CN, UN, CD) for each run
I_S = {"win64_seed0_experimental": [0.10, 1.70, 0.12, 2.25], "win64_seed0_reference": [2.30, 1.70, 1.80, 2.26],
       "win64_seed5_experimental": [0.08, 1.80, 0.07, 2.23], "win64_seed5_reference": [2.30, 1.79, 1.57, 2.27]}
I_PROJ = {"win64_seed0_experimental": [0.05, 1.60, 0.05, 2.25], "win64_seed0_reference": [2.30, 1.62, 1.75, 2.26],
          "win64_seed5_experimental": [0.03, 1.75, 0.02, 2.23], "win64_seed5_reference": [2.30, 1.70, 1.55, 2.27]}


def _write(d, job, recomputed: bool, source="a" * 64, shift=0.0):
    d.mkdir(parents=True)
    seed, condition = int(job.split("seed")[1][0]), job.split("_")[-1]
    zone_of_state = [CATS[s % 4] for s in range(144)]
    rc = {"kernel_draws": 1358, "l1_floor_by_zone": {c: 0.0 for c in CATS},
          "I_s_by_zone": {c: {"mean": I_S[job][k] + shift, "se": 0.01} for k, c in enumerate(CATS)},
          "I_z_by_zone": {c: {"mean": I_S[job][k] + 0.2, "se": 0.02} for k, c in enumerate(CATS)}}
    results = {"seed": seed, "condition": condition, "zone_of_state": zone_of_state, "accuracies": {"acc_replay_hard": 0.995},
               "z_only_probe": {"test_accuracy": 0.995}, "real_context": rc}
    if recomputed:
        rc["I_s_projected_by_zone"] = {c: {"mean": I_PROJ[job][k], "se": 0.01} for k, c in enumerate(CATS)}
        results.update(run_dir=f"/scratch/x/results/E3_worldmodel_4zones/{RUNS[job][0]}", weights_sha256=RUNS[job][1],
                       reproducibility={"source_sha256": source, "device": "cuda:0", "gpu_name": "A100", "allow_tf32_matmul": False,
                                        "allow_tf32_cudnn": False})
    (d / "results.json").write_text(json.dumps(results), encoding="utf-8")
    P = np.zeros((144, 5, 144))
    P[np.arange(144), :, np.arange(144)] = 1.0
    leak = {"win64_seed0_experimental": 0.05, "win64_seed5_experimental": 0.06}.get(job, 0.01)
    p_hat = P * (1 - leak)
    p_hat[:, :, 143 - np.arange(144) % 3] += leak  # mass on far states: off the support
    per_state = np.array([I_S[job][s % 4] for s in range(144)])
    np.savez(d / "arrays.npz", N=np.full((144, 5, 144), 1.0), P_true=P, T_hat_B=P, real_p_hat=p_hat, real_I_s=per_state + shift,
             real_I_z=per_state + 0.2)


def _archives(tmp_path, source=None, shift=None):
    measures, originals = tmp_path / "measures", tmp_path / "originals"
    measures.mkdir()
    originals.mkdir()
    for job, (name, _) in RUNS.items():
        org = tmp_path / "build" / "org" / name
        rec = tmp_path / "build" / "rec" / name / "recomputed" / "final__20261001T000000Z"
        _write(org, job, False)
        _write(rec, job, True, source=(source or {}).get(job, "a" * 64), shift=(shift or {}).get(job, 0.0))
        with tarfile.open(originals / f"{name}.tgz", "w:gz") as t:
            t.add(org, arcname=f"results/E3_worldmodel_4zones/{name}")
        with tarfile.open(measures / f"{name}.recomputed.tgz", "w:gz") as t:
            t.add(tmp_path / "build" / "rec" / name, arcname=name)
    return measures, originals


def test_thresholds_and_verdict_in_one_command(tmp_path):
    measures, originals = _archives(tmp_path)
    out = tmp_path / "out"
    assert tau.main(["--measures", str(measures), "--originals", str(originals), "--out", str(out)]) == 0
    rules = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    # CN: b(I_s) - b(I_s projected) = (1.70 - 1.70) - (1.60 - 1.62) = +0.02 for seed 0, (1.80 - 1.79) - (1.75 - 1.70) = -0.04 for seed 5
    assert rules["tau_b"][CATS[1]] == pytest.approx(2 * np.std([0.02, -0.04], ddof=1))
    assert set(rules["tau_b"]) == set(rules["tau_off"]) == {CATS[1], CATS[3]}
    assert set(rules["floor"]["zones"]) == {CATS[0], CATS[2]}
    text = (out / "TAU.md").read_text(encoding="utf-8")
    assert "0 of 16 (run, zone) rows differ" in text and "- passed" in text
    assert text.count("| 0 | ") == 4 and text.count("| 5 | ") == 4  # every zone of both pairs has a verdict
    assert (out / "B_BY_ZONE.md").exists()
    # the output directory is never written over
    with pytest.raises(SystemExit):
        tau.main(["--measures", str(measures), "--originals", str(originals), "--out", str(out)])


def test_a_re_measurement_that_does_not_reproduce_the_original_is_flagged(tmp_path):
    measures, originals = _archives(tmp_path, shift={"win64_seed5_reference": 0.01})
    out = tmp_path / "out"
    assert tau.main(["--measures", str(measures), "--originals", str(originals), "--out", str(out)]) == 1
    text = (out / "TAU.md").read_text(encoding="utf-8")
    assert "4 of 16 (run, zone) rows differ" in text and "DIFFERS -- to be reported" in text


def test_two_code_versions_in_one_batch_fail_the_check(tmp_path):
    measures, originals = _archives(tmp_path, source={"win64_seed0_reference": "b" * 64})
    out = tmp_path / "out"
    assert tau.main(["--measures", str(measures), "--originals", str(originals), "--out", str(out)]) == 1
    assert "the four runs differ in source_sha256" in (out / "TAU.md").read_text(encoding="utf-8")
