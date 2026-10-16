"""E4's batch table and the commands of its submission script (no cluster involved)."""

from __future__ import annotations

import pytest

from _experiment_loader import load_experiment_module
from helplessness.cluster.kubeflow import runtime_patch_secrets

batches = load_experiment_module("E4_generalization_gradient", "batches.py", "e4_batches_under_test")
submit = load_experiment_module("E4_generalization_gradient", "submit_kubeflow.py", "test_e4_submit")
e4_setup = load_experiment_module("E4_generalization_gradient", "e4_setup.py", "test_e4_batches_setup")
CFG = e4_setup.load_config()


def test_two_waves_of_the_four_diagonal_1_configurations_balanced_in_orientation():
    seen = [cid for wave in batches.WAVES.values() for cid in wave["configurations"]]
    assert sorted(seen) == ["p1-a-d1", "p1-b-d1", "p2-a-d1", "p2-b-d1"]
    for name, wave in batches.WAVES.items():
        confs = [CFG["e4"]["configurations"][cid] for cid in wave["configurations"]]
        assert sorted(c["inc"] for c in confs) == ["a", "b"], name
        assert {c["diagonal"] for c in confs} == {1} and len({c["pair"] for c in confs}) == 1, name
        assert len(batches.batch_runs(name)) == 4
    assert batches.WAVES["e4_wave1"]["configurations"] == ("p1-a-d1", "p1-b-d1") and batches.diagnostics_at("e4_wave1") is None
    replicas = [cid for r in batches.REPLICAS.values() for cid in r["configurations"]]
    assert {CFG["e4"]["configurations"][cid]["diagonal"] for cid in replicas} == {2}
    seeds = [c["seed"] for c in CFG["e4"]["configurations"].values()]
    assert len(set(seeds)) == len(seeds)


def test_wave1_commands_need_no_secret_and_train_four_runs():
    jobs = submit.commands("train", "PY", "e4_wave1")
    assert [name for name, _ in jobs] == ["p1-a-d1_experimental", "p1-b-d1_experimental", "p1-a-d1_reference", "p1-b-d1_reference"]
    for _, cmd in jobs:
        assert cmd[:3] == ["PY", "-u", "experiments/E4_generalization_gradient/run_e4.py"]
        assert not any(a.startswith("--push") for a in cmd)
        assert "--max-env-steps" not in cmd and "--diagnostics-at" not in cmd
        assert cmd[cmd.index("--config") + 1] == "configs/e4/e4.yaml" and cmd[cmd.index("--device") + 1] == "cuda"
    pushed = submit.commands("train", "PY", "e4_wave1", {"repo": "https://x/r.git", "token_file": "/etc/t/TOKEN"})
    assert all(cmd[cmd.index("--push-batch") + 1] == "e4_wave1" for _, cmd in pushed)


def test_probes_commands():
    jobs = submit.commands("probes", "PY", "e4_wave1", checkpoint=175000)
    assert [n for n, _ in jobs] == ["probes_p1-a-d1_env175000", "probes_p1-b-d1_env175000"]
    for _, cmd in jobs:
        assert cmd[2] == "experiments/E4_generalization_gradient/measure_probes.py"
        assert cmd[cmd.index("--checkpoint") + 1] == "175000" and cmd[cmd.index("--find-runs") + 1] == "results"
        assert "--save-steps" in cmd and cmd[cmd.index("--device") + 1] == "cuda"
    assert len(submit.commands("probes", "PY", "e4_wave1", checkpoint=1, configurations=["p1-b-d1"])) == 1
    with pytest.raises(ValueError):
        submit.commands("probes", "PY", "e4_wave1")
    with pytest.raises(ValueError):
        submit.commands("probes", "PY", "e4_wave1", checkpoint=1, configurations=["p2-a-d1"])


def test_later_batches_wait_for_b_and_take_fewer_diagnostics(monkeypatch):
    with pytest.raises(ValueError):
        submit.commands("train", "PY", "e4_wave2")
    monkeypatch.setattr(submit.batches, "B_AFTER_PILOT", 175000)
    cmd = submit.commands("train", "PY", "e4_wave2")[0][1]
    assert cmd[cmd.index("--max-env-steps") + 1] == "175000"
    at = cmd[cmd.index("--diagnostics-at") + 1:cmd.index("--diagnostics-at") + 1 + len(batches.CONTROL_CHECKPOINTS)]
    assert at == [str(s) for s in batches.CONTROL_CHECKPOINTS]


def test_runtime_patch_without_and_with_secret():
    empty = runtime_patch_secrets("vol", "/scratch", {})
    spec = empty["template"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    assert [v["name"] for v in spec["volumes"]] == ["scratch"]
    patch = runtime_patch_secrets("vol", "/scratch", {"push-secret": "/etc/e4-results-push"})
    spec = patch["template"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    assert {"name": "secret-0", "mountPath": "/etc/e4-results-push", "readOnly": True} in spec["containers"][0]["volumeMounts"]


def test_dry_runs(capsys):
    import json

    assert submit.main(["--batch", "e4_wave1", "--repo-dir", "/scratch/x", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["push"] is False and len(out["jobs"]) == 4 and len(out["runtime_patch"]["template"]["spec"]["replicatedJobs"][0]
                                                                   ["template"]["spec"]["template"]["spec"]["volumes"]) == 1
    assert submit.main(["--batch", "e4_wave1", "--phase", "probes", "--checkpoint", "175000", "--repo-dir", "/scratch/x", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["phase"] == "probes" and out["max_parallel"] == batches.PROBES_MAX_PARALLEL and len(out["jobs"]) == 2
