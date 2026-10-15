"""E4's batch table and the commands of its submission script (no cluster involved)."""

from __future__ import annotations

import pytest

from _experiment_loader import load_experiment_module
from helplessness.cluster.kubeflow import runtime_patch_secrets

batches = load_experiment_module("E4_generalization_gradient", "batches.py", "e4_batches_under_test")
submit = load_experiment_module("E4_generalization_gradient", "submit_kubeflow.py", "test_e4_submit")
e4_setup = load_experiment_module("E4_generalization_gradient", "e4_setup.py", "test_e4_batches_setup")


def test_waves_cover_the_eight_configurations_once_and_are_balanced():
    cfg = e4_setup.load_config()
    seen = [cid for wave in batches.WAVES.values() for cid in wave["configurations"]]
    assert sorted(seen) == sorted(cfg["e4"]["configurations"])
    for name, wave in batches.WAVES.items():
        confs = [cfg["e4"]["configurations"][cid] for cid in wave["configurations"]]
        assert sorted(c["inc"] for c in confs) == ["a", "b"], name                      # both orientations
        assert len({(c["pair"], c["diagonal"]) for c in confs}) == 1, name              # one pair, one diagonal
        assert len(batches.batch_runs(name)) == 4
    assert {c["pair"] for c in (cfg["e4"]["configurations"][cid] for cid in batches.WAVES["e4_wave1"]["configurations"])} == {"pair1"}
    seeds = [c["seed"] for c in cfg["e4"]["configurations"].values()]
    assert len(set(seeds)) == len(seeds)


def test_wave1_commands():
    jobs = submit.commands("train", "PY", "e4_wave1", "https://example/repo.git", "/etc/x/RESULTS_PUSH_TOKEN")
    assert [name for name, _ in jobs] == ["p1-a-d1_experimental", "p1-b-d1_experimental", "p1-a-d1_reference", "p1-b-d1_reference"]
    for name, cmd in jobs:
        assert cmd[:3] == ["PY", "-u", "experiments/E4_generalization_gradient/run_e4.py"]
        assert "--max-env-steps" not in cmd and cmd[cmd.index("--config") + 1] == "configs/e4/e4.yaml"
        assert cmd[cmd.index("--push-batch") + 1] == "e4_wave1" and cmd[cmd.index("--device") + 1] == "cuda"
    resume = submit.commands("resume", "PY", "e4_wave1", "u", "t", resume="results/E4_generalization_gradient/r")
    assert resume[0][1][resume[0][1].index("--resume") + 1] == "results/E4_generalization_gradient/r"


def test_later_waves_wait_for_b():
    with pytest.raises(ValueError):
        submit.commands("train", "PY", "e4_wave2", "u", "t")


def test_runtime_patch_mounts_every_secret_read_only():
    patch = runtime_patch_secrets("vol", "/scratch", {"push-secret": "/etc/e4-results-push"})
    spec = patch["template"]["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
    assert {"name": "secret-0", "secret": {"secretName": "push-secret"}} in spec["volumes"]
    assert {"name": "secret-0", "mountPath": "/etc/e4-results-push", "readOnly": True} in spec["containers"][0]["volumeMounts"]
