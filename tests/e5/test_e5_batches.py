"""E5's batch table and the commands of its submission script (no cluster involved): 12 runs in three waves of four,
the batch parameters (configuration, switch step, post steps, E4's results), the branch of immunisation B only, the
resumable states of the baseline reference of seed B, the probe pairs; the config holds E4's parameters unchanged."""

from __future__ import annotations

import pytest

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline import two_zones_setup

batches = load_experiment_module("E5_belief_training", "batches.py", "e5_batches_under_test")
submit = load_experiment_module("E5_belief_training", "submit_kubeflow.py", "test_e5_submit")
e5_setup = load_experiment_module("E5_belief_training", "e5_setup.py", "test_e5_batches_setup")


def test_waves_after_the_failed_first_wave():
    """docs/e5/DECISIONS.md (29): wave 1 = shuffle f = 1, seeds A and B, both conditions; wave 2 = seed B's baseline; f = 0.5
    and the immunisation (E8) deferred, never in a wave."""
    assert batches.WAVES["e5_wave1"] == {"A_sh100_experimental": ("A", "shuffle", 1.0, "experimental"),
                                         "A_sh100_reference": ("A", "shuffle", 1.0, "reference"),
                                         "B_sh100_experimental": ("B", "shuffle", 1.0, "experimental"),
                                         "B_sh100_reference": ("B", "shuffle", 1.0, "reference")}
    assert batches.WAVES["e5_wave2"] == {"B_base_experimental": ("B", "baseline", None, "experimental"),
                                         "B_base_reference": ("B", "baseline", None, "reference")}
    in_waves = {spec for wave in batches.WAVES.values() for spec in wave.values()}
    assert not any(spec[1] == "immunisation" or spec[2] == 0.5 for spec in in_waves)
    assert {spec[1] for d in batches.DEFERRED.values() for spec in d.values()} == {"shuffle", "immunisation"}
    for name in batches.DEFERRED:
        with pytest.raises(ValueError):
            batches.batch_runs(name)


def test_train_commands():
    jobs = submit.commands("train", "PY", "e5_wave1", "p1-b-d1")
    assert [n for n, _ in jobs] == list(batches.WAVES["e5_wave1"])
    for _, cmd in jobs:
        assert cmd[:3] == ["PY", "-u", "experiments/E5_belief_training/run_e5.py"]
        assert cmd[cmd.index("--configuration") + 1] == "p1-b-d1" and cmd[cmd.index("--config") + 1] == "configs/e5/e5.yaml"
        assert cmd[cmd.index("--training") + 1] == "shuffle" and cmd[cmd.index("--fraction") + 1] == "1.0"
        assert "--max-env-steps" not in cmd and cmd[cmd.index("--device") + 1] == "cuda"
    w2 = dict(submit.commands("train", "PY", "e5_wave2", "p1-b-d1"))
    ref = w2["B_base_reference"]
    assert ref[ref.index("--save-resumable-at") + 1:ref.index("--device")] == ["100000"]
    assert "--save-resumable-at" not in w2["B_base_experimental"]
    with pytest.raises(ValueError):
        submit.commands("train", "PY", "e5_wave1", "")  # the configuration is required


def test_probe_commands():
    jobs = dict(submit.commands("probes", "PY", "e5_wave1", "p1-b-d1", checkpoint=100000))
    assert sorted(jobs) == ["probes_A_sh100_k100000", "probes_B_sh100_k100000"]
    assert all("--e4-results" not in cmd for cmd in jobs.values())
    jobs = dict(submit.commands("probes", "PY", "e5_wave2", "p1-b-d1", checkpoint=100000))
    assert sorted(jobs) == ["probes_B_base_k100000"]


def test_config_is_e4_unchanged_plus_e5():
    cfg = e5_setup.load_config(REPO_ROOT / "configs" / "e5" / "e5.yaml")
    e4 = two_zones_setup.load_config(REPO_ROOT / "configs" / "e4" / "e4.yaml")
    assert {k: v for k, v in cfg.items() if k != "e5"} == e4
    assert e5_setup.configuration(cfg, "p1-a-d1", "A")["seed"] == e4["e4"]["configurations"]["p1-a-d1"]["seed"]
    assert e5_setup.configuration(cfg, "p1-a-d1", "B")["seed"] == e4["e4"]["configurations"]["p1-a-d1"]["seed"] + 100
    assert cfg["e5"]["immunisation"]["post_steps"] == 175000 and 175000 in e4["checkpoints"]["grid"]


def test_e5_config_refuses_other_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("base: configs/e4/e4.yaml\ne5: {}\ntrain: {lr: 1.0}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        e5_setup.load_config(p)


@pytest.mark.parametrize("spec", [
    {"training": "baseline", "condition": "reference"},
    {"training": "shuffle", "condition": "experimental", "fraction": 0.5},
    {"training": "immunisation", "switch_step": 1000, "post_steps": 175000},
])
def test_every_e5_run_uses_the_same_background_statistic(spec):
    """docs/e5/DECISIONS.md (19): one procedure for every E5 run."""
    run_e5 = load_experiment_module("E5_belief_training", "run_e5.py", "test_e5_batches_run")
    cfg = e5_setup.load_config(REPO_ROOT / "configs" / "e5" / "e5.yaml")
    assert cfg["e5"]["switch_pairs"] == "same_background"
    assert "switch_pairs" not in cfg["train"]["latent_estimator"]  # E4's block itself is untouched
    assert cfg["train"]["latent_estimator"]["switch_max_steps"] == 25000
    cfg["e5"]["run"] = {"configuration": "p1-a-d1", "seed_label": "B", **spec}
    run_e5.build(cfg)
    assert cfg["train"]["latent_estimator"]["switch_pairs"] == "same_background"
    assert cfg["e5"]["switch_max_steps"] == 75000 and cfg["train"]["latent_estimator"]["switch_max_steps"] == 75000
    est = cfg["train"]["latent_estimator"]  # the late threshold (docs/e5/DECISIONS.md (28))
    assert (est["switch_late_threshold"], est["switch_late_after_steps"]) == (1.5, 6100)


def test_configuration_is_p1_b_d1_and_baseline_a_is_reprobed_from_e4(capsys):
    """docs/e5/DECISIONS.md (21) and (25): the submission takes p1-b-d1 only; e5_probes_baseline_A trains nothing and
    probes E4's own pair of the configuration (--e4-results), with E5's probe script."""
    assert batches.CONFIGURATION == "p1-b-d1"
    with pytest.raises(SystemExit):
        submit.main(["--batch", "e5_wave1", "--configuration", "p1-a-d1", "--repo-dir", "/scratch/x", "--dry-run"])
    with pytest.raises(SystemExit):
        submit.main(["--batch", "e5_probes_baseline_A", "--repo-dir", "/scratch/x", "--dry-run"])  # no training phase
    capsys.readouterr()
    assert submit.main(["--batch", "e5_wave1", "--repo-dir", "/scratch/x", "--dry-run"]) == 0  # the default configuration
    assert '"configuration": "p1-b-d1"' in capsys.readouterr().out
    jobs = submit.commands("probes", "PY", "e5_probes_baseline_A", "p1-b-d1", checkpoint=300000, e4_results="/scratch/e4/results")
    assert [n for n, _ in jobs] == ["probes_A_base_k300000"]
    cmd = jobs[0][1]
    assert cmd[cmd.index("--training") + 1] == "baseline" and cmd[cmd.index("--seed-label") + 1] == "A"
    assert cmd[cmd.index("--e4-results") + 1] == "/scratch/e4/results" and "--n-walks" not in cmd  # e5.probes.n_walks (150)
    with pytest.raises(ValueError):
        submit.commands("probes", "PY", "e5_probes_baseline_A", "p1-b-d1", checkpoint=300000)
    cfg = e5_setup.load_config()
    assert cfg["e5"]["probes"]["n_walks"] == 150 and cfg["e4"]["checkpoint"]["n_walks"] == 100  # training-time walks: E4's
    assert e5_setup.configuration(cfg, "p1-b-d1", "A")["seed"] == 2 and e5_setup.configuration(cfg, "p1-b-d1", "B")["seed"] == 102
