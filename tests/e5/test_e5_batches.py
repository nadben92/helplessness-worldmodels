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
                                         "B_base_reference": ("B", "baseline", None, "reference"),
                                         "E_sh100_experimental": ("E", "shuffle", 1.0, "experimental"),
                                         "F_sh100_experimental": ("F", "shuffle", 1.0, "experimental")}
    assert "e5_wave3" not in batches.WAVES  # two waves only (docs/e5/DECISIONS.md (32))
    # the replacements of seed B's locked-in experimental run are measured against seed B's f = 1 reference
    assert batches.REFERENCE_SEED_OF == {("E", "shuffle"): "B", ("F", "shuffle"): "B"}
    assert batches.PROBES_CHECKPOINT == 300000
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
    assert sorted(w2) == ["B_base_experimental", "B_base_reference", "E_sh100_experimental", "F_sh100_experimental"]
    ref = w2["B_base_reference"]
    assert ref[ref.index("--save-resumable-at") + 1:ref.index("--device")] == ["100000"]
    for job, label, training in (("B_base_experimental", "B", "baseline"), ("E_sh100_experimental", "E", "shuffle"),
                                 ("F_sh100_experimental", "F", "shuffle")):
        cmd = w2[job]
        assert cmd[cmd.index("--seed-label") + 1] == label and cmd[cmd.index("--training") + 1] == training
        assert "--save-resumable-at" not in cmd and cmd[cmd.index("--condition") + 1] == "experimental"
    with pytest.raises(ValueError):
        submit.commands("train", "PY", "e5_wave1", "")  # the configuration is required


def test_probe_commands():
    jobs = dict(submit.commands("probes", "PY", "e5_wave1", "p1-b-d1", checkpoint=100000))
    assert sorted(jobs) == ["probes_A_sh100_k100000", "probes_B_sh100_k100000"]
    assert all("--e4-results" not in cmd for cmd in jobs.values())
    jobs = dict(submit.commands("probes", "PY", "e5_wave2", "p1-b-d1", checkpoint=100000))
    assert sorted(jobs) == ["probes_B_base_k100000", "probes_E_sh100_k100000", "probes_F_sh100_k100000"]
    for name in ("probes_E_sh100_k100000", "probes_F_sh100_k100000"):
        cmd = jobs[name]
        assert cmd[cmd.index("--reference-seed-label") + 1] == "B"
    assert "--reference-seed-label" not in jobs["probes_B_base_k100000"]


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


def test_replay_of_the_corrected_criterion_on_the_failed_wave_1():
    """docs/e5/DECISIONS.md (29), (30): non-regression on the real series of the first wave 1's f = 1 runs (seed 2; the
    same-background statistic at every evaluation, tests/e5/data). E4's main rule alone never switches (the runs failed
    at the deadline); the corrected criterion of configs/e5/e5.yaml switches the reference at 6,301 and the experimental
    run at 7,151, both by the late threshold, and the experimental run's collapse (an abrupt fall to 0 at 13,250, the
    single-step collapse E3 met at lr 1e-3) comes 6,099 gradient steps after its switch."""
    import json
    from pathlib import Path

    replay = load_experiment_module("E5_belief_training", "replay_switch.py", "test_e5_replay_wave1")
    data = json.loads((Path(__file__).parent / "data" / "e5_wave1_f1_switch_series.json").read_text(encoding="utf-8"))
    cfg = e5_setup.load_config(REPO_ROOT / "configs" / "e5" / "e5.yaml")
    late = cfg["e5"]["late_switch"]
    rule = {"threshold_bits": 3.0, "consecutive": 5, "eval_every": 50, "min_steps": 500, "collapse_threshold_bits": 1.0,
            "max_steps": cfg["e5"]["switch_max_steps"]}
    expected = {"reference": 6301, "experimental": 7151}
    for role, run in data["runs"].items():
        evals = [(int(s), float(v)) for s, v in run["evals"]]
        assert len(evals) == 1497 and replay.gaps(evals) == []
        assert replay.replay(evals, rule, None, None) == {"switch_step": None, "rule": None}
        assert replay.replay(evals, rule, late["threshold_bits"], late["after_steps"]) == {"switch_step": expected[role], "rule": "late_threshold"}
    exp = [(int(s), float(v)) for s, v in data["runs"]["experimental"]["evals"]]
    assert replay.collapse_onset(exp, 1.0) == 13250 and 13250 - expected["experimental"] == 6099
    assert replay.collapse_onset([(int(s), float(v)) for s, v in data["runs"]["reference"]["evals"]], 1.0) is None



def test_seeds_c_and_d_and_no_stream_collision():
    """docs/e5/DECISIONS.md (31), (32): seeds A-F of p1-b-d1 are 2, 102, 202, 302, 402, 502 (A + 100 per label, B's rule), the same in
    e5_setup and in the export's text reading; every stream root of every E4 and E5 run is distinct -- the experimental
    root SeedSequence(seed), the reference's [seed, 1], the assignment stream's [seed, 2] -- except seed A, which IS E4's
    run of p1-b-d1 on purpose (its baseline is E4's pair)."""
    import numpy as np

    cfg = e5_setup.load_config(REPO_ROOT / "configs" / "e5" / "e5.yaml")
    e5_seeds = {label: e5_setup.configuration(cfg, "p1-b-d1", label)["seed"] for label in e5_setup.SEED_LABELS}
    assert e5_seeds == {"A": 2, "B": 102, "C": 202, "D": 302, "E": 402, "F": 502}
    assert submit.seeds(REPO_ROOT, "p1-b-d1") == e5_seeds
    e4_seeds = {c["seed"] for c in cfg["e4"]["configurations"].values()}
    seeds = sorted(e4_seeds | set(e5_seeds.values()))
    roots = [np.random.SeedSequence(s) for s in seeds] + [np.random.SeedSequence([s, k]) for s in seeds for k in (1, 2)]
    states = [tuple(r.generate_state(8)) for r in roots]
    assert len(set(states)) == len(states)
    assert e5_seeds["A"] in e4_seeds and not (set(e5_seeds.values()) - {e5_seeds["A"]}) & e4_seeds


def test_continuation_batch_and_probe_checkpoints():
    """docs/e5/DECISIONS.md (33): e5_extend continues the five healthy shuffle runs (A exp/ref, B ref, E, F) to EXTEND_TO (650,000, a ceiling)
    from their resume point; e5_probes_all probes the baselines at 300,000 (converged) and the shuffle pairs at --checkpoint."""
    jobs = dict(submit.commands("train", "PY", "e5_extend", "p1-b-d1"))
    assert sorted(jobs) == ["A_sh100_experimental", "A_sh100_reference", "B_sh100_reference", "E_sh100_experimental", "F_sh100_experimental"]
    for name, cmd in jobs.items():
        assert cmd[cmd.index("--extend") + 1] == "auto" and cmd[cmd.index("--to") + 1] == str(batches.EXTEND_TO) == "650000"
        assert cmd[cmd.index("--training") + 1] == "shuffle" and cmd[cmd.index("--fraction") + 1] == "1.0"
    assert batches.BATCH_CPUS["e5_extend"] == 3  # the quota: one pod of 3 CPUs
    probes = dict(submit.commands("probes", "PY", "e5_probes_all", "p1-b-d1", checkpoint=500000, e4_results="/e4"))
    assert sorted(probes) == ["probes_A_base_k300000", "probes_A_sh100_k500000", "probes_B_base_k300000", "probes_B_sh100_k500000",
                              "probes_E_sh100_k500000", "probes_F_sh100_k500000"]



def test_continuation_in_two_batches_and_two_probe_sets():
    """docs/e5/DECISIONS.md (33): e5_extend_a = seed A's f = 1 pair, e5_extend_b = seed B's reference with E and F, each
    3 CPUs to 650,000 (e5_extend kept, unused); probes in two sets -- e5_probes_a (baselines A and B at 300,000, seed A's
    pair at --checkpoint K_a), e5_probes_b (E and F at K_b)."""
    a = dict(submit.commands("train", "PY", "e5_extend_a", "p1-b-d1"))
    b = dict(submit.commands("train", "PY", "e5_extend_b", "p1-b-d1"))
    assert sorted(a) == ["A_sh100_experimental", "A_sh100_reference"]
    assert sorted(b) == ["B_sh100_reference", "E_sh100_experimental", "F_sh100_experimental"]
    for cmd in [*a.values(), *b.values()]:
        assert cmd[cmd.index("--extend") + 1] == "auto" and cmd[cmd.index("--to") + 1] == "650000"
    assert batches.BATCH_CPUS["e5_extend_a"] == batches.BATCH_CPUS["e5_extend_b"] == 3
    pa = dict(submit.commands("probes", "PY", "e5_probes_a", "p1-b-d1", checkpoint=550000, e4_results="/e4"))
    assert sorted(pa) == ["probes_A_base_k300000", "probes_A_sh100_k550000", "probes_B_base_k300000"]
    pb = dict(submit.commands("probes", "PY", "e5_probes_b", "p1-b-d1", checkpoint=600000))
    assert sorted(pb) == ["probes_E_sh100_k600000", "probes_F_sh100_k600000"]
    assert all(cmd[cmd.index("--reference-seed-label") + 1] == "B" for cmd in pb.values())
