"""E8 end to end on E5's mini configuration (CPU; the brief's section 7): start files from a short mini training, an
agents file naming them with their SHA-256, then run_e8.py --phase train for a baseline pair and a shuffle pair (therapy
on nw), --phase probes at their checkpoints (--n-reps 1, --n-walks 3) and analyze_e8.py on the outputs. Its numbers mean
nothing; it checks that every stage runs and that the outputs hold what the analysis reads."""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import ClosedLoopHooks, run_one
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import make_hooks as e4_hooks
from helplessness.probing.theme_probe_measure import sha256

run_e8 = load_experiment_module("E8_therapy_relapse", "run_e8.py", "test_e8_e2e_run")
analyze_e8 = load_experiment_module("E8_therapy_relapse", "analyze_e8.py", "test_e8_e2e_analyze")
e8_setup = run_e8.e8_setup


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("e8e2e")
    runs_root = root / "runs_root"
    cfg = e8_setup.load_config(REPO_ROOT / "configs" / "e8" / "e8_mini.yaml")
    two_zones_setup.apply_e5_procedure(cfg)
    cfg["e4"]["run_configuration"] = "p1-b-d1"
    cfg["checkpoints"]["grid"] = [100]
    models = {}
    for mid, seed, kind, role in (("base_X_exp", 2, "baseline", "experimental"), ("base_X_ref", 2, "baseline", "reference"),
                                  ("shuffle_X_exp", 402, "shuffle", "experimental"), ("shuffle_Y_ref", 102, "shuffle", "reference")):
        conf = {**two_zones_setup.configuration(cfg, "p1-b-d1"), "seed": seed}
        base = e4_hooks(cfg, conf)
        hooks = ClosedLoopHooks(episode=base.episode, build_world=base.build_world, checkpoint_record=lambda s: {}, finalize=lambda r, w, d: {})
        run = f"start/{mid}"
        run_one(cfg, seed, role, runs_root / run, configure_runtime(cfg), hooks=hooks, max_env_steps=100)
        models[mid] = {"kind": kind, "role": role, "configuration": "p1-b-d1", "seed": seed, "run": run, "checkpoint": 100,
                       "sha256": sha256(runs_root / run / "checkpoints" / "agent_env000100.pt")}
    agents = root / "agents.yaml"
    agents.write_text(yaml.safe_dump({"models": models, "pairs": {"base_X": {"exposed": "base_X_exp", "reference": "base_X_ref"},
                                                                  "shuffle_X": {"exposed": "shuffle_X_exp", "reference": "shuffle_Y_ref"}}}),
                      encoding="utf-8")
    raw = yaml.safe_load((REPO_ROOT / "configs" / "e8" / "e8_mini.yaml").read_text(encoding="utf-8"))
    raw["e8"].update(agents=str(agents), max_env_steps=100, checkpoints=[0, 50, 100])
    config = root / "e8_e2e.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = root / "out"
    for mid, q in (("base_X_exp", None), ("base_X_ref", None), ("shuffle_X_exp", "nw"), ("shuffle_Y_ref", "nw")):
        args = ["--phase", "train", "--model", mid, "--config", str(config), "--runs-root", str(runs_root), "--device", "cpu", "--out-dir", str(out)]
        assert run_e8.main(args + ([] if q is None else ["--treated-quadrant", q])) == 0
    runs = {yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["e8"]["run"]["model"]: d
            for d in (out / e8_setup.EXPERIMENT_ID).glob("grid12x12__*")}
    for pairs, models_ in ((["base_X"], ("base_X_exp", "base_X_ref")), (["shuffle_X"], ("shuffle_X_exp", "shuffle_Y_ref"))):
        for k in (0, 100):
            args = ["--phase", "probes", "--config", str(config), "--pairs", *pairs, *[a for m in models_ for a in ("--run", f"{m}={runs[m]}")],
                    "--checkpoint", str(k), "--device", "cpu", "--out-dir", str(out), "--n-reps", "1", "--n-walks", "3"]
            if pairs == ["shuffle_X"]:
                args += ["--images", "all"]
            assert run_e8.main(args) == 0
    return root, out, runs


def test_train_outputs(campaign):
    _, out, runs = campaign
    for mid, d in runs.items():
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        assert meta["status"] == "done" and meta["fresh_start"]["origin_env_step"] == 100
        recs = [json.loads(x) for x in (d / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["env_step"] for r in recs] == [0, 50, 100]
        assert all((d / "checkpoints" / f"agent_env{k:06d}.pt").is_file() and (d / "checkpoints" / f"resumable_env{k:06d}.pt").is_file()
                   for k in (0, 50, 100))
        assert (d / "steps.npz").is_file()


def test_probe_rows(campaign):
    _, out, _ = campaign
    jobs = sorted((out / e8_setup.EXPERIMENT_ID).glob("probes__*"))
    assert len(jobs) == 4 and all(json.loads((j / "status.json").read_text())["status"] == "done" for j in jobs)
    rows = [r for j in jobs for r in json.loads((j / "rows.json").read_text(encoding="utf-8"))]
    base = [r for r in rows if r["pair"] == "base_X"]
    assert {r["image"] for r in base} == {"training"} and {r["role"] for r in base} == {"exposed", "reference"}
    shuf = [r for r in rows if r["pair"] == "shuffle_X" and r["checkpoint"] == 0]
    assert len({r["assignment"] for r in shuf}) == 12
    assert {r["assignment"] for r in shuf if r["treated"]} == {0, 1, 2}
    for r in rows:
        for key in ("b_walks", "walks_valid", "b_step1", "directed_share", "C1_loc", "C3_loc", "loc_loc_cell"):
            assert key in r, key
    oracle = [r for r in rows if r.get("oracle_directed_share") is not None]
    assert {(r["pair"], r["checkpoint"]) for r in oracle} == {("base_X", 0), ("base_X", 100), ("shuffle_X", 0), ("shuffle_X", 100)}


def test_analysis_runs(campaign):
    root, out, _ = campaign
    res = root / "analysis"
    assert analyze_e8.main(["--probes", str(out / e8_setup.EXPERIMENT_ID / "probes__*"), "--runs", str(out / e8_setup.EXPERIMENT_ID / "grid12x12__*"),
                            "--out", str(res)]) == 0
    forms = json.loads((res / "forms.json").read_text(encoding="utf-8"))
    assert set(forms) >= {"R", "M", "P1", "P2", "P3", "P4", "P4_overall"}
    assert (res / "curves.csv").is_file() and (res / "recovery.csv").is_file() and (res / "figures" / "recovery_base_X.png").is_file()
    shutil.rmtree(res)


def test_campaign_config_has_no_test_only_setting():
    """The real runs keep E6's 150 validity walks: no test-only key in the campaign's config (only e8_mini.yaml has one)."""
    def keys(d, prefix=""):
        return [f"{prefix}{k}" for k, v in d.items()] + [x for k, v in d.items() if isinstance(v, dict) for x in keys(v, f"{prefix}{k}.")]
    cfg = e8_setup.load_config(REPO_ROOT / "configs" / "e8" / "e8.yaml")
    assert not [k for k in keys(cfg) if "test_only" in k]
    mini = e8_setup.load_config(REPO_ROOT / "configs" / "e8" / "e8_mini.yaml")
    assert mini["e8"]["probes"]["test_only_validity_n_walks"] == 10
