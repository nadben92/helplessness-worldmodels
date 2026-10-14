"""The campaign's analysis pipeline (docs/e3/DECISIONS.md, 2026-10-13 (31)): integrity of a handoff, the analyses of B to
E on synthetic inputs, and the one-command report (RSA included) on a TINY environment / reference pair."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from _experiment_loader import load_experiment_module

ci = load_experiment_module("E3_worldmodel_4zones", "campaign_integrity.py", "test_report_integrity")
ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "test_report_analysis")
cl = load_experiment_module("E3_worldmodel_4zones", "campaign_learning.py", "test_report_learning")
cr = load_experiment_module("E3_worldmodel_4zones", "campaign_report.py", "test_report_report")
mh = load_experiment_module("E3_worldmodel_4zones", "make_handoff.py", "test_report_make_handoff")
run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "test_report_run_agent")
rsa_latents = load_experiment_module("E3_worldmodel_4zones", "rsa_latents.py", "test_report_rsa")
run_mini = run_agent.run_mini
REPO = Path(__file__).resolve().parents[2]


# --- synthetic -----------------------------------------------------------------------------------------------------------
def test_wilson_interval():
    p, lo, hi = cl.wilson(50, 100)
    assert p == 0.5 and lo < 0.5 < hi and np.isclose(hi - 0.5, 0.5 - lo)
    assert np.isnan(cl.wilson(0, 0)[0])


def test_noise_sigma_and_corridor_scaling():
    sig = cl.zone_sigma()
    assert np.isclose(sig["controllable_deterministic"], np.sqrt(np.mean(np.square(cl.NOISE_X4_PILOT["controllable_deterministic"]))))
    assert np.isclose(sig["corridor"], sig["controllable_deterministic"] * np.sqrt(25 / 43))


def test_seeds_ci_and_verdicts():
    s = ca.seeds_ci([0.1, 0.12, 0.09, 0.11])
    assert s["excludes_0"] and s["n"] == 4 and s["bound"] >= s["ci95"][1]
    assert not ca.seeds_ci([0.1, -0.1, 0.05, -0.05])["excludes_0"]
    neg, zero = ca.seeds_ci([-0.6, -0.5, -0.7, -0.55]), ca.seeds_ci([0.1, -0.1, 0.05, -0.05])
    assert ca.distance_shape_verdict(neg, neg, zero) == "propagation shape present"
    assert ca.distance_shape_verdict(zero, neg, zero).startswith("absent")
    assert ca.distance_shape_verdict(neg, neg, neg) == "undetermined"
    assert ca.generalisation_verdict([-0.1, -0.2, -0.05, -0.3]).startswith("generalises")
    assert ca.generalisation_verdict([0.1, -0.2, 0.0, -0.3]) == "mixed"
    assert ca.generalisation_verdict([0.1, float("nan")]).startswith("not testable")


def _fake_run(zone, I_by_c, n_by_c, seed=0, condition="experimental"):
    """A Run-like object for equal_visits_pair: counts N_c with min_a N[s, a] = n, measures with real_I_s."""
    S = len(zone)
    counts, measures = {}, {}
    for c, n in n_by_c.items():
        N = np.zeros((S, 5, S))
        N[np.arange(S), :, 0] = np.asarray(n)[:, None]
        counts[f"N_{c}"] = N
        measures[c] = ({}, {"real_I_s": np.asarray(I_by_c[c], dtype=float)})
    return SimpleNamespace(zone_of_state=np.array(zone), counts=counts, measure_dirs={c: None for c in n_by_c},
                           measure=lambda c: measures[c], seed=seed, condition=condition)


def test_equal_visits_separates_visits_from_training_time(monkeypatch):
    """I_s = f(n) in both runs, the reference reaching each n later: d = 0 at equal n whatever the delay, so b0 = 0;
    then a constant offset of the environment shows as b0."""
    monkeypatch.setattr(ca, "goal_state", lambda: 3)
    zone = ["controllable_deterministic"] * 3 + ["corridor"]  # state 3: the goal, left out
    cs = [10_000, 30_000, 100_000, 300_000]
    n_env = {c: [c // 100, c // 120, c // 90, 0] for c in cs}
    n_ref = {c: [c // 300, c // 400, c // 250, 0] for c in cs}  # fewer visits at each checkpoint
    f = lambda n: 2.0 - 1.0 / np.log10(np.asarray(n, dtype=float) + 10)
    exp = _fake_run(zone, {c: f(n_env[c]) for c in cs}, n_env)
    ref = _fake_run(zone, {c: f(n_ref[c]) for c in cs}, n_ref, condition="reference")
    out = ca.equal_visits_pair(exp, ref)["controllable_deterministic"]
    assert out["n_states"] == 3 and abs(out["b0"]) < 0.02 and abs(out["mean_d"]) < 0.02
    exp2 = _fake_run(zone, {c: f(n_env[c]) + 0.1 for c in cs}, n_env)
    assert np.isclose(ca.equal_visits_pair(exp2, ref)["controllable_deterministic"]["b0"], out["b0"] + 0.1, atol=1e-6)


def test_distance_to_uncontrollable_on_the_real_layout():
    world = run_mini.build_world(run_mini.load_mini_config(REPO / "configs" / "e3" / "gpu_campaign.yaml"), 1, "experimental")
    P = world.env.transition_tensor()
    zone = np.array([run_mini.zone_of_state(world.layout, s) for s in range(144)])
    run = SimpleNamespace(truth=ca.state_truth(P), zone_of_state=zone)
    d = ca.distance_to_uncontrollable(run)
    unc = np.isin(zone, ["uncontrollable_deterministic", "uncontrollable_noisy"])
    assert (d[unc] == 0).all() and (d[~unc] > 0).all()
    assert d[zone == "controllable_deterministic"].min() >= 2  # the corridor lies between the quadrants


def _record(c, succ, eps, mass, delta):
    return {"env_step": c, "successes": succ, "episodes": eps, "goal_diagnostics": {"median_mass_on_goal_pairs": mass},
            "agent_delta": {"real_contexts": {"delta_by_zone": {"a": delta}}}}


def test_convergence_rule_and_bonferroni():
    zones = np.array(["uncontrollable_deterministic"] * 2 + ["controllable_deterministic"] * 2 + ["corridor"] * 2 + ["corridor"])

    def make(seed, cond, jitter):
        cs = [25_000, 63_096, 100_000, 300_000]
        I = {c: np.array([0.0, 0.0, 2.3, 2.3, 2.3, 2.3, 0.0]) + jitter(c) for c in cs}
        recs, k, e = {}, 0, 0
        for c in cs:
            k, e = k + 900, e + 1000
            recs[c] = _record(c, k, e, 1.0 if c > 25_000 else 0.5, 0.2)
        return SimpleNamespace(seed=seed, condition=cond, zone_of_state=zones, records=recs, measure_dirs={c: None for c in cs},
                               measure=lambda c, I=I: ({}, {"real_I_s": I[c]}))

    goal = cl.ca.goal_state
    try:
        cl.ca.goal_state = lambda: 6
        stable = [make(s, c, lambda c: 0.0) for s in (1, 6) for c in ("experimental", "reference")]
        res = cl.convergence(stable)
        assert res["candidates"] == [25_000, 63_096, 100_000]
        assert res["first_checkpoint"] == 63_096  # (a) fails at 25,000 (goal mass 0.5)
        row = res["table"][1]
        assert row["comparisons_per_run_zone"] == 3 and row["m"] == 4 * 3 * 3  # 63,096 -> 100,000 -> 300,000, and 63,096 -> 300,000
        assert row["k"] > 2.5
        drift = [make(s, c, lambda c: 0.05 * np.log10(c / 25_000)) for s in (1, 6) for c in ("experimental", "reference")]
        assert cl.convergence(drift)["first_checkpoint"] is None and "stays at 300,000" in cl.convergence(drift)["verdict"]
    finally:
        cl.ca.goal_state = goal


# --- on a tiny pair --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_pair(tmp_path_factory):
    from test_e3_campaign_run import campaign_tiny

    root = tmp_path_factory.mktemp("tiny_campaign")
    cfg = campaign_tiny()
    dirs = []
    for cond in ("experimental", "reference"):
        d = run_mini.make_seed_run_dir(root / "runs", "agent-tiny", 0, cond)
        run_agent.run_one(cfg, 0, cond, d, run_mini.configure_runtime(cfg))
        dirs.append(d)
    return root, dirs


def _handoff_from(dirs: list[Path], out: Path) -> dict:
    """A handoff laid out as make_handoff.py writes it, for the given run directories (jobs seed0_<condition>)."""
    out.mkdir()
    rows = []
    for d in dirs:
        cond = yaml.safe_load((d / "meta.yaml").read_text())["condition"]
        arc = f"results/E3_worldmodel_4zones/{d.name}"
        with tarfile.open(out / f"{d.name}.tgz", "w:gz") as tar:
            tar.add(d, arcname=arc, filter=mh._excluded_run)
        with tarfile.open(out / f"{d.name}.measures.tgz", "w:gz") as tar:
            tar.add(d / "measures", arcname=f"{arc}/measures", filter=mh._excluded)
        with tarfile.open(out / f"{d.name}.weights.tgz", "w:gz") as tar:
            tar.add(d / "checkpoints", arcname=f"{arc}/checkpoints", filter=mh._weights_only)
            tar.add(d / "replay_final.npz", arcname=f"{arc}/replay_final.npz")
        rows.append({"job": f"seed0_{cond}", "exit_code": 0, "status": "done", "finished": True, "not_started": False, "run_dir": d.name,
                     "measured_checkpoints": len(list((d / "measures").glob("env*/results.json")))})
    (out / "handoff.json").write_text(json.dumps({"runs": rows}))
    (out / "handoff.md").write_text("# GPU batch handoff\n")
    return {r["job"]: r for r in rows}


@pytest.fixture()
def tiny_rules(monkeypatch, tiny_pair):
    _, dirs = tiny_pair
    meta = yaml.safe_load((dirs[0] / "meta.yaml").read_text())
    for mod in (ci, cr.ci):  # the report loads its own instance of the module
        monkeypatch.setattr(mod, "expected_jobs", lambda: {f"seed0_{c}": {"seed": 0, "condition": c, "pilot": False} for c in ci.CONDITIONS})
        monkeypatch.setattr(mod, "BUDGET", 300)
        monkeypatch.setattr(mod, "grid_points", lambda: [100, 200, 300])
        monkeypatch.setattr(mod, "CAMPAIGN_SOURCE_SHA256", meta["source_sha256"])
    return dirs


def test_integrity_passes_extracts_and_refuses(tmp_path, tiny_rules):
    dirs = tiny_rules
    _handoff_from(dirs, tmp_path / "h")
    report = ci.check_handoff(tmp_path / "h")
    assert report["passed"] and all(r["interruptions"]["log_lines"] == [] for r in report["runs"])
    extracted = ci.extract(tmp_path / "h", report, tmp_path / "x")
    for d, src in zip(extracted, dirs):
        assert (d / "measures").is_dir() and (d / "checkpoints" / "agent_env000300.pt").exists() and (d / "replay_final.npz").exists()
        assert sorted(p.name for p in (d / "measures").iterdir()) == sorted(p.name for p in (src / "measures").iterdir())
    with pytest.raises(FileExistsError):
        ci.extract(tmp_path / "h", report, tmp_path / "x")
    assert "passed" in ci.markdown(report)
    # a failed job, a missing measure, a wrong code hash: refused, with the reason
    h2 = tmp_path / "h2"
    shutil.copytree(tmp_path / "h", h2)
    rows = json.loads((h2 / "handoff.json").read_text())
    rows["runs"][0]["exit_code"] = 1
    (h2 / "handoff.json").write_text(json.dumps(rows))
    with pytest.raises(ci.IntegrityError, match="exit_code"):
        ci.check_handoff(h2)
    h3 = tmp_path / "h3"
    shutil.copytree(tmp_path / "h", h3)
    name = dirs[1].name
    with tarfile.open(h3 / f"{name}.measures.tgz", "w:gz") as tar:  # the last checkpoint's measures dropped
        tar.add(dirs[1] / "measures", arcname=f"results/E3_worldmodel_4zones/{name}/measures",
                filter=lambda m: None if "env000300" in m.name else m)
    with pytest.raises(ci.IntegrityError, match="measures at"):
        ci.check_handoff(h3)
    (h3 / f"{name}.weights.tgz").unlink()
    with pytest.raises(ci.IntegrityError, match="archive"):
        ci.check_handoff(h3)


def test_integrity_wrong_code_and_resume_reported(tmp_path, tiny_rules, monkeypatch):
    dirs = tiny_rules
    _handoff_from(dirs, tmp_path / "h")
    monkeypatch.setattr(ci, "CAMPAIGN_SOURCE_SHA256", "0" * 64)
    with pytest.raises(ci.IntegrityError, match="expected the campaign's"):
        ci.check_handoff(tmp_path / "h")


def test_the_pilot_handoff_without_measures_and_weights_is_refused():
    h = REPO / "handoff" / "gpu_pilot_cl"
    if not h.exists():
        pytest.skip("pilot handoff not in this checkout")
    with pytest.raises(ci.IntegrityError) as err:
        ci.check_handoff(h)
    assert "absent from handoff.json" in str(err.value)  # the campaign's new jobs are not there


def test_campaign_hash_is_the_runbook_s():
    text = (REPO / "experiments" / "E3_worldmodel_4zones" / "RUN_GPU.md").read_text(encoding="utf-8")
    assert ci.CAMPAIGN_SOURCE_SHA256 in text


def test_rsa_slot_is_free():
    used = {run_agent.SLOT_AGENT_DELTA, run_agent.SLOT_GOAL_KERNEL, *range(1, 5), *range(6, 14)}
    assert rsa_latents.SLOT_RSA not in used and rsa_latents.SLOT_RSA < run_agent.e3_rng.N_MEASURE_SLOTS


def test_one_command_report_on_the_tiny_pair(tmp_path, tiny_rules):
    """--handoff mode end to end (integrity, extraction, analyses, RSA on the CPU, figures, tables, the section, the
    append between markers); then --runs-dir mode marks its output as not the campaign's."""
    dirs = tiny_rules
    _handoff_from(dirs, tmp_path / "h")
    report = tmp_path / "REPORT.md"
    report.write_text("# E3\n\nexisting text\n")
    work = tmp_path / "w"
    assert cr.main(["--handoff", str(tmp_path / "h"), "--work", str(work), "--append-report", str(report)]) == 0
    for name in ("RESULTS_SECTION.md", "analysis.json", "CAMPAIGN_ANALYSIS.md", "tables/learning.csv", "tables/rsa.csv",
                 "figures/curves_learning.png", "figures/curves_b_alpha.png"):
        assert (work / name).exists(), name
    text = (work / "RESULTS_SECTION.md").read_text(encoding="utf-8")
    for head in ("### Integrity", "### Main", "### Propagation", "### Measurement error", "### Convergence", "### Learning", "### RSA"):
        assert head in text
    assert "NOT THE CAMPAIGN" not in text
    rsa_files = list((work / "rsa").rglob("env*.json"))
    assert rsa_files
    summary = json.loads(rsa_files[0].read_text())
    assert summary["slot"] == rsa_latents.SLOT_RSA and set(summary["states"]["latents"]) == {"h", "z"}
    assert summary["states"]["latents"]["h"]["regression"]["n_points"] <= 143
    for f in rsa_files:  # the reference's controllability predictor: removed explicitly, absent from its regression
        s = json.loads(f.read_text())
        reg = s["states"]["latents"]["z"]["regression"]
        if s["condition"] == "reference":
            assert "controllability" in s["states"]["removed_predictors"] and "controllability" not in reg["unique_r2"]
        assert {"vif", "variance_partition", "predictor_correlations"} <= set(reg)
        ratio = s["states"]["latents"]["z"]["ratios"]["controllable_vs_uncontrollable"]  # by appearance, in both runs (33)
        assert ratio["n_a"] > 0 and ratio["n_b"] > 0 and np.isfinite(ratio["ratio"])
    assert "Partition of the explained variance" in text and "p = 1/16" in text and "not the JS divergence of the joint" in text
    appended = report.read_text(encoding="utf-8")
    assert appended.startswith("# E3\n\nexisting text") and appended.count(cr.MARK_START) == 1
    cr.append_report(report, text)  # a second append replaces the section
    assert report.read_text(encoding="utf-8").count(cr.MARK_START) == 1
    work2 = tmp_path / "w2"
    assert cr.main(["--runs-dir", str(dirs[0].parent), "--work", str(work2), "--no-rsa"]) == 0
    assert "NOT THE CAMPAIGN'S HANDOFF" in (work2 / "RESULTS_SECTION.md").read_text(encoding="utf-8")
    assert cr.main(["--runs-dir", str(dirs[0].parent), "--work", str(tmp_path / "w3"), "--no-rsa", "--append-report", str(report)]) == 1
