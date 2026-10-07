"""The pilot's runs as campaign seeds (docs/e3/DECISIONS.md, 2026-10-07 (30)) on a TINY configuration: a run trained with
the pilot's keys and measured afterwards by campaign_measure_pilot.py gives, at every checkpoint up to its last resume
point, exactly what a run trained with the campaign's keys measured during training -- same training, same weights,
same replay, same streams. The last checkpoint, after the last resume point, is flagged."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import yaml

from _experiment_loader import load_experiment_module

cmp = load_experiment_module("E3_worldmodel_4zones", "campaign_measure_pilot.py", "test_campaign_measure_pilot")
run_agent, run_mini = cmp.run_agent, cmp.run_mini


def _configs():
    from test_e3_campaign_run import campaign_tiny
    from test_e3_closed_loop import tiny_config

    campaign = campaign_tiny()
    campaign["measure"]["checkpoint_measures"]["min_in_window"] = 0  # no extra checkpoint (a pilot run has no weights for one)
    pilot = tiny_config()
    pilot["measure"].update(final=True, checkpoint_goal_kernel={"draws": 8})
    return pilot, campaign


def test_a_pilot_run_measured_afterwards_is_a_campaign_run(tmp_path):
    pilot_cfg, campaign_cfg = _configs()
    # short names (Windows' path length): the pattern <...>__worldmodel-<config>__seed<NN>__<...> of a run directory
    pilot_dir = tmp_path / "p" / "x__pair-experimental__worldmodel-agent-t__seed00__1"
    pilot_dir.mkdir(parents=True)
    run_agent.run_one(pilot_cfg, 0, "experimental", pilot_dir, run_mini.configure_runtime(pilot_cfg))
    campaign_dir = tmp_path / "c" / "x__pair-experimental__worldmodel-agent-t__seed00__1"
    campaign_dir.mkdir(parents=True)
    run_agent.run_one(campaign_cfg, 0, "experimental", campaign_dir, run_mini.configure_runtime(campaign_cfg))
    # the campaign's keys do not change the training: same weights at the end
    wa = torch.load(pilot_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    wb = torch.load(campaign_dir / "checkpoints" / "agent_final.pt", weights_only=False)
    for part in ("rssm", "goal_head", "critic"):
        assert all(torch.equal(wa[part][k], wb[part][k]) for k in wa[part]), part
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(campaign_cfg), encoding="utf-8")
    before = sorted((p.relative_to(pilot_dir).as_posix(), p.stat().st_mtime_ns) for p in pilot_dir.rglob("*"))
    out = cmp.measure_pilot_run(pilot_dir, config_path, "cpu", tmp_path / "m")
    assert sorted((p.relative_to(pilot_dir).as_posix(), p.stat().st_mtime_ns) for p in pilot_dir.rglob("*")) == before  # read only
    measured = [json.loads(line) for line in (out / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    in_run = {r["env_step"]: r for r in (json.loads(line) for line in (campaign_dir / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines())}
    resume_env = int(torch.load(pilot_dir / "checkpoints" / "resume.pt", weights_only=False)["progress"]["env_step"])
    assert [r["env_step"] for r in measured] == list(in_run) == [100, 200, 300] and resume_env == 250
    for r in measured:
        c = r["env_step"]
        assert r["twin_matches_pilot_record"] and r["twin_by_zone"] == in_run[c]["twin_by_zone"]
        assert ("measures" in r) == ("measures" in in_run[c])
        if c <= resume_env:  # exact: same weights, same replay, same streams
            assert r["goal_diagnostics"] == in_run[c]["goal_diagnostics"]
            if "measures" in r:
                assert r["measures"] == in_run[c]["measures"]
                a, b = np.load(out / r["measures"]["dir"] / "arrays.npz"), np.load(campaign_dir / r["measures"]["dir"] / "arrays.npz")
                assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
        else:
            assert r["replay_env_steps"] == resume_env  # the last checkpoint, on the last resume point's replay: flagged
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["replay_env_steps_final"] == resume_env and res["seed"] == 0 and res["condition"] == "experimental"
    t1, t2 = np.load(out / "twin.npz"), np.load(campaign_dir / "twin.npz")
    assert set(t1.files) == set(t2.files) and all(np.array_equal(t1[k], t2[k]) for k in t1.files)
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == ["agent_env000100.pt", "agent_env000200.pt", "agent_env000300.pt",
                                                                       "agent_final.pt"]
    assert (out / "replay_final.npz").exists() and yaml.safe_load((out / "meta.yaml").read_text(encoding="utf-8"))["status"] == "done"
    # campaign_analysis reads it as a run
    ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "test_campaign_measure_pilot_ca")
    assert sorted(ca.Run(out).measure_dirs) == sorted(c for c, r in in_run.items() if "measures" in r)


def test_a_run_with_another_training_is_refused(tmp_path):
    pilot_cfg, campaign_cfg = _configs()
    other = {**pilot_cfg, "train": {**pilot_cfg["train"], "replay_ratio": 2}}
    with pytest.raises(ValueError, match="training differs"):
        cmp.campaign_config_for(other, campaign_cfg)
    assert cmp.campaign_config_for(pilot_cfg, campaign_cfg)["measure"] == campaign_cfg["measure"]
