"""E4's training run (run_e4.py, through pipeline.agent_run.run_one with hooks), on a tiny configuration (CPU):

- budget equivalence: a run stopped at B with --max-env-steps (planned budget unchanged) has, at B, exactly the weights
  of the full run's checkpoint B; a run PLANNED for a shorter budget does not (beta_rep's ramp after the switch is 20%
  of the planned gradient steps left: train.rep_warmup_steps null, E3's rule) -- the dependency the README documents;
- measuring never changes the training: the same run with E4's checkpoint records replaced by nothing gives the same
  weights."""

from __future__ import annotations

import copy

import pytest
import torch

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline.agent_run import ClosedLoopHooks, run_one
from helplessness.pipeline.runtime import configure_runtime

run_e4 = load_experiment_module("E4_generalization_gradient", "run_e4.py", "test_e4_training_run")
e4_setup = run_e4.e4_setup


def tiny_cfg(total_steps: int = 260) -> dict:
    cfg = e4_setup.load_config(REPO_ROOT / "configs" / "e4" / "e4_mini.yaml")
    cfg["budget"]["total_steps"] = total_steps
    cfg["checkpoints"]["grid"] = [100, 200, 300]
    cfg["e4"]["run_configuration"] = "p1-a-d1"
    return cfg


def run(cfg: dict, tmp_path, name: str, max_env_steps: int | None = None, quiet_hooks: bool = False) -> dict:
    conf = e4_setup.configuration(cfg, "p1-a-d1")
    hooks = run_e4.make_hooks(cfg, conf)
    if quiet_hooks:
        hooks = ClosedLoopHooks(episode=hooks.episode, build_world=hooks.build_world, checkpoint_record=lambda state: {},
                                finalize=lambda records, world, run_dir: {})
    run_dir = tmp_path / name
    run_dir.mkdir()
    run_one(copy.deepcopy(cfg), conf["seed"], "experimental", run_dir, configure_runtime(cfg), max_env_steps=max_env_steps, hooks=hooks)
    return run_dir


def weights(run_dir, step: int) -> dict:
    return torch.load(run_dir / "checkpoints" / f"agent_env{step:06d}.pt", map_location="cpu", weights_only=False)


def same_weights(a: dict, b: dict) -> bool:
    return all(torch.equal(a[m][k], b[m][k]) for m in ("rssm", "goal_head", "critic") for k in a[m])


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    return run(tiny_cfg(), tmp_path_factory.mktemp("full"), "full")


def test_stopping_early_with_the_same_planned_budget_equals_the_checkpoint(full_run, tmp_path):
    short = run(tiny_cfg(), tmp_path, "short", max_env_steps=200)
    assert same_weights(weights(full_run, 200), weights(short, 200))


def test_a_shorter_planned_budget_changes_the_trajectory(full_run, tmp_path):
    planned_short = run(tiny_cfg(total_steps=160), tmp_path, "planned_short")  # 160 + 40 prefill = 200 steps
    assert weights(full_run, 100)["switch_step"] is not None
    assert not same_weights(weights(full_run, 200), weights(planned_short, 200))


def test_checkpoint_measures_never_change_the_training(full_run, tmp_path):
    quiet = run(tiny_cfg(), tmp_path, "quiet", quiet_hooks=True)
    for step in (100, 200, 300):
        assert same_weights(weights(full_run, step), weights(quiet, step))


def test_run_outputs(full_run):
    import json

    records = [json.loads(line) for line in (full_run / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["env_step"] for r in records] == [100, 200, 300]
    last = records[-1]["e4"]
    assert set(last["walks"]) == {"nw", "ne", "se", "sw"} and "final" in last
    assert set(last["real_context"]["mass_true_by_region"]) == {"corridor", "nw", "ne", "se", "sw"}
    header = (full_run / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("env_step,grad_step,region,role")
    assert (full_run / "replay_final.npz").exists() and (full_run / "checkpoints" / "agent_final.pt").exists()
