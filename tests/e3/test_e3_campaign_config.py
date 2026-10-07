"""The campaign's configuration (docs/e3/DECISIONS.md, 2026-10-07 (29)) is the pilot's, with only the measurement
protocol (x4 contexts), the seeds and the measures added; the budget is the pilot's 300,000 environment steps."""

from __future__ import annotations

import copy

from _experiment_loader import load_experiment_module

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "test_campaign_config_run_mini")
env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_campaign_config_env")
CONFIGS = run_mini.REPO_ROOT / "configs" / "e3"


def test_campaign_is_the_pilot_with_the_retained_protocol_and_budget():
    pilot = run_mini.load_mini_config(CONFIGS / "gpu_pilot_closed_loop.yaml")
    campaign = run_mini.load_mini_config(CONFIGS / "gpu_campaign.yaml")
    # training, agent, environment, budget: identical
    for block in ("model", "render", "budget", "collection", "train", "agent", "log", "runtime", "conditions"):
        assert campaign[block] == pilot[block], block
    assert campaign["budget"]["total_steps"] + campaign["budget"]["prefill_steps"] == 300_000
    # the protocol: x4 the pilot's contexts per state, nothing else in real_context
    assert campaign["measure"]["real_context"]["max_per_state"] == 4 * pilot["measure"]["real_context"]["max_per_state"] == 200
    p, c = copy.deepcopy(pilot["measure"]), copy.deepcopy(campaign["measure"])
    p["real_context"].pop("max_per_state"), c["real_context"].pop("max_per_state")
    added = {"checkpoint_measures", "light_arrays"}
    assert {k: v for k, v in c.items() if k not in added | {"checkpoint_goal_kernel"}} == \
        {k: v for k, v in p.items() if k != "checkpoint_goal_kernel"}
    assert c["checkpoint_goal_kernel"] == {"draws": 256, "goal_draws": 10_000, "agent": True}
    assert c["checkpoint_measures"] == {"window_after_switch": 50_000, "min_in_window": 2, "extra_offsets": [25_000, 50_000]}
    assert c["light_arrays"] is True and c["final"] is True
    assert campaign["twin"] == {**pilot["twin"], "save_kernels": True}
    assert campaign["checkpoints"] == {**pilot["checkpoints"], "save_final_replay": True}
    # seeds: 1, 2 (class A), 6, 7 (class B), none of the pilot's
    assert campaign["seeds"] == [1, 2, 6, 7] and not set(campaign["seeds"]) & set(pilot["seeds"])
    assert [env.block_class_for_seed(s) for s in campaign["seeds"]] == ["A", "A", "B", "B"]
    assert campaign["conditions"] == ["experimental", "reference"]
