"""twin.replay_counts (docs/e8/DECISIONS.md (20)): the counts rebuilt from a run's replay_final.npz equal run_one's own
counts.npz at every checkpoint of a mini run (which stops mid-episode, so its last transition is the pending one), and
every rebuilt transition is one of the run's environment's; synthetic replays pin the convention (the goal, the
truncation and the reset are not transitions out; a run ending on an episode end has nothing pending)."""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module
from helplessness.config import REPO_ROOT
from helplessness.pipeline import two_zones_setup
from helplessness.pipeline.agent_run import ClosedLoopHooks, run_one
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import make_hooks as e4_hooks
from helplessness.twin.replay_counts import counts_at, replay_transitions

run_e4 = load_experiment_module("E4_generalization_gradient", "run_e4.py", "test_replay_counts_run_e4")


def test_synthetic_convention():
    # episode 1: 0 -a1-> 1 -a2-> 2 (goal); episode 2: 3 -a0-> 4 -a4-> 5 (truncated); episode 3: 6 -a3-> (pending)
    s = np.array([0, 1, 2, 3, 4, 5, 6])
    a = np.array([1, 2, 0, 0, 4, 0, 3])
    terminal = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    truncated = np.array([0, 0, 0, 0, 0, 1, 0], bool)
    is_first = np.array([1, 0, 0, 1, 0, 0, 1], bool)
    P = np.zeros((8, 5, 8))
    P[6, 3, 7] = 1.0
    t = replay_transitions(s, a, terminal, truncated, is_first, P)
    assert list(zip(t["s"], t["a"], t["s_next"])) == [(0, 1, 1), (1, 2, 2), (3, 0, 4), (4, 4, 5), (6, 3, 7)]
    assert t["accounting"] == {"stored": 7, "transitions": 5, "with_stored_successor": 4, "pending": 1, "end_observations": 2,
                               "terminal": 1, "truncated": 1, "episodes_started": 3}
    with pytest.raises(ValueError):
        replay_transitions(s, a, terminal, truncated, is_first)  # pending without P
    done = replay_transitions(s[:6], a[:6], terminal[:6], truncated[:6], is_first[:6])
    assert done["accounting"]["pending"] == 0 and len(done["s"]) == 4
    assert counts_at(done, 4, 8, 5).sum() == 4 and counts_at(done, 2, 8, 5)[1, 2, 2] == 1


def test_broken_flags_are_refused():
    s, a = np.array([0, 1, 2]), np.array([1, 0, 1])
    with pytest.raises(ValueError):  # an end not followed by a first observation
        replay_transitions(s, a, np.array([0, 1, 0], bool), np.zeros(3, bool), np.array([1, 0, 0], bool))


def test_rebuilt_counts_equal_run_one_counts(tmp_path):
    cfg = run_e4.e4_setup.load_config(REPO_ROOT / "configs" / "e4" / "e4_mini.yaml")
    cfg["e4"]["run_configuration"] = "p1-b-d1"
    cfg["checkpoints"]["grid"] = [37, 120, 299]
    assert cfg["checkpoints"]["save_final_replay"]
    conf = {**two_zones_setup.configuration(cfg, "p1-b-d1"), "seed": 2}
    base = e4_hooks(cfg, conf)
    hooks = ClosedLoopHooks(episode=base.episode, build_world=base.build_world, checkpoint_record=lambda s: {}, finalize=lambda r, w, d: {})
    run_one(cfg, 2, "experimental", tmp_path, configure_runtime(cfg), hooks=hooks, max_env_steps=299)
    counts = dict(np.load(tmp_path / "counts.npz"))
    P = two_zones_setup.world(cfg, conf, "experimental").env.transition_tensor()
    replay = np.load(tmp_path / "replay_final.npz")
    t = replay_transitions(*(replay[k] for k in ("s", "a", "terminal", "truncated", "is_first")), P=P)
    assert t["accounting"]["transitions"] == 299 and t["accounting"]["pending"] == 1  # the mini run stops mid-episode
    assert {f"N_{k}" for k in (37, 120, 299)} <= set(counts)
    for key, N in counts.items():
        assert np.array_equal(counts_at(t, int(key[2:]), N.shape[0], N.shape[1]), N), key
    assert (P[t["s"], t["a"], t["s_next"]] == 1.0).all()
