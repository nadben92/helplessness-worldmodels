"""E5's training conditions on a tiny configuration (CPU, configs/e5/e5_mini.yaml, 300 environment steps, checkpoints
100 / 200 / 300), the shape tests decided before any launch (docs/e5/DECISIONS.md):

- f = 0 equals E4: the shuffle machinery with fraction 0 (assignment stream, 12-assignment replay table, per-step
  backgrounds all ON) gives E4's run (library options OFF) bit for bit -- the weights at every checkpoint, the training
  losses recorded there, and every stored step (observation row, action, reward, flags) -- for both conditions;
- under shuffle (f = 1): the assignment changes only at an episode's first step (is_first, where the model resets), so no
  replay window ever carries a transition from one assignment into another; the episodes' assignments are the
  stream's draws, the same in the reference by episode index; the background of every sampled step is its own
  assignment's, and the observation differs from it only on the agent's cell;
- immunisation: the run equals the reference of its seed up to the switch (the full state at the nominal switch step,
  weights, optimiser, replay, streams); the dynamics change at the first episode starting once switch_step transitions
  are done, every transition before it follows the reference's tensor and every one after it the experimental one; the
  run branched from the reference's resumable state equals the run recomputed from zero after the switch."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import torch

from _experiment_loader import load_experiment_module
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.config import REPO_ROOT
from helplessness.envs.layouts import two_zones
from helplessness.pipeline.agent_run import ClosedLoopHooks, background_groups, run_one
from helplessness.pipeline.episode_sources import ASSIGNMENTS, background_uint8, canonical_assignment
from helplessness.pipeline.runtime import configure_runtime
from helplessness.pipeline.two_zones_records import make_hooks as e4_hooks
from helplessness.streams.assignment import assignment_generator

run_e5 = load_experiment_module("E5_belief_training", "run_e5.py", "test_e5_training_run")
e5_setup = run_e5.e5_setup
N = two_zones.N_STATES
CANONICAL = canonical_assignment(1)
SWITCH, POST = 200, 200


def tiny_cfg() -> dict:
    cfg = e5_setup.load_config(REPO_ROOT / "configs" / "e5" / "e5_mini.yaml")
    cfg["budget"]["total_steps"] = 260
    cfg["checkpoints"]["grid"] = [100, 200, 300]
    cfg["e4"]["run_configuration"] = "p1-a-d1"
    return cfg


CONF = e5_setup.configuration(tiny_cfg(), "p1-a-d1", "A")


def quiet(hooks: ClosedLoopHooks) -> ClosedLoopHooks:
    """The same training without the checkpoint records (they never change the training: E4's test)."""
    return ClosedLoopHooks(episode=hooks.episode, build_world=hooks.build_world, checkpoint_record=lambda state: {},
                           finalize=lambda records, world, run_dir: {}, episode_source=hooks.episode_source)


def train(root, name: str, condition: str, hooks: ClosedLoopHooks, **kwargs):
    run_dir = root / name
    run_dir.mkdir()
    cfg = tiny_cfg()
    run_one(copy.deepcopy(cfg), CONF["seed"], condition, run_dir, configure_runtime(cfg), hooks=hooks, **kwargs)
    return run_dir


def weights(run_dir, step: int) -> dict:
    return torch.load(run_dir / "checkpoints" / f"agent_env{step:06d}.pt", map_location="cpu", weights_only=False)


def same_weights(a: dict, b: dict) -> bool:
    return all(torch.equal(a[m][k], b[m][k]) for m in ("rssm", "goal_head", "critic") for k in a[m])


def replay_arrays(run_dir) -> dict:
    return dict(np.load(run_dir / "replay_final.npz"))


def train_losses(run_dir) -> list[dict]:
    return [r["train"] for r in e5_setup.records(run_dir)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5")
    out = {}
    for condition in ("experimental", "reference"):
        out[f"e4_{condition}"] = train(root, f"e4_{condition}", condition, quiet(e4_hooks(tiny_cfg(), CONF)))
        out[f"f0_{condition}"] = train(root, f"f0_{condition}", condition, quiet(run_e5.shuffle_hooks(tiny_cfg(), CONF, 0.0)))
        out[f"f1_{condition}"] = train(root, f"f1_{condition}", condition, quiet(run_e5.shuffle_hooks(tiny_cfg(), CONF, 1.0)))
    out["reference_resumable"] = train(root, "ref_resumable", "reference", quiet(run_e5.baseline_hooks(tiny_cfg(), CONF)),
                                       save_resumable_at=(SWITCH,))
    total = SWITCH + POST + tiny_cfg()["episode"]["max_steps"]
    imm = quiet(run_e5.immunisation_hooks(tiny_cfg(), CONF, SWITCH, POST))
    out["imm_recomputed"] = train(root, "imm_recomputed", "reference", imm, total_env_steps=total, save_resumable_at=(SWITCH,))
    out["imm_branched"] = train(root, "imm_branched", "reference", quiet(run_e5.immunisation_hooks(tiny_cfg(), CONF, SWITCH, POST)),
                                total_env_steps=total,
                                branch_from=out["reference_resumable"] / "checkpoints" / f"resumable_env{SWITCH:06d}.pt")
    return out


@pytest.mark.parametrize("condition", ["experimental", "reference"])
def test_fraction_zero_equals_e4_bit_for_bit(runs, condition):
    e4, f0 = runs[f"e4_{condition}"], runs[f"f0_{condition}"]
    for step in (100, 200, 300):
        assert same_weights(weights(e4, step), weights(f0, step)), step
    assert train_losses(e4) == train_losses(f0)
    a, b = replay_arrays(e4), replay_arrays(f0)
    for key in ("s", "a", "r", "terminal", "truncated", "is_first"):
        assert np.array_equal(a[key], b[key]), key
    assert "obs" not in a and np.array_equal(b["obs"], CANONICAL * N + b["s"].astype(np.int64))


def episodes(arrays: dict) -> list[np.ndarray]:
    starts = np.flatnonzero(arrays["is_first"])
    return np.split(np.arange(len(arrays["s"])), starts[1:])


@pytest.mark.parametrize("condition", ["experimental", "reference"])
def test_shuffle_assignment_changes_only_at_episode_starts(runs, condition):
    arrays = replay_arrays(runs[f"f1_{condition}"])
    block = arrays["obs"].astype(np.int64) // N
    assert np.array_equal(arrays["obs"].astype(np.int64) % N, arrays["s"].astype(np.int64))
    changes = np.flatnonzero(block[1:] != block[:-1]) + 1
    assert changes.size > 0 and arrays["is_first"][changes].all()
    ends = np.flatnonzero(arrays["terminal"] | arrays["truncated"])
    assert arrays["is_first"][ends[ends + 1 < len(block)] + 1].all()  # an episode's last step is followed by a reset
    # the episodes' assignments are the stream's draws (f = 1: every episode drawn), in order
    rng = assignment_generator(CONF["seed"])
    expected = []
    for _ in range(len(episodes(arrays))):
        rng.random()
        expected.append(int(rng.integers(len(ASSIGNMENTS))))
    assert [int(block[e[0]]) for e in episodes(arrays)] == expected


def test_shuffle_same_assignments_in_the_reference_by_episode_index(runs):
    blocks = []
    for condition in ("experimental", "reference"):
        arrays = replay_arrays(runs[f"f1_{condition}"])
        blocks.append([int(arrays["obs"][e[0]]) // N for e in episodes(arrays)])
    n = min(map(len, blocks))
    assert n >= 5 and blocks[0][:n] == blocks[1][:n]


def test_shuffle_backgrounds_follow_the_episode_assignment(runs):
    cfg = tiny_cfg()
    hooks = run_e5.shuffle_hooks(cfg, CONF, 1.0)
    world = hooks.build_world(cfg, CONF["seed"], "experimental")
    source = hooks.episode_source(world, cfg, CONF["seed"], "experimental")
    replay = ReplayBuffer.from_compact(replay_arrays(runs["f1_experimental"]), source.table, source.background_table, source.background_of_row)
    batch, labels = replay.sample(8, 8, np.random.default_rng(0))
    rows = replay.obs_rows_and_flags()[0]
    assert len(set((rows // N).tolist())) > 1
    canonical_bg = background_uint8(world)
    k_px = cfg["render"]["k"]
    seen_other = False
    for i in range(8):
        for t in range(8):
            x, bg = np.asarray(batch.x[i, t]), np.asarray(batch.background[i, t])
            k = int(np.flatnonzero((source.table == x).all(axis=(1, 2, 3)))[0]) // N
            assert np.array_equal(bg, source.background_table[k])
            diff = np.argwhere((x != bg).any(-1))
            r, c = divmod(int(labels.s[i, t]), two_zones.GRID_SHAPE[1])
            # the observation differs from its background on the agent's cell only
            assert diff.size and (diff[:, 0] // k_px == r).all() and (diff[:, 1] // k_px == c).all()
            if k != CANONICAL:
                seen_other = True
                assert (x != canonical_bg).any(-1).sum() > k_px * k_px  # the canonical background would mask far more
    assert seen_other


def _state(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_immunisation_equals_the_reference_until_the_switch(runs):
    a = _state(runs["reference_resumable"] / "checkpoints" / f"resumable_env{SWITCH:06d}.pt")
    b = _state(runs["imm_recomputed"] / "checkpoints" / f"resumable_env{SWITCH:06d}.pt")
    for key in ("rssm", "goal_head", "critic", "slow_critic"):
        assert all(torch.equal(a[key][k], b[key][k]) for k in a[key]), key
    for key in ("optimizer", "goal_optimizer", "critic_optimizer"):
        sa, sb = a[key]["state"], b[key]["state"]
        assert sa.keys() == sb.keys() and all(torch.equal(sa[p][f], sb[p][f]) for p in sa for f in sa[p] if torch.is_tensor(sa[p][f]))
    ra, rb = pickle.loads(a["replay"]).export_compact(), pickle.loads(b["replay"]).export_compact()
    assert all(np.array_equal(ra[k], rb[k]) for k in ra)
    assert a["streams"].keys() == b["streams"].keys()
    for name in a["streams"]:
        va, vb = a["streams"][name], b["streams"][name]
        assert (torch.equal(va, vb) if torch.is_tensor(va) else va == vb), name
    for key in ("env_step", "grad_step", "credit", "rep_warmup", "switch_env_step"):
        assert a["progress"][key] == b["progress"][key], key
    assert a["planned_env_steps"] == b["planned_env_steps"]


def test_immunisation_switches_dynamics_at_the_first_episode_starting_at_switch_step(runs):
    arrays = replay_arrays(runs["imm_recomputed"])
    cfg = tiny_cfg()
    P = {c: run_e5.two_zones_world.build_world(CONF["training_colors"], CONF["agent_color"], 1, c, cfg["render"], cfg["episode"], 1)
         .env.transition_tensor() for c in ("reference", "experimental")}
    done, effective, n_after_ud = 0, None, 0
    ud_states = set(two_zones.region_states(two_zones.build_layout(1, "experimental"))[two_zones.roles(1)["ud"]].tolist())
    for ep in episodes(arrays):
        if effective is None and done >= SWITCH:
            effective = done
        P_ep = P["experimental"] if effective is not None else P["reference"]
        s, a = arrays["s"].astype(np.int64)[ep], arrays["a"].astype(np.int64)[ep]
        for t in range(len(ep) - 1):
            assert P_ep[s[t], a[t], s[t + 1]] == 1.0
            n_after_ud += int(effective is not None and s[t] in ud_states)
        done += len(ep) - 1
    assert effective is not None and SWITCH <= effective < SWITCH + cfg["episode"]["max_steps"]
    assert n_after_ud > 0  # the conveyor was met after the switch (the check above bears on it)
    # an episode cut by the run's end has one more transition than its stored observations show (its last observation
    # is never stored): the run's length is the effective switch + post_steps
    unfinished = not (arrays["terminal"][-1] or arrays["truncated"][-1])
    assert done + int(unfinished) == effective + POST
    ck = sorted(int(p.stem[len("agent_env"):]) for p in (runs["imm_recomputed"] / "checkpoints").glob("agent_env*.pt"))
    assert ck == [effective + 100, effective + 200]  # E4's grid in post-switch steps (100; 200 = post_steps)


def test_branched_immunisation_equals_the_recomputed_one(runs):
    a, b = runs["imm_recomputed"], runs["imm_branched"]
    ck_a = sorted(p.name for p in (a / "checkpoints").glob("agent_env*.pt"))
    assert ck_a == sorted(p.name for p in (b / "checkpoints").glob("agent_env*.pt")) and ck_a
    for name in ck_a:
        assert same_weights(torch.load(a / "checkpoints" / name, map_location="cpu", weights_only=False),
                            torch.load(b / "checkpoints" / name, map_location="cpu", weights_only=False)), name
    assert train_losses(a) == train_losses(b)
    ra, rb = replay_arrays(a), replay_arrays(b)
    assert all(np.array_equal(ra[k], rb[k]) for k in ra)


def test_background_groups_are_the_assignments(runs):
    """The proposed switch-pair option (docs/e5/DECISIONS.md (17)) groups a batch's items by their background image: on
    a shuffled replay these groups are exactly the items' assignments."""
    cfg = tiny_cfg()
    hooks = run_e5.shuffle_hooks(cfg, CONF, 1.0)
    world = hooks.build_world(cfg, CONF["seed"], "experimental")
    source = hooks.episode_source(world, cfg, CONF["seed"], "experimental")
    replay = ReplayBuffer.from_compact(replay_arrays(runs["f1_experimental"]), source.table, source.background_table, source.background_of_row)
    starts = np.arange(0, len(replay) - 16, 37)[:8]
    batch, _ = replay.windows(starts, 16)
    groups = background_groups(batch)
    block = replay.obs_rows_and_flags()[0][(starts[:, None] + np.arange(16)[None, :]).reshape(-1)] // N
    assert len(set(block.tolist())) > 1
    _, expected = np.unique(block, return_inverse=True)
    assert np.array_equal(np.unique(groups, return_inverse=True)[1], expected)
