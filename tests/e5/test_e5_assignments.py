"""E5's zone assignments (pipeline.episode_sources, envs.layouts.two_zones' ud_slot option), the shape tests decided
before any launch (docs/e5/DECISIONS.md):

- 12 distinct assignments, the canonical one is the configuration's (UD at nw, CD at se on diagonal 1);
- pairing, for EACH assignment: every (s, a) a point mass in the experimental environment and its reference (same
  H(s'|s,a) = 0, same support size 1), I(a;s'|s) differs exactly on the UD slot's states (0 vs log2 5);
- invariants: the corridor, the goal and the region of every state never change; the conveyor's drift is resolved for
  its own slot (away from the goal); the theme follows the type (T_inc on UD, T_ctrl on CD, corridor colour elsewhere);
- the draw: over 10^4 episodes, the share of redrawn episodes is f within the binomial precision (4 sd), the 12
  assignments are uniform among the redrawn ones (chi-square), the others are canonical; the same draws for a run and
  its reference (by episode index); f = 0 never leaves the canonical assignment;
- the default arguments reproduce E4 (the options pinned);
- model/simulation agreement of a moved conveyor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR
from helplessness.envs.dynamics.controllable import ControllableDeterministic
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.layouts import four_zones_pixel, two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.zones import resolve_drift
from helplessness.measures.controllability import conditional_entropy_per_state, controllability_per_state
from helplessness.pipeline.episode_sources import (
    ASSIGNMENTS,
    ZoneAssignmentShuffle,
    assignment_colors,
    assignment_roles,
    canonical_assignment,
)
from helplessness.streams.assignment import assignment_generator
from helplessness.streams.worldmodel import spawn_streams

LOG2_5 = np.log2(5)
T_INC, T_CTRL = (227, 17, 15), (22, 0, 244)


def tensor(layout) -> np.ndarray:
    themes = two_zones.themes({q: CORRIDOR_COLOR for q in QUADRANT_ORDER}, (0, 0, 0))
    return four_zones_pixel.build_pixel_env(layout, themes, goal_reward=50.0).transition_tensor()


def test_twelve_distinct_assignments_and_the_canonical_one():
    assert len(ASSIGNMENTS) == 12 and len(set(ASSIGNMENTS)) == 12
    assert all(ud != cd and {ud, cd} <= set(QUADRANT_ORDER) for ud, cd in ASSIGNMENTS)
    for diagonal in (1, 2):
        r = two_zones.roles(diagonal)
        assert ASSIGNMENTS[canonical_assignment(diagonal)] == (r["ud"], r["cd"])
        assert assignment_colors(canonical_assignment(diagonal), diagonal, T_INC, T_CTRL) == two_zones.training_colors(diagonal, T_INC, T_CTRL)


def test_defaults_reproduce_e4():
    for diagonal in (1, 2):
        for condition in two_zones.CONDITIONS:
            a, b = two_zones.build_layout(diagonal, condition), two_zones.build_layout(diagonal, condition, ud_slot=None)
            assert np.array_equal(tensor(a), tensor(b))
            same = two_zones.build_layout(diagonal, condition, ud_slot=two_zones.roles(diagonal)["ud"])
            assert np.array_equal(tensor(a), tensor(same))


@pytest.mark.parametrize("k", range(12))
def test_pairing_for_every_assignment(k):
    ud, cd = ASSIGNMENTS[k]
    exp, ref = (tensor(two_zones.build_layout(1, c, ud_slot=ud)) for c in ("experimental", "reference"))
    layout = two_zones.build_layout(1, "experimental", ud_slot=ud)
    for P in (exp, ref):
        assert np.allclose(P.max(-1), 1.0)  # H(s'|s,a) = 0 everywhere
    assert np.array_equal((exp > 0).sum(-1), (ref > 0).sum(-1))  # support of 1 everywhere, in both
    assert np.allclose(conditional_entropy_per_state(exp), conditional_entropy_per_state(ref))
    I_exp, I_ref = controllability_per_state(exp), controllability_per_state(ref)
    ud_states = two_zones.region_states(layout)[ud]
    assert set(np.flatnonzero(~np.isclose(I_exp, I_ref))) == set(ud_states)
    assert np.allclose(I_exp[ud_states], 0.0) and np.allclose(I_ref[ud_states], LOG2_5)


@pytest.mark.parametrize("k", range(12))
def test_invariants_corridor_goal_regions_drift_and_theme(k):
    ud, cd = ASSIGNMENTS[k]
    canonical = two_zones.build_layout(1, "experimental")
    layout = two_zones.build_layout(1, "experimental", ud_slot=ud)
    P0, P = tensor(canonical), tensor(layout)
    regions0, regions = two_zones.region_states(canonical), two_zones.region_states(layout)
    assert all(np.array_equal(regions0[r], regions[r]) for r in regions0)
    corridor = regions["corridor"]
    assert np.array_equal(P0[corridor], P[corridor])  # the corridor (goal included) is untouched
    assert two_zones.GOAL_STATE in corridor
    kinds = {z.appearance.category: z.dynamics for z in layout.zones}
    assert isinstance(kinds[ud], UncontrollableDeterministic)
    assert kinds[ud].drift == resolve_drift(tuple(two_zones.CFG["blocks"][ud]["rows"]), two_zones.GRID_SHAPE, two_zones.GOAL_RC,
                                            two_zones.DIRECTION)
    assert all(isinstance(kinds[q], ControllableDeterministic) for q in QUADRANT_ORDER if q != ud)
    colors, roles = assignment_colors(k, 1, T_INC, T_CTRL), assignment_roles(k)
    assert colors[ud] == T_INC and colors[cd] == T_CTRL and roles[ud] == "ud" and roles[cd] == "cd"
    assert all(colors[q] == tuple(CORRIDOR_COLOR) and roles[q] == "neutral" for q in QUADRANT_ORDER if q not in (ud, cd))


@pytest.mark.parametrize("ud", QUADRANT_ORDER)
def test_model_simulation_agreement_of_a_moved_conveyor(ud):
    env = four_zones_pixel.build_pixel_env(two_zones.build_layout(1, "experimental", ud_slot=ud),
                                           two_zones.themes({q: CORRIDOR_COLOR for q in QUADRANT_ORDER}, (100, 45, 0)), goal_reward=50.0, seed=0)
    P = env.transition_tensor()
    for s in two_zones.region_states(env.zone_layout)[ud]:
        for a in range(5):
            env.reset(start_state=int(s))
            _, _, _, _, info = env.step(a)
            assert P[s, a, info["state"]] == 1.0


def fake_worlds(n_states: int = 3):
    """Light stand-ins for the 12 worlds (the draw needs none of their content)."""
    return [SimpleNamespace(env=f"env{k}", templates_np=np.full((n_states, 2, 2, 3), k, dtype=np.uint8),
                            background_t=torch.full((3, 2, 2), k / 255.0)) for k in range(12)]


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 1.0])
def test_draw_frequency_and_uniformity(fraction):
    n = 10_000
    canonical = canonical_assignment(1)
    src = ZoneAssignmentShuffle(fake_worlds(), canonical, fraction, assignment_generator(7), n_states=3)
    out = [src.next_episode(i) for i in range(n)]
    drawn = np.array(src.episode_drawn)
    k = np.array(src.episode_assignment)
    sd = np.sqrt(n * fraction * (1 - fraction))
    assert abs(drawn.sum() - n * fraction) <= 4 * sd + 1e-9
    assert np.all(k[~drawn] == canonical)
    if fraction > 0:
        counts = np.bincount(k[drawn], minlength=12)
        assert chisquare(counts).pvalue > 1e-3
    assert [o[0] for o in out] == [f"env{j}" for j in k] and [o[1] for o in out] == [3 * j for j in k]


def test_same_draws_for_a_run_and_its_reference_by_episode_index():
    """The assignment stream depends on the seed only: two sources of one seed draw the same sequence, whatever the
    steps at which their episodes start."""
    a = ZoneAssignmentShuffle(fake_worlds(), canonical_assignment(1), 0.5, assignment_generator(3), n_states=3)
    b = ZoneAssignmentShuffle(fake_worlds(), canonical_assignment(1), 0.5, assignment_generator(3), n_states=3)
    for i in range(500):
        a.next_episode(10 * i)
        b.next_episode(36 * i + 5)
    assert a.episode_assignment == b.episode_assignment and a.episode_drawn == b.episode_drawn
    c = ZoneAssignmentShuffle(fake_worlds(), canonical_assignment(1), 0.5, assignment_generator(4), n_states=3)
    for i in range(500):
        c.next_episode(i)
    assert c.episode_assignment != a.episode_assignment


def test_assignment_stream_is_independent_of_the_training_streams():
    """Its root differs from both conditions' roots: its first draws match none of their streams' first draws."""
    first = assignment_generator(1).random(4)
    for condition in ("experimental", "reference"):
        for name, g in spawn_streams(1, condition).items():
            assert not np.allclose(g.random(4), first), (condition, name)


def test_within_group_pairs_never_cross_groups():
    """The proposed switch statistic (docs/e5/DECISIONS.md (17)) pairs items of the same group only: two groups whose
    items are identical within and opposite across give 0, the default statistic does not; no pair gives NaN."""
    from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon, mean_within_group_pair_jensen_shannon

    a, b = np.zeros((1, 4, 2)), np.zeros((1, 4, 2))
    a[..., 0], b[..., 1] = 1.0, 1.0
    probs = np.concatenate([np.repeat(a, 6, 0), np.repeat(b, 6, 0)])
    groups = np.array([0] * 6 + [1] * 6)
    assert mean_within_group_pair_jensen_shannon(probs, groups, np.random.default_rng(0)) == 0.0
    assert mean_random_pair_jensen_shannon(probs, np.random.default_rng(0)) > 1.0
    assert np.isnan(mean_within_group_pair_jensen_shannon(probs[:2], np.array([0, 1]), np.random.default_rng(0)))


def test_within_group_statistic_with_one_group_is_the_default_one_bit_for_bit():
    """docs/e5/DECISIONS.md (19): one background -> the same draws and value as E3/E4's statistic, and the stream left in
    the same state."""
    from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon, mean_within_group_pair_jensen_shannon

    probs = np.random.default_rng(1).dirichlet(np.ones(4), size=(50, 8))
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    assert mean_within_group_pair_jensen_shannon(probs, np.zeros(50, dtype=np.int64), r1) == mean_random_pair_jensen_shannon(probs, r2)
    assert r1.random() == r2.random()


# ---- the conveyor in the 12 assignments (docs/e5/DECISIONS.md (24)) ------------------------------------------------

def _e3_symmetry():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("e3_symmetry_for_e5", Path(__file__).parents[1] / "e3" / "test_e3_symmetry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPECTED_DRIFT = {"nw": (1, 0), "ne": (1, 0), "se": (-1, 0), "sw": (-1, 0)}  # away from the goal (row 0), along the rows


@pytest.mark.parametrize("k", range(12))
def test_every_assignment_is_the_canonical_conveyor_up_to_a_symmetry(k):
    """For each assignment, both conditions: the UD slot holds a conveyor with a well-defined drift away from the goal,
    and the goal-centred axial reflection named by assignment_symmetry transports the canonical environment's dynamics
    EXACTLY onto the assignment's (E3's _transports_exactly, every (s, a), actions permuted). Its class is E3's rotation
    class of the UD slot; the per-episode log carries it."""
    from helplessness.pipeline.episode_sources import REFLECTIONS, assignment_symmetry

    e3_symmetry = _e3_symmetry()
    sym = assignment_symmetry(k, 1)
    ud, cd = ASSIGNMENTS[k]
    assert (sym["ud"], sym["cd"]) == (ud, cd) and REFLECTIONS[sym["reflection"]]["nw"] == ud
    assert tuple(sym["drift"]) == EXPECTED_DRIFT[ud]
    assert sym["rotation_class"] == ("A" if ud in ("nw", "se") else "B")
    layout = two_zones.build_layout(1, "experimental", ud_slot=ud)
    conveyor = next(z for z in layout.zones if z.appearance.category == ud)
    assert isinstance(conveyor.dynamics, UncontrollableDeterministic) and tuple(conveyor.dynamics.drift) == EXPECTED_DRIFT[ud]
    for condition in ("experimental", "reference"):
        canonical = tensor(two_zones.build_layout(1, condition))
        moved = tensor(two_zones.build_layout(1, condition, ud_slot=ud))
        assert e3_symmetry._transports_exactly(canonical, moved, e3_symmetry.REFLECTION_GROUP[sym["reflection"]]), (k, condition)


def test_episode_log_carries_the_symmetry_class():
    from helplessness.pipeline.episode_sources import SYMMETRY_CLASSES, assignment_symmetry

    src = ZoneAssignmentShuffle(fake_worlds(), canonical_assignment(1), 1.0, assignment_generator(11), n_states=3)
    for i in range(200):
        src.next_episode(i)
    log = src.log_arrays()
    expected = [SYMMETRY_CLASSES.index(assignment_symmetry(int(k), 1)["rotation_class"]) for k in log["episode_assignment"]]
    assert log["episode_rotation_class"].tolist() == expected and set(expected) == {0, 1}


def test_late_threshold_switch():
    """docs/e5/DECISIONS.md (28): the late threshold never acts before late_after_steps (a history that switches by the
    main rule is unchanged), and after it a plateau above the late threshold switches, recorded as such."""
    from helplessness.agents.worldmodel.loop import SeparationSwitch

    def run(history, **late):
        sw = SeparationSwitch(max_steps=100_000, **late)
        for step, stat in history:
            sw.update(step, stat)
            if sw.switched:
                break
        return sw.switch_step, sw.switch_rule

    rising = [(s, min(4.0, s / 1000)) for s in range(0, 8000, 50)]  # crosses 3 bits at 3,050
    plateau = [(s, 0.0 if s < 3000 else 2.0) for s in range(0, 20000, 50)]  # stays at 2 bits
    assert run(rising) == run(rising, late_threshold_bits=1.5, late_after_steps=10000) == (3251, "threshold")
    assert run(plateau) == (None, None)
    assert run(plateau, late_threshold_bits=1.5, late_after_steps=10000) == (10201, "late_threshold")
    with pytest.raises(ValueError):
        SeparationSwitch(max_steps=100_000, late_threshold_bits=1.5)


def test_late_threshold_changes_no_switch_of_e4():
    """The late threshold acts from gradient step 10,000 on; every one of E4's eight runs switched by the main rule by
    6,001 (both waves, their logs; docs/e5/DECISIONS.md (27), (28)), so by construction it changes none of them
    (test_late_threshold_switch: a run that switches by the main rule is unchanged)."""
    from helplessness.config import REPO_ROOT
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "e5" / "e5.yaml").read_text(encoding="utf-8"))["e5"]
    e4_switches = [2551, 3901, 3051, 3451, 4801, 6001, 3701, 3701]
    assert cfg["late_switch"]["after_steps"] > max(e4_switches)
