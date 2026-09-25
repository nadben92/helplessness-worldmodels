"""T2, exact part (spec section 9): "pour chaque permutation, la
reflexion associee transporte P sur P exactement."

Corrected understanding (2026-09-23, see docs/e3/DECISIONS.md): the
right symmetry group is the goal-centred AXIAL REFLECTION group --
{identity, row-reflection (r -> -r, fixing the goal's row), column-
reflection (c -> -c, fixing the goal's column), point-reflection / a
180-degree turn (both)} -- not the full rotation group. A drift that
moves along the row axis, away from the goal, is preserved by all four
of these transforms (row-reflection negates it correctly; column-
reflection doesn't touch it since it never crosses a row band; point-
reflection is their composition) -- and this group sends any quadrant to
any other (it acts simply transitively on the 4 quadrants), which is
exactly what a counterbalancing scheme needs.

E2's own 4 counterbalancing rotations (rotation_for_seed_index /
quadrant_assignment_for_rotation, reused unchanged, frozen) are the
CYCLIC group C4 (successive 90-degree turns of which TYPE sits where),
which is a DIFFERENT group from the axial-reflection one above -- C4 has
an order-4 element (a genuine quarter turn), while the reflection group
has none (every element is its own inverse). The two groups happen to
share exactly two elements: identity and the 180-degree turn
(point-reflection). This is why E2's rotation=0 and rotation=2 ARE exact
symmetries (tested below, against E2's own built layouts) while
rotation=1 and rotation=3 are not (kept as a documentary regression pin,
second half of this file) -- not because drift "breaks" rotation in
general, but because rotation=1/3 are elements of a group (C4) that was
never the relevant symmetry group in the first place.

Block-factor consequence: E2's 4 rotations split into two classes under
this reflection group -- class A = {0, 2} (related to each other by
point-reflection), class B = {1, 3} (also related to each other by
point-reflection), and NO reflection relates a class-A rotation to a
class-B one. See experiments/E3_worldmodel_4zones/env.py::ROTATION_CLASS
and ::block_class_for_seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from _experiment_loader import load_experiment_module

e3_env = load_experiment_module("E3_worldmodel_4zones", "env.py", "test_e3_symmetry_env")

GRID_SHAPE = e3_env.GRID_SHAPE
N_ROWS, N_COLS = GRID_SHAPE
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))  # up, right, down, left, stay


def _rc_to_s(r: int, c: int) -> int:
    return (r % N_ROWS) * N_COLS + (c % N_COLS)


# The goal-centred axial-reflection group: the correct symmetry group
# for this layout (see module docstring), NOT the full D4/rotation group.
REFLECTION_GROUP = {
    "identity": lambda r, c: (r, c),
    "row_reflect": lambda r, c: ((-r) % N_ROWS, c),
    "col_reflect": lambda r, c: (r, (-c) % N_COLS),
    "point_reflect": lambda r, c: ((-r) % N_ROWS, (-c) % N_COLS),
}

# The quadrant_assignment a given reflection implies, expressed directly
# (not all of these are among E2's own quadrant_assignment_for_rotation
# outputs -- row_reflect and col_reflect are not; both are still valid
# inputs to the imported, unmodified build_layout()).
REFLECTION_ASSIGNMENTS = {
    "identity": {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"},
    "row_reflect": {"nw": "sw", "sw": "nw", "ne": "se", "se": "ne"},
    "col_reflect": {"nw": "ne", "ne": "nw", "se": "sw", "sw": "se"},
    "point_reflect": {"nw": "se", "se": "nw", "ne": "sw", "sw": "ne"},
}


def _action_permutation_for(posmap) -> list[int]:
    base = posmap(0, 0)
    perm = []
    for dr, dc in ACTIONS:
        nr, nc = posmap(dr, dc)
        ndr = nr - base[0]
        ndc = nc - base[1]
        ndr = ((ndr + 1) % N_ROWS) - 1 if abs(ndr) > 1 else ndr
        ndc = ((ndc + 1) % N_COLS) - 1 if abs(ndc) > 1 else ndc
        perm.append(ACTIONS.index((ndr, ndc)))
    return perm


def _transports_exactly(P0: np.ndarray, Pr: np.ndarray, posmap) -> bool:
    """True iff, for every state/action of P0, transporting P0's row
    through posmap (positions) and the induced action permutation
    reproduces Pr's own row exactly -- covers every zone type, drift
    included (drift-based zones simply never reference the action, so
    the action permutation is a no-op check for them; controllable zones
    exercise it fully)."""
    aperm = _action_permutation_for(posmap)
    n_states = N_ROWS * N_COLS
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = _rc_to_s(r, c)
            nr, nc = posmap(r, c)
            s_mapped = _rc_to_s(nr, nc)
            for a in range(5):
                row0 = P0[s, a]
                transported = np.zeros(n_states)
                nz = np.flatnonzero(row0)
                for s2 in nz:
                    r2, c2 = divmod(int(s2), N_COLS)
                    nr2, nc2 = posmap(r2, c2)
                    transported[_rc_to_s(nr2, nc2)] += row0[s2]
                if not np.allclose(Pr[s_mapped, aperm[a]], transported, atol=1e-9):
                    return False
    return True


def _find_matching_reflection(P0: np.ndarray, Pr: np.ndarray) -> str | None:
    for name, posmap in REFLECTION_GROUP.items():
        if _transports_exactly(P0, Pr, posmap):
            return name
    return None


def _tensor_for_assignment(quadrant_assignment) -> np.ndarray:
    layout = e3_env.build_layout(quadrant_assignment)
    return e3_env.build_pixel_env(layout, _themes_for(layout), goal_reward=50.0).transition_tensor()


def _themes_for(layout):
    from helplessness.envs.appearance.theme_sampling import build_themes

    categories = sorted({z.appearance.category for z in layout.zones if z.appearance.category != "corridor"})
    return build_themes(categories, goal_state=0, rng=np.random.default_rng(0), min_dist=5.0)


@pytest.fixture(scope="module")
def identity_tensor():
    return _tensor_for_assignment(REFLECTION_ASSIGNMENTS["identity"])


# ---------------------------------------------------------------------
# T2 exact: the axial-reflection group transports P exactly, every zone
# type, drift included.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["identity", "row_reflect", "col_reflect", "point_reflect"])
def test_each_axial_reflection_transports_p_exactly(identity_tensor, name):
    Pr = _tensor_for_assignment(REFLECTION_ASSIGNMENTS[name])
    assert _transports_exactly(identity_tensor, Pr, REFLECTION_GROUP[name])


def test_axial_reflection_group_sends_every_quadrant_to_every_other():
    """The group acts simply transitively on the 4 quadrants -- the
    geometric property that makes it a valid counterbalancing group at
    all (spec 2.2's premise)."""
    images = set()
    for posmap in REFLECTION_GROUP.values():
        images.add(posmap(3, 3))  # an interior point of the NW block
    assert len(images) == 4  # one image per quadrant, all distinct


# ---------------------------------------------------------------------
# Block-factor structure of E2's OWN 4 rotations under this group.
# ---------------------------------------------------------------------


def test_rotation_pairwise_reflection_classification_matches_class_a_b():
    """Class A = {0, 2}, class B = {1, 3}: every intra-class pair is
    related by some element of the axial-reflection group; no
    inter-class pair is."""
    tensors = {r: _tensor_for_assignment(e3_env.quadrant_assignment_for_rotation(r)) for r in range(4)}
    for i in range(4):
        for j in range(i + 1, 4):
            match = _find_matching_reflection(tensors[i], tensors[j])
            same_class = e3_env.ROTATION_CLASS[i] == e3_env.ROTATION_CLASS[j]
            if same_class:
                assert match is not None, f"rotation {i} and {j} are same-class but no reflection relates them"
            else:
                assert match is None, f"rotation {i} and {j} are different-class but {match} relates them"


def test_seed_to_class_split_is_exactly_10_10():
    classes = [e3_env.block_class_for_seed(seed) for seed in range(20)]
    assert classes.count("A") == 10
    assert classes.count("B") == 10


def test_seed_to_class_membership_is_the_expected_partition():
    class_a_seeds = {s for s in range(20) if e3_env.block_class_for_seed(s) == "A"}
    class_b_seeds = {s for s in range(20) if e3_env.block_class_for_seed(s) == "B"}
    # rotation 0 -> seeds 0-4, rotation 2 -> seeds 10-14 (class A);
    # rotation 1 -> seeds 5-9, rotation 3 -> seeds 15-19 (class B).
    assert class_a_seeds == set(range(0, 5)) | set(range(10, 15))
    assert class_b_seeds == set(range(5, 10)) | set(range(15, 20))


# ---------------------------------------------------------------------
# Documentary regression pin: rotation=1 and rotation=3, tested against
# the FULL D4 group (reflections above plus the two 90-degree rotations
# and the two diagonal reflections), match none of it -- not just the 4
# axial reflections. Confirms these two rotations are not secretly
# ANOTHER kind of exact symmetry either; they simply are not members of
# any goal-fixing rigid symmetry of this layout's dynamics.
# ---------------------------------------------------------------------

_FULL_D4_GROUP = dict(REFLECTION_GROUP)
_FULL_D4_GROUP.update(
    {
        "rot90": lambda r, c: ((-c) % N_ROWS, r % N_COLS),
        "rot270": lambda r, c: (c % N_ROWS, (-r) % N_COLS),
        "diag": lambda r, c: (c, r),
        "anti_diag": lambda r, c: ((-c) % N_ROWS, (-r) % N_COLS),
    }
)


@pytest.mark.parametrize("rotation", [1, 3])
def test_rotation_1_and_3_match_no_element_of_the_full_d4_group(identity_tensor, rotation):
    """Regression pin, not a bug: rotation=1/3 belong to C4 (a genuine
    quarter-turn group), which is simply not the relevant symmetry group
    for this layout's row-axis-locked drift convention (see module
    docstring). If this ever starts matching something, that means
    E1/E2's frozen drift construction changed -- worth flagging, not
    silently accepting."""
    Pr = _tensor_for_assignment(e3_env.quadrant_assignment_for_rotation(rotation))
    match = None
    for name, posmap in _FULL_D4_GROUP.items():
        if _transports_exactly(identity_tensor, Pr, posmap):
            match = name
            break
    assert match is None


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_controllable_zone_position_masks_transport_under_the_full_rotation_group(rotation):
    """The pure-geometry claim holds for every rotation regardless of
    dynamics: each target quadrant's cell mask is the exact rot90^rotation
    image of the corresponding source quadrant's cell mask under E2's OWN
    cyclic assignment -- confirms the counterbalancing's geometric design
    is sound; it is specifically the row-axis-only drift convention
    (relevant only for the reflection group, not the rotation group) that
    makes rotation=1/3 not a symmetry of the full DYNAMICS, not the
    geometry."""
    assignment = e3_env.quadrant_assignment_for_rotation(rotation)
    layout = e3_env.build_layout(assignment)
    rot = _FULL_D4_GROUP["rot90"]

    def rot_k(r, c, k):
        for _ in range(k):
            r, c = rot(r, c)
        return r, c

    identity_layout = e3_env.build_layout(e3_env.quadrant_assignment_for_rotation(0))
    for target_q in e3_env.QUADRANT_ORDER:
        source_q = assignment[target_q]
        target_cfg = e3_env.CFG["blocks"][target_q]
        r0, c0 = target_cfg["rows"][0], target_cfg["cols"][0]
        target_zone = layout.zone_at((r0, c0))

        source_cfg = e3_env.CFG["blocks"][source_q]
        sr0, sc0 = source_cfg["rows"][0], source_cfg["cols"][0]
        source_zone_id = identity_layout.zone_at((sr0, sc0))
        source_cells = np.argwhere(source_zone_id.mask)
        expected_target_cells = {rot_k(int(r), int(c), rotation) for r, c in source_cells}
        actual_target_cells = {(int(r), int(c)) for r, c in np.argwhere(target_zone.mask)}
        assert expected_target_cells == actual_target_cells, f"rotation={rotation} target={target_q}"
