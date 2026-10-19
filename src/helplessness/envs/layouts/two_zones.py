"""E4's environments (docs/E4_generalization_gradient.md): E3's 12x12 torus geometry (configs/zones/e1_4zones.yaml:
the cross corridor, the four 5x5 quadrant slots, the goal at the corridor intersection (0, 0)) with two zones on one
diagonal -- an uncontrollable-deterministic conveyor (UD, drift away from the goal, as E1-E3) and a
controllable-deterministic zone (CD) -- while the two slots of the other diagonal take the corridor's dynamics
(controllable-deterministic) and, in the training image, the corridor's colour.

Appearance: every quadrant slot is its own appearance category, named after the slot ("nw", "ne", "se", "sw"), and a
run's Themes maps each slot to a colour. The training image colours the UD slot T_inc, the CD slot T_ctrl and the two
other slots like the corridor; a probe changes the colour of one slot only. A slot coloured like the corridor renders
exactly like corridor cells (the renderer reads colours, not categories).

Conventions (fixed, relative to the configuration):
  DIAGONALS     diagonal 1 = (nw, se), diagonal 2 = (ne, sw); in each, UD sits at the first slot and CD at the second.
                The 180-degree turn about the goal (point reflection) maps one slot of a diagonal onto the other and
                preserves the dynamics (E3, tests/e3/test_e3_symmetry.py), so placing UD first loses no generality.
  neutral slot  the slot of the other diagonal in UD's row band (same rows, so the same drift band): nw <-> ne,
                se <-> sw.
The reference (paired control) is the experimental layout with envs.dynamics.pairing.derive_control applied to every
zone (UD -> CD), never a hand-written layout. The pixel environment is four_zones_pixel.build_pixel_env (same goal,
same rewards, same renderer).
"""

from __future__ import annotations

import numpy as np

from helplessness.config import load_yaml_block
from helplessness.envs.appearance.base import AppearanceSpec
from helplessness.envs.appearance.theme_sampling import CORRIDOR_COLOR, GOAL_COLOR, Themes
from helplessness.envs.dynamics.controllable import ControllableDeterministic
from helplessness.envs.dynamics.pairing import derive_control
from helplessness.envs.dynamics.uncontrollable import UncontrollableDeterministic
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.zones import Zone, ZoneLayout, cross_corridor_mask, resolve_drift, torus_block_mask

CFG: dict = load_yaml_block("zones/e1_4zones.yaml")
DIRECTION = "away_from_goal"
GRID_SHAPE: tuple[int, int] = tuple(CFG["grid_shape"])
GOAL_RC: tuple[int, int] = tuple(CFG["goal"])
N_STATES = GRID_SHAPE[0] * GRID_SHAPE[1]
GOAL_STATE = GOAL_RC[0] * GRID_SHAPE[1] + GOAL_RC[1]
CONDITIONS = ("experimental", "reference")
DIAGONALS: dict[int, tuple[str, str]] = {1: ("nw", "se"), 2: ("ne", "sw")}
NEUTRAL_OF: dict[str, str] = {"nw": "ne", "ne": "nw", "se": "sw", "sw": "se"}


def roles(diagonal: int) -> dict[str, str]:
    """Slot of each role for a diagonal: "ud", "cd", "neutral" (probed), "other" (the fourth slot, never probed)."""
    ud, cd = DIAGONALS[diagonal]
    neutral = NEUTRAL_OF[ud]
    other = next(q for q in QUADRANT_ORDER if q not in (ud, cd, neutral))
    return {"ud": ud, "cd": cd, "neutral": neutral, "other": other}


def quadrant_mask(quadrant: str) -> np.ndarray:
    block = CFG["blocks"][quadrant]
    return torus_block_mask(GRID_SHAPE, tuple(block["rows"]), tuple(block["cols"]))


def build_layout(diagonal: int, condition: str, ud_slot: str | None = None) -> ZoneLayout:
    """The experimental layout of a diagonal (UD at roles["ud"], every other zone controllable-deterministic), or its
    reference (derive_control on every zone of the experimental one). ud_slot (E5's per-episode assignments; None, the
    default, is the diagonal's own UD slot): the conveyor at that slot instead, its drift resolved for that slot's rows
    (away from the goal); the corridor, the goal and the other slots are unchanged."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    ud = roles(diagonal)["ud"] if ud_slot is None else ud_slot
    if ud not in QUADRANT_ORDER:
        raise ValueError(f"ud_slot must be one of {QUADRANT_ORDER}, got {ud!r}")
    corridor = CFG["corridor"]
    zones = [Zone(mask=cross_corridor_mask(GRID_SHAPE, tuple(corridor["rows"]), tuple(corridor["cols"])),
                  dynamics=ControllableDeterministic(), appearance=AppearanceSpec(category=corridor["appearance"]))]
    for q in QUADRANT_ORDER:
        if q == ud:
            dynamics = UncontrollableDeterministic(drift=resolve_drift(tuple(CFG["blocks"][q]["rows"]), GRID_SHAPE, GOAL_RC, DIRECTION))
        else:
            dynamics = ControllableDeterministic()
        zones.append(Zone(mask=quadrant_mask(q), dynamics=dynamics, appearance=AppearanceSpec(category=q)))
    experimental = ZoneLayout(GRID_SHAPE, zones)
    if condition == "experimental":
        return experimental
    return ZoneLayout(GRID_SHAPE, [Zone(mask=z.mask, dynamics=derive_control(z.dynamics), appearance=z.appearance)
                                   for z in experimental.zones])


def training_colors(diagonal: int, t_inc, t_ctrl, ud_slot: str | None = None, cd_slot: str | None = None) -> dict[str, tuple[int, int, int]]:
    """Slot -> colour of the training image: UD slot T_inc, CD slot T_ctrl, the other two the corridor's colour.
    ud_slot / cd_slot (E5's assignments; None: the diagonal's own slots): T_inc / T_ctrl at those slots instead."""
    r = roles(diagonal)
    ud = r["ud"] if ud_slot is None else ud_slot
    cd = r["cd"] if cd_slot is None else cd_slot
    if ud == cd:
        raise ValueError(f"the UD and CD slots must differ, got {ud!r} twice")
    colors = {q: tuple(int(v) for v in CORRIDOR_COLOR) for q in QUADRANT_ORDER}
    colors[ud] = tuple(int(v) for v in t_inc)
    colors[cd] = tuple(int(v) for v in t_ctrl)
    return colors


def themes(slot_colors: dict[str, tuple[int, int, int]], agent_color) -> Themes:
    """A Themes object for a slot -> colour map, with E4's fixed agent-marker colour and E3's corridor/goal colours."""
    if set(slot_colors) != set(QUADRANT_ORDER):
        raise ValueError(f"slot_colors must give the four slots {QUADRANT_ORDER}, got {sorted(slot_colors)}")
    return Themes(zone_colors={q: tuple(int(v) for v in slot_colors[q]) for q in QUADRANT_ORDER},
                  agent_marker_color=tuple(int(v) for v in agent_color), goal_state=GOAL_STATE,
                  corridor_color=CORRIDOR_COLOR, goal_color=GOAL_COLOR)


def region_of_state(layout: ZoneLayout, s: int) -> str:
    """"corridor" or the slot name of a state."""
    r, c = divmod(int(s), GRID_SHAPE[1])
    return layout.zone_at((r, c)).appearance.category


def region_states(layout: ZoneLayout) -> dict[str, np.ndarray]:
    """Region -> its states (the goal is in the corridor region)."""
    regions = np.array([region_of_state(layout, s) for s in range(N_STATES)])
    return {name: np.flatnonzero(regions == name) for name in ("corridor", *QUADRANT_ORDER)}


def corridor_states(layout: ZoneLayout) -> np.ndarray:
    """The corridor's states without the goal."""
    states = region_states(layout)["corridor"]
    return states[states != GOAL_STATE]


def toric_distance_to_goal(s: int) -> int:
    """Manhattan distance on the torus from a state to the goal."""
    r, c = divmod(int(s), GRID_SHAPE[1])
    dr = min(abs(r - GOAL_RC[0]), GRID_SHAPE[0] - abs(r - GOAL_RC[0]))
    dc = min(abs(c - GOAL_RC[1]), GRID_SHAPE[1] - abs(c - GOAL_RC[1]))
    return dr + dc
