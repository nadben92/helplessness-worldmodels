"""E5's episode sources (docs/E5_belief_training.md, docs/e5/DECISIONS.md): the environment of each episode of a
closed-loop run, chosen at the episode's reset (agents.worldmodel.closed_loop, episode_source; pipeline.agent_run,
ClosedLoopHooks.episode_source). Never within an episode: no episode mixes two environments.

  ZoneAssignmentShuffle  at each episode, with probability `fraction`, a zone assignment drawn uniformly among the 12
                         ordered (UD slot, CD slot) pairs of the four quadrant slots; otherwise the configuration's
                         canonical one. The theme follows the type (T_inc on the UD slot, T_ctrl on the CD slot, the
                         corridor's colour on the two neutral slots); the conveyor's drift is resolved for its slot;
                         corridor, goal and torus never change. In the reference, the same draws (the assignment stream,
                         streams.assignment, shared by episode index) on the reference's dynamics (UD -> CD, T_inc kept).
                         The replay keeps every assignment's images (one block of n_states rows per assignment) and the
                         background of each, so the loss's mask stays the agent's patch.
  DynamicsSwitch         the immunisation: the `before` environment until the first episode that starts once
                         `switch_step` transitions are done, the `after` environment from that episode on (the same
                         images, so the same replay table). The step it actually happens (dynamics_switch_step) sets the
                         run's end (+ post_steps) and its checkpoints (+ each grid point below post_steps).
"""

from __future__ import annotations

import numpy as np
import torch

from helplessness.envs.layouts import two_zones
from helplessness.envs.layouts.four_zones import QUADRANT_ORDER
from helplessness.envs.zones import resolve_drift
from helplessness.pipeline import two_zones as two_zones_world

# the 12 assignments: (UD slot, CD slot), the two other slots neutral; fixed order (index = position in this tuple)
ASSIGNMENTS: tuple[tuple[str, str], ...] = tuple((ud, cd) for ud in QUADRANT_ORDER for cd in QUADRANT_ORDER if cd != ud)


def assignment_index(ud: str, cd: str) -> int:
    return ASSIGNMENTS.index((ud, cd))


def assignment_roles(index: int) -> dict[str, str]:
    """Slot -> role ("ud", "cd", "neutral") of an assignment."""
    ud, cd = ASSIGNMENTS[index]
    return {q: "ud" if q == ud else "cd" if q == cd else "neutral" for q in QUADRANT_ORDER}


def canonical_assignment(diagonal: int) -> int:
    r = two_zones.roles(diagonal)
    return assignment_index(r["ud"], r["cd"])


def assignment_colors(index: int, diagonal: int, t_inc, t_ctrl) -> dict[str, tuple[int, int, int]]:
    ud, cd = ASSIGNMENTS[index]
    return two_zones.training_colors(diagonal, t_inc, t_ctrl, ud_slot=ud, cd_slot=cd)


# the goal-centred axial reflections of the torus (tests/e3/test_e3_symmetry.py's REFLECTION_GROUP), as maps of the
# quadrant slots; and the quarter turns about the goal (E3's rotations, nw -> ne -> se -> sw)
REFLECTIONS: dict[str, dict[str, str]] = {
    "identity": {"nw": "nw", "ne": "ne", "se": "se", "sw": "sw"},
    "col_reflect": {"nw": "ne", "ne": "nw", "se": "sw", "sw": "se"},
    "row_reflect": {"nw": "sw", "sw": "nw", "ne": "se", "se": "ne"},
    "point_reflect": {"nw": "se", "se": "nw", "ne": "sw", "sw": "ne"},
}
ROTATION_ORDER = ("nw", "ne", "se", "sw")
SYMMETRY_CLASSES = ("A", "B")


def assignment_symmetry(index: int, diagonal: int) -> dict:
    """How an assignment relates to the configuration's canonical one (docs/e5/DECISIONS.md (24)).

    reflection      the goal-centred axial reflection that sends the canonical UD slot to this one. Only the UD slot's
                    dynamics differ from the corridor's (CD and the neutral slots are controllable-deterministic alike),
                    so this reflection transports the canonical environment's dynamics exactly onto this assignment's,
                    the conveyor's drift included (tested for the 12, both conditions): every assignment is the
                    canonical one up to that symmetry, for the dynamics; the images differ (where T_ctrl sits).
    rotation_class  E3's block factor (envs.layouts.four_zones_pixel.ROTATION_CLASS) of the quarter turn about the goal
                    that sends the canonical UD slot to this one: "A" for 0 or 180 degrees, "B" for 90 or 270.
    ud_cd           where the CD slot sits relative to the UD slot: "diagonal" (opposite slots, as in the canonical
                    layouts), "same_rows" (same row band), "same_columns" (same column band).
    drift           the conveyor's (row, column) drift, away from the goal, resolved for the UD slot's rows."""
    ud, cd = ASSIGNMENTS[index]
    canon_ud = two_zones.roles(diagonal)["ud"]
    reflection = next(name for name, m in REFLECTIONS.items() if m[canon_ud] == ud)
    turns = (ROTATION_ORDER.index(ud) - ROTATION_ORDER.index(canon_ud)) % 4
    if REFLECTIONS["point_reflect"][ud] == cd:
        relation = "diagonal"
    elif REFLECTIONS["col_reflect"][ud] == cd:
        relation = "same_rows"
    else:
        relation = "same_columns"
    drift = resolve_drift(tuple(two_zones.CFG["blocks"][ud]["rows"]), two_zones.GRID_SHAPE, two_zones.GOAL_RC, two_zones.DIRECTION)
    return {"index": int(index), "ud": ud, "cd": cd, "reflection": reflection, "rotation_class": SYMMETRY_CLASSES[turns % 2],
            "quarter_turns": int(turns), "ud_cd": relation, "drift": [int(drift[0]), int(drift[1])]}


def assignment_world(index: int, conf: dict, condition: str, render_cfg: dict, episode: dict, seed: int):
    """The pixel world of one assignment (pipeline.world.World): its colours and, in the experimental condition, the
    conveyor at its UD slot; the reference's dynamics otherwise. conf: a two-zone configuration
    (pipeline.two_zones_setup.configuration) -- its colours, diagonal and agent colour."""
    ud, _ = ASSIGNMENTS[index]
    return two_zones_world.build_world(assignment_colors(index, conf["diagonal"], conf["t_inc"], conf["t_ctrl"]), conf["agent_color"],
                                       conf["diagonal"], condition, render_cfg, episode, seed, ud_slot=ud)


def background_uint8(world) -> np.ndarray:
    """(H, W, C) uint8, the world's agent-free background (World.background_t is that image / 255, exact)."""
    return (world.background_t.detach().cpu().permute(1, 2, 0) * 255.0).round().to(torch.uint8).numpy()


class ZoneAssignmentShuffle:
    """worlds: one World per assignment, in ASSIGNMENTS order (the canonical one is the run's own world, so that its
    environment is the very object of a run without shuffle). rng: streams.assignment.assignment_generator(seed).

    Every episode draws u ~ U[0, 1) then k ~ U{0..11} (both, always, so the stream advances by episode whatever is
    drawn); the episode's assignment is k when u < fraction (a "drawn" episode), the canonical one otherwise. The
    per-episode log (episode_start: transitions done at its reset, episode_assignment, episode_drawn) is kept."""

    def __init__(self, worlds: list, canonical: int, fraction: float, rng: np.random.Generator, n_states: int = two_zones.N_STATES):
        if len(worlds) != len(ASSIGNMENTS):
            raise ValueError(f"one world per assignment ({len(ASSIGNMENTS)}), got {len(worlds)}")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self.envs = [w.env for w in worlds]
        self.canonical, self.fraction, self.rng, self.n_states = int(canonical), float(fraction), rng, int(n_states)
        self.table = np.concatenate([w.templates_np for w in worlds])
        self.background_table = np.stack([background_uint8(w) for w in worlds])
        self.background_of_row = np.repeat(np.arange(len(worlds)), self.n_states)
        self.episode_start: list[int] = []
        self.episode_assignment: list[int] = []
        self.episode_drawn: list[bool] = []

    def __getstate__(self) -> dict:  # the tables live in the replay; the resume state keeps the source light
        return {k: v for k, v in self.__dict__.items() if k not in ("table", "background_table", "background_of_row")}

    def draw(self) -> tuple[int, bool]:
        u, k = float(self.rng.random()), int(self.rng.integers(len(ASSIGNMENTS)))
        drawn = u < self.fraction
        return (k if drawn else self.canonical), drawn

    def next_episode(self, env_steps_done: int) -> tuple[object, int]:
        k, drawn = self.draw()
        self.episode_start.append(int(env_steps_done))
        self.episode_assignment.append(k)
        self.episode_drawn.append(bool(drawn))
        return self.envs[k], k * self.n_states

    @property
    def current(self) -> int | None:
        return self.episode_assignment[-1] if self.episode_assignment else None

    def log_arrays(self) -> dict[str, np.ndarray]:
        """The per-episode log; episode_rotation_class: 0 = "A", 1 = "B" (assignment_symmetry, relative to the canonical
        assignment) so that the analysis can split by symmetry class."""
        diagonal = next(d for d in two_zones.DIAGONALS if canonical_assignment(d) == self.canonical)
        cls = np.array([SYMMETRY_CLASSES.index(assignment_symmetry(k, diagonal)["rotation_class"]) for k in range(len(ASSIGNMENTS))],
                       dtype=np.int8)
        assignment = np.asarray(self.episode_assignment, dtype=np.int8)
        return {"episode_start": np.asarray(self.episode_start, dtype=np.int64), "episode_assignment": assignment,
                "episode_drawn": np.asarray(self.episode_drawn, dtype=bool), "episode_rotation_class": cls[assignment.astype(np.int64)]}


class DynamicsSwitch:
    """before / after: the two worlds (the same images, checked). switch_step: the nominal step; the switch happens at
    the reset of the first episode that starts once that many transitions are done (dynamics_switch_step = the
    transitions done then). post_steps: the run's length after it; post_grid: the checkpoint grid in post-switch steps
    (the points below post_steps, and post_steps itself). No checkpoint before the switch (the run equals its
    reference there)."""

    background_table = None
    background_of_row = None

    def __init__(self, before, after, switch_step: int, post_steps: int, post_grid: list[int]):
        if not np.array_equal(before.templates_np, after.templates_np):
            raise ValueError("the immunisation's two environments must render identically")
        self.before_env, self.after_env = before.env, after.env
        self.switch_step, self.post_steps = int(switch_step), int(post_steps)
        self.post_grid = sorted(int(g) for g in post_grid)
        self.table = before.templates_np
        self.dynamics_switch_step: int | None = None
        self._pending = False

    def __getstate__(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "table"}

    @property
    def switched(self) -> bool:
        return self.dynamics_switch_step is not None

    def checkpoints(self, n_env_steps: int) -> list[int]:
        return []

    def post_checkpoints(self, effective: int) -> list[int]:
        end = effective + self.post_steps
        return sorted({effective + g for g in self.post_grid if 0 < g < self.post_steps} | {end})

    def next_episode(self, env_steps_done: int) -> tuple[object, int]:
        if not self.switched and env_steps_done >= self.switch_step:
            self.dynamics_switch_step, self._pending = int(env_steps_done), True
        return (self.after_env if self.switched else self.before_env), 0

    def schedule_update(self) -> tuple[int, list[int]] | None:
        if not self._pending:
            return None
        self._pending = False
        return self.dynamics_switch_step + self.post_steps, self.post_checkpoints(self.dynamics_switch_step)
