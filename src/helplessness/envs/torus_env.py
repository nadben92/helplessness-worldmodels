"""TorusZonedEnv: the new base environment. Torus topology (no walls, no
absorbing borders), dynamics determined per-cell by a ZoneLayout, one of
two interchangeable observation backends (tabular index or pixel), reward
supplied as a pluggable RewardFn.

Implements the same Env ABC as the legacy GridWorld (reset, step,
transition_tensor, reward_vector, terminal_mask, controllability) so
everything downstream that already works via that interface (evaluate(),
value_iteration(), controllability_from_tensor()) needs no changes to run
on this environment instead -- verified directly in tests/envs/test_torus_env.py.

step() samples the next state from exactly the row transition_tensor()
would report for the same (position, action) -- both go through
zone_layout.transition_row(), so the two can never silently diverge (the
same discipline CLAUDE.md requires of the legacy environment, enforced
here by construction rather than by a separate empirical test needing to
catch a hand-written duplicate).

No terminal states by default (terminal_mask() is all-False): the zone
family is about controllability, not about goal/trap cells the way the
legacy layout was, and reward is not yet a decided structure (see
rewards/base.py) -- max_steps is a truncation-only stopping condition
until a concrete task defines what "terminal" means here. If a future
reward design introduces absorbing states, this is the method to extend
(a pluggable TerminationFn analogous to RewardFn, not a hidden default).
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.base import Env
from helplessness.envs.dynamics.base import N_ACTIONS
from helplessness.envs.observation.base import ObservationBackend
from helplessness.envs.rewards.base import RewardFn, reward_vector as _reward_vector
from helplessness.envs.zones import ZoneLayout
from helplessness.measures.controllability import controllability_from_tensor


class TorusZonedEnv(Env):
    def __init__(
        self,
        zone_layout: ZoneLayout,
        observation_backend: ObservationBackend,
        reward_fn: RewardFn,
        start_position: tuple[int, int] = (0, 0),
        max_steps: int = 100,
        seed: int | None = None,
        theme_seed: int = 0,
    ):
        self.zone_layout = zone_layout
        self.grid_shape = zone_layout.grid_shape
        self.n_rows, self.n_cols = self.grid_shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = N_ACTIONS

        self.observation_backend = observation_backend
        self.reward_fn = reward_fn
        self.start_position = (int(start_position[0]) % self.n_rows, int(start_position[1]) % self.n_cols)
        self.max_steps = max_steps
        self.theme_seed = theme_seed

        self.seed_dynamics(seed)
        self.reset()

    # ------------------------------------------------------------------
    def rc_to_s(self, rc: tuple[int, int]) -> int:
        return int(rc[0]) * self.n_cols + int(rc[1])

    def s_to_rc(self, s: int) -> tuple[int, int]:
        return divmod(int(s), self.n_cols)

    def _obs(self):
        return self.observation_backend.observe(self.position, self.zone_layout, self.theme_seed)

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def seed_dynamics(self, seed: int | None) -> None:
        """(Re)initialises the dynamics' noise generator -- kept distinct
        from any exploration generator, same invariant as the legacy env
        (see CLAUDE.md, RNG separation)."""
        self._dynamics_rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None, start_state: int | None = None, theme_seed: int | None = None):
        if seed is not None:
            self.seed_dynamics(seed)
        if theme_seed is not None:
            self.theme_seed = theme_seed
        self.position = self.start_position if start_state is None else self.s_to_rc(start_state)
        self.t = 0
        return self._obs(), {"state": self.rc_to_s(self.position)}

    def step(self, action: int):
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"invalid action: {action}")

        row = self.zone_layout.transition_row(self.position, int(action))
        s_next = int(self._dynamics_rng.choice(self.n_states, p=row))
        self.position = self.s_to_rc(s_next)
        self.t += 1

        reward = float(self.reward_fn(s_next))
        terminated = False  # see module docstring: no terminal states yet
        truncated = self.t >= self.max_steps

        info = {"state": s_next}
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # exact model
    # ------------------------------------------------------------------
    def transition_tensor(self) -> np.ndarray:
        P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                s = self.rc_to_s((r, c))
                for a in range(self.n_actions):
                    P[s, a] = self.zone_layout.transition_row((r, c), a)
        return P

    def reward_vector(self) -> np.ndarray:
        return _reward_vector(self.reward_fn, self.n_states)

    def terminal_mask(self) -> np.ndarray:
        return np.zeros(self.n_states, dtype=bool)

    def controllability(self) -> float:
        return controllability_from_tensor(self.transition_tensor(), self.terminal_mask())
