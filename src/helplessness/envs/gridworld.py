"""GridWorld with tunable controllability.

The agent moves cell by cell, over 4 directions plus an optional 5th "stay"
action (see n_actions below). At each step, with probability p, the executed
action is drawn uniformly among the n_actions available, independently of the
chosen action. The probability of executing the intended action is therefore
(1 - p) + p/n_actions, and p = 1 corresponds to a random walk where the
action has no effect at all.

Gymnasium-compatible API (reset -> (obs, info), step -> (obs, r, terminated, truncated, info))
without a mandatory dependency on gymnasium.
"""

from __future__ import annotations

import numpy as np

from helplessness.envs.base import Env
from helplessness.metrics.controllability import controllability_from_tensor

# actions: 0 = up, 1 = right, 2 = down, 3 = left, 4 = stay
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))
ACTION_NAMES = ("up", "right", "down", "left", "stay")
N_ACTIONS = 4  # default action count (4 directions); kept for backward
# compatibility with existing call sites (offline experiments/tests) that
# import this module constant instead of reading env.n_actions. Pass
# n_actions=5 to GridWorld's constructor to enable the "stay" action.

# Default layout. S = start, G = terminal reward, T = trap,
# # = wall, . = empty cell.
DEFAULT_LAYOUT = """
S....
.##T.
..#..
.T#..
....G
"""

ALLOWED_CHARS = frozenset(".#STG")


class GridWorld(Env):
    """GridWorld with tunable controllability (see module docstring).

    trap_reward: reward received upon falling into a trap. -1.0 by default
        (terminal trap: a single hit is taken). If terminal_traps=False, the
        trap no longer ends the episode and can be visited (and penalised)
        an arbitrary number of times within the same episode; a value around
        -0.3 is then more appropriate, otherwise an agent that oscillates on
        the trap accumulates a very negative return that swamps the dynamic
        range of the other measures.
    """

    def __init__(
        self,
        layout: str = DEFAULT_LAYOUT,
        p: float = 0.0,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        max_steps: int = 100,
        terminal_traps: bool = True,
        obs_mode: str = "index",  # "index" | "onehot" | "grid"
        seed: int | None = None,
        n_actions: int = N_ACTIONS,
    ):
        self.grid = np.array(
            [list(row) for row in layout.strip().splitlines()], dtype="<U1"
        )
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols

        unknown = set(np.unique(self.grid)) - ALLOWED_CHARS
        if unknown:
            raise ValueError(
                f"unknown characters in layout: {sorted(unknown)} "
                f"(allowed characters: {sorted(ALLOWED_CHARS)})"
            )

        if n_actions not in (4, 5):
            raise ValueError(f"n_actions must be 4 (no stay) or 5 (with stay), got {n_actions}")
        self.n_actions = n_actions

        self.p = float(p)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.max_steps = max_steps
        self.terminal_traps = terminal_traps
        self.obs_mode = obs_mode
        self.seed_dynamics(seed)

        starts = np.argwhere(self.grid == "S")
        if len(starts) != 1:
            raise ValueError("the layout must contain exactly one 'S'")
        self.start_rc = tuple(int(x) for x in starts[0])

        self.walls = self.grid == "#"
        self.goals = self.grid == "G"
        self.traps = self.grid == "T"

        if self.walls[self.start_rc]:
            raise ValueError("the start cell is a wall")

        self._transitions = None  # cache of the exact tensor P[s, a, s']
        self.reset()

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------
    def rc_to_s(self, rc) -> int:
        return int(rc[0]) * self.n_cols + int(rc[1])

    def s_to_rc(self, s: int):
        return divmod(int(s), self.n_cols)

    def _obs(self):
        s = self.rc_to_s(self.agent_rc)
        if self.obs_mode == "index":
            return s
        if self.obs_mode == "onehot":
            v = np.zeros(self.n_states, dtype=np.float32)
            v[s] = 1.0
            return v
        if self.obs_mode == "grid":
            # 4 channels: agent, walls, reward, traps
            g = np.zeros((4, self.n_rows, self.n_cols), dtype=np.float32)
            g[0][self.agent_rc] = 1.0
            g[1] = self.walls
            g[2] = self.goals
            g[3] = self.traps
            return g
        raise ValueError(f"unknown obs_mode: {self.obs_mode}")

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _move(self, rc, a: int):
        """Deterministic move. A wall or a border leaves the agent in place."""
        dr, dc = ACTIONS[a]
        r, c = rc[0] + dr, rc[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols):
            return rc
        if self.walls[r, c]:
            return rc
        return (r, c)

    def seed_dynamics(self, seed: int | None) -> None:
        """(Re)initialises the dynamics' noise generator.

        Only _sample_effective_action uses this generator. It is kept
        distinct from any exploration generator of the agent, so that the
        environment's noise can be held fixed while the collection policy
        varies, and vice versa.
        """
        self._dynamics_rng = np.random.default_rng(seed)

    def _sample_effective_action(self, a: int) -> int:
        """With probability p, the action is replaced by a uniform draw over
        the n_actions available (4, or 5 if "stay" is enabled)."""
        if self._dynamics_rng.random() < self.p:
            return int(self._dynamics_rng.integers(self.n_actions))
        return int(a)

    def reset(self, seed: int | None = None, start_state: int | None = None):
        """Resets the episode (start position, step counter).

        Does not touch the dynamics' noise generator. The seed parameter, if
        given, is routed explicitly to seed_dynamics -- beware, calling
        reset(seed=k) at the start of every episode then replays exactly the
        same noise stream from one episode to the next. To hold the noise
        fixed or vary it independently of the reset/step cycle, call
        seed_dynamics() directly instead of going through here.

        start_state: explicit start state (index), to evaluate a policy from
        a state other than S. None (default) keeps the historical behaviour
        (fixed start at self.start_rc) -- the choice of which states to draw
        and under what law belongs to the caller (evaluate() in
        evaluation/rollout.py), not to the environment.
        """
        if seed is not None:
            self.seed_dynamics(seed)
        self.agent_rc = self.start_rc if start_state is None else self.s_to_rc(start_state)
        self.t = 0
        return self._obs(), {"state": self.rc_to_s(self.agent_rc)}

    def step(self, action: int):
        """Executes an action, returns (obs, reward, terminated, truncated, info).

        The underlying MDP is infinite-horizon discounted; max_steps is not
        part of it, it is merely a convenience limit for looping rollouts.
        terminated=True signals a genuinely terminal state (goal, trap in
        terminal mode, wall); truncated=True signals that only max_steps was
        reached. An agent learning on this environment must bootstrap on
        V(s') when truncated=True, and treat only terminated=True as a
        terminal state -- conflating the two would teach the agent that the
        world ends at max_steps, a behavioural artefact indistinguishable
        from the helplessness phenotype under study.
        """
        if not 0 <= int(action) < self.n_actions:
            raise ValueError(f"invalid action: {action}")

        a_eff = self._sample_effective_action(int(action))
        slipped = a_eff != int(action)
        self.agent_rc = self._move(self.agent_rc, a_eff)
        self.t += 1

        reward = self.step_reward
        terminated = False
        if self.goals[self.agent_rc]:
            reward += self.goal_reward
            terminated = True
        elif self.traps[self.agent_rc]:
            reward += self.trap_reward
            terminated = self.terminal_traps

        truncated = (not terminated) and self.t >= self.max_steps
        info = {
            "state": self.rc_to_s(self.agent_rc),
            "effective_action": a_eff,
            "slipped": slipped,
        }
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # exact model -- oracle, evaluation of the learned model, interventions on p
    # ------------------------------------------------------------------
    def transition_tensor(self, p: float | None = None) -> np.ndarray:
        """Exact P[s, a, s'] for a given value of p (defaults to the env's own).

        Walls and goals are absorbing. Traps are too, unless
        terminal_traps=False (they then remain ordinary, traversable and
        revisitable cells).
        """
        p = self.p if p is None else float(p)
        P = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        for s in range(self.n_states):
            rc = self.s_to_rc(s)
            if self.walls[rc] or self.goals[rc] or (self.traps[rc] and self.terminal_traps):
                P[s, :, s] = 1.0
                continue
            for a in range(self.n_actions):
                for a_eff in range(self.n_actions):
                    prob = p / self.n_actions + (1.0 - p) * (a_eff == a)
                    if prob > 0.0:
                        P[s, a, self.rc_to_s(self._move(rc, a_eff))] += prob
        return P

    def reward_vector(self) -> np.ndarray:
        """R[s'] : reward received upon arriving at s' (simple terminal version)."""
        R = np.full(self.n_states, self.step_reward, dtype=np.float64)
        for s in range(self.n_states):
            rc = self.s_to_rc(s)
            if self.goals[rc]:
                R[s] = self.step_reward + self.goal_reward
            elif self.traps[rc]:
                R[s] = self.step_reward + self.trap_reward
        return R

    def terminal_mask(self) -> np.ndarray:
        m = np.zeros(self.n_states, dtype=bool)
        for s in range(self.n_states):
            rc = self.s_to_rc(s)
            m[s] = self.goals[rc] or self.walls[rc] or (self.traps[rc] and self.terminal_traps)
        return m

    def controllability(self, p: float | None = None) -> float:
        """Average mutual information I(a ; s' | s) in bits, over non-terminal states.

        Equals ~2 bits when the 4 actions lead to 4 distinct cells and p = 0,
        and exactly 0 when p = 1. Delegates to controllability_from_tensor.
        """
        return controllability_from_tensor(self.transition_tensor(p), self.terminal_mask())

    # ------------------------------------------------------------------
    def render(self) -> str:
        out = self.grid.copy()
        out[out == "S"] = "."
        r, c = self.agent_rc
        out[r, c] = "A"
        return "\n".join("".join(row) for row in out)
