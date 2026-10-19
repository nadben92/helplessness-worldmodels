"""Closed-loop data collection (spec section 4 and condition A of section
6; Phase 5): the agent acts in the environment with its own policy
(policy.py), one environment step at a time, and stores what it sees.

REPLAY CONVENTION (docs/e3/DECISIONS.md, 2026-09-30 (5)). A stored step is an
OBSERVATION with what was received on reaching it:

    x_t        the observation
    a_t        the action taken from it (0 for an episode's last observation: never used, the next step is a reset)
    r_t        the reward received ON ARRIVING at x_t (0 for an episode's first observation)
    terminal_t x_t is the goal
    truncated_t the episode was cut after x_t (H reached, x_t not terminal)
    is_first_t x_t starts an episode

so the goal's observation IS stored (terminal, r = goal reward): the goal
head's target 1[r_t > 0] then bears on the posterior state of the goal, and
the world model sees the transition into it. E3-0's fixed-policy collection
(loop.collect_fixed_policy_data) stores the reward of the transition LEAVING
x_t and never the terminal observation; it is left unchanged. An episode of
n transitions therefore stores n + 1 steps: the replay capacity of a
closed-loop run is budgeted for it (stored_steps_bound).

The true state label s is stored for logging and for the tabular twin's
counts N[s, a, s'] (condition B, spec 6), kept here from the environment's
own transitions; it never reaches the model, the heads or the policy.

EPISODE SOURCE (E5, 2026-10-19; off by default). With episode_source, the
environment of each episode is the source's choice, made at the episode's
reset and never within an episode: next_episode(env_steps_done) returns
(env, obs_offset), env_steps_done being the transitions taken so far. The
replay then stores each observation's table row as obs_offset + s (E5's
per-episode zone assignments: one block of n_states rows per appearance).
The episode boundary is the replay's is_first flag, as always: the model
resets there, so no window ever carries a transition from one episode's
environment into another's. Without a source, nothing changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead
from helplessness.agents.worldmodel.loop import draw_start_state, non_terminal_states
from helplessness.agents.worldmodel.policy import LatentFilter, act, q_values
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM


def stored_steps_bound(n_env_steps: int) -> int:
    """The most steps n_env_steps transitions can store: every episode
    stores one more observation than it has transitions, and has at least
    one transition."""
    return 2 * int(n_env_steps)


@dataclass
class EpisodeStats:
    """Running behavioural counts of a closed-loop run (spec section 10)."""

    env_steps: int = 0
    episodes: int = 0
    successes: int = 0
    greedy_steps: int = 0
    tied_greedy_steps: int = 0
    explored_steps: int = 0
    lengths: list[int] = field(default_factory=list)
    reached_goal: list[bool] = field(default_factory=list)

    def window(self, n_episodes: int) -> dict[str, float]:
        """Success rate and mean length over the last n_episodes finished episodes."""
        if not self.lengths:
            return {"success_rate": float("nan"), "episode_length": float("nan"), "n_episodes": 0}
        lengths, goals = self.lengths[-n_episodes:], self.reached_goal[-n_episodes:]
        return {"success_rate": float(np.mean(goals)), "episode_length": float(np.mean(lengths)), "n_episodes": len(lengths)}


class ClosedLoopCollector:
    """Plays the agent's policy in `env` and fills `replay`. step() is ONE
    environment transition (a reset is never counted as a step). Streams
    (kept separate, CLAUDE.md): start_rng (episode starts), explore_rng and
    tiebreak_rng (the policy's epsilon and tie-break, E2's), act_generator
    (the filter's posterior draws and the lookahead's prior draws)."""

    def __init__(
        self, env, replay: ReplayBuffer, rssm: RSSM, goal_head: GoalHead, critic: Critic, cfg: AgentConfig, n_states: int,
        start_rng: np.random.Generator, explore_rng: np.random.Generator, tiebreak_rng: np.random.Generator,
        act_generator: torch.Generator | None, device: torch.device, start_states: np.ndarray | None = None,
        episode_source=None,
    ):
        self.env, self.replay, self.rssm, self.goal_head, self.critic, self.cfg = env, replay, rssm, goal_head, critic, cfg
        self.start_rng, self.explore_rng, self.tiebreak_rng = start_rng, explore_rng, tiebreak_rng
        self.act_generator, self.device = act_generator, device
        self.start_states = non_terminal_states(env) if start_states is None else np.asarray(start_states)
        self.filter = LatentFilter(rssm, device, act_generator)
        self.N = np.zeros((n_states, env.n_actions, n_states), dtype=np.int64)
        self.stats = EpisodeStats()
        self._obs = None
        self._s = None
        self._arrival_reward = 0.0
        self._is_first = True
        self._episode_length = 0
        self.episode_source = episode_source
        self._obs_offset = 0

    def _obs_kwargs(self, s: int) -> dict:
        return {} if self.episode_source is None else {"obs_index": self._obs_offset + int(s)}

    def _reset(self) -> None:
        if self.episode_source is not None:
            self.env, self._obs_offset = self.episode_source.next_episode(self.stats.env_steps)
        self._obs, info = self.env.reset(start_state=draw_start_state(self.start_states, self.start_rng))
        self._s, self._arrival_reward, self._is_first, self._episode_length = info["state"], 0.0, True, 0

    def step(self, estimator=None) -> dict:
        """One transition under the agent's policy. estimator: the
        posterior estimator of the world model's current training phase
        (loop.SeparationSwitch.estimator()), so the filter reads the
        latent as training does. Returns the transition (s, a, r, s_next,
        terminated, truncated) and the choice's flags."""
        if self._obs is None:
            self._reset()
        was_training = self.rssm.training
        self.rssm.eval()
        x = torch.as_tensor(self._obs, device=self.device).to(torch.float32).permute(2, 0, 1) / 255.0
        h, z = self.filter.observe(x, self._is_first, estimator=estimator)
        Q = q_values(self.rssm, self.goal_head, self.critic, h, z, self.cfg.k_act, self.cfg, self.act_generator)[0]
        self.rssm.train(was_training)
        choice = act(Q, self.cfg, self.explore_rng, self.tiebreak_rng)
        a = choice["action"]
        self.filter.set_action(a)

        s = self._s
        self.replay.add(x=self._obs, a=a, r=self._arrival_reward, terminal=False, truncated=False, is_first=self._is_first, s=s,
                        **self._obs_kwargs(s))
        obs_next, r, terminated, truncated, info = self.env.step(a)
        s_next = info["state"]
        self.N[s, a, s_next] += 1
        self.stats.env_steps += 1
        self.stats.explored_steps += int(choice["explored"])
        self.stats.greedy_steps += int(not choice["explored"])
        self.stats.tied_greedy_steps += int(choice["tied"] and not choice["explored"])
        self._episode_length += 1

        if terminated or truncated:
            # The episode's last observation, with the reward of reaching it; no action is taken from it.
            self.replay.add(x=obs_next, a=0, r=r, terminal=bool(terminated), truncated=bool(truncated and not terminated),
                            is_first=False, s=s_next, **self._obs_kwargs(s_next))
            self.stats.episodes += 1
            self.stats.successes += int(terminated)
            self.stats.lengths.append(self._episode_length)
            self.stats.reached_goal.append(bool(terminated))
            self._obs = None  # the next step() resets
        else:
            self._obs, self._s, self._arrival_reward, self._is_first = obs_next, s_next, float(r), False
        return {"s": s, "a": a, "r": float(r), "s_next": s_next, "terminated": bool(terminated), "truncated": bool(truncated),
                "tied": choice["tied"], "explored": choice["explored"]}
