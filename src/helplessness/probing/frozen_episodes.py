"""Episodes of a FROZEN world-model agent in a test environment, measured at every step (E6's protocol, E7's
interventions; docs/E6_phenotypes_latent.md, docs/e6/DECISIONS.md). New module (2026-10-23): it calls the RSSM's own
methods (observation_step, imagination_step, decode), the policy's own Q (agents.worldmodel.policy.q_values, tie_set),
E4's one-step controllability (probing.fixed_trajectory.step_controllability) and E4's localisation
(probe_validity.decoded_agent_cell, template matching) -- never a reimplementation. Nothing is learned.

An episode (batch of episodes, one row each; a row leaves the batch when its episode ends):
  t = 0     reset: the learned initial state (is_first), the start state's image observed;
  each t    h_t, z_t = observation_step(h_{t-1}, z_{t-1}, a_{t-1}, x(s_t)) -- the posterior draw is straight-through, as
            the training's filter after the estimator switch; then the optional INTERVENTION, h_t <- hook(...): the
            posterior z_t was drawn from the unmodified h_t (z is recomputed from the image, never modified), the
            modified h_t is the one the measures, the policy and the recurrence (h_{t+1} = GRU(h_t, z_t, a_t)) use;
            the measures at (h_t, z_t) (`measure`): I_s (16 prior draws per action decoded and matched, E4's probe
            rule), the mass on the true successor of every action, the localisation (the decoded (h_t, z_t) nearest
            template, and the cell nearest to the agent's colour), p_goal(h_t, z_t), the Q values and their spread;
            the action: epsilon-greedy on Q (agents.worldmodel.policy.q_values, the training's K_act draws), the tie
            set of the training's tolerance (policy.tie_set); two test-time options of the decision rule
            (docs/e6/DECISIONS.md (52)), both off by default: `tau`, the action drawn from softmax(Q / tau) by the inverse
            CDF of the step's explore uniform u (no epsilon branch), and `q_fn`, another Q (E7's k-step lookahead);
  end       the goal (terminated) or H transitions (truncated: max_steps is a loop limit).

Random streams (docs/e6/DECISIONS.md (36)): per EPISODE a numpy generator (explore_seqs[i]) draws, at every step and
whatever the model, u ~ U[0, 1) (explore if u < epsilon), a uniform action and one uniform score per action (the tie-break:
the highest score in the tie set), so two models given the same seeds see the same starts and the same epsilon draws
step by step; the torch draws are split in two generators, act (the posterior and the lookahead's prior draws) and
measure (I_s's prior draws), so measuring never changes the behaviour. The training's act() consumes its tie-break stream
only on greedy steps; here every step consumes the same amount -- the same law, not the same draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead
from helplessness.agents.worldmodel.policy import q_values, tie_set
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.measures.decision_rule import softmax_choice
from helplessness.probing.fixed_trajectory import step_controllability, template_distances_torch
from helplessness.probing.probe_validity import decoded_agent_cell

# hook(t, h, z, states, in_zone, rows) -> (h_new, info): t the step, h (m, h_dim) and z (m, V, C) the filtered state of
# the m active rows, states (m,) their true states, in_zone (m,) bool, rows (m,) their episode indices; info: name ->
# (m,) array recorded at (rows, t).
Hook = Callable[[int, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray], tuple[torch.Tensor, dict]]
# q_fn(h, z, states, act_generator) -> Q (m, n_actions): the Q the policy reads in place of policy.q_values.
QFn = Callable[[torch.Tensor, torch.Tensor, np.ndarray, torch.Generator], torch.Tensor]


@dataclass
class FrozenAgent:
    rssm: RSSM
    goal_head: GoalHead
    critic: Critic
    cfg: AgentConfig
    path: str = ""


def load_frozen_agent(path, device: torch.device) -> FrozenAgent:
    """An agent checkpoint (agent_env<N>.pt: rssm, goal head, critic, agent config), every parameter frozen, eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    rssm = RSSM(RSSMConfig(**state["rssm_config"]))
    rssm.load_state_dict(state["rssm"])
    cfg = AgentConfig(**state["agent_config"])
    goal_head = GoalHead(rssm.cfg.h_dim, rssm.cfg.latent_dim, cfg.hidden_dim)
    critic = Critic(rssm.cfg.h_dim, rssm.cfg.latent_dim, cfg.hidden_dim)
    goal_head.load_state_dict(state["goal_head"])
    critic.load_state_dict(state["critic"])
    for module in (rssm, goal_head, critic):
        module.to(device).eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return FrozenAgent(rssm, goal_head, critic, cfg, str(path))


class TestWorld:
    """What the episodes need of a test environment: the state images (templates_t (n_states, C, H, W) float in
    [0, 1]), the deterministic successors (succ (n_states, n_actions), from the environment's transition_tensor, which
    the library keeps consistent with step()), the goal, the agent's colour and the pixels per cell."""

    def __init__(self, templates_t: torch.Tensor, succ: np.ndarray, goal_state: int, agent_color, kpx: int, device: torch.device):
        self.templates_t = templates_t.to(device)
        self.succ = np.asarray(succ)
        self.goal_state = int(goal_state)
        self.agent = torch.tensor(agent_color, dtype=torch.float32, device=device)[:, None, None] / 255.0
        self.kpx = int(kpx)
        self.flat64 = self.templates_t.reshape(self.templates_t.shape[0], -1).double()
        self.sq = (self.flat64 ** 2).sum(1)


def localise(rssm: RSSM, world: TestWorld, h: torch.Tensor, z: torch.Tensor) -> dict[str, np.ndarray]:
    """E4's localisation of (h, z): the state whose image is nearest to the reconstruction (the kernel's rule) and the
    cell nearest to the agent's colour (direct), as probe_validity.reconstruction_diagnostic."""
    with torch.no_grad():
        dec = rssm.decode(h, z)
        nearest = template_distances_torch(dec, world.flat64, world.sq).argmin(axis=1)
        cell = decoded_agent_cell(dec.clamp(0, 1), world.agent, world.kpx)
    return {"nearest_template": nearest, "agent_cell": cell}


def measure_contexts(agent: FrozenAgent, world: TestWorld, h: torch.Tensor, z: torch.Tensor, states: np.ndarray, draws: int,
                     margin: float, generator: torch.Generator, action_kernel: bool = False) -> dict[str, np.ndarray]:
    """Every per-step measure at the contexts (h, z) whose true states are `states`. action_kernel: also argmax_true (n, A),
    whether the I_s kernel's argmax cell for each action is its true successor (NaN where every draw is a ghost); the
    kernel is the one I is computed from (the same draws)."""
    sc = step_controllability(agent.rssm, h, z, world.templates_t, draws, margin, generator=generator,
                              true_successor=world.succ[states], return_kernel=action_kernel)
    extra = {}
    if action_kernel:
        p_hat = sc["p_hat"]
        ok = np.isfinite(p_hat).all(axis=-1)
        hit = np.nan_to_num(p_hat, nan=-1.0).argmax(axis=-1) == world.succ[states]
        extra["argmax_true"] = np.where(ok, hit.astype(float), np.nan)
    loc = localise(agent.rssm, world, h, z)
    with torch.no_grad():
        p_goal = agent.goal_head.p_goal(h, z).cpu().numpy()
    return {"I": sc["I"], "mass_true": sc["mass_true"], "ghost": np.nanmean(sc["ghost_fraction"], axis=1),
            "loc_template": loc["nearest_template"] == states, "loc_agent": loc["agent_cell"] == states, "p_goal": p_goal,
            "decoded_template": loc["nearest_template"], "decoded_cell": loc["agent_cell"], **extra}


def fresh_start_contexts(rssm: RSSM, world: TestWorld, states: np.ndarray, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The context of the first step of an episode started at each of `states`: is_first, the start image observed, the
    posterior drawn straight-through (as run_episodes at t = 0)."""
    n = len(states)
    device = world.templates_t.device
    h, z = rssm.initial_state(n, device)
    a = torch.zeros(n, rssm.cfg.n_actions, device=device)
    with torch.no_grad():
        out = rssm.observation_step(h, z, a, world.templates_t[torch.as_tensor(states, device=device)],
                                    torch.ones(n, dtype=torch.bool, device=device), generator=generator)
    return out["h"], out["z"]


def fresh_start_controllability(agent: FrozenAgent, world: TestWorld, states: np.ndarray, n_samples: int, draws: int, margin: float,
                                seed: int) -> np.ndarray:
    """I_s at the first step of an episode started at each state, the mean over n_samples posterior draws (one generator
    of `seed` for the posterior and the prior draws). (len(states),)."""
    gen = torch.Generator(device=world.templates_t.device).manual_seed(int(seed))
    rep = np.repeat(np.asarray(states), n_samples)
    h, z = fresh_start_contexts(agent.rssm, world, rep, gen)
    I = step_controllability(agent.rssm, h, z, world.templates_t, draws, margin, generator=gen)["I"]
    return np.nanmean(I.reshape(len(states), n_samples), axis=1)


def explore_sequences(seed_entropy: list[int], n_episodes: int) -> list[np.random.SeedSequence]:
    """One SeedSequence per episode, spawned from SeedSequence(seed_entropy): episode i's stream never depends on the
    model, on the other episodes nor on the batch."""
    return np.random.SeedSequence([int(v) for v in seed_entropy]).spawn(n_episodes)


FIELDS_STEP = ("I", "mass_played", "ghost", "loc_template", "loc_agent", "p_goal", "q_gap", "explored", "tied", "decoded_template",
               "decoded_cell")


def run_episodes(agent: FrozenAgent, world: TestWorld, starts: np.ndarray, H: int, epsilon: float,
                 explore_seqs: list[np.random.SeedSequence], act_seed: int, measure_seed: int, zone: np.ndarray,
                 draws: int = 16, margin: float = 1e-3, measure: str = "all", hook: Hook | None = None, tau: float | None = None,
                 q_fn: QFn | None = None, action_kernel: bool = False) -> dict[str, np.ndarray]:
    """The episodes from `starts` (n,), at most H transitions each (module docstring). measure: "all" (every step), "zone"
    (the steps whose state is in `zone` and t = 0), "first" (t = 0 only); unmeasured steps are NaN.
    Returns: states (n, H + 1) and actions (n, H) (-1 beyond the end), length, terminated, and per step (n, H): I,
    mass_played (the mass on the true successor of the action taken), ghost, loc_template, loc_agent, decoded_template and
    decoded_cell (the decoded positions, states as floats, NaN where unmeasured), p_goal, q_gap (max -
    min Q), explored, tied, plus the hook's fields; Q (n, H, n_actions). tau: None (epsilon-greedy, E6) or the softmax's
    temperature; q_fn: None (policy.q_values with the act generator, E6) or another Q. action_kernel (off by default; the
    other arrays unchanged): also mass_true_all and argmax_true (n, H, A), every action's mass on its true successor and
    whether the kernel's argmax is it (docs/e6/DECISIONS.md (59))."""
    rssm, cfg = agent.rssm, agent.cfg
    device = world.templates_t.device
    n, A = len(starts), rssm.cfg.n_actions
    zone = np.asarray(zone, dtype=bool)
    act_gen = torch.Generator(device=device).manual_seed(int(act_seed))
    meas_gen = torch.Generator(device=device).manual_seed(int(measure_seed))
    explore = [np.random.default_rng(s) for s in explore_seqs]
    states = np.full((n, H + 1), -1, dtype=np.int64)
    actions = np.full((n, H), -1, dtype=np.int64)
    out = {f: np.full((n, H), np.nan) for f in FIELDS_STEP}
    Q_all = np.full((n, H, A), np.nan)
    per_action = {f: np.full((n, H, A), np.nan) for f in ("mass_true_all", "argmax_true")} if action_kernel else {}
    extra: dict[str, np.ndarray] = {}
    length = np.zeros(n, dtype=np.int64)
    terminated = np.zeros(n, dtype=bool)
    states[:, 0] = starts
    h, z = rssm.initial_state(n, device)
    a_prev = torch.zeros(n, A, device=device)
    rows = np.arange(n)
    for t in range(H):
        s = states[rows, t]
        x = world.templates_t[torch.as_tensor(s, device=device)]
        first = torch.full((len(rows),), t == 0, dtype=torch.bool, device=device)
        with torch.no_grad():
            obs = rssm.observation_step(h, z, a_prev, x, first, generator=act_gen)
        h, z = obs["h"], obs["z"]
        if hook is not None:
            h, info = hook(t, h, z, s, zone[s], rows)
            for name, values in info.items():
                extra.setdefault(name, np.full((n, H), np.nan))[rows, t] = np.asarray(values, dtype=float)
        sel = {"all": np.ones(len(rows), bool), "zone": zone[s] | (t == 0), "first": np.full(len(rows), t == 0)}[measure]
        if sel.any():
            idx = torch.as_tensor(np.flatnonzero(sel), device=device)
            m = measure_contexts(agent, world, h[idx], z[idx], s[sel], draws, margin, meas_gen, action_kernel)
            r = rows[sel]
            for name in ("I", "ghost", "loc_template", "loc_agent", "p_goal", "decoded_template", "decoded_cell"):
                out[name][r, t] = m[name]
            mass_t = m["mass_true"]
            if action_kernel:
                per_action["mass_true_all"][r, t], per_action["argmax_true"][r, t] = mass_t, m["argmax_true"]
        Q = q_values(rssm, agent.goal_head, agent.critic, h, z, cfg.k_act, cfg, act_gen) if q_fn is None else q_fn(h, z, s, act_gen)
        tied_set = tie_set(Q, cfg).cpu().numpy()
        Qn = Q.cpu().numpy()
        a = np.empty(len(rows), dtype=np.int64)
        for j, i in enumerate(rows):
            u, uniform, scores = explore[i].random(), explore[i].integers(A), explore[i].random(A)
            if tau is None:
                explored = u < epsilon
                a[j] = uniform if explored else int(np.argmax(np.where(tied_set[j], scores, -1.0)))
            else:
                explored = False
                a[j] = softmax_choice(Qn[j], tau, u)
            out["explored"][i, t] = explored
        out["tied"][rows, t] = tied_set.sum(axis=1) > 1
        out["q_gap"][rows, t] = Qn.max(axis=1) - Qn.min(axis=1)
        Q_all[rows, t] = Qn
        if sel.any():
            out["mass_played"][rows[sel], t] = mass_t[np.arange(sel.sum()), a[sel]]
        actions[rows, t] = a
        s_next = world.succ[s, a]
        states[rows, t + 1] = s_next
        length[rows] = t + 1
        done = s_next == world.goal_state
        terminated[rows[done]] = True
        keep = ~done
        rows = rows[keep]
        if rows.size == 0:
            break
        keep_t = torch.as_tensor(np.flatnonzero(keep), device=device)
        h, z = h[keep_t], z[keep_t]
        a_prev = F.one_hot(torch.as_tensor(a[keep], device=device), num_classes=A).to(h.dtype)
    return {"states": states, "actions": actions, "length": length, "terminated": terminated, **out, **extra, "Q": Q_all, **per_action}
