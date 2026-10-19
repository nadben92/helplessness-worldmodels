"""E3's closed-loop run (the agent with its tabular twin), moved from
experiments/E3_worldmodel_4zones/run_agent.py on 2026-10-15 so that E4 runs the
same training procedure without a copy of it. The constants, helpers and run_one
are run_agent.py's own, bodies unchanged except that the module prefixes
(run_mini., run_mini.e3_env.) became library imports of the same objects; run_agent.py
re-exports every name. One addition, off by default: run_one's `hooks`
(ClosedLoopHooks) -- None reproduces E3's run exactly; an experiment with another
world (E4) passes its own world builder, episode parameters and checkpoint records.
See run_agent.py's docstring for the run itself (outputs, resume, campaign keys).

E5's additions (2026-10-19), every one off by default (E3's and E4's runs unchanged, tested bit for bit):
  hooks.episode_source  the environment of each episode chosen at its reset (closed_loop.ClosedLoopCollector's
                        episode source: E5's per-episode zone assignments, E5's dynamics switch), its replay table and
                        per-step backgrounds; it may also set the run's checkpoints and, once, move its end and its
                        checkpoints (schedule_update: E5's post-switch grid);
  total_env_steps       the run's length when it exceeds its planned budget (budget.total_steps + prefill): beta_rep's
                        ramp stays the planned budget's (E5's immunisation runs past E4's 300,000 steps with E4's ramp);
  save_resumable_at     a copy of the full resume state (checkpoints/resumable_env<NNNNNN>.pt) at these steps, kept;
  branch_from           a new run that starts from such a state (the same planned budget), its own records from there.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.acceleration import compile_from_config
from helplessness.agents.worldmodel.closed_loop import ClosedLoopCollector, EpisodeStats, stored_steps_bound
from helplessness.agents.worldmodel.heads import AgentConfig, Critic, GoalHead, make_slow_critic
from helplessness.agents.worldmodel.imagination import train_heads_step
from helplessness.agents.worldmodel.loop import (
    CollapseAfterSwitchError,
    SeparationNotReachedError,
    WorldModelLossConfig,
    phase_learning_rate,
    rep_beta,
    set_learning_rate,
    train_world_model_step,
)
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.envs.layouts import four_zones_pixel as e3_env
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon, mean_within_group_pair_jensen_shannon
from helplessness.pipeline.checkpoints import _atomic_torch_save, build_rssm, e2_grid_env_steps
from helplessness.pipeline.measure import ARRAY_FIELDS, measure_final, real_context_settings
from helplessness.pipeline.runtime import (
    RunLog,
    _jsonable,
    build_separation_switch,
    environment_metadata,
    max_images_per_forward,
    seed_streams,
)
from helplessness.pipeline.world import GRID_SHAPE, N_STATES, build_world, load_episode_config, zone_of_state, zone_states
from helplessness.probing import probe as e3_probe
from helplessness.runs import require_code_provenance, write_checkpoint_pointer, write_run_meta
from helplessness.streams import worldmodel as e3_rng
from helplessness.twin import tabular_twin as e2_twin


AGENT_KEYS = ("epsilon", "k_act", "k_imag", "horizon", "lam", "slow_tau", "tie_tol_min", "hidden_dim")
# tie_fraction (absolute tolerance, 2026-10-01 (12)) or, in configs made before it, tie_tol_rel (legacy, to read old runs)
OPTIONAL_AGENT_KEYS = ("tie_fraction", "tie_tol_rel")
TWIN_KEYS = ("I_hat", "I_ref", "b", "delta")
SLOT_AGENT_DELTA = 14  # measurement slot of the agent's Delta (run_mini 1-4, 6-9, 11, 12; action_in_latent 10; hybrid codes 13)
SLOT_GOAL_KERNEL = 15  # measurement slot of the checkpoint goal diagnostics (2026-10-02 (19))
NUMPY_STREAMS = ("explore", "tiebreak", "start_state", "replay_sample", "switch_pairs")


@dataclasses.dataclass
class ClosedLoopHooks:
    """What an experiment whose world is not E3's supplies to run_one (added 2026-10-15 for E4). Each field replaces
    one E3-specific step; run_one(hooks=None) is E3's run, unchanged.

      episode            the episode block (gamma, goal_reward, step_reward, max_steps), in place of E3's
                         load_episode_config()
      build_world        (cfg, seed, condition, themes=None) -> pipeline.world.World, in place of build_world
      checkpoint_record  (state) -> dict merged into each checkpoint's record, in place of E3's tabular twin and agent
                         Delta; state holds rssm, goal_head, critic, world, replay, N, cfg, acfg, seed, condition,
                         env_step, grad_step, n_env_steps, device, run_dir, log, episode_source (None without one). It must not draw from the training
                         streams nor the global torch RNG (own streams, torch.random.fork_rng), so that measuring never
                         changes the training trajectory.
      finalize           (records, world, run_dir) -> dict of report fields, in place of E3's metrics.csv and its
                         zone_class / zone_kinds fields
      episode_source     optional (E5): (world, cfg, seed, condition) -> an episode source, with next_episode(env_steps_done)
                         -> (env, obs_offset) (closed_loop.py), table, background_table, background_of_row (the replay's
                         table and per-step backgrounds; None, None for one appearance) and, optionally,
                         checkpoints(n_env_steps) -> the run's initial checkpoints and schedule_update() -> None or, once,
                         (new n_env_steps, new checkpoints). It is saved with the resume state. None: E3's and E4's run.
    """

    episode: dict
    build_world: Callable
    checkpoint_record: Callable
    finalize: Callable
    episode_source: Callable | None = None


class Interrupted(RuntimeError):
    """Raised by run_one(interrupt_after=...) right after a resume point: the resume test's simulated crash."""


def agent_config(cfg: dict, episode: dict | None = None) -> AgentConfig:
    """The run config's agent block with E2's episode values (gamma, rewards, H = max_steps for the tie tolerance)."""
    episode = load_episode_config() if episode is None else episode
    return AgentConfig(gamma=episode["gamma"], goal_reward=episode["goal_reward"], step_reward=episode["step_reward"],
                       episode_horizon=episode["max_steps"], **{key: cfg["agent"][key] for key in AGENT_KEYS},
                       **{key: cfg["agent"][key] for key in OPTIONAL_AGENT_KEYS if key in cfg["agent"]})


def checkpoint_env_steps(cfg: dict, n_env_steps: int) -> list[int]:
    """E2's grid points within the run (environment steps), and the end."""
    grid = cfg["checkpoints"]["grid"]
    points = e2_grid_env_steps() if grid == "e2" else [int(c) for c in grid]
    return sorted({c for c in points if c <= n_env_steps} | {n_env_steps})


def by_zone_mean(values: np.ndarray, world) -> dict[str, float]:
    return {cat: float(np.nanmean(values[zone_states(world.layout, cat)])) for cat in e3_env.ZONE_CATEGORIES}


def twin_at_checkpoint(N: np.ndarray, world, P_ref: np.ndarray, cfg: dict, acfg: AgentConfig, seed: int, condition: str,
                       env_step: int, arrays_out: dict | None = None) -> dict:
    """Condition B at one checkpoint: E2's twin_from_counts on the agent's
    counts, summarised by zone. The matched-count reference draws from a
    child of the run's `reference` stream keyed by the checkpoint, so a
    checkpoint's value does not depend on which others were computed.
    arrays_out (twin.save_kernels, the campaign: 2026-10-07 (29)): receives
    the twin's kernel T_hat (float32) and its per-state I_hat, I_ref, b and
    Delta, keyed <name>_<env_step>."""
    R_true, terminal = e3_env.reward_and_terminal(acfg.goal_reward, acfg.step_reward)
    seq = e3_rng.spawn_sequences(seed, condition)["reference"]
    reference_rng = np.random.default_rng(np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step))))
    twin = e2_twin.twin_from_counts(
        N=N, alpha=cfg["twin"]["alpha"], support=geographic_support_mask(GRID_SHAPE), P_true=world.env.transition_tensor(),
        R_true=R_true, terminal=terminal, P_ref=P_ref, gamma=acfg.gamma, reference_rng=reference_rng,
    )
    if arrays_out is not None:
        arrays_out[f"T_hat_{env_step}"] = twin["T_hat"].astype(np.float32)
        for key in ("I_hat", "I_ref", "b", "delta"):
            arrays_out[f"{key}_{env_step}"] = np.asarray(twin[key], dtype=np.float64)
    visits = N.sum(axis=(1, 2))
    by_zone = {}
    for cat in e3_env.ZONE_CATEGORIES:
        states = zone_states(world.layout, cat)
        by_zone[cat] = {**{key: float(np.mean(twin[key][states])) for key in TWIN_KEYS},
                        "l1": float(np.mean(twin["l1"][states])), "visit_share": float(visits[states].sum() / max(visits.sum(), 1))}
    return by_zone


def agent_delta_at_checkpoint(rssm, goal_head, critic, world, replay, cfg: dict, acfg: AgentConfig, seed: int, condition: str,
                              env_step: int, device: torch.device) -> dict:
    """Delta under P of the world-model agent (spec 5.7), by zone: its greedy
    policy read on real contexts (the main regime, as the other measures)
    and on the canonical probe (the spec's own reading, secondary). Own
    measurement streams, keyed by the checkpoint."""
    seq = e3_rng.spawn_sequences(seed, condition)["measure"].spawn(e3_rng.N_MEASURE_SLOTS)[SLOT_AGENT_DELTA]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step)))
    rng, gen = np.random.default_rng(child), e3_rng.torch_generator(child, device)
    R, terminal = e3_env.reward_and_terminal(acfg.goal_reward, acfg.step_reward)
    P = world.env.transition_tensor()
    was_training = rssm.training
    rssm.eval()
    rc = real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=N_STATES, rng=rng, generator=gen, max_images=max_images_per_forward(cfg))
    out = {}
    readings = {"real_contexts": (ctx["h"], ctx["z"], ctx["states"])}
    h0, z0 = e3_probe.canonical_latents(rssm, world.templates_t)
    readings["canonical"] = (h0, z0, np.arange(N_STATES))
    for name, (h, z, states) in readings.items():
        greedy = e3_probe.agent_greedy_policy(rssm, goal_head, critic, h, z, states, N_STATES, acfg, gen)
        delta = e3_probe.delta_under_p(greedy["policy_probs"], P, R, terminal, acfg.gamma)
        out[name] = {"delta_by_zone": by_zone_mean(delta, world), "delta_mean": float(delta[~terminal].mean()),
                     "n_states_read": int(greedy["measured"].sum())}
    rssm.train(was_training)
    return out


def goal_kernel_at_checkpoint(rssm, world, replay, N: np.ndarray, cfg: dict, seed: int, condition: str, env_step: int,
                              device: torch.device, goal_head=None, critic=None, acfg: AgentConfig | None = None) -> dict:
    """The checkpoint goal diagnostics (docs/e3/DECISIONS.md, 2026-10-02 (19)), cheap: goal arrivals in the replay; the
    prior's mass on the goal for the pairs that lead into it; the median mass on the right successor over the
    deterministic pairs off the goal visited 2 to 8 times (N at this checkpoint). The kernel of those pairs only
    (probe.real_context_kernel_pairs), measure.checkpoint_goal_kernel.draws draws each, from real contexts (the run's
    real_context settings); own measurement slot, a child stream keyed by the checkpoint.

    The campaign's additions (2026-10-07 (29)), on the same contexts, each behind its key of
    measure.checkpoint_goal_kernel: goal_draws -- the four pairs into the goal drawn that many times (the n 2-8 pairs
    keep `draws`); agent: true -- at the four states next to the goal, the agent's Q (probe.agent_greedy_policy, as the
    goal control of measure_noise.py) and the gap between the action into the goal and the best other action against
    the tie-break tolerance; V (the critic on the contexts' posterior latents) averaged by state and by zone; and the
    separation of the zones in the latent (probe.latent_zone_separation on h and z concatenated)."""
    gk = cfg["measure"]["checkpoint_goal_kernel"]
    seq = e3_rng.spawn_sequences(seed, condition)["measure"].spawn(e3_rng.N_MEASURE_SLOTS)[SLOT_GOAL_KERNEL]
    child = np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step)))
    rng, gen = np.random.default_rng(child), e3_rng.torch_generator(child, device)
    P, goal = world.env.transition_tensor(), e3_env.goal_state()
    det_s, det_a = np.nonzero(P.max(-1) >= 1.0 - 1e-12)
    succ = P[det_s, det_a].argmax(-1)
    n = N[det_s, det_a].sum(-1)
    band = (det_s != goal) & (succ != goal) & (n >= 2) & (n <= 8)
    goal_s, goal_a = np.nonzero(P[:, :, goal] > 0)
    keep = goal_s != goal
    goal_pairs = list(zip(goal_s[keep].tolist(), goal_a[keep].tolist()))
    band_pairs = list(zip(det_s[band].tolist(), det_a[band].tolist()))
    was_training = rssm.training
    rssm.eval()
    rc = real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=N_STATES, rng=rng, generator=gen, max_images=max_images_per_forward(cfg))
    margin = cfg["measure"]["kernel_margin_threshold"]
    if gk.get("goal_draws") is None:  # the pilot's diagnostics: one call, every pair at `draws`
        kernel = e3_probe.real_context_kernel_pairs(rssm, ctx, world.templates_t, goal_pairs + band_pairs, draws=gk["draws"],
                                                    margin_threshold=margin, generator=gen)
    else:
        kernel = {**e3_probe.real_context_kernel_pairs(rssm, ctx, world.templates_t, goal_pairs, draws=gk["goal_draws"],
                                                       margin_threshold=margin, generator=gen),
                  **e3_probe.real_context_kernel_pairs(rssm, ctx, world.templates_t, band_pairs, draws=gk["draws"],
                                                       margin_threshold=margin, generator=gen)}
    band_mass = [kernel[p]["p_hat"][int(P[p].argmax())] for p in band_pairs if p in kernel]
    out = {"goal_arrivals_in_replay": int(np.count_nonzero(replay.export_compact()["terminal"])),
           "goal_pairs": [{"s": s, "a": a, "n": int(N[s, a].sum()), "mass_on_goal": float(kernel[(s, a)]["p_hat"][goal]) if (s, a) in kernel else None}
                          for s, a in goal_pairs],
           "median_mass_on_goal_pairs": float(np.median([kernel[p]["p_hat"][goal] for p in goal_pairs if p in kernel]))
           if any(p in kernel for p in goal_pairs) else float("nan"),
           "median_mass_right_successor_n2_8": float(np.median(band_mass)) if band_mass else float("nan"), "pairs_n2_8": len(band_mass),
           "draws": gk["draws"]}
    if gk.get("goal_draws") is not None:
        out["goal_draws"] = gk["goal_draws"]
    if gk.get("agent"):
        out.update(agent_at_goal_and_latent(rssm, goal_head, critic, acfg, world, ctx, goal_pairs, cfg, rng, gen))
    rssm.train(was_training)
    return out


def agent_at_goal_and_latent(rssm, goal_head, critic, acfg: AgentConfig, world, ctx: dict, goal_pairs: list, cfg: dict,
                             rng: np.random.Generator, gen: torch.Generator) -> dict:
    """goal_kernel_at_checkpoint's campaign part (2026-10-07 (29)) on its real contexts: the Q gaps next to the goal, V by
    state and zone, and the zones' separation in the latent."""
    states = np.asarray(ctx["states"])
    with torch.no_grad():
        greedy = e3_probe.agent_greedy_policy(rssm, goal_head, critic, ctx["h"], ctx["z"], states, N_STATES, acfg, gen)
        v_ctx = torch.cat([critic(ctx["h"][i:i + 4096], ctx["z"][i:i + 4096]).reshape(-1)
                           for i in range(0, ctx["h"].shape[0], 4096)]).cpu().numpy()
    tol = float(acfg.tie_tol_abs) if acfg.tie_tol_abs is not None else float("nan")
    neighbours = []
    for s, a in goal_pairs:
        q = greedy["Q"][s]
        best_other = float(np.nanmax(np.delete(q, a))) if np.isfinite(q).any() else float("nan")
        neighbours.append({"s": s, "a_into_goal": a, "contexts": int(np.sum(states == s)), "Q": q.tolist(),
                           "gap_into_goal_minus_best_other": float(q[a] - best_other), "tie_tolerance": tol})
    counts = np.bincount(states, minlength=N_STATES)
    V = np.full(N_STATES, np.nan)
    V[counts > 0] = np.bincount(states, weights=v_ctx, minlength=N_STATES)[counts > 0] / counts[counts > 0]
    zones = [zone_of_state(world.layout, s) for s in range(N_STATES)]
    x = torch.cat([ctx["h"].reshape(ctx["h"].shape[0], -1), ctx["z"].reshape(ctx["z"].shape[0], -1)], dim=1)
    separation = e3_probe.latent_zone_separation(x, states, zones, list(e3_env.ZONE_CATEGORIES),
                                                 cfg["measure"]["z_only_test_frac"], rng)
    return {"goal_neighbours": neighbours, "V_by_zone": by_zone_mean(V, world), "V_by_state": V.tolist(),
            "latent_zone_separation": separation}


def extra_checkpoints(switch_env_step: int, grid: list[int], n_env_steps: int, spec: dict) -> list[int]:
    """measure.checkpoint_measures (the campaign, 2026-10-07 (29)): when fewer than spec["min_in_window"] checkpoints of
    the grid fall within the spec["window_after_switch"] environment steps after the switch, checkpoints at the switch
    + each of spec["extra_offsets"], in order, until there are enough (none at or past the run's end)."""
    window_end = switch_env_step + spec["window_after_switch"]
    n_in = sum(switch_env_step < c <= window_end for c in grid)
    extra = []
    for offset in spec["extra_offsets"]:
        if n_in >= spec["min_in_window"]:
            break
        c = switch_env_step + int(offset)
        if c < n_env_steps and c not in grid and c <= window_end:
            extra.append(c)
            n_in += 1
    return extra


LIGHT_DROPPED = ("real_prior_probs", "real_post_probs")  # per-context latent distributions: recomputable from the weights and replay


def light_arrays(arrays: dict, cfg: dict) -> dict:
    """measure.light_arrays (the campaign): the measures' arrays without the per-context latent distributions (37 MB of
    prior probabilities per measurement at 200 contexts per state), which the handed-off weights and replay reproduce."""
    return {k: v for k, v in arrays.items() if k not in LIGHT_DROPPED} if cfg["measure"].get("light_arrays") else arrays


def full_measures_at_checkpoint(rssm, world, replay, N: np.ndarray, cfg: dict, seed: int, condition: str, run_dir: Path, log,
                                grad_step: int, env_step: int, device: torch.device, extra: dict | None = None) -> tuple[dict, dict, dict]:
    """The complete E3 measures (run_mini.measure_final: I_s, I_s projected, off-support mass, I_z, kernels, probes) of the
    weights at a checkpoint, on the replay as it is then, into <run dir>/measures/env<NNNNNN>/ (results.json, arrays.npz,
    artifacts/). The run's own measurement streams (run_mini.seed_streams), as the final measures; the global torch RNG
    is forked and the model's training mode restored, so the training that follows is the one it would have been.
    extra: fields added to results.json (the pilot's runs measured afterwards: the replay's provenance).
    Returns (fields, arrays, the summary recorded in checkpoints.jsonl)."""
    from helplessness.measures.kernel_probe import off_support_mass

    out = run_dir / "measures" / f"env{env_step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"full measures at environment step {env_step} (gradient step {grad_step}) into {out.relative_to(run_dir)} ...")
    t0 = time.perf_counter()
    was_training = rssm.training
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        fields, arrays = measure_final(rssm, world, replay, N, cfg, seed_streams(seed, device, condition), out, log, grad_step)
    rssm.train(was_training)
    fields.pop("diagnostics_final")
    arrays = light_arrays(arrays, cfg)
    seconds = time.perf_counter() - t0
    results = {"seed": seed, "condition": condition, "checkpoint_env_step": env_step, "grad_step": grad_step, "seconds": seconds,
               **(extra or {}), **fields}
    (out / "results.json").write_text(json.dumps(_jsonable({k: v for k, v in results.items() if k not in ARRAY_FIELDS}),
                                                 indent=1), encoding="utf-8")
    np.savez_compressed(out / "arrays.npz", **arrays)
    rc = fields["real_context"]
    off = off_support_mass(arrays["real_p_hat"], geographic_support_mask(GRID_SHAPE))
    zone = np.array(fields["zone_of_state"])
    summary = {"dir": f"measures/env{env_step:06d}",  # no duration here: checkpoints.jsonl stays identical across a resume
               **{f"{m}_by_zone": {cat: rc[f"{m}_by_zone"][cat]["mean"] for cat in e3_env.ZONE_CATEGORIES}
                  for m in ("I_s", "I_s_projected", "I_z")},
               "off_support_by_zone": {cat: float(np.nanmean(off[zone == cat])) if np.isfinite(off[zone == cat]).any() else float("nan")
                                       for cat in e3_env.ZONE_CATEGORIES}}
    log(f"full measures at {env_step} done in {seconds:.0f} s: I_s by zone {json.dumps({k: round(v, 4) for k, v in summary['I_s_by_zone'].items()})}")
    return fields, arrays, summary


def background_groups(model_batch) -> np.ndarray:
    """(B * T,) a label per item of the batch, equal for items whose per-step background images are equal (computed from
    the images; the batch's order, as the posteriors'). Needs ModelBatch.background."""
    if model_batch.background is None:
        raise ValueError("switch_pairs: same_background needs per-step backgrounds (ModelBatch.background)")
    bg = torch.as_tensor(model_batch.background)
    flat = bg.reshape(bg.shape[0] * bg.shape[1], -1)
    return torch.unique(flat, dim=0, return_inverse=True)[1].cpu().numpy()


def _stream_states(streams: dict, gens: dict) -> dict:
    return {**{name: streams[name].bit_generator.state for name in NUMPY_STREAMS}, **{name: g.get_state() for name, g in gens.items()},
            "torch_global": torch.get_rng_state()}


def _restore_streams(state: dict, streams: dict, gens: dict) -> None:
    for name in NUMPY_STREAMS:
        streams[name].bit_generator.state = state[name]
    for name, g in gens.items():
        g.set_state(state[name])
    torch.set_rng_state(state["torch_global"])


def run_one(cfg: dict, seed: int, condition: str, run_dir: Path, device: torch.device, max_env_steps: int | None = None,
            resume: bool = False, interrupt_after: int | None = None, hooks: ClosedLoopHooks | None = None,
            total_env_steps: int | None = None, save_resumable_at: tuple[int, ...] = (), branch_from: Path | None = None) -> dict:
    """One closed-loop run in run_dir. resume=True continues it from
    checkpoints/resume.pt (the config and the planned length are the run's
    own). interrupt_after (tests): raise Interrupted right after the resume
    point written at that environment step. E5's options (module docstring):
    total_env_steps, save_resumable_at, branch_from (a resumable state file
    of a run with the same planned budget)."""
    if resume and branch_from is not None:
        raise ValueError("resume and branch_from are exclusive")
    log = RunLog(run_dir / "console.log", f"[agent seed={seed} {condition}]")
    meta = environment_metadata(device)
    require_code_provenance(meta)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    acfg = agent_config(cfg, None if hooks is None else hooks.episode)
    resume_path = run_dir / "checkpoints" / "resume.pt"
    if resume:
        write_run_meta(run_dir, seed, status="running", resumed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       **{f"resume_{k}": v for k, v in meta.items() if k in ("device", "git_commit", "source_sha256")})
        log(f"RESUMING from {resume_path}")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        e5_meta = {**({"total_env_steps": int(total_env_steps)} if total_env_steps is not None else {}),
                   **({"branched_from": str(branch_from)} if branch_from is not None else {})}
        write_run_meta(run_dir, seed, status="running", condition=condition, phase="5 closed loop", agent=dataclasses.asdict(acfg),
                       rng_streams=list(e3_rng.STREAM_NAMES), episode=load_episode_config() if hooks is None else hooks.episode, max_env_steps=max_env_steps,
                       **e5_meta, **meta)
    log(f"code: git commit {meta['git_commit']}, source sha256 {meta['source_sha256']}")

    world_builder = build_world if hooks is None else hooks.build_world
    world = world_builder(cfg, seed, condition)
    P_ref = (world_builder(cfg, seed, "reference", themes=world.themes).env.transition_tensor() if condition == "experimental"
             else world.env.transition_tensor())
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    budget, prefill = cfg["budget"]["total_steps"], cfg["budget"]["prefill_steps"]
    planned_env_steps = budget + prefill
    n_env_steps = planned_env_steps if max_env_steps is None else min(planned_env_steps, max_env_steps)
    if total_env_steps is not None:  # E5: the run's own length, past its planned budget (beta_rep's ramp stays the budget's)
        n_env_steps = int(total_env_steps)
    source = (hooks.episode_source(world, cfg, seed, condition) if hooks is not None and hooks.episode_source is not None
              else None)
    if source is None:
        replay = ReplayBuffer.for_run(budget=stored_steps_bound(n_env_steps), prefill_steps=0, table=world.templates_np)
    else:
        replay = ReplayBuffer.for_run(budget=stored_steps_bound(n_env_steps), prefill_steps=0, table=source.table,
                                      background_table=source.background_table, background_of_row=source.background_of_row)

    streams, seqs = e3_rng.spawn_streams(seed, condition), e3_rng.spawn_sequences(seed, condition)
    gens = {name: e3_rng.torch_generator(seqs[seq], device)
            for name, seq in (("latent", "latent_sample"), ("act", "act_latent"), ("imagination", "imagination"))}
    torch.manual_seed(e3_rng.torch_seed(seqs["network_init"]))  # built on the CPU, then moved: same weights on every device
    rssm = compile_from_config(build_rssm(cfg).to(device), cfg)  # runtime.compile: off unless set (acceleration.py)
    torch.manual_seed(e3_rng.torch_seed(seqs["agent_init"]))
    goal_head = GoalHead(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim).to(device)
    critic = Critic(rssm.cfg.h_dim, rssm.cfg.latent_dim, acfg.hidden_dim).to(device)
    slow_critic = make_slow_critic(critic)

    tr = cfg["train"]
    optimizer = torch.optim.Adam(rssm.parameters(), lr=tr["lr"], eps=tr["adam_eps"])
    goal_optimizer = torch.optim.Adam(goal_head.parameters(), lr=cfg["agent"]["goal_lr"], eps=cfg["agent"]["adam_eps"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg["agent"]["critic_lr"], eps=cfg["agent"]["adam_eps"])
    loss_cfg = WorldModelLossConfig(k=cfg["render"]["k"], w_patch=tr["w_patch"], beta_dyn=tr["beta_dyn"], beta_rep=tr["beta_rep"],
                                    free_bits=tr["free_bits"], free_bits_mode=tr["free_bits_mode"])
    switch, n_pairs = build_separation_switch(cfg)
    # E5's option (docs/e5/DECISIONS.md (17)): the switch statistic on pairs of the same background (absent: any pair)
    same_background_pairs = tr["latent_estimator"].get("switch_pairs", "any") == "same_background"
    planned_grad_steps = int(round((planned_env_steps - prefill) * tr["replay_ratio"]))

    collector = ClosedLoopCollector(world.env, replay, rssm, goal_head, critic, acfg, N_STATES, start_rng=streams["start_state"],
                                    explore_rng=streams["explore"], tiebreak_rng=streams["tiebreak"], act_generator=gens["act"], device=device,
                                    episode_source=source)
    checkpoints = checkpoint_env_steps(cfg, n_env_steps)
    if source is not None and hasattr(source, "checkpoints"):
        checkpoints = sorted(int(c) for c in source.checkpoints(n_env_steps))
    measures_spec = cfg["measure"].get("checkpoint_measures")  # the campaign's full measures at the checkpoints (2026-10-07 (29))
    save_twin = bool(cfg["twin"].get("save_kernels"))
    resume_every = cfg["checkpoints"].get("resume_every")
    monitor_every, window = cfg["log"]["monitor_every"], cfg["log"]["behaviour_window"]
    progress = {"env_step": 0, "grad_step": 0, "credit": 0.0, "last_stat": float("nan"), "rep_warmup": None, "last": {},
                "records": [], "counts": {}, "seconds": 0.0, "switch_env_step": None, "extra_checkpoints": [], "twin_arrays": {}}

    def agent_state() -> dict:
        return {"rssm": rssm.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg), "goal_head": goal_head.state_dict(),
                "critic": critic.state_dict(), "slow_critic": slow_critic.state_dict(), "agent_config": dataclasses.asdict(acfg),
                "env_step": progress["env_step"], "step": progress["grad_step"], "switch_step": switch.switch_step}

    def resume_state() -> dict:
        state = {
            "progress": progress, "rssm": rssm.state_dict(), "goal_head": goal_head.state_dict(), "critic": critic.state_dict(),
            "slow_critic": slow_critic.state_dict(), "optimizer": optimizer.state_dict(), "goal_optimizer": goal_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "switch": {f.name: getattr(switch, f.name) for f in dataclasses.fields(switch)},
            "replay": pickle.dumps(replay), "env": pickle.dumps(collector.env),
            "collector": {"N": collector.N, "stats": dataclasses.asdict(collector.stats), "obs": collector._obs, "s": collector._s,
                          "arrival_reward": collector._arrival_reward, "is_first": collector._is_first,
                          "episode_length": collector._episode_length,
                          "filter": {"h": collector.filter.h, "z": collector.filter.z, "a_prev": collector.filter.a_prev}},
            "streams": _stream_states(streams, gens), "n_env_steps": n_env_steps, "planned_env_steps": planned_env_steps,
        }
        if collector.episode_source is not None:  # the episode's environment and its source in one pickle: one object
            state["env"], state["episode_source"] = pickle.dumps((collector.env, collector.episode_source)), True
        return state

    def save_resume() -> None:
        _atomic_torch_save(resume_state(), resume_path)
        write_checkpoint_pointer(run_dir, resume_path.name, progress["env_step"])

    if resume or branch_from is not None:
        state = torch.load(resume_path if resume else branch_from, map_location="cpu", weights_only=False)
        if resume:
            expected = state["progress"]["schedule"][0] if "schedule" in state["progress"] else n_env_steps
            if state["planned_env_steps"] != planned_env_steps or state["n_env_steps"] != expected:
                raise ValueError("resume.pt was written for another budget")
        elif state["planned_env_steps"] != planned_env_steps:
            raise ValueError(f"{branch_from} was planned for another budget (beta_rep's ramp would differ)")
        for module, key in ((rssm, "rssm"), (goal_head, "goal_head"), (critic, "critic"), (slow_critic, "slow_critic"),
                            (optimizer, "optimizer"), (goal_optimizer, "goal_optimizer"), (critic_optimizer, "critic_optimizer")):
            module.load_state_dict(state[key])
        for name, value in state["switch"].items():
            setattr(switch, name, value)
        replay = pickle.loads(state["replay"])
        if branch_from is not None:
            replay.grow(stored_steps_bound(n_env_steps))
        if state.get("episode_source"):
            collector.env, collector.episode_source = pickle.loads(state["env"])
        else:
            collector.env = pickle.loads(state["env"])
        collector.replay = replay
        world.env = collector.env
        c = state["collector"]
        collector.N, collector.stats = c["N"], EpisodeStats(**c["stats"])
        collector._obs, collector._s, collector._arrival_reward = c["obs"], c["s"], c["arrival_reward"]
        collector._is_first, collector._episode_length = c["is_first"], c["episode_length"]
        collector.filter.h, collector.filter.z, collector.filter.a_prev = (c["filter"][k].to(device) for k in ("h", "z", "a_prev"))
        _restore_streams(state["streams"], streams, gens)
        progress = state["progress"]
        if branch_from is not None:  # a new run from here: its own records, counts and schedule
            progress.update(records=[], counts={}, twin_arrays={}, seconds=0.0, extra_checkpoints=[])
            progress.pop("schedule", None)
        if "schedule" in progress:
            n_env_steps, checkpoints = progress["schedule"][0], list(progress["schedule"][1])
        (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(_jsonable(r)) + "\n" for r in progress["records"]),
                                                   encoding="utf-8")
        log(f"{'resumed' if resume else f'branched from {branch_from}'} at environment step {progress['env_step']} "
            f"(gradient step {progress['grad_step']}, switch {switch.switch_step})")
    if device.type == "cuda":
        replay.to_device(device)
    log(f"condition {condition}; device {device}; {n_env_steps} environment steps (planned {planned_env_steps}, prefill {prefill} under "
        f"the agent's own policy, epsilon {acfg.epsilon}); {tr['replay_ratio']} gradient step(s) per environment step; checkpoints at "
        f"{checkpoints}; resume point every {resume_every} environment steps; agent heads trained from the estimator switch on")

    def fail(err: Exception) -> None:
        write_run_meta(run_dir, seed, status="failed", error=str(err), env_steps=progress["env_step"], grad_steps=progress["grad_step"],
                       switch_step=switch.switch_step)

    t0 = time.perf_counter() - progress["seconds"]
    last = progress["last"]
    final_measures = None
    env_step = progress["env_step"] + 1
    while env_step <= n_env_steps:
        collector.step(estimator=switch.estimator())
        if collector.episode_source is not None and hasattr(collector.episode_source, "schedule_update"):
            update = collector.episode_source.schedule_update()
            if update is not None:  # E5: the source moved the run's end and its checkpoints (once)
                n_env_steps, checkpoints = int(update[0]), sorted(int(c) for c in update[1])
                progress["schedule"] = (n_env_steps, checkpoints)
                log(f"schedule from the episode source at environment step {env_step}: {n_env_steps} environment steps, "
                    f"checkpoints at {checkpoints}")
        if env_step > prefill:
            progress["credit"] += tr["replay_ratio"]
        while progress["credit"] >= 1.0:
            progress["credit"] -= 1.0
            grad_step = progress["grad_step"]
            set_learning_rate(optimizer, phase_learning_rate(grad_step, switch.switch_step, tr["lr"], tr["lr_after_switch"]))
            beta = rep_beta(grad_step, switch.switch_step, progress["rep_warmup"] or 0, loss_cfg.beta_rep)
            model_batch, _ = replay.sample(tr["batch_size"], tr["seq_len"], streams["replay_sample"])
            eval_now = switch.is_eval_step(grad_step)
            out = train_world_model_step(rssm, optimizer, model_batch, world.background_t, dataclasses.replace(loss_cfg, beta_rep=beta),
                                         clip_norm=tr["clip_norm"], estimator=switch.estimator(), return_posteriors=eval_now,
                                         generator=gens["latent"], return_states=switch.switched)
            last.update({k: v for k, v in out.items() if k not in ("post_probs", "h", "z")})
            if switch.switched:
                last.update(train_heads_step(rssm, goal_head, critic, slow_critic, goal_optimizer, critic_optimizer, out["h"], out["z"],
                                             torch.as_tensor(np.asarray(model_batch.r)), acfg, gens["imagination"],
                                             critic_clip_norm=cfg["agent"]["critic_clip_norm"]))
            if eval_now:
                if same_background_pairs:  # E5's option (off by default): pairs of items with the same background image
                    progress["last_stat"] = mean_within_group_pair_jensen_shannon(out["post_probs"], background_groups(model_batch),
                                                                                  streams["switch_pairs"], n_pairs)
                else:
                    progress["last_stat"] = mean_random_pair_jensen_shannon(out["post_probs"], streams["switch_pairs"], n_pairs)
                was_switched = switch.switched
                try:
                    switch.update(grad_step, progress["last_stat"])
                except CollapseAfterSwitchError as err:
                    fail(err)
                    raise
                if switch.switched and not was_switched:
                    progress["switch_env_step"] = env_step
                    if measures_spec:
                        progress["extra_checkpoints"] = extra_checkpoints(env_step, checkpoints, n_env_steps, measures_spec)
                    progress["rep_warmup"] = (tr["rep_warmup_steps"] if tr["rep_warmup_steps"] is not None
                                              else int(round(0.2 * (planned_grad_steps - switch.switch_step))))
                    log(f">>> SWITCH to straight-through effective at gradient step {switch.switch_step} (batch-pair JS "
                        f"{progress['last_stat']:.3f} bits, environment step {env_step}); the agent's heads train from here"
                        + (f"; full measures at every checkpoint after it, extra checkpoints {progress['extra_checkpoints']}"
                           if measures_spec else ""))
            progress["grad_step"] += 1
            try:
                switch.check_deadline(progress["grad_step"])
            except SeparationNotReachedError as err:
                fail(err)
                raise
        progress["env_step"] = env_step
        at_checkpoint = env_step in checkpoints or env_step in progress.get("extra_checkpoints", ())

        if env_step % monitor_every == 0 or at_checkpoint:
            behaviour, stats = collector.stats.window(window), collector.stats
            tie = stats.tied_greedy_steps / max(stats.greedy_steps, 1)
            log(f"env step {env_step:6d}/{n_env_steps} | grad {progress['grad_step']:6d} | JS {progress['last_stat']:6.3f} | "
                f"{'straight-through' if switch.switched else 'probabilities   '} | success {behaviour['success_rate']:.3f} "
                f"len {behaviour['episode_length']:.1f} ({behaviour['n_episodes']} ep) | ties {tie:.3f} | recon_patch "
                f"{last.get('recon_patch', float('nan')):.4f} | goal_loss {last.get('goal_loss', float('nan')):.4f} critic_loss "
                f"{last.get('critic_loss', float('nan')):.4f} | {time.perf_counter() - t0:.0f}s")
        if at_checkpoint:
            stats = collector.stats
            record = {"env_step": env_step, "grad_step": progress["grad_step"], "switch_step": switch.switch_step,
                      "batch_pair_js": progress["last_stat"], "episodes": stats.episodes, "successes": stats.successes,
                      **collector.stats.window(window), "tie_fraction": stats.tied_greedy_steps / max(stats.greedy_steps, 1),
                      "explored_fraction": stats.explored_steps / max(stats.env_steps, 1), "stored_steps": len(replay),
                      "train": {k: float(v) for k, v in last.items()}}
            if hooks is None:
                record["twin_by_zone"] = twin_at_checkpoint(collector.N, world, P_ref, cfg, acfg, seed, condition, env_step,
                                                            progress["twin_arrays"] if save_twin else None)
                record["agent_delta"] = agent_delta_at_checkpoint(rssm, goal_head, critic, world, replay, cfg, acfg, seed, condition,
                                                                  env_step, device)
            else:
                record.update(hooks.checkpoint_record({
                    "rssm": rssm, "goal_head": goal_head, "critic": critic, "world": world, "replay": replay, "N": collector.N,
                    "cfg": cfg, "acfg": acfg, "seed": seed, "condition": condition, "env_step": env_step,
                    "grad_step": progress["grad_step"], "n_env_steps": n_env_steps, "device": device, "run_dir": run_dir, "log": log,
                    "episode_source": collector.episode_source, "switch_env_step": progress.get("switch_env_step")}))
            if cfg["measure"].get("checkpoint_goal_kernel"):
                record["goal_diagnostics"] = goal_kernel_at_checkpoint(rssm, world, replay, collector.N, cfg, seed, condition, env_step, device,
                                                                       goal_head=goal_head, critic=critic, acfg=acfg)
            progress["counts"][f"N_{env_step}"] = collector.N.copy()
            # the weights at every checkpoint (docs/e3/DECISIONS.md, 2026-10-01 (12)): written, nothing drawn
            _atomic_torch_save(agent_state(), run_dir / "checkpoints" / f"agent_env{env_step:06d}.pt")
            if measures_spec and progress.get("switch_env_step") is not None and env_step > progress["switch_env_step"]:
                fields_c, arrays_c, record["measures"] = full_measures_at_checkpoint(
                    rssm, world, replay, collector.N, cfg, seed, condition, run_dir, log, progress["grad_step"], env_step, device)
                if env_step == n_env_steps:  # the run's final measures are this checkpoint's
                    final_measures = (fields_c, arrays_c)
            progress["records"].append(record)
            with open(run_dir / "checkpoints.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record)) + "\n")
        if env_step < n_env_steps and (at_checkpoint or (resume_every and env_step % resume_every == 0)):
            progress["seconds"] = time.perf_counter() - t0
            save_resume()
            if interrupt_after is not None and env_step >= interrupt_after:
                raise Interrupted(f"simulated interruption after the resume point at environment step {env_step}")
        if env_step in save_resumable_at:  # E5: a kept copy of the full state, from which another run may branch
            progress["seconds"] = time.perf_counter() - t0
            _atomic_torch_save(resume_state(), run_dir / "checkpoints" / f"resumable_env{env_step:06d}.pt")
            log(f"resumable state saved at environment step {env_step}")
        env_step += 1

    progress["seconds"] = time.perf_counter() - t0
    records, counts = progress["records"], progress["counts"]
    torch.save(agent_state(), run_dir / "checkpoints" / "agent_final.pt")
    np.savez_compressed(run_dir / "counts.npz", **counts)
    if save_twin:  # the twin's kernel and per-state values at every checkpoint (2026-10-07 (29))
        np.savez_compressed(run_dir / "twin.npz", **progress["twin_arrays"])
    if cfg["checkpoints"].get("save_final_replay"):  # the whole replay, compact (states, actions, rewards, flags): with the
        # checkpoint weights, every measure can be recomputed off the cluster (2026-10-07 (29))
        np.savez_compressed(run_dir / "replay_final.npz", **replay.export_compact())
    if hooks is None:  # E3's metrics.csv (an experiment with hooks writes its own in finalize)
        with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["env_step", "grad_step", "zone", "dynamics", "success_rate", "episode_length", "tie_fraction", "visit_share",
                             "twin_I_hat", "twin_I_ref", "twin_b", "twin_delta", "twin_l1", "agent_delta_real", "agent_delta_canonical"])
            for r in records:
                for cat, z in r["twin_by_zone"].items():
                    writer.writerow([r["env_step"], r["grad_step"], cat, world.zone_kinds[cat], r["success_rate"], r["episode_length"],
                                     r["tie_fraction"], z["visit_share"], z["I_hat"], z["I_ref"], z["b"], z["delta"], z["l1"],
                                     r["agent_delta"]["real_contexts"]["delta_by_zone"][cat],
                                     r["agent_delta"]["canonical"]["delta_by_zone"][cat]])

    fields, arrays = {}, {}
    if cfg["measure"].get("final", True) and final_measures is not None:
        log(f"final measures: those of the last checkpoint ({final_measures[0]['final_step']} gradient steps, measures/env{n_env_steps:06d})")
        fields, arrays = final_measures
        np.savez_compressed(run_dir / "arrays.npz", **arrays)
    elif cfg["measure"].get("final", True):
        log("final measures of the world model in closed loop (run_mini.measure_final) ...")
        streams_m = seed_streams(seed, device, condition)
        fields, arrays = measure_final(rssm, world, replay, collector.N, cfg, streams_m, run_dir, log, progress["grad_step"])
        fields.pop("diagnostics_final")
        arrays = light_arrays(arrays, cfg)
        np.savez_compressed(run_dir / "arrays.npz", **arrays)
    world_fields = ({"zone_class": e3_env.block_class_for_seed(seed), "zone_kinds": world.zone_kinds} if hooks is None
                    else hooks.finalize(records, world, run_dir))
    report = {"seed": seed, "condition": condition, **world_fields, "env_steps": n_env_steps, "planned_env_steps": planned_env_steps,
              "grad_steps": progress["grad_step"], "switch_step": switch.switch_step,
              "truncated_by_max_env_steps": n_env_steps < planned_env_steps, "seconds": progress["seconds"], "checkpoints": records,
              "reproducibility": meta, **fields}
    (run_dir / "results.json").write_text(json.dumps(_jsonable({k: v for k, v in report.items() if k not in ARRAY_FIELDS}),
                                                     indent=1), encoding="utf-8")
    write_run_meta(run_dir, seed, status="done", env_steps=n_env_steps, grad_steps=progress["grad_step"], switch_step=switch.switch_step)
    log(f"run done in {progress['seconds']:.0f}s; results in {run_dir}")
    return report
