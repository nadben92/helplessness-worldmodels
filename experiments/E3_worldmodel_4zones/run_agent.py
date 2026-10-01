"""E3 Phase 5: the agent in closed loop (condition A of spec section 6, or
Ref in the reference environment) with its tabular twin (condition B) at
every checkpoint. CPU test bench: configs/e3/agent_mini.yaml (mini_long's
network and world-model procedure).

One environment step at a time, the agent filters its latent state, acts
epsilon-greedily on a one-step lookahead (agents.worldmodel.policy), and
stores what it sees (agents.worldmodel.closed_loop: the goal's observation
is stored). After a minimal prefill played under the same policy, every
environment step is followed by train.replay_ratio gradient steps:

  world model   the retained procedure of run_mini.py, unchanged: posterior
                probabilities until the label-free batch-pair JS criterion,
                then straight-through; lr and beta_rep scheduled on the
                switch; SeparationNotReachedError / CollapseAfterSwitchError;
  agent heads   from the switch on (before it the latent is not a discrete
                code): the goal head on the batch's posterior states (BCE
                against 1[r > 0]) and the critic in imagination
                (agents.worldmodel.imagination), slow critic by EMA.
                Before the switch the policy is the same one, on untrained
                heads: their output layers start at zero (heads.py), so
                every Q-value ties and the random tie-break explores
                uniformly until the heads have learnt something.

At E2's checkpoints (environment steps): the twin's full chain from the
agent's own counts N[s, a, s'] (experiments/E2_online_4zones/twin.py,
imported: Dirichlet alpha on the 5 successors, VI, b with the matched-count
reference, Delta under P); Delta under P of the WORLD-MODEL agent itself
(spec 5.7: its greedy policy read on its own latent states, real contexts
and canonical probe, evaluated exactly under P; probe.agent_greedy_policy);
the behaviour (success rate, episode length, tie fraction, visitation by
zone) and the training scalars.

At the end (measure.final): the E3 measures of the world model in closed
loop, run_mini.measure_final on the closed-loop replay (I_s, I_s projected,
off-support mass, I_z, kernels, probes, D1 / D2), written as run_mini writes
them, so aggregate_b.py pairs a closed-loop environment run with its
reference (b on I_s and on I_s projected, bias control).

Resume (docs/e3/DECISIONS.md, 2026-09-30 (8)): every checkpoints.resume_every
environment steps and at every checkpoint, checkpoints/resume.pt holds the
whole state -- models, optimisers, switch, replay, environment, counts,
behaviour, every random stream -- written atomically;
`run_agent.py --resume <run dir>` continues in the same directory and gives
the same results as an uninterrupted run (CPU; tested).

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/run_agent.py [--config configs/e3/agent_mini.yaml] [--seeds 0] [--conditions experimental]
        [--max-env-steps N] [--device cpu] [--out-dir results]
    python experiments/E3_worldmodel_4zones/run_agent.py --resume results/E3_worldmodel_4zones/<run directory>
Outputs: results/E3_worldmodel_4zones/<run name>/ with config.yaml, meta.yaml, console.log, metrics.csv (one row per
checkpoint and zone), checkpoints.jsonl, results.json, arrays.npz (the final measures), counts.npz (N at every
checkpoint), checkpoints/agent_final.pt and checkpoints/resume.pt.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pickle
import sys
import time
from pathlib import Path

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
from helplessness.envs.zones import geographic_support_mask
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon
from helplessness.runs import require_code_provenance, write_run_meta

run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "run_agent_run_mini")
e3_rng = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "run_agent_rng_streams")
e3_probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "run_agent_probe")
e2_twin = load_experiment_module("E2_online_4zones", "twin.py", "run_agent_e2_twin")

DEFAULT_CONFIG = run_mini.REPO_ROOT / "configs" / "e3" / "agent_mini.yaml"
AGENT_KEYS = ("epsilon", "k_act", "k_imag", "horizon", "lam", "slow_tau", "tie_tol_rel", "tie_tol_min", "hidden_dim")
TWIN_KEYS = ("I_hat", "I_ref", "b", "delta")
SLOT_AGENT_DELTA = 14  # measurement slot of the agent's Delta (run_mini 1-4, 6-9, 11, 12; action_in_latent 10; hybrid codes 13)
NUMPY_STREAMS = ("explore", "tiebreak", "start_state", "replay_sample", "switch_pairs")


class Interrupted(RuntimeError):
    """Raised by run_one(interrupt_after=...) right after a resume point: the resume test's simulated crash."""


def agent_config(cfg: dict) -> AgentConfig:
    """The run config's agent block with E2's episode values (gamma, rewards)."""
    episode = run_mini.load_episode_config()
    return AgentConfig(gamma=episode["gamma"], goal_reward=episode["goal_reward"], step_reward=episode["step_reward"],
                       **{key: cfg["agent"][key] for key in AGENT_KEYS})


def checkpoint_env_steps(cfg: dict, n_env_steps: int) -> list[int]:
    """E2's grid points within the run (environment steps), and the end."""
    grid = cfg["checkpoints"]["grid"]
    points = run_mini.e2_grid_env_steps() if grid == "e2" else [int(c) for c in grid]
    return sorted({c for c in points if c <= n_env_steps} | {n_env_steps})


def by_zone_mean(values: np.ndarray, world) -> dict[str, float]:
    return {cat: float(np.nanmean(values[run_mini.zone_states(world.layout, cat)])) for cat in run_mini.e3_env.ZONE_CATEGORIES}


def twin_at_checkpoint(N: np.ndarray, world, P_ref: np.ndarray, cfg: dict, acfg: AgentConfig, seed: int, condition: str,
                       env_step: int) -> dict:
    """Condition B at one checkpoint: E2's twin_from_counts on the agent's
    counts, summarised by zone. The matched-count reference draws from a
    child of the run's `reference` stream keyed by the checkpoint, so a
    checkpoint's value does not depend on which others were computed."""
    R_true, terminal = run_mini.e3_env.reward_and_terminal(acfg.goal_reward, acfg.step_reward)
    seq = e3_rng.spawn_sequences(seed, condition)["reference"]
    reference_rng = np.random.default_rng(np.random.SeedSequence(entropy=seq.entropy, spawn_key=(*seq.spawn_key, int(env_step))))
    twin = e2_twin.twin_from_counts(
        N=N, alpha=cfg["twin"]["alpha"], support=geographic_support_mask(run_mini.GRID_SHAPE), P_true=world.env.transition_tensor(),
        R_true=R_true, terminal=terminal, P_ref=P_ref, gamma=acfg.gamma, reference_rng=reference_rng,
    )
    visits = N.sum(axis=(1, 2))
    by_zone = {}
    for cat in run_mini.e3_env.ZONE_CATEGORIES:
        states = run_mini.zone_states(world.layout, cat)
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
    R, terminal = run_mini.e3_env.reward_and_terminal(acfg.goal_reward, acfg.step_reward)
    P = world.env.transition_tensor()
    was_training = rssm.training
    rssm.eval()
    rc = run_mini.real_context_settings(cfg)
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"], max_per_state=rc["max_per_state"],
                                      n_states=run_mini.N_STATES, rng=rng, generator=gen, max_images=run_mini.max_images_per_forward(cfg))
    out = {}
    readings = {"real_contexts": (ctx["h"], ctx["z"], ctx["states"])}
    h0, z0 = e3_probe.canonical_latents(rssm, world.templates_t)
    readings["canonical"] = (h0, z0, np.arange(run_mini.N_STATES))
    for name, (h, z, states) in readings.items():
        greedy = e3_probe.agent_greedy_policy(rssm, goal_head, critic, h, z, states, run_mini.N_STATES, acfg, gen)
        delta = e3_probe.delta_under_p(greedy["policy_probs"], P, R, terminal, acfg.gamma)
        out[name] = {"delta_by_zone": by_zone_mean(delta, world), "delta_mean": float(delta[~terminal].mean()),
                     "n_states_read": int(greedy["measured"].sum())}
    rssm.train(was_training)
    return out


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
            resume: bool = False, interrupt_after: int | None = None) -> dict:
    """One closed-loop run in run_dir. resume=True continues it from
    checkpoints/resume.pt (the config and the planned length are the run's
    own). interrupt_after (tests): raise Interrupted right after the resume
    point written at that environment step."""
    log = run_mini.RunLog(run_dir / "console.log", f"[agent seed={seed} {condition}]")
    meta = run_mini.environment_metadata(device)
    require_code_provenance(meta)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    acfg = agent_config(cfg)
    resume_path = run_dir / "checkpoints" / "resume.pt"
    if resume:
        write_run_meta(run_dir, seed, status="running", resumed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       **{f"resume_{k}": v for k, v in meta.items() if k in ("device", "git_commit", "source_sha256")})
        log(f"RESUMING from {resume_path}")
    else:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        write_run_meta(run_dir, seed, status="running", condition=condition, phase="5 closed loop", agent=dataclasses.asdict(acfg),
                       rng_streams=list(e3_rng.STREAM_NAMES), episode=run_mini.load_episode_config(), max_env_steps=max_env_steps, **meta)
    log(f"code: git commit {meta['git_commit']}, source sha256 {meta['source_sha256']}")

    world = run_mini.build_world(cfg, seed, condition)
    P_ref = (run_mini.build_world(cfg, seed, "reference", themes=world.themes).env.transition_tensor() if condition == "experimental"
             else world.env.transition_tensor())
    world.templates_t, world.background_t = world.templates_t.to(device), world.background_t.to(device)
    budget, prefill = cfg["budget"]["total_steps"], cfg["budget"]["prefill_steps"]
    planned_env_steps = budget + prefill
    n_env_steps = planned_env_steps if max_env_steps is None else min(planned_env_steps, max_env_steps)
    replay = ReplayBuffer.for_run(budget=stored_steps_bound(n_env_steps), prefill_steps=0, table=world.templates_np)

    streams, seqs = e3_rng.spawn_streams(seed, condition), e3_rng.spawn_sequences(seed, condition)
    gens = {name: e3_rng.torch_generator(seqs[seq], device)
            for name, seq in (("latent", "latent_sample"), ("act", "act_latent"), ("imagination", "imagination"))}
    torch.manual_seed(e3_rng.torch_seed(seqs["network_init"]))  # built on the CPU, then moved: same weights on every device
    rssm = compile_from_config(run_mini.build_rssm(cfg).to(device), cfg)  # runtime.compile: off unless set (acceleration.py)
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
    switch, n_pairs = run_mini.build_separation_switch(cfg)
    planned_grad_steps = int(round((planned_env_steps - prefill) * tr["replay_ratio"]))

    collector = ClosedLoopCollector(world.env, replay, rssm, goal_head, critic, acfg, run_mini.N_STATES, start_rng=streams["start_state"],
                                    explore_rng=streams["explore"], tiebreak_rng=streams["tiebreak"], act_generator=gens["act"], device=device)
    checkpoints = checkpoint_env_steps(cfg, n_env_steps)
    resume_every = cfg["checkpoints"].get("resume_every")
    monitor_every, window = cfg["log"]["monitor_every"], cfg["log"]["behaviour_window"]
    progress = {"env_step": 0, "grad_step": 0, "credit": 0.0, "last_stat": float("nan"), "rep_warmup": None, "last": {},
                "records": [], "counts": {}, "seconds": 0.0}

    def save_resume() -> None:
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
        run_mini._atomic_torch_save(state, resume_path)
        run_mini.write_checkpoint_pointer(run_dir, resume_path.name, progress["env_step"])

    if resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["planned_env_steps"] != planned_env_steps or state["n_env_steps"] != n_env_steps:
            raise ValueError("resume.pt was written for another budget")
        for module, key in ((rssm, "rssm"), (goal_head, "goal_head"), (critic, "critic"), (slow_critic, "slow_critic"),
                            (optimizer, "optimizer"), (goal_optimizer, "goal_optimizer"), (critic_optimizer, "critic_optimizer")):
            module.load_state_dict(state[key])
        for name, value in state["switch"].items():
            setattr(switch, name, value)
        replay = pickle.loads(state["replay"])
        collector.replay, collector.env = replay, pickle.loads(state["env"])
        world.env = collector.env
        c = state["collector"]
        collector.N, collector.stats = c["N"], EpisodeStats(**c["stats"])
        collector._obs, collector._s, collector._arrival_reward = c["obs"], c["s"], c["arrival_reward"]
        collector._is_first, collector._episode_length = c["is_first"], c["episode_length"]
        collector.filter.h, collector.filter.z, collector.filter.a_prev = (c["filter"][k].to(device) for k in ("h", "z", "a_prev"))
        _restore_streams(state["streams"], streams, gens)
        progress = state["progress"]
        (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(run_mini._jsonable(r)) + "\n" for r in progress["records"]),
                                                   encoding="utf-8")
        log(f"resumed at environment step {progress['env_step']} (gradient step {progress['grad_step']}, switch {switch.switch_step})")
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
    for env_step in range(progress["env_step"] + 1, n_env_steps + 1):
        collector.step(estimator=switch.estimator())
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
                progress["last_stat"] = mean_random_pair_jensen_shannon(out["post_probs"], streams["switch_pairs"], n_pairs)
                was_switched = switch.switched
                try:
                    switch.update(grad_step, progress["last_stat"])
                except CollapseAfterSwitchError as err:
                    fail(err)
                    raise
                if switch.switched and not was_switched:
                    progress["rep_warmup"] = (tr["rep_warmup_steps"] if tr["rep_warmup_steps"] is not None
                                              else int(round(0.2 * (planned_grad_steps - switch.switch_step))))
                    log(f">>> SWITCH to straight-through effective at gradient step {switch.switch_step} (batch-pair JS "
                        f"{progress['last_stat']:.3f} bits, environment step {env_step}); the agent's heads train from here")
            progress["grad_step"] += 1
            try:
                switch.check_deadline(progress["grad_step"])
            except SeparationNotReachedError as err:
                fail(err)
                raise
        progress["env_step"] = env_step

        if env_step % monitor_every == 0 or env_step in checkpoints:
            behaviour, stats = collector.stats.window(window), collector.stats
            tie = stats.tied_greedy_steps / max(stats.greedy_steps, 1)
            log(f"env step {env_step:6d}/{n_env_steps} | grad {progress['grad_step']:6d} | JS {progress['last_stat']:6.3f} | "
                f"{'straight-through' if switch.switched else 'probabilities   '} | success {behaviour['success_rate']:.3f} "
                f"len {behaviour['episode_length']:.1f} ({behaviour['n_episodes']} ep) | ties {tie:.3f} | recon_patch "
                f"{last.get('recon_patch', float('nan')):.4f} | goal_loss {last.get('goal_loss', float('nan')):.4f} critic_loss "
                f"{last.get('critic_loss', float('nan')):.4f} | {time.perf_counter() - t0:.0f}s")
        if env_step in checkpoints:
            stats = collector.stats
            record = {"env_step": env_step, "grad_step": progress["grad_step"], "switch_step": switch.switch_step,
                      "batch_pair_js": progress["last_stat"], "episodes": stats.episodes, "successes": stats.successes,
                      **collector.stats.window(window), "tie_fraction": stats.tied_greedy_steps / max(stats.greedy_steps, 1),
                      "explored_fraction": stats.explored_steps / max(stats.env_steps, 1), "stored_steps": len(replay),
                      "train": {k: float(v) for k, v in last.items()},
                      "twin_by_zone": twin_at_checkpoint(collector.N, world, P_ref, cfg, acfg, seed, condition, env_step),
                      "agent_delta": agent_delta_at_checkpoint(rssm, goal_head, critic, world, replay, cfg, acfg, seed, condition, env_step,
                                                               device)}
            progress["records"].append(record)
            progress["counts"][f"N_{env_step}"] = collector.N.copy()
            with open(run_dir / "checkpoints.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(run_mini._jsonable(record)) + "\n")
        if env_step < n_env_steps and (env_step in checkpoints or (resume_every and env_step % resume_every == 0)):
            progress["seconds"] = time.perf_counter() - t0
            save_resume()
            if interrupt_after is not None and env_step >= interrupt_after:
                raise Interrupted(f"simulated interruption after the resume point at environment step {env_step}")

    progress["seconds"] = time.perf_counter() - t0
    records, counts = progress["records"], progress["counts"]
    torch.save({"rssm": rssm.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg), "goal_head": goal_head.state_dict(),
                "critic": critic.state_dict(), "slow_critic": slow_critic.state_dict(), "agent_config": dataclasses.asdict(acfg),
                "env_step": n_env_steps, "step": progress["grad_step"], "switch_step": switch.switch_step},
               run_dir / "checkpoints" / "agent_final.pt")
    np.savez_compressed(run_dir / "counts.npz", **counts)
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
    if cfg["measure"].get("final", True):
        log("final measures of the world model in closed loop (run_mini.measure_final) ...")
        streams_m = run_mini.seed_streams(seed, device, condition)
        fields, arrays = run_mini.measure_final(rssm, world, replay, collector.N, cfg, streams_m, run_dir, log, progress["grad_step"])
        fields.pop("diagnostics_final")
        np.savez_compressed(run_dir / "arrays.npz", **arrays)
    report = {"seed": seed, "condition": condition, "zone_class": run_mini.e3_env.block_class_for_seed(seed),
              "zone_kinds": world.zone_kinds, "env_steps": n_env_steps, "planned_env_steps": planned_env_steps,
              "grad_steps": progress["grad_step"], "switch_step": switch.switch_step,
              "truncated_by_max_env_steps": n_env_steps < planned_env_steps, "seconds": progress["seconds"], "checkpoints": records,
              "reproducibility": meta, **fields}
    (run_dir / "results.json").write_text(json.dumps(run_mini._jsonable({k: v for k, v in report.items() if k not in run_mini.ARRAY_FIELDS}),
                                                     indent=1), encoding="utf-8")
    write_run_meta(run_dir, seed, status="done", env_steps=n_env_steps, grad_steps=progress["grad_step"], switch_step=switch.switch_step)
    log(f"run done in {progress['seconds']:.0f}s; results in {run_dir}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--out-dir", type=Path, default=run_mini.DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--conditions", nargs="+", choices=e3_rng.CONDITIONS, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None, help="caps the number of environment steps (smoke tests)")
    parser.add_argument("--resume", type=Path, default=None, help="continue the interrupted run in this directory")
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.resume:
        cfg = run_mini.load_mini_config(args.resume / "config.yaml")
        meta = yaml.safe_load((args.resume / "meta.yaml").read_text(encoding="utf-8"))
        if args.device is not None:
            cfg["runtime"]["device"] = args.device
        device = run_mini.configure_runtime(cfg)
        run_one(cfg, int(meta["seed"]), meta["condition"], args.resume, device, max_env_steps=meta.get("max_env_steps"), resume=True)
        return 0
    cfg = run_mini.load_mini_config(args.config)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    device = run_mini.configure_runtime(cfg)
    failures = 0
    for seed in (args.seeds or cfg["seeds"]):
        for condition in (args.conditions or cfg["conditions"]):
            run_dir = run_mini.make_seed_run_dir(args.out_dir, f"agent-{Path(args.config).stem}", seed, condition)
            try:
                run_one(cfg, seed, condition, run_dir, device, max_env_steps=args.max_env_steps)
            except (SeparationNotReachedError, CollapseAfterSwitchError) as err:
                failures += 1
                print(f"[run_agent] seed {seed} {condition} STOPPED by a training guard ({type(err).__name__}): {err}", flush=True)
            except BaseException as err:
                write_run_meta(run_dir, seed, status="failed", error=f"{type(err).__name__}: {err}")
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
