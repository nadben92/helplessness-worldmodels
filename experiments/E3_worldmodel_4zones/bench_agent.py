"""Speed bench of the closed-loop agent (run_agent.py) in a given configuration
(docs/e3/DECISIONS.md, 2026-10-02 (19)). INFORMATIVE: it never enters a
preflight verdict; it measures what the multi-seed runner and the campaign
budgets will be decided on. The pilot itself runs without any acceleration.

Variants, each timed after a warm-up (its duration reported as the
compilation time), on fresh models and a replay filled by the agent's own
collector (straight-through phase, the long one):
  world_model          train_world_model_step, the run's batch shape;
  heads                train_heads_step (goal head, critic in imagination);
  env                  ClosedLoopCollector.step (filter, one-step lookahead, act);
  stacked_<n>          stacked_train_step with n = 1, 2, 4, 8 seeds (acceleration.py);
  compiled             world-model step with runtime.compile (inductor), only if
                       the equivalence gate (acceleration.check_compile_equivalence)
                       passes here -- otherwise its failure is reported;
  stacked_<n>_compiled the stacked step with its vectorised calls compiled, same gate.
Per variant: seed-steps per second, warm-up seconds, and the largest relative loss
difference from the plain world-model step over a few identical steps (same weights,
batches, generators). A time budget (--budget-minutes) stops the remaining variants,
which are reported as skipped.

--loop SECONDS: instead, run the closed loop's inner loop (an environment step and
replay_ratio x (world-model + heads) steps) for SECONDS and print environment steps per
second -- what the preflight runs in 2 and 4 parallel processes to choose the parallelism.

Usage: python experiments/E3_worldmodel_4zones/bench_agent.py --config configs/e3/gpu_pilot_closed_loop.yaml --device cuda [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel import acceleration as acc
from helplessness.agents.worldmodel.closed_loop import ClosedLoopCollector, stored_steps_bound
from helplessness.agents.worldmodel.heads import Critic, GoalHead, make_slow_critic
from helplessness.agents.worldmodel.imagination import train_heads_step
from helplessness.agents.worldmodel.loop import WorldModelLossConfig, train_world_model_step
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import LatentEstimator
from helplessness.experiment_loader import load_experiment_module

run_agent = load_experiment_module("E3_worldmodel_4zones", "run_agent.py", "bench_agent_run_agent")
run_mini = run_agent.run_mini
ST = LatentEstimator("straight_through")
N_CHECK = 4


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Setup:
    """Fresh models, optimisers and a replay of `fill` steps collected by the agent's own collector."""

    def __init__(self, cfg: dict, device: torch.device, fill: int = 3000, seed: int = 0):
        self.cfg, self.device = cfg, device
        self.acfg = run_agent.agent_config(cfg)
        self.world = run_mini.build_world(cfg, 0, "experimental")
        self.world.templates_t, self.bg = self.world.templates_t.to(device), self.world.background_t.to(device)
        tr = cfg["train"]
        self.B, self.T = tr["batch_size"], tr["seq_len"]
        self.loss_cfg = WorldModelLossConfig(k=cfg["render"]["k"], w_patch=tr["w_patch"], beta_dyn=tr["beta_dyn"], beta_rep=tr["beta_rep"])
        self.rssm = self.model(seed)
        self.goal = GoalHead(self.rssm.cfg.h_dim, self.rssm.cfg.latent_dim, self.acfg.hidden_dim).to(device)
        self.critic = Critic(self.rssm.cfg.h_dim, self.rssm.cfg.latent_dim, self.acfg.hidden_dim).to(device)
        self.slow = make_slow_critic(self.critic)
        self.replay = ReplayBuffer.for_run(budget=stored_steps_bound(fill + 2000), prefill_steps=0, table=self.world.templates_np)
        rngs = [np.random.default_rng(seed + k) for k in range(3)]
        self.collector = ClosedLoopCollector(self.world.env, self.replay, self.rssm, self.goal, self.critic, self.acfg, run_mini.N_STATES,
                                             rngs[0], rngs[1], rngs[2], torch.Generator(device=device).manual_seed(seed), device)
        for _ in range(fill):
            self.collector.step(estimator=ST)
        if device.type == "cuda":
            self.replay.to_device(device)
        self.rng = np.random.default_rng(seed + 10)
        self.check_batches = [self.replay.sample(self.B, self.T, np.random.default_rng(99))[0] for _ in range(N_CHECK)]

    def model(self, seed: int):
        torch.manual_seed(seed)
        return run_mini.build_rssm(self.cfg).to(self.device)

    def batch(self):
        return self.replay.sample(self.B, self.T, self.rng)[0]


def _wm_steps(s: Setup, model, n: int, batches=None) -> list[dict]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
    gen = torch.Generator(device=s.device).manual_seed(0)
    out = []
    for i in range(n):
        mb = batches[i] if batches is not None else s.batch()
        out.append(train_world_model_step(model, opt, mb, s.bg, s.loss_cfg, clip_norm=s.cfg["train"]["clip_norm"], estimator=ST, generator=gen,
                                          sync=batches is not None))
    return out


def _timed(fn, n_warm: int, n: int, device) -> tuple[float, float]:
    t0 = time.perf_counter()
    for _ in range(n_warm):
        fn()
    _sync(device)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / n, warm


def bench(cfg: dict, device: torch.device, n: int = 100, budget_minutes: float = 30.0, platform: str | None = None) -> dict:
    t_start = time.perf_counter()
    s = Setup(cfg, device)
    out: dict = {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "batch": [s.B, s.T], "timed_steps": n, "setup_seconds": time.perf_counter() - t_start, "variants": {}}
    reference = _wm_steps(s, s.model(0), N_CHECK, s.check_batches)

    def over_budget() -> bool:
        return (time.perf_counter() - t_start) / 60.0 > budget_minutes

    def record(name, fn, seeds=1, check=None):
        if over_budget():
            out["variants"][name] = {"skipped": f"time budget of {budget_minutes:g} min reached"}
            return
        try:
            per_call, warm = _timed(fn, 5, n, device)
            entry = {"seed_steps_per_s": seeds / per_call, "seconds_per_call": per_call, "warmup_seconds": warm}
            if check is not None:
                entry["max_rel_loss_diff"] = acc.max_relative_difference(check(), reference)
            out["variants"][name] = entry
        except Exception as err:  # informative bench: a failing variant is reported, never fatal
            out["variants"][name] = {"error": f"{type(err).__name__}: {str(err).splitlines()[0][:300] if str(err) else ''}"}

    # plain world-model step, heads step, environment step
    wm_model = s.model(0)
    wm_opt = torch.optim.Adam(wm_model.parameters(), lr=1e-4, eps=1e-8)
    gen = torch.Generator(device=device).manual_seed(0)
    record("world_model", lambda: train_world_model_step(wm_model, wm_opt, s.batch(), s.bg, s.loss_cfg, clip_norm=None, estimator=ST,
                                                        generator=gen, sync=False))
    state = train_world_model_step(s.rssm, torch.optim.Adam(s.rssm.parameters(), lr=1e-4), s.batch(), s.bg, s.loss_cfg, estimator=ST,
                                   generator=gen, return_states=True)
    mb = s.batch()
    gopt = torch.optim.Adam(s.goal.parameters(), lr=1e-4)
    copt = torch.optim.Adam(s.critic.parameters(), lr=3e-5)
    r = torch.as_tensor(np.asarray(mb.r), device=device)
    record("heads", lambda: train_heads_step(s.rssm, s.goal, s.critic, s.slow, gopt, copt, state["h"], state["z"], r, s.acfg, gen))
    record("env", lambda: s.collector.step(estimator=ST))

    # stacked seeds, eager
    def stacked(nseeds, compiled=False):
        stack = acc.StackedRSSM([s.model(k) for k in range(nseeds)])
        if compiled:
            stack.compile()
        opt = acc.StackedAdam(stack.parameters(), nseeds, lr=1e-4, eps=1e-8)
        gens = [torch.Generator(device=device).manual_seed(0) for _ in range(nseeds)]
        call = lambda batches: acc.stacked_train_step(stack, opt, batches, [s.bg] * nseeds, [s.loss_cfg] * nseeds, [ST] * nseeds,  # noqa: E731
                                                                 gens, clip_norm=s.cfg["train"]["clip_norm"])
        return stack, opt, gens, call

    for nseeds in (1, 2, 4, 8):
        def make_check(ns=nseeds, compiled=False):
            def check():
                _, _, _, call = stacked(ns, compiled)
                return [call([b] * ns)[0] for b in s.check_batches]
            return check
        _, _, _, call = stacked(nseeds)
        record(f"stacked_{nseeds}", lambda call=call, ns=nseeds: call([s.batch() for _ in range(ns)]), seeds=nseeds, check=make_check())

    # torch.compile: only behind the equivalence gate (Linux / CUDA)
    platform = sys.platform if platform is None else platform
    if not (platform.startswith("linux") and device.type == "cuda"):
        out["compile_gate"] = {"passed": False, "error": f"not run: the inductor gate runs on Linux / CUDA only (here {platform} / {device.type})"}
    else:
        out["compile_gate"] = acc.check_compile_equivalence(lambda: run_mini.build_rssm(cfg), (s.B, s.T), device, backend="inductor",
                                                            k=cfg["render"]["k"])
    if out["compile_gate"].get("passed"):
        torch._dynamo.reset()
        cm = acc.compile_rssm(s.model(0))
        copt2 = torch.optim.Adam(cm.parameters(), lr=1e-4, eps=1e-8)
        record("compiled", lambda: train_world_model_step(cm, copt2, s.batch(), s.bg, s.loss_cfg, clip_norm=None, estimator=ST, generator=gen,
                                                         sync=False),
               check=lambda: _wm_steps(s, acc.compile_rssm(s.model(0)), N_CHECK, s.check_batches))
        for nseeds in (1, 2, 4, 8):
            torch._dynamo.reset()
            _, _, _, call = stacked(nseeds, compiled=True)

            def check(ns=nseeds):
                torch._dynamo.reset()
                _, _, _, c2 = stacked(ns, compiled=True)
                return [c2([b] * ns)[0] for b in s.check_batches]
            record(f"stacked_{nseeds}_compiled", lambda call=call, ns=nseeds: call([s.batch() for _ in range(ns)]), seeds=nseeds, check=check)
    else:
        for name in ["compiled", *[f"stacked_{k}_compiled" for k in (1, 2, 4, 8)]]:
            out["variants"][name] = {"skipped": f"compile gate failed: {out['compile_gate'].get('error')}"}
    out["seconds"] = time.perf_counter() - t_start
    v = out["variants"]
    if all("seconds_per_call" in v.get(k, {}) for k in ("world_model", "heads", "env")):
        ratio = cfg["train"]["replay_ratio"]
        per_env = v["env"]["seconds_per_call"] + ratio * (v["world_model"]["seconds_per_call"] + v["heads"]["seconds_per_call"])
        out["one_process_seconds_per_env_step"] = per_env
    return out


def loop(cfg: dict, device: torch.device, seconds: float) -> dict:
    """The closed loop's inner loop for `seconds`: environment steps per second of one process."""
    s = Setup(cfg, device, fill=500)
    opt = torch.optim.Adam(s.rssm.parameters(), lr=1e-4, eps=1e-8)
    gopt = torch.optim.Adam(s.goal.parameters(), lr=1e-4)
    copt = torch.optim.Adam(s.critic.parameters(), lr=3e-5)
    gen = torch.Generator(device=device).manual_seed(0)
    ratio, credit = cfg["train"]["replay_ratio"], 0.0
    t0, steps = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds:
        s.collector.step(estimator=ST)
        credit += ratio  # as run_agent.py: replay_ratio gradient steps per environment step, fractional ratios carried over
        while credit >= 1.0:
            credit -= 1.0
            mb = s.batch()
            o = train_world_model_step(s.rssm, opt, mb, s.bg, s.loss_cfg, estimator=ST, generator=gen, sync=False, return_states=True)
            train_heads_step(s.rssm, s.goal, s.critic, s.slow, gopt, copt, o["h"], o["z"], torch.as_tensor(np.asarray(mb.r), device=device),
                             s.acfg, gen)
        steps += 1
    _sync(device)
    return {"env_steps_per_s": steps / (time.perf_counter() - t0), "pid": os.getpid()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=100, help="timed calls per variant")
    parser.add_argument("--budget-minutes", type=float, default=30.0)
    parser.add_argument("--loop", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = run_mini.load_mini_config(args.config)
    cfg["runtime"]["device"] = args.device
    device = run_mini.configure_runtime(cfg)
    result = loop(cfg, device, args.loop) if args.loop is not None else bench(cfg, device, args.steps, args.budget_minutes)
    result = run_mini._jsonable(result)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / ("loop.json" if args.loop is not None else "bench_agent.json")).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
