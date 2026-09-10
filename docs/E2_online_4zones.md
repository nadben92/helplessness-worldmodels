# E2 — Online validation, 4 zones

**Status: not started.** The agent/loop/recovery-protocol library exists
and is tested (`helplessness.agents.online`), validated on E0's 5x5
environment as a prototype (`experiments/E0_offline_5x5/05_online_prototype/`)
-- not yet run on the real environment.

## What it tests

The definitional criterion of learned helplessness (failure to *relearn*
that control has returned), on the same 4-zone environment as E1, once E1
has produced clean offline results (gating criterion, see CLAUDE.md: do
not move to the next step until the previous one produces clean results).

## Environment

Same as E1 (`helplessness.envs.torus_env.TorusZonedEnv`, 4 zones, tabular
backend).

## Agent

`helplessness.agents.online.agent.OnlineTabularAgent`: periodically
replanned (full value iteration, never a sampled Dyna-Q backup), epsilon-
greedy, Dirichlet prior restricted to geographically reachable successors.

## What carries over from the E0 prototype

- The training loop (`helplessness.agents.online.loop.train_online`,
  `train_online_with_burnin`) -- fully environment-agnostic already.
- The Phase-1 burn-in fix and its rationale (see
  `experiments/E0_offline_5x5/05_online_prototype/README.md`): a uniform-
  random preamble that fixed a real exploration deadlock at the task's
  default epsilon=0.1, applied identically across every training condition.
- The recovery-protocol mechanics (`helplessness.agents.online.recovery`:
  `RecoveryTracker`, `recovery_time`) -- environment-agnostic, takes an
  `eval_env_factory` rather than assuming GridWorld.
- `geographic_support()` will need reworking: on the new torus (no walls),
  "the 4 neighbours + self" is directly computable from grid geometry --
  simpler than the p=1-trick the prototype used, which assumed a single
  global noise parameter that no longer exists once dynamics are per-zone.

## Configuration

Not yet defined -- depends on E1's zone layout being finalised first.
