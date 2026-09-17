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

## Convention: alpha is the TOTAL prior mass per (s, a) pair

`alpha` (the Dirichlet prior hyperparameter, `helplessness.agents.tabular.TabularModelBasedAgent`'s
`alpha` argument together with an `alpha_support` mask) always denotes
the **total** pseudo-count mass spread over a (s, a) pair's support, not
a per-successor value:

```
T_hat(s' | s, a) = (N[s, a, s'] + alpha/5) / (N[s, a].sum() + alpha)
```

for the 5-state geographic support (`helplessness.envs.zones.geographic_support_mask`)
every dynamics type in this family uses. Each of the 5 reachable
successors receives a pseudo-count of exactly `alpha/5`; the sum of the
5 pseudo-counts is `alpha`, matching a single Dirichlet concentration
parameter, not `alpha` per successor (which would make the effective
prior mass `5*alpha`).

This was a real bug, not just a naming risk: `TabularModelBasedAgent.T_hat`'s
`alpha_support` branch originally put the *full* `alpha` on each support
cell (`alpha_support * alpha`), giving `5*alpha` total prior mass per
pair -- invisible at zero data (a symmetric prior's alpha cancels top and
bottom of the unvisited-pair formula, so the zero-data row is `1/5`
regardless of which convention is used), but wrong as soon as any real
counts exist. Fixed in `src/helplessness/agents/tabular.py`; see
`tests/agents/test_agent.py::test_alpha_support_prior_mass_is_alpha_not_a_raw_zero_one_mask`
and `tests/agents/test_e1_estimation.py::test_agent_from_counts_alpha_support_prior_is_exact`
for the regression pins (on the shared class directly, and through E1's
own `agent_from_counts` wrapper respectively).

## Configuration

Not yet defined -- depends on E1's zone layout being finalised first.
