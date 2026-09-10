# Online prototype (E0, 5x5) -- not a numbered paper experiment

This is where the online model-based agent (periodic replanning, Phase-1/2
recovery protocol) was built and validated, on the legacy 5x5 GridWorld,
while the actual zoned/torus environment (E1+) didn't exist yet. **The
library code is what carries over** -- `helplessness.agents.online`
(agent, loop, recovery mechanics) -- imported here, not copy-pasted. This
directory is the GridWorld-specific glue and CLI on top of it, kept as a
reproducible reference, not as E2 itself: E2 reruns this protocol on the
new environment when it exists.

The online agent's policy determines the data it collects: a periodically
replanned (full value iteration, never a sampled Dyna-Q backup), tabular
model-based agent. See `helplessness.agents.online.agent` (model + policy)
and `.loop` (the training loop, plus the Phase-1 burn-in below).

## Status

The agent, loop, and Phase-1/2 recovery protocol are implemented and
tested (see `tests/agents/online/` and `tests/agents/test_base.py` at the
repo root). A 3-seed validation subset of the recovery protocol has been
run (p_train=0.95 vs naive, all four conditions) -- see
`results/online/recovery_summary.csv`. The full 10-seed x 2-budget sweep
was never run: superseded by the move to the zoned environment.

## The Phase-1 burn-in

At the task's literal defaults (epsilon=0.1, replan_every=500, 50k
transitions), a **fresh** agent trained at p_train=0 -- the fully
controllable, no-slip "healthy" world, where nothing about the task itself
is hard -- converges **bimodally**: over 20 seeds, 8 (40%) never discovered
the goal at all and stayed permanently stuck. Confirmed not to be "slow":
6x the budget grew visits to the blocking state only 21 -> 42, not
proportionally -- a self-reinforcing avoidance, not a rate problem.

Mechanism: the default layout's two routes to the goal both pass through a
cell adjacent to a trap. The Dirichlet prior restricted to geographically
reachable successors (required by the task -- see
`helplessness.agents.online.agent.geographic_support`) is necessarily
per-state, not per-action, so an untried *safe* action from a
trap-adjacent cell inherits the same pessimism as the actual unsafe one,
until that specific action is sampled. At epsilon=0.1, with that cell
visited only rarely once the greedy policy starts avoiding it, the
unlocking sample can fail to occur within budget.

Fix (per-user direction, see conversation): a **Phase-1-only** burn-in --
`burn_in_steps` transitions under a uniform random policy over all 5
actions, reusing `helplessness.agents.collect.collect` (imported, not
copy-pasted), before the epsilon-greedy loop begins. It counts against the
existing budget (never added on top) and is applied identically to every
p_train condition, so it introduces no asymmetry between them. Empirically:
19/20 seeds converge with `burn_in_steps=2000` (up from 12/20 without),
consistent with a finite random burn-in rather than an absolute coverage
guarantee -- occasional single-seed residual failures are expected and
average out over the 10 seeds/condition design.

**Deliberately not applied to Phase 2 (recovery).** The same deadlock there
is part of what is being measured: if the naive group also gets stuck under
epsilon=0.1, that is itself informative (slowness is generic to this agent
in this environment, not caused by training at high p), not a bug to patch
away. Kept as a reportable result: an agent can develop passivity that is
behaviourally indistinguishable from learned helplessness in a fully
controllable world, purely from epistemic pessimism near a known danger --
which is exactly why the aetiology in `CLAUDE.md` (phase 3, component
swaps) and the naive-vs-inescapable comparison (not escapable-vs-inescapable)
matter: behaviour alone never settles the question.

## Running

From this directory (same convention as every other experiment under
`experiments/E0_offline_5x5/`):

```
python train.py --p-train 0.0 --seed 1
python train.py --seed 1 --naive
python recovery.py --p-train 0.95 --seeds 0 1 2 --budgets 50000
```

Writes `results/online/phase1.csv` / `results/online/recovery.csv` plus
adjacent `.config.yaml` files (resolved config, seed, git commit hash) --
see `helplessness.config.OnlineTrainConfig` / `RecoveryConfig`.
