# Online experiment

Separate from `experiments/` (the offline pipeline, protocol A) and from
every offline result -- see the root `CLAUDE.md` for the scientific
question and `CONTRIBUTING.md` for the src/ vs experiments/ split this
package follows (imports shared primitives from `helplessness.*`, never
copy-pastes them).

The online agent's policy determines the data it collects: a periodically
replanned (full value iteration, never a sampled Dyna-Q backup), tabular
model-based agent. See `agent.py` (model + policy) and `loop.py` (the
training loop, plus the Phase-1 burn-in below).

## Status

Step 1 (5-action GridWorld) and step 2 (the online agent and loop) are
implemented and tested; step 3 (the two-phase protocol and recovery 2x2)
awaits approval before running the full sweep.

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
reachable successors (required by the task -- see `agent.geographic_support`)
is necessarily per-state, not per-action, so an untried *safe* action from a
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

From the repo root (needed for `from online.* import ...` and the installed
`helplessness` package to resolve):

```
python -m online.train --p-train 0.0 --seed 1
```

Writes `results/online/train_validation.csv` plus an adjacent
`.config.yaml` (resolved config, seed, git commit hash) -- see
`helplessness.config.OnlineTrainConfig`.
