# E0 — Offline, 5x5 gridworld

**Status: done.** Terminal state, kept exactly as it was, never to be
modified again except to fix a reported bug.

## What it tests

Whether an agent's perceived controllability of its training environment
causally determines its behaviour when evaluated in a fully controllable
("healthy") environment, and whether that quantity is directly readable
from and editable in a model-based agent's transition belief.

## Environment

`helplessness.envs.legacy.gridworld.GridWorld`: 5x5 layout, one scalar
noise parameter `p` (uniform over the whole grid, not per-zone), bounded
plateau with walls (not the torus the new environment family uses), 4 or 5
actions (stay added after the phase below was written; see
`tests/envs/test_gridworld_stay_action.py`).

## Agent

`helplessness.agents.tabular.TabularModelBasedAgent`: Dirichlet-count
transition/reward beliefs, planning by value iteration.

## Configuration

Full sweep: `p in {0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0}`, 10 seeds,
budgets `{5000, 50000}` transitions, `alpha=0.1`, `gamma=0.95`. Exact
resolved configs for every run are in `results/*.csv.config.yaml`.

## Phases (`experiments/E0_offline_5x5/`)

1. `01_offline_sweep/` — training + evaluation across the full `p` sweep.
2. `02_swaps/` — component swaps (T_hat/R_hat) to localise the phenotype.
3. `03_intervention/` — causal manipulation of T_hat (the controllability
   dial), induction and rescue.
4. `04_rescue_injection/` — structure injection as a second rescue route,
   compared against the dial.
5. `05_online_prototype/` — **not a numbered paper phase**: where the
   online model-based agent (periodic replanning, Phase-1/2 recovery
   protocol) was built and validated against this same 5x5 environment,
   before the zoned/torus environment existed. The reusable logic moved to
   `helplessness.agents.online`; this directory is the GridWorld-specific
   glue kept as a reproducible reference. E2 reruns the protocol on the
   real environment, not this one.

## Write-up

`reports/week1_offline_report.md` — full findings, all figures.

## Reproducing

```
pip install -e .
python scripts/regenerate_agents.py
make sweep swaps intervention rescue
```

See each phase's own `README.md` for exact commands and dependencies
between phases.
