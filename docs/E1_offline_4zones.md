# E1 — Offline validation, 4 zones

**Status: not started.** Foundation built (this restructuring), no
experiment code written yet.

## What it tests

Validates the new base environment (`helplessness.envs.torus_env.TorusZonedEnv`)
and its zone/dynamics/appearance framework against the same offline
protocol E0 already validated on the 5x5 gridworld -- the first checkpoint
before moving to the online agent (E2) or the world model (E3).

## Environment

Base environment (torus, no walls), 4 distinct zones (masks, dynamics,
appearance to be specified when this experiment is designed). Tabular
(index) observation backend.

## Agent

Tabular model-based agent (`helplessness.agents.tabular.TabularModelBasedAgent`),
same as E0.

## Configuration

Not yet defined: grid size, the 4 zones' masks/dynamics/appearance, and
which (if any) is the reference zone are all open. Per CLAUDE.md, none of
these get a default baked into the code -- they must be explicit in this
experiment's own config when it is written.

## Dependencies

`helplessness.envs.{zones,dynamics,appearance,observation,torus_env}`,
`helplessness.measures` (unchanged from E0), `helplessness.agents.tabular`
(unchanged from E0).
