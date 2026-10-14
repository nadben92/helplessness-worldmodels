# Contributing (to yourself, in four months)

This file exists so the structure survives future sessions of work on this
project -- including future sessions of the agent that set it up. See also
`CLAUDE.md` for the full set of code conventions and invariants; this file
covers the src/ vs experiments/ boundary and the results/versioning rules
specifically.

## The src/ vs experiments/ rule

`src/helplessness/` is reusable code. `experiments/` is disposable code that
uses it.

- **src/ must not import from experiments/.** Never. If an experiment
  script defines something a second experiment needs, that thing belongs in
  src/, not in a cross-import between two experiment directories.
- **experiments/ imports from src/, and from its own directory only.**
  A runner script may import a sibling file in the same experiment
  directory (e.g. `experiments/E0_offline_5x5/03_intervention/plot_intervention.py`
  importing `rescue_diagnostics.py` next to it), never a file living under
  a different experiment directory.
- **`src/helplessness/measures/` and `src/helplessness/interventions/` must
  not import the environment or agent classes.** They take arrays in and
  return arrays/scalars out. This is what lets them run unchanged on a
  learned belief from the future neural world model (roadmap step 2/3 in
  CLAUDE.md) -- verified at restructuring time to hold for the existing
  code, and worth re-checking by eye whenever you touch either package.
  `src/helplessness/agents/base.py::materialize_transition_tensor` is the
  bridge on the agent side precisely because measures/ itself can't import
  Agent -- see CLAUDE.md.
- Everything else in `src/` (envs, agents, planning, evaluation) may import
  freely within `src/`.

If you're not sure where something goes: will a second experiment need it?
If yes, src/. If it's a one-off analysis script, config, or a runner's own
plotting code, experiments/<name>/<NN_phase>/.

## Adding a new experiment

Copy the shape of an existing one: `run.py` (or `run.py` + a couple of
sibling scripts), a `config.yaml` with every parameter that varies, and a
`README.md` stating what was tested and what came out (fill in the second
half after running it, not before). See `docs/EXPERIMENTS.md`, the
central tracker (E0..E7), and the one-file-per-experiment docs -- update
both when an experiment's status changes.

## Adding an environment variant

Subclass `helplessness.envs.base.Env` (see `helplessness/envs/legacy/gridworld.py`
for the frozen 5x5 pattern, or `helplessness/envs/torus_env.py` for the
zone/dynamics/appearance family every new experiment should build on).
Every downstream module (agents, measures, evaluation) already goes
through the four array-producing methods (`transition_tensor`,
`reward_vector`, `terminal_mask`, `controllability`) plus the Gymnasium-
style `reset`/`step` -- if a call site needs something environment-specific
instead, that's a sign the interface needs to grow, not that the call site
should special-case the environment. A new dynamics type or appearance
category is a new entry in `envs/dynamics/` or `envs/appearance/`, not a
new Env subclass -- see CLAUDE.md.

## Results and configs

Every CSV a runner writes must have an adjacent `<name>.config.yaml`
recording exactly the parameters that produced it (see
`src/helplessness/config.py`, `dump_config`). Don't hand-edit a results CSV
without regenerating or hand-editing its config.yaml to match -- a CSV
without a trustworthy config next to it is worse than no CSV. For the
zone/dynamics/appearance family, `helplessness.config.compose()` produces
that same fully-resolved config from shared blocks + overrides -- dump the
result of compose(), never the pre-composition pieces.

## What gets versioned

Code, configs, `results/**/*.csv` (data is small here -- a few hundred KB to
a few MB), and final figures used in `reports/`. Not versioned: raw saved
agent tensors and world-model checkpoints (`results/**/snapshots/**/*.npz`,
`results/**/checkpoints/**` -- see `scripts/regenerate_agents.py` for the
offline agents), matplotlib scratch output, `__pycache__`, notebook
checkpoints.
