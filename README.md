# helplessness

## Research question

When an agent learns that some region of its environment is beyond its
control, according to what geometry does that belief propagate to regions
that actually are controllable? The project studies this in a model-based
reinforcement learning agent, where a transition belief `T_hat` and a
reward belief `R_hat` are structurally separate -- which makes perceived
controllability (measured as `I(action ; next_state | state)`) directly
readable from the belief, and, as it turns out, directly editable. Work
progresses from a tabular agent on a small gridworld, to a tabular agent on
a richer environment with heterogeneous, zone-level controllability, to a
neural world model on the same environment rendered as pixels, ending in
causal intervention on the world model's latent representation.

See `CLAUDE.md` for the code-level conventions and invariants, and `docs/`
for what each experiment (E0 through E6) tests and its current status.

## Repository structure

```
src/helplessness/
  envs/          Environments. legacy/gridworld.py is the frozen 5x5 (E0).
                 zones.py, dynamics/, appearance/, observation/, torus_env.py
                 are the zone-based family (E1 onwards): dynamics and
                 appearance are independent per zone, torus topology, an
                 interchangeable tabular/pixel observation backend.
  agents/        tabular.py (counts + value iteration), online/ (periodic
                 replanning + the Phase-1/2 recovery protocol), worldmodel/
                 (reserved, E3+). base.py declares the one thing every
                 agent must expose: transition_distribution().
  measures/      I(a;s'|s) -- one implementation, used everywhere.
  interventions/ Causal manipulation of a transition belief.
  planning/, evaluation/, config.py, runs.py
configs/         Shared YAML blocks (zones, themes, dynamics presets) for
                 experiments to compose, some of them frozen (see CLAUDE.md).
experiments/     One directory per experiment; orchestration only.
docs/            One file per experiment: what it tests, its config, its status.
results/         Run outputs -- see "where results land" below.
reports/         Write-ups and figures. Start with week1_offline_report.md.
tests/           Mirrors src/.
```

## Running an experiment

```bash
pip install -e .
pytest
```

E0 (offline, 5x5 -- done):

```bash
python scripts/regenerate_agents.py   # trains and saves the tabular agents
make sweep swaps intervention rescue  # or: cd experiments/E0_offline_5x5/<phase> && python run.py --config config.yaml
```

Each experiment directory has its own `README.md` with exact commands and
dependencies between phases. An experiment built on the zone/dynamics/
appearance family composes its config via `helplessness.config.compose()`
and writes results under the naming convention in `CLAUDE.md`.

## Where results land

E0 (`experiments/E0_offline_5x5/`): `results/<name>.csv` plus an adjacent
`<name>.csv.config.yaml` recording exactly what produced it. Raw agent
tensors (`results/agents/`) are regenerable, not versioned.

E1 onwards: `results/<experiment_id>/<run_name>/`, one self-contained
directory per run (`config.yaml`, `meta.yaml`, metrics, `checkpoints/`,
`artifacts/`) -- see `CLAUDE.md`, "results and run naming", for the exact
convention and what `run_name` encodes.
