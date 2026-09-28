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

## Experiments

The seven experiments form one chain. Each step checks that the
measurement still works before the next step changes the agent or the
environment. Status as of 2026-09-28.

| | Experiment | Agent | Environment | Status |
|---|---|---|---|---|
| E0 | Offline, 5x5 | tabular | 5x5 gridworld, global noise `p` | done |
| E1 | Offline validation, 4 zones | tabular (offline) | 12x12 torus, 4 zones | done |
| E2 | Online validation, 4 zones | tabular (online) | same as E1 | sweep done |
| E3 | World model, 4 zones | RSSM world model | same as E1, rendered as pixels | in progress |
| E4 | Generalization pilot | world model | 1 contrasted zone pair + derived control | not started |
| E5 | Generalization campaign | world model | 2 contrasted pairs + 2 derived controls | not started |
| E6 | Latent perturbations | E5's trained world models | E5's environments | not started |

**E0 -- proof of concept (5x5 gridworld).** Does perceived
controllability, read from the tabular agent's `T_hat`, causally drive
behaviour in a fully controllable test environment? A training sweep over
the noise level `p`, `T_hat`/`R_hat` swaps, a controllability "dial" on
`T_hat` (to induce and to rescue the phenotype), and structure injection
as a second rescue. Write-up: `reports/week1_offline_report.md`.

**E1 -- validate the instrument on the new environment.** Introduces the
base environment used from here on: a 12x12 torus with a controllable
cross-shaped corridor and four 5x5 zones, one per dynamics type
(controllable or uncontrollable, deterministic or noisy). E1 does not
look for helplessness. With offline, fixed-policy data, it checks that the
measurement chain (`I_hat`, `b`, `Delta`) recovers the analytic
controllability of each zone, and measures how precise it is as a function
of the data budget. Result: `b` matches the analytic value in every zone.
The recommended budget for later experiments is n=100 visits per
state-action pair. It also fixes `gamma=0.9`, `R=50` and the drift
direction. Details: `docs/E1_offline_4zones.md`.

**E2 -- close the loop (online tabular agent).** Same environment, but the
agent now collects its own data (epsilon-greedy, periodic value
iteration). The question: does learning online, where the agent's beliefs
decide where it samples, bias the controllability estimate compared with a
matched control that has exactly the same visit counts drawn passively
from the true dynamics? The sweep (alpha in {0.5, 1, 5} x epsilon in
{0.1, 1}, 20 counterbalanced seeds, 1e6 steps) finds a small bias
(~0.01-0.04 bits). It appears only in the controllable-noisy zone,
concentrates on the optimal action, and disappears at epsilon=1. The
uncontrollable zones converge to their analytic values. This gives E3 its
null model. Results: `experiments/E2_online_4zones/RESULTS.md`.

**E3 -- the neural step (world model on pixels).** The same four zones,
rendered as images with a colour theme per zone, learned by a simplified
DreamerV3-style RSSM (`src/helplessness/agents/worldmodel/`). The question:
does the world model's learned dynamics encode controllability, measured
as `I(a; z'|context)` in latent space, and does it match the tabular twin
on the same data? The experiment also checks whether that code is driven
by *position* or by *appearance* (the `alpha_app` index, from
counterfactual re-renders). That diagnostic decides whether E4/E5 can run
at all. Done so far: phases 0-3, the probabilities-phase fix for
posterior collapse, a long CPU diagnostic run (`mini_long`), the theme
sensitivity study, and a main measure taken in real contexts. The first
GPU runs (E3-0) are prepared. The agent policy (phase 5) is not built
yet. Details: `docs/E3_worldmodel_4zones.md`, `docs/e3/`.

**E4 -- generalization pilot.** A pilot for E5. One contrasted pair of
zones (e.g. controllable vs uncontrollable, same appearance family) plus
its control, derived automatically by `derive_control`. Does a
controllability belief learned in one zone carry over to zones that
*look* similar?

**E5 -- generalization campaign.** The full version: two contrasted pairs
and two derived controls, crossing dynamics and appearance freely. This
crossing is the reason the two are decoupled in the framework. The
question: does the controllability axis found in E3/E4 generalize across
appearance families, and along what geometry does the belief spread?

**E6 -- causal test (latent perturbations).** Steer E5's trained world
models along the controllability axis found in their latent space, and
check whether this reproduces E0's phenotype (worse returns,
perseveration, trap avoidance). This is the neural version of E0's dial
and structure injection, and it reuses `interventions/` unchanged. E6 is
the step that shows the latent information is *used*, not just decodable.

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
                 (E3's RSSM, losses, replay, training loop). base.py
                 declares the one thing every agent must expose:
                 transition_distribution().
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

E1, E2, E3: see `experiments/E1_offline_4zones/README.md`,
`experiments/E2_online_4zones/README.md` and
`experiments/E3_worldmodel_4zones/README.md` (GPU runs:
`experiments/E3_worldmodel_4zones/RUN_GPU.md`).

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
