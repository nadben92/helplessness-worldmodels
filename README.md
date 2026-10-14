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

See `CLAUDE.md` for the code-level conventions and invariants, and
`docs/EXPERIMENTS.md` for every experiment (E0 through E7): its status,
objective, entry point and report.

## Experiments

The experiments form one chain. Each step checks that the measurement
still works before the next step changes the agent or the environment.
E0-E3 are finished and frozen; the project was reframed on 2026-10-14
(E4-E7 below, deterministic dynamics only; the previous E4-E6 plan is in
`docs/archive/plan_2026-09/`). Status as of 2026-10-14.

| | Experiment | Agent | Environment | Status |
|---|---|---|---|---|
| E0 | Offline, 5x5 | tabular | 5x5 gridworld, global noise `p` | done |
| E1 | Offline validation, 4 zones | tabular (offline) | 12x12 torus, 4 zones | done |
| E2 | Online validation, 4 zones | tabular (online) | same as E1 | done |
| E3 | World model, 4 zones | RSSM world model | same as E1, rendered as pixels | done |
| E4 | Generalisation gradient | world model | 2 deterministic zones (controllable / uncontrollable) + derived reference; zero-data probes | not started |
| E5 | Manipulating the belief through training | world model | E4's, with varied training histories | not started |
| E6 | Behavioural phenotypes and latent organisation | E4/E5's agents | fully controllable test battery | not started |
| E7 | Direct manipulation of the latent | E4-E6's world models | E4's probes, E6's battery | not started |

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
counterfactual re-renders). Finished: phases 0-5 (closed-loop agent
included), the GPU batches and the final campaign, analysed. Details:
`docs/E3_worldmodel_4zones.md` (results section), `paper/reports/e3_report.tex`,
decisions in `docs/e3/DECISIONS.md`.

**E4 -- generalisation gradient.** Two deterministic zones (one
controllable, one uncontrollable, on a diagonal) and the derived reference.
Zero-data probes on fully controllable test environments, changing the
theme of one quadrant along RGB axes at three positions: is the belief of
controllability carried by position, appearance, both, or neither?
`docs/E4_generalization_gradient.md`.

**E5 -- manipulating the belief through training.** Prior exposure to
control, similar themes across opposite controllabilities, latent
capacity, measured with E4's probes. `docs/E5_belief_training.md`.

**E6 -- behavioural phenotypes and latent organisation.** A test battery
in a fully controllable environment with norms from the reference agents,
RSA on the latent, and the link between latent geometry and phenotype.
`docs/E6_phenotypes_latent.md`.

**E7 -- direct manipulation of the latent.** For example, interpolating
zone codes, applied both to the belief (E4's probes) and to E6's
phenotypes: the final causal test. `docs/E7_latent_manipulation.md`.

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
  streams/, twin/, probing/, pipeline/, cluster/
                 Promoted from E2/E3 on 2026-10-14: random streams, the
                 tabular twin, the torch probes, the world-model run
                 pipeline, the GPU cluster tooling (see CLAUDE.md);
                 envs/layouts/ holds the 4-zone layouts.
  planning/, evaluation/, config.py, runs.py
configs/         Shared YAML blocks (zones, themes, dynamics presets) for
                 experiments to compose, some of them frozen (see CLAUDE.md).
experiments/     One directory per experiment; orchestration only.
docs/            EXPERIMENTS.md (the tracker), one file per experiment, e3/ (E3's
                 decisions and reports), archive/ (abandoned plans).
results/         Run outputs -- see "where results land" below.
reports/         Frozen outputs of E0 and E3 (see reports/README.md).
paper/           LaTeX paper and reports, figure and number scripts.
archive/         Retired material (notebooks/).
tests/           Mirrors src/.
```

## Running an experiment

```bash
pip install -e .
pytest
```

E0 (offline, 5x5 -- done):

```bash
cd experiments/E0_offline_5x5/<phase> && python run.py --config config.yaml   # see experiments/E0_offline_5x5/README.md
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
convention and what `run_name` encodes. From E4 on, what a GPU batch
returns goes to an orphan branch `results/<batch>`; weights and buffers stay
on the cluster (`docs/RESULTS_CONVENTION.md`).
