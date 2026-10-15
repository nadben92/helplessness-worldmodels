Code conventions

Python 3.10+, numpy. No heavy dependencies in the tabular phases (matplotlib, scipy, pyyaml are already in pyproject.toml; add nothing else without a reason tied to a specific experiment). torch is the `e3` extra: only `agents/worldmodel/`, `probing/`, `pipeline/`, `streams/worldmodel.py` and the world-model experiments import it.
Gymnasium-compatible API (reset -> (obs, info), step -> (obs, reward, terminated, truncated, info)) without a mandatory gymnasium import.
Type hints on public signatures.
English docstrings and comments throughout the codebase; keep that style for consistency.
No external RL framework. Tabular agents must stay readable line by line.
measures/ and interventions/ must have no side effects, and must never import an environment or agent class, nor torch -- see "the single source of truth" below.

Structure

```
src/helplessness/                 the shared, importable library
  envs/
    base.py              Env ABC: reset, step, transition_tensor, reward_vector, terminal_mask, controllability.
    legacy/gridworld.py  E0's 5x5 environment. Frozen.
    zones.py             Zone, ZoneLayout, the reference zone.
    dynamics/            DynamicsSpec registry (4 types) + pairing (derive_control) + factory (YAML -> object).
    appearance/          AppearanceSpec, themes, CIELAB theme sampling, theme_axes.py (E4's colour axes).
    observation/         ObservationBackend: index (tabular) / pixel (world model).
    rewards/             GoalReward, ZeroReward.
    layouts/             Zone layouts built from configs/zones/: four_zones.py (E1/E2, with the counterbalancing
                         permutation), four_zones_pixel.py (E3: pixels, themes, seed -> rotation -> block class),
                         two_zones.py (E4: UD and CD on a diagonal, slots as appearance categories).
    torus_env.py         TorusZonedEnv: the base environment from E1 on.
  agents/
    base.py              Agent ABC (transition_distribution only) + materialize_transition_tensor.
    tabular.py           TabularModelBasedAgent: counts, T_hat/R_hat, value iteration.
    collect.py           Fixed-policy trajectory collection (offline protocol).
    online/              OnlineTabularAgent, train_online(_with_burnin), RecoveryTracker -- env-agnostic.
    worldmodel/          E3's RSSM, losses, replay, training loop, separation switch, closed-loop agent.
  measures/              Arrays in, arrays/scalars out (see below). controllability.py holds THE I(a;s'|s).
  interventions/         Causal manipulation of a transition belief (the T_hat dial, component swaps).
  planning/value_iteration.py   GAMMA = 0.95 is the library default; every caller may pass its own
                         (E1's calibration sweeps it, then E1-E3 pass gamma=0.9 explicitly).
  evaluation/            rollout.py, tiebreak.py, offline_support.py (E0's offline plumbing).
  streams/               Independent random streams by SeedSequence.spawn: tabular.py (E2), worldmodel.py (E3).
  twin/                  The tabular twin: counts in, the full tabular measurement chain out (E2, reused by E3).
  probing/               Torch-side measurement of a world model (forward passes through a frozen RSSM -> arrays):
                         probe.py (E3), fixed_trajectory.py (I_s along fixed walks, E4), walks.py (the walks, numpy),
                         theme_probes.py (E4's probe set).
  pipeline/              The world-model run pipeline: runtime.py, world.py, measure.py, checkpoints.py (E3's run_mini),
                         agent_run.py (E3's closed-loop run; ClosedLoopHooks for another world, E4 on), two_zones.py
                         (the world of a two-zone environment).
  cluster/               GPU cluster tooling: kubeflow.py (supervisor, runtime patches), handoff.py (handoff and the
                         results-branch layout), results_push.py (automatic push to results/<batch>), wandb_logger.py.
  config.py              Dataclass configs (E0) + compose()/FROZEN_BLOCKS (E1+).
  runs.py                Results-naming convention, run metadata, code provenance.
  experiment_loader.py   load_experiment_module: loads a module of a finished experiment by path (legacy only).

configs/                 Shared YAML blocks (zones/, themes/, dynamics/) and one directory per world-model
                         experiment (e3/, e4/, ...). Data only, no Python.
experiments/             One directory per experiment: orchestration only (runners, analysis, batch tables).
  E0_offline_5x5/        01-04 (the first paper's phases), 05_online_prototype.          FINISHED, frozen.
  E1_offline_4zones/                                                                     FINISHED, frozen.
  E2_online_4zones/                                                                      FINISHED, frozen.
  E3_worldmodel_4zones/                                                                  FINISHED, frozen.
  E4_generalization_gradient/   code ready (branch e4); launch: RUN_GPU.md.
  E5_belief_training/, E6_phenotypes_latent/, E7_latent_manipulation/
docs/                    EXPERIMENTS.md (the central tracker), one file per experiment, RESULTS_CONVENTION.md,
                         e3/ (E3's decisions, spec status, phase reports), archive/ (abandoned plans).
results/                 Run outputs (see "results" below). Heavy artefacts are never versioned.
reports/                 Frozen outputs of E0 and E3 (reports/README.md). Nothing new writes here.
paper/                   LaTeX paper and reports, figure and number scripts (see "Paper" below).
archive/                 Retired material.
tests/                   Mirrors src/ (plus tests/e3/ for E3's orchestration).
```

Finished experiments are frozen

E0 (phases 01-05), E1, E2 and E3 are finished; tags `e0-final` ... `e3-final` (and `v0.1-offline` for E0's report)
mark their final state, and `pre-reorg` the state before the 2026-10-14 reorganisation.

- Their logic never changes: no change of computation, default value, seed, stream order or config. They may be
  moved and their imports updated; a module promoted to the library leaves a re-export shim at its old place
  (`experiments/E2_online_4zones/env.py`, `rng.py`, `twin.py`, `matched_control.py`; `experiments/E3_worldmodel_4zones/env.py`,
  `rng_streams.py`, `probe.py`, `wandb_logger.py`; `run_mini.py` re-exports what moved to `pipeline/`, `run_agent.py`
  what moved to `pipeline/agent_run.py`). Shims are
  never removed and never edited except to follow a move.
- A genuine bug in a finished experiment is flagged before it is fixed, never folded into unrelated work.
- `configs/e3/*.yaml`, `configs/zones/e1_4zones.yaml` and every finished experiment's config are never edited.

Extending the library (E4-E7)

- New behaviour goes into new modules (a new layout in `envs/layouts/`, a new probe in `probing/`, a new pipeline
  piece in `pipeline/`), or into an option whose default reproduces the previous behaviour exactly, with a test
  pinning that default.
- An experiment imports the library, never another experiment: no new `load_experiment_module` call, no
  `sys.path` insertion of another experiment's directory. If an experiment needs code that lives in a finished
  experiment, the code moves to the library first (with a shim), in its own commit, verified bit for bit.
- An experiment's configs go to `configs/<eN>/`; its runners, analysis and cluster batch table to
  `experiments/<EN_name>/`; its status to `docs/EXPERIMENTS.md` and `docs/<EN_name>.md` (before the runner is
  written, and on every status change).
- Every change to the library is checked against the finished experiments' reduced runs (the non-regression
  protocol of the 2026-10-14 reorganisation: same outputs bit for bit on the same machine, volatile fields
  aside).

The single source of truth

- **I(a;s'|s) has exactly one implementation**: `measures.controllability.controllability_from_tensor` (and its
  per-state form `controllability_per_state`). It takes a `(S, A, S)` array and a terminal mask, nothing else.
  Every other computation of this quantity -- `env.controllability()`, `perceived_controllability()`, a
  `DynamicsSpec.analytic_controllability()` closed form, a world model's s-space kernel (I_s, E3's main measure,
  via `pipeline.measure.controllability_of_kernel`) -- either calls this function or is tested against it.
- **I_z is a different quantity**, not a second implementation: the information between the action and the
  predicted *latent code*, computed in latent space by `measures.latent_controllability` (E3's secondary measure).
  It never stands in for I(a;s'|s); a third mutual-information computation over actions and *states* would be a
  bug.
- **measures/ and interventions/ take arrays in, return arrays/scalars out, and never import an Env, an Agent or
  torch.** The bridge goes on the other side: `agents.base.materialize_transition_tensor` for tabular beliefs,
  `probing/` for world models (forward passes -> arrays). `probing/` is the only place where torch code measures.
- **Every environment's `transition_tensor()`/`transition_row()` stays exactly consistent with `step()`**,
  verified by a model/simulation-agreement test (`tests/envs/test_gridworld_stay_action.py`,
  `tests/envs/test_torus_env.py::test_model_simulation_agreement`); a new environment needs its own instance.
- **A new `DynamicsSpec` ships its `analytic_controllability()` tested against the numeric measure**
  (`tests/envs/dynamics/test_analytic_agreement.py`).

What never changes

- `terminated` and `truncated` are never conflated. Every MDP here is infinite-horizon discounted; `max_steps` is
  a loop limit, not part of the MDP. On truncation, bootstrap on `V(s')`; only `terminated=True` is absorbing.
- RNGs are kept separate (`streams/`): environment dynamics, exploration, tie-breaking, start states, and every
  measurement its own slot, each derived by `SeedSequence.spawn`, never two generators seeded with the same
  integer. It must always be possible to hold one fixed while varying another.
- The planner is never varied across agents compared in the same experiment.
- From E4 on, dynamics are deterministic only.

Agent interface

`agents.base.Agent` is deliberately minimal: `transition_distribution(s, a) -> array`. The tabular, online and
world-model agents differ in their `observe()`/`plan()` signatures by design; do not force them into one shape.
Add to `Agent` only what every backend genuinely shares.

Adding an environment variant

A new variant is a new `DynamicsSpec` (`envs/dynamics/`), a new `AppearanceSpec` category or theme rule
(`envs/appearance/`), and/or a new zone layout (`configs/zones/` + a module in `envs/layouts/`) -- never a new
`Env` subclass and never a change to `TorusZonedEnv`. Zone geometry is validated strictly (`ZoneLayout` refuses
holes and overlaps); fix the config, never catch the error. Grid size is always explicit in a config.
`n_actions` for the zone family is 5 (`envs.dynamics.N_ACTIONS`, includes "stay"); the legacy environment's
`N_ACTIONS=4` is a separate constant, never imported under the same bare name.

The paired control is always derived (`envs.dynamics.pairing.derive_control`), never hand-written. The reference
zone is built by `envs.zones.reference_zone` and checked by `assert_reference_unchanged` whenever a config claims
to use it.

Results

- E0: `results/<name>.csv` + `<name>.csv.config.yaml` (CONTRIBUTING.md), unchanged.
- E1 on: `results/<experiment_id>/<run_name>/` with `config.yaml` (fully resolved), `meta.yaml`
  (`runs.write_run_meta`: seed, git commit, code hash, status, times), metrics, `artifacts/`. `run_name =
  grid<RxC>__<zone_preset>__pair-<experimental|control>__<agent_type>__seed<NN>__<UTC timestamp>`
  (`runs.make_run_name`); `pair_role` always explicit; `make_run_dir` never overwrites.
- GPU cluster batches (E4 on): what a batch returns goes to **its own orphan branch `results/<batch>`**, never
  merged, read through `git worktree` (`docs/RESULTS_CONVENTION.md`; built by `cluster.handoff.write_results_layout`,
  `make_handoff.py --results-layout`). Weights, buffers, `arrays.npz` and checkpoints stay on the cluster's
  storage; `handoff.json` lists them with their absolute cluster path and SHA-256.
- Never versioned: `handoff/`, `logs/`, `*.pt`, `**/replay.npz`, `**/wandb/`, checkpoints, bulk arrays (see
  `.gitignore`). E3's handoffs versioned before 2026-10-14 stay in the history only.

Environments: Windows and Linux

Development runs on Windows (PowerShell, `.venv\Scripts\python.exe`); training runs on Linux on the UCL ARC GPU
cluster (Kubeflow, the repo checked out under `/scratch`). Code never contains an absolute path: paths derive
from `helplessness.config.REPO_ROOT` or come in as arguments (cluster paths: `--repo-dir`, `--mount`, batch
tables in the experiment's own submit script). Shell scripts for the cluster are LF (`.gitattributes`). Keep
Windows' 260-character path limit in mind for run directories. Bit-exact reproducibility is claimed only on the
same machine, device, library versions and number of threads; across Windows and Linux, compare with a tolerance.

Git

- Commits list their files explicitly; messages in French; **no `Co-Authored-By` line and no mention of Claude or
  of any AI tool**.
- **Never push** (branches or tags): pushes are made by the user.
- No history rewrite (no amend of a published commit, no rebase of shared history, no force).

What must never be duplicated

- The mutual-information computation (`measures.controllability`).
- The transition dynamics of an environment (one `transition_tensor()`/`transition_row()`, sampled by `step()`).
- The reference zone's definition (`envs.zones.reference_zone`).
- The paired-control transformation (`envs.dynamics.pairing.derive_control`).
- Config blocks marked frozen (`helplessness.config.FROZEN_BLOCKS`: `zones.reference`, `themes.test_pool`);
  `compose()` raises `FrozenConfigError` rather than merging over them.
- Analysis logic that already exists: behavioural metrics in `measures/behavioural.py`, RSA in `measures/rsa.py`,
  the E3 measures in `pipeline/measure.py` -- import them, never recompute inline.

Paper

`paper/` holds the LaTeX sources of the paper and the scripts that generate its figures (`paper/figures/fig_*.py`) and numbers (`paper/numbers/gen_numbers.py` -> `paper/numbers.tex`). Build and conventions: `paper/README.md`.

- **The prose in `paper/sections/` belongs to the author.** Never write, rewrite or rephrase it unless explicitly asked to. Infrastructure work (figure/number scripts, macros, build, checks) may touch section files only for `% src:` comments, figure inclusions and macros, never for the text itself.
- **Audit mode** (when asked to audit the paper): check every claim in `paper/sections/` against the code, the configs and the results, and list each discrepancy (file:line, the claim, what the source actually says, the source path). Do not correct the text.
- Every number in the text comes from a `numbers.tex` macro, and every figure comes from a `fig_*.py` script that reads results read-only and reuses the experiments' own analysis functions. Never invent a value, a data point or a bibliography entry: a missing source is an explicit error, never a fallback.
