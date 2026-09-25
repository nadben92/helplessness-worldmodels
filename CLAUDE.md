Code conventions

Python 3.10+, numpy. No heavy dependencies in the tabular phases (matplotlib, scipy, pyyaml are already in pyproject.toml; add nothing else without a reason tied to a specific experiment).
Gymnasium-compatible API (reset -> (obs, info), step -> (obs, reward, terminated, truncated, info)) without a mandatory gymnasium import.
Type hints on public signatures.
English docstrings and comments throughout the codebase; keep that style for consistency.
No external RL framework. Tabular agents must stay readable line by line.
measures/ and interventions/ must have no side effects, and must never import an environment or agent class -- see "the single source of truth" below.

Structure

```
src/helplessness/
  envs/
    base.py            Env ABC: reset, step, transition_tensor, reward_vector, terminal_mask, controllability.
    legacy/gridworld.py  E0's 5x5 environment. Frozen. See "what never changes" below.
    zones.py             Zone, ZoneLayout, the reference zone.
    dynamics/            DynamicsSpec registry (4 types) + pairing (derive_control) + factory (YAML -> object).
    appearance/          AppearanceSpec + the theme system (themes.py).
    observation/         ObservationBackend: index (tabular) / pixel (world model).
    rewards/             RewardFn: reserved, not designed yet.
    torus_env.py          TorusZonedEnv: the base environment for E1 onwards.
  agents/
    base.py              Agent ABC (transition_distribution only -- see below) + materialize_transition_tensor.
    tabular.py            TabularModelBasedAgent: counts, T_hat/R_hat, value iteration.
    collect.py             collect()/uniform_policy(): fixed-policy trajectory collection (offline protocol).
    online/                agent.py (OnlineTabularAgent), loop.py (train_online, train_online_with_burnin),
                           recovery.py (RecoveryTracker, recovery_time) -- environment-agnostic, duck-typed.
    worldmodel/            Reserved, empty until E3.
  measures/
    controllability.py     controllability_from_tensor: THE ONLY implementation of I(a;s'|s). value_gap, value_gap_second.
    belief.py               perceived_controllability (wraps controllability_from_tensor), coverage, model_error.
    behavioural.py           Policy entropy, perseveration, cycle stats, outcome rates -- computed on trajectories.
  interventions/            Causal manipulation of a transition belief (the T_hat dial, component swaps).
                           Reused as-is by E6 (latent perturbations) -- do not fold into measures/ or analysis code.
  planning/value_iteration.py   value_iteration(P, R, terminal, gamma). GAMMA = 0.95, the library default --
                           every caller may override it explicitly; E2 does, passing gamma=0.9 throughout
                           (experiments/E2_online_4zones/sweep.py), so "0.95" is not a claim about what any
                           given experiment actually used.
  evaluation/               rollout.py (evaluate, draw_start_states), tiebreak.py (argmax_random_tie).
  config.py                 Dataclass configs (E0, online prototype) + compose()/FROZEN_BLOCKS (E1+, see below).
  runs.py                   Results-naming convention (E1+, see below).

configs/                  Shared YAML blocks for compose() -- zones/, themes/, dynamics/. Data only, no Python.
experiments/
  E0_offline_5x5/          01_offline_sweep, 02_swaps, 03_intervention, 04_rescue_injection (the paper's phases),
                           05_online_prototype (not a numbered phase -- see its README.md).
  E1.. (future)             One directory per experiment in docs/. Orchestration only -- see "adding an experiment".
docs/                     One file per experiment (E0-E6): what it tests, its config, its status. Update on every status change.
results/                  See "results and run naming" below.
tests/                    Mirrors src/ (tests/envs/dynamics/ mirrors src/helplessness/envs/dynamics/, etc.).
```

The single source of truth

- **I(a;s'|s) has exactly one implementation**: `measures.controllability.controllability_from_tensor`. It takes a `(S, A, S)` array and a terminal mask, nothing else. Every other computation of this quantity -- `env.controllability()`, `perceived_controllability()`, a `DynamicsSpec.analytic_controllability()` closed form -- either calls this function directly or is tested against it (see `tests/envs/dynamics/test_analytic_agreement.py`). If you ever find a second place computing mutual information over actions and states, that is a bug: unify it, don't leave two implementations that could silently diverge.
- **measures/ and interventions/ take arrays in, return arrays/scalars out, and never import an Env or Agent class.** This is what lets them run unchanged on a tabular T_hat today and a neural world model's belief later. If a measure needs something from an agent, the bridge goes on the agent side: `agents.base.materialize_transition_tensor(agent, S, A)` builds the tensor by calling `agent.transition_distribution(s, a)` for every pair, then hands it to measures/ as a plain array. Never make measures/ agent-aware to avoid writing this bridge.
- **Every environment's `transition_tensor()`/`transition_row()` must stay exactly consistent with `step()`.** Any change to the dynamics must be propagated to both and verified by an empirical model/simulation-agreement test (sample many times from a fixed state, compare the empirical distribution to the exact tensor). `tests/envs/test_gridworld_stay_action.py` and `tests/envs/test_torus_env.py::test_model_simulation_agreement` are the two existing instances of this test; a third environment needs a third instance, not an exception.
- **A new `DynamicsSpec` must ship its `analytic_controllability()` tested against the numeric measure** (`tests/envs/dynamics/test_analytic_agreement.py` is the pattern), not just against intuition.

What never changes

- `src/helplessness/envs/legacy/gridworld.py` and everything under `experiments/E0_offline_5x5/`: frozen. E0 is a finished paper result; a change here is only ever a genuine bug fix, and it gets flagged before it happens, not silently folded into unrelated work.
- `terminated` and `truncated` are never conflated in any environment. Every MDP here is infinite-horizon discounted; `max_steps` is a convenience limit for loops, not part of the MDP. On truncation, bootstrap on `V(s')`; only `terminated=True` is genuinely absorbing. Conflating the two teaches an agent that the world ends at `max_steps` -- an artefact indistinguishable from a real phenotype.
- RNGs are kept separate: one for environment dynamics (`seed_dynamics`), one for agent exploration, one for tie-breaking at the policy-extraction step, one for burn-in if used. It must always be possible to hold one fixed while varying another.
- The planner is never varied across agents being compared in the same experiment. Behavioural differences must be attributable to the belief, not to the planning budget.

Agent interface

`agents.base.Agent` is deliberately minimal: `transition_distribution(s, a) -> array`. `TabularModelBasedAgent` and `OnlineTabularAgent` already differ in their `observe()`/`plan()` signatures by design (the online agent learns `terminal[s]` from experience and plans against a tie-break RNG; the offline agent takes an externally-supplied terminal mask and a gamma) -- do not force them into one shape to make the ABC "more complete". Add to `Agent` only once a second concrete agent (the world-model one, E3) shows which parts are genuinely load-bearing across every backend; until then, a shared method here is a convention documented in the class docstring, not an abstract requirement.

Adding an environment variant

A new variant is a new `DynamicsSpec` (in `envs/dynamics/`) and/or a new `AppearanceSpec` category (in `envs/appearance/`) and/or a new zone layout (in `configs/zones/` + an experiment's own config) -- never a new `Env` subclass and never a change to `TorusZonedEnv` itself. `TorusZonedEnv` already reads dynamics per-cell from whatever `ZoneLayout` it is given; if an experiment needs a fifth dynamics type, add it next to the other four and give it `transition_row()` + `analytic_controllability()`, tested per the pattern above. Only touch `legacy/gridworld.py` for a genuine bug fix (see "what never changes").

Zone geometry is validated strictly, not defensively: `ZoneLayout` refuses to construct if its zones don't tile the grid exactly once (a hole or an overlap is a `ValueError` naming the offending cells). Never catch that error to "fix" a zone layout at runtime -- fix the config.

Grid size is always explicit in a config, never a code default. `n_actions` for the zone family is fixed at 5 (`envs.dynamics.N_ACTIONS`, includes "stay") -- the legacy environment's own `N_ACTIONS=4` default is a separate constant in a separate module and the two must never be imported under the same bare name (see `envs/__init__.py`'s docstring note).

The paired control for an experimental environment is always derived, never hand-written: `envs.dynamics.pairing.derive_control(spec)` on each zone's dynamics (via the experiment's config, before constructing the `ZoneLayout`), not a second YAML file maintained by hand. If you need a different pairing convention than "uncontrollable -> ControllableDeterministic", change `derive_control`, don't special-case one experiment.

The reference zone (the one that must be byte-identical across every environment that uses it) is built by `envs.zones.reference_zone(grid_shape, anchor)` and checked by `envs.zones.assert_reference_unchanged(zone, anchor)`. Call the assertion when loading any config that claims to use the reference zone -- don't just trust the config.

Adding an experiment

1. Add `docs/<ExperimentID>_<name>.md` (or update the existing stub) stating what it tests, its config, and its status -- before writing the runner, not after.
2. Create `experiments/<parent>/<NN_name>/` (or `experiments/<ExperimentID>_<name>/` for a new top-level experiment) with a `run.py`, a `config.yaml`, and a `README.md`.
3. The runner composes its config via `helplessness.config.compose(blocks, overrides)` if it uses the zone/dynamics/appearance family (E1+), or a dataclass from `helplessness.config` if it's part of the E0 family. Either way, the fully-resolved config gets dumped next to the results -- never the pre-composition pieces.
4. For E1+, use `helplessness.runs.make_run_dir`/`make_run_name` for output layout. For E0-family additions, follow the existing `results/<name>.csv` + `<name>.csv.config.yaml` convention (see CONTRIBUTING.md).
5. Update the experiment's `docs/` status when it actually runs.

Results and run naming (E1 onwards)

```
results/<experiment_id>/<run_name>/
    config.yaml       full resolved config (helplessness.config.compose() output)
    meta.yaml          seed, git commit, status, started_at/updated_at (helplessness.runs.write_run_meta)
    metrics.csv         or whatever the experiment's own metric file is called
    checkpoints/         world-model runs: periodic saves + latest.json pointer (never a symlink)
    artifacts/            figures, rendered samples
```

`run_name = grid<RxC>__<zone_preset>__pair-<experimental|control>__<agent_type>__seed<NN>__<UTC timestamp>`
(`helplessness.runs.make_run_name`). `pair_role` is always explicit in the name -- never infer "this is the control" from which config file was passed. `make_run_dir` refuses to overwrite an existing run directory; a rerun gets a new timestamp, not a clobbered old one.

This convention is for E1 onwards. E0's `results/*.csv` layout (see CONTRIBUTING.md, "results and configs") is untouched and stays that way -- it is not being migrated to the new convention.

What must never be duplicated

- **The mutual-information computation.** One function, `measures.controllability.controllability_from_tensor`. See "the single source of truth" above.
- **The transition dynamics of an environment.** One `transition_tensor()`/`transition_row()` per environment, and `step()` must sample from exactly that. Never a second, hand-written approximation for "just this one experiment's plotting script."
- **The reference zone's definition.** One frozen definition (`envs.zones.reference_zone`), included by reference everywhere, checked by `assert_reference_unchanged`, never retyped into a second config.
- **The paired-control transformation.** One function (`envs.dynamics.pairing.derive_control`), applied to a config, never a second hand-authored environment.
- **Config blocks marked frozen** (`helplessness.config.FROZEN_BLOCKS`, currently `zones.reference` and `themes.test_pool`). An experiment's own config may only extend or override the non-frozen parts; `compose()` raises `FrozenConfigError` rather than silently merging over a frozen path.
- **Plotting/analysis logic that already exists.** Behavioural metrics live in `measures/behavioural.py`; if a new experiment's plotting script needs a metric already computed there, import it, don't recompute it inline.
