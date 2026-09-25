# E3 — Phase 0 report (repo reading, no code)

Repo state at time of reading: branch `e2/online-tabular`, HEAD `724bf64`
("E2: seed-level bootstrap CI for the budget power-law exponent (b_hat,
L1)"). All citations below are `file:line` or `file::function` against
that commit.

---

## 1. Repo structure and E2's entry point

Library code lives under `src/helplessness/` (installed package
`helplessness`), orchestration under `experiments/<ID>_<name>/`, one doc
per experiment under `docs/`, tests mirroring `src/` under `tests/`. This
matches `CLAUDE.md`'s structure section exactly; nothing has drifted from
it as of this commit.

E2's own code lives entirely in `experiments/E2_online_4zones/` (not
under `src/helplessness/`): `env.py` (layout + counterbalancing),
`agent.py` (`OnlineTabularAgentKnownReward`), `loop.py`
(`train_online_e2`), `rng.py` (seed→streams), `matched_control.py`
(`collect_matched`), `twin.py` (`twin_from_counts`, the tabular
measurement pipeline), `measurement.py` (row-building for CSVs),
`sweep.py` (the actual runnable entry point — `python sweep.py` runs
`main()`, the full 4-condition × 20-seed sweep), plus `pilot.py`
(single-seed dry run), `analysis*.py`/`fig_f*.py`/`tables.py` (post-hoc
analysis and figures), and `phase_a.py` (status checks against saved
sweep artifacts). **`sweep.py::run_condition_seed` is the one function
that does everything end-to-end for one (condition, seed) job** — this is
the closest thing to a single entry point, and the one E3 should read
line-by-line as "what does one full run of E2 do."

Nothing under `experiments/E2_online_4zones/` is imported by library code
under `src/helplessness/` — it is a leaf, orchestration-only, exactly as
`CLAUDE.md`'s "Adding an experiment" section prescribes.

---

## 2. Torus 12×12 environment

### 2.1 Zone coordinates, corridor, goal

From `configs/zones/e1_4zones.yaml` (loaded unchanged by E2 via
`helplessness.config.load_yaml_block("zones/e1_4zones.yaml")`,
`sweep.py:50`):

- `grid_shape: [12, 12]`, `goal: [0, 0]`.
- Corridor: `cross_corridor_mask` (`src/helplessness/envs/zones.py:152-162`)
  over `rows=[0,6]`, `cols=[0,6]` — i.e. row 0, row 6, col 0, col 6 form a
  cross, `ControllableDeterministic` dynamics, appearance `corridor`.
- Four 5×5 blocks via `torus_block_mask` (`zones.py:137-149`), closed
  ranges inclusive both ends:
  - `nw`: rows [1,5], cols [1,5] — `UncontrollableDeterministic`
    (conveyor), appearance `uncontrollable_deterministic`.
  - `ne`: rows [1,5], cols [7,11] — `ControllableNoisy(p=0.125)`,
    appearance `controllable_noisy`.
  - `sw`: rows [7,11], cols [1,5] — `ControllableDeterministic`,
    appearance `controllable_deterministic`.
  - `se`: rows [7,11], cols [7,11] — `UncontrollableNoisy(p=0.125)`,
    appearance `uncontrollable_noisy`.
- `12 = 1 + 5 + 1 + 5`: corridor rows/cols at 0 and 6, blocks fill
  `{1..5}` and `{7..11}` (comment in the YAML itself).

E2 additionally permutes **which dynamics type sits in which geometric
quadrant** per seed (never the corridor, never the block boundaries
themselves) — see §2.2 below and `experiments/E2_online_4zones/env.py`.

### 2.2 Action order, `stay`

`ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))`,
`ACTION_NAMES = ("up", "right", "down", "left", "stay")`, `N_ACTIONS = 5`
(`src/helplessness/envs/dynamics/base.py:28-30`). This is a **separate**
constant from the legacy `GridWorld`'s own 4-action `ACTIONS`/`N_ACTIONS`
(`envs/legacy/gridworld.py`, re-exported under the same bare names by
`envs/__init__.py`) — the two must never be imported under the same name
(`envs/__init__.py:1-7`'s own docstring note, and `CLAUDE.md`).

### 2.3 Kernels of the four dynamics types

All four live in `src/helplessness/envs/dynamics/{controllable,uncontrollable}.py`,
operate on the torus (`_wrap`, modulo both axes), stateless, one
`transition_row(rc, a, grid_shape)` + one `analytic_controllability()`
each:

- `ControllableDeterministic`: intended action always executes.
  `I = log2(5)` exactly (`controllable.py:12-26`).
- `ControllableNoisy(p)`: with prob. `p` the executed action is replaced
  by a **uniform draw over all 5 actions** (independent of the chosen
  one), else the chosen action executes (`controllable.py:29-72`). Closed
  form `analytic_controllability` is the symmetric-channel formula.
- `UncontrollableDeterministic(drift)`: fixed displacement every step,
  action read and ignored. `I = 0` exactly (`uncontrollable.py:21-39`).
- `UncontrollableNoisy(drift, p)`: with prob. `(1-p)` the fixed `drift`
  applies, with prob. `p` a uniform random direction applies — **the
  action never appears in the row at all**, so `I = 0` exactly for every
  `p`, not just approximately (`uncontrollable.py:42-76`).

`measures.controllability.controllability_from_tensor` is the one place
`I(a;s'|s)` is ever computed numerically; every `analytic_controllability()`
above is tested against it in `tests/envs/dynamics/test_analytic_agreement.py`
(pattern cited by `CLAUDE.md`).

---

## 3. Direction in the dynamics: goal-relative, not absolute

**Goal-relative**, by construction, not absolute. `envs/zones.py::resolve_drift`
(`zones.py:170-211`) takes `(row_range, grid_shape, goal_rc, mode)` where
`mode ∈ {"away_from_goal","toward_goal"}`, and resolves the row-axis drift
sign by comparing **toric row-distance to the goal's row** from the
block's near edge — explicitly documented to handle the case where a
block on the far side of the torus has "down" as `toward_goal` even
though row indices increase away from row 0 (`zones.py:170-186`
docstring, SE block example).

Both `experiments/E1_offline_4zones/env.py::_build_block_dynamics`
(`env.py:29-42`) and `experiments/E2_online_4zones/env.py::_build_block_dynamics`
(`env.py:77-97`) call `resolve_drift(row_range, grid_shape, goal_rc, direction)`
for both `UncontrollableDeterministic` and `UncontrollableNoisy` — the raw
`configs/dynamics/*.yaml` files' literal `drift: [1, 0]` values
(`configs/dynamics/uncontrollable_{deterministic,noisy}_default.yaml`) are
**never used directly**; they exist only as a factory-registry default
(`envs/dynamics/factory.py`) unrelated to E1/E2's own zone construction
path.

Critically for E3's permutation requirement (spec §2.2): E2's
`_build_block_dynamics` resolves drift from the **target** quadrant's own
`row_range`, never the source config's `rows` — `env.py:77-85`'s
docstring calls this out explicitly as a bug the counterbalancing review
caught, and `tests/envs/test_e2_4zone_layout.py::test_drift_points_away_from_goal_for_every_uncontrollable_zone_at_every_rotation`
and `::test_drift_is_not_silently_copied_from_the_source_quadrant` pin it
directly. **No adapter is needed in `e3_worldmodel/` for direction
resolution** — `resolve_drift` already satisfies spec §2.2's requirement
("toute direction ... doit être exprimée dans le repère relatif au but"),
and E3 can call it exactly as E1/E2 do.

---

## 4. Paired reference environment

Construction: `envs.dynamics.pairing.derive_control(spec)`
(`pairing.py:28-33`) — `UncontrollableDeterministic → ControllableDeterministic()`,
`UncontrollableNoisy(drift,p) → ControllableNoisy(p)` (noise level `p`
preserved), controllable zones untouched. Applied per-zone by
`build_control_layout()` in both E1's and E2's `env.py`
(`E1/env.py:70-79`, `E2/env.py:139-148`) — `derive_control` is never
hand-authored per experiment.

Matching test: `tests/envs/test_e2_4zone_layout.py::test_pairing_agreement_survives_quadrant_permutation`
(`test_e2_4zone_layout.py:207-242`) checks, per non-corridor zone and at
every counterbalancing rotation, that `conditional_entropy_per_state(P_exp)
== conditional_entropy_per_state(P_ctrl)` pointwise and that the support
size of `P[s,a]` (`(P[s,a] > 0).sum()`) matches exactly between exp and
ctrl, while asserting `I_exp != I_ctrl` on the originally-uncontrollable
zones — i.e. **exactly** spec §2.3's "H(s'|s,a) and support size
identical, only I differs" appariement test, already implemented for the
tabular family and directly reusable as the pattern for E3's pixel
environment (same `P` tensors, only the observation backend changes).

E2's actual reference/control values, at runtime, are never the closed-form
`I=0`/`I=log2(5)`: **`b` is computed from a matched-counts reference**
(§7 below), not the analytic kernel directly — this matters for E3's
spec §5.6/§6, which also specifies a matched-counts reference (the "Ref"
condition run end-to-end in the pixel environment, not a closed form).

---

## 5. E2's policy

- **ε-greedy**: `explore_rng.random() < epsilon` → uniform random action
  via `explore_rng`; else `agent.policy[s]` (`loop.py:57-61`, E2's own
  `train_online_e2`, not the shared `agents.online.loop.train_online`).
  `epsilon` is swept: `0.1` (main conditions) and `1.0` (the "bridge to
  E1" condition) — `sweep.py:65-70`.
- **Tie-break**: `evaluation.tiebreak.argmax_random_tie` for the
  **acting/collection** policy, drawn from a dedicated `tiebreak_rng`
  (`agent.py:96-123`, `plan()`). A **different** tie-break convention
  (`uniform_tie_policy_probs`, a full mixture, not a single sampled
  action) is used only for evaluating Δ (`twin.py:87`) — the two must not
  be confused; `plan()`'s own `self.policy` is never used for Δ
  (`agent.py:104-107`'s docstring is explicit about this).
- **Burn-in**: **none**, in the real E2 sweep. `experiments/E2_online_4zones/loop.py`'s
  `train_online_e2` has no burn-in parameter at all; its module docstring
  states plainly: *"no burn-in, terminated/truncated never conflated, the
  reset transition never counted in N — already held and is preserved
  here"* (`loop.py:15-17`). The burn-in fix
  (`helplessness.agents.online.loop.train_online_with_burnin`) belongs
  only to the E0 5×5 prototype (`experiments/E0_offline_5x5/05_online_prototype/`)
  and is **not** part of E2's shipped pipeline — `docs/E2_online_4zones.md`'s
  own wording ("carries over from the E0 prototype... the Phase-1 burn-in
  fix") is aspirational/historical, not a description of `sweep.py`'s
  actual behaviour. **Flagging this as a real discrepancy between
  `docs/E2_online_4zones.md` and the shipped `sweep.py`/`loop.py`** —
  worth a one-line correction to that doc, not something E3 should
  silently resolve by picking one or the other. E3's own spec (§4.5)
  asks to "reproduce exactly E2's treatment (burn-in at ε=1 or not), as
  identified in phase 0": **E3 should default to no burn-in**, matching
  what E2 actually ran, and only add `prefill_steps` as its own,
  separately-logged, world-model-specific addition (§4.5's second bullet).
- **Starts**: uniform draw over **non-terminal** states
  (`NON_TERMINAL_STATES = [s for s in range(N_STATES) if s != GOAL_S]`,
  `sweep.py:56`), from a dedicated `start_rng`, at the first reset and at
  every subsequent episode boundary (`loop.py:51-54,69-71`).
- **Truncation**: `H = 36` (`sweep.py:59`, `max_steps` on
  `TorusZonedEnv`). `terminated` is set only on arrival at the goal state
  (`terminal_states=[goal_s]` passed to `TorusZonedEnv`, `env.py:196`);
  `truncated`/`terminated` are never conflated anywhere in the loop
  (`torus_env.py:116-117` sets them independently every step).
- **Reward**: `GoalReward(goal_state, step_reward=-1.0, goal_reward=50.0)`
  (`envs/rewards/base.py:33-53`; E2's defaults:
  `GOAL_REWARD = 50.0`, `step_reward` defaults to `-1.0` and is never
  overridden in `sweep.py`) — **not** "50 at goal, 0 elsewhere": it is
  **50 at goal, −1 per step elsewhere** (replacing, not adding to, the
  step cost on arrival). `gamma = 0.9` (`sweep.py:58`, `GAMMA` local to
  the sweep) — **not** the shared `planning.value_iteration.GAMMA = 0.95`
  default. E2 passes `gamma=0.9` explicitly everywhere it plans
  (`agent.py:46`, `sweep.py:128`, `twin.py`'s callers) — it does not rely
  on the module-level default. **This matches spec §4.2/§4.3's own
  `γ = 0.9`** (the one-step Q and critic-target discount), so E3's choice
  of `γ = 0.9` for the agent is consistent with what E2 actually ran, even
  though it diverges from `CLAUDE.md`'s "GAMMA = 0.95, the one definition"
  language — worth flagging explicitly since `CLAUDE.md` calls 0.95 *the*
  discount and E2 does not use it.

---

## 6. Checkpoint grid

`CHECKPOINTS = tuple(sorted(set(int(round(x)) for x in np.logspace(3, 6, 16))))`
(`sweep.py:63`, identical in `pilot.py:62` and `analysis_data.py:210`) —
**16** log-spaced integer checkpoints from `1e3` to `1e6`:
`[1000, 1585, 2512, 3981, 6310, 10000, 15849, 25119, 39811, 63096,
100000, 158489, 251189, 398107, 630957, 1000000]` (verified by direct
computation, no rounding collisions).

**Deviation flag**: the spec's phase-0 question says "grille des 11
checkpoints." The actual full grid is 16 points. "11" appears in
`tables.py:164-182` / `RESULTS.md:158-187` — it is the **subset of the 16
checkpoints restricted to `cp >= 10_000`**
(`high_checkpoints = tuple(cp for cp in ad.CHECKPOINTS if cp >= 10_000)`,
`tables.py:178`), used only for one specific power-law fit table
(`table3_budget_trend_power_law`), not E2's actual checkpoint schedule.
**E3 should use the full 16-point grid** as "the E2 checkpoint grid" per
spec §4.6/§6 ("checkpoints sur la grille d'E2"), unless told otherwise.

---

## 7. Measures

All of E2's per-checkpoint measurement routes through
`experiments/E2_online_4zones/twin.py::twin_from_counts(N, alpha, support,
P_true, R_true, terminal, P_ref, gamma, reference_rng)` — **the** function
E3's "jumeau tabulaire" (spec §6, condition B) should reuse verbatim
(it already takes `N[s,a,s']` in and returns arrays out, with no
dependency on any online-agent internal state — `twin.py:1-16`'s own
docstring is explicit that this is designed to "run identically on a
logged N[s,a,s'] table produced by a future world model (E3)").

- **`b̂`**: `b = I_hat - I_ref`, `I_hat = controllability_per_state(T_hat)`,
  `T_hat` from `TabularModelBasedAgent(...).T_hat` built from the raw
  `N` (`twin.py:69-77`). No second implementation of `I(a;s'|s)`
  anywhere in E2.
- **`Î_ref`**: **matched to counts, not analytic.** `N_ref =
  collect_matched(P_ref, N_target, reference_rng)` — exactly `N_target[s,a]
  = N.sum(axis=2)[s,a]` draws from the **reference kernel** `P_ref`
  (`matched_control.py::collect_matched`, vectorized per-`(s,a)`
  multinomial draw), then `T_ref` and `I_ref = controllability_per_state(T_ref)`
  computed the same way as the online side (`twin.py:73-75`). This directly
  matches spec §5.6/§7's "b(s) = Î_A(s) − Î_ref(s)... même noyau réel
  apparié aux comptages," and confirms E3's `Ref` condition (spec §6)
  must be a **real, closed-loop run in the reference environment**, not a
  reweighting of the same trajectory.
- **`Δ` evaluated under `P`**: `policy_probs = uniform_tie_policy_probs(Q_hat,
  tol)` (mixture over `Q_hat`'s ties, `Q_hat` from `T_hat`+`R_true`), then
  `V_pi_true = policy_evaluation_stochastic(policy_probs, P_true, R_true,
  terminal, gamma)` — the policy is **derived from the belief** (`T_hat`)
  but **evaluated under the true kernel** `P_true`
  (`twin.py:79-91`). `V_star` also from `value_iteration(P_true, ...)`.
  `delta = V_star - V_pi_true`, clamped to exactly 0 under `1e-9` absolute
  (float-noise floor, `twin.py:92`). **Never evaluated under `T_hat`.**
  This is E1's own convention (`twin.py:85-88`'s docstring: "E2's Delta
  convention... not E1's plain argmax" — the *tie-break* differs from E1,
  the *evaluate-under-true-kernel* principle does not).
- **Dirichlet prior, α restricted to the 5 reachable successors**:
  `support = geographic_support_mask(GRID_SHAPE)` (`envs/zones.py:307-321`,
  purely geometric — the 5 torus neighbours up/down/left/right/stay, no
  dynamics involved), passed as `alpha_support` to
  `TabularModelBasedAgent` (`sweep.py:119,126-129`). Convention (pinned
  by `docs/E2_online_4zones.md`'s own "alpha is the TOTAL prior mass"
  section, and enforced by an assertion in
  `agents/tabular.py::TabularModelBasedAgent.T_hat`, lines 91-107): `alpha`
  is the **total** pseudo-count mass per `(s,a)` pair, `alpha/5` per
  reachable successor, **not** `alpha` per successor. E2 sweeps
  `alpha ∈ {0.5, 1, 5}`; `alpha=1` is the "reference condition" used in
  several of E2's own status checks (e.g. `phase_a.py::counterbalancing_dispersion_status`,
  `epsilon=0.1`). **Spec §7's "α=1" for E3-0/A/B should be read as: use
  E2's own reference condition `(alpha=1, epsilon=0.1)`** as E3's single
  fixed value, consistent with E2's own designated reference point rather
  than an arbitrary choice.
- **Aggregation by zone**: done entirely in orchestration/analysis code
  (`measurement.py::per_state_rows`/`per_pair_rows` tag every row with
  `zone = zone.appearance.category`, `zone_type = type(zone.dynamics).__name__`,
  `region = "interior"|"boundary"` via `envs.zones.interior_mask`), **not**
  inside `twin.py` or any `measures/` function — the per-state/per-pair
  arrays are indexed by a zone mask downstream, exactly as
  `CLAUDE.md`'s "measures/ ... never agent/env-aware" principle requires.
  There is no single reusable "aggregate by zone" function in
  `measures/behavioural.py` or elsewhere; each analysis script
  (`analysis_data.py`, `fig_f*.py`, `tables.py`) does its own
  groupby-style aggregation over the CSV rows. **E3 should build its own
  zone-aggregation helper** (reusing `interior_mask`/`zone.appearance.category`
  the same way) rather than expect a ready-made function to import.

---

## 8. E2's seed scheme

`experiments/E2_online_4zones/rng.py::spawn_streams(seed, alpha, epsilon)`
derives **6 independent streams** from one integer seed via
`np.random.SeedSequence(_condition_entropy(seed, alpha, epsilon)).spawn(6)`
— never `np.random.default_rng(seed)` called twice with the same
integer (the module docstring, `rng.py:1-16`, cites this as a **real
found bug in E1**: two independently-constructed `Generator`s seeded with
the same integer stay in lockstep across an entire draw sequence whenever
the underlying distributions are identical, silently erasing sampling
variance — `experiments/E1_offline_4zones/estimation.py::spawn_exp_ctrl_rngs`'s
own docstring is the primary source for this). Stream names:
`env_dynamics, explore, tiebreak, start_state, matched_control, reference`
(`rng.py:22-29`). A **second** function, `spawn_checkpoint_streams(seed,
checkpoint_step, alpha, epsilon)`, independently re-derives fresh
`matched_control`/`reference` streams **per checkpoint**, keyed on
`(seed, checkpoint_step, alpha, epsilon)` directly (not by advancing a
shared stream) — so a checkpoint's matched/reference draw is independent
of which other checkpoints were computed, in what order, or whether the
run was parallelised (`rng.py:70-104`).

**Independence between experimental and reference**: confirmed by
inspection of `sweep.py::run_condition_seed` — `P_ref` (the reference
kernel) is built once from `ctrl_layout = e2_env.build_control_layout(CFG,
DIRECTION, quadrant_assignment=assignment)` (**same** `assignment`/rotation
as the experimental layout — this is the one thing genuinely shared, per
spec §6), but the actual reference **draws** come from
`cp_streams_online["reference"]`/`cp_streams_matched["reference"]`
(`sweep.py:223-238`), streams that never touch `env_dynamics`, `explore`,
`tiebreak`, or `start_state` — i.e. the online run's own environment
noise, exploration, tie-breaks and start states are **never** reused to
generate the reference/matched draws, and vice versa. This matches spec
§6's requirement exactly: *"seuls la permutation et les thèmes sont
partagés... tous les autres flux sont indépendants."* (E2 has no themes —
tabular backend — so "permutation" alone is the shared element there;
E3 additionally shares themes per spec §2.3/§2.5, which is new relative
to E2 and needs no adapter, just an explicit design choice honoured in
E3's own seed-stream layout, see open questions below.)

There is no separate stream for "network initialisation" in E2 (no
network exists) — E3 will need to add one, per spec §6's own stream list.

---

## Things that exist but do not, as of this commit, satisfy E3's spec as-is

1. **`envs/observation/pixel_backend.py::PixelObservationBackend`** is a
   flat 1-pixel-per-cell renderer (`observe(position, zone_layout,
   theme_seed) -> (n_rows, n_cols, 3) float64`), with no `k`-pixel
   upscaling, no grid lines, no CIELAB-distance-constrained agent marker
   colour, no `render_background`/`render_counterfactual` split, and a
   signature (`position`, not state index `s`; `theme_seed`, not a
   `themes` object) that does not match spec §2.4's
   `render(s, layout, themes) -> uint8[H,W,3]`. **This is not something to
   patch in place** — spec §2.4 needs a materially different renderer.
   Open question below on where it should live.
2. **`envs/appearance/themes.py`** is a deterministic
   `(category, theme_seed) -> RGB` hash+rotation scheme that guarantees
   equal pairwise angular spacing between categories in every theme (by
   construction), but has **no explicit CIELAB distance computation, no
   `min_dist` enforcement, and no `pair_distance` parameter** — spec
   §2.5's `sample_themes(n_zones, rng, min_dist, metric="cielab",
   pair_distance=...)` does not exist yet, and there is no CIELAB
   conversion utility anywhere in the repo (`sRGB→XYZ→L*a*b*` is not in
   numpy/scipy/matplotlib; nothing pulls in `scikit-image`). Implementing
   this needs either a small self-contained CIELAB conversion (pure numpy,
   no new dependency) or a new dependency (`scikit-image`) — flagged as an
   open question below, leaning toward the self-contained version per
   `CLAUDE.md`'s "no heavy dependencies... add nothing without a reason."
3. **PyTorch is not installed** in `.venv` and not listed in
   `pyproject.toml` (`dependencies = [numpy, matplotlib, scipy, pyyaml]`).
   Adding it is squarely "a reason tied to a specific experiment"
   (`CLAUDE.md`'s own carve-out) — flagged so the PowerShell install
   command can be confirmed with you before Phase 2 needs it (not needed
   for Phase 1).
4. **`agents/base.py::Agent`/`materialize_transition_tensor`**: usable
   as-is for E3's tabular twin (B), exactly as E2's own `twin.py` uses it
   indirectly through `TabularModelBasedAgent`. For the world model
   itself, `transition_distribution(s, a)` is awkward to define directly
   over a *continuous/categorical latent* belief rather than a table row
   — `docs/E3_worldmodel_4zones.md` already flags this as "an open design
   question for whoever builds this agent" (lines 27-31). Per spec §5.2,
   E3's principal measure is computed **in z-space directly** (canonical
   probe + Monte-Carlo entropy), not via `materialize_transition_tensor`
   + `controllability_from_tensor` on some materialised `(S,A,S)` tensor
   — so this bridge is likely **not** the mechanism E3 actually uses for
   its primary measure, only (possibly) for a sanity cross-check. Flagged
   as a design point to confirm in Phase 3, not resolved here.

---

## Open questions (spec says: ask, do not improvise)

1. **Where does `e3_worldmodel/` actually live?** The spec asks for "a
   new module `e3_worldmodel/`, named consistently with the repo's
   current structure — to verify in phase 0." The repo's actual
   convention splits reusable library code (`src/helplessness/<pkg>/`,
   importable, tested under `tests/`, no heavy deps outside the tabular
   phases) from orchestration (`experiments/<ID>_<name>/`, self-contained
   scripts, not imported by anything else). `CLAUDE.md`'s own structure
   table already reserves `src/helplessness/agents/worldmodel/` for
   exactly this ("Reserved, empty until E3"). E3's own spec, though, asks
   for one self-contained module holding environment/rendering/theme
   code, the RSSM+agent, measures, diagnostics, calibration reports, and
   phase reports together — which cuts across `envs/`, `agents/`,
   `measures/` if placed inside `src/helplessness/`.

   I am proceeding for now with **`e3_worldmodel/` as a new top-level
   directory** (sibling to `src/`, `experiments/`, `docs/`, `tests/` — the
   same level chosen for this Phase 0 report already, since I had to put
   it somewhere to deliver this document), containing its own internal
   split (e.g. `e3_worldmodel/env/`, `e3_worldmodel/model/`,
   `e3_worldmodel/measure/`, `e3_worldmodel/agent/`) rather than
   scattering pieces across `src/helplessness/{envs,agents,measures}/`.
   Rationale: (a) it keeps torch-dependent code entirely out of
   `src/helplessness/`, which today has zero torch dependency and whose
   `measures`/`interventions` packages are explicitly required to import
   neither an `Env` nor an `Agent` class, let alone a neural model; (b) it
   matches the spec's own framing of E3 as a self-contained deliverable
   "reused as-is in E4/E5/E6"; (c) it avoids editing
   `CLAUDE.md`'s frozen structure table mid-experiment (the spec's own
   final instruction is to update `CLAUDE.md` **at the end of P6**, not
   now). **This is a real structural choice, not a neutral default —
   please confirm or redirect before Phase 1 writes anything under it.**
   The concrete alternative is to honour the `agents/worldmodel/` reserved
   slot literally (RSSM + agent under `src/helplessness/agents/worldmodel/`,
   pixel rendering under a new `src/helplessness/envs/observation/`
   backend or `envs/appearance/` extension, measures under
   `src/helplessness/measures/worldmodel.py`, and only orchestration +
   reports under `experiments/E3_worldmodel_4zones/`) — more consistent
   with how E1/E2 actually split reusable vs. run-specific code, at the
   cost of adding a torch dependency to the core library package.

2. **Counterbalancing/permutation logic**: E2's quadrant-rotation
   machinery (`quadrant_assignment_for_rotation`, `rotation_for_seed_index`,
   goal-relative drift recomputation) lives entirely inside
   `experiments/E2_online_4zones/env.py`, which is frozen for this task
   ("on l'importe, on ne le modifie jamais"). E3 needs the same logic
   (spec §2.2's Latin-square-by-rotation requirement is structurally
   identical to E2's). Should E3 (a) reimplement an equivalent
   `build_layout`/`quadrant_assignment_for_rotation` inside
   `e3_worldmodel/`, built from the same shared primitives
   (`helplessness.envs.zones.{cross_corridor_mask,torus_block_mask,
   resolve_drift}`, `helplessness.envs.dynamics.pairing.derive_control`)
   the way E2 itself does — accepting some duplication of ~80 lines
   relative to E2's `env.py` — or (b) should this permutation logic be
   promoted into `helplessness.envs.zones` itself (new library code, not
   a modification of anything frozen) so E2 and E3 both import the same
   function? Leaning toward (a) for Phase 1 (matches "E1/E2 frozen, never
   modified, adapters live in e3_worldmodel and get flagged" — promoting
   shared code into the library is a **library change**, not an
   E1/E2 modification, so it would technically be allowed, but it is a
   scope decision beyond "read and reuse" that I'd rather have you
   confirm).

3. **CIELAB**: implement a small, self-contained sRGB→CIELAB conversion in
   `e3_worldmodel/` (numpy only, ~30 lines, standard formulas), or add
   `scikit-image` as a new dependency? No CIELAB utility exists anywhere
   in the repo today.

4. **`docs/E2_online_4zones.md`'s burn-in claim**: flagged in §5 above as
   inconsistent with the shipped `sweep.py`/`loop.py` (no burn-in in the
   real sweep). Worth a one-line correction to that doc — should I make
   it as part of E3's Phase 0/1 work, or leave E2's docs untouched since
   E2 itself is frozen for this task?

---

Stopping here per the phase protocol. Awaiting go-ahead (and answers to
the open questions above, or a "proceed with your stated defaults")
before starting Phase 1.
