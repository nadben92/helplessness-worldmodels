# E3 — decisions log

Running record of every point where the spec was clarified, corrected, or
deliberately deviated from, with who decided and why. Updated at the end
of every phase gate, never rewritten retroactively (a superseded
decision gets a new dated entry, not an edit to the old one).

**Priority rule (2026-09-25).** When this log and the initial E3 prompt
(`prompt_e3_worldmodel.md`) conflict, this log wins. The prompt remains the
reference for everything this log does not change. Section by section,
`docs/e3/SPEC_STATUS.md` records which parts of the prompt are unchanged,
amended (with the entry below that amends them) or under review.

**Working mode (user, 2026-09-25).** The implementer decides alone, without
asking, and records each decision in one line in this log (under
"Engineering decisions" in the entry of the step), for:
- implementation bugs, tests, refactorings, performance, documentation, tooling;
- random streams, reproducibility, run infrastructure;
- alignment with E2 where E2's behaviour is unambiguous;
- fixes to tests broken by a legitimate change (keeping what the test checks).
The implementer stops and reports to the user only:
- anything that changes what is measured or how it is interpreted (I, b,
  canonical probe, kernel, diagnostics);
- any deviation from E2's protocol or from this log that is not a plain bug;
- any hyperparameter or criterion chosen while looking at results;
- any result that contradicts an expected shape;
- any irreversible action or action outside the local repository (the push
  stays with the user; commits are free).
Reports: one per completed step, not one per decision, with three sections --
results; scientific decisions to take (if any); engineering decisions taken
alone (short list).

## 2026-09-23 — Phase 0 validation

1. **Checkpoint grid.** The spec's "11 checkpoints" is corrected to E2's
   real grid: 16 log-spaced integer checkpoints from 1e3 to 1e6
   (`experiments/E2_online_4zones/sweep.py::CHECKPOINTS`). Any analysis
   that filters to the `cp >= 1e4` subset (11 points, as E2's own
   `table3_budget_trend_power_law` does) applies that same explicit
   filter, not a different checkpoint schedule.

2. **Burn-in / prefill.** E2 ran with no burn-in (confirmed in Phase 0:
   `experiments/E2_online_4zones/loop.py::train_online_e2` has no
   burn-in parameter). E3 introduces none either. Instead, E3 adds
   `prefill_steps` (~1024, the minimum to fill one training batch)
   **before the first world-model update**, played under the *same*
   epsilon-greedy policy as training (epsilon=0.1, random tie-break) —
   never epsilon=1. These steps are logged separately and **counted
   inside** the 1e6-step budget, not added on top of it.

3. **Documentation corrections (E1/E2 code itself stays frozen).**
   - `docs/E2_online_4zones.md`: corrected the "carries over from the E0
     prototype" section — E2's actual loop is
     `experiments/E2_online_4zones/loop.py::train_online_e2`, distinct
     from `helplessness.agents.online.loop.train_online`, and the shipped
     sweep runs with no burn-in at all. (Note, not requested for
     correction this round: that same doc section still names
     `helplessness.agents.online.agent.OnlineTabularAgent` as E2's agent
     class, when E2 actually uses
     `experiments/E2_online_4zones/agent.py::OnlineTabularAgentKnownReward`
     — flagging here in case a further correction pass is wanted.)
   - `CLAUDE.md`: the `GAMMA = 0.95, the one definition` line corrected
     to note it is the *library default*, explicitly overridden by E2
     (`gamma=0.9` throughout `experiments/E2_online_4zones/sweep.py`).

4. **Module location.** Follows the structure `CLAUDE.md` already
   reserved: RSSM/heads/agent/replay under
   `src/helplessness/agents/worldmodel/`. New rendering code under a new
   file in `src/helplessness/envs/observation/` (not a patch to
   `PixelObservationBackend`). New theme-sampling/CIELAB code under new
   files in `src/helplessness/envs/appearance/` (not a patch to
   `themes.py`). Measures, scripts and configs follow the existing
   per-package conventions (see `docs/e3/ARCHITECTURE.md`). The Phase 0
   report moved from a stray top-level `e3_worldmodel/` directory to
   `docs/e3/PHASE0_REPORT.md`. Full proposed tree in
   `docs/e3/ARCHITECTURE.md`, awaiting validation before Phase 1 code.

5. **Quadrant-permutation reuse.** `experiments/E2_online_4zones/env.py`
   is cleanly importable via the existing
   `tests/_experiment_loader.py::load_experiment_module` pattern (already
   used by `tests/envs/test_e2_4zone_layout.py`) — E3 imports E2's
   `quadrant_assignment_for_rotation`/`rotation_for_seed_index` directly,
   no reimplementation. Verified computationally: E2's seed→permutation
   table (`rotation_for_seed_index(seed, seeds_per_rotation=5)` composed
   with `quadrant_assignment_for_rotation`) is a balanced cyclic Latin
   square — every (dynamics type, quadrant) pair occurs in exactly 5 of
   the 20 seeds. E3 reuses this exact table (not the spec's originally
   proposed independent Latin square) so that E3's seed `k` is directly
   comparable to E2's seed `k`.

6. **CIELAB.** Self-contained sRGB→CIELAB conversion (`envs/appearance/cielab.py`),
   numpy only, tested against published reference sRGB/Lab value pairs.
   No new dependency (`scikit-image` rejected).

7. **PyTorch.** New optional dependency group `[project.optional-dependencies].e3`
   in `pyproject.toml` (`torch>=2.2`). **This machine has no NVIDIA
   GPU** (only an integrated AMD Radeon, PCI VEN_1002 — confirmed via
   `Get-CimInstance Win32_VideoController`, no `nvidia-smi`, no CUDA
   toolkit installed) — there is no CUDA wheel to target here. Decision:
   install the **CPU-only** PyTorch wheel now, from PyTorch's own CPU
   index (not the default PyPI wheel, which bundles unused CUDA
   runtimes). Phases 1-3 (environment, world model correctness,
   measures) run on this machine on CPU. Phase 4 onward (calibration,
   real training budgets) runs on a separate Linux GPU machine —
   probably a cluster with a scheduler — target to be confirmed before
   Phase 4 starts.

   Constraints adopted immediately as a result (apply from Phase 1
   onward, not just Phase 4):
   - Every E3 entry point is a cross-platform Python script (argparse,
     `pathlib`, no hardcoded Windows paths, no PowerShell-only workflow
     steps). PowerShell commands given in chat are for local convenience
     only, never encoded into the scripts themselves.
   - Device is a config value (`auto` / `cpu` / `cuda`), read once at
     startup. No hardcoded `.cuda()` call or hardcoded device anywhere.
     Tests run on CPU.
   - Phase 2's throughput profiling on this machine is indicative only;
     Phase 4's cost estimate is redone on the actual target machine.
   - `docs/e3/` carries an explicit note (see `CALIBRATION_REPORT.md`
     when it exists, and each run's `meta.yaml`) that bit-for-bit
     determinism is not guaranteed across CPU/GPU or across GPU models —
     reproducibility is guaranteed per-seed **on a given machine type**,
     not across machine types.

   PowerShell install command (CPU wheel, into the project's `.venv`,
   run from the repo root):

   ```powershell
   .venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
   ```

8. **Two spec amendments:**
   - **§4.3 (tie-break tolerance).** Changed from a fixed absolute
     `tol=1e-3` to a **relative** tolerance:
     `tol = max(1e-6, 0.01 * std(Q(h,z,·)))` for the current step's own
     Q-values across actions, configurable. The fraction of steps where
     the tie-break actually fires (more than one action within `tol` of
     the max) is logged.
   - **§7, criterion C-b.** Replaced the fixed `L1(p_hat_s, P) <= 0.1`
     threshold with a **relative-to-the-tabular-twin** criterion:
     `L1_WM <= L1_B + 0.05`, per zone type, restricted to `(s,a)` pairs
     with `N[s,a] >= 100`. `L1_B` is the tabular twin's (condition B) own
     L1 error at the matching checkpoint/zone/pair-count restriction —
     the world model is judged against what the tabular estimator itself
     achieves with the same data, not against an arbitrary fixed bound.

9. **Architecture adjustments (your Phase-0-validation reply):**
   - **A.** `AllocentricPixelBackend.observe()` raises `ValueError`
     (message citing spec section 2.5) whenever it is called with a
     non-`None` `theme_seed`, instead of silently ignoring it. Tested in
     `tests/envs/observation/test_allocentric_pixel_backend.py`.
   - **B.** No torch in `measures/`, confirmed. Two guardrails to enforce
     in Phase 3: `probe.py` must call the RSSM's own methods (never
     reimplement the forward pass), routed through `latent_hook` exactly
     as the agent does — test: for a fixed `(h, z, a)` and a non-trivial
     hook, the probe and the agent must produce the same prior. Δ under
     P must import E2's own policy-evaluation pipeline (as
     `twin_from_counts` already does), never reimplemented.
   - **C.** Import utility: a new file in `src/helplessness/`
     (`src/helplessness/experiment_loader.py`, an exact duplicate of
     `tests/_experiment_loader.py`'s logic — the test file itself stays
     untouched). "Frozen values" test added:
     `tests/e3/test_e3_seed_table.py` hardcodes the 20-seed→permutation
     table and checks it against the live import from E2.
   - **D.** `replay.py` (Phase 2): `sample()` will return two separate
     objects — the model batch (`x, a, r, terminal, truncated, is_first`)
     and the label batch (`s`) — with a test verifying the label batch
     cannot reach the world-model/goal-head/critic training functions.
     Not yet implemented (Phase 2 scope); recorded here so it isn't
     forgotten when `replay.py` is written.

## 2026-09-23 — Phase 1 finding: T2's "exact reflection transport" does
## not hold for rotations 1 and 3

Verified computationally (`tests/e3/test_e3_symmetry.py`, all passing —
this is a regression pin of an actual property of the frozen E1/E2 code,
not a bug introduced by E3): of E2's 4 counterbalancing rotations
(`rotation_for_seed_index` / `quadrant_assignment_for_rotation`, reused
unchanged, see decision 5 above), only **rotation=0** (identity,
trivially) and **rotation=2** (point-reflection through the goal, `r ->
-r, c -> -c`) correspond to an exact grid symmetry that transports the
full transition tensor `P` onto itself. **Rotation=1 and rotation=3** —
genuine 90-degree turns of the zone POSITIONS (confirmed: the position
map matches `(r,c) -> (-c, r)` exactly, and CONTROLLABLE zones' dynamics
transport correctly under it) — do **not** correspond to any of the 8
symmetries of the goal-fixing D4 group for the DRIFT-based
(uncontrollable) zones specifically.

Root cause: `envs.zones.resolve_drift` always returns a row-axis-only
drift vector (`dc=0`, by fixed, documented convention — "Drift is always
along the row axis only... arbitrary but must be single and
consistent"). A true 90-degree rotation would need to turn a row-axis
vector into a column-axis one; `resolve_drift`'s convention never
produces that, so drift transport only checks out when the target and
source quadrants share the transform's own axis-alignment (true for
row-reflection and point-reflection, false for the two quarter-turns).

This is E1/E2 dynamics-construction logic, frozen for this task. Not
resolved unilaterally — options, for your decision:
- (a) Read spec section 2.2's "reflection group" claim as applying only
  to the position/geometry level (already exactly true — see
  `test_controllable_zone_position_masks_transport_under_the_full_rotation_group`)
  and to the CONTROLLABLE zones' dynamics (also exactly true), treating
  the drift-based zones' weaker invariant (E2's own
  `test_v_star_multiset_per_zone_type_is_identical_across_rotations`,
  reused) as the actually-intended T2 criterion for those zones.
- (b) Treat this as a genuine, pre-existing gap in E1/E2's counterbalancing
  design (not something E3 can fix, since E1/E2 are frozen) and simply
  document it, which is what has been done for Phase 1.
- (c) Something else you specify.

No code changed as a result of this finding (E1/E2 untouched, as
required). Current default: (b), pending your review.

## 2026-09-23 — T2 reformulated (your review of the Phase 1 finding above)

Neither option (a) nor (b) above: the finding's premise was checking the
wrong group. **The correct symmetry group is the goal-centred AXIAL
REFLECTION group** -- {identity, row-reflection (fixes the goal's row),
column-reflection (fixes the goal's column), point-reflection / 180-degree
turn}, not the full rotation group. A drift along the row axis, pointing
away from the goal, is preserved by all four of these transforms, and
this group sends any quadrant to any other (acts simply transitively on
the 4 quadrants) -- the actual geometric property a counterbalancing
scheme needs.

**Verified exhaustively** (`tests/e3/test_e3_symmetry.py`, all 14 tests
passing):
1. Each of the 4 axial reflections transports `P` onto `P` exactly, for
   every zone type including drift, with the correspondingly-permuted
   action index. Row-reflection and column-reflection are not among
   E2's own `quadrant_assignment_for_rotation` outputs, but are valid
   inputs to E2's unmodified `build_layout()` (constructed explicitly,
   `REFLECTION_ASSIGNMENTS` in the test file) -- confirmed exact
   regardless.
2. The group acts simply transitively on the 4 quadrants (sends the NW
   block to all 4 quadrants across its 4 elements).
3. E2's own 4 counterbalancing rotations (the CYCLIC group C4, a
   genuinely different group from the axial-reflection one -- they share
   only their two order-<=2 elements, identity and the 180-degree turn)
   split into exactly two classes under the axial-reflection group:
   - **Class A** = {rotation 0, rotation 2}, related to each other by
     point-reflection.
   - **Class B** = {rotation 1, rotation 3}, also related to each other
     by point-reflection.
   - No axial reflection (nor any element of the full 8-element D4
     group, checked separately) relates a class-A rotation to a
     class-B one.
4. Seed-to-class membership (`rotation_for_seed_index`, `seeds_per_rotation=5`,
   reused unchanged from E2): **class A = seeds {0-4, 10-14} (10 seeds),
   class B = seeds {5-9, 15-19} (10 seeds)** -- exactly the 10/10 split
   expected. Recorded as `experiments/E3_worldmodel_4zones/env.py::ROTATION_CLASS`
   and `::block_class_for_seed(seed_index)`.

**Downstream consequences (recorded now, implemented when their owning
phase starts):**
- **Phase 4 (E3-0 population test).** T2's population-level check
  becomes: *within* each class, `b` per zone type does not depend on
  quadrant location. A difference *between* class A and class B is
  allowed and reported separately, never pooled silently into one
  cross-class statistic.
- **Phase 4/6 (metadata and tables).** `block_class_for_seed(seed)`
  ("A"/"B") is written into every run's `meta.yaml` and becomes an
  explicit column in every aggregated table from Phase 4 onward -- a
  declared block factor, not folded into "seed" or averaged away.

Code changed: only `tests/e3/test_e3_symmetry.py` (rewritten around the
correct group) and `experiments/E3_worldmodel_4zones/env.py`
(`ROTATION_CLASS`, `block_class_for_seed` added). E1/E2 untouched.

## 2026-09-23 — Phase 2 closed

RSSM (`src/helplessness/agents/worldmodel/rssm.py`), losses (`losses.py`,
masked reconstruction + exact KL dyn/rep), replay
(`replay.py`, model/label batch split per adjustment D), and the fixed
epsilon=1 data-collection + world-model training step
(`loop.py`) are implemented and tested (T0's 6 checklist items in
`tests/agents/worldmodel/test_t0_unit.py`, plus broader unit/integration
tests per module, 42 new tests, all passing).

Throughput profiling (`experiments/E3_worldmodel_4zones/profile_throughput.py`,
spec's full-scale config: base_depth=32, embed_dim=h_dim=256, n_vars=
n_classes=8, k=6 -> 72x72 images), on this CPU-only machine, indicative
only (point 7): ~2350 env steps/s, ~0.5 gradient steps/s at batch=8,
seq_len=32 (256 transitions/gradient step) -- training is clearly the
bottleneck on CPU, as expected; Phase 4's real replay_ratio/cost estimate
will be redone on the GPU training machine, not extrapolated from these
numbers.

Deferred, not decided here: whether the Phase-5 agent implements
`helplessness.agents.base.Agent.transition_distribution(s, a)` directly
or via a decoded/kernel-probe bridge (see `src/helplessness/agents/worldmodel/README.md`
and `docs/e3/PHASE0_REPORT.md`) -- spec 5.2's principal measure is
computed directly in z-space, so this is not blocking.

`imagination_step` (prior-only rollout, `latent_hook` wired through it)
is implemented now even though Phase 2 doesn't strictly need it, so
Phase 3's probe and Phase 5's agent both call this SAME method rather
than either one reimplementing the forward pass (adjustment B). Only
half of adjustment B's own test is possible yet (latent_hook is
genuinely invoked with the real (h, z) and changes the prior, tested in
`tests/agents/worldmodel/test_rssm.py`) -- the "probe and agent produce
the same prior for the same hook" cross-check needs Phase 3's probe.py
to exist and will be added then.

## 2026-09-24 — Phase 3, points 1-2 (pre-phase adjustments)

1. **Replay strictness.** `ReplayBuffer.for_run(budget, prefill_steps,
   image_shape, capacity=None)` validates `capacity >= budget +
   prefill_steps` at construction (`ReplayCapacityError` otherwise) and
   always builds in `raise_on_overwrite=True` mode; `add()` then raises
   `ReplayOverwriteError` rather than silently overwriting if the buffer
   is ever asked to forget its oldest surviving transition. The
   permissive (default) constructor is unchanged, so every existing
   test and ad hoc/dev usage keeps working. New test:
   `test_no_sampled_window_ever_crosses_the_oldest_newest_join` (500
   sampled windows across several full wrap cycles, all chronologically
   consecutive).

2. **`Agent.transition_distribution`, decided.**
   `agents/worldmodel/agent.py::WorldModelAgent` implements it via the
   decode+template bridge (spec 5.3): canonical `(h0, z0=mode)` at
   state `s`, `imagination_step` for action `a` (so `latent_hook` is
   exercised exactly as the future policy will), `K_samples` decoded
   draws matched against the 144 templates
   (`measures.kernel_probe`), kernel renormalised over non-ghost draws
   only. Ghost fraction exposed via `self.last_ghost_fraction` (an
   attribute — `Agent.transition_distribution`'s ABC signature is fixed
   to a single `ndarray` return). `act()` is a Phase-5 placeholder
   (`NotImplementedError`); a mock test
   (`tests/agents/worldmodel/test_agent.py::test_act_raises_not_implemented_and_never_calls_transition_distribution`)
   pins that it never calls `transition_distribution`, written to keep
   passing unmodified once Phase 5 replaces `act()` with the real
   policy.

## 2026-09-24 — Phase 3 closed

Two-level measures testing, both done:

**(a) Analytical, numpy-only** (`measures/latent_controllability.py`,
`measures/kernel_probe.py`, `measures/diagnostics.py`, hand-built
factorised distributions, all in `tests/measures/`): identical actions
give `I_z == 0`; disjoint one-hot actions give `I_z == log2(5)` exactly
(zero MC variance by construction); a hand-computed intermediate case
(binary-entropy(0.6) bits) recovered through the real MC codepath; MC
standard error empirically scales ~1/sqrt(M); the unimix floor's
resulting `I_z` matches `ControllableNoisy(p=ratio).analytic_controllability()`
exactly (reusing that existing closed form, not re-deriving one); a
synthetic factorised "cross combination" is classified a ghost with the
analytically-exact ghost rate (3/6 = 0.5 by construction).

**(b) "Mini" end-to-end** (`configs/e3/mini.yaml`, base_depth=8,
h_dim=embed_dim=64, n_vars=n_classes=8, k=2 → 24×24 images, ~20000
gradient steps, profiled at ~8 grad steps/s on this CPU-only machine —
~35-45 min total including diagnostics, under the hour target with
margin). `experiments/E3_worldmodel_4zones/run_mini.py` is the real
run (not executed as part of this session — too long for one turn; the
PowerShell command is below). `tests/e3/test_e3_mini_pipeline.py` is
the FAST smoke-test version (a much tinier model, ~30s total) exercising
every function in `probe.py` for structural correctness (shapes,
0<=I_z<=log2(5) bounds, ghost-fraction validity, alpha_app
definedness logic) — comparative claims (conveyor's I_z lower than
controllable-deterministic's) need real training progress and are only
asserted by `run_mini.py`'s own shape-check report, not by the fast
smoke test.

Bug found and fixed during this work (not a design deviation, a real
bug): `torch.no_grad()` does NOT strip `requires_grad` from tensors
produced by pure VIEW operations (`.expand()`, `.unsqueeze()`) on a
leaf `nn.Parameter` (only genuine ops like `Conv2d`/`Linear` do) — every
`.cpu().numpy()` call in `probe.py`/`agent.py` reachable from `RSSM.h0`
needed an explicit `.detach()` first. Fixed everywhere (grep-verified).
Also: `jensen_shannon_categorical` clips its result to `>= 0` — JS
divergence is mathematically non-negative, but float64 subtraction of
two near-equal cross-entropy terms can produce a ~1e-8 negative sliver;
clipped at the source rather than left for every caller to tolerate.

Housekeeping: `tests/agents/worldmodel/test_agent.py` renamed to
`test_worldmodel_agent.py` -- it collided with the pre-existing
`tests/agents/test_agent.py`'s bare module name (neither directory has
an `__init__.py`, so pytest imports both as `test_agent` and refuses the
second one). Matches the repo's existing convention for this exact
situation (e.g. `tests/agents/online/test_online_agent.py`).

## 2026-09-24 — Phase 3 code-review follow-ups (points 1-3)

1. **Unimix-floor test, replaced.** The old test compared
   `latent_controllability()`'s output against
   `ControllableNoisy(p=ratio).analytic_controllability()` -- a
   different implementation, but built on the same underlying
   symmetric-channel structure, so a shared conceptual bug could have
   passed both. Replaced with
   `test_unimix_floor_matches_a_hand_computed_value`
   (`tests/measures/test_latent_controllability.py`): a fully worked,
   independent by-hand derivation (n=5, r=0.1) exploiting an EXACT
   symmetry argument -- the marginal m(z) is provably uniform (1/n per
   class) because the construction is symmetric under jointly permuting
   action and class labels, so H(z'|s) = log2(n) exactly, no MC or
   external formula needed -- with the literal computed float
   (`1.759748904685089`) hardcoded and the full arithmetic shown in a
   comment. `ControllableNoisy` is no longer imported by this test file.

2. **`transition_distribution` restructured.** `self.last_ghost_fraction`
   removed. New method
   `WorldModelAgent.transition_distribution_with_diagnostics(s, a) ->
   (kernel, ghost_fraction, n_valid_draws)`; `transition_distribution(s,
   a)` is now a thin wrapper (ABC's fixed single-`ndarray` signature).
   All-ghost case: `kernel` is NaN-filled (never zero-filled, never
   silently renormalised), `ghost_fraction == 1.0`, `n_valid_draws ==
   0`, and a `logging.getLogger("helplessness.agents.worldmodel.agent")`
   warning is emitted -- never a silent division by zero. Tests updated
   (`tests/agents/worldmodel/test_worldmodel_agent.py`, `caplog`-based
   warning check added); the `materialize_transition_tensor`
   integration test now tolerates NaN rows instead of assuming
   zero-filled ones.

3. **`run_mini.py` rewritten to produce `docs/e3/MINI_REPORT.md`.** Runs
   TWO seeds by default (`configs/e3/mini.yaml`'s `seeds: [0, 5]` --
   class A and class B respectively, see the T2 reformulation above),
   collecting `N[s,a,s']` counts alongside the replay fill (a new,
   run_mini.py-local `collect_and_count`, not a modification of Phase
   2's `collect_fixed_policy_data`) so the tabular-twin comparison uses
   the exact same transitions. New library additions in support of the
   report: `losses.patch_only_reconstruction_error` (diagnostic-only,
   never part of the training loss) and `world_model_loss`'s returned
   dict gained a `recon_patch` key (additive, no existing key changed --
   confirmed no test asserted an exact key set). The report covers, per
   seed/class and per zone type: loss curves (`docs/e3/mini_report_figures/loss_curves.png`)
   for total/recon/recon_patch/kl_dyn/kl_rep; canonical I_z mean+/-std+MC
   se (`iz_by_zone.png`); KL(posterior‖prior) by zone (post-hoc from the
   canonical probe -- see the function's own docstring for why this,
   not a per-batch training-time breakdown, which would need per-sample
   zone attribution inside random training windows); template-match
   accuracy and z-only-probe accuracy; L1 of the s-space kernel vs the
   true `P` and vs the tabular twin `B` (built via
   `helplessness.agents.tabular.TabularModelBasedAgent` directly, alpha=1,
   E2's own geographic support -- reused, not reimplemented); a T4
   preview (mean p_hat(frequent outcome) in the two noisy zones); ghost
   rate; context dispersion; the full D1 (alpha_app, local and global)
   table; D2 boundary-vs-within-zone JS distances; and the
   expected-shape checklist table.

   `replay_ratio` reduced from 4 to 2 in `mini.yaml` so the TWO-seed
   total stays close to the original one-seed "under an hour" budget
   (~40-50 min for both seeds combined, not per-seed on top of that --
   flagged explicitly since the target was originally phrased as a
   single run).

   **Dry-run validated** (this session, budget overridden to 60
   steps/10 prefill/1 gradient-step-per-transition, both seeds): the
   full pipeline ran end to end with no exceptions and produced a
   complete, correctly-structured report (all tables, both figures).
   The dry run's own numbers are exactly what an undertrained toy model
   should show (I_z near zero with occasional small negative MC
   excursions inside tolerance, 0% template-match accuracy, D1
   universally "undefined -- no contrast") -- not evidence of a bug, and
   NOT kept: the placeholder `MINI_REPORT.md`/figures this dry run
   produced were deleted afterward so they cannot be mistaken for real
   results. The real run (~40-50 min, not executed this session) is
   yours to launch.

To run the real "under an hour" mini config (PowerShell, repo root):

    .venv\Scripts\python.exe experiments\E3_worldmodel_4zones\run_mini.py

Writes `docs/e3/MINI_REPORT.md` and `docs/e3/mini_report_figures/`.

## 2026-09-24 — Posterior-collapse diagnosis (after the first real mini run)

The first real `run_mini.py` run (both seeds) showed a complete
posterior collapse: KL identical in every zone to 4 decimals, D2 = 0,
template-match accuracy ~8%, I_z ~0 within MC noise. Diagnosed before
touching any rendering or calibration hyperparameter (neither was
changed).

### 1. Loss scale -- confirmed, fixed

- **Aggregation (before):** `weighted_reconstruction_loss` took the MEAN
  over batch x time x channels x pixels (1728 elements per 24x24x3
  image). The KL terms are exact, SUMMED over the n_vars latent
  variables, averaged over batch x time, in nats. Pixels are in [0, 1]
  at the decoder input and in the target; the decoder output is linear
  (unbounded), compared to targets in [0, 1].
- **Values before** (reproduction, seed 0, mini config, 2000 gradient
  steps, first / last 50-step means; weights recon x1 with w_patch=10
  inside the mask, kl_dyn x0.5, kl_rep x0.1): recon 0.290 -> 0.0066,
  recon_patch 0.334 -> 0.036 (per-element mean on the patch), kl_dyn
  0.0165 -> 4e-6 nats, kl_rep identical (free bits off: same KL value,
  only the stop-gradients differ). KL collapsed within ~100 steps.
- **Why collapse was optimal:** at k=2 the agent marker is exactly ONE
  pixel of 576 (measured, every state). A decoder that ignores z and
  draws only the background errs by 0.00029 per element (0.49 summed
  per image) -- the recon plateau (~0.003 after the x10 patch weight)
  is exactly that background-only solution. Identifying the state costs
  >= ln(144) ~ 4.97 nats of KL (x0.6 ~ 3.0 in loss units) against a
  reconstruction gain of at most ~0.003 under the mean: collapse wins by
  three orders of magnitude.
- **Fix:** SUM over channels and pixels of each image, MEAN over batch x
  time only (DreamerV3 convention), mask weighting unchanged
  (`losses.weighted_reconstruction_loss`; `patch_only_reconstruction_error`
  switched to the same aggregation so the two stay comparable). Pinned
  by `test_weighted_reconstruction_loss_reduces_to_summed_squared_error_when_w_patch_is_1`
  and `test_weighted_reconstruction_loss_does_not_shrink_with_image_size`.
- **Values after** (same reproduction): recon 498 -> 10.6 per image,
  recon_patch 3.57 -> 0.37, kl_dyn = kl_rep 0.115 -> 0.0034 nats. The
  scales are now commensurate (encoding the state is now worth ~4.9 in
  reconstruction, 0.49 x w_patch, against ~3.0 of KL -- a thin margin,
  because of the 1-pixel marker), but **the KL still goes to ~0**: the
  aggregation fix is necessary, not sufficient -- see point 2.

### 2. Autoencoder isolation -- the categorical bottleneck fails

`experiments/E3_worldmodel_4zones/ae_isolation.py`: RSSM.encoder ->
RSSM.posterior_probs at the fixed initial h -> z -> RSSM.decode, on the
144 isolated state images, same weighted loss, no KL / prior / GRU /
dynamics. Criterion: template accuracy >= 95%.

| z path | k | accuracy | mean posterior entropy (ceiling 24 bits) |
|---|---|---|---|
| straight-through categorical (the RSSM's own) | 2 | 0.007 (chance) | 0.86 |
| straight-through categorical | 6 | 0.007 (chance) | 1.21 |
| deterministic (z = posterior probabilities) | 2 | **0.986** | 17.97 |
| deterministic | 6 | **0.993** | 14.76 |

(lr 1e-3, 3000 steps. At the world model's lr 1e-4 the deterministic
path is simply not converged after 3000 steps -- loss 8.3 vs 0.9 --
not failing.)

- **Rendering, mask and encoder/decoder are not the problem:** with a
  deterministic z the same modules represent all 144 states (>= 98.6%,
  both render scales, two seeds).
- **The straight-through categorical bottleneck is:** it fails at
  chance with NO KL at all, and not by staying uniform -- the posterior
  settles on one near-deterministic code (~1 bit of 24) shared by every
  image. This is why fixing the aggregation alone leaves the KL at ~0.
- In the suite: `tests/e3/test_e3_ae_isolation.py` -- deterministic path
  must pass; straight-through is a STRICT xfail (fails red the moment a
  fix makes it pass, so the marker gets removed deliberately).
- **Not fixed -- your decision.** The fix belongs to the latent
  mechanism (initialisation of the logit layers, the gradient estimator,
  or free bits, the last being a spec 3.2 parameter currently at 0). None
  touched, per the instruction not to change hyperparameters at this
  stage.

### 3-4. Report changes (`run_mini.py`)

- Collapse diagnostics: I(z; s) over the canonical posteriors
  (`measures.latent_controllability.latent_state_information`, uniform
  over the 144 states, ceiling log2(144) = 7.17 bits), mean posterior
  entropy (canonical and over every replay posterior), and a per-seed
  grid figure (input / mask / posterior reconstruction for 8 states, two
  per zone).
- I(z; s) and I_z now share ONE core, `mixture_information` (components =
  actions for I_z, states for I(z; s)) -- no second implementation.
- Bounds 0 <= I_z <= log2(5): per state, within 3 MC standard errors.
- Orderings: "OK"/"FAIL" only when the gap exceeds 3 combined standard
  errors (per-zone mean MC se = sqrt(sum se_s^2)/n; combined =
  quadrature sum), otherwise "non concluant". Only MC noise counts: the
  states are a fixed, complete set, so their spread is signal, not error.
- Template-match accuracy < 0.5: s-space kernel, L1, T4 preview, ghosts,
  D1, D2 are reported as "non évaluable (mécanique en échec)".
- z-only probe trained on every replay posterior
  (`ReplayBuffer.chronological_windows`, ~5.2k samples, not 410).
- Loss-term table (first / last 50 steps, with weights) in the report.
- Rules pinned in `tests/e3/test_e3_mini_report.py`.

Also fixed along the way: `run_mini.py` never seeded torch (network
initialisation and latent sampling were not reproducible) --
`torch.manual_seed(seed)` per seed now, full per-stream seed layout
deferred to Phase 4a; and the KL-by-zone table was labelled bits while
computed in nats -- now converted to bits.

The report code was dry-run end to end on a tiny budget; your real
`MINI_REPORT.md` and its figures were backed up first and restored
afterwards (the dry run's placeholder output was discarded).

Found by the full suite: `ae_isolation.run()` first imported `run_mini`
with a bare `import`, which (in a full test session where E1's `env.py`
is already cached under the bare name `env`) made run_mini bind the
WRONG `env` module -- and the strict xfail silently counted that
AttributeError as its expected failure. Fixed: loaded through
`helplessness.experiment_loader.load_experiment_module`, and the xfail
now only accepts an `AssertionError` (the accuracy criterion itself).
Full suite: 645 passed, 2 skipped, 1 xfailed (the straight-through test,
for the right reason).

## 2026-09-24 — Categorical-bottleneck fix: decisions and audit

**0. Free bits stay at 0.** The collapse reproduces in the autoencoder
isolation test, which has NO KL term at all -- so it is not a KL
problem, and a KL-side remedy (free bits) would treat a symptom the KL
does not cause. `free_bits` remains 0 in every config.

**1. Audit (before comparing estimators).**
- **Straight-through implementation: correct.** On a toy case (one
  variable, 3 classes, unimix r = 0.01, L = w . z) the autograd gradient
  equals the analytic `(1 - r) (diag(p) - p p^T) w` to 1e-12; the same
  check for Gumbel-softmax ST (noise replayed from the seed) matches its
  analytic chain rule to 1e-10, and its forward pass samples exactly
  Categorical(q) (`tests/agents/worldmodel/test_rssm.py`). The collapse is
  not an implementation bug.
- **Logit output layers now start at small scale** (new
  `RSSMConfig.logit_init_scale = 0.01`: default init x 0.01, bias zeroed,
  on both the prior and the posterior head), so both distributions start
  near uniform, as in DreamerV3 (`test_logit_layers_start_near_uniform`).
- **Between-state JS diagnostic**: `measures.diagnostics.mean_pairwise_jensen_shannon`
  (mean over all pairs of states of the per-variable-summed JS between
  their posteriors; 0 = every state gets the same posterior, n_vars bits
  = pairwise-disjoint point-mass codes). Recorded at initialisation and
  every 100 steps in the isolation test. A first functional check (k=2,
  60 steps) already shows plain straight-through and Gumbel-ST reaching
  a near-deterministic posterior (~1 bit of 24) with JS = 0 -- one
  confident code shared by every state -- within 60 steps, from a near-
  uniform start.
- **Performance bug found and fixed (no change to any result):**
  `measures.kernel_probe.match_templates` computed squared distances by
  broadcasting a (decoded x templates x pixels) array -- 144 x 144 x
  15552 float64 = 2.6 GB per call at k=6 (9x smaller at k=2, which is why
  it went unnoticed). The first estimator-comparison launch spent most
  of its time there (3.2 GB processes, no seed finished after ~40 min)
  and was stopped. Now |a|^2 + |b|^2 - 2 a.b (memory K x n_templates):
  0.05 s for the same call, identical nearest template and margins to
  1e-9 (`test_match_templates_distances_equal_the_direct_pairwise_computation`).
  run_mini.py's kernel probe and template-match accuracy at k=6 would
  have hit the same wall.

**2. Estimator comparison -- (c) retained.** Full report and curves:
`docs/e3/ESTIMATOR_COMPARISON.md` (raw curves:
`results/E3_worldmodel_4zones/estimator_comparison/`). Autoencoder
isolation, no KL, k=6, 3 seeds, schedules fixed before the runs: 6000
steps, lr 1e-3, batch 32; (b) tau 1 -> 0.1 exponentially over the first
3000 steps, then held; (c) posterior probabilities for the first 3000
steps, then plain straight-through. (The half-way switch point was set
from what the deterministic path was already known to need to separate
the states at this lr -- ~3000 steps -- before any categorical run at
k=6; an earlier 3000-step draft would have switched before separation
could exist.) Evaluation: hard one-hot sample, template accuracy
averaged over 5 draws; criterion >= 0.95 on all 3 seeds.

| estimator | hard-sample accuracy (seeds 0/1/2) | passes |
|---|---|---|
| (a) straight-through | 0.007 / 0.007 / 0.008 | no |
| (b) Gumbel-softmax ST | 0.007 / 0.007 / 0.008 | no |
| (c) probabilities warm-up, then ST | 0.999 / 0.999 / 0.994 | **yes** |

- (a) and (b) collapse within the first ~100 steps (posterior entropy
  24 -> ~1 bit, between-state JS 0) and never recover.
- (c) separates the states progressively during the warm-up, then the
  switch to straight-through SHARPENS rather than re-collapses (seed 0:
  hard accuracy 0.56 -> 0.85 in 100 steps, 0.99 by step 3500; JS keeps
  rising to ~6.5 of 8 bits). The collapse is a start-up trap: once the
  states are separated, plain straight-through is stable.
- Two cautions from the curves: the separation onset varies by seed
  (~1000-2000 steps for two seeds, ~2600 -- just before the switch -- for
  the third), so a shorter warm-up could have failed; and one seed shows
  a transient accuracy dip (to ~0.60 near step 5900) that recovers by
  6000.
- **Selected: (c)** (the only option that passes; also the preferred one
  on a tie).
- **Deviation from DreamerV3, training procedure only:** DreamerV3 feeds
  straight-through samples to the decoder from the first step; here the
  decoder receives the posterior probabilities for the first
  `warmup_steps`, then straight-through samples. Unchanged: the model's
  probabilistic definition (categorical posterior/prior, 1% unimix), the
  prior samples, the canonical probe, and I_z by exact densities. Also
  new (DreamerV3-like): small-scale init of the logit output layers.
  Implemented as `rssm.LatentEstimator` + `loop.EstimatorSchedule`;
  training entry points take `estimator=`, default plain
  straight-through (every other code path unchanged).
- Set in `configs/e3/mini.yaml` as `latent_estimator: {kind:
  warmup_probs, warmup_steps: 5256}` -- half the mini training budget,
  the fraction the comparison validated. **Open risk:** the comparison
  ran at lr 1e-3, the mini config trains at the spec's 1e-4, so the states
  may separate later than step 5256. run_mini.py now records the
  canonical separation trajectory (hard-sample accuracy, between-state JS,
  posterior entropy every 250 steps; figure `separation_curves.png`, with
  the switch and the end of the KL warm-up marked) so the run itself
  shows whether the switch came after separation. Not tuned.

**3. KL warm-up (full RSSM).** beta_dyn and beta_rep ramp linearly from
0 to 0.5 / 0.1 over `train.kl_warmup_steps`, then hold
(`loop.kl_warmup_betas`); mini default 2102 = 20% of the training budget
((5000 + 256) x 2 = 10512 gradient steps). Rationale: the KL cost of
encoding the state (~ln 144 ~ 5 nats) only exists at the start of
training, before the prior has learned the dynamics -- once the prior
predicts the posterior, the KL is small again. The warm-up removes that
start-up cost without leaving any permanent bias on the learned kernel
(the full weights apply for the remaining 80% of training).

**4. k=6 restored** in `configs/e3/mini.yaml` (the spec default; the
1-pixel marker at k=2 was not the design). Consequence: ~2.4 gradient
steps/s instead of ~8, so run_mini.py now takes ~2.5 h for both seeds
(~73 min of training per seed). Budget and replay_ratio deliberately
unchanged.

**5. Categorical isolation test** (`tests/e3/test_e3_ae_isolation.py`):
now a real test with the retained estimator, expected-failure marker
removed. Short version: same switch point (step 3000), stopped 500 steps
after it (seed 0 was already at 0.986 at step 3500 in the full run);
passes, ~2.5 min. The deterministic-path test left the suite (runtime at
k=6); still available via `ae_isolation.py --estimator deterministic`.

Smoke test on the full RSSM (`tests/e3/test_e3_mini_pipeline.py`, now
training with the retained estimator schedule and the KL warm-up, step
counts scaled to its tiny budget): passes. The canonical separation
evaluation now lives in one place, `probe.canonical_separation`, shared
by the isolation test and run_mini.py. run_mini.py was dry-run end to
end at k=6 on a tiny budget; the real MINI_REPORT.md and its figures
were backed up first and restored afterwards.

## 2026-09-24 — Option (c) validated; criterion-triggered switch and restructured KL schedule

**1. Switch triggered by a label-free criterion, not a fixed step.**
- Statistic: mean Jensen-Shannon divergence (bits, summed over the
  latent variables) between the posteriors of `switch_n_pairs` = 64
  random pairs of images from the CURRENT training batch
  (`measures.diagnostics.mean_random_pair_jensen_shannon`). No state
  label is involved -- the function has no label argument, and the
  training step hands back only the batch posteriors
  (`train_world_model_step(..., return_posteriors=True)`). Two random
  images almost always show different states, so the statistic tracks
  between-state separation. Its pair sampling uses its own RNG stream.
- Rule (`loop.SeparationSwitch`): evaluated every `switch_eval_every` =
  50 steps; the decoder switches from the posterior probabilities to
  straight-through samples once the statistic exceeds
  `switch_js_threshold` = 3 bits on `switch_consecutive` = 5 consecutive
  evaluations taken at step >= `switch_min_steps` = 500, effective from
  the next step. If it has not fired by `switch_max_steps` (default 60%
  of the training budget), the run stops with `SeparationNotReachedError`
  ("separation non atteinte") -- never a silent switch. A configuration
  whose deadline precedes the earliest possible switch is rejected at
  construction. The effective switch step is printed, stored in the run's
  report, and marked on the separation figure.
- Tracking check: see `docs/e3/ESTIMATOR_COMPARISON.md`, section
  "Label-free switch criterion" (correlation between the batch statistic
  and the labelled between-state JS; the criterion-triggered procedure on
  the same 3 seeds).

**2. KL schedule restructured by the role of each term** (replaces the
linear warm-up of both terms).
- beta_dyn = 0.5 from step 0. KL(sg(q) || p) has the posterior under a
  stop-gradient, so it only trains the prior and never pulls the
  posterior toward it: it cannot cause or worsen collapse, and letting it
  run from the start lets the prior learn the dynamics while the
  posterior separates.
- beta_rep = 0 until the switch, then a linear ramp to 0.1 over
  `rep_warmup_steps` (default: 20% of the training budget left after the
  switch), then held (`loop.rep_beta`). KL(q || sg(p)) is the term that
  pulls the posterior toward the prior -- the pressure that makes
  encoding the state costly (~ln 144 nats) before the prior has learned
  anything. It is withheld until the posteriors are separated, then
  introduced gradually.
- `kl_warmup_betas` and `train.kl_warmup_steps` removed.

**3. Console monitoring** (`run_mini.py`, every `measure.monitor_every` =
500 steps): step, latest batch-pair JS, phase (probabilities /
straight-through), current beta_rep, patch reconstruction loss -- plus a
line at the effective switch. Enough to interrupt early.

**4. What the short smoke test on the full RSSM checks** (now stated in
`tests/e3/test_e3_mini_pipeline.py`'s docstring): STRUCTURE ONLY -- every
pipeline stage runs without error with the retained procedure (both
phases, the switch, the beta_rep ramp -- asserted explicitly), shapes,
simplices, I_z bounds within MC tolerance, ghost fractions in [0, 1],
alpha_app definedness logic. It does NOT check separation: 40 steps on a
tiny model cannot separate 144 states, so its switch threshold is scaled
to a value that is always met. Separation is checked by the categorical
isolation test (no dynamics) and, for the full RSSM, only by run_mini.py.

Results of the criterion (isolation test, k=6, same 3 seeds, retained
defaults, deadline 3600 = 60% of 6000; `docs/e3/ESTIMATOR_COMPARISON.md`):
- **Tracking**: the (c) runs were re-run with the same seeds and
  settings to add the batch statistic; all 3 x 61 curve points
  reproduced the stored values exactly. Label-free batch-pair JS vs
  labelled between-state JS: Pearson r = 0.999, Spearman rho = 0.981 (180
  points), mean gap 0.09 bits; on the criterion-triggered runs, 0.999 /
  0.947. Scope: in the isolation test the batch images are distinct
  states at a fixed initial h, so this mostly validates the estimator; in
  the full RSSM (real contexts h_t, possible repeated states) the
  tracking is shown by run_mini.py's separation figure, which plots both.
- **Criterion-triggered switch**: effective at steps 2301 / 1701 / 3101
  (seeds 0 / 1 / 2) -- the late-separating seed switches later on its own
  -- final hard-sample accuracy 0.999 / 1.000 / 0.999: passes on all 3.
- The categorical isolation suite test now uses this procedure: same
  rule and deadline, stops 500 steps after the effective switch; passes
  (~2 min).
- Full suite: 669 passed, 2 skipped.

## 2026-09-24 -- the lr 1e-4 separation failure: diagnosis and fixes

The user's run_mini.py at lr 1e-4 hit the switch deadline
(SeparationNotReachedError at step 6307; label-free JS 0.067, labelled JS
0.064, moving together -- so not an artefact of the context h) although
recon_patch fell from 74 to ~1.5. Hypothesis (user): the state is
carried by tiny differences between near-uniform probabilities, which
the decoder can read, with no pressure to commit to a code.

**Failure diagnostic.** run_mini.py now saves, when
SeparationNotReachedError is raised, the checkpoint (model + optimizer)
and `probe.latent_usage_diagnostic` into
`results/E3_worldmodel_4zones/mini_failures/seedNN__stepN__<UTC>/`:
template accuracy with the decoder fed (a) the probabilities, (b) the
mode one-hot, (c) a hard sample; mean posterior entropy; I(z;s)
(`latent_state_information`); mean posterior logit norm (centred over
classes, and raw). The failed run saved no checkpoint, so seed 0 was
re-run to the deadline (`run_mini.py --seeds 0`; torch is seeded since
the previous round, so it is the same run).

**Reproduction in the isolation test** (no KL, k=6, (c') unchanged, lr
1e-4, 10,000-step budget, deadline 6000, batch 32;
`experiments/E3_worldmodel_4zones/compare_probs_phase_aids.py`):
reproduced -- seeds 0 and 1 hit the deadline. At the deadline:
accuracy with probabilities 0.354 / 0.812, with the mode 0.014 / 0.076,
with a hard sample 0.019 / 0.006; posterior entropy 23.34 / 23.28 bits
(max 24); centred logit norm 0.95 / 1.06; I(z;s) 0.03 / 0.22 bits. The
hypothesis holds: the probabilities carry the state, a discrete draw
does not. The batch posterior entropy even RISES during the phase (15.2
-> 16.2 nats, max 16.64): at this lr the reconstruction pushes towards
near-uniform posteriors rather than away from them.

**eta_max for (ii), fixed before the (ii) runs.** eta_max = 1 (loss
weight per nat, H(q) summed over the 8 variables as the KL is). Scale:
the per-image weighted reconstruction at the end of the probabilities
phase in the reproduction runs is ~44-68; the maximum entropy is
8 ln 8 = 16.6 nats, so at the end of the ramp the penalty on a
near-uniform posterior is ~25-38% of the reconstruction -- a real
pressure that does not dominate it. A penalty equal to the
reconstruction (eta ~3) was rejected: in a 720-step functional check at
eta_max = 1 with a short ramp (reconstruction still ~500), the penalty
already produced a low-entropy code shared by all states (JS 0.001),
i.e. commitment without separation is a live failure mode. Ramp: linear
from 0 at step 0 to eta_max at the switch deadline (6000) -- the only
length of the probabilities phase known in advance (flagged: "over the
phase" read as "over the phase's maximum length"); removed at the switch.
(iii)'s rho ramps over the same horizon, for the same reason; the
sampled variables are drawn at random, independently per example and per
step (exactly round(rho * 8) of them).

**Results.** Every row, and the curves: `docs/e3/ESTIMATOR_COMPARISON.md`,
last section.

- **Mini RSSM, seed 0, re-run to the deadline**: reproduced the user's
  run (deadline at step 6307, last batch-pair JS 0.067 bits). Diagnostic
  at the deadline (`results/E3_worldmodel_4zones/mini_failures/seed00__step6307__20260924T155341Z/`):
  accuracy with probabilities 0.236, with the mode 0.035, with a hard
  sample 0.018; posterior entropy 23.62 / 24 bits; labelled JS 0.063
  bits; I(z;s) 0.13 bits; centred logit norm 0.73. This matches the
  isolation test: the hypothesis holds in the full RSSM too.
- **(ii) entropy penalty, eta_max = 1**: fails on all 3 seeds. It commits
  without separating: the posterior entropy falls to 4.5 / 7.8 / 0.78 bits
  but the labelled JS stays at 0.03 / 0.05 / 0.00 bits. Seed 2 ends on a
  saturated code shared by every state.
- **(iii) progressive samples**: fails on all 3 seeds. Seeds 1 and 2
  collapse to a constant code (JS 0) as rho rises; seed 0 separates only
  partly (JS 1.3 bits, hard-sample accuracy 0.056).
- **(i) lr 1e-3**: passes on all 3 seeds. Switches at steps 2301 /
  1701 / 3101, with >= 0.95 first reached at steps 2900 / 2100 / 3600;
  final hard-sample accuracy 1.000 / 0.997 / 1.000. At the triggering
  evaluation:
  - accuracy with probabilities 0.979 / 0.965 / 0.972, with a hard
    sample 0.417 / 0.496 / 0.439;
  - labelled JS 3.30 / 3.93 / 3.99 bits.
- **An instability seen on the way, which must not be lost.** The first
  batch of (i) runs had a bug: the at-switch diagnostic snapshot drew its
  hard samples from torch's global RNG, which shifted every later
  straight-through draw. A measurement must not perturb training, so the
  snapshot now runs under `torch.random.fork_rng()` (without the fix,
  the short isolation suite test ended at 0.946 instead of passing: 500
  steps after the switch, the margin over 0.95 is thin). (i) was re-run
  with the fix, and its results above come from that re-run.
  In that first, discarded batch, seed 0 reached 1.000 and then
  collapsed in a SINGLE step. A re-run keeping the per-step loss
  reproduced it deterministically: the loss is 1.1-2.1 up to step 9981
  and 40.1 at step 9982, and it never recovers. The final posterior is
  one saturated code for every state: entropy 0.7756 bits, exactly 8 x a
  unimix'd one-hot. So the straight-through phase at lr 1e-3 CAN blow up
  late, depending on the sampling stream. It is 1 event in 6 runs of
  10,000 steps, and the corrected stream does not hit it. The isolation
  loop has no gradient clipping, and run_mini's clip_norm of 1000 is
  effectively none either.
- The corrected-stream re-run also used 3 threads instead of 1, and
  seed 2 then switched at 3101 instead of 3001. Trajectories are not
  bit-exact across thread counts (reductions are ordered differently).
  Seeds 0 and 1 switched at the same steps as in the 6000-step
  comparison.

**Selection (pre-declared rule, order (ii), (iii), (i)): (i).** Scored on
the final evaluation at step 10,000 -- the reading encoded before the
runs; the literal reading "reached before 10,000 steps" gives the same
selection. Applied: `configs/e3/mini.yaml` train.lr 1e-4 -> 1e-3, with
procedure (c') unchanged and no aid (ProbsPhaseAid stays in the library,
unused by run_mini). Per the rule, lr is now a hyperparameter calibrated
in Phase 4.
**Flag for the user:** the late one-step collapse above is a documented
risk of this setting. Gradient clipping at a finite norm, or a lower lr
after the switch, are the obvious guards. Neither was run or applied:
both are the user's decision.

## 2026-09-24 -- (i) validated; the late collapse: diagnosis, fix, guard, reproducibility

**Diagnosis of the late collapse.** Deterministic reproduction of the
discarded (i) batch, seed 0 (1 thread, at-switch snapshot drawing from
the global torch RNG: `ae_isolation.run(..., legacy_snapshot_rng=True,
log_step_diagnostics=True)`), logged per step. The isolation loop has no
GRU and no prior; its modules are encoder, posterior_net, decoder and h0.
- **Before the event (straight-through phase, steps 2800-9960, medians):**
  global gradient norm ~53, almost all of it in the decoder (51.4). The
  encoder gets 0.38 and posterior_net 1.4. The posterior logits grow
  steadily: centred norm 4.8 at the switch, 17 at step 5000, ~25 at step
  9975. Batch posterior entropy is ~1.9 bits (24 max): the softmax is
  saturated, so the straight-through gradient into the encoder is tiny,
  and so is Adam's second-moment estimate for the encoder.
- **The event:**
  - step 9980: the encoder gradient jumps from < 1 to 64.7 (global 77.7);
  - step 9981: 9.4;
  - step 9982: the loss jumps from 1.3 to 40.1 (encoder gradient 251,
    decoder 259, global 361);
  - then the centred logit norm runs away (30 -> 93 -> 154 by step 9989),
    batch entropy falls to 0.78 bits (one saturated code), and the
    encoder gradient drops to ~0.001.
- **Probable cause:** a rare batch gives the encoder a gradient ~100x
  its usual scale. Its Adam second moment, built on near-zero gradients,
  is tiny, so the normalised step is a large coordinated jump of order
  lr on every encoder parameter (~1e-3 at lr 1e-3). That jump throws the
  posterior into a saturated, state-independent code. Once saturated,
  the straight-through gradient through the probabilities vanishes, so
  nothing can bring it back. Root conditions: a saturated posterior (so
  a starved Adam state), plus lr 1e-3 in the straight-through phase.
- **Consequence for the fix, flagged:** a global clip at ~10x the median
  would NOT have acted on this event. The global norm was 77.7 at the
  kick (step 9980), and the per-parameter Adam step is insensitive to
  gradient scale anyway. The operative fix is lr_after_switch (10x
  smaller step). Clipping guards against global spikes of a different
  kind (see below).

**clip_norm, fixed before the verification runs: 500.** Global gradient
norm in the stable straight-through phase (from 500 steps after the
switch) of the existing (i) runs, re-run instrumented:
- median 50.0 / 65.7 / 47.7 (seeds 0 / 1 / 2 of the retained re-run),
  51.7 (discarded seed 0 before the event); pooled median ~51;
- 10 x ~51 = 510, rounded to 500.
What a clip at 500 would have touched:
- the first ~200 steps (initial reconstruction transients: 41-94% of
  those steps exceed 500);
- nothing else in the probabilities phase;
- 0-5 isolated straight-through spikes per run (up to 1770).
**Scale caveat:** these are ISOLATION-TEST norms. The RSSM's gradient
(sequences, KL, GRU/prior) has a different scale, and no RSSM run has
reached its straight-through phase yet. The 500 in mini.yaml is this
value carried over. run_mini's console line now prints the global
gradient norm so its scale can be checked.

**Two-phase lr.** lr 1e-3 until the effective switch, then
lr_after_switch = 1e-4 (the spec's value) on every world-model
parameter (`loop.phase_learning_rate` / `set_learning_rate`). The Adam
state is kept across the change. run_mini logs the change.

**Guard after the switch.** `SeparationSwitch` keeps evaluating the
label-free batch-pair JS after the switch and raises
`CollapseAfterSwitchError` once it is < 1 bit on 5 consecutive
evaluations. run_mini saves the checkpoint and the diagnostic, as for
SeparationNotReachedError, under
`results/E3_worldmodel_4zones/mini_failures/seedNN__collapse_after_switch__stepN__<UTC>/`.
The console line gains the global gradient norm and the current lr.

**Reproducibility.** `runtime.num_threads` in configs/e3/mini.yaml
(6 = this machine's torch default, at which the throughput was
profiled). run_mini applies it and records num_threads, the torch
version and the git commit in each run's report and failure
diagnostics; isolation-test results record num_threads. Reproducibility
is documented as "same machine, same number of threads". Observed: seed
2 of (i) switches at 3001 with 1 thread and at 3101 with 3.
**Note for Phase 4b:** define the GPU determinism policy
(torch.use_deterministic_algorithms, cuDNN deterministic/benchmark
flags, CUBLAS_WORKSPACE_CONFIG) before the cluster runs.

**Verification of the fix (lr 1e-4 after the switch + global clip at
500), isolation test, 3 seeds x 2 lots: FAILS.** No run collapsed after
the switch, but the clip breaks the probabilities phase.
- **Seed 0 passes in both lots:**
  - current stream: switch at 1101, final accuracy 0.992;
  - the discarded batch's stream, i.e. the deterministic reproduction of
    the collapse: switch at 1101, final 0.988, so it no longer collapses.
- **Seeds 1 and 2 fail in both lots:** SeparationNotReachedError at the
  deadline, with batch/labelled JS 0.000 from step 200 on, entropy
  21-24 bits, and reconstruction falling to ~50 (the mean image, latent
  unused).
- **How little the clip touched:** only the initial transient -- the
  first 60 steps for seed 2, 79 steps up to step 919 for seed 1.
- **Probable cause:** the very large initial gradients (up to ~20,000)
  that the clip truncates are what make the posterior image-dependent at
  the start; without them, training takes the "mean image" path and
  never leaves it.
- Options put to the user: clip only after the switch (recommended), no
  clip, or a threshold above the initial transient. mini.yaml
  temporarily holds clip_norm 500; run_mini must not be launched until
  this is decided.

## 2026-09-25 -- global clipping disabled; lr after the switch is the only fix

Decision (user): no global gradient clipping. `clip_norm: null` in
configs/e3/mini.yaml (the key stays for the record; `clip_norm=None` is
now supported by `train_world_model_step`, which still reports the
pre-clip global norm). Only lr 1e-3 -> 1e-4 at the effective switch is
retained as the fix. Reasons:
- **The clip does not act on the identified mechanism.** At the shock,
  the global norm was only 77.7. The mechanism is a tiny Adam second
  moment on the saturated encoder, then a gradient ~100x larger. A
  global threshold cannot see this, and a clip applied only after the
  switch would give a false sense of safety.
- **The clip breaks the probabilities phase.** The 500 clip made seeds 1
  and 2 fail in both lots (previous verification, archived in
  `results/.../probs_phase_aids/discarded_clip500_verification/`).
- **Only without the clip is the reproduction a controlled experiment.**
  The "discarded stream" run is then strictly identical to the collapsing
  run up to the switch and differs only by the lr afterwards. In the
  previous verification the clip changed the first steps, so its
  non-collapse proved nothing. The identity is checked, not assumed:
  the report compares switch step, per-step loss and global gradient
  norm against the collapsed run.
- **Limit:** the collapse was seen on 1 run out of 6. Six clean runs do
  not prove it cannot happen. `CollapseAfterSwitchError` stays active. If
  a collapse reappears, the envisaged guard is an adaptive per-parameter
  clip (AGC-style), not a global threshold; check what DreamerV3's
  reference implementation does at that point.
- **Note for Phase 4b:** the initial separation depends on the very
  large gradients at the start (~20,000 in the isolation test). Watch
  this with the full configuration; do not add a clip that truncates the
  first steps without re-running the separation check.
- The 500 figure and its derivation (previous section) are kept in this
  log as the record of what was tried.

**Verification without clipping: PASSES on all 6 runs** (isolation test,
k=6, no KL, 10,000 steps, lr 1e-3 -> 1e-4 at the effective switch, no
clip, `CollapseAfterSwitchError` guard active; full table in
ESTIMATOR_COMPARISON.md).
- Current stream (3 threads): switches at 2301 / 1701 / 3101, final
  hard-sample accuracy 0.994 / 0.992 / 0.993.
- Discarded batch's stream (1 thread): switches at 2301 / 1701 / 3001,
  final 0.990 / 0.993 / 0.996.
- No collapse: the minimum batch-pair JS after the switch is 3.4-4.5
  bits, far above the 1-bit guard.
- **Controlled reproduction, checked:** the discarded-stream lot's seed
  0 is bit-identical to the run that collapsed (same switch step 2301,
  same per-step loss and global gradient norm over steps 0-2301). The
  first loss that differs is step 2302: the lr change acts on the update
  of step 2301. Where the lr-1e-3 run went to loss 40 at step 9982, this
  run ends at 0.990. The same seed's collapse is therefore removed by
  the lr drop alone.
- Cost: the final accuracy is ~0.99 rather than ~1.000, with a posterior
  entropy of 3.4-4.5 bits against ~1.3 at lr 1e-3. The lr-1e-4 phase
  leaves the posteriors less sharp; still far above the 0.95 criterion.
- Retained for mini.yaml: lr 1e-3, lr_after_switch 1e-4, clip_norm null,
  collapse guard (1 bit, 5 consecutive), runtime.num_threads 6. Caveat as
  above: 0 collapses in 6 runs does not prove the risk is gone.

## 2026-09-25 -- after the mini run: labelled accuracies, D1 rule, absolute deadline, device, run outputs, RNG streams, long run

Context (user): the mini run's mechanics pass (replay template accuracy
0.99, z-only probe 0.98, I(z;s) ~6.86 of 7.17 bits, D2 3-4 bits), but the
prior has not learned the dynamics:
- KL(posterior || prior) is 9-18 bits per step;
- the L1 of the kernel vs P is 1.6-1.9;
- p_hat(frequent outcome) is 0.26-0.35;
- I_z of the conveyor is 0.5-0.65 bit.
Hypothesis: budget (~7 visits per (s, a), 4000-5700 steps after the
switch at lr 1e-4). This entry covers step A of the plan that follows:
the long diagnostic run is prepared, not run.

**Two template accuracies, labelled.** The report's 0.99 and the
separation figure's 0.70-0.87 measure different things:
- 0.99 = **replay windows (real contexts h_t), hard sample**
  (`probe.template_match_accuracy`: the replay unroll, z_t one
  straight-through draw).
- 0.70-0.87 = **canonical probe (fresh h0), hard sample, one draw**
  (`probe.canonical_separation`): each state's render encoded from the
  learned initial state h0. That context only occurs at the start of a
  replay window, so this number is expected to be lower.
Every report now gives three labelled numbers: replay/hard,
canonical/hard (mean of 5 draws), canonical/mode (run_mini.ACC_LABELS).
The mechanics gate (floor 0.5) uses replay/hard, as before.

**D1 rule.** A contrast (L, L') is accepted only if
|I_L' - I_L| > max(0.25 bit, 3 x combined MC se), where the zone means'
MC se come from `zone_mean_mc_se` (`measures.diagnostics.d1_contrast`;
config `measure.d1_contrast_min_bits`, which replaces
`contrast_threshold`). For an accepted pair, alpha_app is computed for
**every** state of L (not 2 per zone as before), local and global:
- each per-state value gets a standard error by first-order error
  propagation over I_swap, I_L and I_L' (`alpha_app_with_se`);
- a value outside [0, 1] by more than 3 of its se is an **anomaly**:
  listed with its state, never averaged;
- the others are summarised by their mean over the zone's states
  +/- 3 sd/sqrt(n) (`summarize_alpha_app`).
All three are pure functions in measures/, tested against hand values
and finite differences. The report's D1 table gives, per pair: contrast,
threshold, accepted, local and global mean [interval] (n kept / n), and
the anomalies. Cost: ~0.016 s per state, ~10 s for the 600 values.

**Switch deadline in absolute steps.** `train.latent_estimator.switch_max_steps`
is an absolute number of gradient steps in every config: 6307 in
mini.yaml (the value "60% of 10512" resolved to), 10000 in mini_long.yaml.
`build_separation_switch` rejects null. The margin (deadline - switch
step) is printed at the switch and recorded in results.json, meta.yaml
and the report (the mini run's seed 5 switched at 6201 for a deadline of
6307: margin 106). The isolation test's own `make_schedule` keeps its
documented "60% of n_steps" default: it is a test tool whose recorded
results used it, not a run config.

**Device (audit B1 + 1.4).**
- `runtime.device` (auto | cpu | cuda) in the config, and `--device` on
  the command line.
- The RSSM is built on the CPU right after `torch.manual_seed(seed)`,
  then moved, so its initial weights do not depend on the device. The
  background and the templates are moved too.
- The probes take their device from the model's parameters and move
  their own inputs (`canonical_separation` and `latent_usage_diagnostic`
  did not).
- The resolved device is printed at startup and recorded in meta.yaml.
- At startup, `torch.backends.cuda.matmul.allow_tf32 = False` and
  `torch.backends.cudnn.allow_tf32 = False`, also recorded. The A100
  default enables TF32 for convolutions, and the probabilities phase
  separates the states on tiny probability differences.

**Run outputs (audit B3 + 4.1 + 4.4).** One directory per seed through
`runs.make_run_dir` (never overwritten), named with `runs.make_run_name`:
`grid12x12__4zones__pair-experimental__worldmodel-<config>__seedNN__<UTC>`.
- `config.yaml`: the resolved config.
- `meta.yaml`: status, seed, commit and "working tree modified" flag,
  python / torch / CUDA / cuDNN / GPU, device, TF32 and determinism
  flags, threads, switch step, deadline and margin.
- `console.log`.
- `diagnostics.jsonl`: appended per point.
- `results.json` and `arrays.npz`, written at the end of each seed.
- `REPORT.md` and `artifacts/*.png`.
- `failure/`: checkpoint and diagnostic, if a guard fired. The guards
  used to write to `results/E3_worldmodel_4zones/mini_failures/`.
With several seeds, a combined report goes to its own directory. Nothing
is written to `docs/` any more; `docs/e3/MINI_REPORT.md` stays as the
record of the mini run that produced it.

A seed stopped by a guard no longer stops the others: it is marked
`failed`, the next seed runs, and the process exits with code 1.

Options: `--config`, `--device`, `--out-dir`, `--seeds`, `--max-steps`
(caps the gradient steps; the report says so). Windows limits paths to
260 characters and the run name is long: an `--out-dir` deep in a long
path (e.g. a temp directory) fails on Windows. The default `results/` in
this repository stays under 190 characters.

**Separate random streams (audit 3.1).** Each training consumer has its
own stream, and every measurement has streams of its own:
- `collect`: environment actions, `default_rng(seed)` as before, so the
  same data;
- `replay`: training minibatches, `default_rng([seed, 1])`;
- `switch_pairs`: `default_rng([seed, 4])`, the label-free statistic on
  its fixed schedule;
- `latent`: a dedicated `torch.Generator` on the model's device, used
  for every latent draw of the training step;
- measurements: separation curve, training diagnostics, I(z;s), z-only
  probe, context dispersion, D1 each draw from their own (numpy, torch)
  pair.
The network initialisation keeps the global torch RNG, seeded just
before the model is built.

To make this possible, `rssm.sample_categorical_index` draws through
`torch.multinomial(..., generator=g)` when a generator is given, and
through the historical `Categorical.sample()` path when it is None. The
generator is threaded through `straight_through_sample`, Gumbel,
`partial_straight_through`, `sample_posterior`, `observation_step`,
`imagination_step`, `unroll_observation`, `train_world_model_step`,
`sample_categorical_batch` and the probes.

Test (`tests/e3/test_e3_run_outputs.py`): the real `run_one_seed` on a
tiny config, same seed, run twice.
- Measured often: separation every 10 steps, training diagnostics every
  20 steps.
- Measured rarely: separation every 50 steps, no training diagnostics.
- Result: every training loss, gradient norm, lr and beta_rep is
  **bit-identical** (CPU, same threads), and so is the switch step.

**Trajectories are no longer bit-comparable to the runs before this
entry.** Two things changed:
- the latent draws now come from a different sampler call on a
  different stream;
- the minibatches come from a stream that is no longer shared with the
  measurements.
Same seed, same data, same initial weights; different training
trajectory. The isolation-test results (ESTIMATOR_COMPARISON.md) are
unaffected: that loop does not pass a generator, so it uses the
historical path.

**Long diagnostic run, prepared, not run: `configs/e3/mini_long.yaml`.**
- Budget: 20,000 environment steps at epsilon=1 (19,744 + prefill 256;
  ~28 visits per (s, a)) and 40,000 gradient steps (replay ratio 2).
- Switch deadline: 10,000 absolute steps. Seed 0 only.
- Procedure unchanged from mini.yaml; only the budget, the deadline,
  `measure.diagnostics_every: 2000` and the seeds differ.
Every 2000 gradient steps (and at the end), one line is appended to
`diagnostics.jsonl` and the figure `artifacts/training_diagnostics_seed00.png`
is redrawn:
- KL(posterior || prior) by zone;
- L1 of the s-space kernel vs P and vs B by zone, and L1(B, P);
- canonical I_z by zone with its MC se;
- p_hat(frequent outcome) in the two noisy zones;
- the canonical posterior entropy;
- the three labelled template accuracies.
Expected shapes, fixed in advance (`run_mini.shape_verdicts`), judged
between the first diagnostic at or after the switch and the last one:
- KL in the deterministic zones decreasing, to < 1 bit;
- L1(WM, P) decreasing toward L1(B, P);
- I_z of the conveyor decreasing;
- I_z of the controllable-deterministic zone increasing.
For the two I_z checks, a change within 3 combined se is "non concluant".

**Note, no action yet -- the canonical probe starts from h0.** The
canonical probe, which gives the main I_z measure, encodes every state's
render from the learned initial state h0. Training shows the model that
context less often than a real one, which explains the lower canonical
template accuracy: 0.70-0.87, against 0.99 in real contexts (replay).
Measured on mini.yaml, seed 0, 2000 x 8 sampled windows of seq_len 16:
- h0 is the context of 10.8% of training steps (~1 in 9);
- window starts contribute 1 in 16, and episode starts contribute 4.8%
  of the buffer's steps;
- episode starts always show the start cell, so the pairing the
  canonical probe uses -- h0 with the render of an arbitrary state --
  only occurs at window starts: ~6% of steps.
The figure first put forward (~1 in 64) corresponds to the spec's
seq_len of 64 without episode starts.
The long run (mini_long.yaml) will show whether the gap closes. If it
does not, the definition of the main measure is revisited in Phase 4a:
a probe from a short standardised context, or from replay contexts,
using the context-dispersion measure as the basis.

## 2026-09-25 -- episode starts: uniform over the non-terminal states (E2 protocol)

**Deviation found and corrected.** Every E3 run before this commit --
including the mini run of 2026-09-24 and every isolation-test data
collection that used the world-model loop -- started EVERY episode at
(0, 0). Both collection loops called `env.reset()` without a start state,
and `TorusZonedEnv` then falls back to `start_position=(0, 0)`
(`experiments/E3_worldmodel_4zones/env.py::build_pixel_env`). This has
been the case since Phase 2 (`loop.collect_fixed_policy_data`), and Phase 3
reused it in `run_mini.collect_and_count`. The fixed start was never a
decision: `docs/e3/ARCHITECTURE.md` planned E2's `start_state` stream and
it was never wired. It is a deviation from the spec (prompt sections 4.6
and 6: uniform starts) and from E2 (`experiments/E2_online_4zones/loop.py`:
uniform draw over the non-terminal states at every reset, dedicated
`start_state` stream).

**Correction (user decision).** At every reset, both loops now draw the
start state uniformly among the non-terminal states
(`loop.non_terminal_states` = E2's `NON_TERMINAL_STATES`: every state but
the goal) with `loop.draw_start_state`, the same draw as E2's
`_draw_start`.
- The stream is dedicated. `run_mini` takes it from E2's own scheme by
  import: `experiments/E2_online_4zones/rng.py::spawn_streams(seed)["start_state"]`,
  derived by `SeedSequence(seed).spawn`.
- `start_rng` is a required argument of `collect_fixed_policy_data`, so
  no caller can fall back to the fixed start.
- New config block `collection: {start_distribution: uniform_non_terminal,
  epsilon: 1.0}` in mini.yaml and mini_long.yaml. It is validated by
  run_mini (the only supported values) and recorded in meta.yaml.
- The start sequence is saved in `arrays.npz` (`episode_starts`).
Tests:
- a chi-square test of uniformity over the non-terminal states, for both
  loops (40,000 and 30,000 steps), with no start on the goal;
- the start sequence is identical whatever the measurement frequency, and
  equal to the uniform draws of E2's `start_state` stream
  (`tests/e3/test_e3_run_outputs.py`);
- the comparison with E2 below.

**What the earlier results depend on.** The mini run's conclusions on the
mechanics (template accuracy, z-only probe, I(z;s), D2) and on the prior
(KL, L1, frequent outcome, conveyor I_z) do not depend on the start
distribution. Their argument is the separation of the posterior and the
learning of the transition kernel, not the visitation pattern. The
numbers themselves would differ under uniform starts.

**Exposure of the canonical-probe regime** (h0 with the render of an
arbitrary state; mini.yaml, seed 0, 2000 x 8 sampled windows of 16 steps):
- fixed start: ~6% of training steps (window starts only; episode starts
  always showed state 0, and h0 overall was 10.8%);
- uniform starts: **8.9%** (~1 in 11.3), and every h0 step now shows an
  arbitrary state.
The total h0 share falls because the episodes get longer. With uniform
starts, 148 episodes in 5,256 steps (mean ~35.5, almost all truncated at
36); with the fixed start, 253 episodes (mean ~21). The figure first
proposed for this entry (~10.8%) was the fixed-start total.

**Conformance with E2** (`tests/e3/test_e3_protocol_conformance.py`),
against E2 as amended by this log and SPEC_STATUS.md. Every E2 value is
read from E2's code (`sweep.py`, `pilot.py`, `env.py`), never copied.
Passing:
- start support and distribution;
- collection epsilon (= E2's epsilon=1 bridge condition, E3-0);
- uniform actions;
- truncation at H = `sweep.H` (truncated, not terminal);
- terminal at the goal only;
- reward vector = E2's `reward_and_terminal`: goal_reward 50 at the goal,
  -1 per step everywhere else. E2's step cost is -1, not 0;
- gamma in E3's config = `sweep.GAMMA`;
- budget and the 16-point grid = `sweep.BUDGET` / `sweep.CHECKPOINTS`;
- the start_state stream = E2's spawned stream.
Strict xfail, with the reason:
- closed-loop epsilon and gamma consumed by the agent (Phase 5);
- model checkpoints on the grid (Phase 4a, B5);
- **independence of the collection, dynamics and theme streams**, a
  deviation reported below.

**Other deviations found, reported, not corrected (user to decide):**
1. **Correlated streams (spec 6).** The collection actions
   (`default_rng(seed)`), the environment dynamics
   (`TorusZonedEnv(seed=seed)` -> `default_rng(seed)`) and the themes
   (`default_rng(seed)`) are seeded with the same integer: identical bit
   streams, the pattern E2's rng.py calls a real bug found in E1. E2
   derives them by `SeedSequence(seed).spawn`.
2. **Metadata (spec 10).** meta.yaml records the seed, the config and
   the commit, but not the permutation (quadrant assignment), the themes
   or the block class A/B. The class is in results.json and the report,
   not in meta.yaml.
3. **Episode parameters not read from the config.** run_mini hard-codes
   goal_reward=50 and relies on `build_pixel_env`'s defaults for the step
   reward (-1) and H (36), instead of reading
   `experiments/E3_worldmodel_4zones/config.yaml` (`episode`). The values
   agree today.
4. **MC sample count (spec 5.2).** The default M = 4096; the configs use
   `iz_mc_samples: 2000`. This log has no entry for it.
5. **Spec 10 outputs** not produced yet: `transitions.parquet`, model
   weights per checkpoint (the latter is B5).

**Addendum to the h0 note (user, 2026-09-25).** The share of training
steps in the h0 regime depends on seq_len: a window start is 1 step in
16 in the mini configurations, and would be 1 in 64 in the spec's.
- The measurement has to be made again with the full configuration in
  Phase 4a.
- If the canonical-vs-replay accuracy gap closes on the long mini run,
  that does not carry over to the full configuration.

**The fixed start was the goal itself.** The goal is state 0, the cell
(0, 0) (`env.goal_state() == 0`), so every episode of the earlier runs
began ON the terminal state:
- the first step always left the goal;
- an episode ended by returning to it or at H = 36;
- hence the short episodes (mean ~21 steps against ~35.5 with uniform
  starts);
- the "start cell" of the h0 analysis above is the goal cell.
This does not change the statement on the mini run's conclusions (their
argument is the separation of the posterior and the learning of the
kernel). The visitation of the earlier runs, however, was concentrated
around the goal. The truncated mini_long run after the correction:
570 starts over 140 of the 143 non-terminal states, none on the goal.

## 2026-09-25 -- independent random streams (SeedSequence.spawn), metadata, episode parameters, Monte-Carlo counts, reward convention

**1. Streams: deviation 1 corrected (user decision, blocking).**
`experiments/E3_worldmodel_4zones/rng_streams.py` derives every stream of
a run by `np.random.SeedSequence(seed).spawn`, as E2 does. No component
receives the integer seed any more.
- The first six children are **E2's own streams**, in E2's order,
  imported from `experiments/E2_online_4zones/rng.py::STREAM_NAMES`:
  env_dynamics, explore, tiebreak, start_state, matched_control,
  reference. For the same seed they are the very generators of E2's
  `spawn_streams(seed)` (tested).
- E3 appends theme, network_init, replay_sample, latent_sample,
  switch_pairs and measure.
- Use:
  - collection actions = explore;
  - dynamics = `seed=int(env_dynamics.integers(2**31))`, exactly as E2's
    sweep.py / pilot.py;
  - episode starts = start_state;
  - themes = theme;
  - `torch.manual_seed` from network_init, just before the model is built;
  - training minibatches = replay_sample;
  - the training step's latent draws = a torch.Generator from
    latent_sample;
  - the switch statistic = switch_pairs;
  - each measurement = its own child of measure (32 slots).
- Tests (`tests/e3/test_e3_protocol_conformance.py`): the independence
  test is now a normal test (it was a strict xfail). New tests: the first
  draws of the action and dynamics streams differ for the same seed; E3's
  first streams equal E2's; the dynamics seed is drawn as E2 draws it.

**Every E3 run before this entry had correlated streams.** The collection
actions (`default_rng(seed)`), the environment dynamics
(`TorusZonedEnv(seed=seed)` -> `default_rng(seed)`) and the themes
(`default_rng(seed)`) were identical bit streams. That covers the mini
run of 2026-09-24 and its re-runs, and the truncated test runs. The
action draws and the transition noise shared one underlying sequence.
**The noisy-zone statistics of those runs (frequent outcome, kernel L1 in
the noisy zones, I_z of the noisy zones) are not reliable.** The
isolation test (ae_isolation.py) has no environment dynamics and no
actions; its themes came from `default_rng(seed)`, the same integer that
seeded its batch draws.

**Consequences.**
- Themes and trajectories change for every seed: runs made from now on
  are not comparable seed-for-seed with the earlier ones.
- Re-running `ae_isolation.py` now draws different themes (it builds its
  images with `run_mini.build_run`), so the recorded results in
  ESTIMATOR_COMPARISON.md are not bit-reproducible by a re-run. They stay
  the record of what was run.

**2. Metadata (deviation 2).** meta.yaml now records:
- the rotation and the quadrant assignment (the permutation);
- the block class A/B;
- the theme colours (zones, agent marker, corridor, goal);
- the names of the random streams;
- the episode parameters.

**3. Episode parameters (deviation 3).** `run_mini.build_run` reads
goal_reward, step_reward and max_steps (H) from
`experiments/E3_worldmodel_4zones/config.yaml` (`episode`). There are no
hard-coded values any more. The test compares them, and gamma, with E2:
`sweep.GOAL_REWARD`, the default step_reward of E2's `build_env`,
`sweep.H` and `sweep.GAMMA`.

**4. Monte-Carlo sample counts (deviation 4).**
- M = 2000 for the diagnostics during training
  (`measure.iz_mc_samples_training`).
- M = 4096, the spec's value, for the final measures
  (`measure.iz_mc_samples_final`): the final canonical I_z, I(z;s), the
  context dispersion, D1 and the failure diagnostic. Also for the
  calibration (Phase 4).
- Applied in mini.yaml and mini_long.yaml; each diagnostics record
  carries its M.

**5. Reward convention of E2, exactly.** The reward is given on ARRIVAL
in s': `TorusZonedEnv.step` computes `reward = reward_fn(s_next)`
(`src/helplessness/envs/torus_env.py:115`). With `GoalReward`, arriving
at the goal gives goal_reward **in place of** the step reward, not added
to it: `src/helplessness/envs/rewards/base.py:52-53` (`return
self.goal_reward if int(s) == self.goal_state else self.step_reward`),
docstring lines 34-38. The same convention is in
`experiments/E2_online_4zones/env.py::reward_and_terminal`
(`R = full(n_states, step_reward); R[goal_s] = goal_reward`), with E2's
`build_env(..., step_reward=-1.0)` default and `sweep.GOAL_REWARD = 50`.
So:
- the step that reaches the goal is rewarded **+50** (not 49 = 50 - 1);
- every other step is rewarded **-1**.
**Amendment of spec 4.1 (goal head), user decision.** r = 50·p_goal is
replaced by **r = 50·p_goal - 1·(1 - p_goal)**. It gives 50 when
p_goal = 1 and -1 when p_goal = 0, i.e. exactly E2's convention. Not
implemented yet (Phase 5).

## 2026-09-25 -- theme sensitivity: pinned test themes, switch step as a covariate, longer deadline, prompt file, reward decided

**Why.** Once the themes come from the theme stream, the isolation test
(seed 0, lr 1e-3) switches at step 4901 instead of 2301. With the new
colours of seed 0 it misses its 3600-step deadline, and 500 steps after
the switch its accuracy is 0.786. The time the probabilities phase needs
to separate the states depends strongly on the colour draw.

**1. The isolation test's themes are pinned (user decision, option 2).**
`tests/e3/test_e3_ae_isolation.py` passes a fixed `Themes` object, colours
written out with their provenance: the draw the test was validated with,
`build_themes(ZONE_CATEGORIES, goal_state=0, rng=np.random.default_rng(0),
min_dist=30, marker_min_dist=30)`. The test no longer depends on the
theme stream. Criterion and deadlines are unchanged, and it passes.
`run_mini.build_run` and `ae_isolation.run` accept an optional `themes`
for this.

**2. The switch step is a covariate.** Every run must record its
effective switch step, and every analysis must treat it as a covariate.
This matters most in E4/E5, which manipulate the colours on purpose.
Checked: the switch step is already recorded in
- meta.yaml (`switch_step`, `switch_deadline`, `switch_margin`, written at
  the end of a seed and on a guard failure);
- results.json (`training_procedure.switch_step`, `switch_margin`);
- the console and the report.
The colour dependence is characterised in `docs/e3/THEME_SENSITIVITY.md`
(10 theme draws in the isolation test).

**3. Switch deadlines.**
- mini_long.yaml: 25,000 steps (62.5% of its 40,000-step budget; it was
  10,000).
- mini.yaml: to be set from the distribution measured in
  THEME_SENSITIVITY.md (at least the maximum observed x a justified
  margin, at most 80% of its 10,512-step budget); a later entry records
  it. mini.yaml is not being run meanwhile.

**4. The prompt is in the repository.** `docs/e3/prompt_e3_worldmodel.md`,
verbatim as provided. Checked: sections §0 to §11 and all 23
subsections are present. Word by word, it matches the text of the first
message of the E3 session; the only differences are the title line and
the list numbering, so nothing is truncated. SPEC_STATUS.md points to
it.

**5. Reward convention decided.** E2's convention is confirmed: +50 on
arrival at the goal (replacing the step cost) and -1 for every other
step. E3 aligns with it: the environment already does, and the goal head
will use r = 50·p_goal - 1·(1 - p_goal) (Phase 5). The spec 4.1
amendment is marked "decided" in SPEC_STATUS.md.

## 2026-09-25 -- rule for mini.yaml's switch deadline, fixed before the theme-sensitivity results

Set by the user before any result of the characterisation
(THEME_SENSITIVITY.md: 10 colour draws in the isolation test, deadline
8,000, censored draws reported as "> 8,000").

1. **If no draw is censored:**
   - mini.yaml's deadline = (maximum switch step observed in the
     autoencoder) x (RSSM / autoencoder factor) x 1.25, capped at 80% of
     mini's budget (0.8 x 10,512 = 8,409).
   - The RSSM / autoencoder factor is the LARGER of the two available
     pairs:
     - seed 0, old themes: RSSM 4,801 / AE 2,301;
     - seed 0, new themes: RSSM switch step of mini_long / AE 4,901.
   - If the deadline exceeds the cap, **mini's budget is increased**; the
     margin is never reduced.
2. **If at least one draw is censored:**
   - mini.yaml's deadline is set to the cap (8,409), with no other
     adjustment.
   - The case is treated as a **robustness problem of the start-up
     procedure**, not as a deadline setting.
   - The report gives the correlations with the colour statistics,
     keeping the agent-theme contrast apart from the distances between
     themes.
   - No remedy is proposed before the user has read the report. If a
     constraint is considered, it would bear on the agent-theme contrast,
     never on the distances between themes (which E4/E5 manipulate).
3. **In every case,** the report adds mini_long's switch step (when it
   is available) and the RSSM / autoencoder factor it implies.

## 2026-09-28 -- first GPU run (E3-0): main measure in real contexts, recomputation, themes closed, B4-B7, budget

Change of plan (user): no re-run of mini_long, no Colab test. The first
GPU run is E3-0 (epsilon = 1 data) in the environment and in its reference,
on the lab A100, launched by the supervisor. Scientific decisions 1-5 below
are the user's; the implementation choices under them and the engineering
list are the implementer's.

### Scientific decisions (user)

**1. Main measure: real contexts (spec 5.1 amended).** The canonical probe
(learned initial state h0) is not valid as the main measure: template
accuracy 0.40 from h0 against 0.98 in real contexts on mini_long, and the
agent almost never acts from h0. New main measure:
- for each state s, the contexts (h_t, z_t) of replay windows unrolled
  with their true history where s_t = s, excluding the first 8 steps of
  each window and the is_first steps (the h0 regime); up to 50 contexts per
  state (configurable), drawn with the measurement stream;
- I_z computed in every context (same exact densities), then averaged per
  state; the number of contexts per state is reported and a state with no
  context is "non mesuré";
- KL(posterior || prior) by zone, the s-space kernel (decoding prior
  samples from these contexts), L1 vs P and vs B, p_hat(frequent outcome),
  alpha_app and D2: all computed in this regime.
The canonical probe stays in every report as secondary.

**2. Everything recomputable without re-running.** Model weights at every
checkpoint of E2's grid and at the end (and on failure, as before), the
compact replay (state indices, actions, rewards, flags) and the theme
configuration are saved; a script recomputes every measure from a run
directory.

**3. Theme sensitivity: conclusion by the criterion fixed in advance**
(recorded as is in THEME_SENSITIVITY.md): the variability between
replicates is of the same order as between colours, the correlations are
not interpretable, and the switch step depends mostly on the randomness of
training. The RSSM / autoencoder factor is not stable (2.09 and 0.85). 1
run out of 16 did not separate before 8,000 steps: the start-up procedure
is not fully robust. No further investigation for E3.

**4. mini.yaml retired from current use**; mini_long.yaml is the CPU
reference. No budget increase; mini's deadline rule (2026-09-25 entry) is
closed without object.

**5. GPU deadlines large and absolute.** A run that does not separate is
reported as a failure, never re-run with another seed without it being
recorded.

### How decision 1 is implemented (`probe.py`, `run_mini.real_context_measures`)

- **Windows.** The replay (logical order) is cut into consecutive windows
  of `measure.real_context.window_len` steps (null = the training seq_len:
  16 in mini_long, 64 in gpu_e30), each followed by at least one stored step
  (n_windows = (n_steps - 1) // window_len). Each window is unrolled from
  the learned initial state at its start, exactly as training unrolls a
  window, with hard straight-through posterior samples from the measurement
  generator. Selection uses the labels only (`select_real_contexts`: pure
  function, tested); labels never enter the model.
- **z_t of a context** is the unroll's own posterior sample (the value the
  recurrence carried forward), not the posterior mode used by the canonical
  probe. The one-step predictions from a context are
  p(z_{t+1} | GRU(h_t, z_t, a)) for the 5 actions (imagination_step, as the
  canonical probe).
- **Standard error of a state's mean**: sd over its contexts / sqrt(n) (the
  spread already contains each estimate's Monte-Carlo noise); the context's
  own MC se when n = 1 (`measures/context_summary.py`, tested). Zone means
  are over the measured states, se = sqrt(sum se_s^2) / n_measured. These
  se's are the ones the D1 rule, the orderings and the shape checks use.
- **KL(posterior || prior) in real contexts** is KL(q_{t+1} || p_{t+1}) for
  the transition the data actually took from the context (action a_t) --
  the training loss's pairing -- NaN when step t+1 starts an episode;
  averaged per state, then per zone. **Finding, reported:** the canonical
  "KL by zone" (`kl_post_prior_by_zone`, since Phase 3) compares the
  posterior of the CURRENT state s at h0 with the prior predicted for the
  NEXT state after each action. The two describe different states whenever
  s' != s, so the quantity is large even for a perfect model (with 1%
  unimix, one latent variable that differs costs ~9.6 bits). The "9-18
  bits" of the mini run and the KL shape checks of mini_long were judged on
  it: they were not a measure of the prior's fit. The canonical KL is kept,
  labelled as such, in the secondary section.
- **s-space kernel**: for every measured state and action, 32 prior samples
  (`kernel_draws`), the i-th from the state's (i mod n)-th context, decoded
  from h_{t+1,a} and matched to the 144 templates (the same
  measures.kernel_probe functions).
- **D1**: in every context of s, x_t is replaced by the counterfactual render
  (local or global), the history h_t is kept, z_t is a hard sample of
  q(. | h_t, e(x_cf)); I_swap(s) = mean over the contexts, se = sd / sqrt(n).
  Zone means and their se's from the real-context I_z. Only measured states
  of L enter; a pair with an unmeasured zone is "non mesuré".
- **D2**: for neighbouring cells (s, s2), the mean over i < min(n_s, n_s2)
  of the JS distance between the posteriors of their i-th contexts.
- **Context dispersion (spec 5.5)** is now read from the same contexts: per
  zone, the mean |I_real(s) - I_canonical(s)| and the mean sd of I_z over a
  state's contexts (the old 100-sample version is superseded).
- **The goal (state 0) is never measured in real contexts**, by
  construction: it is terminal, so it is never the current state of a
  stored step (an episode ends on arrival and the next step is a reset).
  Every report lists it as "non mesuré"; the canonical probe still covers it.
- **Template accuracy** gets a fourth labelled number: decode(h_t, z_t) at
  the real contexts themselves.
- **Shape checks** (unchanged rules, fixed 2026-09-25) are judged in real
  contexts, the canonical ones printed as secondary. In the reference
  environment the rules bear on the dynamics: "deterministic zone" and the
  two I_z rules select zones by their dynamics type there, so the
  conveyor-looking zone (controllable-deterministic in the reference) falls
  under "I_z increasing" and "I_z of the conveyor decreasing" has no zone;
  the checklist's conveyor ordering is "non applicable". This is the direct
  application of the rules to the reference, not a new expected shape; it
  is flagged in the report.
- Cost at gpu_e30's size, measured on this CPU with the mini model: one
  training-diagnostic point ~2-3 min (7,200 I_z at M = 2000, 23,040 decodes).

### How decision 2 is implemented

- `replay.npz` (after collection): s int16, a int8, r float32, terminal,
  truncated, is_first, the true s_next of every step (the next stored step
  is a reset state at an episode's end, so N is only recoverable with it),
  N and the episode starts. `world.json`: seed, condition, permutation,
  theme colours, render settings, the dynamics seed and the sha256 of the
  144 pre-rendered images (a rebuild must reproduce it, or the recomputation
  stops).
- `checkpoints/weights_stepNNNNNNNN.pt` at every checkpoint of E2's grid,
  `weights_final.pt`, and `failure/checkpoint.pt` as before. **E3-0 mapping
  of the grid**: all data is collected first, so grid point c (environment
  steps, c <= the run's total) maps to gradient step round(c x replay_ratio)
  -- "the training a run at this ratio would have done after c environment
  steps". gpu_e30: 600, 951, 1,507, 2,389, 3,786, 6,000, 9,509, 15,071,
  23,887, 37,858, and the end (60,000). The training diagnostics are taken
  at these points (`measure.diagnostics_at_checkpoints`).
- `recompute_measures.py <run dir> --checkpoint final|all|failure|<step>`
  runs the same `run_mini.measure_final` with the same measurement streams:
  on the same device and threads it reproduces the run's own final arrays
  bit for bit (tested, and printed by the script).
- The z-only probe's linear classifier was initialised from the global
  torch RNG; it is now initialised from its measurement stream (a forked
  RNG), so a probe gives the same numbers whatever ran before it.
- **mini_long cannot be recomputed**: its weights were not saved. Its
  report stays the record of what was measured then.

### GPU run: `configs/e3/gpu_e30.yaml`, budget and expected duration

Seeds 0 (class A) and 5 (class B), experimental and reference: 4 runs,
one process each, side by side on the A100 (`launch_gpu_e30.sh`). (A fifth, bridge run was added the same day: see the next entry.)
- **Architecture: the spec's** (3.1: base depth 32, h_dim 256; embed_dim
  256, RSSMConfig's default) with the 8x8 latent of mini_long, and the
  spec's batch of 16 x 64 (3.3). The mini sizes were a CPU reduction; the
  capacity sweep comes later. Procedure unchanged from mini_long.
- **Data: 100,000 environment steps** (99,744 + 256 prefill), ~139 visits
  per (s, a) on average (mini_long: 28), enough for C-b's N[s,a] >= 100 on
  most pairs, and a point of E2's grid.
- **Training: 60,000 gradient steps** (`replay_ratio: 0.6` gradient steps
  per environment step = 614 replayed steps per environment step; mini_long
  had 256). **Note (convention):** run_mini's `replay_ratio` has always
  been gradient steps per environment step; the spec (3.5) defines it as
  replayed steps per environment step. Both numbers are given here; the
  code's meaning is unchanged.
- **Switch deadline 40,000** (absolute): more than twice the largest RSSM
  estimate available (6,401 x 2.09 = 13,400 from THEME_SENSITIVITY.md), and
  leaves 20,000 steps after a switch at the deadline.
- **Duration, extrapolated from the CPU profile** (no GPU here):
  - a training step at this size is ~12,000 aten calls (torch.profiler,
    after 7.1) and ~0.32 TFLOP (encoder + decoder, forward and backward,
    1,024 images of 72x72);
  - host side (eager dispatch, ~5-10 us per call): 0.06-0.12 s per step;
    device side (FP32 convolutions, TF32 off, ~6-8 TFLOP/s effective on an
    A100): ~0.05 s; the two overlap, so one process alone: ~0.1-0.15 s per
    step; four processes sharing the GPU: up to ~0.25 s per step each;
  - training: 60,000 x 0.1-0.25 s = 1.7-4.2 h; 10 diagnostic points x
    ~3 min = 0.5 h; final measures (real contexts, z-only probe, D1 in real
    contexts) ~0.3 h; collection ~2 min;
  - **total: ~2.5-5 h per run, the 4 side by side**, under the ~6 h target
    at the pessimistic end.
  `preflight.py` measures the real rates (alone and 4 side by side) and
  prints the projected duration before launch; the budget is not adjusted
  to the machine after the fact.

### Engineering decisions taken alone (audit step B and follow-ups)

- **B4** -- `z_only_probe_chunked`: the buffer is unrolled 256 windows at
  a time without decoding; only z and the entropies are kept. The same
  windows give the same samples in one chunk or several (tested).
- **B5** -- checkpoints as above; `checkpoints/resume.pt` (model, Adam,
  switch state, progress, loss history, curves, diagnostics, every training
  and measurement random state, torch global RNG) written atomically at
  every checkpoint and every `checkpoints.resume_every` steps (2,000),
  `latest.json` pointer; `run_mini.py --resume <run dir>` continues in the
  same directory. Tested: a run crashed after its resume point and resumed
  gives bit-identical losses, switch step, diagnostics and final measures
  to an uninterrupted run (CPU).
- **B6** -- `ReplayBuffer(capacity, table=...)`: a private int16 index into
  the run's 144 pre-rendered images instead of the pixels; `add()` asserts
  x == table[s] bit for bit (`ObservationTableMismatch`); `to_device()` puts
  the table on the GPU and minibatches are gathered there (no image crosses
  the host-device link). Same windows as pixel mode for the same stream
  (tested); pixel mode kept for tests and tools whose images are not
  renders. 100,000 steps: ~2 MB instead of 1.6 GB.
- **B7 -- GPU determinism policy**: `runtime.deterministic: true` sets
  `torch.use_deterministic_algorithms(True, warn_only=True)`,
  cuDNN deterministic, no cuDNN benchmark, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  (set before CUDA starts), TF32 off; all recorded in meta.yaml.
  warn_only: an op without a deterministic CUDA implementation warns in the
  console instead of killing a multi-hour run. The target is bit-exact
  reproduction on the same GPU model and software stack, not across devices;
  `preflight.py` checks it (same 5 steps twice).
- **7.1** (done, the profile shows it is needed): the encoder runs once on
  a window's B x T images before the recurrence and the decoder once after
  it. Same samples for a given generator, values equal up to reduction
  order (tested against the step-by-step version). CPU, 16 x 64: encoder +
  decoder 3.2 s batched against 6.1 s step by step; a full training step
  of the spec model 4.6-4.8 s against 8.8 s (under load). On a GPU the
  step-by-step version adds ~2T small kernel launches per step. **New runs
  are not bit-comparable to mini_long** (float reduction order).
- **7.2** (done): `train_world_model_step(sync=False)` returns device
  tensors; run_mini converts the per-step losses in one transfer every 50
  steps (at the switch evaluations) instead of 8 synchronisations per step.
  Same values.
- uint8 images cross to the device and are converted there (audit 1.8).
- Reference-condition streams: the experimental run's streams derive from
  SeedSequence(seed) (E2's, unchanged); the reference's from
  SeedSequence([seed, 1]), except the themes, always the experimental
  seed's theme stream (spec 6: only the permutation and the themes are
  shared). Run names use `pair-control` for the reference
  (`runs.make_run_name`'s vocabulary).
- NaN values are written as null in the JSON outputs (they were written
  as the non-standard token NaN).
- `preflight.py` (one command, ~10-15 min), `launch_gpu_e30.sh` (4
  processes, nohup), `run_mini.py --combine` (combined report of finished
  runs); README updated for the A100.

## 2026-09-28 -- bridge run for the first GPU run; test count

**1. Bridge run (user decision).** gpu_e30's 4 runs keep the spec
architecture. A fifth run is added: seed 0, experimental environment,
with exactly mini_long.yaml's `model:` block
(`configs/e3/gpu_e30_bridge.yaml`). **Why:** the start-up procedure
(probabilities phase, label-free switch, lr 1e-3 -> 1e-4 at the switch)
has only been validated on the small network (every isolation test, the
mini run, mini_long); the bridge relates the GPU results to the CPU
reference. It differs from mini_long.yaml only by:
- the budget: 100,000 environment steps and 60,000 gradient steps
  (`replay_ratio` 0.6), as gpu_e30. Same seed, same condition, same
  streams: its data are exactly those of gpu_e30's seed 0 / experimental
  run, so the two differ only by the network (and mini_long's batch of
  8 x 16 and its switch deadline of 25,000);
- the device and runtime (cuda, 4 threads, the GPU determinism policy);
- the outputs: measured at the checkpoints of E2's grid instead of every
  2,000 steps, console line every 1,000 steps, z-only windows of 64 steps.
`tests/e3/test_e3_gpu_configs.py` checks that nothing else differs.
`launch_gpu_e30.sh` starts 5 processes; `preflight.py --extra-config
configs/e3/gpu_e30_bridge.yaml` measures the throughput with the 5
processes side by side (4 gpu_e30, 1 bridge) and projects the duration of
each config.

**2. Test count (716 -> 327 -> 733).** No test was removed or moved: every
test id collected at commit d748416 is still collected (checked by
comparing the two collections). 716 was the full suite (`tests/`: 721
collected = 716 passed + 2 skipped + 3 xfailed). 327 was a subset
(`tests/agents/worldmodel tests/e3 tests/measures`), reported without
saying so. Full suite after commit b52902e: 738 collected = 733 passed + 2
skipped + 3 xfailed; the +17 are the new tests of that commit
(test_replay_table 8, test_unroll_batched 1, test_e3_real_context 6,
test_context_summary 2). With test_e3_gpu_configs (2): 740. Reports now
give the full-suite count and name any subset.

## 2026-09-28 -- optional W&B tracking; light results archive

**W&B (user request; implementation choices by the implementer).**
`experiments/E3_worldmodel_4zones/wandb_logger.py`, extra `wandb` in
pyproject.toml, config block `logging.wandb` (enabled: auto | true | false,
project, entity, mode), options `--wandb` and `--wandb-artifact`.
- On if WANDB_API_KEY is set (enabled: auto), or with --wandb; WANDB_MODE
  overrides the mode (offline runs are uploaded later with `wandb sync`).
- W&B run name = run directory name; W&B config = resolved config +
  meta.yaml; the W&B run id is written to meta.yaml, and a resumed run
  continues the same W&B run.
- Logged only where run_mini.py already synchronises, no new
  synchronisation: the console line every `measure.monitor_every` steps
  (1,000 in the GPU configs: losses, gradient norm, lr, beta_rep, phase,
  batch-pair and labelled JS), the switch step, every diagnostic point (the
  fields of diagnostics.jsonl, flattened), and at the end the scalars of
  results.json (as the run summary), the figures and REPORT.md. The x axis
  is `grad_step`, not W&B's own step, so a resumed run can log again from
  its resume point.
- `--wandb-artifact`: the run directory is uploaded as an artifact at the
  end of the run (also after a guard failure), weights included,
  checkpoints/resume.pt and the local wandb/ directory excluded.
  launch_gpu_e30.sh passes it to the 5 processes, with WANDB_API_KEY and
  WANDB_MODE; it has no effect when W&B is off.
- Robustness: a failure of the import, the initialisation or any later call
  prints one warning in the console and turns the tracking off for the rest
  of the run; a run never stops because of W&B.
- Tests (`tests/e3/test_e3_wandb.py`): losses, lr, beta_rep and switch step
  bit-identical with and without W&B (real wandb, offline mode); a failing
  wandb at init or at log only warns and the run finishes; the artifact
  holds every file of the run directory except resume.pt and wandb/; off by
  default without a key.

**Results archive (RUN_GPU.md, section 4).** First a light archive, sent
right away: the run directories without checkpoints/ and failure/*.pt,
plus the logs. Measured on the smoke run of gpu_e30 (20 gradient steps):
10.0 MB for one run. A full run adds its 60,000-step history to
results.json (~5 MB compressed, upper estimate): ~15 MB per run, ~75 MB for
the 5 runs. The full archive (with the weights, ~2 GB) only on request.
