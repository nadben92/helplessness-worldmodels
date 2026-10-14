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

## 2026-09-29 -- first GPU batch: outcomes as they stand; bridge measures recovered; K_s corrected; engineering for the next batch

### 1. Outcomes of TrainJob m3e9461fad23 (recorded as they are, nothing relaunched)

One A100 80 GB shared by the 5 processes (preflight bc7e34cec174 had passed; it tested throughput, not the memory of concurrent measurements).

| run | outcome |
|---|---|
| seed 0 reference (gpu_e30, class A) | **done**, 60,000 / 60,000 gradient steps; switch at 7,051 |
| seed 0 experimental (gpu_e30) | **separation not reached** by the 40,000 deadline (SeparationNotReachedError); batch-pair JS never above 0.003 bits |
| seed 5 reference (gpu_e30, class B) | **separation not reached** by 40,000; batch-pair JS never above 0.24 bits |
| seed 5 experimental (gpu_e30) | **CUDA out of memory** at step 600, first real-context diagnostic (encoder LayerNorm, 2.57 GiB request) |
| bridge seed 0 experimental (gpu_e30_bridge) | training **done** 60,000 / 60,000 (switch at 4,201); **CUDA out of memory** in the final z-only probe (972 MiB request: the float images of one 256-window chunk) |

One success, two separation failures, two out-of-memory errors. Seed 5 experimental's step-600
weights are not a resume point: its outcome stays "out of memory at 600", and any later run of it is a
new run, recorded separately. All five meta.yaml have git_commit "unknown" (no git in the container).

### 2. Bridge: final measures recovered on CPU

`recompute_measures.py <bridge> --checkpoint final --device cpu` (4 threads), then `--checkpoint all`
for the trajectory (section 7). There is no run-side arrays.npz to compare with, and a CPU
recomputation is not bit-identical to the A100. Outputs under `<run>/recomputed/` (not versioned).

- **C-a** (>= 0.99): 0.9941 on the designated measure (replay windows, one batch of 32 x 16 = 512
  frames, binomial se 0.0034); on the 6,806 real contexts 0.9875 at K_s = 32 and 0.9904 at K_s = 1,358
  (the contexts' posterior draws come from the same stream, after the canonical kernel). At the threshold.
- **C-c** (>= 0.99): 0.9947 (z-only probe, 19,993 held-out samples). Met.
- **C-b** (per zone, pairs with N[s,a] >= 100, L1_WM <= L1_B + 0.05), at K_s = 1,358 (section 3), with
  the exact-kernel floor: uncontrollable deterministic 0.044 vs 0.061 (floor 0, 24 pairs) met;
  controllable deterministic 0.045 vs 0.059 (floor 0, 93 pairs) met; controllable noisy 0.265 vs 0.104
  (floor 0.020, 94 pairs) **not met**; uncontrollable noisy 0.191 vs 0.107 (floor 0.020, 27 pairs)
  **not met**. Same verdicts at K_s = 32 (0.029 / 0.050 / 0.289 / 0.202; floor 0.13 in the noisy zones).
- I_z in real contexts (mean [se]): uncontrollable deterministic 0.287 [0.006], controllable noisy
  2.069 [0.002], uncontrollable noisy 0.285 [0.006], controllable deterministic 2.257 [0.002]; s-space
  truth 0, 1.653, 0, 2.322.
- T4 (p_hat at the true frequent outcome, real-context kernel, mean over the zone's pairs): controllable
  noisy 0.789 (se over states 0.030), uncontrollable noisy 0.634 (0.051); target 0.9; twin B 0.894 / 0.877.
- Ghosts 0.002 (0.001-0.004 by zone). Context dispersion (mean |I_real - I_canonical| / mean sd over a
  state's contexts), zone order as above: 0.245/0.176, 0.139/0.084, 0.247/0.187, 0.152/0.061.
  D2 (real contexts): boundary 4.294, within-zone 2.907 bits.

### 3. C-b's K_s: corrected after seeing the bridge fail, for every configuration

With K draws per (s, a), even a model whose kernel IS the true P scores an expected L1 above zero: the
empirical law of K draws differs from P. Exactly (the expectation is linear: a sum of binomial mean
absolute deviations, `measures.kernel_probe.expected_l1_sampling_floor`), in a 0.9 / 4 x 0.025 zone the
floor is 0.131 at K = 32, above C-b's bound L1_B + 0.05 (about 0.105-0.11 with this data): C-b could
not be met in the noisy zones by any model, the seed 0 reference included (0.167 / 0.183, within 0.05
of the floor).

**Decision (user):** correct the measurement's precision, not the threshold. K_s is the smallest K for
which the floor is <= 0.02 in the noisiest zone, at K and at every K' up to 2K
(`measures.kernel_probe.draws_for_l1_floor`): **K_s = 1,358** (1,357 is above 0.02). Set as
`measure.kernel_k_s` and `measure.real_context.kernel_draws` in **every** E3 config (mini, mini_long,
gpu_e30, gpu_e30_bridge). The floor is reported next to every L1 (`l1_floor_by_zone` in the records, a
column in REPORT.md and in the recomputation's SUMMARY.md). `recompute_measures.py --kernel-draws K`
measures an older run at the corrected K_s (output labelled `__ks<K>`).

**This correction was made after seeing the bridge fail C-b.** It is justified independently of that
result: the floor is a property of the estimator and of P alone, computed without any model, and it
applies to every run and configuration alike. At K_s = 1,358 the bridge still fails C-b in the noisy
zones, by 8-13 times the floor. Cost: a kernel is 144 x 5 x 1,358 decodes (about 12 min of decoding
for the bridge network on 4 CPU threads; each diagnostic point has two kernels, canonical and real).

### 4. Seed 0 reference: T4 and I_z against the s-space value

In the reference both noisy zones are controllable noisy (0.9 / 4 x 0.025): I(a;s'|s) = 1.653 bits.
- **T4**: 0.9225 (se 0.0051) in the controllable-noisy zone, 0.9011 (0.0075) in the zone that is
  uncontrollable noisy in the experimental environment; twin B 0.898 / 0.893. One seed: no interval
  over seeds.
- **I_z (z-space, real contexts): 2.142 [0.004] and 1.965 [0.005], i.e. +0.49 and +0.31 bits above
  1.653.** Deterministic zones: 2.225 and 2.316 against log2 5 = 2.322.
- The run's own s-space kernel (K_s = 32): I_s 1.970 / 1.895, against 1.749 expected from 32 exact draws
  of P: +0.22 / +0.15 bits beyond the sampling bias. **The model overestimates the controllability of
  the noisy zones**, while its probability of the frequent outcome is right (T4).
- The reference's weights are not in this checkout (the light archive omits every .pt): it could not be
  measured at K_s = 1,358, nor probed as in section 5.

### 5. Hypothesis (user): the predicted code carries the action

I_z measures the action dependence of the predicted latent code; the code of one arrival state could
encode, through h_{t+1,a}, the action that led there -- inflating I_z without changing the belief about
the state (and possibly the conveyor's ~0.4-bit plateau in mini_long; I_z being capped at log2 5, it
would not cancel in b). Tested on the bridge only (weights available), final checkpoint. **The main
measure is unchanged; the decision is the user's.**

a. I_s by zone from the real-context kernel at K_s = 1,358 (sampling bias <= 0.004 bits), against I_z
and the truth: uncontrollable deterministic I_s 0.071 / I_z 0.287 / truth 0; controllable noisy
1.706 / 2.069 / 1.653; uncontrollable noisy 0.069 / 0.285 / 0; controllable deterministic 2.128 /
2.257 / 2.322. In the controllable-noisy zone the s-space kernel is within +0.05 bits of the truth
while I_z is +0.42; in the uncontrollable zones I_s is 0.07, I_z 0.29.

b. `action_in_latent.py` (new): from the 6,806 real contexts, 4 prior draws of z' per context and
action, each decoded to s'; at fixed s' (groups >= 50 draws, 80/20 split), a linear probe z' -> a. Test
accuracy, weighted over the s' groups (about 4,100-4,900 test draws per zone, binomial se about 0.007):

| source zone | chance | majority given s' | (s, s') only | z' given s' | z' + s given s' |
|---|---|---|---|---|---|
| uncontrollable deterministic | 0.200 | 0.221 | 0.222 | 0.258 | 0.265 |
| controllable noisy | 0.200 | 0.350 | 0.860 | 0.401 | 0.869 |
| uncontrollable noisy | 0.200 | 0.222 | 0.223 | 0.251 | 0.260 |
| controllable deterministic | 0.200 | 0.331 | 0.964 | 0.376 | 0.972 |

In the uncontrollable zones, where the states tell nothing about the action, z' predicts it 0.03-0.04
above what (s, s') allow (about 5 se); in the controllable zones z' adds 0.008-0.009 to (s, s').

### 6. Engineering for the next batch (implementer's decisions)

- **Probes by blocks.** `loop.encode_windows(rssm, x, device, max_images)`: the real-context and z-only
  probes convert and encode their block of windows at most `measure.max_images_per_forward` images at
  a time (2,048 in both GPU configs), then unroll from the embeddings (`unroll_observation(...,
  embeds=)`). The latent draws, taken step by step over the whole block, are unchanged; embeddings
  agree to ~1e-6 (tested: same drawn classes, probabilities within 1e-6). null (every older config)
  keeps the one-call path bit for bit.
- **Peak GPU memory** of every diagnostic point (`diagnostics.jsonl`: gpu_memory) and of the final
  measures (results.json: gpu_memory_final), allocated and reserved; the cache is then returned to the
  device (`torch.cuda.empty_cache`): the spec-size processes held 29 GiB each after their measurements.
- **Preflight memory test** with the planned number of processes: each trains the config's model a few
  steps, waits at a file barrier for the others, then takes one diagnostic point and one final
  measurement at the same time; fails on any process failure (OOM) or if the processes' worst reserved
  peaks add up to more than 90% of the GPU. Not yet run on a GPU (none here).
- **Launcher**: `submit_kubeflow.py --runs <subset> --max-parallel N` (a queue: at most N at once, the
  next starts when one ends; status.json per run, supervisor.json); its preflight phase with the same
  flags builds `preflight.py --parallel N` with the worst mix (spec-size runs first).
  `launch_gpu_e30.sh` delegates to the same supervisor (RUNS, MAX_PARALLEL).
- **Code version**: `runs.code_provenance` records git_commit when git works, and always
  `source_sha256` (contents of src/, experiments/, configs/, pyproject.toml; CRLF read as LF; caches
  excluded); `run_one_seed` refuses to start with neither (`UnknownCodeVersionError`).

### 7. Separation failures (description only: no correction, no hypothesis tested by a run)

From diagnostics.jsonl, failure/diagnostic.json, the console lines (every 1,000 steps) and the
batch-pair JS history (every 50 steps; the bridge's from its resume.pt).

| | seed 0 exp. (spec) | seed 5 ref. (spec) | seed 0 ref. (spec) | bridge seed 0 exp. |
|---|---|---|---|---|
| outcome | no switch by 40,000 | no switch by 40,000 | switch 7,051 | switch 4,201 |
| batch-pair JS first > 0.1 / 1 / 3 bits | never | 5,650 / never / never | 5,500 / 5,650 / 6,250 | 2,600 / 2,850 / 3,350 |
| JS max before switch / deadline | 0.003 (5,400) | 0.236 (5,650) | 4.81 | 4.12 |
| JS median, 10,000-40,000 | 1e-7 | 2e-9 | 6.1 (after switch) | 6.0 (after switch) |
| posterior entropy, canonical (bits, ceiling 24), at 600 / 3,786 / 6,000 / 9,509 / 37,858 | 23.54 / 23.84 / 23.89 / 23.99 / 23.99 | 23.26 / 23.94 / 24.00 / 24.00 / 23.99 | 23.60 / 23.94 / 17.17 / 3.43 / 1.18 | 20.93 / 17.24 / 9.20 / 5.85 / 3.74 |
| recon_patch at 1k / 5k / 10k / 39k | 1.76 / 1.40 / 1.00 / 0.51 | 2.77 / 3.17 / 2.81 / 1.05 | 1.66 / 1.74 / 0.089 / 0.007 | 2.22 / 0.35 / 0.25 / 0.047 |
| grad norm, median (max) over 1k-39k | 44 (250) | 50 (199) | 13 (147) | 19 (160) |
| real-context template accuracy at 37,858 | 0.46 | 0.57 | 1.00 | 0.98 |
| failure diagnostic (canonical) | acc probs/mode/hard 0.007/0.007/0.010; I(z;s) -0.001; centred logit norm 0.10 | 0.007/0.007/0.007; -0.001; 0.12 | - | - |

What distinguishes the runs that separate:
- All four start alike: batch-pair JS about 1e-7 and a posterior entropy rising from step 600 toward
  the 24-bit ceiling.
- Each shows one window in which the JS rises (bridge 2,400-3,350; the three spec runs 3,500-7,000). In
  the seed 0 reference it goes from 1e-3 to 3 bits in 1,000 steps (5,250 -> 6,250) and the posterior
  entropy falls (23.94 at 3,786, 17.2 at 6,000). In the two failures the rise stops at 0.003 and 0.24
  bits and reverts: the JS is back near 1e-7 by 10,000 and the posterior reaches maximal entropy
  (23.99-24.00 bits, centred logit norm 0.10-0.12), where it stays to 40,000.
- Before that window, recon_patch and gradient norms do not separate successes from failures (seed 0
  reference vs experimental: 1.66-1.74 vs 1.40-1.76; gradient norms 100-250 in every run). After it,
  recon_patch falls by an order of magnitude with the separation in the successes and declines slowly
  in the failures, whose gradient norms stay 2-4 times higher.
- In the failures the reconstruction still improves and the real-context template accuracy reaches
  0.46 and 0.57 with a posterior at maximal entropy: state information reaches the decoder through h,
  not z (observation).
- With the same data (seed 0 experimental), the bridge network's window comes earlier (first JS > 0.1
  at 2,600 vs never) and its posterior is further from uniform at step 600 (20.9 vs 23.5 bits).
- Spec architecture: 1 separation in 3 runs; the bridge network: 1 in 1.

Bridge trajectory (recomputed at every checkpoint, K_s = 32; the final point recomputed with the new
code is bit-identical to the one recomputed before the changes of section 6):

| step | I_z real ud / cn / un / cd | acc replay / real | z-only | post. entropy real | L1(WM, P) real ud / cn / un / cd | T4 preview cn / un |
|---|---|---|---|---|---|---|
| 600 | 0.00 / 0.00 / 0.00 / 0.00 | 0.016 / 0.008 | 0.025 | 21.08 | 1.97 / 1.99 / 1.97 / 1.99 | 0.14 / 0.15 |
| 2,389 | 0.00 / 0.00 / 0.00 / 0.00 | 0.016 / 0.010 | 0.028 | 22.68 | 1.95 / 1.99 / 1.97 / 1.99 | 0.21 / 0.21 |
| 3,786 | 0.06 / 0.06 / 0.06 / 0.09 | 0.506 / 0.349 | 0.637 | 16.97 | 1.97 / 1.78 / 1.93 / 1.92 | 0.14 / 0.13 |
| 6,000 | 0.15 / 0.28 / 0.25 / 0.52 | 0.896 / 0.673 | 0.921 | 8.82 | 1.86 / 1.81 / 1.84 / 1.84 | 0.21 / 0.24 |
| 9,509 | 0.29 / 0.76 / 0.51 / 1.19 | 0.965 / 0.796 | 0.957 | 5.67 | 1.70 / 1.73 / 1.73 / 1.61 | 0.21 / 0.29 |
| 15,071 | 0.40 / 1.23 / 0.59 / 1.67 | 0.969 / 0.890 | 0.973 | 4.57 | 1.49 / 1.49 / 1.48 / 1.17 | 0.30 / 0.35 |
| 23,887 | 0.43 / 1.51 / 0.51 / 2.01 | 0.982 / 0.957 | 0.983 | 3.96 | 1.23 / 1.04 / 1.20 / 0.62 | 0.46 / 0.44 |
| 37,858 | 0.37 / 1.87 / 0.37 / 2.18 | 0.994 / 0.980 | 0.991 | 3.40 | 0.97 / 0.65 / 0.97 / 0.30 | 0.66 / 0.52 |
| 60,000 | 0.28 / 2.06 / 0.30 / 2.26 | 0.994 / 0.988 | 0.995 | 2.94 | 0.71 / 0.39 / 0.71 / 0.15 | 0.79 / 0.64 |

(ud / cn / un / cd: uncontrollable deterministic, controllable noisy, uncontrollable noisy,
controllable deterministic; all states of each zone, before the N >= 100 restriction.) At 60,000 the
L1 errors and T4 are still moving: the bridge's kernel has not converged. The uncontrollable zones'
I_z peaks at 0.43 / 0.59 (15,000-24,000) and is still falling at the end (0.28 / 0.30).

## 2026-09-30 -- I_s becomes the main measure; second GPU batch gpu_e30b (objectives fixed before the launch); b aggregation

### 1. Main measure: I_s; I_z secondary (user decision)

**I_s** -- the controllability I(a ; s' | s) of the world model's s-space kernel in real contexts
(prior draws of z' from a state's contexts, decoded to their arrival state and matched to the
templates) -- replaces I_z as the main measure. I_z stays reported, as secondary: an upper bound.

**Why.** On the bridge (2026-09-29, section 5), I_z / I_s / truth are 2.07 / 1.71 / 1.65 bits in the
controllable-noisy zone and 0.29 / 0.07 / 0 in the uncontrollable zones: I_z overestimates the
controllability. The predicted code carries information on the action beyond the arrival state
(action_in_latent.py: at fixed s', z' predicts the action 0.03-0.04 above what (s, s') allow in the
uncontrollable zones, about 5 se), and this bias would compress b (I_z is capped at log2 5 and the
inflation does not cancel between the environment and its reference). I_s recovers the truth where
it is known (within +0.05 bits in the controllable-noisy zone, 0.07 in the uncontrollable ones).

**Implementation.**
- `run_mini.real_context_measures`: I_s per state = `measures.controllability.controllability_per_state`
  of the real-context kernel (NaN for a state not measured or with an all-ghost action row);
  standard error and plug-in bias by parametric bootstrap
  (`measures.controllability.controllability_bootstrap`, `measure.is_bootstrap` = 200 replicates:
  redraw K outcomes per (s, a) from the estimated kernel); by zone, the mean over measured states
  with the se of the mean and the mean bias (`I_s_by_zone`). The canonical probe gets the same
  (secondary). Records, diagnostics.jsonl, results.json (`I_s_by_category_real`), arrays.npz
  (`real_I_s`, `real_I_s_se`, `real_I_s_bias`), REPORT.md (I_s table first, I_z "SECONDARY, upper
  bound"), the expected-shape checks (the I rules judged on I_s, then on I_z as secondary), the
  checklist and the figures.
- b = I_s(experimental) - I_s(reference) (`aggregate_b.py`, section 4).
- D1 (alpha_app) is still computed in z-space: no s-space version yet.

**Cost and estimator (implementer's proposal).** The plug-in I(a ; s' | s) of a kernel estimated
from K draws per (s, a) is biased upward; for an exact 0.9 / 4 x 0.025 kernel (simulation, 1,000
replicates): 0.047 bits at K = 64, 0.019 at 128, 0.013 at 256, 0.005 at 512, 0.003 at 1,024,
**0.002 at 1,358**. Keeping the bias <= 0.01 per context would take >= 512 draws in each of the 50
contexts of a state (128,000 decodes per state and action, 50 times the present cost). So:
- **the draws of a state's contexts are pooled** into one kernel before I is computed (the existing
  estimator: K draws per (s, a), the i-th from the state's (i mod n)-th context -- about 27 per
  context at K = 1,358 with 50 contexts). This measures the controllability of the mixture of the
  contexts' kernels, which is at most the mean of their controllabilities (mutual information is
  convex in the channel);
- **final measures: 50 contexts per state, K = 1,358 per (s, a)** (the value C-b already needs):
  bias about 0.002-0.004 bits in the noisy zones, 0 in the deterministic ones; 144 x 5 x 1,358 =
  0.98 M decodes and template matches per kernel, two kernels (real contexts and canonical);
- **training diagnostics: K = 256** (`measure.real_context.kernel_draws_training`,
  `measure.kernel_k_s_training`, in every E3 config): bias <= 0.013 bits, L1 floor 0.046; each record
  states the K it used.
- **Gap between pooled and per-context I_s, measured on the bridge** (`is_pooling_check.py`, final
  checkpoint, 4 states per zone, 10 real contexts each, 512 draws per context and action; bits,
  bootstrap plug-in bias in brackets):

  | zone | per-context mean I_s | pooled I_s (same contexts) | gap [se over states] | truth |
  |---|---|---|---|---|
  | uncontrollable deterministic | 0.143 (0.006) | 0.092 (0.001) | +0.051 [0.029] | 0 |
  | controllable noisy | 1.691 (0.009) | 1.626 (0.002) | +0.065 [0.031] | 1.653 |
  | uncontrollable noisy | 0.111 (0.017) | 0.058 (0.003) | +0.053 [0.006] | 0 |
  | controllable deterministic | 1.963 (0.005) | 1.927 (0.001) | +0.036 [0.019] | 2.322 |

  The gap (+0.04 to +0.065 bits) is larger than the 0.01 bias target and is not a sampling bias
  (the pooled bias is <= 0.003; the per-context one 0.005-0.017 at 512 draws): the two are different
  quantities -- the controllability of the state's average predicted kernel (pooled) and the average
  controllability of its context-specific kernels (per context), which also counts how the context
  changes the action's effect. **Kept: pooled** (the main measure, bias <= 0.004, and the nearer to
  the truth where it is known: 1.626 vs 1.653 in the noisy zone, 0.06-0.09 vs 0 in the
  uncontrollable ones). The per-context estimand with bias <= 0.01 in every zone would need about
  1,024 draws per context (the uncontrollable-noisy bias is 0.017 at 512), i.e. 10 contexts x 1,024
  draws = 7.5 times the present final cost; available on request, not implemented as a measure.

### 2. Second GPU batch: gpu_e30b (configs/e3/gpu_e30b_win64.yaml, gpu_e30b_win16.yaml)

The bridge's network and procedure (mini_long's model block, lr 1e-3 -> 1e-4 at the label-free
switch, deadline 25,000), 60,000 gradient steps on 100,000 environment steps (replay_ratio 0.6),
**1,024 images per gradient step** (8 times the bridge) in two shapes:
- **A (win64): 16 windows x 64 steps**; **B (win16): 64 windows x 16 steps**.
Seeds 0 (block class A) and 5 (class B), experimental environment and reference: 8 runs. One CPU
thread per process (and one BLAS/OpenMP thread, set by the Kubeflow worker). The real contexts
are windows of the training length (64 or 16 steps, burn-in 8): the measurement regime follows
the training regime, so it differs between the shapes.

**Objectives, written before the launch:**
- (i) **Does the window length explain the separation failures?** Hypothesis: with long windows the
  state information can travel through h, and z does not need to separate. Read: the switch step
  and the batch-pair JS trajectory of win64 vs win16 (same seeds, same data), with the new h -> s
  and z -> s probes (real contexts, every diagnostic point; full replay at the end).
- (ii) **Do 8 times more images per step than the bridge make the noisy zones' kernel converge?**
  Read: C-b (L1 per zone against L1_B + 0.05, with the floor) and T4 (p_hat of the frequent outcome
  -> 0.9) in the noisy zones, at the end and along the checkpoints.
- (iii) **A first floor b = I_s - I_s,ref by zone**, for the configuration that passes the
  calibration criteria (C-a, C-b, C-c), from `aggregate_b.py`, by block class.
No run of this batch is relaunched on failure without a new entry here; a guard failure is a result.

**Launch (implementer's proposal).** `submit_kubeflow.py --batch gpu_e30b --max-parallel 8`, preflight
first with the same flags: its memory test runs 8 processes (7 win64 + 1 win16, the heaviest mix)
measuring together; if it fails, the same with `--max-parallel 4` (then 4 win64 in the first wave).

### 3. h -> s probe (user request)

`probe.z_only_probe_chunked(latent="h")` (the z-only probe's linear classifier, factored as
`probe.linear_state_probe`, on h_t; own measurement slot 11) at the end over the whole replay;
`probe.context_state_probes` (h -> s and z -> s) on the real contexts at every diagnostic point.
Reported in REPORT.md, SUMMARY.md, diagnostics.jsonl (`state_probe_h`, `state_probe_z`) and the
console line.

### 4. b aggregation

`aggregate_b.py <run or recomputation dirs>`: pairs the runs by (config, seed), b = experimental -
reference by zone for I_s (main) and I_z (secondary) with the combined se; class means (block class
A / B) with se sqrt(sum se^2) / n and the spread over seeds; each run's C-a, C-b (by zone, with the
floor) and C-c next to it; B_BY_ZONE.md and b_by_zone.json.

## 2026-09-30 (2) -- pooled I_s retained; per-context I_s as a secondary diagnostic; --cpu; git rule

### 1. I_s pooled over a state's contexts: the main measure (user decision)

**Retained: I_s pooled over the contexts of a state** (one kernel per state, K = 1,358 draws per
(s, a) spread over up to 50 contexts). **Why:**
- it is the exact analogue of E2's per-state kernel and of the tabular twin B -- one kernel per
  state, histories merged -- so it is comparable to them;
- its gap to the per-context computation (+0.04 to +0.065 bits on the bridge, 2026-09-30 first
  entry) is small against the differences between zones, and it is computed the same way in the
  environment and in its reference;
- the per-context computation would cost 7.5 times more.

**Per-context I_s stays as a SECONDARY diagnostic, reported with every final measurement**
(`run_mini.per_context_is_check`, config `measure.is_per_context`: 4 states per zone, 10
contexts each, 512 draws per context and action, in every E3 config; own measurement slot 12):
per zone, the mean per-context I_s, the pooled I_s of the same contexts, their bootstrap biases,
and the gap pooled - per-context with its se over states (<= 0 up to sampling: mutual information
is convex in the channel). In results.json (`is_per_context`) and REPORT.md (under the I_s table).
`is_pooling_check.py` runs the same function on any checkpoint (other sizes on request). Cost:
4 x 4 x 10 x 5 x 512 = 0.41 M decodes and matches per final measurement.

### 2. Job CPUs (user request)

`submit_kubeflow.py --cpu N` (default 3, as before; now an integer) sets the Kubernetes CPU
request, and is passed to the worker and to the preflight (`--cpus N`, with the launch's runs per
config, `--batch-runs`). The preflight records the request with the CPUs it can actually use
(cgroup quota, affinity, count; a warning if they differ), measures under that allocation, and
prints the projected wall time of the whole batch (`projected_batch_hours`: runs queued at most
`--max-parallel` at once, longest first; a config left out of the parallel mix is costed as the
slowest measured one; every diagnostic point costed as the final one, an upper bound).
RUN_GPU.md says how to request it.

### 3. Git rule (user decision)

The user committed and pushed the state of 2026-09-30 (merge with origin/main included,
41260d8). **From now on the implementer makes no commit, not even local, and does not modify the
existing history.** At the end of each step it gives the list of files to add and a commit
message, **without any Co-Authored-By line or mention of Claude**; the user runs git add, commit
and push.

## 2026-09-30 (3) -- RUN_GPU.md for an unattended executor; preflight verdict; handoff script

**User decision.** RUN_GPU.md holds only the current batch (gpu_e30b), by the Kubeflow path only,
to be followed from top to bottom by the supervisor's AI assistant on the GPU environment: rules
for the executor (no change to tracked files, no relaunch of a failed run, no weights committed, stop
and report on anything unforeseen), code check (git pull, source hash), preflight (`--batch gpu_e30b
--max-parallel 8 --cpu 5`), an automatic decision rule (launch if PREFLIGHT PASSED and
projected_batch_hours <= 20; after a memory failure at 8, the preflight again at 4 and the same rule;
stop otherwise), follow-up commands, and the handoff committed alone to main. W&B is left out of this
batch (no Secret configured). Everything else (first batch, server launcher, manual install, W&B by
`export`) moved to RUN_GPU_HISTORY.md, marked as history not to be used for the current batch.

**Engineering needed for the rule to be applied without judgement (implementer's decisions):**
- `preflight.py` ends with `<out-dir>/preflight_verdict.json` (passed, problems, memory_problems,
  warnings, parallel, projected_batch_hours, decision) and a last line `PREFLIGHT DECISION: launch |
  retry-max-parallel-4 | stop`, computed by `preflight.decide` -- the rule of RUN_GPU.md step 3
  (`--max-batch-hours 20`, `--fallback-parallel 4`; tested). The memory test's problems are kept
  apart so that "the memory test failed" is readable. A missing projection is a failure.
- The per-run "above the ~6 h target" message of the first batch becomes an informational note: it
  would have fired for every win64 run and blocked the launch, while the launch rule bears on the
  batch (<= 20 h). The CPU mismatch stays a warning (it stops the launch).
- `runs.source_tree_hash` leaves out the Markdown files: RUN_GPU.md states the expected hash, so it
  cannot be part of it, and documentation does not change a run. `runs.py` imports yaml only where
  it writes meta.yaml, so the hash command runs on the standard library alone (the notebook's Python).
  No run had recorded a hash before, so no recorded value changes meaning.
- `make_handoff.py`: from the supervisor's status.json and the run directories, handoff.md (status,
  exit code, gradient steps, switch step, error of every run; the preflight's decision), one light
  archive per run (no .pt, no checkpoints/, no wandb/) and logs.tgz, each below 90 MB (GitHub refuses
  files over 100 MB; one archive for 8 runs of ~16 MB each would approach it); refuses to overwrite.
  Also without a training (after a stop).
- `tests/e3/test_e3_run_gpu_doc.py` reads RUN_GPU.md: every submission and handoff command parses with
  its options, the hash command runs with `python -S` (no site-packages) and prints the hash the file
  states, and the decision lines are the preflight's. **Any change to the code or configs changes the
  hash: that test then fails until RUN_GPU.md is updated.**

## 2026-09-30 (4) -- gpu_e30b results, as they stand; answers to the objectives of entry (1); decisions pending

Full analysis: `reports/E3_GPU_gpu_e30b_analysis_2026-09-30.md` (tables per run, figures, checks). Read-only
analysis of the handoff (`handoff/gpu_e30b/`); no run relaunched, no measurement code changed.

**Outcomes (recorded as they are).** 8 / 8 runs `done` (60,000 gradient steps), `source_sha256` as stated in
RUN_GPU.md in 8 / 8, `git_commit` `unknown` (no git in the container). Batch wall time 14.6 h (preflight
projection 18.1 h); GPU peak 2.2 to 2.5 GiB per process. The preflight log says 3 CPUs requested and usable,
whereas RUN_GPU.md prescribes `--cpu 5` (to confirm with the executor).

**(i) Window length and the separation failures.** All 8 runs separate (switch 5,501 to 7,001; margins 17,999 to
19,499). The first batch's shape (16 x 64) does not prevent separation with the bridge's network. win16 switches
later than win64 in 4 / 4 paired comparisons (mean 6,614 vs 5,814; n = 4, p = 0.125 two-sided sign test: a
tendency). h -> s (0.905 to 0.918) and z -> s (0.993 to 0.999) are the same in both shapes; h is never ahead of z
beyond noise. Between `gpu_e30.yaml` and `gpu_e30b_win64.yaml` only the network (and the deadline, the number of
threads) differs: two paired failures (seed 0 experimental, seed 5 reference) now separate. **Established:** window
length 64 is not sufficient to cause a failure; the failures go with the spec-size network (2 paired cases).
**Not established:** that window length is irrelevant for the spec-size network (no such run), that network size is
the only cause, that the failed runs' state information passed through h (no h -> s probe then).

**(ii) Kernel convergence of the noisy zones at 1,024 images per step.** No. C-b fails in cn (0.150 to 0.173 vs
0.103 to 0.111) and un (0.157 to 0.313 vs 0.107 to 0.113) in 8 / 8 runs; C-a, C-c hold in 8 / 8, C-b holds in ud and
cd in 8 / 8. T4 is met in cn (0.891 to 0.907), not in the environment's un zone (0.687 to 0.766; reference un 0.796
to 0.889). L1 and T4(un) still improve at 60,000 steps, decelerating; structural components do not: in cn the
residual mass is concentrated and 4 to 5 % of the mass is off the true support (H(s'|s,a) right, H(s'|s) too high),
and **I_s in cn is 1.93 to 1.95 against 1.653 (+0.27 to +0.29, 8 / 8 runs), still rising**. Hypothesis (not tested):
a product-of-categoricals prior produces hybrid codes when the alternatives differ in many latent variables (the
noise is spread over 6.8 to 7.7 of 8 variables in the noisy zones, 2.4 to 3.1 in the deterministic ones).

**(iii) First floor of b (I_s, `aggregate_b.py`).** No pair meets all the calibration criteria (C-b, noisy zones).
Where the criteria hold: ud b = -2.19 to -2.27 (theory -2.322), cd and cn b = -0.013 to +0.008 (theory 0).
un b = -1.55 to -1.82 (theory -1.653): not calibrated, dominated by the reference's un zone (I_s 1.60 to 1.88, not
converged). One seed per (config, class): no within-class variance.

**Checks.** I_z - I_s bias of the bridge found again (+0.16 to +0.38 in the environment's uncontrollable zones, +0.25
to +0.29 in cn, +0.02 to +0.03 in cd). Reference seed 0 (win64) against the first batch's (spec network) at the same
K = 32: same order of magnitude everywhere; spec-size more precise in the deterministic zones, worse C-c (0.959 vs
0.995). **Contradictions of expected forms:** C-b in the noisy zones; T4 not near 0.9 in the environment's un zone;
I_s in cn above the truth (so "I_s recovers the truth where it is known" holds for ud, un (environment) and cd, not
cn); the "KL of a deterministic zone ends below 1 bit" form fails narrowly for the environment's ud zone in 3 / 4
runs (1.057, 1.060, 1.099 bits); one state of win16 seed 5 reference exceeds I_z <= log2 5 by 3.03 se (secondary).

**Decisions pending (the user's; nothing chosen here).** (A) which window shape to carry forward (win64: earlier
switch in 4 / 4 pairs, the spec's shape, +40 % training time; win16: 30 % cheaper, latest and most dispersed
separations, worst reference un zone; both: twice the cost); (B) the noisy zones' C-b: train longer / accept with the
documented reserve / revise the criterion (post hoc, to be recorded as such) / investigate the structural cause first;
(C) whether I_s stays the main measure with its documented cn bias; (D) which b to report as the first floor (ud and
the null controls are reliable; un is not); more seeds per class; (E) cost of the next batch (CPU-bound, memory not).

## 2026-09-30 (5) -- decisions after gpu_e30b: window shape, C-b kept, hybrid-codes test, off-support mass, b floor, CPU check, Phase 5 started

### A. Window shape: win64 (16 x 64) retained (user decision)

Why: the switch comes earlier in 4 pairs out of 4 (same seed, condition and data; mean 5,814 vs 6,614), it is
the shape of the first batch and of the spec (comparability), and its final measures cost half as much (11 to 14
min against 23 to 28). **The choice was made without looking at b or at the calibration criteria.**

### B. C-b in the noisy zones: kept as it is; hybrid-codes test before any new batch (user decision)

The criterion is not revised. Before any new batch, the factorised-prior hypothesis is tested on existing
weights (requested from the supervisor: weights_final.pt of win64 seed 0 environment, win64 seed 5 reference,
win16 seed 5 reference). `experiments/E3_worldmodel_4zones/hybrid_codes_check.py <run dir> --weights <file>`
(tested; dry run on the bridge, the only weights in this checkout): for the (s, a) pairs of the noisy zones, from
real contexts,
- factorised draws z' ~ prod_i p_i(. | h') decoded and matched: the share outside the 5 reachable successors;
- two non-factorised comparisons: the joint mode of the same prior, and the posterior codes the model actually
  produced after (s, a) in the replay, decoded with their own h;
- each factorised draw's code classified against the posterior codes of the whole replay: novel (full code never
  produced), novel pair (two variables' classes never produced together), hybrid (novel, and every variable's
  class is one the posterior produced for a reachable successor).

**Criterion, fixed before any result (user): the hypothesis is confirmed if the off-support mass comes mostly from
combinations of variables never observed together in the posteriors** -- implemented as: more than half of the
off-support factorised draws (noisy zones pooled) have a novel code.

Reserves noted by the implementer before the three runs are tested (from the bridge dry run, which is NOT one
of the three runs and was trained at 128 images per step): (1) the base rate is high -- 41 % of ALL the bridge's
factorised draws have a novel code (7,716 distinct posterior codes in 100,000), so "73 % of the off-support draws
are novel" must be read against it; the script also reports the off-support share among novel and among observed
codes (0.31 against 0.08 on the bridge), which is the contrast the hypothesis predicts; (2) no pair of classes
was unobserved (novel-pair share 0): the novelty is of higher order than pairs; (3) the joint mode itself decodes
off support 8 % of the time and the observed-code draws 8 %: a part of the off-support mass is not hybrid.

### C. I_s stays the main measure, on the kernel as it is; off-support mass and projected I_s added (user decision)

- **I_s (main)** is computed on the kernel as it is: the model's actual belief, off-support mass included.
- **Off-support mass** (diagnostic, by zone): the share of p_hat(s' | s, a) on states that are not among the 5
  reachable successors of s (`measures.kernel_probe.off_support_mass`, support =
  `envs.zones.geographic_support_mask`, the support of the twin's prior).
- **I_s projected on the support** (secondary): the same draws matched among the successors' templates only
  (`measures.kernel_probe.match_from_distances(allowed=...)`, the single matching rule), then renormalised and
  passed to the same controllability function, with its own bootstrap se and bias. It serves the comparison with
  the tabular twin B, whose prior is restricted to the same support.
- In every final measurement (results.json, arrays.npz, REPORT.md, the recomputation's SUMMARY.md) and in
  `aggregate_b.py` (the off-support mass is recomputed there from the saved kernel, so it is available for the 8
  gpu_e30b runs: 0.04 to 0.055 in cn, 0.14 to 0.19 in the environment's un zone, 0.05 to 0.07 in its ud zone,
  below 0.01 in cd; the projected I_s needs a decoding and exists only for runs measured from now on).
- **Why the projected version is not the main measure:** it uses the environment's truth (which successors are
  reachable) to measure the model's belief, and would erase the model's errors -- including those that will guide
  the agent in closed loop.
- Implementer's reading of "ghost if the distance to the best support template exceeds the existing ghost
  threshold": the existing ghost rule is a MARGIN (second-best minus best distance below
  `kernel_margin_threshold`), not a distance; the projection applies that same rule among the support's
  templates. To be confirmed.

### D. b floor (user decision)

Established: the conveyor (uncontrollable-deterministic zone), b = -2.19 to -2.27 for a theory of -2.32, and the
null controls (controllable-noisy and controllable-deterministic zones, b = -0.013 to +0.008 for 0). The
uncontrollable-noisy zone is **pending a calibrated kernel**.

### E. Latent size; CPUs (user decision; implementer's engineering)

- The latent stays 8 x 8 as long as test B has not confirmed the hypothesis; no sweep of the latent for now. If
  it is confirmed, before any GPU run: an isolation test on the CPU and a short local run in 4 x 32, to check
  that the separation still happens with the same switch criterion.
- **Why gpu_e30b ran on 3 CPUs instead of 5.** The preflight's own command (status.json) carried `--cpus 3` and the
  container's quota was 3: the submission was made without `--cpu 5` (or with 3). `--cpu` defaulted to 3 and any
  value was accepted, so the prescribed number could be missed silently; the submission record
  (`*.submission.json`) was not in the archive. Now: each batch carries its prescribed number
  (`submit_kubeflow.BATCH_CPUS`: gpu_e30b 5), `--cpu` defaults to it, another value is refused at submission
  unless `--allow-cpu-override`; the preflight verdict records the CPUs requested and usable and prints them;
  RUN_GPU.md's step 3 requires both to equal the prescribed number; `make_handoff.py` archives the submission
  record and reports the CPUs. RUN_GPU.md is marked as finished for gpu_e30b.

### F. Phase 5 started on the CPU (user decision; first version, implementer's choices flagged)

`src/helplessness/agents/worldmodel/`: `heads.py` (goal head, critic, slow critic, r = 50 p - (1 - p),
c = 1 - p), `policy.py` (one-step lookahead Q through RSSM.imagination_step, epsilon-greedy with the relative
tie-break, the latent filter), `imagination.py` (rollouts, lambda-returns, the two losses), `closed_loop.py`
(collection). `experiments/E3_worldmodel_4zones/run_agent.py` with `configs/e3/agent_mini.yaml` (mini_long's
network and procedure): world model trained as in run_mini.py, heads from the switch on, twin B at E2's
checkpoints through E2's own `twin_from_counts`. New streams appended (agent_init, act_latent, imagination).

Implementer's choices to confirm:
1. **Replay convention of the closed loop.** E3-0's replay stores, at each step, the observation before the action
   and the reward of the transition LEAVING it, and never the terminal observation: the goal's image is in no E3-0
   replay, and the model never sees a transition into it. The goal head (target 1[r > 0] on posterior states) and
   the lookahead (r and c read at the predicted next state) need the goal's observation. The closed loop therefore
   stores observations with the reward received ON ARRIVING at them, the goal's observation included (terminal,
   r = 50); an episode of n transitions stores n + 1 steps. E3-0's collection is unchanged. Consequence: a
   closed-loop world model also learns the goal's image and the transitions into it, which no E3-0 model has.
2. **Untrained heads.** Their output layers start at zero (as DreamerV3): every Q ties and the random tie-break
   explores uniformly until the heads have learnt something. With a random initialisation the first version
   followed an arbitrary, 90 %-greedy policy for thousands of steps (tie fraction 0.015).
3. **The heads train from the estimator switch on** (before it the latent is not a discrete code). So the
   behaviour before the switch is uniform exploration in effect, although no epsilon = 1 phase is scheduled.
4. Not given by the spec: the goal head's learning rate (1e-4) and the heads' width (128).
5. Tie-break tolerance: population std of the 5 Q-values.

Not in this first version: the E3 measures of the world model in closed loop (I_s, kernels, D1 / D2), Delta
under P of the world-model agent itself (spec 5.7; the twin's Delta is computed), resume, W&B, the reference
pipeline run side by side (the runner accepts `--conditions reference`).

## 2026-09-30 (6) -- hybrid-codes test: criterion revised before the three runs; decision rule on the latent fixed in advance

Written BEFORE the test is run on the three target runs (win64 seed 0 environment, win64 seed 5 reference,
win16 seed 5 reference; their weights arrived in `handoff/gpu_e30b/weights/`).

**Criterion of the hybrid-codes test, revised (user).** The hypothesis (the off-support mass of the noisy zones
comes from the factorised prior) is **confirmed for a run if the off-support mass obtained by drawing the
posterior codes actually observed after (s, a) is at most half of the off-support mass of the factorised prior
draws** (noisy zones pooled). The off-support mass of the joint mode is reported next to it. **This criterion was
fixed after the dry run on the bridge and before any result on the three target runs.** It replaces the one of
entry (5) B (share of the off-support draws with a novel code above one half), whose base rate the dry run showed
to be high (41 % of all the bridge's draws have a novel code); that statistic is still reported, as a description.

**Decision rule on the latent, fixed in advance (user).**
- If the hypothesis is confirmed on at least 2 of the 3 runs: start-up test of the 4 x 32 latent on the CPU (the
  isolation test on 3 seeds, then a short local run on the basis of mini_long). Criterion: separation in the 3
  seeds, with the present switch criterion and before the deadline. If it passes, the next GPU batch is in
  4 x 32. If it fails: stop and report.
- Otherwise: the next GPU batch stays in 8 x 8, with a doubled gradient budget (120,000 steps), to test whether
  the noisy zones' kernel converges with more training.

### Results of the hybrid-codes test (three target runs, final weights; `hybrid_codes_check.py`, 500 real contexts of the noisy zones' states x 5 actions x 64 factorised draws per run)

| run | off-support mass, factorised prior | posterior codes after (s, a) | ratio | joint mode | verdict |
|---|---|---|---|---|---|
| win64 seed 0 environment | 0.0983 (cn 0.048, un 0.149) | 0.0041 | 0.042 | 0.0272 | confirmed |
| win64 seed 5 reference | 0.0467 (cn 0.040, un 0.054) | 0.0012 | 0.027 | 0.0108 | confirmed |
| win16 seed 5 reference | 0.0627 (cn 0.047, un 0.079) | 0.0033 | 0.053 | 0.0116 | confirmed |

**Confirmed on 3 of 3 runs** (criterion: ratio <= 0.5). Description, same runs: 63 to 85 % of the off-support draws
have a code the posterior never produced (26 to 37 % of all draws do); 55 to 79 % are hybrid (novel, made of the
reachable successors' own classes); no pair of classes is unobserved; the off-support share is 0.11 to 0.29 among
novel codes against 0.014 to 0.05 among observed ones.

Reserve (implementer): the posterior sees the image of the state actually reached, and its reconstruction is right
99.6 to 99.8 % of the time (C-a), so its off-support mass is near zero almost by construction; the criterion
mostly measures that the prior's draws err and the posterior's do not. The evidence that bears on the mechanism is
the description above and the joint mode: decoding the mode of every variable (no mixing of alternatives) leaves
0.011 to 0.027 off support, 3.6 to 5.4 times less than the factorised draws -- most, not all, of the off-support
mass comes from mixing.

**Consequence, by the rule above:** start-up test of the 4 x 32 latent on the CPU.

### Start-up test of the 4 x 32 latent: isolation test FAILED on 3 of 3 seeds -- stopped here, as the rule says

`ae_isolation.run(6000, 1e-3, 32, seed, "warmup_probs_until_separated", k=6, switch_max_steps=3600, n_vars=4,
n_classes=32)` (the retained procedure's isolation protocol, seeds 0, 1, 2, each with its own themes; the latent's
shape is the only change, through the new `n_vars` / `n_classes` arguments):

| seed | 4 x 32: switch | batch-pair JS at the deadline (3,600) | canonical mean JS at 3,500 | 8 x 8, same protocol (control): switch, final hard accuracy |
|---|---|---|---|---|
| 0 | none | 0.000 | 0.00 | none (JS 0.17 at the deadline) |
| 1 | none | 0.000 | 0.00 | 2,601, 0.997 |
| 2 | none | 2.892 | 2.80 (mode accuracy 0.76) | 1,851, 0.999 |

Criterion (separation in the 3 seeds, present switch criterion, before the deadline): **not met, 0 of 3**. The short
local run on the basis of mini_long was therefore NOT made, and no GPU batch in 4 x 32 is prepared.

What the control and the numbers show (description; nothing was changed):
- the same protocol in 8 x 8 switches in 2 of 3 seeds (seed 0's themes do not separate within 3,600 steps in either
  shape: the isolation deadline is tight for that seed, see THEME_SENSITIVITY.md), so 0 of 3 is a difference of the
  latent's shape for seeds 1 and 2, not of the protocol;
- seed 1 shows no separation at all in 4 x 32 at 3,500 steps where 8 x 8 had switched at 2,601: the start-up is
  slower;
- seed 2 does separate in 4 x 32 (JS rising from step 1,000, 2.8 bits at 3,500) but stays under the threshold:
  the batch-pair JS is a sum over the latent variables, each bounded by 1 bit, so its ceiling is 4 bits for 4
  variables against 8 for 8, while the threshold stayed at 3.0 bits (75 % of the ceiling instead of 37.5 %; the
  8 x 8 runs plateau at 6.2 to 6.5 of 8). The "present switch criterion" is not scale-free in the number of
  variables.

Decision pending (user): the latent of the next GPU batch. The rule's two branches do not cover this case (the
hypothesis is confirmed, the 4 x 32 start-up fails).

### Expected shapes of the agent_mini bench run -- PROPOSED by the implementer, to be confirmed before the run is read

No expected shape of this run was recorded before. Proposal (configs/e3/agent_mini.yaml, seed 0, experimental,
20,000 environment steps):
1. the world model separates: the switch happens before the 25,000-gradient-step deadline, and no collapse guard
   fires afterwards;
2. before the heads have learnt anything the greedy tie fraction is close to 1 (zero-initialised heads) and the
   success rate is that of uniform exploration (about 0.1 within H = 36); after the switch the tie fraction falls;
3. the goal head learns the reward signal: goal_loss decreases after the switch and goal_recall rises toward 1;
4. behaviour improves after the switch: the success rate over the last 100 episodes rises above its pre-switch
   level and the mean episode length falls;
5. the twin (condition B): Delta >= 0 in every zone at every checkpoint (T9); its L1 to P decreases with the
   counts; its b is negative in the two uncontrollable zones and near 0 in the two controllable ones.
A run that contradicts one of these is reported as such.

## 2026-09-30 (7) -- decisions after the hybrid-codes and 4 x 32 tests; expected shapes of agent_mini (final); bias control; scale-free switch

### Expected shapes of the agent_mini bench run -- FINAL (validated by the user before the run)

`configs/e3/agent_mini.yaml`, seed 0, experimental environment, 20,000 environment steps:
1. the world model separates: the switch happens before the 25,000-gradient-step deadline, and no collapse guard
   fires afterwards;
2. before the switch the policy is exactly uniform (zero-initialised heads: every Q ties, and the epsilon branch is
   uniform too): the greedy tie fraction is 1 and the success rate is that of a uniform policy in this environment,
   computed exactly under P (uniform starts over the non-terminal states, H = 36, the goal absorbing):
   **0.0459** for seed 0 experimental (expected episode length 34.97; 95 % range of a 100-episode window
   0.005 to 0.087). For reference: 0.1173 in the reference environment (same value for seed 5, whose layout is the
   same up to the permutation of the zones). After the switch the tie fraction falls;
3. the goal head learns the reward signal: goal_loss decreases after the switch and goal_recall rises toward 1;
4. behaviour improves after the switch: the success rate over the last 100 episodes rises above the uniform level
   and the mean episode length falls;
5. the twin (condition B): Delta >= 0 in every zone at every checkpoint (T9); its L1 to P decreases with the counts;
   its b is negative in the two uncontrollable zones and near 0 in the two controllable ones.
A run that contradicts one of these is reported as such.

### Hybrid codes: conclusion (user decision)

The main cause of the off-support mass in the noisy zones is confirmed: the factorised prior mixes the latent
variables of different successors. **The conclusion rests on two measures, not on the counterfactual criterion:**
the joint mode (3.6 to 5.4 times less off-support mass than the factorised draws) and the over-representation of
never-produced codes (63 to 85 % of the off-support draws against 26 to 37 % of all draws). The counterfactual
criterion (posterior codes after (s, a) against factorised draws) was almost met by construction: the posterior
codes decode their own image correctly (C-a). The bias is structural for a factorised prior in a multimodal zone;
more training budget would not remove it.
**Corrected 2026-10-05 (22):** the pilot (180,000 gradient steps) shows that the bias falls clearly with training
but does not disappear (C-b still fails in the noisy zones); "would not remove it" stands, "would not reduce it" does not.

### Latent and calibration floor (user decision)

- The latent stays **8 x 8**. The 4 x 32 latent is abandoned for E3 (slower start-up, seed 1).
- No further GPU calibration batch: the win64 8 x 8 runs of gpu_e30b are the **provisional E3-0 floor**; the final
  floor will be rerun with the final campaign (goal-storage convention aligned with the closed loop, more seeds).
  **Superseded 2026-10-02 (16): E3-0 is not rerun; the floor and the reference of every measure come from the final
  closed-loop campaign itself (its environment and reference runs).**
- C-b in the noisy zones: the criterion is kept; its failure is recorded with its structural cause.
- The next GPU batch is the **closed-loop pilot** (seeds 0 and 5, environment and reference), prepared once the
  closed-loop measures are done and agent_mini has been read, with RUN_GPU.md in the same format (replaces gpu_e30c).

### Switch criterion: scale-free in the number of latent variables (user decision; implementation below)

The threshold is expressed as a fraction of the ceiling: the batch-pair JS averaged over the latent variables
(each bounded by 1 bit), `switch_js_threshold_per_var` and `collapse_js_threshold_per_var`; in bits of the summed
statistic, threshold = fraction x n_vars. Fixed to reproduce the present 8 x 8 behaviour exactly: 3 bits of 8 =
0.375 per variable (collapse: 1 bit of 8 = 0.125). 0.375 x 8 = 3.0 exactly in floating point, so an 8 x 8 run is
unchanged bit for bit; the existing configs keep their `switch_js_threshold` in bits and are unchanged (tested).

### Bias control in closed loop: reading rule and thresholds (user decision; thresholds proposed by the implementer)

**Partly superseded by (9): the rule applies to the controllable zones only, the thresholds are per zone and defined differently (TAU_B = 0.14 and TAU_OFF = 0.02 below are dropped), and UN / UD are read against the E3-0 floor.**

For every environment / reference pair, `aggregate_b.py` gives, by zone: b computed on I_s and on I_s projected on
the geographic support, and the difference of off-support mass between the two environments (environment minus
reference). **Reading rule: b is interpretable in a zone only if the two versions of b agree,
|b(I_s) - b(I_s projected)| <= TAU_B, and the off-support mass does not differ between the environment and its
reference, |off(env) - off(ref)| <= TAU_OFF.** A run without I_s projected (measured before 2026-09-30 (5)) gets the
verdict "partial" (the off-support condition only). Both thresholds are options of `aggregate_b.py`
(`--tau-b`, `--tau-off`).

Thresholds, from the between-seed variability of gpu_e30b: for each (window shape, condition, zone), the paired
difference between seeds 0 and 5 estimates sqrt(2) x the standard deviation of one run; the threshold is twice that
standard deviation (a disagreement or a difference beyond it is larger than the seed-to-seed spread of one run),
by zone and pooled over the four zones (root mean square):

| zone | TAU_B (b on I_s) | TAU_OFF (off-support mass) |
|---|---|---|
| uncontrollable deterministic | 0.061 | 0.013 |
| controllable noisy | 0.014 | 0.007 |
| uncontrollable noisy | 0.265 | 0.032 |
| controllable deterministic | 0.013 | 0.004 |
| **pooled (retained)** | **0.136 -> 0.14 bits** | **0.018 -> 0.02** |

What the rule gives on gpu_e30b (8 runs, no I_s projected, so "partial"): the off-support condition holds in the
two controllable zones (differences -0.003 to +0.012) and **fails in the two uncontrollable zones in every pair**:
uncontrollable deterministic +0.039 to +0.058, uncontrollable noisy +0.089 to +0.137 (the environment's kernel
leaks more than the reference's, which is controllable-deterministic there). Under the rule as stated, the b of
gpu_e30b in the uncontrollable zones -- including the conveyor's, about -2.2 bits -- is **not interpretable**.
Recorded as it is; a remark for the user, nothing changed: the rule does not scale with the size of b (an
off-support difference of 0.05 cannot by itself produce 2.2 bits), and the reference of an uncontrollable zone is a
controllable-deterministic zone, whose kernel is sharper by construction, so the off-support condition may fail
wherever b is large. Recomputing I_s projected for the gpu_e30b runs (recompute_measures.py, CPU, long at
K = 1,358) would give the full verdict; not done.

## 2026-09-30 (8) -- closed-loop measures (4c) and accelerations (4b): implementation

### 4c: the E3 measures in closed loop, Delta under P of the agent, resume (run_agent.py)

- **Delta under P of the world-model agent** (spec 5.7) at every checkpoint, by zone: the agent's greedy policy
  (probe.agent_greedy_policy: Q of the one-step lookahead averaged over a state's contexts, uniform over the tie set,
  uniform where no context was read), evaluated exactly under P (probe.delta_under_p: V* - V^pi, >= 0). Two
  readings: real contexts (main, the latent states the agent actually had in the replay, as the other measures) and
  the canonical probe (secondary). Own measurement slot (14), a child stream keyed by the checkpoint.
  Columns agent_delta_real / agent_delta_canonical of metrics.csv, field agent_delta of checkpoints.jsonl.
- **Final measures** (measure.final, true in agent_mini.yaml): run_mini.measure_final on the closed-loop replay and
  counts at the end of the run -- I_s, I_s projected, off-support mass, I_z, kernels, probes, D1 / D2 -- written as
  run_mini writes them (arrays.npz, results.json), so aggregate_b.py pairs a closed-loop environment run with its
  reference, with the bias control above (tested on a tiny pair). agent_mini.yaml's measure block is mini_long's,
  unchanged (tested); its schedule keys (diagnostics_every, separation_every, ...) are run_mini's and unused here.
- **Resume**: every checkpoints.resume_every environment steps (2,000 in agent_mini.yaml) and at every checkpoint,
  checkpoints/resume.pt holds the whole state -- models, optimisers, switch, replay and environment (pickled), the
  collector (counts, episode statistics, current observation, latent filter), every random stream (numpy and torch,
  and torch's global state) and the records -- written atomically. `run_agent.py --resume <run dir>` continues in the
  same directory. Tested: an interruption at step 150 then --resume gives the same checkpoints.jsonl, metrics.csv,
  counts and final weights (bit for bit on CPU) as the uninterrupted run.

### 4b: two accelerations of the training step, off by default (src/helplessness/agents/worldmodel/acceleration.py)

- **torch.compile** of the encoder, the decoder and `RSSM.observation_core` (observation_step without its posterior
  draw -- a refactoring with no change of behaviour: the draw, and so the latent stream, stays eager). In place:
  parameters and state_dict unchanged. Option `runtime.compile: {backend, mode}` (mode "reduce-overhead" = CUDA
  graphs); absent in every config, so nothing changes. Equivalence test (5 steps, both phases): tracing backend
  (aot_eager) passes here; inductor needs a C++ compiler absent on this Windows machine (the test is skipped here and
  will run on Linux / GPU).
- **Several seeds in one process**: parameters stacked (torch.func.stack_module_state), every dense call vectorised
  over the seeds (vmap + functional_call); per seed and outside the vectorised calls: the minibatch, the latent
  draws from the seed's own generator (same calls, same order: the stream is consumed identically, tested bit for
  bit), beta_rep, the estimator phase, the learning rate (seeds switch at different steps: StackedAdam, Adam with one
  learning rate per seed) and the gradient-norm clipping. The GRU cell is written with its gate equations in the
  stacked path (vmap has no batching rule for aten::gru_cell and would loop over the seeds). Equivalence test: two
  seeds with different weights, batches, beta_rep and learning rates, one switching a step before the other, binding
  clipping: losses within 1e-4 relative and weights within 2e-5 of the sequential step over 5 steps. Library only:
  not wired into run_mini.py / run_agent.py (one seed per process); a multi-seed runner is worth writing only if the
  bench shows the gain.
- **Speed bench** in the preflight (GPU, informational, never in the verdict; alone: `preflight.py --config ...
  --bench-accel`): the current step against compile (default and reduce-overhead) and against 2 stacked seeds, 200
  steps per phase after warm-up, in seed-steps per second, with the warm-up (compilation) time and each variant's
  loss difference from the current step. Results in preflight_summary.json / acceleration_bench.json. Nothing is
  enabled on the strength of the bench without a decision.

## 2026-09-30 (9) -- reading rule by zone type (correction of (7)), per-zone thresholds, one execution mode per campaign

### A. Scope of the reading rule (user decision; correction of (7))

The rule of (7) applies to the **controllable zones (CN, CD)** only: that is where the propagation is tested, and the
local dynamics are the same in the environment and in its reference. In an uncontrollable zone the reference is a
controllable zone, so the off-support condition (environment against reference) fails by construction.

- **CN, CD**: b is interpretable only if the two versions of b agree, |b(I_s) - b(I_s projected)| <= tau_b(zone), and
  the off-support mass does not differ between the environment and its reference, |difference| <= tau_off(zone).
- **UN, UD**: b is read against the **E3-0 floor of the zone** (the b of the same zone on the E3-0 pairs), provided the
  off-support mass of the zone on the environment side, in closed loop, stays within the interval observed on the E3-0
  environment runs, widened by +- tau_off_floor(zone). Otherwise the zone is reported as **diagnostic only**.

**This correction was made after seeing the rule fail in the uncontrollable zones of gpu_e30b** (calibration runs,
before any closed-loop result), with the structural reason above. It is recorded as such: the first version of the
rule ((7)) was tested on gpu_e30b and found inapplicable to UN and UD; nothing in the closed-loop pilot informed it.

Implementation: `aggregate_b.py`. `aggregate(runs, tau_b, tau_off, floor)` applies the rule by zone type
(`is_controllable`: the zone category is named after its dynamics); verdicts: "interpretable" / "NOT interpretable" /
"partial: off-support ok|NOT ok; projected b unavailable" (CN, CD), "read against the floor" / "diagnostic only
(off-support outside the floor's interval)" / "floor unavailable" (UN, UD). `--e30 <E3-0 runs>` derives the
thresholds and the floor (`thresholds_from_pairs`, `floor_from_pairs`; `--write-thresholds` saves them), or
`--thresholds <json>` (default `configs/e3/bias_control_thresholds.json`). The scalar thresholds of (7) (TAU_B = 0.14,
TAU_OFF = 0.02) are removed. Tested (tests/e3/test_e3_aggregate_b.py).

### B. Thresholds (user decision; definition fixed before the values were computed)

TAU_B = 0.14 is dropped: dominated by UN, where the rule no longer applies, and seven times the size of E2's effect in
CN. New definition, **per zone, for CN and CD**: twice the standard deviation of the tested quantity itself,
estimated over the win64 pairs of gpu_e30b (seeds 0 and 5; sample sd, ddof = 1):

- tau_b(zone) = 2 x sd over pairs of [ b(I_s) - b(I_s projected) ];
- tau_off(zone) = 2 x sd over pairs of [ off-support(environment) - off-support(reference) ].

Limits, stated now: **two seeds only, of different block classes**, so a standard deviation over n = 2 values: a fragile
estimate (one degree of freedom), which also mixes the seed effect with the class effect. The thresholds are an order
of magnitude, not a calibrated noise level.

For UN and UD the text of the decision fixes the interval as "+- TAU_OFF of the zone" without a definition of that
TAU_OFF (B defines thresholds for CN and CD only). **Implementer's reading, to be confirmed**: tau_off_floor(zone) =
2 x sd, over the E3-0 environment runs, of the environment-side off-support mass of the zone (the quantity the interval
bounds -- same principle: twice the sd of the tested quantity itself; here also n = 2).

### C. Accelerations: one execution mode per campaign (user decision)

The execution modes (compile or not, stacked seeds or not) are not bit-equivalent. **All the conditions of one
campaign -- environment, reference, and E3-0 if it is rerun -- run in the same mode.** Implementation
(`acceleration.execution_mode`, `mixed_modes`, `preflight.execution_mode_problems`):

- the preflight reads every config of the campaign (`--config`, `--extra-config`, and `--campaign-config` for
  configs that belong to the campaign but are not launched by this job) and **fails if their modes differ**;
- **runtime.compile is refused** unless the equivalence gate passes, which only runs on **Linux with CUDA**
  (`acceleration.check_compile_equivalence`): the eager and the inductor-compiled step take the same 4 steps (both
  phases) from the same weights, batches and generators, the loss difference must be <= 1e-4 (relative), and **TF32
  must still be off after the compiled steps** (torch.backends.cuda.matmul / cudnn allow_tf32 false,
  float32_matmul_precision "highest", in the state `configure_runtime` leaves). The compilation's own failure is a
  refusal. On any other platform or device the option is refused as well: **no silent fallback** to the eager step;
- `runtime.stacked_seeds` is refused: no runner trains stacked seeds (library only);
- the mode of each config is printed and stored in preflight_summary.json (`execution_mode`).
Tested on the CPU with the tracing backend (the gate's logic, failing and passing cases, the TF32 check) and with
mocks for the Linux / CUDA branch; the inductor run itself happens on the GPU machine.

### gpu_e30b: state of the recomputation and what could be computed (status at the end of the day)

- **Blocked: the weights** (2026-09-30). Only 2 of the 4 win64 runs had their `weights_final.pt` here (experimental seed
  0, control seed 5: two seeds AND two conditions, so no pair). I_s projected needs the weights (it was not measured on
  the GPU). The CPU recomputation of these two runs was started, then **interrupted before the end on the user's
  decision** (30 Sept., evening: too long; more than 35 CPU-minutes each, single thread, K = 1,358). **No result of it
  is retained** (its partial outputs were deleted) and **it is not to be relaunched**: the measurement is redone on the
  GPU for the four runs by the batch gpu_e30b_measures, 2026-10-01 (10). Nothing was changed in the definition.
- **Computable from the existing arrays (GPU runs, no recomputation), definition (B) unchanged**, n = 2 pairs:
  tau_off(CN) = 0.0154, tau_off(CD) = 0.0035 (differences env - ref: CN +0.0124, +0.0015; CD -0.0015, +0.0009);
  tau_off_floor (implementer's reading, to be confirmed): UD = 0.0175, UN = 0.0214.
  E3-0 floor (win64 pairs, b on I_s): UD -2.254 (-2.239, -2.268), UN -1.771 (-1.792, -1.750); environment-side
  off-support interval: UD [0.047, 0.060], UN [0.142, 0.157].
  tau_b(CN, CD): not computed (needs I_s projected).

## 2026-10-01 (10) -- gpu_e30b_measures: the four win64 runs of gpu_e30b measured again on the GPU (no training)

### What and why (user decision)

**No training: only the measurement is redone.** The four win64 runs of gpu_e30b (experimental and reference, seeds 0
and 5) are measured again by `recompute_measures.py` from their own final weights, on the A100, in one job. Reason:
**I_s projected on the support did not exist when gpu_e30b ran** (added 2026-09-30 (5)), and the thresholds of the
reading rule ((9), B) need it on both sides of both pairs. Both sides of each pair are measured on the same platform
with the same version of the code: that is the reason for the batch (a CPU recomputation of some runs next to GPU
measurements of others would mix platforms). The measurement configuration is each run's own `config.yaml` (K = 1,358,
measurement streams, ghost thresholds, real-context settings), TF32 off and deterministic algorithms on, as in gpu_e30b;
nothing is overridden (no `--kernel-draws`).

The two CPU recomputations started on 2026-09-30 (experimental seed 0, control seed 5) were **interrupted before the
end by the user's decision; no result of them is retained, and they are not relaunched** (see (9)).

### Inventory and integrity of the weights (2026-10-01)

`handoff/gpu_e30b/weights/` holds the four win64 final weights (the last two pushed by the supervisor on 2026-10-01).
- Checksums: the SHA-256 of each file is the one `handoff/gpu_e30b/handoff.md` lists for its source run (4 / 4). The
  run archives of gpu_e30b contain no weights (by design), so there is no second, independent checksum to compare with.
- Identity, checked on the content (not on the file names): each file's model, run on its run's own pre-rendered images
  (canonical probe, deterministic), reproduces the posterior probabilities stored in **its** run's `arrays.npz` to
  1.3e-6 to 1.6e-6 (GPU float vs CPU float), and differs from those of the three other runs by 0.99. Gradient step
  60,000 and switch step equal to the run's (`meta.yaml`) in all four. Seed, condition and window of each directory
  name agree with its `meta.yaml` and `config.yaml`. **No discrepancy.**

### The weights do not travel

The jobs read the runs in place, on the volume where gpu_e30b wrote them
(`/scratch/helplessness-worldmodels-gpu_e30b/results/E3_worldmodel_4zones/<run>`), and write only into the job's own
directory (`recompute_measures.py --out-dir`); each job checks its weights' SHA-256 before measuring
(`--weights-sha256`, the values above). The copies in `handoff/gpu_e30b/weights/` are not used by the batch.

### Engineering (implementer)

- `submit_kubeflow.py`: measurement batches (`MEASURE_BATCHES`, phases `preflight` and `measure` only; `train` and
  `resume` refused for them, `measure` refused for a training batch); batch `gpu_e30b_measures`, 5 CPUs prescribed.
- `preflight_measures.py`: CUDA, TF32 off / deterministic after `configure_runtime`; **CPUs: requested = prescribed =
  usable = 5, otherwise a problem (not a warning)**; the checkout's source hash = the one RUN_GPU.md states; each source
  run present, its name / `meta.yaml` / window = the job's, weights SHA-256 as expected; one full measurement of the
  first run on the GPU (timed, outputs complete, TF32 off recorded); its comparison with the run's original arrays is
  reported, not judged. Decision launch / stop, no retry branch.
- `make_handoff.py --measure-dir`: one light archive per run (`<run>.recomputed.tgz`: results.json, arrays.npz,
  SUMMARY.md, comparison_with_run.json, artifacts; no `.pt`), `handoff.md` with each job's code hash, weights checksum
  and comparison with the original; into `handoff/gpu_e30b_measures/`.
- `recompute_measures.py`: `--out-dir` (the run directory only read), `--weights-sha256`, and the measuring
  environment (code hash, device, GPU, TF32) recorded in results.json (`reproducibility`).
- `tau_gpu_e30b.py`: one command once the archive is in `handoff/` (below).

### Consistency check (to be read in the aggregation)

gpu_e30b's original measurements were made on the same GPU. The re-measurement must therefore find again the stored
I_s, I_z (real contexts) and off-support mass. `tau_gpu_e30b.py` compares them by run and zone (and the per-state
arrays real_I_s, real_I_z, real_p_hat), with a tolerance of 1e-6; any difference above it is flagged **"DIFFERS -- to
be reported"** (either the measurement code changed or the platform differs) and the script exits with 1. It is
reported to the user, not resolved by the implementer.

### Thresholds: the computation, prepared before any value (definitions of (9) unchanged)

`python experiments/E3_worldmodel_4zones/tau_gpu_e30b.py` checks the batch (the four runs, expected weights, one code
hash, one device and GPU, TF32 off, same K), the consistency above, then computes on the two win64 pairs, by
`aggregate_b.thresholds_from_pairs`: for CN and CD, tau_b = 2 x sd of b(I_s) - b(I_s projected) and tau_off = 2 x sd of
off-support(env) - off-support(ref); for UN and UD, tau_off_floor and the E3-0 floor (the reading of (9) for UN / UD,
still to be confirmed); and the rule's verdict on every pair and zone. Outputs in `reports/tau_gpu_e30b/` (TAU.md,
B_BY_ZONE.md, thresholds.json); it refuses an output directory that already holds files.

**Raised to the user before any value (applicability with two seeds).** The definition is computable, but:
1. one degree of freedom: with n = 2, sd = |q0 - q5| / sqrt(2), so tau = sqrt(2) |q0 - q5|. The 95 % interval of a
   standard deviation estimated with 1 degree of freedom runs from about 0.45 to 32 times the estimate: tau is an order
   of magnitude at best, and can be arbitrarily small when the two seeds happen to agree;
2. **circularity of the verdict on gpu_e30b itself**: on the two pairs that set the thresholds, pair i passes the b
   condition iff |q_i| <= sqrt(2) |q0 - q5|. When the two seeds share a bias of the same sign and size (a systematic
   bias), tau is small and both pairs fail; when their differences have opposite signs, both pass. The verdict on
   gpu_e30b is thus set by the relative position of two numbers, not by an independent noise level: it can be reported
   as a description, not as a test. The thresholds are meaningful for pairs that did not set them (the closed-loop
   pilot);
3. the two seeds are of different block classes (A and B), so the spread mixes seed and class effects.
None of this changes the definition; the script applies it as fixed. Whether the gpu_e30b verdict is reported as a
test or as a description is the user's decision.

## 2026-10-01 (11) -- agent_mini recorded as FAILED; read-only diagnostics; closed-loop GPU pilot suspended

### Status (user decision)

The bench run `results/E3_worldmodel_4zones/grid12x12__4zones__pair-experimental__worldmodel-agent-agent_mini__seed00__20260930T193212Z`
(configs/e3/agent_mini.yaml, seed 0, experimental, 20,000 environment steps, finished, 18,383 s on the CPU) is recorded
as **failed**: expected shapes 3 and 4 of (7) are contradicted (shapes 1, 2 and 5 hold). The run directory is kept as
it is: not overwritten, not relaunched. The closed-loop GPU pilot is suspended.

### Diagnostics (read-only; `experiments/E3_worldmodel_4zones/diagnose_agent_run.py`, output in
`reports/diagnostics_<run directory name>/diagnostics.json`)

Only two weight sets exist (the resume point, environment step 18,000, and the end, 20,000): run_agent.py saves no
weights at the checkpoints. Held-out frames: 4,000 fresh episodes of the uniform policy (own seeds; 175 reach the goal).

a. **Goal frames.** 4 goal observations in the replay at the switch (environment step 2,382), 13 of 18,510 stored at
   18,000 (0.07 %). A training batch (8 x 16 = 128 frames) holds 0.21 positives on average at the switch, 0.09 at
   18,000; **81 % to 91 % of batches hold none**. The goal head saw at most about 3,200 positive frames over ~31,000
   updates. Cumulative successes: 4 by 2,512, 13 by 18,000 (510 episodes), 20 by 20,000 (568 episodes).
b. **The goal head separates perfectly and is calibrated**, at both weight sets, on the replay and on the held-out
   frames: AUC 1.000; p_goal on positives = 1.00 (5 % quantile 1.00), on negatives 0.00 (95 % quantile 0.00); mean
   p_goal over all frames = the true frequency of positives (0.0007 vs 0.0007 on the replay, 0.0012 vs 0.0012 held
   out). The recall logged in checkpoints.jsonl is computed on ONE batch with 0/max(1, n_pos): a batch without positive
   (81-91 % of them) logs 0. Shape 3's "goal_recall rises toward 1" was therefore not a valid test of the head.
c. **The break is upstream of the head: the world model's prior never predicts the goal.** For the four
   (state, action) pairs that lead deterministically into the goal (visited 2 to 8 times), the decoded kernel gives the
   goal a probability of 0.000 to 0.001 (its prediction is diffuse elsewhere, max 0.10-0.36). On the frames one step
   from the goal, p_goal of the imagined next state for the action into the goal is ~1e-11 (median), and the argmax
   picks that action in 10 % of them (chance 20 %). Hence r_hat = -1.0000 for every imagined step, and the critic has
   learnt V = -10.01 (5-95 %: -10.05 to -9.98) everywhere = -1/(1 - gamma): the value of never reaching the goal. Its
   ranking of states is unrelated to V* (Spearman -0.08 at 18,000, +0.14 at 20,000).
   **Policy.** The Q gap between the best and the second action is 0.005-0.007 (median), entirely from gamma c V
   (spread over actions 0.024-0.027; r_hat's spread is 0), i.e. the critic's residual noise around -10. The tie
   tolerance, relative (1 % of the std of Q over the actions), is ~1e-4: the gap is 60-75 times the tolerance (median),
   so ties vanish (0.8 % of frames) **without any information** -- this is the fall of the tie fraction from 1.0 to
   0.13. The argmax is neither constant nor cyclic in itself: all five actions win somewhere, a state's argmax agrees
   across its contexts in 74 % of reads (context-dependent in the rest), and it is an optimal action under P in 41-46 %
   of frames against 41 % for a uniform choice. Played actions become skewed (after 10,000: 0.23 / 0.23 / 0.13 / 0.30 /
   0.12; repeating the previous action 27 % vs 20 % before the switch). Unlike E2's tabular agent, whose Q ties exactly
   where it has no information (uniform tie-break), this agent turns the absence of information into an arbitrary
   deterministic preference.
d. **Success after the switch, pooled**: from the checkpoint records (2,512 to 20,000) 16 / 496 = 0.032, Wilson 95 %
   [0.020, 0.052], exact two-sided binomial p = 0.16 against the uniform rate 0.0459; from the replay (2,512 to 18,000)
   9 / 437 = 0.021 [0.011, 0.039], p = 0.008. Observed afterwards, not tested before: the last 2,000 steps hold 7 / 58
   successes (0.12); a window chosen after seeing the curve, not a result.
e. **Shape 5 holds**: the twin's L1 to P decreases at every checkpoint in all four zones (UD 1.27 -> 0.52, CN 0.76 ->
   0.19, UN 1.03 -> 0.31, CD 0.83 -> 0.17); Delta >= 0 everywhere; b at 20,000: UD -1.03, UN -1.06, CN +0.01, CD 0.00.

### Options raised to the user (nothing implemented: each changes what the agent believes or does)

1. Tie tolerance: absolute, or tied to the estimation noise of Q (e.g. its Monte Carlo standard error over the K draws,
   or a fixed fraction of |goal_reward|), instead of 1 % of the spread of Q. No effect on p_goal. Restores E2's
   behaviour when the value information is absent (exact-tie analogue -> uniform tie-break); the threshold is a
   choice that also sets when real but small preferences are ignored.
2. Goal head: positive weighting of the BCE, or balanced batches for the head only. The head is not the bottleneck
   (AUC 1, calibrated); a weight w inflates p_goal's odds by w, so r_hat and c would have to be corrected (odds / w)
   to stay calibrated. Little expected effect.
3. Replay sampling for the world model: over-sample windows that contain a goal arrival, so the prior learns the
   transitions into the goal (the actual break). Changes the world model's data distribution: its predicted kernel
   near the goal (and p_goal's base rate seen by the head) is biased toward the goal unless importance-corrected; it
   departs from E2's twin, which counts every transition once.
4. Data: the goal observation is rare because the agent rarely gets there (and E3-0's replay never stored it).
   Seeding the world model with goal arrivals (e.g. prefill under the uniform policy for longer) changes the protocol
   of section 4.5 (minimal prefill).

## 2026-10-01 (12) -- reading rule of the world-model diagnostics (fixed BEFORE computing them); tie-break; per-checkpoint weights

### Reading rule (user decision, recorded before any of the numbers below was computed)

Question: is the world model's failure to predict the goal (11) specific to the goal, or a general insufficiency of
the prior in rarely visited contexts?

Quantity: for every (s, a) pair whose true transition is deterministic (one successor s* with P = 1), with s and s*
both different from the goal, the **mass of the prior on the right successor**: p_hat(s* | s, a) of the s-space kernel
in real contexts (the main measure's kernel, probe.real_context_kernel: prior draws decoded and matched to the
templates, renormalised over the non-ghost draws as the kernel is defined; the raw mass including ghosts,
p_hat x (1 - ghost fraction), is reported next to it), against the number n of real visits of (s, a) (the run's
counts N at the weights' environment step).

Rule: over the pairs with 2 <= n <= 8, the **median** mass on the right successor
- >= 0.9, while the median mass on the goal of the four pairs that lead into the goal is < 0.1: the failure is
  **specific to the goal**;
- <= 0.5: **general insufficiency** of the prior in rarely visited contexts;
- in between: **undetermined, no action**.
Applied at each of the two weight sets (environment steps 18,000 and 20,000), each with its own replay and counts
(18,000: the replay and counts of the resume point; 20,000: the run's final measures, arrays.npz). A verdict holds
only if both weight sets give it; otherwise undetermined.

### Diagnostics of the world model around the goal (computed after the rule above was committed)

`experiments/E3_worldmodel_4zones/diagnose_world_model_goal.py`, output in
`reports/diagnostics_wm_goal_<run directory name>/diagnostics.json`; kernels with the run's own settings (K = 1,358
draws per pair, ghost margin 0.001). 461 deterministic pairs off the goal, all measured.

a. **Verdict of the rule: general insufficiency of the prior in rarely visited contexts**, at both weight sets.
   Median mass on the right successor over the pairs with 2 <= n <= 8: **0.284** at 18,000 (170 pairs) and **0.325**
   at 20,000 (final measures, 158 pairs) -- both <= 0.5; median mass on the goal of the four goal pairs: 0.000 at
   both. The mass rises with the visits (median of the per-n medians): n 0-1: 0.11 / 0.12 (63 / 59 pairs); 2-8: 0.30 /
   0.32; 9-20: 0.81 / 0.79 (100 / 99); 21-50: 0.94 / 0.93 (66 / 73); 51-100: 0.97 / 0.97 (33 / 41); > 100: 0.99 /
   0.99 (29 / 31). Ghosts take 1 to 4 % of the draws (raw masses within 0.02 of the renormalised ones).
   **Corrected 2026-10-02 (13)**: the sentence first written here, "the four goal pairs sit where any pair with so few
   visits sits; they are not singled out", was wrong. Their percentile rank among the deterministic pairs with the
   same number of visits is 0 (1.5 for one pair at 18,000): they are the lowest of their visit class at both weight
   sets (see (13)). This is descriptive, not a criterion: the verdict of the rule is unchanged.
b. The four goal pairs: mass on the goal 0.000 to 0.001 at both weight sets (their argmax: other states, 0.10 to
   0.38). p_goal on latents drawn from the PRIOR at their real contexts (512 draws per context): mean 1e-10 to 0.004,
   at most 0.3 % of the draws above 0.5 -- the head recognises the goal in the rare draws that hold its code, and the
   prior almost never draws it. Pair (12, 0) went from 1 visit at 18,000 to 8 at 20,000 (the late successes) with no
   change of its mass on the goal.
c. Dynamic KL (posterior vs prior, bits, per frame): goal frames median 28.1 / 29.6 (13 frames; 5-95 %: 21-44);
   other frames median 0.42 / 0.35 (mean 2.0 / 1.8; 95 %: 8.9 / 8.4). The goal frame is the most surprising frame of
   the replay for the prior: it is in the loss (d) with a large error, and is seen 13 times.
d. Code read and test (`tests/e3/test_e3_terminal_frame_loss.py`): the collector stores the pre-goal observation with
   the action taken from it and the goal's observation after it (terminal, not is_first, action 0, reward 50); the
   training sampler draws windows over every stored step; the unroll feeds the GRU step into frame t with the action
   stored at t - 1; the loss averages the dynamic and representation KL over every step of the window, with no
   terminal mask. Tested on a synthetic episode ending at the goal: the goal step's KL enters the dynamic term with
   weight 1 / (B T), changing only the goal's image changes that term and its gradient on the prior network, and the
   GRU step into the goal receives the action that led there. **No bug found.**
e. The templates cover the goal: the observation is the goal's template by construction (table mode raises
   otherwise); the goal's template is the most distinctive of the 144 (squared-L2 distance to its nearest other
   template 11.7, against 2.6 median for the others); the decoded posterior of the 13 goal frames matches the goal
   template 13 / 13, no ghost; the decoded prior draws at the four goal pairs match the goal 0 %, are ghosts 0.4 to
   7 %, and sit as far from the goal's template as any other state (median distance 10.6 to 11.7). No
   under-estimation of the goal by ghost classification.

**Reading (implementer, for the user; corrected 2026-10-02 (13)):** the rule's verdict is the general failure of the
prior on transitions seen fewer than ~9 times; within it, the four goal pairs are the lowest of their visit class
(percentile rank 0), with the goal frame the most distinctive image of the grid. Nothing is changed on this
basis (options 2-4 of (11) stay unimplemented).

### Tie-break: absolute tolerance from the protocol (user decision, implemented)

**Deviation from the spec (section 4.3 as amended in Phase 0: a relative tolerance, max(1e-6, 0.01 x std of Q over
the actions)), decided after seeing agent_mini.** Reason: with a relative tolerance, any non-zero spread of Q sets
its own threshold at 1 % of itself, so differences of the order of the critic's noise become strict preferences: in
agent_mini, where the agent had no value information (r_hat = -1 everywhere), the ties fell from 1 to 0.13 and the
policy became an arbitrary deterministic one, whereas E2's tabular agent ties exactly when it has no information and
then plays uniformly.

Required form: no value information (r_hat constant across actions, Q gaps of the order of the critic's noise) ->
uniform policy, as E2; the tolerance is absolute, in units of value, derived from the protocol (not from agent_mini's
gaps), proportional to the smallest Q gap that information at the truncation horizon produces.

Formula (heads.horizon_information_gap, heads.AgentConfig): a state d steps from the goal along a deterministic path
has V(d) = r_step (1 - gamma^(d-1)) / (1 - gamma) + gamma^(d-1) r_goal (E2: r_goal on the arriving step instead of the
step cost, goal absorbing). The smallest Q gap an informative action makes -- one step closer against an action that
leaves the distance unchanged -- in the one-step lookahead Q = r_hat + gamma c V is

    delta(d) = gamma (V(d-1) - V(d)) = ((1 - gamma) r_goal - r_step) gamma^(d-1);
    delta(36) = 6 x 0.9^35 = 0.1502;     tol = 0.5 x delta(H = 36) = **0.0751**.

The factor 0.5 keeps information at the horizon at twice the tolerance. `agent.tie_fraction: 0.5` in agent_mini.yaml
(replaces tie_tol_rel); the episode values (gamma, rewards, H = max_steps) come from the experiment's episode config.
The relative rule stays only to read the runs made with it (AgentConfig.tie_tol_rel, legacy).

Shape tests (`tests/agents/worldmodel/test_tie_break_shapes.py`, through the agent's own q_values with K = 16 draws
and act with epsilon 0.1; stub world model and heads):
(i) r_hat constant, V = constant + Gaussian noise of sd sigma on every evaluated latent, sigma = 0.005, 0.01, 0.02,
    0.03, 0.05: ties >= 99 % (observed 100 %) and actions uniform (chi-square, p > 0.001) -- passed;
(ii) a goal 20 steps away (gap 0.81), then 36 (gap 0.150), between the action one step closer and four that leave
    the distance unchanged, with and without noise sigma = 0.05: every greedy step picks the informative action,
    never as a tie -- passed.
Limit, stated: the noise in (i) is drawn per evaluated latent, so Q averages it over the K = 16 draws (sd 0.9 sigma /
4). Noise shared by the K draws of an action (a critic error that depends on h'_a only) is not averaged: then ties
are 99.7 % at sigma = 0.03 and 93.7 % at sigma = 0.05 (all five actions tied 72 % and 24 %). No absolute tolerance
can satisfy (i) for shared noise of sd 0.05 (it would need ~0.21) and (ii) at 36 steps (0.150) at once. In agent_mini
the spread of gamma c V over the actions was 0.024-0.027 (median), below the tolerance.

No agent run was made with the new tolerance.

### Weights at every checkpoint (engineering)

`run_agent.py` writes `checkpoints/agent_env<NNNNNN>.pt` (the same fields as agent_final.pt) at every checkpoint,
atomically; nothing is drawn. Checked: a tiny run with the previous run_agent.py and with this one (same legacy tie
rule) gives identical checkpoints.jsonl and final weights.

## 2026-10-02 (13) -- gradients against data: protocol and criterion (fixed BEFORE any computation; not launched)

Question (user): is the prior's poor mass on rarely visited transitions ((12): median 0.28-0.33 at n = 2-8) limited by
the number of gradient steps or by the data?

Protocol (user decision): start from the agent_mini world model, continue training the world model ALONE on the
frozen replay -- same loss, same learning rates and schedule, no new data, no agent heads -- for N more gradient steps;
then measure the s-space kernel in real contexts with the run's own settings (K = 1,358, ghost margin), as in (12).

Criterion (user decision), on the deterministic pairs off the goal with 2 <= n <= 8 (n = the frozen replay's visits):
- median mass of the prior on the right successor >= 0.5 after the N steps: **limited by the gradients**;
- rise of that median < 0.1 over its value at the start: **limited by the data**;
- otherwise: **mixed**.
Reported alongside, not part of the criterion: by zone, the L1 to P of the NOISY pairs before / after with their
visits (memorising a few samples must not pass for success: a pair seen n times whose kernel moves toward its n
empirical successors rather than toward P is not learning the dynamics).

Not launched: N and the starting point await the user's agreement (cost and the starting point's constraint in the
report of 2026-10-02).

### Goal pairs' percentile ranks (item 1 of 2026-10-02; descriptive, the rule's verdict is unchanged)

Mass on the right successor (the goal) of each pair into the goal, ranked among the deterministic pairs off the goal
with the same number of visits n (ties count half); `diagnose_world_model_goal.py --ranks`, output
`reports/diagnostics_wm_goal_ranks_<run>/ranks.json` (the 18,000 kernel recomputed with the same seeds: its rule median
0.2836 is the one of (12); saved as kernel_18000.npz).

| pair (s, a) | 18,000: n, mass, peers, rank, peers' median | 20,000: n, mass, peers, rank, peers' median |
|---|---|---|
| (1, 3) | 3, 0.000, 29, **0.0**, 0.15 | 3, 0.000, 25, **0.0**, 0.16 |
| (11, 1) | 7, 0.000, 18, **0.0**, 0.44 | 7, 0.001, 18, **0.0**, 0.35 |
| (12, 0) | 1, 0.000, 34, **1.5**, 0.14 | 8, 0.000, 16, **0.0**, 0.53 |
| (132, 2) | 2, 0.000, 40, **0.0**, 0.23 | 2, 0.000, 34, **0.0**, 0.23 |

At 20,000 no peer of any of the four classes has a mass below 0.01; the goal pairs are below all of them.

### Gradients against data: cost and starting point (proposal; nothing launched)

- Starting point: the weights at 20,000 have no optimiser state (agent_final.pt) and the replay at 20,000 was not
  saved (only the resume point's, at 18,000). The only consistent start is the **resume point at 18,000** (weights,
  Adam state, separation switch, learning-rate / beta_rep schedule -- constant there: lr 1e-4 after the switch, the
  representation warm-up over -- and its replay of 18,510 observations). Its baseline is already measured: median mass
  at n = 2-8 = **0.284** (kernel_18000.npz). Starting from the 20,000 weights would mean a fresh Adam state and a
  replay without the last 2,000 steps: proposed against.
- Cost on this CPU (4 threads, measured): world-model step 0.24 s in the run (0.29 s in an isolated benchmark); the
  kernel measurement 6.2 min (K = 1,358). N = 10,000: about 45 min; **N = 20,000 (proposed): about 1.5 h**, plus the
  measurement. Minibatches from the diagnostic's own stream; the run directory only read.
- Noisy pairs at the start (reported alongside): controllable noisy zone, 125 pairs, visits median 16 (10-90 %: 3-76),
  L1(world model, P) median 0.61; uncontrollable noisy zone, 125 pairs, visits median 5 (1-17), L1 median 0.97.
  Memorisation reference: the empirical successor frequencies of those visits have L1 0.20 to P (median, both zones)
  -- P puts 0.9 on one successor and 0.025 on each of four, so a pair whose few samples all fell on the main successor
  scores exactly 0.2. A kernel approaching 0.2 by collapsing onto the main successor (mass on the four minor successors
  -> 0 instead of 0.1) would be memorising: the mass on the minor successors is reported with the L1.

### Forecast for the closed-loop pilot (item 3; an estimate, not a decision)

Assumptions: behaviour = the uniform policy (the agent without value information, with the absolute tie tolerance of
(12)); 2 gradient steps per environment step, as agent_mini; the mass-versus-visits curve of (12) holds at any run
length (data-limited -- if the gradients-vs-data diagnostic says gradient-limited, this is pessimistic); the critic
and goal head follow once the prior predicts the goal (their own delay not modelled).
- Visits under the uniform policy (exact, from P: uniform starts over the non-terminal states, H = 36, goal absorbing,
  E[length] 34.97): each of the four goal pairs gets 3.1e-4 to 3.5e-4 visits per environment step, ~6.5 per 20,000
  steps (agent_mini's 2-8 visits agree).
- Mass versus visits ((12), median per visit class, 18,000 / 20,000): n 6-7: 0.49 / 0.50; 8-9: 0.60 / 0.54; 10-12:
  0.66 / 0.72; 13-15: 0.82 / 0.74; 16-20: 0.86 / 0.82; 21-25: 0.86 / 0.92; 31-40: 0.93 / 0.92; 41-60: 0.94 / 0.94.
- Goal pairs at n ~ 9 (mass ~0.5) after **~27,000** environment steps; at n ~ 20 (mass ~0.8-0.9) after **~60,000**;
  at n ~ 30 after ~90,000. For the information to reach states far from the goal, the pairs along the paths matter
  too: 90 % of the deterministic pairs reach n = 20 after ~94,000 steps and n = 30 after ~140,000.
- **Estimate: no improvement beyond the uniform policy before ~60,000 environment steps (near the goal); a general one
  around 100,000-150,000** (200,000-300,000 gradient steps). The goal pairs being at percentile rank 0 of their class,
  the goal may need more visits than the median pair: the estimate is optimistic on that side.

### Costs for the budget (measured on this CPU, 4 threads; agent_mini's configuration)

- Gradient step: world model 0.24 s + agent heads 0.25 s (run log: 0.48 s per environment step before the switch, 2
  world-model steps; 0.98 s after it); isolated benchmark 0.29 s + 0.31 s.
- Environment step (filter, one-step lookahead with K = 16 draws per action, epsilon-greedy): 0.006 s.
- Checkpoint measurements (twin B, agent Delta): negligible (< 2 s each in the run log).
- Final measurements: 15 min 16 s for a 20,000-step replay (file times of the run), of which the kernel 6.2 min; the
  replay-wide probes grow with the replay.
- 10^6 environment steps = 2 x 10^6 gradient steps: about 2 x 10^6 x 0.49 s = **~11 days per run on this CPU**, plus
  ~1.7 h of environment steps. The GPU throughput for this batch shape (8 x 16 windows, the bench network) is not
  measured; the preflight's bench would give it.

### Measurement code of the batch gpu_e30b_measures (item 4)

origin/main is at e0889bd (the batch commit); its RUN_GPU.md states the hash 99efffaf... / 261 files, which is that
commit's own tree (recomputed from `git archive e0889bd`). Since e0889bd, no file read by recompute_measures.py has
changed (its import closure: run_mini, probe, env, rng_streams, the src modules measures/*, kernel_probe, controllability,
agents/worldmodel/{rssm, loop, replay, losses, agent, acceleration}, envs/*, runs, config): the commits since touch
heads.py, policy.py, run_agent.py, agent_mini.yaml, the diagnostic scripts, docs and tests. The batch launched from
origin/main measures with the same code as this branch. The branch's RUN_GPU.md carries the branch's own hash (it
changes with every code commit); whichever tree main holds at launch, the preflight checks its RUN_GPU.md against it.

## 2026-10-02 (14) -- recorded BEFORE computing: gradients-vs-data launch settings; target-state frequency rule

The user accepts the correction of (12): the four goal pairs are at percentile rank 0 of their visit class, so the goal
carries a penalty on top of the general insufficiency; the verdict of (12) is unchanged.

### Gradients against data: approved and launched with these settings (protocol and criterion of (13) unchanged)

- Start: the resume point at environment step 18,000 (weights, Adam state, separation switch, schedule, replay of
  18,510 observations); the world model alone, trained N = 20,000 more gradient steps on that frozen replay, with the
  run's loss, learning rate and schedule (phase_learning_rate, rep_beta), the separation switch's collapse guard
  evaluated as in the run; minibatches and latent draws from the diagnostic's own streams. The run directory is only
  read; everything is written under reports/.
- Kernel measured (K = 1,358, the run's settings, the same measurement seeds as the baseline kernel_18000.npz) at
  N = 5,000, 10,000 and 20,000. **Only N = 20,000 decides the criterion of (13); N = 5,000 and 10,000 are descriptive.**
- Reported at each point: the median mass on the right successor over the deterministic pairs off the goal with
  2 <= n <= 8 (n = the visits at 18,000) and their number; by zone, the L1 to P of the noisy pairs and their mass on
  the four minor successors (0.1 under P); the four goal pairs' mass on the goal.

### Target-state frequency (read-only on kernel_18000.npz; reading rule fixed here, before computing)

Question: does the rarity of the goal as an observed state (13 occurrences in the replay at 18,000) explain the goal
pairs' penalty?
Group: the deterministic pairs off the goal with 2 <= n <= 8 whose successor state appears at most 20 times in the
replay (as an observation, at 18,000). If the group has fewer than 10 pairs, say so and do not apply the rule.
Rule: the group's median mass on the right successor
- <= 0.05: the frequency of the target state explains the goal's penalty;
- >= 0.2: there is another factor specific to the goal;
- in between: undetermined.
Reported alongside: the mass as a function of the successor's occurrences (classes 1-5, 6-20, 21-50, > 50), at fixed
pair visits n.

## 2026-10-02 (15) -- gpu_e30b_measures received: code version, CPUs, consistency, TAU, verdict, floor intervals

### Code and CPUs (user decisions)

- The archive (handoff/gpu_e30b_measures/, imported alone from origin/main; no merge of origin/main yet) was measured
  with source_sha256 **dae7ff2e...** (261 files), the tree of b887419 on main, not 99efffaf... (e0889bd). Accepted:
  b887419 changes only submit_kubeflow.py among the code (plus RUN_GPU.md, DECISIONS and two tests); no file read by
  recompute_measures.py differs from this branch. The merge of origin/main comes after the gradients-vs-data
  diagnostic; the supervisor's entry on main will then take the next free number here, marked "numbered (11) on main on
  2026-10-01"; our entries (11) to (13) are not renumbered.
- **Protocol deviation: 3 CPUs requested and usable instead of 5** (the notebook reserves 2 of the namespace's 5).
  No effect expected on the measures: made on the GPU, and the re-measured arrays are bit-identical to the originals
  (below). Consequence for planning: if the notebook keeps 2 CPUs, the cost estimates of the closed loop must assume 3
  usable CPUs, not 4.

### Consistency with the original gpu_e30b measurements (same GPU)

`tau_gpu_e30b.py` (report: reports/tau_gpu_e30b/TAU.md): the four runs measured from the expected weights, one code
hash, one device (A100-SXM4-80GB), TF32 off, K = 1,358. I_s, I_z (real contexts) and off-support mass, by run and
zone: **0 of 16 rows differ; the per-state arrays real_I_s, real_I_z and real_p_hat are bit-identical** (largest
difference 0). The re-measurement reproduces the stored values exactly.

### TAU (definitions of (9), B, unchanged; two seeds of different block classes A and B, an sd over n = 2 with one
### degree of freedom: a fragile estimate)

| zone | values over the pairs (seed 0, seed 5) | threshold |
|---|---|---|
| CN | b(I_s) - b(I_s projected): +0.0160, +0.0018 | **TAU_B = 0.0200** |
| CN | off-support env - ref: +0.0124, +0.0015 | **TAU_OFF = 0.0154** |
| CD | b(I_s) - b(I_s projected): -0.0121, +0.0080 | **TAU_B = 0.0285** |
| CD | off-support env - ref: -0.0015, +0.0009 | **TAU_OFF = 0.0035** |

No value is zero or aberrant; one value per pair and zone, so each threshold is sqrt(2) times the absolute difference
of two numbers. The limits raised in (10) apply as stated there (one degree of freedom; the verdict on these two pairs
is set by the relative position of the two values, so it is a description of gpu_e30b, not a test).

### Verdict of the rule on gpu_e30b (CN, CD) and the b table

| seed | zone | b (I_s) | b (I_s projected) | off-support env / ref | verdict |
|---|---|---|---|---|---|
| 0 | UD | -2.2394 | -2.2432 | 0.0472 / 0.0085 | read against the floor (floor run) |
| 0 | CN | +0.0084 | -0.0075 | 0.0498 / 0.0374 | interpretable |
| 0 | UN | -1.7921 | -1.7777 | 0.1573 / 0.0457 | read against the floor (floor run) |
| 0 | CD | -0.0028 | +0.0093 | 0.0048 / 0.0063 | interpretable |
| 5 | UD | -2.2681 | -2.2689 | 0.0596 / 0.0051 | read against the floor (floor run) |
| 5 | CN | -0.0040 | -0.0058 | 0.0431 / 0.0416 | interpretable |
| 5 | UN | -1.7504 | -1.7375 | 0.1421 / 0.0527 | read against the floor (floor run) |
| 5 | CD | -0.0090 | -0.0170 | 0.0079 / 0.0070 | interpretable |

CN and CD: the two versions of b agree within TAU_B and the off-support masses within TAU_OFF in both pairs
(descriptive on these pairs, see above).

### UN, UD: calibration interval of the environment-side off-support mass (condition of the E3-0 floor in closed loop)

| zone | E3-0 floor b(I_s) (seeds 0, 5; mean) | observed off-support (env) | tau_off_floor (implementer's reading, to be confirmed) | interval |
|---|---|---|---|---|
| UD | -2.2394, -2.2681; **-2.2538** | 0.0472 - 0.0596 | 0.0175 | **[0.0297, 0.0771]** |
| UN | -1.7921, -1.7504; **-1.7713** | 0.1421 - 0.1573 | 0.0214 | **[0.1207, 0.1787]** |

Saved as reports/tau_gpu_e30b/thresholds.json (aggregate_b.py's --thresholds format); not copied into configs/.

## 2026-10-02 (16) -- role of E3-0, aberration control, campaign rule for the bias, scope of E3, assumptions for E4 / E5

Recorded before any closed-loop result (user decisions).

1. **Role of E3-0.** E3-0 served to understand and control the measurement biases and to calibrate the world model.
   Its thresholds (TAU per zone, (15)) and its off-support intervals (UN, UD, (15)) are now **only an aberration
   control** for the closed loop; they are reused neither as thresholds nor as a floor. The floor and the reference of
   every measure and every bias are those of the final closed-loop campaign (N seeds, environment and reference).
   **E3-0 is not rerun**: the question of the goal-storage convention no longer arises, since floor and measures will
   come from the same runs. This replaces the "final floor rerun with the final campaign" of (7) and the "E3-0 if it is
   rerun" of (8)/(9) C (the single-execution-mode rule still holds for every run of a campaign).
2. **Aberration control.** A closed-loop value outside E3-0's ranges -- the off-support mass of UN / UD outside its
   interval, or in CN / CD |b(I_s) - b(I_s projected)| > TAU_B or |off-support(env) - off-support(ref)| > TAU_OFF -- is
   flagged in aggregate_b.py's output and must be explained before interpretation; it is not excluded automatically.
   E3-0's values are read from a versioned file, `configs/e3/e30_aberration_control.yaml` (the values of (15)).
3. **Campaign rule for the bias (hybrid codes), N a parameter** (N even, block classes A and B in equal numbers).
   Per controllable zone (CN, CD), over the N seeds: the mean of b(I_s) - b(I_s projected) and its 95 % confidence
   interval (Student t, N - 1 degrees of freedom); the same for off-support(env) - off-support(ref). b is interpretable
   in the zone if both intervals lie within +- FRACTION x REFERENCE EFFECT. **N, FRACTION and REFERENCE EFFECT are left
   unset; the user fixes them before the campaign.** Implemented in aggregate_b.py (campaign_rule) and tested; with
   the parameters unset, or with two seeds (the pilot), the rule is not applied and the differences are reported as a
   description.
4. **Scope of E3.** The campaign will probably have fewer than 20 seeds. With a small N, E3 serves as a reference and a
   validation, and can only detect effects of the representation clearly larger than the between-seed spread seen in
   calibration (0.01 to 0.015 bit in CN, of the order of E2's effect, -0.02 bit). **A non-significant result in a
   controllable zone will not be read as the absence of an effect.**
5. **E3-0's thresholds in aggregate_b.py**: no default values any more (the old 0.14 / 0.02 and the (9) derivation from
   E3-0 runs are gone from the command line); E3-0's thresholds and intervals are read from the versioned file of 2 and
   serve only the aberration control. The rule of (9) as applied to gpu_e30b in (15) is kept in tau_gpu_e30b.py, for
   the record of (15) only.
6. **Assumptions for what follows:**
   - the floor of the E3 campaign applies to the two-zone environments (E4, E5) only if the zone types and their
     visitations are comparable; to be checked then;
   - E4 serves to estimate the between-seed variance that sets the number of seeds of E5. For that: the smallest
     effect size aimed at in E5 is fixed before E4; the variance estimated is that of the paired quantity that will be
     E5's main measure; E4 includes the E5 condition expected to be the most variable, or a margin recorded in advance.

## 2026-10-02 (17) -- results: gradients against data, target-state frequency; GPU budget

### Gradients against data (protocol and criterion of (13), settings of (14); `gradients_vs_data.py`)

From the resume point at 18,000 (gradient step 35,488), the world model alone trained 20,000 more steps on the frozen
replay (6,646 s on this CPU, 4 threads; the collapse guard never fired). Kernel K = 1,358 with the baseline's
measurement seeds; outputs in reports/gradients_vs_data_<run>/ (result.json, kernels; the weights there are not
versioned).

| point | median mass, right successor, n = 2-8 (170 pairs) | CN noisy: L1 median / minor-successor mass median | UN noisy: L1 median / minor-successor mass median | goal pairs' mass on the goal |
|---|---|---|---|---|
| +0 (start) | 0.284 | 0.612 / 0.147 | 0.965 / 0.135 | 0, 0, 0, 0 |
| +5,000 | 0.359 | 0.415 / 0.124 | 0.774 / 0.114 | 0, 0, 0, 0 |
| +10,000 | 0.435 | 0.391 / 0.125 | 0.709 / 0.104 | 0, 0.001, 0, 0 |
| **+20,000** | **0.494** | 0.341 / 0.106 | 0.554 / 0.087 | 0.001, 0.022, 0, 0 |

(noisy pairs: 125 per zone, visits median 16 in CN, 5 in UN; true minor-successor mass 0.10.)

**Criterion (N = 20,000 only): rise 0.21 >= 0.1 and end 0.494 < 0.5 -> MIXED.** Descriptive: the median was still
rising at the last point (+0.076, +0.059 per 5,000-10,000 steps), not saturated. The noisy pairs' L1 falls toward P
while their mass on the four minor successors stays near the true 0.10 (CN 0.106; UN 0.087, mean 0.099): no collapse
onto the main successor, so the gain is not the memorisation signature defined in (13). The goal pairs stay at ~0
(largest 0.022): the extra gradient steps raise the rarely visited pairs in general but barely touch the goal.

### Target-state frequency (rule of (14); read-only on kernel_18000.npz)

The goal appears 13 times in the replay at 18,000. Group: deterministic pairs off the goal, 2 <= n <= 8, successor
seen <= 20 times: **49 pairs** (rule applicable). Median mass on the right successor **0.097 -> undetermined**
(between 0.05 and 0.2).
Mass by pair visits n and successor occurrences (median; pairs):

| pair visits n | successor seen 1-5 | 6-20 | 21-50 | > 50 |
|---|---|---|---|---|
| 0-1 | - | 0.092 (42) | 0.164 (21) | - |
| 2-3 | 0.007 (1) | 0.114 (26) | 0.180 (30) | 0.540 (12) |
| 4-5 | - | 0.103 (16) | 0.326 (24) | 0.622 (6) |
| 6-8 | - | 0.072 (6) | 0.480 (29) | 0.849 (20) |
| 9-20 | - | 0.142 (9) | 0.489 (30) | 0.866 (61) |
| > 20 | - | - | 0.191 (2) | 0.980 (126) |

At fixed pair visits, the successor's frequency has a large effect (n = 6-8: 0.07 / 0.48 / 0.85); pairs into rarely
seen states (6-20 occurrences) stay near 0.1 whatever their own visits. The goal pairs (0 to 0.001) sit below even
that group.

### Budget (user request; nothing launched)

GPU throughput for agent_mini's configuration is **not measured**: there is no preflight for run_agent.py, and the
acceleration bench (stacked seeds) never ran on a GPU. What exists (A100, gpu_e30b preflight, bridge network = the
bench network): world-model step 0.289 s single process for 16 x 64 windows (T = 64), 0.47 s per process with 8
processes for 64 x 16 (T = 16); final measures 511 s (recompute, 100,000-step replay). agent_mini's windows are 8 x 16
(T = 16); the step is launch-bound, so a single-process T = 16 step is estimated at ~0.07 s (scaled by T, not
measured). Not measured at all on the GPU: the agent heads' step (imagination, H = 15), the environment step, stacked
seeds. Missing to close the budget: a preflight bench of run_agent.py's three steps on the A100, alone and with
stacked seeds.

Run time = E x (environment step + ratio x (world-model step + heads step)). CPU: measured, 4 threads (0.24 + 0.25 s
per gradient step, 0.006 s per environment step); with the 3 usable CPUs of (15), expect up to ~4/3 longer. GPU: a
range, assuming heads step = world-model step, between ~0.07 s (estimated) and 0.289 s (measured at T = 64) each;
environment step taken as on the CPU.

| environment steps | ratio 1 | ratio 2 | ratio 4 |
|---|---|---|---|
| 200,000 | CPU 28 h; GPU 8-32 h | CPU 55 h; GPU 16-65 h | CPU 109 h; GPU 32-129 h |
| 300,000 | CPU 41 h; GPU 12-49 h | CPU 82 h; GPU 24-97 h | CPU 164 h; GPU 48-193 h |
| 1,000,000 | CPU 138 h; GPU 42-162 h | CPU 274 h; GPU 82-323 h | CPU 546 h; GPU 162-644 h |

With stacked seeds: no number (no runner, no GPU bench). Plus one final measurement per run (~8.5 min on the GPU,
~15 min or more on the CPU, growing with the replay).
**The forecast of (13) assumes ratio 2** (the mass-versus-visits curve was measured at ~2 gradient steps per
environment step); at another ratio the curve, hence the forecast, shifts -- and the gradients-vs-data result (mixed,
still rising) says more gradient steps per visit raise the mass. **For the goal it is a lower bound** on the number of
environment steps: the goal pairs are at percentile rank 0 of their visit class, their successor is among the
rarest states, and the extra gradient steps of this diagnostic barely moved them.

## 2026-10-01 (18) -- CPU capacity adjustment for gpu_e30b_measures (numbered (11) on main on 2026-10-01; renumbered at the merge of main into e3/worldmodel, 2026-10-02)

The user authorized running the measurement-only batch with 3 CPUs rather than the originally prescribed 5:
the notebook reserves 2 of the namespace's 5 CPUs, so the 5-CPU GPU job cannot be admitted alongside it.
This supersedes only the batch's CPU prescription in (10), not its four source runs, weights, measurement
configuration, parallelism (4), deterministic GPU settings, source-hash check, or preflight decision rule.
`BATCH_CPUS`, the runbook, and their tests now prescribe 3 requested and 3 usable CPUs; the preflight still
rejects any other allocation and any projected duration over 4 hours. The existing dirty cluster checkout
must be preserved separately before installing a clean copy of this version at the canonical path.

## 2026-10-02 (19) -- conclusion of the gradients-vs-data diagnostic; the closed-loop GPU pilot (recorded before its launch)

### Conclusion of (17) (user decision)

The verdict "mixed" is accepted. The extra gradient steps raise the rarely visited pairs in general, without
memorisation, but not the goal, whose limiting factor is the frequency of the target state (13 frames). Only a longer
closed-loop run produces more goal frames: **the pilot becomes the test of this hypothesis.**

### The pilot (batch gpu_pilot_cl; config configs/e3/gpu_pilot_closed_loop.yaml)

- Runs: seeds 0 and 5, environment and reference; the final configuration (absolute tie-break of (12), weights at
  every checkpoint, resume, final measures), **no acceleration** (no runtime.compile, no stacked seeds). agent_mini's
  network, procedure and agent; only the budget, runtime, seeds / conditions, resume interval (10,000) and the
  checkpoint goal diagnostics differ (tested).
- **Ratio 2 gradient steps per environment step.** Reason: the diagnostic of (17) shows that gradient steps matter;
  ratio 4 doubles the cost with no guarantee.
- Target length **300,000 environment steps** (E2's checkpoints up to 251,189, and 300,000); the runs can be read at
  their checkpoints before the end. With 3 CPUs, if 4 runs side by side are slower in total than 2, the preflight
  chooses 2 and the experimental runs go first, then the references.
- At every checkpoint, besides the twin, the agent's Delta under P and the success: the goal arrivals in the replay,
  the prior's mass on the goal for the four pairs into it, the median mass on the right successor of the pairs visited
  2 to 8 times (those pairs only, 256 draws each from real contexts; own measurement slot 15).
- Speed bench at the start of the batch, in the preflight, **informative only** (never in the verdict), ~30 minutes:
  world-model, heads and environment steps of run_agent.py in the pilot's configuration; the world-model step with
  1, 2, 4 and 8 stacked seeds, with and without torch.compile (the compiled variants only if the inductor equivalence
  gate passes on the node; otherwise its failure is reported and the bench continues without); per variant seed-steps
  per second, warm-up (compilation) time, loss difference from the plain step. For the decision on a multi-seed runner
  and the campaign budgets.
- **The pilot's runs will not be reused as campaign seeds if the campaign enables an acceleration** (the modes are not
  bit-equivalent, (9) C).

### Expected shapes (fixed before the launch)

1. The separation switch happens, with no collapse after it.
2. Before any value information: ties ~100 % and the actions played uniform.
3. **Goal criterion**, at the first checkpoint whose replay holds at least 100 goal arrivals, then at the end of the
   run: median prior mass on the goal over the four pairs into it
   - >= 0.5: **the data suffice**;
   - < 0.1: **a structural problem specific to the goal**;
   - in between: **undetermined**.
4. Pooled success indistinguishable from 0.0459 while the mass on the goal stays low, above it afterwards; an
   improvement without a rise of that mass must be explained.
5. The twin's L1 falls in all four zones; Delta >= 0.
6. Final measures: aberration control against configs/e3/e30_aberration_control.yaml; the campaign rule for the bias
   descriptive only (two seeds).

## 2026-10-02 (20) -- the pilot runs the configuration retained for E3 (win64), not agent_mini's (user decision)

Check before the push: configs/e3/gpu_pilot_closed_loop.yaml inherited agent_mini.yaml's training batch. Full diff of
the world-model and training hyperparameters against configs/e3/gpu_e30b_win64.yaml (model, render and every other
train key -- learning rates and schedule, latent estimator and switch, losses, clipping -- were already identical):

| key | pilot (as committed in c3a6b99) | gpu_e30b_win64 |
|---|---|---|
| train.batch_size | 8 | 16 |
| train.seq_len | 16 | 64 |
| train.replay_ratio | 2 (world-model updates per environment step, closed loop) | 0.6 (E3-0, offline) |
| measure.z_only_seq_len | absent (default 64) | 64 |
| measure.max_images_per_forward | absent (one-call encoding) | 2048 |
| budget, collection.epsilon, schedule keys, checkpoints, runtime | closed loop's own | E3-0's own |

Decision: **the pilot runs win64 -- B = 16 windows of T = 64 steps, and what depends on it (the real-context windows,
window_len null = T = 64; the z-only / h-only probe windows 64; the probes' encoder block 2048)** -- the rest of the
pilot's protocol of (19) unchanged. The identity test now takes gpu_e30b_win64.yaml as its reference (agent_mini.yaml
only for the agent, the twin and the logging). No technical constraint: run_agent.py ran with 16 x 64 windows (CPU
smoke, tiny network).

**Ratio 2 = 2 world-model updates per environment step = 2 x 16 x 64 = 2,048 replayed frames per environment step**
(agent_mini: 2 x 8 x 16 = 256; E3-0 win64: 0.6 x 1,024 = 614). **The gradients-vs-data diagnostic (17) and the
forecast of (13) were made in 8 x 16 (256 frames per environment step): they do not transpose as they are** -- eight
times more replayed frames per environment step here.

Projection (the preflight measures the real figure): world-model step 0.289 s (A100, one process, win64,
straight-through: gpu_e30b's preflight); the heads' step at this batch is not measured -- assumed equal to the
world-model step, the environment step negligible: ~2 x (0.289 + 0.289) = 1.16 s per environment step, i.e. **~97 h
(4 days) for 300,000 environment steps in one process** (48 h if the heads cost nothing). Several processes share the
GPU: in gpu_e30b, 8 processes cut the per-process rate from 3.46 to 1.01 steps/s; the preflight's 2- and 4-process
loops give the figure for this batch.
**100 goal arrivals**: under the uniform policy, 0.0459 successes per episode of 34.97 steps = 1.31e-3 arrivals per
environment step (agent_mini: 20 by 20,000) -> **~76,000 environment steps, ~25 h in one process** at 1.16 s (12 h
at the lower bound), more with several processes on the GPU.

## 2026-10-02 (21) -- the pilot takes win64's replay ratio, 0.6 (user decision, before any result of the pilot)

The pilot's replay ratio is the retained configuration's, **0.6** (gpu_e30b_win64), instead of 2. Reasons:
- the ratio 2 of (19) came from the diagnostic made in 8 x 16 windows (17);
- in replayed frames per environment step, win64 at 0.6 gives 0.6 x 16 x 64 = **614**, already above the diagnostic's
  level (~400 equivalent, where the median mass reached 0.49);
- the goal is limited by the data, not by the gradients ((17), (19)): the compute is better spent on environment
  steps;
- the cost is divided by 3.3 (2 / 0.6).
**Reservation:** at equal replayed frames, fewer optimiser steps (0.6 updates of 1,024 frames per environment step
instead of 2 of 256). If the median mass on the right successor of the pairs visited 2 to 8 times learns poorly at the
pilot's checkpoints, the ratio will be raised for the campaign. This decision was taken before any result of the pilot.

**Rate of the agent heads:** the same as the world model's. run_agent.py makes one gradient step per unit of replay
credit (0.6 per environment step, the fraction carried over: 3 steps every 5 environment steps), and each gradient step
after the switch updates the world model, then the goal head and the critic (in imagination from that batch's 1,024
posterior states) -- one heads update per world-model update, on the same batch. The bench's closed-loop loop
(bench_agent.py --loop) follows the same rule.

The rest of the protocol of (19) and the expected shapes are unchanged; (20)'s "2,048 replayed frames per environment
step" and its projection are superseded by the figures below.

Projection (world-model step 0.289 s, A100, one process, win64; the heads' step assumed equal, the environment step
negligible -> 0.6 x (0.289 + 0.289) = 0.35 s per environment step; lower bound with free heads 0.17 s):

| | environment steps | one process | sharing as observed in gpu_e30b (8 processes: 3.46 -> 1.01 steps/s, x 3.4) |
|---|---|---|---|
| 100 goal arrivals (uniform policy, 1.31e-3 per step) | ~76,000 | ~7 h (lower bound 4 h) | ~25 h (13 h) |
| end of run | 300,000 | ~29 h (lower bound 14 h) | ~99 h (50 h) |

The sharing factor is gpu_e30b's at 8 processes, pessimistic for the pilot's 2 or 4; the preflight measures the
pilot's own.

## 2026-10-05 (22) -- the pilot (gpu_pilot_cl) reviewed against (19); rules recorded before the next diagnostics

The four runs finished (300,000 environment steps, no resume, code 3a71c745...). The closed loop works; **b is not read
at this stage.** The expected shapes of (19) are not modified; recorded about them (user decisions):
- **The shape "success ~ 0.0459" was wrong for the reference**: its uniform rate is 0.1173 ((7)); both reference runs
  are compatible with it (seed 0: 0.135 [0.101, 0.178]; seed 5: 0.089 [0.062, 0.128]). The experimental runs matched
  0.0459 (0.046, 0.045).
- **The goal criterion was badly specified**: the checkpoints are too far apart to catch the passage of 100 arrivals
  (the first checkpoint past 100 already lies inside the transition), and the median over 4 pairs is not the statistic
  relevant to behaviour. **The verdict retained is the end-of-run one: the data suffice, in all four runs** (median
  mass on the goal 1.000).
- The measures that go through decoding are not readable before the switch (REF 0's mass 0.14-0.26 between 1,585 and
  3,981 steps, with the decoder still trained on probabilities).
- The n = 2-8 population empties because the data cover everything: **the reservation on the ratio 0.6 ((21)) is
  lifted.**
- **The off-support mass in UD and UN is well below E3-0's intervals** (UD 0.008-0.009 against [0.030, 0.077]; UN
  0.050-0.053 against [0.121, 0.179]): an aberration signal in the right direction, attributed to the much longer
  training (180,000 gradient steps against E3-0's 60,000), without exclusion. This corrects (7): the hybrid-codes
  bias falls clearly with training but does not disappear (C-b still fails in the noisy zones). The excess over
  TAU_OFF in CD (seed 5, 0.0048 > 0.0035) is the fragility anticipated in (16) (two seeds, one degree of freedom).
- **torch.compile is abandoned** (its equivalence gate failed on the A100: loss difference 9.2e-2).

### Rule for the per-pair goal masses (recorded before reading them)

For each run, at every checkpoint, the prior's mass on the goal for each of the 4 pairs into it (checkpoints.jsonl
keeps them per pair), with their maximum and median, against the success over the interval the checkpoint ends. At the
checkpoints that end an interval where the success clearly exceeds the uniform rate (0.0459 environment, 0.1173
reference; "clearly": the interval's 95 % Wilson interval lies above it):
- maximum >= 0.5: the maximum explains why success led the median;
- maximum < 0.1: unexplained -- where the information goes through (critic, imagination) must be looked for;
- in between: undetermined.

### Stability criterion for the campaign's budget (recorded before any re-measurement; [SEUIL] to be set by the user)

On the re-measurement of the pilot's checkpoint weights (63,096; 100,000; 158,489; 251,189; 300,000): **the first
checkpoint from which I_s by zone changes by less than [SEUIL] bit from one checkpoint to the next, in every run**
(4 runs, 4 zones). This criterion bears on the stability of the measures, never on the value of b.

## 2026-10-05 (23) -- per-pair goal masses (read with the rule of (22)); the checkpoint re-measurement batch (prepared, not launched)

### Per-pair goal masses against the success per interval (rule of (22))

checkpoints.jsonl keeps the mass of each of the four pairs (not only the median). Intervals whose success clearly
exceeds the uniform rate (Wilson 95 % above 0.0459 / 0.1173), their end checkpoint's maximum:

| run | interval (env steps) | success [95 %] | 4 pair masses at its end | max | rule |
|---|---|---|---|---|---|
| REF 0 | (10,000; 15,849] | 0.263 [0.205, 0.331] | 0, 0, 0, 0 | 0.000 | **< 0.1: unexplained** |
| REF 0 | (15,849; 25,119] | 0.472 [0.420, 0.524] | 0, 0.104, 0.063, 0 | 0.104 | undetermined |
| REF 0 | from (25,119; 39,811] on | 0.99-1.00 | >= 0.91 | 1.000 | explained |
| REF 5 | from (25,119; 39,811] on | 0.78-1.00 | >= 0.82 | >= 0.93 | explained |
| EXP 0 | (15,849; 25,119] | 0.703 [0.656, 0.746] | 0.361, 0, 0, 0.307 | 0.361 | undetermined |
| EXP 0 | from (25,119; 39,811] on | 0.99-1.00 | max >= 0.996 | 1.000 | explained |
| EXP 5 | (25,119; 39,811] | 0.070 [0.050, 0.098] | 0.020, 0, 0.008, 0 | 0.020 | **< 0.1: unexplained** |
| EXP 5 | from (39,811; 63,096] on | 0.85-1.00 | max >= 0.99 | 1.000 | explained |

Two intervals are unexplained (REF 0 up to 15,849; EXP 5 up to 39,811), two undetermined: there, the information
reached the behaviour without the prior putting mass on the goal at the interval's end; where it goes through (critic,
imagination) remains to be looked for. Limits: the mass is read at the interval's end only, from 256 draws per pair.

### The checkpoint re-measurement batch gpu_pilot_remeasure (prepared; not launched)

The complete final measures of the weights saved at 63,096, 100,000, 158,489, 251,189 and 300,000, for the four runs
(`remeasure_checkpoints.py`; batch in submit_kubeflow.MEASURE_BATCHES, kind "checkpoints"; preflight_measures.py;
make_handoff.py --measure-dir; RUN_GPU.md).
- **Real contexts when only the final replay is kept.** run_agent.py keeps no replay file; the only replay stored is
  the one pickled in each run's resume.pt, at the last resume point (290,000: resume_every 10,000, none at the end).
  The closed-loop replay is strict and chronological, so the replay at checkpoint c is exactly the prefix of its first
  c + episodes(c) observations (tested against a resume point's own replay). Each checkpoint's weights are measured on
  that prefix -- the contexts the run itself would have drawn at c, with the run's measurement streams -- and the
  counts N at c (counts.npz). **Acceptable for the stability question**: same estimator, same draws' streams, the
  data as of the checkpoint (contexts from a later replay would mix states the policy did not yet visit). The one
  exception is 300,000, after the last resume point: measured on the replay at 290,000 (flagged in its outputs); the
  run's own final measures (arrays.npz in the pilot's handoff) are the exact value there, and the pair gives a check.
- **Cost**: the pilot's final measures took about 11 minutes per run with 4 processes on the A100 (from the logs); 5
  checkpoints x 4 runs in 4 parallel jobs: about 1 hour, plus a preflight of about 30 minutes (one smoke measurement).
  Archives about 40 MB per run (arrays of 5 checkpoints).
- The stability criterion of (22) ([SEUIL] to be set by the user) is applied to its output; it bears on the stability
  of the measures, never on the value of b.

## 2026-10-05 (24) -- stability threshold and budget rule; value diagnostic (rule fixed before computing); stacked seeds abandoned

### Stability criterion of (22): [SEUIL] set (user decision, before the re-measurement is launched)

Per zone, **one third of the between-seed spread observed in calibration (gpu_e30b): 0.005 bit in CN and CD, 0.01 bit in
UD, 0.013 bit in UN.** **The campaign's budget is the first checkpoint from which I_s changes by less than this threshold
from one checkpoint to the next, in every zone and in all 4 runs** (on gpu_pilot_remeasure's checkpoints 63,096,
100,000, 158,489, 251,189, 300,000). **If no checkpoint satisfies it, the budget stays at 300,000 environment steps.**
The criterion bears on the stability of the measures, never on the value of b.

### Value diagnostic (added to gpu_pilot_remeasure as a fifth, read-only job; rule fixed before computing)

At REF 0's checkpoint 15,849 (the unexplained interval of (23)), and for comparison REF 0 at 10,000 and EXP 5 at 39,811
(`diagnose_value_generalisation.py`, weights and replay as of the checkpoint): Spearman correlation between V (the
critic on the posterior latents of the real contexts, by state) and the distance to the goal under P (expected steps
under the optimal policy), with its 95 % bootstrap interval over states; the share of greedy actions optimal under P
against a uniform choice's; the distribution of the actions played over the interval and the rate of repeating the
previous action.
Rule of reading (user decision; the operational thresholds in brackets are the implementer's, set here before any
value):
- Spearman clearly negative (V higher near the goal) [rho <= -0.3 with its 95 % interval below 0] together with null
  goal masses: **generalisation of V in the latent space**;
- Spearman ~ 0 [|rho| < 0.1 with its 95 % interval containing 0] together with directed or repeated actions [actions
  played not uniform, chi-square p < 0.001, or a repeat rate > 0.3; under the epsilon-greedy uniform policy it is ~0.2]:
  **wider coverage without information on the goal**;
- otherwise: undetermined.

### Stacked seeds abandoned for the campaign (user decision)

Estimated gain x1.0 to 1.5 over the present 4 processes (2026-10-05 (23)) for a large piece of work. **Planning base:
the measured total throughput of 4 processes on the A100, ~8.9 environment steps per second** (2.23 per process, the
pilot's configuration, ratio 0.6).

## 2026-10-06 (25) -- gpu_pilot_remeasure: the value diagnostic's failure and fix; results (stability, 300,000 control, value diagnostic)

Archive: `handoff/gpu_pilot_remeasure/` (commit 1142454 on main: the four re-measurement archives, handoff.md with its
`## Stop` and `## Delivery adjustment` sections, delivery_provenance.json, logs_summary.tgz, value_diagnostics_retry.tgz;
the full 186 MB logs.tgz stays on the cluster, SHA-256 eca2b610...).

### What happened, and the correction (made after a failure, by the supervisor, on the cluster)

- The four re-measurement jobs ran with the code of ff01253 (source_sha256 84d41d62..., recorded in each of their 20
  outputs), 17:04 to 18:11 UTC on 2026-10-05, exit code 0.
- The fifth job (value diagnostic) started at 18:02:29 and failed 8 s later, on its **first point, REF seed 0 at
  10,000 environment steps**: `np.quantile` of an empty list in `spearman_with_ci`. Cause: **the critic is constant**
  there (V = 0 for every state) -- **expected**: REF 0 switched at environment step 10,758 (gradient step 6,301), the
  agent's heads train only from the switch, and their output layers start at zero (heads.py), so before the switch V is
  exactly 0; the correlation is undefined and every bootstrap resample gave NaN. The point was chosen as a pre-switch
  comparison without anticipating this; the script had no test for a constant critic.
- The job was not retried within the batch (RUN_GPU.md rule 2). The handoff's logs.tgz (186 MB) exceeded the 90 MB
  limit (make_handoff.py archived the measurement directory with its run outputs -- fixed below).
- **Correction by the supervisor (Veith Weilnhammer, with Copilot), on the cluster, after the failure: commit 90c5465
  (2026-10-05 18:41 UTC, after the four re-measurements had finished)**, then a separate run of the value diagnostic
  alone with that code (`logs/kubeflow/value-diagnostics-retry-20261005-b`, exit 0). The diff touches only
  `diagnose_value_generalisation.py` (+17 lines) and its test file (+34 lines; checked: no other file changed between
  ff01253 and 1142454 outside handoff/):
  - `spearman_with_ci` raises on fewer than 2 states or non-finite values; returns rho and its interval as NaN with
    `undefined_reason` ("constant critic values" / "constant goal distances") when an input is constant; skips the
    bootstrap resamples where an input is constant (the original computed them and dropped the NaN); returns NaN
    bounds with "no valid bootstrap correlations" when no resample is usable.
  - **Effect on the results: none wherever the original did not fail** -- same random draws (rng.integers is called
    before the skip), same retained resamples (a constant resample gave NaN, dropped by the original), same values;
    checked against a copy of the original on non-constant inputs, including small tied samples where resamples are
    often constant (tests/e3/test_e3_value_diagnostics.py). The constant case now reports "undefined", which under the
    rule of (24) is "undetermined" and, before the switch, means "no value information". The three results of the
    retry are the ones read below; the two later points never ran with the original code but would have given the
    same values.
- Integrated as is in e3/worldmodel (fast-forwarded from main), with additions (2026-10-06): a test with a real
  pre-switch checkpoint (constant critic, every greedy choice tied); the diagnostic's output now records its code
  version (`reproducibility`) -- the retry's output has none, its provenance is in provenance.txt (commit 90c5465);
  make_handoff.py leaves the measurement directory's run outputs and the preflight's smoke measurement out of logs.tgz.

### Stability criterion (24): no checkpoint satisfies it -> **the campaign's budget stays at 300,000 environment steps**

|I_s(next) - I_s(this)| by zone, the largest over the 4 runs (threshold: CN, CD 0.005; UD 0.010; UN 0.013):

| from -> to | UD | CN | UN | CD |
|---|---|---|---|---|
| 63,096 -> 100,000 | 0.136 | 0.242 | 0.289 | 0.063 |
| 100,000 -> 158,489 | 0.030 | 0.110 | 0.065 | 0.037 |
| 158,489 -> 251,189 | 0.039 | 0.084 | 0.028 | 0.019 |
| 251,189 -> 300,000 | **0.012** | **0.019** | **0.021** | 0.002 |

Even the last step exceeds the threshold in CN (3 runs: 0.007-0.019), UN (REF 0 0.015, REF 5 0.021), UD (REF 0 0.012)
while CD is under it in every run (at most 0.002). I_s is still rising in the reference runs' noisy zones and in CN.

### Control at 300,000: re-measurement (replay of 290,000) against the pilot's own final measures (replay of 300,000)

Differences re-measured - final, by zone (the final measure's bootstrap se in brackets):

| run | UD | CN | UN | CD |
|---|---|---|---|---|
| EXP 0 | -0.0004 (0.0003) | +0.0206 (0.0030) | -0.0006 (0.0007) | +0.0009 (0.0008) |
| EXP 5 | -0.0005 (0.0003) | -0.0132 (0.0029) | -0.0008 (0.0006) | +0.0048 (0.0006) |
| REF 0 | +0.0052 (0.0014) | +0.0263 (0.0030) | -0.0073 (0.0030) | -0.0014 (0.0008) |
| REF 5 | -0.0052 (0.0010) | -0.0031 (0.0028) | +0.0001 (0.0033) | -0.0039 (0.0009) |

Same weights, same estimator and streams; only the replay differs by its last 10,000 steps (hence different real
contexts). The differences reach 4 to 9 bootstrap se in CN (up to 0.026 bit): **the choice of real contexts moves I_s by
about 0.01-0.03 bit in CN, more than the bootstrap se and more than the stability threshold (0.005).** Recorded as a
fact; it bears on how the stability criterion can be read (the threshold is below this measurement variation).

### Value diagnostic (rule of (24))

| point | goal masses | Spearman V vs distance [95 %] | greedy optimal / uniform | actions played (chi2 p; repeat) | rule |
|---|---|---|---|---|---|
| REF 0 at 10,000 (before the switch) | 0.004-0.012 | undefined: constant critic (V = 0) | 0.319 / 0.319 (all tied) | uniform (p 0.29; 0.196) | undetermined (no value information, as expected) |
| **REF 0 at 15,849** | 0, 0, 0, 0 | **-0.563 [-0.669, -0.440]** | 0.450 / 0.319 | not uniform (p 2e-17; 0.209) | **generalisation of V in the latent space** |
| EXP 5 at 39,811 | 0.020, 0, 0.008, 0 | -0.132 [-0.288, +0.042] | 0.598 / 0.543 | not uniform (p 8e-4; 0.202) | undetermined (|rho| >= 0.1: not "~ 0"; not <= -0.3) |

At REF 0's unexplained interval of (23), the critic already ranks states by their distance to the goal (V higher near
the goal) while the prior puts no mass on the goal: the information reached the policy through V's generalisation
in the latent space. EXP 5's unexplained interval stays undetermined under the rule.
**Corrected 2026-10-06 (26), after the result:** the label "generalisation of V in the latent space" does not hold --
the critic is trained on imagined trajectories only (see (26)); the reading retained is a mass on the goal below the
measurement's resolution, amplified by the +50 / -1 asymmetry.

## 2026-10-06 (26) -- budget 300,000 by the rule; the critic is trained on imagination only (correction of (25)); measurement-noise batch: rule fixed before computing

### Budget (user decision)

The campaign's budget is **300,000 environment steps**, by application of the rule of (24). Recorded: the threshold
of 0.005 bit in CN was below the measurement's variation due to the choice of real contexts (0.01 to 0.03 bit, the
300,000 control of (25)), so **the criterion was unattainable by construction; it is not modified after the fact.**

### What the critic is trained on (code check)

`src/helplessness/agents/worldmodel/imagination.py`:
- `train_heads_step` (lines 97-124): the goal head is trained on the replay batch's posterior states against the REAL
  arrival rewards, `goal_head_loss(goal_head, h, z, reward)` (line 112: BCE against 1[r > 0]); the critic is trained
  on `rollout = imagine(rssm, goal_head, critic, h, z, cfg, generator)` (line 116) and
  `critic_loss(critic, slow_critic, goal_head, rollout, cfg)` (line 118).
- `imagine` (lines 36-52): from the replay's posterior states (stop-gradient), `cfg.horizon` steps of the prior
  (`rssm.imagination_step`, line 48), actions from the agent's own epsilon-greedy policy.
- `critic_loss` (lines 66-78): r and c of every imagined state from the goal head, `r, c = reward_from_goal(p, cfg),
  continuation_from_goal(p)` with `p = goal_head.p_goal(h, z)` (lines 71-72); lambda-returns on the slow critic
  (line 73); weights cumprod(c) (line 74).
**The critic receives no target from real transitions** (no real reward, no real termination): only lambda-returns of
imagined trajectories, with r_hat and c predicted by the goal head on imagined (prior) latents. Its only contact with
the replay is the start state s_0.

**Correction of the reading of (25) at REF 0 / 15,849, made after the result.** "V higher near the goal with null goal
masses" means: **a mass on the goal below the measurement's resolution (256 draws per pair: a mass under ~1/256 reads
as 0) guides the agent, amplified by the +50 / -1 asymmetry** (r_hat = 50 p - (1 - p): p = 0.01 already gives r_hat
~ -0.49 against -1, and the critic propagates it over the horizon) -- **not a generalisation of the critic.**
Mechanistic reason: the critic cannot learn the value of the goal unless the goal has a non-zero probability under
the prior on imagined trajectories, since that is its only source of reward.

### Measurement-noise batch (prepared, not launched): rule of choice of the measurement protocol (fixed before computing)

a. On the final weights of the pilot's 4 runs (agent_final.pt, replay of the last resume point, 290,000), I_s by zone
   measured with R = 8 independent draws of real contexts (and of the kernel's prior draws), at the present
   configuration (real_context.max_per_state 50) and with 2x and 4x more contexts per state (100, 200); the kernel
   keeps K = 1,358 draws per (s, a), spread over the state's contexts. Reported: the standard deviation of I_s by
   zone over the 8 repeats, by number of contexts, and the cost of each configuration.
b. Goal control on REF 0 at 15,849: the mass on each of the 4 pairs into the goal with 10,000 draws; at the 4 states
   next to the goal, the Q gap between the action into the goal and the best other action, against the tie-break
   tolerance (0.0751).
**Rule (user decision; the cost bound in brackets is the implementer's, set here):** the measurement protocol is the
smallest number of contexts per state for which the standard deviation of I_s due to the contexts is <= 0.005 bit in
CN and in CD (in each of the 4 runs). If none reaches it at a reasonable cost [one I_s measurement costing at most 3
times the present configuration's, measured in this batch], the largest number tested is taken and the resolution
obtained is recorded. **This protocol will be identical for every run of the campaign.**

## 2026-10-06 (27) -- budget criterion corrected (after the result, before the measurement noise is known); candidate protocols at the checkpoints

### The budget criterion, corrected (user decision)

Made **after the result** of (25), and recorded **before the launch of gpu_pilot_noise, hence before the measurement
noise is known**. Reason: the criterion of (23)/(24) was unattainable by construction (its threshold was below the
variation due to the choice of contexts), so the 300,000 steps it gave do not reflect a need of the model. The
correction bears only on the measurement noise, never on b, and **its result will be accepted even if it gives
300,000 again.**

New criterion: the budget is the first checkpoint (among 63,096, 100,000, 158,489, 251,189, 300,000) from which, in all
4 runs of the pilot:
a. **the agent has finished learning**: median mass over the four pairs into the goal >= 0.9; the success of the next
   interval compatible with that of the last interval (their 95 % Wilson intervals overlap); **the agent's Delta under
   P below 1.0 in every zone** (threshold proposed by the implementer from the scale of Delta at the end of the pilot:
   0 to 0.75 by zone in the 4 runs at 300,000 (2026-10-05 review), against ~19 to 36 for the uniform policy; 1.0 is
   the cost of one environment step, the unit of the episode's reward; Delta = mean over the zone's states of
   V*(s) - V^pi(s) with the agent's greedy policy read on real contexts, checkpoints.jsonl agent_delta.real_contexts);
b. **I_s changes from one checkpoint to the next by less than 2 times the standard deviation due to the contexts**
   (task a of gpu_pilot_noise), by zone, measured with the protocol retained by the rule of (26).
If no checkpoint satisfies both, the budget stays at 300,000.
**Transparency:** the quantities of a (goal masses, success per interval, Delta at every checkpoint) were already read
and reported in the pilot's review (2026-10-05 (22), (23)); the threshold on Delta is proposed after having seen them.
Only b depends on measurements not yet made.

### Candidate protocols at the checkpoints (in gpu_pilot_noise)

Each of the four noise jobs, after its 24 repeats, measures I_s by zone at the 5 checkpoints with 2 and then 4 times
the contexts per state (`measure_noise.py --checkpoint-steps`; the weights at each checkpoint, the replay as of it --
the 300,000 point on the replay at 290,000, flagged --; one measurement per checkpoint and multiplier, the same seed
at every checkpoint). The x1 values exist (gpu_pilot_remeasure, (25)). **Only the measurements at the protocol retained
by the rule of (26) serve criterion b; the others are set aside.**
Cost: one I_s measurement (real contexts + kernel, K = 1,358) on the A100 is not measured yet; the preflight times one
at 4 times the contexts (an upper bound for every measurement of the batch) and projects each job as 24 repeats + 10
checkpoint measurements (5 x2 + 5 x4) of that duration, 4 jobs side by side (x 1.25). **If the whole exceeds the 8-hour
ceiling, the preflight's decision is to launch gpu_pilot_noise_x2: the same batch with the checkpoints at x2 only
(priority to x2).**

## 2026-10-07 (28) -- gpu_pilot_noise read: measurement noise, protocol (rule of (26)), budget (criterion of (27)), goal control

Archive: `handoff/gpu_pilot_noise/` (commit b084f01 on main). No rule is modified here; results only.

### Provenance

- Preflight `preflight-20261006T184406428649Z`: passed, decision **launch** (the complete variant, not the x2
  fallback), projected 2.32 h (fallback 1.98 h), CPUs requested / usable 3 / 3, code source_sha256
  c984cdd6...a31477 (271 files) as stated in RUN_GPU.md, A100, TF32 off, deterministic on.
- Measurement `measure-20261006T184805129601Z`, batch gpu_pilot_noise: the 4 noise jobs 18:48 -> 21:12-21:13 UTC
  (about 2 h 24 each), goal_control 21:12:14 -> 21:12:29; **all 5 exit code 0**; every output records the same
  source_sha256.
- On the cluster: no measurement relaunched or corrected. **The handoff step was run twice**: a first call of
  make_handoff.py with `.venv-cluster/bin/python` instead of the runbook's `python` failed at `import yaml` before
  writing anything; the runbook's command then succeeded with the same arguments (recorded by the supervisor in
  handoff.md, `## Stop`).
- Remark (no effect on any decision): for this batch kind the preflight log line "smoke: 6692 s" prints the projected
  seconds per job (one x4 measurement, 196.8 s, x 34 measurements), not the smoke's duration.

### Task a: standard deviation of I_s due to the contexts (8 repeats; final weights, replay of 290,000)

sd by zone (bit). Every single measurement took 251-261 s, **whatever the number of contexts** (mean per run and
multiplier 254-257 s): the cost is that of the kernel (K = 1,358 draws per (s, a)), not of the contexts.

| run | x | contexts | UD | CN | UN | CD | s / measurement |
|---|---|---|---|---|---|---|---|
| EXP 0 | 1 | 7,198 | 0.0006 | 0.0116 | 0.0009 | 0.0039 | 254 |
| EXP 0 | 2 | 14,204 | 0.0008 | 0.0131 | 0.0009 | 0.0033 | 254 |
| EXP 0 | 4 | 27,640 | 0.0005 | 0.0059 | 0.0004 | 0.0026 | 254 |
| EXP 5 | 1 | 7,118 | 0.0014 | 0.0089 | 0.0015 | 0.0017 | 254 |
| EXP 5 | 2 | 14,087 | 0.0007 | 0.0051 | 0.0007 | 0.0023 | 254 |
| EXP 5 | 4 | 27,630 | 0.0016 | 0.0058 | 0.0008 | 0.0023 | 255 |
| REF 0 | 1 | 7,200 | 0.0044 | 0.0099 | 0.0161 | 0.0045 | 256 |
| REF 0 | 2 | 14,400 | 0.0072 | 0.0067 | 0.0099 | 0.0033 | 256 |
| REF 0 | 4 | 28,732 | 0.0024 | 0.0060 | 0.0050 | 0.0010 | 257 |
| REF 5 | 1 | 7,200 | 0.0023 | 0.0116 | 0.0084 | 0.0050 | 255 |
| REF 5 | 2 | 14,400 | 0.0031 | 0.0079 | 0.0101 | 0.0033 | 256 |
| REF 5 | 4 | 28,800 | 0.0026 | 0.0038 | 0.0045 | 0.0022 | 256 |

In CN the sd goes from ~0.010 (x1) to ~0.0054 (x4), less than the factor 2 expected from the contexts alone (part of
the noise, e.g. the kernel's prior draws, does not decrease with the contexts). An sd over 8 repeats has a relative
error of about 25 %.

### Protocol (rule of (26), applied as is): **x4, real_context.max_per_state 200**

No multiplier gives sd <= 0.005 bit in CN in all 4 runs (x4: EXP 0 0.0059, EXP 5 0.0058, REF 0 0.0060, REF 5 0.0038;
CD is <= 0.005 at every multiplier). The cost bound is not binding (x4 costs 1.0 time the present measurement).
Hence the largest number tested, x4. **Resolution obtained (sd due to the contexts at x4): CN 0.0038-0.0060, CD
0.0010-0.0026, UD 0.0005-0.0026, UN 0.0004-0.0050 bit.**

### Budget (criterion of (27), applied as is): no checkpoint satisfies b -> **the budget stays at 300,000**

Point a (checkpoints.jsonl; success per interval with 95 % Wilson intervals; Delta = agent_delta.real_contexts,
largest zone):

| run | median goal mass at 63,096 / 100,000 / 158,489 / 251,189 / 300,000 | next interval vs last interval | max Delta by zone, same checkpoints | a |
|---|---|---|---|---|
| REF 0 | 0.998 / 1 / 1 / 1 / 1 | all 1.000, overlap | 0.49 / 0.30 / 0.37 / 0.25 / 0.32 | holds at all |
| REF 5 | 1 / 1 / 1 / 1 / 1 | all 1.000, overlap | **1.16** / 0.87 / 0.69 / 0.51 / 0.42 | fails at 63,096, holds from 100,000 |
| EXP 0 | 0.996 / 1 / 1 / 1 / 1 | 0.999-1.000, overlap | **6.69** / 0.54 / 0.62 / 0.32 / 0.25 | fails at 63,096, holds from 100,000 |
| EXP 5 | **0.537** / 0.971 / 1 / 1 / 1 | 0.997-1.000, overlap | **18.09** / 0.62 / 0.55 / 0.54 / 0.75 | fails at 63,096, holds from 100,000 |

Point b (x4 measurements; |I_s(next) - I_s(this)| / 2 x sd of task a at x4, same run and zone, in bit; failures in
bold):

| run | 63,096 -> 100,000 | 100,000 -> 158,489 | 158,489 -> 251,189 | 251,189 -> 300,000 (replay of 290,000) |
|---|---|---|---|---|
| REF 0 | fails in 4 zones | fails in 4 zones | fails in 4 zones | **UD 0.0160 / 0.0048, UN 0.0181 / 0.0099**; CN 0.0081 / 0.0120, CD 0.0019 / 0.0020 |
| REF 5 | fails in 4 zones | fails in 4 zones | fails in 4 zones | **UD 0.0059 / 0.0051, CN 0.0165 / 0.0076**; UN 0.0064 / 0.0091, CD 0.0003 / 0.0044 |
| EXP 0 | fails in 4 zones | fails in 4 zones | fails in 4 zones | **UN 0.0014 / 0.0008**; UD 0.0004 / 0.0010, CN 0.0028 / 0.0118, CD 0.0033 / 0.0053 |
| EXP 5 | fails in 4 zones | fails in 4 zones | fails in 4 zones | **UD 0.0059 / 0.0031, UN 0.0052 / 0.0016**; CN 0.0075 / 0.0117, CD 0.0003 / 0.0047 |

251,189 is the only checkpoint where a holds in all 4 runs and b comes close; b fails there in every run (2 zones in
3 runs, 1 in EXP 0). **Flagged:** the 300,000 point is measured on the replay at 290,000 (the last resume point),
with the final weights; the 2 x sd threshold is the sd measured on the final weights, assumed to hold at the earlier
checkpoints. The x2 measurements are set aside, as recorded in (27).

### Task b: goal control, REF 0 at 15,849 (10,000 draws per pair, 50 contexts per state; tie tolerance 0.0751)

| state | action into the goal | mass on the goal | Q gap (into the goal - best other) | gap / tolerance | tied set (within the tolerance of the max) |
|---|---|---|---|---|---|
| 1 | 3 | 0.00010 | -0.0518 | -0.69 | actions 0, 1, 3 (the goal's among 3) |
| 11 | 1 | 0.00081 | +0.0132 | +0.18 | actions 0, 1, 2 (the goal's is the max, among 3) |
| 12 | 0 | 0 (0 / 10,000) | +0.0263 | +0.35 | actions 0, 1 (the goal's is the max, among 2) |
| 132 | 2 | 0.00010 | -0.0675 | -0.90 | actions 0, 1, 2, 3 (the goal's among 4) |

- **Mass below the resolution: confirmed.** All four masses are below 1/256 (0 to 0.0008); the 256-draw reading
  gave 0.
- **Amplification by the +50 / -1 asymmetry: not shown at the goal's neighbours.** No Q gap reaches the tolerance;
  the action into the goal is the strict maximum in 2 states of 4 and below the best other action in the other 2.
  The largest positive gap (+0.026) is at the state whose mass is exactly 0 (state 12); at the largest mass (0.0008,
  state 11) the expected one-step gain, 51 p ~ 0.04, is not visible beyond the tolerance (+0.013). The action into the
  goal is in the tied set at all 4 states (greedy probability 1/4 to 1/2 against 1/5): a weak push whose origin these
  data do not identify. **The corrected reading of (25) is only partly supported**: the mass part is confirmed, the
  mechanism (amplification guiding the agent) is not.

### Between-seed dispersion and measurement noise (x1, CN)

sd due to the contexts at x1 in CN: 0.0099, 0.0116, 0.0116, 0.0089 (root mean square 0.0106 bit per I_s
measurement). Against the calibration's between-seed spread of 0.01 to 0.015 bit in CN (gpu_e30b): **measurement
noise alone accounts for 50 % (0.0106^2 / 0.015^2) to all (0.0106^2 / 0.010^2 = 1.1) of that variance**; for a
difference of two measures (b) the noise alone is sqrt(2) x 0.0106 = 0.015 bit. Limits: the calibration spread has one
degree of freedom (two seeds), and the noise was measured on the closed-loop pilot's weights, not on gpu_e30b's.

### Stop after the analysis

The final batch is not prepared: the goal control gives an ambiguous answer to the reading of (25) (above), which
the instruction of 2026-10-07 makes a reason to stop and report before part 2.

## 2026-10-07 (29) -- the campaign's final batch (gpu_campaign): protocol x4 and budget 300,000 retained; design, measures and analyses recorded before the launch

### Decisions on (28) (user, 2026-10-07)

- **The goal control (28) does not block the final batch**: it bears on the learning dynamics of one run (REF 0, the
  interval up to 15,849), not on the campaign's measures, protocol or budget. Recorded: **the mechanism of REF 0's early
  success is not identified.** The goal masses are confirmed below the resolution; the amplification by the reward
  asymmetry is not demonstrated (Q gaps below the tolerance, the action into the goal in the tied set). Candidate
  explanations: a weak push at the last step through the tied set, combined with a navigation guided by V upstream; or a
  correlation of V with the distance to the goal due to a spatially structured artefact of the critic (the distance to
  the goal is structured by zones), which does not show that V carries information on the goal. **Question left open,
  to revisit if needed with the campaign's data** (V by zone and the Q gaps next to the goal are recorded at every
  checkpoint for that purpose, below).
- **Measurement protocol x4 (real_context.max_per_state 200) and budget 300,000 environment steps: retained**, by
  application of the rules of (26) and (27).

### The batch (user decision; implemented, not launched)

- Seeds: **N = 4, seeds 1 and 2 (class A), 6 and 7 (class B)**, each in the environment and its reference: 8 runs,
  queued, at most 4 at a time (submit_kubeflow.AGENT_BATCHES["gpu_campaign"]; jobs in pairs, A and B alternating --
  seeds 1, 6, then 2, 7 -- so each round of 4 holds two complete pairs). The pilot's runs are not reused.
- Configuration `configs/e3/gpu_campaign.yaml`: the pilot's (gpu_pilot_closed_loop.yaml) identical for training,
  agent, environment and budget (299,744 + 256 = 300,000 environment steps), with the protocol (max_per_state 200) and
  the measures below; tested (tests/e3/test_e3_campaign_config.py). The measures fork the global torch RNG and restore
  the model's mode: a run with them trains exactly as without them (tested), and a resumed run gives the same outputs
  (tested).

### Rule for the hybrid codes (user decision)

aggregate_b.campaign_rule with **N = 4, FRACTION = 1/2, REFERENCE EFFECT = 0.02 bit** (E2's effect in CN): per
controllable zone, b is interpretable if the 95 % Student-t intervals (3 degrees of freedom) of b(I_s) - b(I_s projected)
and of the off-support difference both lie within +- 0.01 bit. **With N = 4 the rule may declare zones not
interpretable; that will be reported as it is.**

### Measures during the runs (run_agent.py; protocol x4)

- **Full measures** (run_mini.measure_final: I_s, I_s projected, off-support mass, I_z, kernels, probes) **at every
  checkpoint of the E2 grid after the switch**, into `measures/env<NNNNNN>/`; when fewer than 2 grid checkpoints fall
  within the 50,000 environment steps after the switch, extra checkpoints at the switch + 25,000 (then + 50,000) until
  there are 2 (run_agent.extra_checkpoints). With the pilot's switches (environment steps 8,700 to 11,300) that is 8 or
  9 checkpoints (10,000 or 15,849, 25,119, 39,811, 63,096, 100,000, 158,489, 251,189, 300,000), 3 of them within 50,000
  steps after the switch; whatever the switch (the separation deadline puts it before ~41,900), at least 6, at least 2
  in the window, and **the budget's checkpoint, 300,000, whose measures are the run's final measures**.
- **At every checkpoint** (the 14 of the grid and any extra one): the counts N[s, a, s'] (counts.npz); the tabular
  twin's kernel T_hat and its per-state I_hat, I_ref, b, Delta (twin.npz); the prior's mass on the four pairs into the
  goal with **10,000 draws** each (the pairs visited 2-8 times keep 256); at the four states next to the goal, the
  agent's Q and the gap between the action into the goal and the best other action, against the tolerance 0.0751; **V
  averaged by state and by zone** (the critic on the contexts' posterior latents); the zones' separation in the latent
  (below). All in checkpoints.jsonl (goal_diagnostics).
- Light arrays: the measures' arrays leave out the per-context latent distributions (real_prior_probs,
  real_post_probs: 37 MB per measurement at 200 contexts per state), reproducible from the weights and replay handed
  off.

### Analyses (recorded before the launch; campaign_analysis.py)

**Main:** b by zone on I_s at the budget's checkpoint (300,000), with the hybrid-code rule above and E3-0's aberration
control (configs/e3/e30_aberration_control.yaml: flags to be explained, never exclusions).

**Secondary, at every checkpoint measured in both runs of a pair** (operational definitions are the implementer's, set
here before any campaign value):
- **b by visit class**: a state's visits are those of its least-visited action, n(s) = min_a N[s, a] (its I_s rests on
  its 5 kernel rows, the least-visited limiting it); classes 0-9, 10-29, 30-99, 100-299, 300-999, 1000+; in each zone
  and class, the mean I_s over the environment run's states of that class minus the mean over the reference run's
  states of that class (each run classes its own states), with the number of states on each side.
- **Difference of differences**, by zone: [I_WM - I_twin]_env - [I_WM - I_twin]_ref, I_WM the zone's I_s and I_twin
  the twin's I on the same run's counts at the checkpoint.
- **alpha, the weight of an action-independent component in the predicted kernel**
  (measures.kernel_probe.action_independent_weight): per state, the least-squares fit of
  p_hat(.|s, a) = (1 - alpha) P(.|s, a) + alpha m(.), m shared by every action; the shared part drops out of the action
  contrasts, so 1 - alpha = sum_a <p_hat_a - mean p_hat, P_a - mean P> / sum_a |P_a - mean P|^2 (exact for a mixture,
  whatever m; not clipped; undefined where P has no action contrast, i.e. the uncontrollable zones of the environment).
  Zone means, environment, reference and their difference; read in the controllable zones.
- **The zones' separation in the latent** (probe.latent_zone_separation, on h and z concatenated, the real contexts of
  the goal diagnostics): held-out accuracy of a linear probe latent -> zone (4 categories) and latent -> controllable vs
  uncontrollable category (in the reference every zone is controllable: there it separates appearances only, a
  control); the distance between the centroids of the controllable and uncontrollable categories in units of the
  within-category RMS distance, on standardised features (diluted by uninformative dimensions: read between
  environment and reference and across checkpoints, not as an absolute).
- From the records: V by zone and the Q gaps next to the goal, reported without a rule (for the open question above).

**Expectation, recorded before:** in the controllable zones, b ~ 0 and alpha ~ 0 at the budget, compatible with
visually distinct zones; a transient early effect is possible. **A null result will be reported as a bound (no effect
larger than the resolution obtained), not as the absence of an effect.**

**Tested on the pilot's archives** (gpu_pilot_cl runs, gpu_pilot_remeasure measures at x1, 63,096 to 300,000; the
300,000 point on the replay at 290,000): the script runs end to end. Descriptive values, not campaign results:
- alpha (environment / reference), CN: seed 0, 0.237 / 0.290 at 63,096, 0.000 / 0.040 at 300,000; seed 5, 0.101 /
  0.114, then -0.003 / -0.019. CD: 0.054 / 0.037 to 0.009 / 0.010 (seed 0), 0.054 / 0.053 to 0.004 / 0.013 (seed 5).
  UD and UN of the environment: undefined, as defined.
- Difference of differences at 300,000, CN: +0.105 (seed 0), -0.056 (seed 5); CD -0.008, -0.003.
- Separation: the pilot's archives hold z only (real_post_probs), not h, so the probe ran on z: zone and
  controllability probes at 1.000 at every checkpoint, centroid ratio 0.39-0.45 (environment) and 0.33-0.35
  (reference). In the campaign it is computed in the run on h and z.
- b by visit class: at 300,000 most states fall in the classes 10-29 and 30-99 (the least-visited action is visited
  through epsilon); the cells with few states on one side are listed with their counts.
- The hybrid-code rule: "not applied: two seeds (pilot)", as designed.

### Handoff and what can be recomputed afterwards

Per run (measured on the pilot's network and arrays): the weights, 2.6 MB per checkpoint x 15 = **~40 MB**; the final
replay, compact (states, actions, rewards, flags; replay_final.npz), **< 1 MB**; the measures at 8-9 checkpoints,
**~1 MB each** (light arrays) plus a figure; records, counts, twin kernels and final measures **~3 MB**. About **52 MB
per run, ~420 MB for the 8**, every archive under 90 MB: **repatriable, included in the handoff**
(make_handoff.py --with-weights: <run>.tgz, <run>.measures.tgz, <run>.weights.tgz; never resume.pt). With the weights
at every checkpoint, the whole replay and the counts, every measure and analysis can be recomputed off the cluster
(a GPU is preferable for the kernels; a recomputation on another device may differ in the last digits -- the cluster's
values stay the reference). **What cannot be recomputed:** the continuation of training (optimiser states, environment
and stream states: resume.pt stays on the volume), and the per-step training scalars that are not logged.

### Duration, preflight, runbook

- Projection: ~37 h of training per run (2.23 environment steps per second per process with 4 on the GPU, (23)), plus
  ~5 h of measures (9 full measurements of an estimated 20-40 min at x4, and the goal diagnostics), i.e. ~42 h per
  run, **two rounds: ~3.5 days, plus a preflight of ~1 h**. The preflight measures it: on the pilot's REF seed 0 final
  weights (read only), the goal diagnostics and one full measurement at x4, timed; the projection adds 9 full
  measurements and 15 goal diagnostics per run (x 1.25 for contention) to the training time at the measured rate. No
  speed bench (measured for the pilot).
- RUN_GPU.md: preflight, queue of the 8 runs, measures, handoff with the weights, duration; **no relaunch and no code
  correction without telling the user** (a resume after an interruption only with his agreement).
- Fixed: preflight_measures.py printed the projection per job as the smoke's duration ("smoke: 6692 s", (28)); it now
  prints both (smoke_line, tested).

## 2026-10-07 (30) -- the pilot's runs reused as campaign seeds; the difference of differences examined (before the launch)

### Decisions (user, 2026-10-07)

- The operational definitions of (29) are validated: the visit class of a state (its least-visited action) and alpha
  (a mixture with a component shared by every action).
- **Seeds: N = 4. Seeds 0 (class A) and 5 (class B): the pilot's runs (gpu_pilot_cl), measured with the campaign's
  protocol** (x4, every measure of (29) at every checkpoint, the zone probe on (h, z) included), from the weights and
  replay saved on the cluster. **Seeds 1 (A) and 6 (B): new runs**, environment and reference: 4 runs in one round.
  The pilot runs' handoff includes their weights at the checkpoints, as the new runs'.
- **Recorded: the pilot's runs served to fix the budget** (the stability criterion of (24) and the corrected criterion
  of (27), on I_s's stability and the agent's learning -- never on b) **and are reused as campaign seeds.**
  Transparency: their b by zone has been seen (the pilot's review (22), the x1 re-measurement (25) and the test of the
  analyses on their archives, (29)); the analyses and the hybrid-code rule were fixed in (16) and (29), the rule's
  parameters by the user before the campaign.

### Check before anything else: the training of the new runs is the pilot's

- The pilot's code: source_sha256 3a71c745... (its 4 runs' meta.yaml), the tree of commits 002ee50, be79c75 and 720593c
  (recomputed from `git archive`). From 720593c to the branch, in the hash's scope (src, experiments, configs,
  pyproject.toml): no change to src/ except the new measure measures.kernel_probe.action_independent_weight; none to
  run_mini.py, env.py, rng_streams.py, config.yaml, the E2 twin, gpu_pilot_closed_loop.yaml, gpu_e30b_win64.yaml; in
  run_agent.py only the campaign's additions of (29), each behind its config key; the rest are measurement, preflight,
  handoff and analysis scripts. **No difference touches the training.**
- Tested on the CPU (tiny configuration, tests/e3/test_e3_campaign_measure_pilot.py): a run trained with the pilot's
  keys and a run trained with the campaign's keys end with the same weights; the first, measured afterwards by
  campaign_measure_pilot.py, gives at every checkpoint up to its last resume point exactly what the second measured
  during its training (goal diagnostics, full measures' arrays, twin kernels). The measures of (29) do not change the
  training (test of (29)).
- **The checkpoint grid of the new runs is the pilot's** (checkpoints.grid "e2", resume_every 10,000; tested in
  test_e3_campaign_config.py, campaign_measure_pilot.py refuses a run whose grid or training blocks differ).
- Not covered by the hash: the cluster's Python environment (the image's torch, .venv-cluster); each run records its
  own in meta.yaml.

### The pilot's runs as campaign seeds (campaign_measure_pilot.py)

Read only on the run; written into the job's directory laid out as a run (config.yaml with the campaign's measure, twin
and checkpoints blocks, checkpoints.jsonl, counts.npz, twin.npz, measures/env<NNNNNN>/, results.json and arrays.npz,
replay_final.npz, the weights). At every checkpoint (14): the twin (checked identical to the pilot's own record) and
the goal diagnostics of (29); at every checkpoint after the switch (environment steps 8,758 to 11,342: 8 checkpoints,
9 for EXP 5), the full measures at x4 -- the same rule as the new runs, no extra checkpoint needed. Same streams keyed by
the checkpoint as in a campaign run. **Flagged: the pilot's last checkpoint, 300,000 (the budget's), is measured on the
replay of its last resume point, 290,000** (no later replay was saved): its contexts miss the last 10,000 steps; the
order of the effect is that of the context noise ((25): 0.01-0.03 bit in CN at x1, from 290,000 vs 300,000 replays).
replay_final.npz of these runs is that replay. The new runs have their exact final replay.

### The difference of differences (asked before the launch)

On the pilot's archives (x1), CN, did = [I_WM - I_twin]_env - [I_WM - I_twin]_ref = b_WM - b_twin:

| seed | checkpoint | b_WM | b_twin | did | I_WM - I_true env / ref | I_twin - I_true env / ref |
|---|---|---|---|---|---|---|
| 0 | 100,000 | +0.067 | +0.008 | +0.059 | +0.072 / +0.005 | -0.052 / -0.060 |
| 0 | 300,000 | +0.109 | +0.004 | +0.105 | +0.248 / +0.139 | -0.025 / -0.029 |
| 5 | 100,000 | -0.046 | -0.027 | -0.019 | +0.146 / +0.192 | -0.044 / -0.016 |
| 5 | 300,000 | -0.091 | -0.034 | -0.056 | +0.208 / +0.298 | -0.037 / -0.003 |

- **Where the variability comes from: the world model's own excess in CN, not the twin.** I_WM is above the true
  kernel's I by +0.14 to +0.30 bit at 300,000 (the CN kernel predicted sharper than the truth, as in gpu_e30b, (4)); the
  excess grows with training (100,000 -> 300,000) and with the state's visits (Spearman with the least-visited action's
  visits +0.20 to +0.88 over the 25 CN states), and its level differs from run to run. The twin is within -0.06 to 0 of
  the truth and almost the same in both runs (b_twin -0.03 to +0.01). So did ~ b_WM: it varies between seeds as b
  itself does in CN (+0.109 / -0.091 at 300,000).
- Not the twin's counts in little-visited contexts: at 300,000 every CN state's least-visited action has 25 or more
  visits, and restricted to states with 50 or more, the WM's excess is as large (+0.23 to +0.34). Not the weighting:
  both terms are unweighted means over the same 25 states. Not the support: every CN state is measured in every run.
- **Its measurement noise:** that of b_WM, the contexts' (the twin is deterministic given the counts): sqrt(2) x 0.0054
  ~ 0.008 bit in CN at x4 (0.015 at x1, (28)). The twin adds the statistical uncertainty of its counts, ~0.013-0.016 bit
  per run on the CN zone mean (Dirichlet posterior, 100 draws), ~0.02 on b_twin. Both are well below the observed
  between-seed spread (~0.16): did is not too noisy to measure; it varies between runs.
- **Decision recorded (implementer's proposal, before the launch):** no restriction would make it more robust -- the
  variation is the world model's run-to-run behaviour, which is what the quantity contrasts with the count-based
  estimator, and it does not shrink on well-visited states. **The definition is kept; the difference of differences is
  reported as DESCRIPTIVE, with its parts** (b_WM, b_twin, and each run's I_WM - I_true and I_twin - I_true, by zone;
  campaign_analysis.py). Note for the main analysis: the same between-seed variation of b in CN, with N = 4, will be
  read through the hybrid-code rule and the interval over seeds, as decided.

### Batch, duration, handoff

- gpu_campaign: the 4 new runs (seeds 1, 6; environment, reference), then the 4 pilot measurement jobs queued after them
  (at most 4 at a time). The preflight also checks that each pilot run has what the jobs read (weights at every
  checkpoint, resume.pt, counts, console.log with the switch) and projects the queue: **~2 days** (preflight ~1 h; the
  4 runs ~42 h; the pilot jobs ~6 h: 8-9 full measurements and 14 goal diagnostics each).
- Handoff: three archives per run, new or pilot (records and final measures, checkpoint measures, weights and replay):
  ~52 MB per run, ~420 MB in total, every archive under 90 MB.

## 2026-10-13 (31) -- the analysis pipeline of gpu_campaign: definitions, expected shapes and rules recorded before any campaign value

Recorded while gpu_campaign runs: **handoff/gpu_campaign is not in this repository at this commit and no value of the
campaign has been read.** The pipeline was tested on the pilot's archives (gpu_pilot_cl runs, gpu_pilot_remeasure
measures at x1, twin kernels recomputed by run_agent.twin_at_checkpoint) and on a tiny environment / reference pair;
only that the code runs was checked, no value was read for this entry. Everything below is descriptive (N = 4); a null
result is reported as a bound (the largest end of the 95 % interval), never as an absence. Items marked **[to validate]**
are the implementer's proposals on what is measured or how: they stand unless the user changes them before the data.

One command (`campaign_report.py --handoff handoff/gpu_campaign`): integrity, extraction, analyses, RSA, figures,
tables, and a results section in English (RESULTS_SECTION.md; `--append-report docs/E3_worldmodel_4zones.md` appends
it between markers, refused outside the handoff mode). `--runs-dir` (development) marks every output "not the
campaign's handoff".

### A. Integrity (campaign_integrity.py; engineering)

For each of the 8 jobs (seeds 1, 6 new; seeds 0, 5 the pilot measured as campaign seeds): started, exit code 0,
meta.yaml status done, results.json, 300,000 environment steps; its three archives; inside them: the code hash
d84dec84... (RUN_GPU.md) and, for the pilot jobs, the pilot's training code 3a71c745... ((30)); checkpoints.jsonl
covering E2's grid to 300,000 with the campaign's goal diagnostics (goal_draws, Q gaps, V by zone, separation) at every
checkpoint; counts.npz and twin.npz at every checkpoint; the full measures at every checkpoint after the switch and only
there; the weights at every checkpoint, agent_final.pt and replay_final.npz; both conditions of each seed. **Any failure
refuses the whole handoff** (nothing extracted, nothing analysed). Interruptions (meta.yaml resumed_at, "RESUMING" in the
log, a `## Interruptions` section of handoff.md) are reported, not refused (a resume with the user's agreement is
allowed, RUN_GPU.md 4b). The pilot jobs' final replay at 290,000 is flagged in the integrity table.

### B. Learning and behaviour (campaign_learning.py; descriptive, no rule)

Per run, zone (UD, CN, UN, CD, corridor) and checkpoint. At every checkpoint: the twin's L1 against P (mean over the
zone's (s, a)); success over the last 100 episodes and over the interval since the previous checkpoint (95 % Wilson),
episode length (last 100); the agent's Delta under P by zone (real contexts; the four zones as recorded -- the corridor
is not recorded by the run); the prior's mass on the goal for the 4 pairs into it and their median (10,000 draws); the Q
gap (into the goal - best other) at the 4 states next to the goal against the tolerance 0.0751; V by zone. At every
measured checkpoint: C-a, C-c, C-b by zone (aggregate_b's definitions, the corridor added); the prior's mean mass on the
true successor over the zone's deterministic (s, a) pairs (real-context kernel).
**The goal state is left out of every per-state analysis and of the corridor's zone mean** (terminal: no transition out
of it, I_true = 0); the four quadrant zones keep the measures' own zone means.

### C. Measurement error in the noisy zones (descriptive)

excess = I_s - I_true (zone means, I_true from the run's own kernel) in CN and UN, per run and measured checkpoint; over
the seeds, at the checkpoints measured in every pair: mean and sd of the excess per condition, mean and sd of b, and
**the resolution an extension to the noisy zones would have, defined as the half-width of b's 95 % Student-t interval
over the N seeds, t(0.975, N - 1) sd(b) / sqrt(N)** [to validate]; the hybrid-code rule's verdict of (29).

### D. Propagation (campaign_analysis.py)

D1, D2: the main analysis of (29) and b by zone at every checkpoint, unchanged; the corridor added (its states' mean).
Over the seeds, at the checkpoints measured in every pair: mean and 95 % t interval of b and of alpha_env - alpha_ref.
**Only alpha_env - alpha_ref is interpreted as a propagation; alpha itself is not.**

D3, b at equal visits [to validate]. States of the zones controllable in both runs (CN, CD, corridor; goal out). For each
run and state, over its measured checkpoints c: n(s, c) = min_a N_c[s, a] and I_s(s, c); I_s and log10 c interpolated
linearly in log10 n (n >= 1; checkpoints with the same n averaged). On 8 log-spaced n within the range both runs cover
(a state without overlap is excluded and counted): d(s, n) = I_env(s, n) - I_ref(s, n), dtau(s, n) = log10 c_env(s, n) -
log10 c_ref(s, n) (the training time at which each run reached n visits). Per pair and zone, least squares
d = b0 + b1 dtau + b2 (log10 n - mean): **b0 is b at equal visits and equal training time**, b1 the part due to the
difference in training time. The points of one state are not independent: the seeds are the replicates (mean and 95 % t
interval of b0, b1 over the 4 seeds).

D5, b against the position. delta(s): torus (Manhattan) distance from s to the nearest state of an uncontrollable zone of
the environment (UD or UN), the same geometry for the reference's states. Per pair and checkpoint, over a zone's states:
Spearman rho of |b(s)| with delta(s), and of each run's |I_s(s) - I_true(s)| with delta(s); mean |b| by distance. Main
zone CD; CN and corridor secondary. **Expected shape of a propagation by position (recorded before): |b| larger near
the boundary, decreasing with the distance, in the environment only, rather early after the switch.** Rule at a
checkpoint, over the seeds [thresholds to validate]: shape **present** if the 95 % t intervals of rho(|b|, delta) and of
rho(env error, delta) both lie below 0 while that of rho(ref error, delta) contains 0; **absent** if the interval of
rho(|b|, delta) contains 0; otherwise undetermined. "Rather early" is read from the checkpoints where it is present.

D6, useful generalisation [to validate]. **"Per context" is read as per (s, a) pair, the world model's kernel pooled
over the state's real contexts** (the per-context kernels are not kept: light arrays, (29); they could be recomputed on a
GPU from the weights). Per run and measured checkpoint, L1 of the world model's kernel and of the twin's T_hat_c against
P, by n(s, a) = N_c[s, a] (bins 0, 1-2, 3-5, 6-10, 11-30, 31-100, 101-300, 301+): medians and the median paired
difference L1_WM - L1_twin. Rule per zone, condition and checkpoint, on the median difference over the pairs with
n <= 10: **the model generalises if it is negative in every seed**, does not if >= 0 in every seed, mixed otherwise; not
testable if a run has fewer than 5 such pairs (expected at the late checkpoints).

**Expectation (recorded before the computation):** in the controllable zones, b ~ 0 and alpha_env - alpha_ref ~ 0 at
the budget ("compatible with 0": the 95 % t interval over the seeds contains 0; the bound is then reported); a
transient early effect is possible (the earlier checkpoints whose interval excludes 0 are listed).

### E. Convergence of the deterministic zones: the budget of the next runs (rule fixed before any computation)

Candidates: the checkpoints measured in all 8 runs, before 300,000. A candidate c is retained if, in every run:
(a) **the agent has finished learning ((27) a), at c and at every later candidate**: median goal mass >= 0.9; the success
over the interval after the checkpoint compatible with the last interval's (251,189 -> 300,000; 95 % Wilson intervals
overlap); the agent's Delta < 1.0 in every zone;
(b) **I_s of UD, CD and the corridor does not move beyond the measurement noise**: for each zone Z, every difference
between consecutive candidates from c to 300,000, and the cumulative difference I_s(300,000) - I_s(c), is below
T_Z = k sqrt(2) sigma_Z.
- sigma_Z: the sd of a zone's I_s due to the contexts at x4 ((28) task a, the pilot's final weights): **one value per
  zone, the root mean square over the 4 pilot runs** -- UD 0.0020, CD 0.0021 bit -- for all 8 runs (the per-run values
  have ~25 % relative error and the new runs were not measured). **The corridor was never measured: CD's sigma scaled
  by sqrt(25 / 43), its 43 non-terminal states against CD's 25 (sd of a zone mean ~ 1 / sqrt(states)): 0.0016 bit**
  [to validate: an assumption].
- **Correction for multiple comparisons (Bonferroni; the criterion of (27) had none):** k = Phi^-1(1 - 0.05 / (2 m)),
  m = 8 runs x 3 zones x the number of differences tested for c (consecutive ones plus the cumulative one). Under no
  change and Gaussian noise of sd sigma_Z per measurement, the probability that some difference exceeds its threshold
  is at most 5 %. (With 2 x sd and no correction, (27)'s threshold was 1.41 sd of a difference: ~16 % false "change" per
  comparison, so a stable model failed it almost surely.) The cumulative difference guards against a slow drift that
  passes every step.
The first retained candidate is the budget of the following runs (deterministic zones only); **none: the budget stays at
300,000.** Flagged: the pilot's 300,000 measure rests on the replay at 290,000.

### F. RSA of the latent (src/helplessness/measures/rsa.py, generic; experiments/E3_worldmodel_4zones/rsa_latents.py)

Generic module, arrays in and out, no environment or agent import (CLAUDE.md), for E3 and the next steps (two-zone
models on permuted themes and positions; links between the latent's organisation and behavioural phenotypes).
- Contexts: the real contexts of the replay as of the checkpoint (window 64, burn-in 8, **200 per state** -- the measures'
  x4 protocol, at no extra cost: the windows' unroll dominates), own measurement slot 16 keyed by the checkpoint; per
  context h_t and the posterior probabilities q(z_t | h_t, x_t).
- Points: every state but the goal, and every zone (UD, CN, UN, CD, corridor: the mean of its states' means; position:
  the zone's circular centre on the torus).
- Distances: L2 on the mean h; **Jensen-Shannon on z = the sum over the 8 categorical variables of the JS divergence
  between the points' mean distributions (bits)** [to validate].
- Theoretical matrices: torus Manhattan distance (position); different theme (zone background colour); different
  controllability; different stochasticity (both from the run's own true kernel). The zone is not a predictor (in E3
  its matrix is the theme's), only the within-group label of the ratios; in the reference controllability is constant
  (left out of its regression and reported).
- Regression per latent, checkpoint and run: z-scored off-diagonal entries, standardised weights, R^2, each matrix's
  unique share (R^2 lost without it); the split-half reliability of the model matrix (Spearman-Brown), the ceiling of R^2.
  At the zone level (5 points, 10 pairs) the regression is reported, not interpreted.
- Scalars: the confusion ratios -- mean distance between states of controllable and uncontrollable zones over the mean
  distance between distinct states of one zone (the four quadrant zones; also CD vs UD and CN vs UN); in the reference
  the same zones by appearance (a control: appearance only). Over the seeds: env, ref and env - ref.
- Visualisation: classical MDS of the states' mean latents, coloured by zone, theme and controllability.
- Reading: descriptive in E3, no rule.
- Cost on this CPU (4 threads, win64 network, measured on gpu_e30b_win64 seed 0's final weights with its 100,000-step
  replay): ~18 s per checkpoint at 200 contexts per state (17 s at 50).

### G. Engineering (implementer)

- `tests/e3/test_e3_run_gpu_doc.py`: the equality between RUN_GPU.md's hash and the checkout is no longer required
  while RUN_GPU.md records the launched gpu_campaign's code (d84dec84...): the analysis code added since changes the
  checkout's hash, and RUN_GPU.md stays the record of what runs. To be restored with the next batch's runbook.
- campaign_analysis: alpha and b now include the corridor (b_I_s_all); the measures are loaded once per checkpoint.

## 2026-10-13 (32) -- the [to validate] items of (31) decided (user); RSA collinearity; the runbook test (before any campaign value)

Still before any value of the campaign (handoff/gpu_campaign not in the repository).

1. **Corridor noise, sigma_CD x sqrt(25 / 43): accepted as an approximation** (the states' context noise assumed
   independent). Recorded: it makes the convergence criterion of (31) E stricter (a smaller threshold), hence prudent.
2. **One sigma per zone, the root mean square over the pilot's 4 runs: accepted.**
3. **"Per context" = per (s, a) pair, the kernel pooled over the state's real contexts: accepted.**
4. **The thresholds of (31) D5 and D6: accepted as descriptive.** The results section states that "negative in all 4
   seeds" is equivalent to a (one-sided) sign test at p = 1/16.
5. **JS on z = the sum over the 8 variables of the per-variable JS divergences: accepted**, stated in the section as
   not the JS divergence of the joint distribution.
6. **RSA predictors: accepted, with two additions (implemented):**
   - the regression reports the predictors' collinearity (pairwise correlations of the theoretical matrices, variance
     inflation factors) and a partition of the explained variance (each predictor's unique share, the shared share, the
     unexplained share); the section reports them and **no weight is read in isolation**. **In E3 theme, position and
     zone are confounded (each theme occupies one fixed region of the torus): E3 cannot separate theme from position
     -- the object of the next step** (two-zone models on permuted themes and positions);
   - in the reference runs the "same controllability" predictor is constant: **removed explicitly before the
     regression** (rsa_latents.predictors_for, recorded as removed_predictors in each output), no longer left to the
     regression's guard; tested (a constant predictor is reported and not estimated; the tiny reference run's outputs
     have controllability removed and absent from its regression).
7. **The relaxation of tests/e3/test_e3_run_gpu_doc.py ((31) G) is recorded as a user decision**: while RUN_GPU.md
   records the launched gpu_campaign's code (d84dec84...), the equality between its hash and the checkout's is not
   required. **Instruction: restore the requirement with the next batch's runbook** (the next RUN_GPU.md states the
   hash of the code it launches, and the test checks it again).
   The complete test suite (run after this entry: 923 passed, 1 failed) showed a second instance of the same
   requirement, tests/e3/test_e3_preflight_measures.py::test_code_hash_is_run_gpu_md_s: the same rule applied to it
   (skipped while RUN_GPU.md states d84dec84...), to be restored with the first.

## 2026-10-14 (33) -- correction made AFTER a result: the RSA confusion ratio did not follow the definition of (31) F

First run of campaign_report.py on handoff/gpu_campaign (integrity passed; pipeline complete). In its results the
reference's confusion ratio was NaN at every checkpoint. Cause, an implementation error: the ratio grouped the states by
the run's own controllability (corridor included; in the reference every state is controllable, so the "uncontrollable"
group was empty), whereas (31) F defines it on **the four quadrant zones grouped by appearance** (CN, CD vs UD, UN; the
corridor in neither group), the same zones in the reference (the control). **The definition is unchanged; the code is
corrected to follow it** (rsa_latents.RATIOS, attribute appearance_group; test: the ratio is finite and both groups are
non-empty in the reference too). Recorded as a correction made after seeing a result. The environment's ratio changes
too (the corridor no longer counts in the controllable group). The CD vs UD and CN vs UN ratios, the regressions and every
other quantity are unaffected. The pipeline is run again in full; only the second run is reported.

## 2026-10-14 (34) -- reading correction on the early CN transient (user); seed 1 exploratory diagnostic: reading rule recorded before computing (post hoc)

### Correction of the reading of the campaign report (user)

The early transient in CN (b = +0.136 [+0.091, +0.180] at 39,811; alpha_env - alpha_ref negative at 25,119 and 39,811)
was reported as "the transient early effect anticipated". **Corrected: it goes in the direction opposite to a
propagation of uncontrollability (which would give b < 0); it does not correspond to the anticipated effect.** With b
at equal visits ~ 0 in CN (b0 = -0.000 [-0.198, +0.197]), the transient is compatible with a visitation effect (CN more
visited and better learnt early in the environment), not with an effect of the representation. The visits of CN per
seed, environment and reference, at 25,119 and 39,811 are reported with this entry's diagnostic. No rule and no recorded
result is modified.

### Seed 1 (environment), CD: exploratory, post hoc diagnostic (read only)

Recorded as **exploratory and post hoc**: seed 1's b(CD) (-0.27 at 300,000) has been seen. The reading rule (user;
the operational thresholds in brackets are the implementer's, set here before computing anything of the diagnostic):
- **"helplessness" signature**: in seed 1's CD, alpha (the action-independent component,
  measures.kernel_probe.action_independent_weight) clearly larger than in seed 1's reference and than in the other
  seeds' environments [at 300,000: alpha(seed 1 env, CD) minus the largest of alpha(seed 1 ref, CD) and alpha(seed s env,
  CD) for s = 0, 5, 6 > 0.05], **and** errors concentrated near the boundary with an uncontrollable zone [Spearman rho,
  over CD's 25 states at 300,000, between the error |I_s(s) - I_true(s)| and the distance delta(s) of (31) D5 <= -0.3];
- **"failed learning" signature**: alpha comparable to the other seeds [the difference above <= 0.05], **and** errors
  either spread without relation to the boundary [|rho(error, delta)| < 0.3] or concentrated on little-visited states
  [rho(error, n(s)) <= -0.3, n(s) the visits of the state's least-visited action at 300,000];
- otherwise: **undetermined**.
Reported besides, without rule: alpha in CD per checkpoint (seed 1 env, the other seeds' environments, seed 1's
reference); maps of I_s and of the prior's mass on the true successor per CD state; the same errors (and 1 - mass on the
true successor) against delta and against n(s); CD's learning curve; seed 1's RSA confusion ratios.

## 2026-10-14 (35) -- after the seed 1 diagnostic: CN transient, a design flaw of (34)'s rule, complementary test (rule recorded before computing)

### Recorded (user)

- **The visits confirm that the early CN transient is a visitation effect**: at 25,119 and 39,811, CN's share of the
  visits is higher in the environment than in the reference in 3 of 4 seeds (seed 1: 0.27 and 0.29 against 0.10 and
  0.11), consistent with b0 ~ 0 at equal visits ((34)).
- **(34)'s rule had a design flaw, recorded as such, after the result**: it assumed that an unfinished learning would
  give a normal alpha, whereas every run's curve shows that a high alpha is the normal signature of early learning (an
  action-independent prediction before the effect of the actions is learnt). **alpha compared at the same checkpoint
  therefore does not distinguish a delay of learning from a belief of helplessness.** (34)'s verdict ("undetermined")
  stays recorded as it was obtained.

### Complementary test, exploratory, read only (rule: user; operational definitions in brackets: implementer, set here before computing)

For the 8 runs, in CD, at every measured checkpoint: alpha and the prior's mass on the true successor against n(s), the
visits of the state's least-visited action -- per state, then as zone means (the zone's mean alpha and mass against its
mean log10 n). Rule on alpha:
- **seed 1 (environment) lies on the same relation as the other runs (within their dispersion): its gap is a learning
  delay due to the visits**;
- **at equal visits its alpha stays above the other runs' dispersion: an effect proper to this model**;
- otherwise undetermined.
[Per state: points (state, checkpoint) of CD with n >= 1, binned in log2 n (bins [2^k, 2^(k+1))). In a bin, each run's
mean alpha; the other runs' dispersion is the [min, max] of the 7 other runs' bin means. A bin qualifies if seed 1 env
has >= 3 points in it and at least 4 other runs have >= 3 points each. **Delay** if seed 1's bin mean lies within
[min, max] in every qualifying bin; **proper effect** if it lies above max in at least half of the qualifying bins and in
at least 2; otherwise undetermined. Zone means (secondary, the same rule): at each of seed 1's checkpoints, the other runs'
zone-mean alpha interpolated linearly in mean log10 n (only runs whose observed range covers that n, at least 4); delay
if seed 1 lies within their [min, max] at every such point, proper effect if above max at at least half of them (>= 2),
otherwise undetermined. The mass on the true successor is reported in the same way, without rule.]
Reported besides: the share of the visits to CD and to CN over seed 1 environment's first ~25,000 steps (the counts at the
checkpoint 25,119), against the other runs.

## 2026-10-14 (36) -- seed 1, the timing of the visits: exploratory test, rule recorded before computing; E3's analysis then closed

Hypothesis (user): at an equal number of visits, late visits teach less than early ones (loss of plasticity, primacy);
seed 1 environment's CD visits come later than in the other runs. Read only, exploratory.
**Rule (user; operational definitions in brackets: implementer, set here before computing):** regression of alpha per
CD state (all 8 runs, all measured checkpoints) on log(visits) and on the mean training step at which those visits
took place.
- If the coefficient of the visits' timing is positive (later visits, larger alpha) with an interval excluding 0, **and**
  seed 1 environment's gap falls within the other runs' dispersion once the term is included: **the gap is attributed to
  the timing of the visits**;
- if the gap persists despite the term: **it stays proper to the model, with no explanation identified**;
- otherwise: **undetermined**.
[Points: (run, CD state but the goal, measured checkpoint c) with n(s, c) >= 1. Visits: the transitions out of s in the
run's replay (replay_final.npz; the action stored at observation i is the one taken from it -- checked: it reproduces the
counts N at 300,000 but for the replay's last transition); log(visits) = log10 n(s, c), n the visits of the state's
least-visited action (as in (31) and (35)); the timing t(s, c) = the mean environment step (proportional to the training
step: ratio 0.6) of all the transitions out of s up to c, in units of 100,000 steps. The pilot runs' replay stops at
290,000: their 300,000 points are left out. Ordinary least squares on all the points pooled, alpha = b0 + b1 log10 n +
b2 t (model A: b2 omitted). Interval of b2: 95 % percentile bootstrap over the (run, state) clusters, 2,000 resamples,
seed 20261014. Gap of a run = the mean of its points' residuals; the others' dispersion = [min, max] of the other 7 runs'
mean residuals. "Within the dispersion" with model B: seed 1's mean residual <= the others' max; "persists": above it.]
After this test, E3's analysis is closed (user); the results section is written.

