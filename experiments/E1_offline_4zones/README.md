# E1 -- offline validation, 4 zones (12x12 torus)

**What this is.** An instrument-validation experiment, not a results
experiment: on an environment whose controllability is known analytically,
check that the measurement chain (Î, b, Δ) recovers the right answer, and
quantify the chain's precision. No learned-helplessness phenomenon is
expected here -- offline/tabular, no belief-behaviour loop, no transfer
between states. See `docs/E1_offline_4zones.md`.

**Status**: environment + construction tests, the Etape A calibration
sweep, and Etape B (generative collection, estimation, measurement,
permutation/bootstrap null, output artifact, figures, report) are all
done and the full 20-seed grid has been run to completion. (gamma, R,
direction) is **arbitrated and frozen** (see `config.yaml`):
`direction=away_from_goal`, `gamma=0.9`, `R=50`. Direction was unambiguous
(18/18 away_from_goal combinations produce a real interior well, 0/18
toward_goal do). gamma/R were chosen once the original locality rationale
(keep the two wells separable) was released by the flat-exterior finding
below: gamma=0.9 is where the interior well is deepest while keeping
interior V_exp clearly positive at R=50 (a negative-value zone would be a
trap, not an uncontrollable zone) and the accumulated step cost
1/(1-gamma)=10 stays a meaningful fraction of R.

**Etape B results (final, 20-seed grid)**: all spec 3.7 tolerances hold at
n=1000. b matches its analytic prediction closely in every zone (mean
+/- sd over 20 seeds): `uncontrollable_deterministic` -2.32193 +/- 0.0
(predicted -log2(5)=-2.321928, exact), `uncontrollable_noisy` -1.6518
+/- 0.0056 (predicted -1.6529, error 0.0011, well under the 0.03-bit
tolerance), both controllable zones ~0 (`controllable_noisy`: -0.00032
+/- 0.0064; `controllable_deterministic`: exactly 0). Î is exact at n=10
in both deterministic zones (log2(5)=2.321928 exactly). Delta >= 0 holds
with zero violations across all 2400 measurement rows. See
`artifacts/etape_b_report.txt`, `artifacts/measurements.csv`,
`artifacts/permutation_null.csv`, and the figures under `artifacts/`.

**Headline deliverable**: smallest detectable |b| (bootstrap null,
controllable zones, where H0: b=0 is genuinely true) shrinks smoothly from
0.102 bits (n=10) to 0.015 bits (n=1000) in `controllable_noisy`, fit
log-log slope -0.415, 95% CI [-0.479, -0.352] (in the right band -- see
`test_controllable_noisy_detectability_slope_is_near_minus_half` -- but
**-0.5 is genuinely outside the 95% CI**, not just noisily off it: with 20
seeds already averaged into each of the 5 fitted points, this looks like a
real, resolvable difference, not sampling noise.

An earlier explanation offered here ("residual 1/n contamination pulling
the slope shallower") had the **wrong sign** and was retracted: a 1/n
component decays faster than 1/sqrt(n), so it would make the fitted slope
*more* negative, not less -- the opposite of what's observed. Two
correctly-signed candidates were checked instead:

1. **n=10 outside the asymptotic regime** (a power-law fit weights the
   leftmost point most heavily in log-log space, and 1/sqrt(n) may not
   yet hold there). Refitting without n=10: slope -0.449, 95% CI
   [-0.478, -0.419] -- moves toward -0.5, a real and non-trivial partial
   contributor, but **-0.5 remains outside this CI too**.
2. **Resampling noise in the p95 estimate itself** (500 bootstrap draws
   might carry a bias that doesn't shrink with n). Checked directly:
   p95 at n_permutations=500 vs 5000 on the same fixed dataset (n=1000)
   -- mean 0.01847 vs 0.01830, well within sampling error of both (sd
   0.00044 shrinking to 0.00014, matching the ~sqrt(10x) reduction pure
   noise predicts). **Ruled out** -- no bias, only the expected shrinking
   spread. Now a permanent regression guard
   (`test_p95_estimate_has_no_permutation_count_bias`).
3. **Aggregation of heterogeneous per-state rates** (states with
   different amplitudes but all individually at -0.5 could, once
   averaged, look shallower over a finite budget range). Mathematically
   this doesn't hold for a simple linear average -- a mix of same-exponent
   power laws is exactly that exponent, amplitude-averaged -- but checked
   directly rather than argued away: per-state slopes, 5 seeds averaged,
   25 states in `controllable_noisy` -- mean -0.407, median -0.411, range
   [-0.484, -0.349], **none reaching -0.5**, and the zone-aggregate slope
   computed the same way (-0.408) is essentially identical to the
   per-state mean, not shallower than it. **Ruled out** -- the shallow
   scaling is a per-state property, not an averaging artifact. Now a
   permanent regression guard
   (`test_per_state_slopes_are_not_shallower_than_the_zone_aggregate`).

**Net: partially explained, residual gap unresolved.** Three candidates
checked, one (n=10) a real partial contributor, two ruled out. Dropping
n=10 closes about a third of the discrepancy; nothing further was found
to close the rest. Stated plainly rather than offered a mechanism that
hasn't been checked -- flagged for whoever next needs `controllable_noisy`'s
precision curve at high accuracy. `controllable_deterministic` is exactly
degenerate (0) at every budget, reported separately, never pooled with
the live zone.

**Recommended n, corrected**: the first pass recommended n=30 from the
fitted curve's raw crossing of the 0.03-bit tolerance (~n=17) -- but that
crossing is where the TYPICAL run's error *equals* tolerance, not where
the tolerance is reliably *met*. Counted directly from the real 20-seed
data: 3/20 seeds (15%) at n=30 actually exceed 0.03 bits for
`uncontrollable_noisy` -- a real, non-trivial failure rate for something
meant to be a tolerance. Corrected criterion: the budget where the fitted
curve *plus a k-sigma safety margin* falls under tolerance (a `k^2`
factor on n, since sd decays as `1/sqrt(n)`). At k=2: fitted crossing
~67 -> **RECOMMENDED n=100** (0/20 seeds exceed tolerance there,
confirmed empirically). At k=3 (more conservative): fitted crossing ~151
-> n=300 (also 0/20). See `report.py::recommended_n` /
`out_of_tolerance_rate`.

Three real bugs were found and fixed across two review passes, each
confirmed with a regression test, not just reasoned about:

1. **RNG correlation between experimental and control collection.**
   `np.random.default_rng(seed)` was called twice with the *same* seed
   value for "independent" streams. For an already-controllable zone,
   `derive_control` leaves the dynamics unchanged (`P_exp == P_ctrl`
   bit-for-bit), and the two separately-constructed-but-identically-seeded
   Generators stayed in lockstep across the *entire* state loop, not just
   the identical-P zones -- so b measured exactly 0 with zero variance in
   both controllable zones, true by construction, not by measurement, and
   no detectability threshold was ever computed anywhere H0: b=0 actually
   held. Fixed via `estimation.py::spawn_exp_ctrl_rngs`
   (`SeedSequence.spawn()`) -- confirmed by a regression test and a
   deliberately-kept *shared*-seed identity check
   (`tests/agents/test_e1_estimation.py`) verifying `derive_control` is a
   true identity on controllable zones.

2. **The bootstrap null pooled actions before resampling.** Fine for
   testing H0: I=0 (matches the shuffle null's own hypothesis), but wrong
   when reused for b's headline detectability in the controllable zones:
   pooling nulls out the REAL per-action dependence there, making the
   statistic scale like an independence-test quantity (~1/n, Wilks'-
   theorem-style) instead of ordinary estimator sampling noise around a
   non-zero true value (~1/sqrt(n)) -- confirmed empirically before the
   fix (an apparent ~1/n threshold decay) and after (a regression test,
   `test_bootstrap_null_b_zone_scales_slower_than_1_over_n_in_a_live_controllable_zone`,
   pins the corrected scaling). Fixed via per-action resampling
   (`permutation_null.py::_resample_rows_per_action_batch`) -- each
   action now resamples from its OWN observed distribution, not a pooled
   one. A side effect, now covered by tests: `controllable_deterministic`'s
   bootstrap null is degenerate at exactly 0 (deterministic transitions
   have zero estimation noise regardless of resampling method), so it is
   reported separately, never pooled with the live `controllable_noisy`
   zone in the headline table.

3. **The shuffle null was wrongly described as "too narrow".** That
   framing (attributing a small discrepancy to hypergeometric shrinkage
   the bootstrap "corrects") was retracted: a permutation test conditional
   on the observed counts is exact, by construction, not an approximation.
   The two nulls answer different questions -- shuffle: exact conditional
   test of whether Ihat is distinguishable from zero; bootstrap:
   unconditional fresh-dataset precision, used for E2/E4-facing
   detectability numbers -- and a small difference between them is
   expected, not evidence either is wrong. See
   `permutation_null.py`'s module docstring for the full statement.

**Recommended-n criterion also corrected**: an SD estimated from only 5
seeds has relative standard error ~35% (`~1/sqrt(2*(k-1))`), far too noisy
to read a "plateau" pointwise -- what looked like non-monotonicity was
likely just that estimation noise. `SEEDS` raised to 20 (~16% relative
error) and `report.py::plateau_n` replaced with
`fit_inverse_sqrt_n`/`fit_power_law_slope`: a proper log-log curve fit
across all budgets at once, with `recommended_n` now reporting the budget
where the FITTED curve crosses the 0.03-bit tolerance, plus the actual
fitted exponent for the live controllable zone (flagged if it's not near
the theoretically-expected -0.5).

**Performance**: the first correction pass made the null computation
prohibitively slow (independently profiled: >60 min and still running for
the 20-seed grid, killed). Root cause was `rng.choice(..., p=...)`, which
draws individual categorical samples one at a time -- replaced with
`rng.multinomial(n, p, size=n_permutations)` (only the resulting counts
were ever used, confirmed statistically equivalent to the old approach by
`test_multinomial_optimization_matches_choice_based_reference_distribution`,
a chi-squared test comparing the two implementations' output
distributions on a toy case, not sample-for-sample equality, which can't
hold since the two draw from the RNG stream differently) plus batching
the whole permutation/bootstrap loop across all `n_permutations` in one
vectorized numpy call per state instead of a Python loop calling one draw
at a time. Full 20-seed grid completed in this session.

Before committing to the full grid, a smoke run (1 seed, n=10 and n=1000
only, ~56s) checked the three things that mattered most: b non-zero in
`controllable_noisy` (0.015 at n=10, 0.008 at n=1000 -- confirmed), Delta
>= 0 everywhere (0 violations, min=0.0 -- confirmed), and the
detectability slope in the right direction (-0.32 at 1 seed/2 points, a
noisy but correctly-signed early read of what the full grid later
confirmed at -0.415).

Full tables and reasoning are in `artifacts/etape_b_report.txt`
(`report.py`).

## Environment

12x12 torus, 144 states, `|A|=5`. A neutral cross corridor (rows/cols 0 and
6, 44 states, controllable-deterministic -- this is E1's own "reference
zone" in the spec's terminology, *not* the same object as
`envs.zones.reference_zone()`, which is an unrelated, frozen 2x2 pattern
used elsewhere; see `docs/E1_offline_4zones.md`) plus four 5x5 blocks (rows
1-5 or 7-11, crossed with cols 1-5 or 7-11), one of each dynamics type.
Goal at (0,0), a corridor intersection.

Corner assignment (diagonal, see `docs/E1_offline_4zones.md` for the full
reasoning -- summary: diagonal gives the larger of the two achievable
minimum inter-zone distances, 4 vs 2, and makes both controllable blocks
border both uncontrollable ones symmetrically):

| block | rows | cols | dynamics |
|---|---|---|---|
| NW | 1-5 | 1-5 | `UncontrollableDeterministic` (tapis roulant) |
| NE | 1-5 | 7-11 | `ControllableNoisy(p=0.125)` |
| SW | 7-11 | 1-5 | `ControllableDeterministic` |
| SE | 7-11 | 7-11 | `UncontrollableNoisy(drift, p=0.125)` |

`p=0.125` reproduces the spec's own epsilon=0.1 mixture
(0.9 on the intended/drift neighbour, 0.025 on each of the other 4) exactly
under this codebase's existing noise parameterisation -- see
`configs/zones/e1_4zones.yaml`'s comments and
`tests/envs/test_e1_4zone_layout.py::test_p_0_125_reproduces_the_spec_epsilon_0_1_mixture_exactly`.

Drift is always along the row axis (a fixed, arbitrary-but-documented
convention -- both uncontrollable blocks resolve `away_from_goal` /
`toward_goal` via toric row-distance, see `envs.zones.resolve_drift`), both
blocks set to `away_from_goal` by construction (required for an interior
well rather than a bump, confirmed empirically in the calibration sweep).

Units: **bits throughout**, matching the codebase's one, unmodified
`controllability_from_tensor`/`analytic_controllability` implementation
(log2). The spec's own nat-valued tables were replaced with their bit
equivalents at the start of this delivery -- see `docs/E1_offline_4zones.md`.

## Etape A -- calibration sweep

```bash
cd experiments/E1_offline_4zones
python calibration.py
```

Pure DP under the true kernel (`value_iteration`), no learning, no seeds.
Sweeps `gamma in {0.7,0.75,0.8,0.85,0.9,0.95} x R in {20,50,100} x direction
in {away_from_goal, toward_goal}` (36 combinations). Writes to `artifacts/`:

- `calibration_report.csv` -- one row per combination: symmetry check
  (exact reflection invariance of the control environment's V*, not a
  4-block mean), reachability, interior-traversal, the exterior gradient
  contrast (descriptive only, see finding below), the interior V*
  exp-vs-control gap (the real, direction-sensitive signal), and the
  two-distance confounding regression.
- `gradient_curves.csv` / `gradient_curves.png` -- exterior V* vs distance
  to each uncontrollable zone, experimental vs control.
- `interior_gap_vs_gamma.png` -- the direction-recommendation evidence.
- `confounding_r2_vs_gamma.png`.
- `recommendation.txt` -- a candidate reading, **not applied
  automatically**.

**Finding surfaced by this sweep** (not assumed going in), and three
corrections to how it's reported -- see `artifacts/recommendation.txt` for
the full text, generated by `calibration.py`, never hand-edited:

The corridor gives every controllable-region state a guaranteed shortest
controllable-deterministic path to the goal that never requires entering
an uncontrollable zone. Under optimal play this makes V* outside the two
uncontrollable zones exactly insensitive (to numerical precision, every
combination tested) to drift direction and to whether a zone is
controllable at all -- exterior V* is identical between experimental and
control (max diff ~1e-11 across the whole sweep). This is the actual null
model.

The exterior "contraste du gradient" curve is **not** that null model and
**not** a value well: it *decreases* with distance from either
uncontrollable zone (the wrong direction for a well) because the goal
(0,0) sits at a corner of the NW block, so "near the zone" mechanically
means "near the goal" -- it's an exterior distance-to-goal profile,
descriptive only. Its nw- and se-partitioned versions are bit-identical at
every distance bin (confirmed at the frozen gamma=0.9, R=50), which is
further confirmation of the flat-exterior finding, not two independent
gradient measurements -- the disjoint-partition code is correct and
unchanged, only the old "two curves" framing was wrong. The far distance
bin has only 4 states; sample sizes are reported inline in
`recommendation.txt`, not only in the CSV.

The two-distance confounding regression is computed in the *control*
environment, where neither zone is uncontrollable -- its near-zero R^2 is
a mechanical consequence of that, not a separability criterion that
"passed". It's moot, superseded by the flat-exterior finding itself
(there's nothing to confound). Numbers stay in the CSV; only the
interpretation changed.

The real, direction-sensitive signal is the **interior** V* gap
(`interior_gap_nw`/`interior_gap_se`: experimental minus paired control,
same states) -- that's what the direction arbitration above is based on,
not the exterior contrast.

**Scope limit on the flat-exterior finding**: it holds for *true* value
propagation under a perfect model. In E2 and E4 the model is learned and
wrong, and bootstrapping on an imperfect T_hat can propagate estimation
error spatially, producing distance-dependent structure with no belief
generalisation involved -- E1's null model doesn't cover that mechanism;
characterising it is E2's job.

## Etape B -- generative collection, estimation, measurement

```bash
cd experiments/E1_offline_4zones
python run_etape_b.py   # ~5 min: 5 budgets x 5 seeds x 2 envs, plus 500-permutation nulls
python figures.py
python report.py        # headline deliverable: detectable-|b| table + recommended n, artifacts/etape_b_report.txt
```

- Generative model (`estimation.py::collect_generative`), never trajectories:
  exactly n draws per (s,a) pair via `rng.multinomial`, for
  n in {10,30,100,300,1000}, 5 seeds, both environments.
- Primary estimator: plug-in MLE. Reuses `agents.tabular.TabularModelBasedAgent`
  with `alpha=0.0` (verified exact plug-in MLE given every (s,a) has
  nonzero count by construction) rather than reimplementing T_hat -- see
  `estimation.py`. The Dirichlet robustness variant (alpha in {0.5,1},
  restricted to `estimation.py::geographic_alpha_support`) reuses the same
  class's existing `alpha_support` mechanism, config-selectable, never the
  default.
- `Delta(s) = V*(s) - V^pi_hat(s)` via the new
  `planning.value_iteration.policy_evaluation` (no max anywhere --
  structurally tested): pi_hat's own planner values are discarded;
  V^pi_hat is pi_hat evaluated under the TRUE kernel, never the planner's
  return value evaluated under T_hat.
- Permutation null (`permutation_null.py`) on the zone-aggregated Ihat
  only, never state by state: shuffles the action label of every
  (action, successor) draw within a zone, 500 times, for both Ihat (in
  the two I*=0 zones) and b (permuting both paired environments
  independently, per draw -- a wider null than either term's alone).
  Degenerate at exactly 0 in the uncontrollable-deterministic zone, as
  required.
- Output artifact (`artifact.py`): `measurements.csv` (spec 3.6 schema) +
  `permutation_null.csv`, with `write_*`/`load_*`/`join_exp_ctrl` functions
  for E2/E3 to reuse without reimplementing the comparison.

## Running the tests

```bash
pytest tests/envs/test_e1_4zone_layout.py tests/planning/test_e1_calibration.py \
       tests/planning/test_value_iteration.py tests/measures/test_controllability.py \
       tests/measures/test_belief.py tests/measures/test_e1_measurement.py \
       tests/measures/test_e1_permutation_null.py tests/agents/test_e1_estimation.py
```
