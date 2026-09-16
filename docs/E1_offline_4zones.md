# E1 — Offline validation, 4 zones

**Status: done.** Environment construction, Étape A (calibration sweep),
and Étape B (generative collection, estimation, measurement,
permutation/bootstrap null, output artifact, figures, report) are all
complete, tested, and the full 20-seed grid has run to completion — see
`experiments/E1_offline_4zones/README.md` for the full account of three
real bugs found and fixed across two review passes (RNG correlation
between exp/ctrl collection; a pooled-action bootstrap that nulled out
real controllability, scaling wrong — now asserted by a slope test, not
just observed; and a retracted "shuffle null is too narrow" claim), plus
the performance fix (`rng.multinomial` instead of `rng.choice(p=...)`,
statistically verified equivalent, plus vectorizing the permutation loop)
that made the 20-seed grid tractable at all. **(γ, R, direction)
arbitrated and frozen**: `away_from_goal`, `gamma=0.9`, `R=50` (see
`config.yaml`) — unaffected by the Étape B corrections.

**Final results**: all spec 3.7 tolerances hold at n=1000. b matches its
analytic prediction closely in every zone (mean ± sd over 20 seeds):
`uncontrollable_deterministic` −2.32193 ± 0.0 (predicted −log2(5) =
−2.321928, exact), `uncontrollable_noisy` −1.6518 ± 0.0056 (predicted
−1.6529, error 0.0011), both controllable zones ~0. Δ ≥ 0 holds with zero
violations across 2400 rows. Headline detectability (bootstrap null,
controllable zones): `controllable_noisy` shrinks smoothly from 0.102
(n=10) to 0.015 bits (n=1000), fitted log-log slope −0.415, 95% CI
[−0.479, −0.352] — in the right band, but **−0.5 is genuinely outside the
CI**, a real difference given 20 seeds already averaged into each fitted
point, not sampling noise. (An earlier "residual 1/n contamination"
explanation was retracted for having the wrong sign — 1/n decays faster
than 1/√n and would make the slope *more* negative, not less. Three
candidates checked: excluding n=10 from the fit moves the slope to
−0.449 [−0.478, −0.419] — real partial improvement, but −0.5 stays
outside; resampling noise in the p95 estimate, tested at 500 vs 5000
permutations, ruled out (mean shift within sampling error); aggregation
of heterogeneous per-state rates, tested by fitting per-state slopes
directly (25 states, 5 seeds averaged: mean −0.407, none reaching −0.5,
zone-aggregate slope −0.408 essentially identical to the per-state mean)
— also ruled out, since an aggregation artifact would make the aggregate
shallower than the per-state mean, and it isn't. **Net: partially
explained (n=10), residual gap unresolved** — stated plainly, not
attributed to an unverified mechanism.) `controllable_deterministic` is
exactly degenerate (0) at every budget.

**Recommended n, corrected**: the raw fitted-curve crossing (~n=17,
rounding to n=30) is where the *typical* run's error equals tolerance —
15% of the real 20 seeds exceed 0.03 bits at n=30, counted directly, not
inferred. Corrected to a k-sigma safety margin (moves the crossing by
`k²` on n): **n=100 at 2σ** (0/20 seeds exceed tolerance, confirmed),
n=300 at 3σ (also 0/20). **Recommended n=100 for E2/E3.**

## What it tests

Validates the new base environment (`helplessness.envs.torus_env.TorusZonedEnv`)
and its zone/dynamics/appearance framework against the same offline
protocol E0 already validated on the 5x5 gridworld — the first checkpoint
before moving to the online agent (E2) or the world model (E3). E1 is an
instrument-validation experiment: it checks that the measurement chain
(Î, b, Δ) recovers the known-analytically-correct answer on a controlled
environment, and quantifies its precision. No learned-helplessness
phenotype is expected — offline/tabular, belief-behaviour loop cut, no
transfer between states.

## Environment

12x12 torus (144 states), `|A|=5`. A neutral cross corridor (rows/cols 0
and 6, 44 states, controllable-deterministic — this is E1's own "zone de
référence" in the spec's terminology, **not** the same object as
`envs.zones.reference_zone()`, an unrelated frozen 2x2 pattern used
elsewhere in the codebase) plus four 5x5 blocks, one of each `DynamicsSpec`
type, diagonally assigned:

- NW (rows 1-5, cols 1-5): `UncontrollableDeterministic` (tapis roulant)
- NE (rows 1-5, cols 7-11): `ControllableNoisy(p=0.125)`
- SW (rows 7-11, cols 1-5): `ControllableDeterministic`
- SE (rows 7-11, cols 7-11): `UncontrollableNoisy(drift, p=0.125)`

Goal at (0,0). `p=0.125` reproduces the spec's own ε=0.1 noise mixture
(0.9/0.025×4) exactly, under this codebase's existing `ControllableNoisy`/
`UncontrollableNoisy` parameterisation. Drift is always row-axis, resolved
via toric row-distance to the goal's row (`envs.zones.resolve_drift`), set
to `away_from_goal` for both uncontrollable blocks. Full reasoning for the
corner assignment, the drift-axis convention, and the (dropped) hard
disjoint-wells geometric assertion is in
`experiments/E1_offline_4zones/README.md`.

Units are **bits throughout** (the codebase's one, unmodified
`controllability_from_tensor`/`analytic_controllability`, log2) — the
spec's own nat-valued tables were translated to their bit equivalents
before any code was written, rather than adding a unit-conversion path to
the frozen-E0-dependent measure.

**Finding from the Étape A sweep**: under optimal play, V* outside the two
uncontrollable zones is exactly insensitive to drift direction and to
whether a zone is controllable at all — the corridor gives every
controllable-region state a shortest path to the goal that never requires
entering an uncontrollable zone (exp-vs-ctrl exterior diff <= ~1e-11 across
the whole sweep). This is the flat null model. The spec's "contraste du
gradient" curve is *not* that null model and *not* a value well — it
decreases with distance from either uncontrollable zone (the goal sits at
a corner of the NW block, so "near the zone" means "near the goal"),
descriptive only; its nw/se partitions are bit-identical at every bin
(further confirmation of the flat-exterior finding, not two independent
measurements). The direction-sensitive signal lives entirely inside the
two uncontrollable zones instead (`interior_gap_nw`/`interior_gap_se`, see
`experiments/E1_offline_4zones/README.md` and
`artifacts/interior_gap_vs_gamma.png`) — that's what the direction
arbitration is based on. The two-distance confounding regression (computed
in the control environment, where neither zone is uncontrollable) is moot,
not a separability criterion that passed. **Scope limit**: the flat null
model holds for true value propagation under a perfect model; a learned,
wrong model (E2, E4) can propagate estimation error spatially via
bootstrapping, producing distance-dependent structure with no belief
generalisation — E1 does not cover that mechanism.

## Agent

`TabularModelBasedAgent` (unchanged from E0), reused for Étape B's plug-in
MLE (`alpha=0.0`, verified exact given full (s,a) coverage by
construction) and its Dirichlet robustness variant (`alpha` in {0.5,1},
restricted to `estimation.py::geographic_alpha_support` via the class's
existing `alpha_support` mechanism) — see
`experiments/E1_offline_4zones/estimation.py`.

## Configuration

`configs/zones/e1_4zones.yaml` (geometry, data-only) +
`experiments/E1_offline_4zones/config.yaml` (experiment-level, composed via
`helplessness.config.compose()`). γ=0.9, R=50, direction=away_from_goal are
now frozen in `config.yaml`, arbitrated from the Étape A sweep.

## Dependencies

`helplessness.envs.{zones,dynamics,appearance,observation,torus_env}`,
`helplessness.measures.controllability` (additive-only: `controllability_per_state`/
`conditional_entropy_per_state` added, `controllability_from_tensor` refactored
to be the former's mean, regression-pinned against E0-frozen output — see
`tests/measures/test_controllability.py`), `helplessness.measures.belief`
(additive-only: `l1_error_per_pair` added, `model_error` refactored to use
it — see `tests/measures/test_belief.py`), `helplessness.planning.value_iteration`
(additive-only: `policy_evaluation` added, `value_iteration` unchanged —
see `tests/planning/test_value_iteration.py`), `helplessness.agents.tabular`
(unchanged, reused for T̂/R̂ estimation).
