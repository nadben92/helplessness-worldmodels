# E7 — The causal test: correcting h

**Status (2026-10-23): code ready (branch `e6`), developed on E4's weights on the CPU; waiting for E5's agents and
E6's curves.** Decisions: `docs/e6/DECISIONS.md` (33)-(43). Report: `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`.
2026-10-26: diagnostics of the cached value and of the decision rule (`run_e7_diagnostics.py`, decisions (51)-(54),
report section 16), run on E4's weights.

## What it tests

The belief is changed directly in h, nothing else (never z, never a weight): does behaviour then follow E6's curve?

## The correction

For a situation (a state of the zone of interest, h_t from a context walk, (42)): d minimising
sum_a KL(target_a || p(z | GRU(h_t + d, z_t, a))) + lam |d|^2, target_a = the posterior of the true next image given the
unmodified h' (phase 0: the posterior depends on h far less than on the image), no pixel loss; Adam from 0
(`probing/belief_steering.py`). Only on situations the model localises. Tasks: forward (exposed model, the test
environment's controllable successors) and reverse (reference model, the exposed training dynamics: inducing
helplessness).

## A common direction?

On the estimation half: alignment of the corrections with their mean direction (leave-one-out cosines against the
isotropic null of the same dimension); on the test half: the shift of b with v (at the corrections' mean norm) over the
shift with each situation's own optimum; stability over random restarts (`interventions/latent_direction.py`). Reported,
not decided: the user decides from the report.

## Application

E6's episodes (same starts, seeds, measures) with h <- h + alpha x (v, a random direction of the same norm, or the
stepwise optimum on the current h), at every step in the zone, alpha in {0, 0.25, 0.5, 0.75, 1, 1.5}; the reverse
direction on the reference. Abscissa: b at step 1 under intervention. Checks: b moved; the localisation, the mass on the
true successor of the played action and p_goal did not; the random control; alpha = 0 identical to E6 (`analyse_e7.py`).
E7's points are read on E6's main curves (residuals).

## Code, configs, results

`experiments/E7_latent_manipulation/` (`run_e7_directions.py`, `run_e7_apply.py`, `analyse_e7.py`; submitted by
`experiments/E6_phenotypes_latent/submit_kubeflow.py --phase e7-directions | e7-apply`); `configs/e7/e7.yaml`. Results:
`results/E7_latent_manipulation/e7dir__<agent>__<UTC>/` and `e7apply__<agent>__<UTC>/`.
