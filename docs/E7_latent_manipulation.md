# E7 — Direct manipulation of the latent

**Status: not started.** Depends on E4 (probes) and E6 (phenotypes). Deterministic dynamics only.

## What it tests

The final causal test: a direct manipulation of the latent (for example, interpolating zone codes) applied
both to the belief of controllability (measured with E4's probes) and to the behavioural phenotypes of E6.

## Config, code, results

`configs/e7/`, `experiments/E7_latent_manipulation/`, results per `docs/RESULTS_CONVENTION.md`. Manipulations
of a belief belong in `interventions/` (arrays in, arrays out); the torch side of applying them to a world
model belongs in `probing/`.
