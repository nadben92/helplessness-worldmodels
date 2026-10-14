# E6 — Latent perturbations

**Status: not started.** Depends on E5.

## What it tests

Causal manipulation of the world models trained in E5: steering along the
controllability axis found in the latent, and checking whether the induced
behaviour reproduces the E0-style phenotype (return suboptimality,
perseveration, trap avoidance, etc.) -- the neural-phase analogue of E0's
`03_intervention`/`04_rescue_injection` (the controllability dial and
structure injection), which were explicitly built once "to be reused as-is
in the neural phase" (see their module docstrings in
`helplessness.interventions`).

## Environment

The E5 environments, models already trained -- this experiment perturbs
existing models, it does not train new environments.

## Agent

The world models produced by E5 (loaded, not retrained).

## Configuration

Not yet defined -- depends on E5's models.

## Caveat (from CLAUDE.md)

This is the step that actually proves the latent's controllability
information is used, not merely decodable -- the reason E3's linear-decode
result alone was flagged as insufficient.
