# E6 — Behavioural phenotypes and latent organisation

**Status: not started.** Depends on E4/E5's trained models. Deterministic dynamics only.

## What it tests

Whether the agents show behavioural phenotypes of helplessness, and whether those relate to the organisation
of their latent space:

- a battery of behavioural tests in a fully controllable environment, with norms taken from the reference
  agents;
- RSA on the latent (`measures/rsa.py`, as in E3's campaign);
- the link between latent geometry and phenotype.

## Config, code, results

`configs/e6/`, `experiments/E6_phenotypes_latent/`, results per `docs/RESULTS_CONVENTION.md`. Behavioural
metrics already computed in `measures/behavioural.py` are imported, never recomputed.
