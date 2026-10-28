# E6 — The behavioural signature of the controllability belief

**Status (2026-10-28): E6/E7 on all agents (docs/e6/DECISIONS.md (63)) STOPPED by the user -- E5's shuffle models under-trained; E5's probes then E6/E7 to be rerun on the converged models, the same scripts with the new checkpoints. Done: E6 and C1/C2 (64 pairs), E7's directions; E7's anti-oracle 8/21, sweep 0/7 (interrupted); no analysis of the partial E7 runs (IMPLEMENTATION_REPORT.md, section 21).** Earlier status (2026-10-23): code ready (branch `e6`), developed and tested on E4's weights on the CPU; waiting for E5's
agents.** Decisions: `docs/e6/DECISIONS.md` (33)-(43), shared with E7. Implementation report, phase 0 and preliminary
results: `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`. Deterministic dynamics only.

## What it tests

Whether behaviour is a function of the belief alone, whatever the training history, and whether that function is
strictly monotone (so the belief can be read back from behaviour). For every pair (agent, test image): the a-priori
belief b at step 1 and behavioural phenotypes in the same situation.

## Agents and test images

An agent = an exposed model and its reference, listed in an agents FILE (`configs/e6/agents_*.yaml`: id, condition,
configuration, seed, the two runs' weights) -- adding an agent (E8's immunisation, for instance) changes no code. E5's
campaign: `configs/e6/agents_e5.yaml` (baseline and shuffle f = 1 on p1-b-d1; to be confirmed after E5's wave 2).
Development: `configs/e6/agents_dev_e4.yaml` (E4's four configurations).

Test environments: the reference's dynamics (everything controllable) with an image of E4's probe grid (canonical
layout). The list is a parameter (`e6.images`; development: training image, position only, theme only, neutral). The
zone of interest is the slot the probe changes ((34)). A pair is kept only if the agent's reference passes E4's validity
criterion on E4's probe of that image ((38)).

## Protocol

Starts: the central 3 x 3 of the zone of interest, 20 episodes each. The learned initial state, no context. Step 1: b
(E4's computation: I_exposed - I_reference, 16 prior draws per action decoded and matched) and the localisation. Then the
training's policy with frozen weights (epsilon 0.1, (39)) until the goal or H = 36. Logged at every step: true position,
action, I_s and b_local, localisation, Q spread, p_goal, mass on the true successor of the played action.

Phenotypes (`measures/zone_phenotypes.py`), per (agent, image, model), with bootstrap intervals: first exit time (Kaplan-
Meier median, censored share), directed share, recovery latency j_theta ((35)); secondary: share of steps in the zone,
stay share, revisit share (k = 4). Companion curve: b_local by step of the first stay.

## Analysis (`analyse_e6.py`)

Per phenotype: the main cloud (x = b at step 1) with an isotonic and a smooth monotone fit and a bootstrap band over the
agents; by condition; held out by condition and by agent (residuals; the blind test: b predicted from the phenotype);
strict monotonicity and the invertible range of b; the localisation confound. Also the coherence of b at step 1 with
E4's probe, and the invalid pairs.

## Code, configs, results

`experiments/E6_phenotypes_latent/` (`run_e6.py`, `analyse_e6.py`, `batches.py`, `submit_kubeflow.py` -- phases `e6`,
`e7-directions`, `e7-apply` -- `export_batch.py`); library: `pipeline/frozen_setup.py`, `probing/frozen_episodes.py`,
`measures/zone_phenotypes.py`, `measures/monotone_fit.py`. Configs: `configs/e6/`. Results:
`results/E6_phenotypes_latent/e6__<agent>__<UTC>/` (rows, episodes, meta).
