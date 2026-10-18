# Experiments

The central tracker: one row per experiment, from the first tabular work to E7. Update the row (and the
experiment's own `docs/` file) on every status change. Every experiment is deterministic from E4 on.

Perceived controllability is always I(a;s'|s) read from the learned model (the tabular T_hat, or the world
model's prior from E3 on), compared to the paired reference environment: b = I_hat - I_hat_ref.

## Finished experiments (frozen)

Their logic, configs, seeds and defaults never change. They may be moved and their imports updated; a module
promoted to the library leaves a re-export shim at its old place. Each is tagged at its final state.

| ID | Status | What it tested | Entry points | Configs | Report | Tag |
|---|---|---|---|---|---|---|
| E0 01-04 | done | Offline tabular, 5x5 gridworld, one noise parameter p: does the perceived controllability of training determine behaviour in a controllable test environment; component swaps, controllability dial, rescue by structure injection | `experiments/E0_offline_5x5/0{1..4}_*/run.py` (and `eval_v2.py`, `plot_*.py`, `rescue_diagnostics.py`); `Makefile` | `experiments/E0_offline_5x5/*/config*.yaml`, `helplessness.config` dataclasses | `reports/week1_offline_report.md`, `docs/E0_offline_5x5.md` | `v0.1-offline`, `e0-final` |
| E0 05 | done (prototype) | Online tabular agent and recovery protocol on the 5x5 gridworld, before the 4-zone environment existed | `experiments/E0_offline_5x5/05_online_prototype/{train,recovery}.py` | `OnlineTrainConfig`, `RecoveryConfig` | `experiments/E0_offline_5x5/05_online_prototype/README.md`, `results/online/` | `e0-final` |
| E1 | done | Offline tabular validation on the 12x12 torus, 4 zones (controllability x stochasticity): calibration, detectability of b, permutation null | `experiments/E1_offline_4zones/{calibration,run_etape_b,figures,report}.py` | `experiments/E1_offline_4zones/config.yaml`, `configs/zones/e1_4zones.yaml` | `paper/reports/e1_report.tex`, `experiments/E1_offline_4zones/artifacts/etape_b_report.txt`, `docs/E1_offline_4zones.md` | `e1-final` |
| E2 | done | Online tabular, epsilon-greedy: null model of the propagation of b under the agent's own policy, against a matched-counts control | `experiments/E2_online_4zones/{pilot,sweep,analysis,episode_analysis}.py` | constants in `pilot.py` / `sweep.py`, `configs/zones/e1_4zones.yaml` | `experiments/E2_online_4zones/RESULTS.md`, `paper/sections/e2.tex`, `docs/E2_online_4zones.md` | `e2-final` |
| E3 | done | Simplified DreamerV3 world model on the pixel environment: I_s from the prior, closed-loop agent, final campaign (4 seeds) | `experiments/E3_worldmodel_4zones/{run_mini,run_agent,submit_kubeflow,campaign_report}.py` | `configs/e3/*.yaml`, `experiments/E3_worldmodel_4zones/config.yaml` | `docs/E3_worldmodel_4zones.md`, `paper/reports/e3_report.tex`; decisions `docs/e3/DECISIONS.md` | `e3-final` |

## Planned experiments (deterministic only)

| ID | Status | Objective | Entry point | Configs | Report |
|---|---|---|---|---|---|
| E4 | wave 1 (pilot) done, B = 300,000; wave 2 ready (branch `e4`; 4 configurations, 2 waves) | Generalisation gradient: is the belief of controllability carried by position, appearance, both or neither (zero-data probes on fully controllable test environments) | `experiments/E4_generalization_gradient/run_e4.py`, `measure_probes.py`, `pilot_analysis.py`; launch `RUN_GPU.md` | `configs/e4/e4.yaml`, `themes.yaml` | `docs/E4_generalization_gradient.md` |
| E5 | not started | Manipulating the belief through training (prior exposure to control, similar themes across opposite controllabilities, latent capacity), measured with E4's probes | `experiments/E5_belief_training/` | `configs/e5/` | `docs/E5_belief_training.md` |
| E6 | not started | Behavioural phenotypes and latent organisation: test battery with norms from the reference agents, RSA, latent geometry vs phenotype | `experiments/E6_phenotypes_latent/` | `configs/e6/` | `docs/E6_phenotypes_latent.md` |
| E7 | not started | Direct manipulation of the latent (e.g. interpolation of zone codes), applied to the belief (E4's probes) and to E6's phenotypes: the final causal test | `experiments/E7_latent_manipulation/` | `configs/e7/` | `docs/E7_latent_manipulation.md` |

The previous plan for E4-E6 (generalisation pilot, generalisation campaign, latent perturbations) was abandoned
on 2026-10-14; its files are kept in `docs/archive/plan_2026-09/`.

## Where things live

- Shared library: `src/helplessness/` (see `CLAUDE.md`, "Structure").
- Results of a cluster batch: an orphan branch `results/<batch>` (`docs/RESULTS_CONVENTION.md`).
- Paper and reports: `paper/` (`paper/README.md`).
