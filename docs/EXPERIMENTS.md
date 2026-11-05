# Experiments

The central tracker: one row per experiment, from the first tabular work to E8. Update the row (and the
experiment's own `docs/` file) on every status change. Every experiment is deterministic from E4 on.

Perceived controllability is always I(a;s'|s) read from the learned model (the tabular T_hat, or the world
model's prior from E3 on), compared to the paired reference environment: b = I_hat - I_hat_ref.

## Status at a glance (2026-11-05)

Everything below is merged into `main` (merges `2c1662b` e5-robust, `5f0e0bc` e6, `149850a` slides). The branch
column says where each experiment was developed.

| ID | Status | Branch of origin | Report | Key commits / tags |
|---|---|---|---|---|
| E0 | finished, frozen | `main` (pre-reorg) | `reports/week1_offline_report.md`, `docs/E0_offline_5x5.md` | `v0.1-offline`, `e0-final` |
| E1 | finished, frozen | `main` (pre-reorg) | `paper/reports/e1_report.tex`, `docs/E1_offline_4zones.md` | `e1-final` |
| E2 | finished, frozen | `main` (pre-reorg) | `experiments/E2_online_4zones/RESULTS.md`, `docs/E2_online_4zones.md` | `e2-final` |
| E3 | finished, frozen | `e3/worldmodel` | `paper/reports/e3_report.tex`, `docs/E3_worldmodel_4zones.md`, `docs/e3/DECISIONS.md` | `e3-final`, `pre-reorg` (`213fb0d`) |
| E4 | finished | `e4` | `results/E4_generalization_gradient/final/REPORT.md`, `docs/E4_generalization_gradient.md` | `f8b2dc5` (closing) |
| E5 | finished | `e5`, then `e5-robust` | `experiments/E5_belief_training/IMPLEMENTATION_REPORT.md`; `analysis_450k/`, `analysis_650k/`, `analysis_swap/` (with `THEME_EVIDENCE.md`), `analysis_robustness/` `REPORT.md`; `docs/e5/DECISIONS.md` | `9ba3ab0` (650,000), `cb80ae9` (swap probes), `1adcbd8` (pair A vs ref. B), `d82d0a0` (robustness) |
| E6 | finished | `e6` | `experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md` (sections 1-21), `docs/e6/DECISIONS.md` | `966404c` (all agents at 650,000) |
| E7 | finished | `e6` | same report (sections 16-22), `docs/e6/DECISIONS.md` | `d77c13a` (decomposition at k = 1), `861adca` (section 22) |
| E8 | design written, code not started | `e8` | `docs/E8_therapy_relapse.md`, `docs/e8/DECISIONS.md`, `docs/e8/EXPECTED_FORMS.md` (report to come: `experiments/E8_therapy_relapse/IMPLEMENTATION_REPORT.md`) | -- |

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

## Experiments E4-E8 (deterministic only)

| ID | Status | Objective | Entry point | Configs | Report |
|---|---|---|---|---|---|
| E4 | finished (branch `e4`; 4 configurations, 2 waves, probe grid analysed: `results/E4_generalization_gradient/final/REPORT.md`) | Generalisation gradient: is the belief of controllability carried by position, appearance, both or neither (zero-data probes on fully controllable test environments) | `experiments/E4_generalization_gradient/run_e4.py`, `measure_probes.py`, `pilot_analysis.py`; launch `RUN_GPU.md` | `configs/e4/e4.yaml`, `themes.yaml` | `docs/E4_generalization_gradient.md` |
| E5 | finished (branches `e5`, `e5-robust`): waves 1 and 2; shuffle runs continued to 650,000 (DECISIONS (35)-(37)); PRELIMINARY probes at 475,000 (`analysis_450k/REPORT.md`) and final probes at 650,000 (`analysis_650k/REPORT.md`); theme-swap probes (DECISIONS (38), `analysis_swap/REPORT.md`: by (38)'s rule no axis readable) and pair A against reference B (DECISIONS (39): (38)'s conclusion unchanged); EXPLORATORY start of the gradient and `analysis_swap/THEME_EVIDENCE.md`; robustness diagnostic (`analysis_robustness/`, predictions committed first: H1 "not decided"); f = 0.5 abandoned; the immunisation dropped (see E8) | Manipulating the belief through training: does the belief move to the theme when the position stops predicting controllability (shuffled zone assignments); measured with E4's probes | `experiments/E5_belief_training/run_e5.py`, `measure_probes.py`, `choose_switch_step.py`, `rsa_e5.py`; launch `RUN_GPU.md` | `configs/e5/e5.yaml` (E4's blocks + `e5`) | `docs/E5_belief_training.md`, `docs/e5/DECISIONS.md` |
| E6 | finished (branch `e6`; campaign on all agents done 2026-11-04: E4's four pairs and baseline B at 300,000, shuffle A, E, F at 650,000; local CPU): forms 1, 3, 5 held, 2 and 4 not; report section 21, `results/E6_phenotypes_latent/all_agents_650k/` | The behavioural signature of the belief: per (agent, test image), b at step 1 and zone phenotypes (exit time, directed share, recovery latency, ...) in fully controllable test environments; is behaviour a strictly monotone (invertible) function of b alone, whatever the training? | `experiments/E6_phenotypes_latent/run_e6.py`, `analyse_e6.py`, `run_all_cpu.py`, `analyse_all.py`, `submit_kubeflow.py` | `configs/e6/` (E4's blocks + `e6`; agents files) | `docs/E6_phenotypes_latent.md`, `docs/e6/DECISIONS.md` |
| E7 | finished (branch `e6`; with E6 on all agents): decomposition at k = 1 (belief effect against cached-value effect) and section 22 (a correction of h is not transported); report sections 21-22, `results/E6_phenotypes_latent/all_agents_650k/` | The causal test: a correction of h (never z, never a weight) that makes the model predict the controllable dynamics (or, on a reference, the conveyor); a common direction?; applied during E6's episodes, does behaviour follow E6's curve? | `experiments/E7_latent_manipulation/run_e7_directions.py`, `run_e7_apply.py`, `analyse_e7.py`, `run_e7_context.py` | `configs/e7/e7.yaml` (E6's + `e7`) | `docs/E7_latent_manipulation.md`, `docs/e6/DECISIONS.md` |
| E8 | design written (branch `e8`, 2026-11-05), code not started | Therapy and relapse: pre-exposed agents are re-trained in a fully controllable environment and we measure how fast belief and behaviour recover, then test for relapse. Baselines A and B (exposed + reference) receive full exposure. Shuffle agents A, E, F (with reference B) receive partial therapy on the layouts where the uncontrollable theme is in the north-west quadrant; transfer is tested on untreated layouts. 8 runs in 2 waves, baselines first. Prediction from E7: therapy should succeed differently depending on the weight of cached values (heavy in shuffle A and E). | `experiments/E8_therapy_relapse/` (to come) | `configs/e8/` (to come) | `docs/E8_therapy_relapse.md`, `docs/e8/DECISIONS.md`, `docs/e8/EXPECTED_FORMS.md` |

The immunisation (prior exposure to control, once planned as E8) was dropped; its description is archived in
`docs/archive/E8_immunisation.md` and its code stays in E5 (`run_e5.py --training immunisation`).

The previous plan for E4-E6 (generalisation pilot, generalisation campaign, latent perturbations) was abandoned
on 2026-10-14; its files are kept in `docs/archive/plan_2026-09/`.

## Where things live

- Shared library: `src/helplessness/` (see `CLAUDE.md`, "Structure").
- Results of a cluster batch: an archive made by `export_batch.py`, analysed locally; light outputs committed
  (`docs/RESULTS_CONVENTION.md`; the orphan-branch convention `results/<batch>` was not used from E4 on).
- Paper and reports: `paper/` (`paper/README.md`).
- Presentation figures: `slides/` (scripts `slides/figures*/`, outputs `slides/out*/`, `slides/MANIFEST*.md`).
