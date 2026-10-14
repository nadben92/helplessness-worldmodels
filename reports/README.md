# reports/

Frozen. This directory holds outputs of the finished experiments whose paths are written into their code
(E0's plotting scripts, E3's diagnostics) or into the paper's number scripts; nothing is moved out of it, and
no new experiment writes here (E4 onwards: `results/` and `docs/`, see `docs/EXPERIMENTS.md`).

| Content | Origin |
|---|---|
| `week1_offline_report.md` | E0's report (first week, tag `v0.1-offline`) |
| `offline_sweep_curves.png`, `eval_v2_comparison.png` | E0 01: `plot_sweep.py`, `plot_eval_v2.py` |
| `intervention_*.png`, `rescue_two_routes.png` | E0 03: `plot_intervention.py` (paths written in the script) |
| `gridworld_layout.png` | `scripts/plot_gridworld.py` (E0's 5x5 layout) |
| `E3_GPU_2026-09-28.md` | E3: handoff note of the first GPU batch (gpu_e30) |
| `E3_GPU_gpu_e30b_analysis_2026-09-30.md`, `figures/gpu_e30b_*.png` | E3: analysis of the gpu_e30b batch, by `experiments/E3_worldmodel_4zones/gpu_e30b_analysis.py` and `gpu_e30b_figures.py` (in `reports/` until 2026-10-14) |
| `tau_gpu_e30b/` | E3: `experiments/E3_worldmodel_4zones/tau_gpu_e30b.py` (reading-rule thresholds on gpu_e30b re-measured) |
| `diagnostics_<run>/`, `diagnostics_wm_goal_<run>/`, `diagnostics_wm_goal_ranks_<run>/` | E3: `diagnose_agent_run.py`, `diagnose_world_model_goal.py` on the agent_mini bench run (default output directory of those scripts); read by `paper/numbers/e3_numbers.py` |
| `gradients_vs_data_<run>/` | E3: `gradients_vs_data.py` (same run); its `rssm_*.pt` weights are not versioned |
