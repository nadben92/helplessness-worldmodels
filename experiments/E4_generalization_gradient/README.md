# E4 — Generalisation gradient

What it tests, the design and the status: `docs/E4_generalization_gradient.md`. Tracker: `docs/EXPERIMENTS.md`. Launching
wave 1 on the GPU cluster: `RUN_GPU.md`.

Orchestration only. The shared code is in the library: the environments (`envs/layouts/two_zones.py`,
`pipeline/two_zones.py`), the colour axes (`envs/appearance/theme_axes.py`), the probe set (`probing/theme_probes.py`),
the walks and the measure (`probing/walks.py`, `probing/fixed_trajectory.py`), E3's closed-loop training
(`pipeline/agent_run.py`, through `ClosedLoopHooks`), the cluster tooling (`cluster/`).

## Files

| File | Role |
|---|---|
| `choose_themes.py` | Chooses the two colour pairs, the agent's colour and the axes; writes `configs/e4/themes.yaml` (deterministic) |
| `e4_setup.py` | Shared set-up: config, colours, a configuration's worlds, walks, measurement seeds |
| `run_e4.py` | One training run (configuration x condition): E3's closed loop with E4's world and checkpoint records |
| `measure_probes.py` | The zero-data probes of a configuration at a checkpoint (93 images, both models) |
| `export_batch.py` | One archive per batch (light files, final weights, probes, logs, manifest) and its check |
| `pilot_analysis.py` | The pilot's analyses: convergence and B, protocol control, evolution in the zone, forgetting, probes |
| `batches.py` | The two waves (and the optional diagonal-2 replicas) |
| `submit_kubeflow.py` | Submission to the GPU cluster: training, resume, probes at a checkpoint (push optional, off) |
| `render_check.py` | Renders the environments and some probes for a visual check (outside the repo) |
| `RUN_GPU.md` | Launch instructions for wave 1 |

Configs: `configs/e4/e4.yaml` (every run), `configs/e4/e4_mini.yaml` (CPU smoke test), `configs/e4/themes.yaml`
(generated).

## Budget: checkpoints and equivalence

The weights are saved at every checkpoint of `checkpoints.grid` (E2's grid to 100,000 steps, then every 25,000 steps,
plus E2's 158,489 and 251,189, to 300,000), on the cluster's volume, so the probes can be measured at any of them.

Nothing in the training depends on the executed length, but one thing depends on the **planned** budget
(`budget.total_steps`): after the estimator switch, `beta_rep` ramps up over `train.rep_warmup_steps` gradient steps,
and `null` (E3's value, kept) means 20% of the planned gradient steps left after the switch
(`pipeline/agent_run.py`, `rep_warmup`). Everything else -- the learning-rate schedule (two phases, on the switch), the
switch and its absolute deadline, epsilon (constant), the agent's heads, the replay (sampled over what it holds) --
does not depend on the budget, and the measurements at the checkpoints never touch the training streams. Hence
(`tests/e4/test_e4_training.py`, exact equality of the weights):

- a run planned for 300,000 steps and stopped at B (`--max-env-steps B`) has at B exactly the weights of a full run's
  checkpoint B; wave 2 runs this way (`batches.py`), with its checkpoint diagnostics only at B and at control
  checkpoints (`--diagnostics-at`);
- a run *planned* for B steps is not the same run (its `beta_rep` ramp is shorter).

## Running

```bash
# one run (CPU smoke configuration)
python experiments/E4_generalization_gradient/run_e4.py --config configs/e4/e4_mini.yaml --configuration p1-a-d1 --condition experimental --device cpu
# the probes of a configuration at a checkpoint (both runs' weights needed)
python experiments/E4_generalization_gradient/measure_probes.py --configuration p1-a-d1 --checkpoint <B> \
    --experimental-run <dir> --reference-run <dir> [--device cuda]
```

A run writes `results/E4_generalization_gradient/<run name>/` (see `run_e4.py`'s docstring); `export_batch.py create`
gathers a batch into one archive (`exports/`), `export_batch.py verify` checks it once extracted, and every analysis reads
the extracted directory (`--export`). Pushing to a branch `results/<batch>` (`--push-batch`) is optional and off.

Cost of the probes: I_s is computed only at the steps in the zone of interest (the corridor is filtered, never
decoded). With n_in = 10 steps in the zone for the grid and 25 for the six extreme pairs (training image x 4 zones, theme
only, position only), 300 walks, 16 draws per action, one configuration costs about **3.6 hours on CPU** (8 threads; 7.5
hours with n_in = 25 everywhere): 126 s per grid pair and 293 s per extreme pair for the two models, extrapolated from
30 walks (2026-10-16), with E3's trained weights of the same architecture -- their prior is diffuse on E4's images, so
this overestimates E4's own models. On the cluster, decoding and matching run on the GPU (`--phase probes --n-walks 30`
times it in a tenth of the cost). Local analyses with the weights: forgetting about 2-3 minutes, the extreme probes
about 30 minutes per configuration, on CPU.

## Pilot analyses (wave 1, local)

`RUN_GPU.md`, steps 4-6: export the batch on the cluster, extract the archive locally, `export_batch.py verify`, then
`pilot_analysis.py convergence | protocol | evolution | forgetting --export <dir>` (B in `budget_choice.json`), the probes
at B on the cluster (`submit_kubeflow.py --phase probes --checkpoint <B>`), a second export, and
`pilot_analysis.py probes | sufficiency --export <dir>` (the grid, the extreme probes, the number of walks for wave 2).
Then write B into `batches.py` (`B_AFTER_PILOT`) before launching wave 2.
