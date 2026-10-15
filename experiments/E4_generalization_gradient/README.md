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
| `pilot_analysis.py` | The pilot's analyses: convergence and B, protocol control, evolution in the zone, forgetting, probes |
| `batches.py` | The four waves |
| `submit_kubeflow.py` | Submission of a wave to the GPU cluster (supervisor, automatic push of the results) |
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
  checkpoint B; waves 2-4 run this way (`batches.py`);
- a run *planned* for B steps is not the same run (its `beta_rep` ramp is shorter).

## Running

```bash
# one run (CPU smoke configuration)
python experiments/E4_generalization_gradient/run_e4.py --config configs/e4/e4_mini.yaml --configuration p1-a-d1 --condition experimental --device cpu
# the probes of a configuration at a checkpoint (both runs' weights needed)
python experiments/E4_generalization_gradient/measure_probes.py --configuration p1-a-d1 --checkpoint <B> \
    --experimental-run <dir> --reference-run <dir> [--device cuda]
```

A run writes `results/E4_generalization_gradient/<run name>/` (see `run_e4.py`'s docstring); with `--push-batch` its
light files go to the branch `results/<batch>` at the end (`docs/RESULTS_CONVENTION.md`).

Cost of the probes: the full measure of one configuration (96 image x zone pairs, two models, 300 walks, 16 draws per
action) costs about **7.5 hours on CPU** (8 threads): 280 s per image x zone pair for the two models, measured on 6 pairs
(2026-10-15) and extrapolated to the 96 -- a full run, stopped after 6 hours, agrees. It was timed with E3's trained
weights (same architecture), whose prior is diffuse on E4's images (66% of the draws distinct), so it overestimates
the cost for E4's own models. Most of it is the decoding and the template matching, which run on the GPU on the
cluster.

## Pilot analyses (wave 1, local)

```bash
A=experiments/E4_generalization_gradient
R=../results-e4_wave1/results/E4_generalization_gradient/e4_wave1     # the results branch, git worktree
python $A/pilot_analysis.py convergence --runs $R/*/ --out <dir>      # criterion, B (budget_choice.json), figure
python $A/pilot_analysis.py protocol --runs $R/*/ --out <dir>         # b fixed walks vs E3's end-of-training b
python $A/pilot_analysis.py evolution --runs $R/*/ --out <dir>        # b by step since entry
# with the weights of checkpoint B copied from the volume (results/E4_generalization_gradient/<run>/checkpoints/):
python $A/measure_probes.py --configuration p1-a-d1 --checkpoint <B> --experimental-run <dir> --reference-run <dir>
python $A/pilot_analysis.py forgetting --configuration p1-a-d1 --checkpoint <B> --experimental-run <dir> --reference-run <dir> --out <dir>
python $A/pilot_analysis.py probes --probes <measure_probes output dirs> --out <dir>   # extreme probes and the full grid
```

Then write B into `batches.py` (`B_AFTER_PILOT`) before launching waves 2-4.
