# GPU batch handoff -- gpu_pilot_noise

Written 2026-10-07 08:11 UTC by make_handoff.py from `measure-20261006T184805129601Z`. No weights (.pt) in any archive.

## Preflight

- `preflight-20261006T184406428649Z`: decision **launch**, parallel 4, CPUs requested / usable 3 / 3, projected batch hours 2.323783842467755, problems none, warnings none

## Runs

Measurement batch: no training; each run re-measured by recompute_measures.py from its own weights.

| job | exit code | run | output | finished | identical to the original arrays | code source_sha256 | weights SHA-256 |
|---|---|---|---|---|---|---|---|
| win64_seed0_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z/*.json | yes | - | - | - |
| win64_seed5_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z/*.json | yes | - | - | - |
| win64_seed0_reference | 0 | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z/*.json | yes | - | - | - |
| win64_seed5_reference | 0 | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z/*.json | yes | - | - | - |
| goal_control | 0 | goal_control | goal_control/*.json | yes | - | - | - |

## Archives

- `grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z.recomputed.tgz`: 0.0 MB
- `grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z.recomputed.tgz`: 0.0 MB
- `grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z.recomputed.tgz`: 0.0 MB
- `grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z.recomputed.tgz`: 0.0 MB
- `goal_control.recomputed.tgz`: 0.0 MB
- `logs.tgz`: 0.0 MB

## Stop

The first handoff attempt used the cluster virtual environment instead of the runbook's `python` command:

```bash
.venv-cluster/bin/python experiments/E3_worldmodel_4zones/make_handoff.py --measure-dir logs/kubeflow/measure-20261006T184805129601Z --preflight-dir logs/kubeflow/preflight-20261006T184406428649Z --out handoff/gpu_pilot_noise
```

It failed before creating the handoff, with the complete output:

```text
Traceback (most recent call last):
  File "/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/make_handoff.py", line 40, in <module>
    import yaml
ModuleNotFoundError: No module named 'yaml'
command terminated with exit code 1
```

The runbook's `python experiments/E3_worldmodel_4zones/make_handoff.py` command then succeeded with the same arguments; the archives listed above are its output. No measurement was retried or changed.
