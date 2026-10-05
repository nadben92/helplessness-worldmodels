# GPU batch handoff -- gpu_pilot_remeasure

Written 2026-10-05 18:20 UTC by make_handoff.py from `measure-20261005T170355207529Z`. No weights (.pt) in any archive.

## Preflight

- `preflight-20261005T165021437348Z`: decision **launch**, parallel 4, CPUs requested / usable 3 / 3, projected batch hours 1.072238067019498, problems none, warnings none

## Runs

Measurement batch: no training; each run re-measured by recompute_measures.py from its own weights.

| job | exit code | run | output | finished | identical to the original arrays | code source_sha256 | weights SHA-256 |
|---|---|---|---|---|---|---|---|
| win64_seed0_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| win64_seed5_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| win64_seed0_reference | 0 | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| win64_seed5_reference | 0 | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| value_diagnostics | 1 | value_diagnostics | none | no | - | - | - |

## Archives

- `grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z.recomputed.tgz`: 44.0 MB
- `grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z.recomputed.tgz`: 43.4 MB
- `grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z.recomputed.tgz`: 44.6 MB
- `grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z.recomputed.tgz`: 45.3 MB
- `logs.tgz`: 186.3 MB

## Stop

The prescribed value-diagnostics job exited with code 1 and was not retried. The handoff script also stopped because its log archive exceeded 90 MB. Nothing was committed or pushed.

Value-diagnostics command:

````text
/scratch/helplessness-worldmodels/.venv-cluster/bin/python -u experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py --points /scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z:10000 /scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z:15849 /scratch/helplessness-worldmodels/results/E3_worldmodel_4zones/grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z:39811 --device cuda --out-dir /scratch/helplessness-worldmodels/logs/kubeflow/measure-20261005T170355207529Z/runs/value_diagnostics
````

Complete value-diagnostics output:

````text
/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py:55: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  rho = float(spearmanr(x, y).correlation)
/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py:59: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  r = spearmanr(x[i], y[i]).correlation
Traceback (most recent call last):
  File "/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py", line 168, in <module>
    sys.exit(main())
             ^^^^^^
  File "/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py", line 160, in main
    results.append(diagnose_point(Path(run_dir), int(step), args.device))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py", line 140, in diagnose_point
    "spearman_V_vs_distance_to_goal": spearman_with_ci(V[keep], dist[keep], np.random.default_rng(SEED + 1)),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/helplessness-worldmodels/experiments/E3_worldmodel_4zones/diagnose_value_generalisation.py", line 62, in spearman_with_ci
    lo, hi = np.quantile(boots, [0.025, 0.975])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py", line 4496, in quantile
    return _quantile_unchecked(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py", line 4510, in _quantile_unchecked
    return _ureduce(a,
           ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py", line 3880, in _ureduce
    r = func(a, **kwargs)
        ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py", line 4669, in _quantile_ureduce_func
    result = _quantile(arr,
             ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/numpy/lib/_function_base_impl.py", line 4793, in _quantile
    slices_having_nans = np.isnan(arr[-1, ...])
                                  ~~~^^^^^^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
````

Handoff command:

````text
python experiments/E3_worldmodel_4zones/make_handoff.py --measure-dir logs/kubeflow/measure-20261005T170355207529Z --preflight-dir logs/kubeflow/preflight-20261005T165021437348Z --out handoff/gpu_pilot_remeasure
````

Complete handoff command output:

````text
# GPU batch handoff -- gpu_pilot_remeasure

Written 2026-10-05 18:20 UTC by make_handoff.py from `measure-20261005T170355207529Z`. No weights (.pt) in any archive.

## Preflight

- `preflight-20261005T165021437348Z`: decision **launch**, parallel 4, CPUs requested / usable 3 / 3, projected batch hours 1.072238067019498, problems none, warnings none

## Runs

Measurement batch: no training; each run re-measured by recompute_measures.py from its own weights.

| job | exit code | run | output | finished | identical to the original arrays | code source_sha256 | weights SHA-256 |
|---|---|---|---|---|---|---|---|
| win64_seed0_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| win64_seed5_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z | grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| win64_seed0_reference | 0 | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| win64_seed5_reference | 0 | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z | grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z/env* | yes | - | 84d41d62e18c7801a9213b0521a126933f9754930bd792fd50836104add54001 | - |
| value_diagnostics | 1 | value_diagnostics | none | no | - | - | - |

## Archives

- `grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z.recomputed.tgz`: 44.0 MB
- `grid12x12__4zones__pair-experimental__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z.recomputed.tgz`: 43.4 MB
- `grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed00__20261003T144527Z.recomputed.tgz`: 44.6 MB
- `grid12x12__4zones__pair-control__worldmodel-agent-gpu_pilot_closed_loop__seed05__20261003T144527Z.recomputed.tgz`: 45.3 MB
- `logs.tgz`: 186.3 MB
STOP: archive(s) above 90 MB, not committable: [('logs.tgz', 186.30187225341797)]
````

## Delivery adjustment and separate diagnostic attempt

The original measurement batch and its failed value-diagnostics job are unchanged. The full 186.3 MiB `logs.tgz` remains on the cluster at `handoff/gpu_pilot_remeasure/logs.tgz` (SHA-256 `eca2b610f251427bda17362ffb3f6153712314bfd0fdaae0fa865956bfcffb9b`), but is **not included in this GitHub handoff**. `logs_summary.tgz` includes the complete supervisor and worker logs, submission records and preflight log/verdict; it omits `runs/` data already delivered in the four separate result archives and the large preflight smoke output. The smoke output remains on the cluster under `logs/kubeflow/preflight-20261005T165021437348Z/runs/smoke`.

After the original failure, a **separate** run of only the prescribed value diagnostic succeeded with fix commit `90c546597dd0c9b191adb8ee79bc7613f9e0266a`; its three results and log are in `value_diagnostics_retry.tgz`. The original exit code 1 remains recorded above and in `logs_summary.tgz`. At REF seed 0 step 10,000, constant critic values make Spearman and its interval undefined (JSON null); the other two points have finite estimates. See `delivery_provenance.json` for both attempts and the original archive checksum.
