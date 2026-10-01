# GPU batch handoff -- gpu_e30b_measures

Written 2026-10-01 17:21 UTC by make_handoff.py from `measure-20261001T170610310630Z`. No weights (.pt) in any archive.

## Preflight

- `preflight-20261001T165336154684Z`: decision **launch**, parallel 4, CPUs requested / usable 3 / 3, projected batch hours 0.21934990354218623, problems none, warnings none

## Runs

Measurement batch: no training; each run re-measured by recompute_measures.py from its own weights.

| job | exit code | run | output | finished | identical to the original arrays | code source_sha256 | weights SHA-256 |
|---|---|---|---|---|---|---|---|
| win64_seed0_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z | grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z/recomputed/final__20261001T170634Z | yes | True | dae7ff2e302b16dc6744833e04f040098f5076082ef3414eeec8a3b291fddb2d | f60377e0e7a8e46f9e24fd7af203eb724b6d35a7f56ce1033476358daf2f9ae2 |
| win64_seed0_reference | 0 | grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z | grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z/recomputed/final__20261001T170634Z | yes | True | dae7ff2e302b16dc6744833e04f040098f5076082ef3414eeec8a3b291fddb2d | fa60ea69d694db6fd062171296f7af107f9b71f4609491353831a0f5effecbb9 |
| win64_seed5_experimental | 0 | grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z | grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z/recomputed/final__20261001T170634Z | yes | True | dae7ff2e302b16dc6744833e04f040098f5076082ef3414eeec8a3b291fddb2d | 9cc40755cc6110ac0a5e15805f3cf6853fda1bdd1528d34f2b52cc0d4bef6038 |
| win64_seed5_reference | 0 | grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z | grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z/recomputed/final__20261001T170634Z | yes | True | dae7ff2e302b16dc6744833e04f040098f5076082ef3414eeec8a3b291fddb2d | 690ea27d22a4a87eb8a4f310c0eac0fd3925525fd668b6ff006610c948e96fa2 |

## Archives

- `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z.recomputed.tgz`: 9.1 MB
- `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z.recomputed.tgz`: 9.0 MB
- `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z.recomputed.tgz`: 9.2 MB
- `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z.recomputed.tgz`: 8.8 MB
- `logs.tgz`: 45.2 MB
