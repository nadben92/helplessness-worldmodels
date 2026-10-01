# GPU batch handoff -- gpu_e30b

Written 2026-09-30 09:48 UTC by make_handoff.py from `train-20260929T183401410873Z`. No weights (.pt) in any archive.

## Preflight

- `preflight-20260929T174511312153Z`: decision **launch**, parallel 8, projected batch hours 18.0645418612249, problems none, warnings none

## Runs

| job | exit code | status (meta.yaml) | gradient steps done | switch step | finished (results.json) | error |
|---|---|---|---|---|---|---|
| win64_seed0_experimental | 0 | done | - | 6351 | yes | - |
| win64_seed0_reference | 0 | done | - | 5901 | yes | - |
| win64_seed5_experimental | 0 | done | - | 5501 | yes | - |
| win64_seed5_reference | 0 | done | - | 5501 | yes | - |
| win16_seed0_experimental | 0 | done | - | 6451 | yes | - |
| win16_seed0_reference | 0 | done | - | 7001 | yes | - |
| win16_seed5_experimental | 0 | done | - | 6351 | yes | - |
| win16_seed5_reference | 0 | done | - | 6651 | yes | - |

## Archives

- `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z.tgz`: 12.5 MB
- `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z.tgz`: 12.5 MB
- `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z.tgz`: 12.7 MB
- `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z.tgz`: 12.3 MB
- `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win16__seed00__20260929T183427Z.tgz`: 12.3 MB
- `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win16__seed00__20260929T183427Z.tgz`: 11.9 MB
- `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win16__seed05__20260929T183427Z.tgz`: 12.5 MB
- `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win16__seed05__20260929T183427Z.tgz`: 12.3 MB
- `logs.tgz`: 20.9 MB

## Selected final weights

These five files were added separately at a collaborator's request; the archives above still contain no weights. Each file is the source run's `checkpoints/weights_final.pt`:

| File in `weights/` | Source run | SHA-256 |
|---|---|---|
| `win64_seed00_pair-experimental_weights_final.pt` | `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed00__20260929T183427Z` | `f60377e0e7a8e46f9e24fd7af203eb724b6d35a7f56ce1033476358daf2f9ae2` |
| `win64_seed00_pair-control_weights_final.pt` | `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed00__20260929T183428Z` | `fa60ea69d694db6fd062171296f7af107f9b71f4609491353831a0f5effecbb9` |
| `win64_seed05_pair-experimental_weights_final.pt` | `grid12x12__4zones__pair-experimental__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z` | `9cc40755cc6110ac0a5e15805f3cf6853fda1bdd1528d34f2b52cc0d4bef6038` |
| `win64_seed05_pair-control_weights_final.pt` | `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win64__seed05__20260929T183427Z` | `690ea27d22a4a87eb8a4f310c0eac0fd3925525fd668b6ff006610c948e96fa2` |
| `win16_seed05_pair-control_weights_final.pt` | `grid12x12__4zones__pair-control__worldmodel-gpu_e30b_win16__seed05__20260929T183427Z` | `2387ef65317b8390b373a79f21ebade533c2498a10d7363118226842ae612bd5` |
