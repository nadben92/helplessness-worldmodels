# E0 — Offline tabular work on the 5x5 gridworld (finished, frozen)

The project's first experiments, before the 4-zone environment: `docs/E0_offline_5x5.md`, report
`reports/week1_offline_report.md` (tag `v0.1-offline`; final state tag `e0-final`).

| Phase | Directory | Entry points |
|---|---|---|
| 01 offline sweep | `01_offline_sweep/` | `run.py`, `eval_v2.py`, `reanalyze.py`, `plot_sweep.py`, `plot_eval_v2.py` |
| 02 component swaps | `02_swaps/` | `run.py` |
| 03 controllability dial | `03_intervention/` | `run.py`, `rescue_diagnostics.py`, `plot_intervention.py` |
| 04 rescue by structure injection | `04_rescue_injection/` | `run.py` |
| 05 online prototype (not a paper phase) | `05_online_prototype/` | `train.py`, `recovery.py` |

Each phase's README gives its command and outputs (`results/<name>.csv` + `<name>.csv.config.yaml`, figures in
`reports/`).

**Known defect, left as is (E0 is frozen):** the repo `Makefile`'s `agents` target calls
`scripts/regenerate_agents.py`, deleted when the phases were nested here (commit 4210db5); `make sweep`, which
depends on it, fails too. Regenerate the saved agents with `01_offline_sweep/run.py --config
configs/sweep_<budget>.yaml` instead.
