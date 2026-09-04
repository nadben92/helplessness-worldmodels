# 02 -- component swaps (aetiology)

**What was tested.** A model-based agent separates `T_hat`, `R_hat`, and a
planner fixed across all agents. Swapping `T_hat`/`R_hat` between a low-p and
a high-p donor (both trained at budget=50k), replanning, and evaluating at
p = 0 localises which component carries the transfer deficit. Seven
conditions per (low_p, high_p) pair: `intact_low`, `intact_high`,
`T_high_R_low`, `T_low_R_high`, `T_true_R_high`, `T_high_R_true`, `oracle`.
Requires agents already trained by `experiments/01_offline_sweep/run.py`
(budget=50000) -- trains nothing itself.

**What came out.** See `reports/week1_offline_report.md` §3.2: the
phenotype tracks `T_hat` exactly (`T_high_R_low` behaves like `intact_high`,
`T_low_R_high` like `intact_low`), identically across all three high-p
donors. `R_hat` is intact and irrelevant.

**How to reproduce.**

```bash
python run.py --config config.yaml
```

Requires `results/agents/budget50000/` to already exist. Writes
`results/swaps.csv` and `results/swaps.csv.config.yaml`.
