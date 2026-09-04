# 04 -- rescue via structure injection

**What was tested.** A second rescue route, contrasted with the
controllability-dial amplification of `experiments/03_intervention/`:
`T_mu = (1-mu)*T_hat + mu*T_det` progressively replaces the agent's learned
(possibly wrong or noisy) structure with the true deterministic tensor,
rather than rescaling whatever structure the belief already has. Swept over
agents trained at high p_train (0.7, 0.85, 0.95, 1.0), both budgets, 10
seeds. Requires agents already trained by
`experiments/01_offline_sweep/run.py`.

**What came out.** See `reports/week1_offline_report.md` §3.5: injection
restores oracle-level behaviour at every p_train including 1.0, with
40-75% of states changing action -- where amplification (lam<0) does
essentially nothing at p_train >= 0.85. The dissociation is the sharpest
result of the week: the high-p deficit is informational (the belief has
lost its geometry), not one of amplitude.

**How to reproduce.**

```bash
python run.py --config config.yaml
```

Then, from `experiments/03_intervention/`:
`python plot_intervention.py --stage comparison` to produce
`reports/rescue_two_routes.png`, which reads this experiment's
`results/rescue_injection.csv` alongside `results/intervention.csv` and
`results/lam_policy_change_counts.csv`.
