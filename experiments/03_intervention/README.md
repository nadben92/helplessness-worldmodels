# 03 -- controllability dial intervention

**What was tested.** The transition tensor decomposes exactly as
`T(p) = (1-p)*T_det + p*M`, with `M = mean_a T_det` invariant in p. This
gives a closed-form dial `T_lam = (1-lam)*T_hat + lam*M_hat` applicable to
any agent's learned belief: lam=0 leaves it intact, lam=1 collapses
controllability to zero, lam<0 amplifies existing structure. Swept over all
agents saved by `experiments/01_offline_sweep/run.py` (budgets 5k/50k,
p_train in the usual grid, 10 seeds), plus a reference curve computed
directly on the true tensor (no agent, no seed -- rows with `budget=oracle`).
`rescue_diagnostics.py` adds a mechanistic diagnostic on top: how many
states' greedy action actually changes under the dial, and how much of the
tensor gets clipped when lam < 0.

**What came out.** See `reports/week1_offline_report.md` §3.3-3.5:
plotting every phenotype against `p_effective` collapses (budget, p_train,
lam, seed) onto near-single curves for the quantitative/global metrics
(R² up to 0.91) but not for categorical/local ones (frac_trap, cycle
rates) -- perceived controllability predicts severity, not topography.
Amplification (lam<0) fails to rescue agents trained at high p (the belief
has no structure left to amplify); see `experiments/04_rescue_injection/`
for the route that does.

**How to reproduce.**

```bash
python run.py --config config.yaml
python rescue_diagnostics.py --config rescue_diagnostics_config.yaml
python plot_intervention.py --stage all
```

`plot_intervention.py --stage comparison` additionally reads
`results/rescue_injection.csv` (written by `experiments/04_rescue_injection/run.py`)
to produce the two-routes comparison figure -- run that experiment first.

Requires `results/agents/budget5000/` and `results/agents/budget50000/` to
already exist.
