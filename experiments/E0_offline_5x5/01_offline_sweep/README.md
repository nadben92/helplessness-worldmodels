# 01 -- offline sweep

**What was tested.** Protocol A: offline collection of a fixed transition
budget with a uniform random policy (identical across every value of p),
training of a tabular `T_hat`/`R_hat`, planning by value iteration, then
evaluation at p = 0 with learning disabled and the planner frozen. Swept
over p in {0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0}, 10 seeds, at two transition
budgets (5k, 50k).

A mid-week correction is included here rather than as a separate
experiment: the original evaluation protocol (fixed start state, `np.argmax`
tie-break) produced an artefact -- high-p agents pushed against the
top-left border for 100 steps, an indexing/position artefact rather than a
behavioural result. `eval_v2.py` reevaluates the same saved agents (no
retraining) under random starts + random tie-break, alongside the two other
combinations, so the artefact is visible rather than silently fixed.

**What came out.** See `reports/week1_offline_report.md` §3.1: a graded
(not threshold) collapse of goal-directed behaviour as p increases, with
`T_hat` itself remaining an accurate estimate of the training dynamics at
every p (TV(T_hat, T_true) stays below 0.06) -- the deficit is a transfer
problem, not a learning problem. `reports/eval_v2_comparison.png` and
`reports/offline_sweep_curves.png` are the corresponding figures.

**How to reproduce.**

```bash
python run.py --config configs/sweep_5000.yaml
python run.py --config configs/sweep_50000.yaml
python eval_v2.py --config configs/eval_v2_5000.yaml
python eval_v2.py --config configs/eval_v2_50000.yaml
python plot_sweep.py --csv 5000:results/offline_sweep_5000.csv --csv 50000:results/offline_sweep_50000.csv
python plot_eval_v2.py --csv 5000:results/offline_eval_v2_5000.csv --csv 50000:results/offline_eval_v2_50000.csv
```

`reanalyze.py --budget <5000|50000>` recomputes `results/offline_sweep_<budget>.csv`
from the saved agents without retraining -- use it after a metrics bugfix,
never after a change to the environment or the collection policy (those
require rerunning `run.py`, which retrains).

Every CSV this experiment writes carries an adjacent `<name>.config.yaml`
recording exactly the parameters that produced it.
