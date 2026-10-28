# E7 — The causal test: correcting h

**Status: code ready (branch `e6`), waiting for E5's agents and E6's curves.** What it tests, its config and its status:
`docs/E7_latent_manipulation.md`; decisions `docs/e6/DECISIONS.md`; report
`experiments/E6_phenotypes_latent/IMPLEMENTATION_REPORT.md`.

| Script | What it does |
|---|---|
| `run_e7_directions.py` | the corrections of h on context walks, the common direction, alignment, efficacy, stability |
| `run_e7_apply.py` | E6's episodes with the intervention (direction, random, stepwise; reverse on the reference), every alpha |
| `analyse_e7.py` | directions table, alpha = 0 identity with E6, checks, E7's points on E6's curves |

Cluster: `experiments/E6_phenotypes_latent/submit_kubeflow.py --phase e7-directions` then `--phase e7-apply` (it reads the
directions under `results/E7_latent_manipulation/`). Local (from the repo root):

```
python experiments/E7_latent_manipulation/run_e7_directions.py --agents configs/e6/agents_dev_e4.yaml --runs-root <root> --device cpu
python experiments/E7_latent_manipulation/run_e7_apply.py --agents configs/e6/agents_dev_e4.yaml --runs-root <root> \
    --directions "results/E7_latent_manipulation/e7dir__*" --device cpu
python experiments/E7_latent_manipulation/analyse_e7.py --e6-runs "results/E6_phenotypes_latent/e6__*" \
    --apply-runs "results/E7_latent_manipulation/e7apply__*" --directions-runs "results/E7_latent_manipulation/e7dir__*" --out <dir>
```
