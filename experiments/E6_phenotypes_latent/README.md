# E6 — The behavioural signature of the controllability belief

**Status: code ready (branch `e6`), waiting for E5's agents.** What it tests, its config and its status:
`docs/E6_phenotypes_latent.md`; decisions `docs/e6/DECISIONS.md`; tracker `docs/EXPERIMENTS.md`; implementation report
and preliminary results on E4's weights: `IMPLEMENTATION_REPORT.md`.

Orchestration only (shared code in `src/helplessness/`, CLAUDE.md):

| Script | What it does |
|---|---|
| `run_e6.py` | the episodes, beliefs and phenotypes of one or more agents (an agents file), every test image |
| `analyse_e6.py` | the curves, the held-out tests, monotonicity, the localisation confound, from run_e6's outputs |
| `batches.py` | the batch table (an agents file per batch; phases e6, e7-directions, e7-apply) |
| `submit_kubeflow.py` | one phase of a batch on one GPU (`--dry-run` prints the jobs); E7's phases too |
| `export_batch.py` | one archive of a batch's E6 / E7 outputs, and its check |

Local run on E4's weights (from the repo root; `<root>` = the extracted E4 exports' `results/`):

```
python experiments/E6_phenotypes_latent/run_e6.py --agents configs/e6/agents_dev_e4.yaml --runs-root <root> --device cpu
python experiments/E6_phenotypes_latent/analyse_e6.py --runs "results/E6_phenotypes_latent/e6__*" --out <analysis dir>
```
