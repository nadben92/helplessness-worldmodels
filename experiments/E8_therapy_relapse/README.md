# E8 — therapy and relapse: orchestration

Design: `docs/E8_therapy_relapse.md`. Decisions: `docs/e8/DECISIONS.md`. Forms, thresholds and predictions (committed
before any computation): `docs/e8/EXPECTED_FORMS.md`. Report: `IMPLEMENTATION_REPORT.md`.

| File | Role |
|---|---|
| `e8_setup.py` | config loading (E5's config + the `e8` block), the start models and their SHA-256 check, run names |
| `e8_therapy.py` | the therapy world, episode source and checkpoint record (hooks of `pipeline.agent_run.run_one`) |
| `run_e8.py` | `--phase train` (a new therapy run from a start file), `resume`, `probes` |
| `probes_e8.py` | the frozen measures of a probe job (the pairs sharing a reference, one checkpoint) |
| `check_identity.py` | form F0: checkpoint 0 against the 650,000 campaign, exact |
| `analyze_e8.py` | curves, recovery times, forms R and M, predictions P1-P4, decomposition, figures |
| `batches.py` | the batch table (`e8_w1_base`, `e8_w2_shuf_nw`, ... ; `LAUNCHABLE`) |
| `submit_kubeflow.py` | cluster submission (train / resume / probes; `--dry-run`, `--status`) |
| `export_batch.py` | the batch's archive (`create` on the cluster, `verify` locally) |
| `chain_cpu.py` | the probes of an extracted batch on the local CPU, unattended (`--smoke` first) |

Configs: `configs/e8/e8.yaml` (the campaign), `configs/e8/agents.yaml` (start models: exact run directories, checkpoints,
SHA-256; the pairs), `configs/e8/e8_mini.yaml` (CPU tests). Tests: `tests/e8/`.

Every path is an argument: the start runs' root (`--runs-root`: `/scratch` on the cluster, a local copy with the same
layout here), the output root (`--out-dir`), the extracted batch (`--batch-root`). Light outputs only go to
`results/E8_therapy_relapse/`.
