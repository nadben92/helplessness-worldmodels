# Robustness diagnostic (E5, frozen weights, local CPU)

Why some shuffle models stay localised on new colours of the swap probes (F) and others lose the agent (A, E).
Predictions and thresholds: PREDICTIONS.md (committed before the mechanistic measurements); results: REPORT.md.

| file | what |
|---|---|
| models.yaml | the models: run globs, cluster swap-probe directory and model name |
| common.py | model table, weights and probe lookup, swap images, cluster localisation, scores |
| inventory_scores.py | step 1: inventory (weights, SHA-256 against the probes' meta.yaml), cluster localisation and scores |
| local_localisation.py | step 1: consistency check, the swap probes' diagnostic recomputed locally |
| stimuli.py | step 4's images (colours fixed in PREDICTIONS.md) |
| mechanism.py | steps 2-4: a1 / a2 / b localisation, posterior statistics, sensitivity, per walk |
| analyse.py | tables and verdicts from mechanism.py's outputs |
| training_history.py | step 5 (H4): the runs' records only |
| fig_*.py | the figures |

Raw per-walk outputs (`mechanism.py --out-dir`) stay outside the repo; `tables/` holds what the report reads.
Local roots used (Windows): runs `C:\nrw\e5_extends\extracted\e5_extend_{a,b}__*\results`,
`C:\nrw\e6all\helplessness-worldmodels-{e4,e5}\results`; probes `C:\nrw\e5_extends\extracted\e5_probes_swap__20261103T085130Z\results\E5_belief_training`.
