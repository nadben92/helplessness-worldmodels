# E5 — Manipulating the belief through training

What it tests, the design and the status: `docs/E5_belief_training.md`; decisions: `docs/e5/DECISIONS.md`. Tracker:
`docs/EXPERIMENTS.md`. Launch on the GPU cluster: `RUN_GPU.md`. Implementation and shape tests: `IMPLEMENTATION_REPORT.md`.

Orchestration only. The shared code is in the library: E4's set-up, checkpoint records and probes
(`pipeline/two_zones_setup.py`, `pipeline/two_zones_records.py`, `probing/theme_probe_measure.py`, moved there from E4 on
2026-10-19), the episode sources (`pipeline/episode_sources.py`: shuffled zone assignments, the dynamics switch), the
measures by assignment (`pipeline/assignment_records.py`), the assignment stream (`streams/assignment.py`), and the
options of E3's closed loop (`pipeline/agent_run.py`: episode source, run length past the planned budget, resumable
states, branching; `agents/worldmodel/replay.py`: per-episode appearances and per-step backgrounds), all off by default.

## Files

| File | Role |
|---|---|
| `e5_setup.py` | Shared set-up: the config (E4's + `e5`), seeds A and B, finding a run (`e5_names.py`: names, standard library only) |
| `run_e5.py` | One training run: baseline, shuffle (`--fraction`) or immunisation (`--switch-step`, `--branch-from`) |
| `choose_switch_step.py` | Proposes the immunisation's switch step from E4's reference run (recomputes missing diagnostics) |
| `measure_probes.py` | E4's zero-data probes of an E5 pair at checkpoint k (post-switch for the immunisation) |
| `rsa_e5.py` | The RSA of a run at a checkpoint (E3's `measures.rsa`; (assignment, state) points under shuffle) |
| `time_assignment_records.py` | The cost of the checkpoint measures by assignment |
| `batches.py` | The three waves (standard library only) |
| `submit_kubeflow.py` | Submission to the GPU cluster: training, resume, probes |
| `export_batch.py` | One archive per batch and its check |

Configs: `configs/e5/e5.yaml` (every run), `configs/e5/e5_mini.yaml` (CPU smoke test).

## Running (CPU smoke configuration)

```bash
A=experiments/E5_belief_training
python $A/run_e5.py --config configs/e5/e5_mini.yaml --configuration p1-a-d1 --seed-label A --training shuffle --fraction 1.0 \
    --condition experimental --device cpu --out-dir scratch/e5
python $A/run_e5.py --config configs/e5/e5_mini.yaml --configuration p1-a-d1 --seed-label B --training baseline \
    --condition reference --save-resumable-at candidates --device cpu --out-dir scratch/e5
python $A/run_e5.py --config configs/e5/e5_mini.yaml --configuration p1-a-d1 --seed-label B --training immunisation \
    --switch-step 200 --branch-from auto --device cpu --out-dir scratch/e5
python $A/measure_probes.py --checkpoint 400 --find-runs scratch/e5 --config configs/e5/e5_mini.yaml --configuration p1-a-d1 \
    --training shuffle --fraction 1.0 --seed-label A --images extreme --device cpu --out-dir scratch/e5
python $A/rsa_e5.py scratch/e5/E5_belief_training/<run> --checkpoint 600 --max-per-state 5 --out scratch/e5/rsa
```

Tests: `tests/e5/` (assignments and pairing; training shapes: f = 0 = E4, replay invariants, backgrounds, immunisation;
batches).
