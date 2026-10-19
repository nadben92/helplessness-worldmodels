# E5 on the GPU cluster: training, export, probes

E5 trains 12 runs on ONE configuration of E4 (chosen after E4's analysis), in three waves of four runs, each wave on one
GPU (four runs in parallel). It is launched from **its own clone**, `/scratch/helplessness-worldmodels-e5` (branch
`e5`): the e4 clone (`/scratch/helplessness-worldmodels-e4`) keeps E4's runs and probes and is never touched by E5,
except to read E4's results (the reference of immunisation A, the switch step).

| Wave | Runs | Needs | Expected duration |
|---|---|---|---|
| `e5_wave1` | seed A: shuffle f = 1 (exp, ref), shuffle f = 0.5 (exp, ref) | the configuration | about 45-50 h |
| `e5_wave2` | seed B: baseline (exp, ref; the reference keeps resumable states), shuffle f = 1 (exp, ref) | the configuration (the switch step if known) | about 45-50 h |
| `e5_wave3` | seed B: shuffle f = 0.5 (exp, ref); immunisation A (recomputed) and B (branched) | the switch step; wave 2 done | about 45-50 h if S + 175,000 <= 300,000, else about (S + 175,000) / 300,000 x 40-45 h |

The durations scale E4's 40-45 h for 300,000 steps (four runs in parallel) by the shuffle runs' extra checkpoint
measures (every assignment at every checkpoint: IMPLEMENTATION_REPORT.md, "cost"); a wave lasts as long as its longest run.

## Rules

1. Run the commands exactly as written. If anything happens that this file does not foresee, stop and report the
   complete output.
2. Never change a seed, a config, the batch or an option. A run that fails is a result: resume it only with the resume
   command, never relaunch it as a new run.
3. Nothing is committed on the cluster. Results stay on the volume and come back as one archive per batch.

## Step 0 -- on your own machine

Merge `e4` into `e5` (E4's latest analysis code), check the tests (`pytest tests/e4 tests/e5`), push `e5`.

## Step 1 -- in a terminal of the Kubeflow notebook: a clean clone of `e5`

```bash
cd /scratch
git clone --branch e5 --single-branch https://github.com/nadben92/helplessness-worldmodels.git helplessness-worldmodels-e5
cd /scratch/helplessness-worldmodels-e5
git log --oneline -1
git status --short
```

If the directory already exists, do not reuse it: stop.

## Step 2 -- the parameters

- `CONF`: the E4 configuration retained for E5 (e.g. `p1-a-d1`), set after E4's analysis.
- `S`: the immunisation's switch step, proposed by `choose_switch_step.py` on E4's REFERENCE run of `CONF` (the first
  checkpoint where the future UD slot has mass >= 0.97 and |I at the first step - log2 5| <= 0.1 bit; it draws the
  curves in `switch_step.png`). For a wave-1 configuration of E4 (`p1-*`, diagnostics at every checkpoint), locally from
  E4's export:

  ```bash
  python experiments/E5_belief_training/choose_switch_step.py --run <E4 export>/results/E4_generalization_gradient/<CONF reference run> --out ../e5_switch
  ```

  For a wave-2 configuration of E4 (`p2-*`, diagnostics at 10,000 / 100,000 / 300,000 only), on the cluster, where the
  weights of every checkpoint are (the missing checkpoints are recomputed from the weights and `replay_final.npz`):

  ```bash
  cd /scratch/helplessness-worldmodels-e5
  python experiments/E5_belief_training/choose_switch_step.py --recompute --device cpu \
      --run /scratch/helplessness-worldmodels-e4/results/E4_generalization_gradient/<CONF reference run> --out /scratch/e5_switch
  ```

  `S` stays your decision: the script only proposes it.
- `P`: the immunisation's steps after the switch, default 175,000 (`configs/e5/e5.yaml`).

## Step 3 -- wave 1 (check, then launch)

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --configuration $CONF \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --configuration $CONF \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

The dry run prints the four jobs (`A_sh100_experimental`, `A_sh100_reference`, `A_sh050_experimental`,
`A_sh050_reference`), each `run_e5.py --config configs/e5/e5.yaml --configuration $CONF --seed-label A --training shuffle
--condition ... --fraction ... --device cuda --out-dir results`. Then, as E4: `kubectl get pods`, `cat $OUT/status.json`,
`tail -n 3 $OUT/*.log` (`E5 checkpoint <step>: 12 assignment(s) ...` at each checkpoint), `nvidia-smi`.

**Check in the first hours of wave 1** (docs/e5/DECISIONS.md (19), (20)): every run computes its estimator switch on
pairs of images with the same background. In each job's log, the `JS` column of the `env step` lines and the line
`>>> SWITCH to straight-through ...`; in `checkpoints.jsonl`, `e5.estimator_switch` and `e5.decode_accuracy_canonical`
(the agent state's decodability). Report the switch step of each run and the decodability at the checkpoints around it.
A shuffle run stopped by `SeparationNotReachedError` (no switch within 25,000 gradient steps, about 41,000 environment
steps) is a result: report it, do not relaunch.

## Step 4 -- wave 2

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --configuration $CONF [--switch-step $S] \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --configuration $CONF [--switch-step $S] \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

The baseline reference of seed B (`B_base_reference`, `--save-resumable-at candidates [S]`) keeps its full state
(`checkpoints/resumable_env<N>.pt`, about 25-40 MB each) at E4's grid points from 10,000 to 250,000 and at `S` when
given: immunisation B branches from the one at `S`. If `S` is not in that list and was not given here, immunisation B
cannot branch: launch wave 2 with `--switch-step $S`.

## Step 5 -- wave 3 (after wave 2 has finished)

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --configuration $CONF --switch-step $S [--post-steps $P] \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --configuration $CONF --switch-step $S [--post-steps $P] \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

`A_imm` recomputes its phase 1 (E4 kept no full state of its reference); `B_imm` has `--branch-from auto` and starts
from wave 2's `resumable_env<S>.pt` (it fails at once if that file is missing: report it).

## Step 6 -- a run interrupted by the infrastructure: resume it

```bash
RUN=results/E5_belief_training/<the run's directory name>
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --phase resume --resume "$RUN" \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

A run stopped by a training guard (`SeparationNotReachedError`, `CollapseAfterSwitchError`) is a result: report it.

## Step 7 -- export, probes, export again

```bash
python experiments/E5_belief_training/export_batch.py create --batch <wave> --configuration $CONF
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --configuration $CONF --phase probes --checkpoint <k> \
    [--e4-results /scratch/helplessness-worldmodels-e4/results] --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --configuration $CONF --phase probes --checkpoint <k> \
    [--e4-results /scratch/helplessness-worldmodels-e4/results] --repo-dir /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/export_batch.py create --batch <wave> --configuration $CONF
```

`k` is in environment steps for baseline and shuffle, in steps AFTER THE SWITCH for the immunisation (whose weights are
at `switch + k`; its reference is the baseline reference of its seed at `k`). Wave 3 needs `--e4-results` (immunisation
A's reference is E4's reference run of `$CONF`). Locally: `export_batch.py verify <extracted dir>`, then the analyses
(`rsa_e5.py`, the probes).
