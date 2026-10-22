# E5 on the GPU cluster: training, export, probes

E5 trains on ONE configuration of E4, **`p1-b-d1`** (docs/e5/DECISIONS.md (21); seeds A = 2, B = 102), one wave per
GPU, plus one probes-only batch (`e5_probes_baseline_A`). It is launched from **its own clone**,
`/scratch/helplessness-worldmodels-e5` (branch `e5`): the e4 clone (`/scratch/helplessness-worldmodels-e4`) keeps E4's
runs and probes and is never touched by E5, except to read E4's results.

Since 2026-10-22 (docs/e5/DECISIONS.md (29)): the first wave 1 (f = 1 and f = 0.5, seed A) failed at the separation
deadline; the campaign is now:

| Wave | Runs | Expected duration |
|---|---|---|
| `e5_wave1` | shuffle f = 1, seeds A and B (exp, ref): 4 runs | about 45-50 h |
| `e5_wave2` | seed B's baseline (exp, ref; the reference keeps its full state at 100,000 steps, for E8) and shuffle f = 1 seed C (exp, ref): 4 runs | about 45-50 h |
| `e5_wave3` | TO BE CONFIRMED: seed C's baseline and shuffle f = 1 seed D (exp, ref): 4 runs | about 45-50 h |
| `e5_probes_baseline_A` | no training: E4's own pair of p1-b-d1 re-probed with E5's probe script (150 walks) | about 15 min |

Shuffle f = 0.5 is abandoned for E5 (docs/e5/DECISIONS.md (31)); the immunisation moved to E8 (both in `batches.DEFERRED`,
never submitted). The durations scale E4's 40-45 h for 300,000 steps (four runs in parallel) by the shuffle runs' extra
checkpoint measures; a wave lasts as long as its longest run.

## Rules

1. Run the commands exactly as written. If anything happens that this file does not foresee, stop and report the
   complete output.
2. Never change a seed, a config, the batch or an option. A run that fails is a result: resume it only with the resume
   command, never relaunch it as a new run.
3. Nothing is committed on the cluster. Results stay on the volume and come back as one archive per batch.

## Step 0 -- on your own machine

`e4` is merged into `e5` (2026-10-21) and the tests pass. Push `e4`, then `e5`.

## Step 1 -- in a terminal of the Kubeflow notebook: a clean clone of `e5`

The clone uses the same remote (and so the same read token, if the e4 clone's URL carries one) as the e4 clone, without
printing it:

```bash
cd /scratch
git clone --branch e5 --single-branch "$(git -C /scratch/helplessness-worldmodels-e4 remote get-url origin)" helplessness-worldmodels-e5
cd /scratch/helplessness-worldmodels-e5
git log --oneline -1
git status --short
```

If the directory already exists, do not reuse it: stop.

## Step 2 -- the parameters

- `CONF=p1-b-d1`: the E4 configuration of E5 (docs/e5/DECISIONS.md (21)); the default of `--configuration`, and the only
  value the submission accepts.
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

  `S` stays your decision: the script only proposes it. For p1-b-d1 it proposes **`S=100000`** (docs/e5/DECISIONS.md (27);
  run locally from E4's export: mass 0.978, I at step 1 2.289 at the future UD slot).
- `P`: the immunisation's steps after the switch, default 175,000 (`configs/e5/e5.yaml`).

## Step 3 -- wave 1 (check, then launch)

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

The dry run prints the four jobs (`A_sh100_experimental`, `A_sh100_reference`, `A_sh050_experimental`,
`A_sh050_reference`), each `run_e5.py --config configs/e5/e5.yaml --seed-label A --training shuffle
--condition ... --fraction ... --device cuda --out-dir results`. Then, as E4: `kubectl get pods`, `cat $OUT/status.json`,
`tail -n 3 $OUT/*.log` (`E5 checkpoint <step>: 12 assignment(s) ...` at each checkpoint), `nvidia-smi`.

**The batch's status, with the STALE flag** (docs/e5/DECISIONS.md (26)), at every check:

```bash
python experiments/E5_belief_training/submit_kubeflow.py --status $OUT
```

One line per run: `running`, `done`, `failed`, `not started` or **`STALE`** (still running but its log has not changed
for more than an hour; the command then exits with code 2 and prints a `STALE ...` line). A STALE run: check the pod
(`kubectl get pods`), the GPU (`nvidia-smi` in the pod) and `/scratch`, and report.

**Check in the first hours of wave 1** (docs/e5/DECISIONS.md (19), (20)). Every E5 run computes its estimator switch on
pairs of images with the same background, with a separation deadline of 75,000 gradient steps (E4: 25,000). At each new
checkpoint until every run has switched (every hour or two), in the notebook:

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/switch_monitor.py early --runs results/E5_belief_training/*sh*__seed*     --e4-runs /scratch/helplessness-worldmodels-e4/results/E4_generalization_gradient/*__2zones-p1-b-d1__pair-*     --e4-decodability --device cpu --out /scratch/e5_wave1_early.csv
```

It prints, per run and per checkpoint up to its switch: the switch statistic (`js`), the agent state's decodability
(`decodability`; for E4's runs recomputed from their weights) and the reconstruction error on the agent's patch
(`recon_patch`), the shuffle runs against E4's runs of `$CONF`; then the alerts. **If a line `ALERT ...` appears** (a
shuffle run not switched at 50% of the deadline, i.e. 37,500 gradient steps, with a decodability that has not risen over
its last 3 checkpoints): report it to the user at once, with the table. It is not a stop: the runs continue. A run
stopped by `SeparationNotReachedError` (no switch within 75,000 gradient steps) is a result: report it, do not relaunch.

At the end of every wave, the one-line-per-run table of the switch (gradient and environment steps, environment steps
after it), for the final report:

```bash
python experiments/E5_belief_training/switch_monitor.py final --runs results/E5_belief_training/grid* --out /scratch/e5_switch_final.csv
```

## Step 3b -- 2026-10-22: after the failed first wave 1, relaunch it (docs/e5/DECISIONS.md (29))

The first wave 1 (supervisor `logs/kubeflow/e5-train-e5_wave1-20261021T140249396664Z`) failed: its four runs stopped at
the separation deadline (`SeparationNotReachedError`). In this order:

**1. Copy its light files** (the switch statistic at every evaluation is in each run's `checkpoints.jsonl`; no weight):

```bash
cd /scratch/helplessness-worldmodels-e5
SUP=logs/kubeflow/e5-train-e5_wave1-20261021T140249396664Z
mkdir -p exports
tar czf exports/e5_wave1_failed_light.tar.gz $SUP $SUP.submission.json     results/E5_belief_training/grid12x12__2zones-p1-b-d1-sh*/{config.yaml,meta.yaml,checkpoints.jsonl,console.log,assignments.npz}
tar tzf exports/e5_wave1_failed_light.tar.gz | wc -l      # 4 runs x 5 files + the supervisor's files
ls -l exports/e5_wave1_failed_light.tar.gz
```

Download `exports/e5_wave1_failed_light.tar.gz` from the notebook's file browser and send it.

**2. Move the failed runs out of the way** (after the copy):

```bash
mkdir -p results/E5_belief_training_aborted/wave1_20261021
mv results/E5_belief_training/grid12x12__2zones-p1-b-d1-sh* results/E5_belief_training_aborted/wave1_20261021/
ls results/E5_belief_training/                            # no grid12x12__2zones-p1-b-d1-sh* left
```

**3. Update the clone** (once the user has pushed `e5`), **dry run, launch**:

```bash
git -C /scratch/helplessness-worldmodels-e5 pull --ff-only
git -C /scratch/helplessness-worldmodels-e5 log --oneline -1
git -C /scratch/helplessness-worldmodels-e5 status --short       # prints nothing
grep -n "late_switch" configs/e5/e5.yaml                          # threshold_bits: 1.5, after_steps: 6100
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave1 --repo-dir /scratch/helplessness-worldmodels-e5
```

The dry run lists `A_sh100_experimental`, `A_sh100_reference`, `B_sh100_experimental`, `B_sh100_reference`. Then the
checks of step 3 (`switch_monitor.py early`); a switch by the late rule shows `[rule late_threshold]` in the log's
`>>> SWITCH` line.

**4. The latent diagnostic of the failed runs** (CPU, in the notebook, while wave 1 trains; about 10 minutes per run):

```bash
cd /scratch/helplessness-worldmodels-e5
mkdir -p exports/e5_ceiling
for R in results/E5_belief_training_aborted/wave1_20261021/grid*; do
  python experiments/E5_belief_training/latent_ceiling.py $R --batches 100 --out exports/e5_ceiling/$(basename $R).json; done
tar czf exports/e5_ceiling.tar.gz exports/e5_ceiling
```

Download `exports/e5_ceiling.tar.gz` and send it.

## Step 4 -- wave 2 (after wave 1 has finished and is exported: one compute pod at a time)

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/export_batch.py create --batch e5_wave1 --configuration p1-b-d1     # wave 1 first
git pull --ff-only && git status --short                                                                  # prints nothing
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --repo-dir /scratch/helplessness-worldmodels-e5
```

Four runs: seed B's baseline, experimental and reference, and shuffle f = 1 seed C (202), experimental and reference.
The dry run lists `B_base_experimental`, `B_base_reference` (with `--save-resumable-at 100000`), `C_sh100_experimental`,
`C_sh100_reference`. The reference of seed B keeps its full state at 100,000 steps (`checkpoints/resumable_env100000.pt`,
about 40 MB), from which E8's immunisation will branch. The checks of step 3 apply to the shuffle runs.

## Step 5 -- wave 3: TO BE CONFIRMED by the user after wave 1's results (docs/e5/DECISIONS.md (31))

Only once the user has confirmed it, after wave 2 has finished and is exported:

```bash
python experiments/E5_belief_training/export_batch.py create --batch e5_wave2 --configuration p1-b-d1
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --repo-dir /scratch/helplessness-worldmodels-e5
```

Four runs: seed C's baseline and shuffle f = 1 seed D (302), experimental and reference. With waves 1-3: the baseline on
seeds A (E4's pair), B, C and shuffle f = 1 on A, B, C, D, paired by seed.

## Step 6 -- runs interrupted by the infrastructure: resume them

Every run of a wave that is not done, in one pod (as the training; a done run is never relaunched). The dry run first
checks each resume point on the CPU (latest.json, resume.pt complete and loaded entirely, the same step) and prints the
step each run resumes from; on any problem it submits nothing and says which run and why.

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --phase resume \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --phase resume \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

One run only: add `--resume results/E5_belief_training/<the run's directory name>`.

A run stopped by a training guard (`SeparationNotReachedError`, `CollapseAfterSwitchError`) is a result: report it.

## Step 7 -- export, probes, export again

```bash
python experiments/E5_belief_training/export_batch.py create --batch <wave> --configuration p1-b-d1
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --phase probes --checkpoint <k> \
    [--e4-results /scratch/helplessness-worldmodels-e4/results] --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch <wave> --phase probes --checkpoint <k> \
    [--e4-results /scratch/helplessness-worldmodels-e4/results] --repo-dir /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/export_batch.py create --batch <wave> --configuration p1-b-d1
```

`k` is in environment steps for baseline and shuffle, in steps AFTER THE SWITCH for the immunisation (whose weights are
at `switch + k`; its reference is the baseline reference of its seed at `k`). The probes use 150 walks
(`e5.probes.n_walks`) and write, beside E4's files, the validity of every probe (`valid`, `validity_summary.json`) and the
reconstruction diagnostic (`reconstruction.csv`; docs/e5/DECISIONS.md (22)).

Seed A's baseline (E4's own pair of p1-b-d1) is re-probed once with the same protocol:

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_baseline_A --phase probes --checkpoint 300000 \
    --e4-results /scratch/helplessness-worldmodels-e4/results --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_baseline_A --phase probes --checkpoint 300000 \
    --e4-results /scratch/helplessness-worldmodels-e4/results --repo-dir /scratch/helplessness-worldmodels-e5
```

Every finished run is probed this way, wave by wave (`--batch e5_wave1 | e5_wave2 | e5_wave3`, the probe pairs of
`batches.PROBE_PAIRS`; only the baseline A needs `--e4-results`), at the same checkpoint `k` for every condition, chosen
by the convergence criterion. Locally: `export_batch.py verify <extracted dir>`, then the probes of every condition and
seed aggregated (2 to 4 seeds per condition; invalid probes left out):

```bash
python experiments/E5_belief_training/analyse_probes.py --root <extracted results root> --checkpoint <k> --out <dir>
```

