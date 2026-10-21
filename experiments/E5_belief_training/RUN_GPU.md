# E5 on the GPU cluster: training, export, probes

E5 trains 12 runs on ONE configuration of E4, **`p1-b-d1`** (docs/e5/DECISIONS.md (21); seeds A = 2, B = 102), in three
waves of four runs, each wave on one GPU (four runs in parallel), plus one probes-only batch (`e5_probes_baseline_A`). It is launched from **its own clone**, `/scratch/helplessness-worldmodels-e5` (branch
`e5`): the e4 clone (`/scratch/helplessness-worldmodels-e4`) keeps E4's runs and probes and is never touched by E5,
except to read E4's results (the reference of immunisation A, the switch step).

| Wave | Runs | Needs | Expected duration |
|---|---|---|---|
| `e5_wave1` | seed A: shuffle f = 1 (exp, ref), shuffle f = 0.5 (exp, ref) | the configuration | about 45-50 h |
| `e5_wave2` | seed B: baseline (exp, ref; the reference keeps resumable states), shuffle f = 1 (exp, ref) | the configuration (the switch step if known) | about 45-50 h |
| `e5_wave3` | seed B: shuffle f = 0.5 (exp, ref); immunisation A (recomputed) and B (branched) | the switch step; wave 2 done | about 45-50 h if S + 175,000 <= 300,000, else about (S + 175,000) / 300,000 x 40-45 h |
| `e5_probes_baseline_A` | no training: E4's own pair of p1-b-d1 re-probed with E5's probe script (150 walks) | `--e4-results` | about 15 min |

The durations scale E4's 40-45 h for 300,000 steps (four runs in parallel) by the shuffle runs' extra checkpoint
measures (every assignment at every checkpoint: IMPLEMENTATION_REPORT.md, "cost"); a wave lasts as long as its longest run.

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

## Step 4 -- wave 2

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 [--switch-step $S] \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 [--switch-step $S] \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

The baseline reference of seed B (`B_base_reference`, `--save-resumable-at candidates [S]`) keeps its full state
(`checkpoints/resumable_env<N>.pt`, about 25-40 MB each) at E4's grid points from 10,000 to 250,000 and at `S` when
given: immunisation B branches from the one at `S`. If `S` is not in that list and was not given here, immunisation B
cannot branch: launch wave 2 with `--switch-step $S`.

## Step 5 -- wave 3 (after wave 2 has finished)

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --switch-step $S [--post-steps $P] \
    --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave3 --switch-step $S [--post-steps $P] \
    --repo-dir /scratch/helplessness-worldmodels-e5
```

`A_imm` recomputes its phase 1 (E4 kept no full state of its reference); `B_imm` has `--branch-from auto` and starts
from wave 2's `resumable_env<S>.pt` (it fails at once if that file is missing: report it).

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
``` Wave 3 needs `--e4-results` (immunisation
A's reference is E4's reference run of `$CONF`). Locally: `export_batch.py verify <extracted dir>`, then the analyses
(`rsa_e5.py`, the probes).
