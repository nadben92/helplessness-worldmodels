# E5 on the GPU cluster: training, export, probes

E5 trains on ONE configuration of E4, **`p1-b-d1`** (docs/e5/DECISIONS.md (21); seeds A = 2, B = 102), one wave per
GPU, plus one probes-only batch (`e5_probes_baseline_A`). It is launched from **its own clone**,
`/scratch/helplessness-worldmodels-e5` (branch `e5`): the e4 clone (`/scratch/helplessness-worldmodels-e4`) keeps E4's
runs and probes and is never touched by E5, except to read E4's results.

Since 2026-10-22 (docs/e5/DECISIONS.md (29)): the first wave 1 (f = 1 and f = 0.5, seed A) failed at the separation
deadline; since 2026-10-23 the campaign is two waves (docs/e5/DECISIONS.md (32)):

| Wave | Runs | Expected duration |
|---|---|---|
| `e5_wave1` | shuffle f = 1, seeds A and B (exp, ref): 4 runs (seed B's experimental run locked in) | about 45-50 h |
| `e5_wave2` | seed B's baseline (exp, ref; the reference keeps its full state at 100,000 steps, for E8) and two shuffle f = 1 experimental runs, seeds E (402) and F (502), replacing seed B's and measured against seed B's f = 1 reference: 4 runs | about 45-50 h |
| `e5_probes_all` | no training: every healthy pair of waves 1-2 and seed A's baseline (E4's own pair), at K = 300,000 | a few hours |

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
git -C /scratch/helplessness-worldmodels-e5 status --short --untracked-files=no   # prints nothing (results/ is untracked)
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

## Between two waves: export and learning check only (docs/e5/DECISIONS.md (31))

No probe between the waves. Once a wave has finished: its export (`export_batch.py create`), then ONE command that prints
the learning check, one line per run and per pair, from the light files:

```bash
python experiments/E5_belief_training/check_wave.py --batch <wave> --out exports/<wave>_check.json
```

Per run: status; the estimator switch (gradient and environment steps, rule); after it, the longest stretch of the
switch statistic below 1 bit (5 = the collapse guard), its lowest and last value; the smallest quadrant mass at the last
checkpoint (shuffle runs: also by zone type over the 12 assignments); the success rate. Per pair: b at step 1 in the
training world by slot (shuffle: also by zone type over the assignments) and the checkpoints where E4's convergence
criterion holds -- the candidates for the common probe checkpoint. `ATTENTION` lines go to the user before the next wave
is launched. Send `exports/<wave>_check.json` with the export.

A run flagged `LOCK-IN` (no switch, statistic below 0.01 bit from gradient step 6,100 on; docs/e5/DECISIONS.md (32))
fails at the deadline: leave it to its end, then, on the CPU (about 10 minutes), the latent diagnostic -- e.g. wave 1's
seed B experimental run:

```bash
R=$(ls -d results/E5_belief_training/grid12x12__2zones-p1-b-d1-sh100__pair-experimental__worldmodel-agent-e5__seed102__* | tail -1)
python experiments/E5_belief_training/latent_ceiling.py $R --batches 100 --out exports/$(basename $R)_ceiling.json
```

Its console log is `$R/console.log` (and the supervisor's copy `logs/kubeflow/<supervisor>/<job>.log`). Its `goal_loss` and
`critic_loss` print as nan before any switch: the agent's heads train only from the switch on (not a numerical
failure).

## Step 4 -- wave 2 (after wave 1 has finished, is exported and checked: one compute pod at a time)

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/export_batch.py create --batch e5_wave1 --configuration p1-b-d1     # wave 1 first
python experiments/E5_belief_training/check_wave.py --batch e5_wave1 --out exports/e5_wave1_check.json      # the learning check
git pull --ff-only && git status --short --untracked-files=no                                             # prints nothing
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_wave2 --repo-dir /scratch/helplessness-worldmodels-e5
```

Four runs (the dry run lists them): `B_base_experimental`, `B_base_reference` (with `--save-resumable-at 100000`: its full
state at 100,000 steps, `checkpoints/resumable_env100000.pt`, for E8's immunisation), `E_sh100_experimental` (seed 402)
and `F_sh100_experimental` (seed 502) -- two experimental runs replacing seed B's (locked in), both measured against seed
B's f = 1 reference of wave 1 (`batches.REFERENCE_SEED_OF`; docs/e5/DECISIONS.md (32)). The checks of step 3 apply to
the shuffle runs; `check_wave.py --batch e5_wave2` pairs E and F with seed B's reference of wave 1 by itself.

There is no wave 3 (docs/e5/DECISIONS.md (32)).

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

## Step 6b -- continue the shuffle runs before the probes (docs/e5/DECISIONS.md (33))

At 300,000 steps the shuffle runs are still learning (mass 0.74-0.93, b(UD) -1.6 to -1.9; the baselines converged by
100,000). The five healthy shuffle runs (A exp, A ref, B ref, E, F) continue from their last resume point (290,000), in
ONE pod of 3 CPUs, in new run directories (the originals untouched), up to a ceiling of 650,000 steps; the user stops the
pod as soon as every pair meets E4's criterion at one common checkpoint.

**Launch** (after the user has pushed `e5`):

```bash
cd /scratch/helplessness-worldmodels-e5
git pull --ff-only && git status --short --untracked-files=no                                             # prints nothing
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_extend --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_extend --repo-dir /scratch/helplessness-worldmodels-e5
```

The dry run shows `"cpus": 3` and five jobs `run_e5.py --extend auto --to 650000 ...` (the newest finished run of each
seed and condition, from its `checkpoints/resume.pt`). Checkpoints every 25,000 steps past 300,000 (every assignment
measured every 50,000).

**Watch the convergence** (any time, readable while the runs train; E and F are paired with B's reference by
themselves):

```bash
python experiments/E5_belief_training/check_wave.py --batch e5_extend --out exports/e5_extend_check.json
```

Its last lines: `PAIRS ... converged at [...]` for each pair and `ALL 3 PAIRS MEET E4's CRITERION AT: [...]`. As soon
as that list is not empty, its first value is **K**.

**Stop the pod** (only then):

```bash
SUP=$(ls -d logs/kubeflow/e5-train-e5_extend-* | grep -v submission | tail -1)
JOB=$(python -c "import json; print(json.load(open('$SUP.submission.json'))['job_id'])")
echo $JOB
kubectl delete trainjob $JOB
kubectl get pods                                                   # its pod Terminating, then gone
python experiments/E5_belief_training/check_wave.py --batch e5_extend --out exports/e5_extend_check.json   # K still listed
ls results/E5_belief_training/*/checkpoints/agent_env$(printf %06d $K).pt                                 # the five runs' weights at K
```

A stop leaves usable checkpoints: every `agent_env<N>.pt` is written atomically (whole or absent; at most an orphan
`.tmp`), and a checkpoint appears in the records only after its weights are saved. The stopped runs keep status
`running` (they are not failed: the check and the probe plan keep them); a truncated last record line is ignored. They
have no `replay_final.npz` (not needed by the probes).

**Export** (with the weights at K):

```bash
python experiments/E5_belief_training/export_batch.py create --batch e5_extend --configuration p1-b-d1 --weights-checkpoints $K
```

If 650,000 is reached without a common K: report the check; a further continuation (a larger `EXTEND_TO`) starts from
each continuation's own resume point.

## Step 7 -- the probes: ONE batch after wave 2, at K = 300,000 (docs/e5/DECISIONS.md (31), (32))

After wave 2 has finished, been exported and checked, every healthy pair of waves 1 and 2 is probed in one batch,
`e5_probes_all`: the baselines at 300,000 (`batches.PROBES_CHECKPOINT`, converged), the shuffle pairs at `--checkpoint`
(the continuations' common convergence checkpoint K of step 6b): seed A's baseline (E4's own pair of p1-b-d1, under
`--e4-results`), seed B's baseline, shuffle f = 1 seed A, and the replacements E and F against seed B's f = 1 reference.
At submission (dry run included), a pair is LEFT OUT, with its reason printed, when one of its runs is not `done`, never
switched, locked in (check_wave.py's criteria) or lacks its weights at K: seed B's own f = 1 pair (its experimental run
locked in) is left out this way, and a replacement that fails is too; if both replacements succeed, both are probed.

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_all --phase probes --checkpoint 300000     --e4-results /scratch/helplessness-worldmodels-e4/results --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_all --phase probes --checkpoint 300000     --e4-results /scratch/helplessness-worldmodels-e4/results --repo-dir /scratch/helplessness-worldmodels-e5
```

The dry run prints the `LEFT OUT <job>: <reason>` lines, then the jobs; two run at a time on the GPU; 150 walks, the full
grid, the validity of every probe and the reconstruction diagnostic (docs/e5/DECISIONS.md (22)). Then one export per wave
(the probes are included) and, locally, `export_batch.py verify` and the aggregation over the seeds of each condition (2
to 3: the replacements count as experimental runs of f = 1 against B's reference):

```bash
python experiments/E5_belief_training/analyse_probes.py --root <extracted results root(s)> --checkpoint 300000 --out <dir>
```

**Keep every checkpoint until then.** The probes need every run's `checkpoints/agent_env<K>.pt` on the volume, E4's runs of
p1-b-d1 included (`/scratch/helplessness-worldmodels-e4/results/E4_generalization_gradient/`). Nothing in the code deletes
or overwrites them: each is written once under its own step's name (an atomic write of that file only); `resume.pt` and
`latest.json` are the only files rewritten; a resumed run re-trains from its last resume point and rewrites the
checkpoints after it with the same training. Never delete or move a run directory of E4 or E5 (the failed first wave 1
was moved to `results/E5_belief_training_aborted/`, not deleted), never run `git clean` in the checkouts (`results/` is
untracked), and check the volume's space before each wave (`df -h /scratch`). Whether UCL ARC purges old files on this
volume is not something the repository can tell: ask ARC if unsure.
