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
| `e5_extend_a` / `e5_extend_b` | continuations of the shuffle runs (seed A's pair; then B's reference, E, F) from 290,000, stopped at their convergence checkpoint K_a / K_b (ceiling 650,000) | a and b: about 20-35 h each |
| `e5_probes_a` / `e5_probes_b` | no training: baselines A, B at 300,000 and seed A's f = 1 pair at K_a; then E and F at K_b | about 1 h each |

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

## Step 6b -- continue the shuffle runs before the probes, in two batches (docs/e5/DECISIONS.md (33))

At 300,000 steps the shuffle runs are still learning (mass 0.74-0.93, b(UD) -1.6 to -1.9; the baselines converged by
100,000). They continue from their last resume point (290,000), in new run directories (the originals untouched), up
to a ceiling of 650,000 steps, in TWO batches, one pod of 3 CPUs at a time, each stopped by the user at its own
convergence checkpoint:

| Batch | Runs | Its K | Then |
|---|---|---|---|
| `e5_extend_a` | f = 1 seed A, experimental and reference (2 runs) | K_a: its pair meets E4's criterion | `e5_probes_a` |
| `e5_extend_b` | f = 1 seed B's reference, E (402) and F (502) experimental (3 runs) | K_b: both pairs (E and F against B's reference) meet it | `e5_probes_b` |

(`e5_extend`, the five runs in one pod, stays defined but is NOT USED.) The order: `e5_extend_a`, its export, the probes
`e5_probes_a` (step 7), then `e5_extend_b`, its export, `e5_probes_b`.

**`e5_extend_a`'s first launch failed (26 October, docs/e5/DECISIONS.md (34)).** Both runs stopped at their first step
on `ObservationTableMismatch`: the resume states of waves 1-2 predate the saved row offset of the episode in progress;
fixed (the offset is taken from the restored episode source). Before relaunching, move the two failed continuations
(kept, not deleted; only directories with status `failed` and a `branched_from` are moved):

```bash
cd /scratch/helplessness-worldmodels-e5
mkdir -p results/E5_belief_training_aborted/extend_a_20261026
for R in results/E5_belief_training/grid12x12__2zones-p1-b-d1-sh100__pair-*__worldmodel-agent-e5__seed02__20261026T*; do
  grep -q "^status: failed" $R/meta.yaml && grep -q "^branched_from:" $R/meta.yaml && mv $R results/E5_belief_training_aborted/extend_a_20261026/
done
ls results/E5_belief_training_aborted/extend_a_20261026/         # the 2 failed continuations (experimental, control)
ls -d results/E5_belief_training/*-sh100__*seed02__*             # the 2 originals, 20261022T..., nothing of 20261026
```

Then pull, and check which runs `--extend auto` takes (after the pull; the originals of 22 October):

```bash
git pull --ff-only && git log --oneline -1 && git status --short --untracked-files=no                     # prints nothing
python -c "import sys; sys.path.insert(0, 'experiments/E5_belief_training'); from pathlib import Path; import run_e5; [print(c, run_e5.original_run(Path('results'), 'p1-b-d1', 'A', 'shuffle', c, 1.0).name) for c in ('experimental', 'reference')]"
```

Then launch as below. The log's first lines now show the commit (`code: git commit <hash>`, read from `.git`: the
container has no git) and `branched from ...20261022T...resume.pt at environment step 290000`; the first `env step
290...` monitor lines follow (no `ObservationTableMismatch`). `check_wave.py` reads a continuation's switch from its
original until its first checkpoint (`switch ... from the original run`).

**Launch a batch** (`B` = `e5_extend_a`, later `e5_extend_b`; after the user has pushed `e5`):

```bash
cd /scratch/helplessness-worldmodels-e5
git pull --ff-only && git status --short --untracked-files=no                                             # prints nothing
python experiments/E5_belief_training/submit_kubeflow.py --batch $B --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch $B --repo-dir /scratch/helplessness-worldmodels-e5
```

The dry run shows `"cpus": 3` and the batch's jobs, each `run_e5.py --extend auto --to 650000 ...` (the newest finished
run of that seed and condition, from its `checkpoints/resume.pt`). Checkpoints every 25,000 steps past 300,000 (every
assignment measured every 50,000).

**Watch the convergence** (any time, readable while the runs train):

```bash
python experiments/E5_belief_training/check_wave.py --batch $B --out exports/${B}_check.json
```

Its last line, `ALL <n> PAIRS MEET E4's CRITERION AT: [...]` (n = 1 for `e5_extend_a`: seed A's pair; n = 2 for
`e5_extend_b`: E and F against B's reference, which is in the batch). As soon as that list is not empty, its first value
is the batch's K (`K_a`, then `K_b`).

**Stop the pod** (only then):

```bash
SUP=$(ls -d logs/kubeflow/e5-train-${B}-* | grep -v submission | tail -1)
JOB=$(python -c "import json; print(json.load(open('$SUP.submission.json'))['job_id'])")
echo $JOB
kubectl delete trainjob $JOB
kubectl get pods                                                   # its pod Terminating, then gone
python experiments/E5_belief_training/check_wave.py --batch $B --out exports/${B}_check.json   # K still listed
ls results/E5_belief_training/*/checkpoints/agent_env$(printf %06d $K).pt                       # the batch's runs' weights at K
```

A stop leaves usable checkpoints: every `agent_env<N>.pt` is written atomically (whole or absent; at most an orphan
`.tmp`), and a checkpoint appears in the records only after its weights are saved. The stopped runs keep status
`running` (not failed: the check and the probe plan keep them); a truncated last record line is ignored. They have no
`replay_final.npz` (not needed by the probes).

**Export** (with the weights at K):

```bash
python experiments/E5_belief_training/export_batch.py create --batch $B --configuration p1-b-d1 --weights-checkpoints $K
```

If 650,000 is reached without K: report the check; a further continuation (a larger `EXTEND_TO`) starts from each
continuation's own resume point.

### 6b bis -- the canonical masses of the continuations (docs/e5/DECISIONS.md (35))

Until the fix of (35), a continuation (and any resumed shuffle run) measured its canonical assignment against the
dynamics of the assignment in progress at its restore: the experimental runs' quadrant mass stuck near 0.2 (A, E, F),
the references right. Training and weights are right; only those records are wrong. In order, from the notebook:

**1. Stop `e5_extend_b`** (its records since 300,000 are wrong for E and F; it resumes later if needed):

```bash
cd /scratch/helplessness-worldmodels-e5
SUP=$(ls -d logs/kubeflow/e5-train-e5_extend_b-* | grep -v submission | tail -1)
JOB=$(python -c "import json; print(json.load(open('$SUP.submission.json'))['job_id'])")
echo $JOB && kubectl delete trainjob $JOB
kubectl get pods                                                   # its pod Terminating, then gone
```

**2. Update the clone and check that no wave 1-2 run was ever resumed** (only the continuations are affected):

```bash
git pull --ff-only && git log --oneline -1 && git status --short --untracked-files=no                     # prints nothing
grep -l "^resumed_at" results/E5_belief_training/*/meta.yaml results/E5_belief_training_aborted/*/*/meta.yaml   # prints nothing
```

A run listed there is a shuffle run resumed before the fix: report it (its checkpoints after `resumed_at` would need the
recomputation too).

**3. One recompute pod** (GPU, the device the runs measured on): A's, E's and F's continuations (every checkpoint) and the
validation on an untouched checkpoint (E's original at 300,000):

```bash
R=results/E5_belief_training
A_EXT=$(ls -d $R/*-sh100__pair-experimental__*__seed02__* | tail -1)
E_EXT=$(ls -d $R/*-sh100__pair-experimental__*__seed402__* | tail -1)
F_EXT=$(ls -d $R/*-sh100__pair-experimental__*__seed502__* | tail -1)
E_ORIG=$(ls -d $R/*-sh100__pair-experimental__*__seed402__20261024T110418Z)
echo $A_EXT $E_EXT $F_EXT $E_ORIG                                  # three continuations (2026102[6-8]T...), E's original
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_extend_b --phase recompute --repo-dir /scratch/helplessness-worldmodels-e5     --recompute-runs $A_EXT $E_EXT $F_EXT $E_ORIG@300000 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_extend_b --phase recompute --repo-dir /scratch/helplessness-worldmodels-e5     --recompute-runs $A_EXT $E_EXT $F_EXT $E_ORIG@300000
```

Each job's log has one line per checkpoint: `REPRODUCED` (the fields independent of the world as recorded) and the
canonical mass by quadrant, recorded -> recomputed. Once the pod is done:

```bash
python -c "import json, glob; [print(p.split('/')[-2][-45:], x['env_step'], x['reproduced'], '%.1e' % x['largest_difference_unaffected'], '%.1e' % x['largest_difference_canonical_mass'], [round(x['e4']['real_context']['mass_true_by_region'][q], 3) for q in ('nw', 'ne', 'se', 'sw')]) for p in sorted(glob.glob('results/E5_belief_training/*/checkpoints_recomputed.jsonl')) for x in map(json.loads, open(p))]"
```

Expected: every line `True`; E's original at 300,000 with a canonical-mass difference of 0 (the validation: an untouched
record recomputed as recorded); the continuations' masses back near their originals' (0.69-0.77 at 300,000). Report
these lines.

**4. Read the criterion** (check_wave uses the recomputed records, "canonical record recomputed at N checkpoint(s)"):

```bash
python experiments/E5_belief_training/check_wave.py --batch e5_extend_a --out exports/e5_extend_a_check.json
python experiments/E5_belief_training/check_wave.py --batch e5_extend_b --out exports/e5_extend_b_check.json
```

A batch whose `ALL <n> PAIRS MEET E4's CRITERION AT:` lists a checkpoint has its K: no resume, its export and probes
(above, step 7). Otherwise:

**5. Resume** that batch from its runs' last resume point (the fixed code: its new checkpoints are right; one pod at a
time -- `e5_extend_b` first if both need it, then `e5_extend_a`; the resume check on the CPU runs first):

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_extend_b --phase resume --configuration p1-b-d1 --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_extend_b --phase resume --configuration p1-b-d1 --repo-dir /scratch/helplessness-worldmodels-e5
# later, the same with --batch e5_extend_a (A's stopped continuation, NOT --extend auto: that takes the 22 October original)
```

The dry run lists `resume_<job>` for the batch's continuations (status `running`), each `run_e5.py --resume <its directory>`.
Their logs: `RESUMING`, `resumed at environment step <its last resume point>`. Watch and stop as above (check_wave).

## Step 7 -- the probes, in two sets (docs/e5/DECISIONS.md (33))

Each shuffle pair is probed at ITS first checkpoint meeting E4's criterion (K_a, K_b), the baselines at 300,000
(converged): comparable by their level of learning, not by their number of steps.

**First set, `e5_probes_a`** (after `e5_extend_a` is stopped and exported; before `e5_extend_b`): baselines A (E4's own
pair of p1-b-d1, under `--e4-results`) and B at 300,000, seed A's f = 1 pair at K_a:

```bash
cd /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_a --phase probes --checkpoint $K_a     --e4-results /scratch/helplessness-worldmodels-e4/results --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_a --phase probes --checkpoint $K_a     --e4-results /scratch/helplessness-worldmodels-e4/results --repo-dir /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/export_batch.py create --batch e5_probes_a --configuration p1-b-d1
```

Three jobs (`probes_A_base_k300000`, `probes_B_base_k300000`, `probes_A_sh100_k<K_a>`), two at a time on the GPU.

**Second set, `e5_probes_b`** (after `e5_extend_b` is stopped and exported): E and F against B's reference at K_b:

```bash
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_b --phase probes --checkpoint $K_b     --repo-dir /scratch/helplessness-worldmodels-e5 --dry-run
python experiments/E5_belief_training/submit_kubeflow.py --batch e5_probes_b --phase probes --checkpoint $K_b     --repo-dir /scratch/helplessness-worldmodels-e5
python experiments/E5_belief_training/export_batch.py create --batch e5_probes_b --configuration p1-b-d1
```

At submission (dry run included), a pair is LEFT OUT, with its reason printed, when one of its runs failed, never
switched, locked in, or lacks its weights at its checkpoint. 150 walks, the full grid, the validity of every probe and
the reconstruction diagnostic (docs/e5/DECISIONS.md (22)). Locally, `export_batch.py verify`, then the two sets merged,
each pair at its own checkpoint (a pair measured at several checkpoints needs `--pair-k <condition>:<seed>=<k>`):

```bash
python experiments/E5_belief_training/analyse_probes.py --root <extracted results root(s) of both sets> --out <dir>
```

(`e5_probes_all`, one set at one common checkpoint, stays defined but is NOT USED.)

**Keep every checkpoint until then.** The probes need every run's `checkpoints/agent_env<K>.pt` on the volume, E4's runs of
p1-b-d1 included (`/scratch/helplessness-worldmodels-e4/results/E4_generalization_gradient/`). Nothing in the code deletes
or overwrites them: each is written once under its own step's name (an atomic write of that file only); `resume.pt` and
`latest.json` are the only files rewritten; a resumed run re-trains from its last resume point and rewrites the
checkpoints after it with the same training. Never delete or move a run directory of E4 or E5 (the failed first wave 1
was moved to `results/E5_belief_training_aborted/`, not deleted), never run `git clean` in the checkouts (`results/` is
untracked), and check the volume's space before each wave (`df -h /scratch`). Whether UCL ARC purges old files on this
volume is not something the repository can tell: ask ARC if unsure.
