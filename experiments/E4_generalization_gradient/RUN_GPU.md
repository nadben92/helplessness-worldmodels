# E4 -- launching wave 1 (`e4_wave1`) on the GPU cluster

This file covers **one batch only: `e4_wave1`**, the pilot and the first wave of E4's campaign: configurations
`p1-a-d1` and `p1-b-d1` (pair 1, both orientations, diagonal 1), each in its experimental (exposed) and its reference
environment -- four runs of 300,000 environment steps, in parallel on one GPU. Waves 2-4 are launched later, once the
budget B is chosen from this one (`pilot_analysis.py convergence`).

Expected duration: about **40-45 hours** for the four runs together (E3's campaign ran the same training procedure,
four runs on one GPU with 3 CPUs, in 39.1-39.7 hours each, of which about 2.9 hours were E3 measures E4 does not take;
E4 adds its own checkpoint diagnostics at 21 checkpoints and the final protocol control, estimated at 3-6 hours).

## Rules

1. Run the commands exactly as written. If anything happens that this file does not foresee (an error, a command that
   fails, a pod that stays `Pending` more than 15 minutes), stop and report the complete output.
2. Never change a seed, a config, the batch or an option. A run that fails is a result: relaunch it only with the
   resume command of step 6, never as a new run.
3. Nothing is committed by hand on the cluster: each run pushes its own light results to the branch `results/e4_wave1`
   when it ends (success or failure), and the batch's status follows at the end. Weights, buffers and arrays stay on
   the volume, under the checkout's `results/E4_generalization_gradient/`.

## Step 0 -- on your own machine: push the branch `e4`

```bash
git push -u origin e4
```

The cluster clones this branch; nothing else is needed from your machine.

## Step 1 -- in a terminal of the Kubeflow notebook: a clean clone of `e4`

The checkout must be an **https** clone of the branch `e4` (the results are pushed with an https token):

```bash
cd /scratch
git clone --branch e4 --single-branch https://github.com/nadben92/helplessness-worldmodels.git helplessness-worldmodels-e4
cd /scratch/helplessness-worldmodels-e4
git log --oneline -1
git status --short
```

`git log` must show the newest commit of `e4` you pushed at step 0; `git status --short` must print nothing. If the
directory already exists from an earlier attempt, do not reuse it: stop (rule 1).

## Step 2 -- the push token, as a Kubernetes secret (once)

Create on GitHub a fine-grained personal access token restricted to the repository `nadben92/helplessness-worldmodels`
with the permission **Contents: read and write** (nothing else). Then, in the notebook terminal (the token is read
without being echoed and never written into the repository nor into the shell history):

```bash
read -rs TOKEN && printf '%s' "$TOKEN" > /tmp/e4_push_token && unset TOKEN
kubectl create secret generic e4-results-push --from-file=RESULTS_PUSH_TOKEN=/tmp/e4_push_token
rm -f /tmp/e4_push_token
kubectl get secret e4-results-push
```

The last command must list the secret. If `kubectl create secret` is refused (permissions), stop (rule 1).

## Step 3 -- check the commands, then launch

```bash
cd /scratch/helplessness-worldmodels-e4
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 \
    --repo-dir /scratch/helplessness-worldmodels-e4 --push-secret e4-results-push --dry-run
```

This prints, without submitting anything, the four jobs -- `p1-a-d1_experimental`, `p1-b-d1_experimental`,
`p1-a-d1_reference`, `p1-b-d1_reference`, each `run_e4.py --config configs/e4/e4.yaml ... --device cuda --push-batch
e4_wave1` with no `--max-env-steps` -- and the runtime patch mounting `e4-results-push` at `/etc/e4-results-push`.
If it shows anything else, stop. Then launch:

```bash
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 \
    --repo-dir /scratch/helplessness-worldmodels-e4 --push-secret e4-results-push
OUT=<the "output" path printed above, ending in logs/kubeflow/e4-train-<date>>
```

The submission refuses a checkout on another branch, with modified files, or with a non-https remote.

## Step 4 -- check that the four runs are running in parallel

```bash
kubectl get pods
cat "$OUT/supervisor.json"
cat "$OUT/status.json"
```

Within about 15 minutes the pod is `Running`, `supervisor.json` shows `"max_parallel": 4` and the four job names,
and `status.json` shows the four jobs each with a `"pid"` and a `"started_at"` and `"exit_code": null`. Each run
writes its own log:

```bash
tail -n 3 "$OUT"/p1-a-d1_experimental.log "$OUT"/p1-b-d1_experimental.log "$OUT"/p1-a-d1_reference.log "$OUT"/p1-b-d1_reference.log
nvidia-smi
```

Every log ends with recent `env step ...` lines (one every 1,000 environment steps, about every 8 minutes), and
`E4 checkpoint <step>: ...` lines at each checkpoint (1,000, 1,585, 2,512, ... 300,000). The batch has finished when
every job of `status.json` has a numeric `"exit_code"`.

## Step 5 -- follow the results branch (from your machine)

Each run pushes when it ends; the branch exists from the first run that ends:

```bash
git fetch origin results/e4_wave1
git worktree add ../results-e4_wave1 results/e4_wave1
ls ../results-e4_wave1/results/E4_generalization_gradient/e4_wave1
```

There is one directory per run (`config.yaml`, `meta.yaml`, `checkpoints.jsonl`, `metrics.csv`, `results.json`,
`console.log.gz`, `run_status.json`, `heavy_files.json`), then the batch's `status.json` and `handoff.json`. Later
updates: `git -C ../results-e4_wave1 pull`. Never merge this branch into `e4` or `main`.

If a run's results did not arrive (its log ends with `RESULTS PUSH FAILED`), its directory is still complete on the
volume (`/scratch/helplessness-worldmodels-e4/results/E4_generalization_gradient/<run name>/`); report it.

## Step 6 -- a failed run: resume it, never relaunch it

`status.json` gives the failed job's exit code; its log and its `run_status.json` give the error. A run interrupted by
the pod (an infrastructure failure, not a training guard) continues from its last resume point:

```bash
RUN=results/E4_generalization_gradient/<the failed run's directory name>
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase resume --resume "$RUN" \
    --repo-dir /scratch/helplessness-worldmodels-e4 --push-secret e4-results-push
```

A run stopped by a training guard (`SeparationNotReachedError`, `CollapseAfterSwitchError` in its log) is a result:
do not resume it; report it.

## After the batch

The pilot analyses run on your machine (`experiments/E4_generalization_gradient/README.md`, "pilot analyses"): the
convergence criterion and the choice of B need only the results branch; the probes at B and the forgetting check need
the weights of the checkpoint B, copied from the volume (their SHA-256 are in `handoff.json`).
