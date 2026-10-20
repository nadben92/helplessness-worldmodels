# E4 -- wave 1 (`e4_wave1`) on the GPU cluster: training, export, local pilot analysis, probes

Wave 1 is the pilot and the campaign's first wave: configurations `p1-a-d1` and `p1-b-d1` (pair 1, both
orientations, diagonal 1), each in its experimental (exposed) and its reference environment -- four runs of 300,000
environment steps in parallel on one GPU. Wave 2 (pair 2) is launched later, once the budget B is chosen.

The sequence: (1) train on the cluster; (2) export one archive and download it; (3) analyse locally (choice of B,
checks); (4) launch the probes at B on the cluster, export again, analyse locally.

## Rules

1. Run the commands exactly as written. If anything happens that this file does not foresee (an error, a command that
   fails, a pod that stays `Pending` more than 15 minutes), stop and report the complete output.
2. Never change a seed, a config, the batch or an option. A run that fails is a result: resume it only with the
   command of step 3, never relaunch it as a new run.
3. Nothing is committed on the cluster. Results stay on the volume and come back as one archive (step 4).

## Step 0 -- on your own machine: push the branch `e4`

```bash
git push -u origin e4
```

## Step 1 -- in a terminal of the Kubeflow notebook: a clean clone of `e4`

```bash
cd /scratch
git clone --branch e4 --single-branch https://github.com/nadben92/helplessness-worldmodels.git helplessness-worldmodels-e4
cd /scratch/helplessness-worldmodels-e4
git log --oneline -1
git status --short
```

`git log` shows the newest commit of `e4`; `git status --short` prints nothing. If the directory already exists, do
not reuse it: stop.

## Step 2 -- check, then launch the training (about 40-45 hours)

```bash
cd /scratch/helplessness-worldmodels-e4
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 \
    --repo-dir /scratch/helplessness-worldmodels-e4 --dry-run
```

It prints, without submitting anything: `"push": false`, the four jobs `p1-a-d1_experimental`,
`p1-b-d1_experimental`, `p1-a-d1_reference`, `p1-b-d1_reference` (each `run_e4.py --config configs/e4/e4.yaml ...
--device cuda`, no `--max-env-steps`, no `--push-...`) and a runtime patch with the `scratch` volume only. Then:

```bash
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 \
    --repo-dir /scratch/helplessness-worldmodels-e4
OUT=<the "output" path printed above, ending in logs/kubeflow/e4-train-e4_wave1-<date>>
```

Check that the four runs are running in parallel (within about 15 minutes):

```bash
kubectl get pods
cat "$OUT/supervisor.json"      # "max_parallel": 4 and the four job names
cat "$OUT/status.json"          # the four jobs with a "pid", a "started_at" and "exit_code": null
tail -n 3 "$OUT"/*.log          # recent "env step ..." lines; "E4 checkpoint <step>: ..." at each checkpoint
nvidia-smi
```

The batch has finished when every job of `status.json` has a numeric `"exit_code"` (0 = done).

## Step 3 -- a run interrupted by the infrastructure: resume it

```bash
RUN=results/E4_generalization_gradient/<the run's directory name>
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase resume --resume "$RUN" \
    --repo-dir /scratch/helplessness-worldmodels-e4
```

Several runs of a batch interrupted at once (one pod, as the training): leave out `--resume`. Every run of the batch
whose status is not `done` resumes from its `checkpoints/resume.pt`, all in parallel in one pod; a `done` run is never
relaunched. The dry-run (and the submission) first checks each resume point (latest.json, resume.pt loaded entirely,
the same step) and prints, per run, the step it resumes from; on any problem it submits nothing.

```bash
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave2 --phase resume \
    --repo-dir /scratch/helplessness-worldmodels-e4 --dry-run
```

A run stopped by a training guard (`SeparationNotReachedError`, `CollapseAfterSwitchError` in its log) is a result:
do not resume it; report it.

## Step 4 -- export one archive (in the notebook), download it

```bash
cd /scratch/helplessness-worldmodels-e4
python experiments/E4_generalization_gradient/export_batch.py create --batch e4_wave1
```

It writes `exports/e4_wave1__<date>.tar.gz` (about 12-13 MB) and prints its content: for each run its light files
(config, metrics, checkpoint diagnostics, log, results, status) and the weights of its final checkpoint
(`agent_env300000.pt`), the supervisor's logs and status, every probe measurement already made, and `MANIFEST.json`
(size and SHA-256 of every file). To add other checkpoints' weights (about 11 MB per checkpoint for the four runs):
`--weights-checkpoints 175000 200000`. Download the archive from the notebook's file browser
(`exports/` in the checkout), then copy it to Google Drive.

## Step 5 -- on your machine: extract, check, analyse

```bash
mkdir -p ../e4_wave1 && tar -xzf e4_wave1__<date>.tar.gz -C ../e4_wave1
python experiments/E4_generalization_gradient/export_batch.py verify ../e4_wave1        # 0 problem(s), the four runs "done"
A=experiments/E4_generalization_gradient
python $A/pilot_analysis.py convergence --export ../e4_wave1 --out ../e4_wave1_analysis/convergence   # B: budget_choice.json
python $A/pilot_analysis.py protocol    --export ../e4_wave1 --out ../e4_wave1_analysis/protocol
python $A/pilot_analysis.py evolution   --export ../e4_wave1 --out ../e4_wave1_analysis/evolution
python $A/pilot_analysis.py forgetting  --export ../e4_wave1 --configuration p1-a-d1 --checkpoint 300000 \
    --out ../e4_wave1_analysis/forgetting                                              # about 2-3 min on CPU
```

The extreme probes on the final weights, locally (about 30 minutes per configuration on CPU), if wanted before the
cluster's probes: `python $A/measure_probes.py --configuration p1-a-d1 --checkpoint 300000 --images extreme --save-steps
--experimental-run ../e4_wave1/results/E4_generalization_gradient/<exp run> --reference-run <ref run> --out-dir ../e4_wave1_probes`.

Write B (`budget_choice.json` "B") into `batches.py` (`B_AFTER_PILOT`), commit and push it before wave 2.

## Step 6 -- the probes at B, on the cluster (launched by hand)

In the notebook, from the checkout (the weights of every checkpoint are on the volume):

```bash
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase probes --checkpoint <B> \
    --repo-dir /scratch/helplessness-worldmodels-e4 --dry-run
python experiments/E4_generalization_gradient/submit_kubeflow.py --batch e4_wave1 --phase probes --checkpoint <B> \
    --repo-dir /scratch/helplessness-worldmodels-e4
```

Two jobs (`probes_p1-a-d1_env<B>`, `probes_p1-b-d1_env<B>`), both on the GPU at once; one configuration only:
`--configurations p1-a-d1`. A quick timing first is possible with `--n-walks 30` (one tenth of the walks). Each job
writes `results/E4_generalization_gradient/probes__<configuration>__env<B>__<date>/` with its own `status.json`
("done", or "failed" with the error) and its log in the supervisor directory. Then export again (step 4, the probes are
included) and analyse locally:

```bash
python $A/pilot_analysis.py probes      --export ../e4_wave1b --out ../e4_wave1_analysis/probes
python $A/pilot_analysis.py sufficiency --export ../e4_wave1b --out ../e4_wave1_analysis/sufficiency   # n_walks for wave 2
```

## Optional -- automatic push of the results to a branch

Not needed. If wanted, every run can also push its light results to the orphan branch `results/e4_wave1` when it ends
(docs/RESULTS_CONVENTION.md): create a GitHub token restricted to this repository (Contents: read and write), store it as
a Kubernetes secret (`read -rs TOKEN && printf '%s' "$TOKEN" > /tmp/t && unset TOKEN && kubectl create secret generic
e4-results-push --from-file=RESULTS_PUSH_TOKEN=/tmp/t && rm -f /tmp/t`) and add `--push-secret e4-results-push` to the
commands of steps 2 and 3 (the checkout must then be an https clone).
