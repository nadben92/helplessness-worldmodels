#!/usr/bin/env bash
# The first GPU run's batch on one machine: the 4 runs of configs/e3/gpu_e30.yaml (seeds 0 and 5 x
# experimental / reference) and the bridge run of configs/e3/gpu_e30_bridge.yaml (seed 0,
# experimental, mini_long's network). Run from the repository root, after preflight.py has passed
# WITH THE SAME SUBSET AND THE SAME NUMBER OF SIMULTANEOUS PROCESSES:
#
#     bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh
#     RUNS="seed5_experimental bridge_seed0_experimental" MAX_PARALLEL=2 bash experiments/E3_worldmodel_4zones/launch_gpu_e30.sh
#
# BATCH: gpu_e30 (default; the first batch) or gpu_e30b (the second: win64_ / win16_ x seed0 / seed5 x
# experimental / reference, 8 runs). RUNS: a subset of the batch's runs, e.g. for gpu_e30 among
# seed0_experimental seed0_reference seed5_experimental seed5_reference bridge_seed0_experimental
# (default: all of them). MAX_PARALLEL: the most runs sharing the GPU at
# once (default: all of them); the others wait in a queue and start as runs finish. The
# preflight for that launch is
#     python experiments/E3_worldmodel_4zones/preflight.py --config configs/e3/gpu_e30.yaml \
#         --extra-config configs/e3/gpu_e30_bridge.yaml --parallel <MAX_PARALLEL> --device cuda
# (submit_kubeflow.py --phase preflight --runs ... --max-parallel ... builds it).
#
# The supervisor (submit_kubeflow.py --worker, the same one the Kubeflow job runs) writes one log
# per run and status.json (start/end times, exit codes) into logs/<BATCH>/train-<UTC>/; nohup keeps
# it alive if the terminal closes. The script prints the supervisor's PID and returns at once.
# An interrupted run is continued, in its own directory, with
#     python experiments/E3_worldmodel_4zones/run_mini.py --resume results/E3_worldmodel_4zones/<run directory>
# A run that stops on a training guard (no separation by the deadline, collapse) is a result:
# it is reported as a failure, never re-launched with another seed without it being recorded.
#
# Optional W&B tracking (wandb_logger.py): set WANDB_API_KEY before running this script (and
# WANDB_MODE=offline if the machine has no internet; then `wandb sync` the runs' wandb/
# directories afterwards). Both variables are passed to the runs. Without a key, W&B is
# off and the runs are unchanged. With W&B on, each run directory is also uploaded as a W&B
# artifact at the end (weights included, resume.pt excluded).
set -euo pipefail

PYTHON=${PYTHON:-python}
BATCH=${BATCH:-gpu_e30}
RUNS=${RUNS:-}
MAX_PARALLEL=${MAX_PARALLEL:-}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
  export WANDB_MODE=${WANDB_MODE:-online}
  echo "W&B tracking on (mode $WANDB_MODE)"
else
  echo "W&B tracking off (WANDB_API_KEY not set)"
fi

mkdir -p "logs/$BATCH"
out="logs/$BATCH/train-$(date -u +%Y%m%dT%H%M%SZ)"
args=(--worker --phase train --repo-dir . --output "$out" --batch "$BATCH")
# shellcheck disable=SC2206  # RUNS is a space-separated list of run names
[ -n "$RUNS" ] && args+=(--runs $RUNS)
[ -n "$MAX_PARALLEL" ] && args+=(--max-parallel "$MAX_PARALLEL")
nohup "$PYTHON" experiments/E3_worldmodel_4zones/submit_kubeflow.py "${args[@]}" > "$out.supervisor.log" 2>&1 &
echo "supervisor pid $! -> $out.supervisor.log; per-run logs and status.json in $out/"
echo "Results: results/E3_worldmodel_4zones/grid12x12__4zones__pair-*__worldmodel-${BATCH}*__seed0{0,5}__*/"
