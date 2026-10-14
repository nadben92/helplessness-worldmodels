"""The results handoff of a GPU batch (docs/e3/DECISIONS.md, 2026-09-30 (3);
RUN_GPU.md step 5), so the person or assistant running the batch has nothing
to write by hand: from the supervisor's output directory (status.json and the
commands of every run) and the run directories under the results root, it
writes into --out

    handoff.md                      status of every run (exit code, meta.yaml status, gradient steps
                                    done, switch step, error), the preflight's decision, the archives;
    <run directory name>.tgz        one light archive per run: everything except checkpoints/,
                                    failure/*.pt, any .pt file, wandb/, measures/ and replay_final.npz;
    <run directory name>.measures.tgz   the run's checkpoint measures (measures/env<NNNNNN>/: results.json,
                                    arrays.npz, artifacts), when it has some (the campaign, DECISIONS 2026-10-07 (29));
    <run directory name>.weights.tgz    with --with-weights: the weights at every checkpoint and the final ones
                                    (checkpoints/agent_env<NNNNNN>.pt, agent_final.pt) and the final replay
                                    (replay_final.npz) -- never resume.pt (optimisers, environment, streams);
    logs.tgz                        the supervisor's logs and status.json, and the preflight's
                                    directories without their .pt files.

Every archive must stay below MAX_ARCHIVE_MB (GitHub refuses files over 100 MB): the script stops
otherwise. It never writes outside --out, and refuses an --out that already holds files.

Usage, from the repo root:
    python experiments/E3_worldmodel_4zones/make_handoff.py --train-dir logs/kubeflow/train-<UTC> \\
        [--preflight-dir logs/kubeflow/preflight-<UTC> ...] --out handoff/gpu_e30b
    python experiments/E3_worldmodel_4zones/make_handoff.py --measure-dir logs/kubeflow/measure-<UTC> \\
        [--preflight-dir logs/kubeflow/preflight-<UTC> ...] --out handoff/gpu_e30b_measures

A measurement batch (submit_kubeflow.MEASURE_BATCHES, docs/e3/DECISIONS.md 2026-10-01 (10)) is
handed off with --measure-dir instead of --train-dir: one light archive per run,
<run directory name>.recomputed.tgz, of the recomputation the job wrote under <measure dir>/runs/
(results.json, arrays.npz, SUMMARY.md, comparison_with_run.json, console.log, artifacts); the
original run directories are not read.
"""

from __future__ import annotations

import sys
from pathlib import Path

from helplessness.cluster import handoff as _handoff
from helplessness.cluster.handoff import *  # noqa: F401,F403 -- the functions moved to the library on 2026-10-14, re-exported here
from helplessness.cluster.handoff import (  # noqa: F401 -- private helpers, re-exported for the existing tests
    _excluded,
    _excluded_from_logs,
    _excluded_run,
    _weights_only,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "E3_worldmodel_4zones"


def main(argv: list[str] | None = None) -> int:
    """E3's handoff: helplessness.cluster.handoff.main with E3's results root, experiment id and documentation."""
    return _handoff.main(argv, default_results_root=RESULTS_ROOT, experiment_id="E3_worldmodel_4zones", description=__doc__)


if __name__ == "__main__":
    sys.exit(main())
