"""E6 and E7's batch table for the cluster (docs/E6_phenotypes_latent.md, docs/E7_latent_manipulation.md): standard library
only (the Kubeflow notebook runs it without the package's dependencies). A batch = an agents file; a phase = one
script; a job = one process measuring a group of agents (--agents-per-job, default 1), run in parallel on one GPU.

  phase          script                                                   needs
  e6             experiments/E6_phenotypes_latent/run_e6.py               the agents' weights
  e7-directions  experiments/E7_latent_manipulation/run_e7_directions.py  the agents' weights
  e7-constrained experiments/E7_latent_manipulation/run_e7_constrained.py the agents' weights (manipulation A's sweep)
  e7-apply       experiments/E7_latent_manipulation/run_e7_apply.py       the weights, the e7-directions outputs (plan B's reverse
                                                                          direction) and the e7-constrained outputs (plan A)

The agent ids are read from the agents file's text (`id: <name>` entries, comments left out), never by importing yaml.
"""

from __future__ import annotations

import re
from pathlib import Path

PHASES = {
    "e6": "experiments/E6_phenotypes_latent/run_e6.py",
    "e7-directions": "experiments/E7_latent_manipulation/run_e7_directions.py",
    "e7-constrained": "experiments/E7_latent_manipulation/run_e7_constrained.py",
    "e7-apply": "experiments/E7_latent_manipulation/run_e7_apply.py",
}
CONFIGS = {"e6": "configs/e6/e6.yaml", "e7-directions": "configs/e7/e7.yaml", "e7-constrained": "configs/e7/e7.yaml", "e7-apply": "configs/e7/e7.yaml"}
BATCHES = {
    "e6_e5": "configs/e6/agents_e5.yaml",          # the E5 campaign's agents (to be confirmed after wave 2)
    "e6_dev_e4": "configs/e6/agents_dev_e4.yaml",  # E4's four configurations (development; CPU)
}
DIRECTIONS_GLOB = "results/E7_latent_manipulation/e7dir__*"
CONSTRAINED_GLOB = "results/E7_latent_manipulation/e7con__*"
BATCH_CPUS = 3
MAX_PARALLEL = 4


def agent_ids(repo: Path, batch: str) -> list[str]:
    """The ids of the batch's agents file, in order."""
    if batch not in BATCHES:
        raise ValueError(f"unknown batch {batch!r}; choose among {list(BATCHES)}")
    text = (Path(repo) / BATCHES[batch]).read_text(encoding="utf-8")
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())  # commented-out entries (TODO) are not agents
    ids = re.findall(r"(?:^|[{\s,-])id:\s*([A-Za-z0-9_.\-]+)", text, re.M)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{BATCHES[batch]}: duplicate agent ids")
    return ids


def jobs(repo: Path, batch: str, phase: str, runs_root: str, device: str = "cuda", agents_per_job: int = 1,
         only: list[str] | None = None, extra: list[str] | None = None) -> list[tuple[str, list[str]]]:
    """(job name, script arguments) of every job of the phase: one per group of agents_per_job agents."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; choose among {list(PHASES)}")
    ids = agent_ids(repo, batch)
    if only:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise ValueError(f"agents not in {BATCHES[batch]}: {unknown}")
        ids = [i for i in ids if i in only]
    out = []
    for k in range(0, len(ids), max(1, int(agents_per_job))):
        group = ids[k:k + agents_per_job]
        args = [PHASES[phase], "--config", CONFIGS[phase], "--agents", BATCHES[batch], "--runs-root", runs_root, "--agent-ids", *group,
                "--device", device, "--out-dir", "results"]
        if phase == "e7-apply":
            args += ["--directions", DIRECTIONS_GLOB, "--constrained", CONSTRAINED_GLOB]
        if phase == "e7-constrained":
            args += ["--directions", DIRECTIONS_GLOB]  # the reproduction check of the first report
        out.append((f"{phase}_{'_'.join(group)}", args + list(extra or [])))
    return out
