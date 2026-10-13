"""Integrity of the gpu_campaign handoff (docs/e3/DECISIONS.md, 2026-10-13 (31)), before any analysis. Read-only on the
handoff; extracts its archives into a working directory only when every check passes.

For each of the batch's 8 jobs (the 4 new runs, seeds 1 and 6, and the 4 pilot runs measured as campaign seeds, seeds 0
and 5; submit_kubeflow.AGENT_BATCHES["gpu_campaign"]):
  - the supervisor's record (handoff.json): started, exit code 0, meta.yaml status "done", results.json written, the
    budget reached (300,000 environment steps);
  - its three archives (<run>.tgz, <run>.measures.tgz, <run>.weights.tgz), and inside them:
      * meta.yaml: status done, 300,000 environment steps; the code hash -- the campaign's (RUN_GPU.md, step 1) for the
        new runs and for the pilot measurement jobs, plus the pilot's training code (DECISIONS (30)) in the pilot jobs'
        source_run_meta; the environment steps, the switch;
      * checkpoints.jsonl: E2's grid up to 300,000 (and any extra checkpoint of (29)), each record with the campaign's
        goal diagnostics (goal_draws, Q gaps next to the goal, V by zone, the zones' separation);
      * counts.npz and twin.npz: one entry per checkpoint;
      * measures/env<NNNNNN>/ (results.json, arrays.npz) at every checkpoint after the switch, and only those, the last
        one at 300,000; arrays.npz and results.json at the top (the final measures);
      * the weights at every checkpoint and the final ones, and replay_final.npz;
  - interruptions: a resume (meta.yaml resumed_at, "RESUMING" in console.log) or a "## Interruptions" section in
    handoff.md is REPORTED with what it says (a resume with the user's agreement is allowed, RUN_GPU.md step 4b);
  - both conditions of each seed present.
Any failure: IntegrityError listing every problem; nothing is extracted, nothing is analysed.

Usage: python experiments/E3_worldmodel_4zones/campaign_integrity.py handoff/gpu_campaign [--extract-to DIR]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import yaml

from helplessness.experiment_loader import load_experiment_module

# the campaign's code (RUN_GPU.md step 1, commit 7980c9e) and the pilot's training code (DECISIONS (30))
CAMPAIGN_SOURCE_SHA256 = "d84dec84a3bd9542ea4846502ae30b20e9909c9c7b49b6e8ac019a4fd21d1a80"
PILOT_TRAINING_SHA256 = "3a71c7453c4aaa51175fe714ad7fc2e3ea3a825e51659fc4219cabd550bc7e15"
BUDGET = 300_000
NEW_SEEDS, PILOT_SEEDS = (1, 6), (0, 5)
CONDITIONS = ("experimental", "reference")
ROLE = {"experimental": "experimental", "reference": "control"}
GOAL_DIAGNOSTIC_KEYS = ("goal_draws", "goal_neighbours", "V_by_zone", "latent_zone_separation")


class IntegrityError(RuntimeError):
    """The handoff is incomplete or inconsistent: nothing is analysed."""


def expected_jobs() -> dict[str, dict]:
    """job name -> {seed, condition, pilot}: the batch's 8 jobs."""
    jobs = {f"seed{s}_{c}": {"seed": s, "condition": c, "pilot": False} for s in NEW_SEEDS for c in CONDITIONS}
    jobs.update({f"pilot_win64_seed{s}_{c}": {"seed": s, "condition": c, "pilot": True} for s in PILOT_SEEDS for c in CONDITIONS})
    return jobs


def grid_points(budget: int = BUDGET) -> list[int]:
    run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "campaign_integrity_run_mini")
    return sorted({c for c in run_mini.e2_grid_env_steps() if c <= budget} | {budget})


class Archive:
    """A .tgz read in place: member names relative to the run directory, and file contents on demand."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tar = tarfile.open(self.path, "r:gz")
        self.members = {}
        for m in self.tar.getmembers():
            parts = Path(m.name).parts
            if len(parts) >= 3 and parts[0] == "results":
                self.members["/".join(parts[3:])] = m  # results/E3_worldmodel_4zones/<run>/...
            elif len(parts) >= 2:
                self.members["/".join(parts[1:])] = m

    def has(self, name: str) -> bool:
        return name in self.members and self.members[name].isfile()

    def read(self, name: str) -> bytes:
        return self.tar.extractfile(self.members[name]).read()

    def npz_keys(self, name: str) -> list[str]:
        return list(np.load(io.BytesIO(self.read(name))).files)

    def close(self) -> None:
        self.tar.close()


def check_run(job: str, spec: dict, row: dict | None, handoff: Path, grid: list[int]) -> tuple[list[str], dict]:
    """The problems of one job, and what is reported about it (interruptions, switch, measured checkpoints)."""
    problems, info = [], {"job": job, **spec}
    if row is None:
        return [f"{job}: absent from handoff.json"], info
    if row.get("not_started"):
        return [f"{job}: not started"], info
    for key, want in (("exit_code", 0), ("status", "done"), ("finished", True)):
        if row.get(key) != want:
            problems.append(f"{job}: {key} = {row.get(key)!r}, expected {want!r}")
    run = row.get("run_dir")
    if not run:
        return problems + [f"{job}: no run directory"], info
    info["run_dir"] = run
    if f"seed{spec['seed']:02d}" not in run or f"pair-{ROLE[spec['condition']]}" not in run:
        problems.append(f"{job}: run directory {run} is not seed {spec['seed']} {spec['condition']}")
    archives = {kind: handoff / f"{run}{suffix}" for kind, suffix in (("run", ".tgz"), ("measures", ".measures.tgz"), ("weights", ".weights.tgz"))}
    missing = [str(p.name) for p in archives.values() if not p.exists()]
    if missing:
        return problems + [f"{job}: archive(s) missing: {missing}"], info
    a_run, a_meas, a_w = (Archive(archives[k]) for k in ("run", "measures", "weights"))
    try:
        for name in ("meta.yaml", "results.json", "checkpoints.jsonl", "counts.npz", "twin.npz", "arrays.npz", "config.yaml", "console.log"):
            if not a_run.has(name):
                problems.append(f"{job}: {name} missing from {archives['run'].name}")
        if problems:
            return problems, info
        meta = yaml.safe_load(a_run.read("meta.yaml"))
        info["source_sha256"] = meta.get("source_sha256")
        if meta.get("status") != "done":
            problems.append(f"{job}: meta.yaml status {meta.get('status')!r}")
        if int(meta.get("env_steps") or -1) != BUDGET:
            problems.append(f"{job}: meta.yaml env_steps {meta.get('env_steps')}, expected {BUDGET}")
        if meta.get("source_sha256") != CAMPAIGN_SOURCE_SHA256:
            problems.append(f"{job}: code {meta.get('source_sha256')}, expected the campaign's {CAMPAIGN_SOURCE_SHA256}")
        if spec["pilot"]:
            src = (meta.get("source_run_meta") or {}).get("source_sha256")
            info["training_source_sha256"] = src
            if src != PILOT_TRAINING_SHA256:
                problems.append(f"{job}: the pilot run's training code {src}, expected {PILOT_TRAINING_SHA256}")
        if int(meta.get("seed", -1)) != spec["seed"] or meta.get("condition") != spec["condition"]:
            problems.append(f"{job}: meta.yaml seed / condition {meta.get('seed')} / {meta.get('condition')}")
        # interruptions: reported, never a refusal by themselves
        log = a_run.read("console.log").decode("utf-8", errors="replace")
        resumes = [line.strip() for line in log.splitlines() if "RESUMING" in line or re.search(r"resumed at environment step", line)]
        info["interruptions"] = {"resumed_at": meta.get("resumed_at"), "log_lines": resumes}
        records = [json.loads(line) for line in a_run.read("checkpoints.jsonl").decode("utf-8").splitlines() if line.strip()]
        steps = [r["env_step"] for r in records]
        info["checkpoints"] = steps
        if steps != sorted(steps) or not set(grid) <= set(steps) or steps[-1] != BUDGET:
            problems.append(f"{job}: checkpoints {steps} do not cover E2's grid up to {BUDGET}")
        for r in records:
            g = r.get("goal_diagnostics") or {}
            lacking = [k for k in GOAL_DIAGNOSTIC_KEYS if k not in g]
            if lacking:
                problems.append(f"{job}: checkpoint {r['env_step']}: goal diagnostics without {lacking}")
        results = json.loads(a_run.read("results.json"))
        switch_env = results.get("switch_env_step")
        if switch_env is None:  # a new run: the switch is in its console log
            m = re.search(r">>> SWITCH.*?environment step (\d+)", log)
            switch_env = int(m.group(1)) if m else None
        info["switch_env_step"] = switch_env
        if switch_env is None:
            problems.append(f"{job}: no switch found")
            return problems, info
        counts, twin = set(a_run.npz_keys("counts.npz")), set(a_run.npz_keys("twin.npz"))
        for c in steps:
            if f"N_{c}" not in counts:
                problems.append(f"{job}: counts.npz without N_{c}")
            if f"T_hat_{c}" not in twin:
                problems.append(f"{job}: twin.npz without T_hat_{c}")
        after = [c for c in steps if c > switch_env]
        measured = sorted(int(re.search(r"env(\d{6})", n).group(1)) for n in a_meas.members if re.fullmatch(r"measures/env\d{6}/results\.json", n))
        info["measured_checkpoints"] = measured
        if measured != after:
            problems.append(f"{job}: measures at {measured}, expected every checkpoint after the switch ({switch_env}): {after}")
        for c in measured:
            if not a_meas.has(f"measures/env{c:06d}/arrays.npz"):
                problems.append(f"{job}: measures/env{c:06d}/arrays.npz missing")
        if row.get("measured_checkpoints") not in (None, len(measured)):
            problems.append(f"{job}: handoff.json counts {row.get('measured_checkpoints')} measured checkpoints, the archive {len(measured)}")
        for c in steps:
            if not a_w.has(f"checkpoints/agent_env{c:06d}.pt") and not (c == steps[-1] and a_w.has("checkpoints/agent_final.pt")):
                problems.append(f"{job}: weights at {c} missing")
        for name in ("checkpoints/agent_final.pt", "replay_final.npz"):
            if not a_w.has(name):
                problems.append(f"{job}: {name} missing from {archives['weights'].name}")
        info["replay_env_steps_final"] = results.get("replay_env_steps_final", BUDGET)
    finally:
        for a in (a_run, a_meas, a_w):
            a.close()
    return problems, info


def interruption_section(handoff_md: str) -> str | None:
    m = re.search(r"^## Interruptions\s*$(.*?)(?=^## |\Z)", handoff_md, flags=re.M | re.S)
    return m.group(1).strip() if m else None


def check_handoff(handoff: Path) -> dict:
    """Every check; raises IntegrityError with the full list of problems."""
    handoff = Path(handoff)
    problems = []
    for name in ("handoff.json", "handoff.md"):
        if not (handoff / name).exists():
            raise IntegrityError(f"{handoff}: {name} missing")
    rows = {r["job"]: r for r in json.loads((handoff / "handoff.json").read_text(encoding="utf-8"))["runs"]}
    md = (handoff / "handoff.md").read_text(encoding="utf-8")
    jobs = expected_jobs()
    extra = sorted(set(rows) - set(jobs))
    if extra:
        problems.append(f"unexpected jobs in handoff.json: {extra}")
    grid = grid_points()
    infos = []
    for job, spec in jobs.items():
        p, info = check_run(job, spec, rows.get(job), handoff, grid)
        problems += p
        infos.append(info)
    if "STOP" in md and "## Stop" in md:
        problems.append("handoff.md has a '## Stop' section")
    report = {"handoff": str(handoff), "runs": infos, "interruptions_section": interruption_section(md), "problems": problems,
              "passed": not problems}
    if problems:
        raise IntegrityError("the gpu_campaign handoff is incomplete -- nothing analysed:\n  " + "\n  ".join(problems))
    return report


def extract(handoff: Path, report: dict, dest: Path) -> list[Path]:
    """Extracts the three archives of every run into dest/<run>/ (the archives' own layout, results/... stripped);
    refuses a dest that already holds one of the runs. Returns the run directories."""
    dest = Path(dest)
    out = []
    for info in report["runs"]:
        run = info["run_dir"]
        target = dest / run
        if target.exists():
            raise FileExistsError(f"{target} exists: extract into a new directory")
        for suffix in (".tgz", ".measures.tgz", ".weights.tgz"):
            with tarfile.open(Path(handoff) / f"{run}{suffix}", "r:gz") as tar:
                members = []
                for m in tar.getmembers():
                    parts = Path(m.name).parts
                    rel = parts[3:] if parts and parts[0] == "results" else parts[1:]
                    if not rel:
                        continue
                    m.name = str(Path(run, *rel)).replace("\\", "/")
                    members.append(m)
                tar.extractall(dest, members=members, filter="data")
        out.append(target)
    return out


def markdown(report: dict) -> str:
    lines = ["## Integrity of the handoff", "", f"`{report['handoff']}`: **{'passed' if report['passed'] else 'FAILED'}**.", "",
             "| job | seed | condition | code (source_sha256) | switch (env step) | checkpoints measured | final replay at | interruptions |",
             "|---|---|---|---|---|---|---|---|"]
    for r in report["runs"]:
        inter = r.get("interruptions") or {}
        txt = "none" if not inter.get("resumed_at") and not inter.get("log_lines") else f"resumed at {inter.get('resumed_at')}: {len(inter.get('log_lines', []))} log line(s)"
        code = (r.get("source_sha256") or "?")[:12] + ("" if not r.get("training_source_sha256") else f" (training {r['training_source_sha256'][:12]})")
        lines.append(f"| {r['job']} | {r['seed']} | {r['condition']} | {code} | {r.get('switch_env_step')} | {len(r.get('measured_checkpoints', []))} | "
                     f"{r.get('replay_env_steps_final')}{' (FLAGGED: last resume point)' if r.get('replay_env_steps_final') != BUDGET else ''} | {txt} |")
    if report.get("interruptions_section"):
        lines += ["", "handoff.md, `## Interruptions`:", "", report["interruptions_section"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--extract-to", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = check_handoff(args.handoff)
    except IntegrityError as err:
        print(f"REFUSED: {err}", flush=True)
        return 1
    print(markdown(report))
    if args.extract_to is not None:
        extract(args.handoff, report, args.extract_to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
