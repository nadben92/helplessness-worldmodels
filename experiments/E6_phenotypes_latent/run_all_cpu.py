"""E6 / E7 on all agents, locally on the CPU (docs/e6/DECISIONS.md (63)): the jobs of one phase, run by a pool of worker
processes (one thread each), one log per job under --logs, one output directory per job under scratch/all/<job>/ (git
ignores it; the run scripts name their directories by agent and second, which parallel jobs of one agent would share).
The jobs are the existing scripts' command lines; nothing of the measures is here.

Phases:
  e6           run_e6.py, one job per (agent, image) of configs/e6/images_by_agent_all.yaml
  c12          run_e7_d7.py --phase natural --no-overlay --image <name>, one job per (agent, image)
  directions   run_e7_directions.py, one job per agent (the constrained sweep's reproduction check)
  constrained  run_e7_constrained.py, one job per agent (needs directions)
  oracle       run_e7_diagnostics.py --phase lookahead, one job per agent
  anti         run_e7_dissociation.py: references p in 0..0.5 and 0.6..1 (two jobs), exposed p in {0, 1}; k in {1, 5}
  corrected    run_e7_d7.py --phase corrected at each agent's (48) setting (read from its constrained sweep)

Usage, from the repo root (src of this checkout on PYTHONPATH):
  python experiments/E6_phenotypes_latent/run_all_cpu.py --phase e6 --runs-root <root> --workers 12 --logs <dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from helplessness.config import REPO_ROOT

AGENTS = "configs/e6/agents_e5.yaml"
IMAGES = "configs/e6/images_by_agent_all.yaml"
E6_CONFIG, E7_CONFIG = "configs/e6/e6_all.yaml", "configs/e7/e7_all.yaml"
E6 = "experiments/E6_phenotypes_latent/run_e6.py"
E7 = "experiments/E7_latent_manipulation/"
OUT = "scratch/all"
P_LO, P_HI = ["0", "0.1", "0.2", "0.3", "0.4", "0.5"], ["0.6", "0.7", "0.8", "0.9", "1"]


def newest(pattern: str) -> str:
    """The newest finished run directory matching <OUT>/<job>/<pattern> (newest by its name's UTC stamp)."""
    found = sorted((d for d in glob.glob(str(REPO_ROOT / OUT / "*" / pattern))
                    if json.loads(Path(d, "status.json").read_text())["status"] == "done"), key=lambda d: Path(d).name)
    if not found:
        raise SystemExit(f"no finished run matches {pattern}")
    return found[-1]


def best_setting(con_dir: str) -> tuple[float, float]:
    """(48): the kept setting of the agent's sweep (training image) with the largest specific gain."""
    rows = [r for r in json.loads(Path(con_dir, "settings.json").read_text())["settings"] if r["image"] == "training" and r["kept"]]
    if not rows:
        raise SystemExit(f"{con_dir}: no kept setting")
    best = max(rows, key=lambda r: r["specific_gain"])
    return float(best["lam"]), float(best["mu_pos"])


def jobs(phase: str, py: str, runs_root: str) -> list[tuple[str, list[str]]]:
    ids = [a["id"] for a in yaml.safe_load((REPO_ROOT / AGENTS).read_text(encoding="utf-8"))["agents"]]
    images = yaml.safe_load((REPO_ROOT / IMAGES).read_text(encoding="utf-8"))
    common = ["--agents", AGENTS, "--runs-root", runs_root, "--device", "cpu"]
    out = []
    for a in ids:
        one = [*common, "--agent-ids", a]
        if phase == "e6":
            out += [(f"e6_{a}_{im}", [py, "-u", E6, *one, "--config", E6_CONFIG, "--images", im]) for im in images[a]]
        elif phase == "c12":
            out += [(f"c12_{a}_{im}", [py, "-u", E7 + "run_e7_d7.py", "--phase", "natural", "--no-overlay", "--image", im,
                                        "--config", E7_CONFIG, *one]) for im in images[a]]
        elif phase == "directions":
            out.append((f"dir_{a}", [py, "-u", E7 + "run_e7_directions.py", *one, "--config", E7_CONFIG, "--images", "training"]))
        elif phase == "constrained":
            out.append((f"con_{a}", [py, "-u", E7 + "run_e7_constrained.py", *one, "--config", E7_CONFIG,
                                     "--directions", newest(f"E7_latent_manipulation/e7dir__{a}__*")]))
        elif phase == "oracle":
            out.append((f"oracle_{a}", [py, "-u", E7 + "run_e7_diagnostics.py", "--phase", "lookahead", *one, "--config", E7_CONFIG]))
        elif phase == "anti":
            dis = [py, "-u", E7 + "run_e7_dissociation.py", *one, "--config", E7_CONFIG, "--ks", "1", "5", "--h-mode", "mean"]
            out += [(f"anti_{a}_ref_lo", [*dis, "--role", "reference", "--p-values", *P_LO, "--tag", "lo"]),
                    (f"anti_{a}_ref_hi", [*dis, "--role", "reference", "--p-values", *P_HI, "--tag", "hi"]),
                    (f"anti_{a}_exp", [*dis, "--role", "experimental", "--p-values", "0", "1", "--tag", "ctl"])]
        elif phase == "corrected":
            con = newest(f"E7_latent_manipulation/e7con__{a}__*")
            lam, mu = best_setting(con)
            out.append((f"cor_{a}", [py, "-u", E7 + "run_e7_d7.py", "--phase", "corrected", "--lam", f"{lam:g}", "--mu-pos", f"{mu:g}",
                                     "--constrained", con, "--config", E7_CONFIG, *one]))
        else:
            raise SystemExit(f"unknown phase {phase}")
    return [(name, [*cmd, "--out-dir", f"{OUT}/{name}"]) for name, cmd in out]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    todo = jobs(args.phase, sys.executable, args.runs_root)
    if args.dry_run:
        for name, cmd in todo:
            print(name, " ".join(cmd))
        return 0
    args.logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def run(job):
        name, cmd = job
        with open(args.logs / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{time.time() - t0:7.0f} s] {name}: exit {code}", flush=True)
        return name, code

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, todo))
    failed = [n for n, c in results if c != 0]
    print(f"{args.phase}: {len(results) - len(failed)}/{len(results)} done in {time.time() - t0:.0f} s; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
