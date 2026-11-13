"""E7 on all agents at 650,000 (report section 21): is the shuffle agents' directed share under the latent fix
(unconstrained stepwise correction, alpha = 1, training image) an artefact of their poor localisation? Reads the free
phase rerun with its episodes (run_e7_apply.py --save-episodes --episode-fields actions decoded_cell decoded_template
loc_template loc_agent p_goal, out dir scratch/free_eps_650k_fields/free_<agent>/; same command line as the campaign's
free phase otherwise, rows identical to it) read-only. On the zone steps of the exposed model's episodes (mode none) and
of the latent fix's (mode stepwise): the share of steps whose decoded agent cell is the true cell; among the others,
the decoded cell's distance to the goal against the true cell's (lower: shifted toward the goal), the error's size in
cells; the directed share (the true transition lowers the true distance to the goal) on localised and mislocalised
steps; and, on mislocalised steps, the share whose action is directed from the decoded cell (what an agent acting on
its decoded position would do).

Writes results/E6_phenotypes_latent/all_agents_650k/free_offsets.csv. Run from the repo root:
  python experiments/E6_phenotypes_latent/free_offsets_650k.py
"""

from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E7_latent_manipulation"))

import analyse_e7_diagnostics as ad  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.measures import zone_phenotypes as zp  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

AGENTS = ("f1_A", "f1_E", "f1_F")
RUNS = REPO_ROOT / "scratch" / "free_eps_650k_fields"
OUT = REPO_ROOT / "results" / "E6_phenotypes_latent" / "all_agents_650k" / "free_offsets.csv"
N_COLS = 12


def main() -> int:
    cfg = fs.load_layered_config(REPO_ROOT / "configs" / "e7" / "e7_all.yaml", ("e6", "e7"))
    w = ad.World(cfg, {"configuration": "p1-b-d1", "slot": "nw"})  # the shuffle agents' configuration, training image
    rows = []
    for a in AGENTS:
        found = glob.glob(str(RUNS / f"free_{a}" / "*" / "*" / "episodes.npz"))
        if len(found) != 1:
            raise SystemExit(f"{a}: {len(found)} episode files")
        z = np.load(found[0])
        for mode, key in (("exposed", "none__0"), ("latent fix", "stepwise__1")):
            st, act, dec = (z[f"training__{key}__{k}"] for k in ("states", "actions", "decoded_cell"))
            # moved to measures.zone_phenotypes.localisation_split on 2026-11-05 (docs/e8/DECISIONS.md (11))
            rows.append({"agent": a, "mode": mode, **zp.localisation_split(st, act, dec, w.in_zone, w.dist, w.directed, N_COLS)})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
