"""Read-only extraction for E7d (v2): one point per induction-B run of D7, from the existing D7 episodes (scratch/,
nothing is rerun). Reuses D7's own per-step collection (analyse_e7_d7.collect_natural: groups, b_local, localisation,
C1 with the main z-bar) and only aggregates it per run.

Per induced-reference run (agent, mode, alpha), over its well-localised zone steps:
  delta_b  = mean b_local of the run - mean b_local of the same agent's reference without intervention
  delta_C1 = mean C1 of the run      - mean C1 of the same agent's reference without intervention
(D7's correction A uses same-step pairs instead; this is a run-vs-baseline difference, noted in MANIFEST_v2.md.)

Writes slides/data_v2/e7d_induction_runs.csv. Run: .venv/Scripts/python.exe slides/figures_v2/_extract_e7d_induction.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "E7_latent_manipulation"))

import analyse_e7_d7 as a7  # noqa: E402
import analyse_e7_diagnostics as ad  # noqa: E402
import analyse_e7_dissociation as adis  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

RUNS = str(REPO / "scratch" / "e6dev" / "results" / "E7_dissociation" / "e7d7__*__natural__*")
OUT = REPO / "slides" / "data_v2" / "e7d_induction_runs.csv"
ZBAR = "draw"


def main():
    cfg = fs.load_layered_config(REPO / "configs" / "e7" / "e7.yaml", ("e6", "e7"))
    runs = [r for r in adis.runs_of([RUNS]) if r["meta"]["phase"] == "natural"]
    if not runs:
        raise FileNotFoundError(f"no D7 natural runs under {RUNS}")
    worlds = {r["meta"]["agent"]: ad.World(cfg, r["meta"]) for r in runs}
    S = a7.collect_natural(runs, worlds)
    loc = S["localised"]
    run_of = np.array([e.rsplit(":", 1)[0] for e in S["episode"]], dtype=object)  # "agent:mode__alpha"
    agent_of = np.array([e.split(":", 1)[0] for e in S["episode"]], dtype=object)
    c1 = S[f"{ZBAR}_C1"]
    rows = []
    for agent in sorted(set(agent_of)):
        base = (agent_of == agent) & (S["group"] == "references, no intervention") & loc
        b0, c0 = float(np.nanmean(S["b"][base])), float(np.nanmean(c1[base]))
        for run in sorted(set(run_of[(agent_of == agent) & (S["group"] == "induced references")])):
            m = (run_of == run) & (S["group"] == "induced references") & loc
            if not m.any():
                continue
            tag = run.split(":", 1)[1]
            mode, alpha = tag.split("__")
            rows.append({"agent": agent, "mode": mode, "alpha": float(alpha), "n_steps": int(m.sum()),
                         "b": float(np.nanmean(S["b"][m])), "C1": float(np.nanmean(c1[m])),
                         "b_ref_none": b0, "C1_ref_none": c0,
                         "delta_b": float(np.nanmean(S["b"][m])) - b0, "delta_C1": float(np.nanmean(c1[m])) - c0})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} induction runs over {len(runs)} agents -> {OUT}")


if __name__ == "__main__":
    main()
