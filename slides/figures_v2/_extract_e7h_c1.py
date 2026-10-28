"""Read-only extraction for E7h: C1 (D7's critic controllability, main z-bar) per agent for the exposed and reference
models without intervention, from D7's existing natural-run episodes (scratch/; nothing is rerun). Reuses D7's own
per-step collection (analyse_e7_d7.collect_natural) and only averages it per agent over the well-localised zone steps,
the steps D7's bins use.

Writes slides/data_v2/e7h_c1_natural.csv. Run: .venv/Scripts/python.exe slides/figures_v2/_extract_e7h_c1.py
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
OUT = REPO / "slides" / "data_v2" / "e7h_c1_natural.csv"
ZBAR = "draw"
GROUPS = {"exposed": "exposed", "references, no intervention": "reference"}


def main():
    cfg = fs.load_layered_config(REPO / "configs" / "e7" / "e7.yaml", ("e6", "e7"))
    runs = [r for r in adis.runs_of([RUNS]) if r["meta"]["phase"] == "natural"]
    if not runs:
        raise FileNotFoundError(f"no D7 natural runs under {RUNS}")
    S = a7.collect_natural(runs, {r["meta"]["agent"]: ad.World(cfg, r["meta"]) for r in runs})
    rows = []
    for agent in sorted(set(S["agent"])):
        for group, role in GROUPS.items():
            m = (S["agent"] == agent) & (S["group"] == group) & S["localised"]
            rows.append({"agent": agent, "role": role, "n_steps": int(m.sum()),
                         "C1": float(np.nanmean(S[f"{ZBAR}_C1"][m]))})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
