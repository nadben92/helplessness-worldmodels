"""E7 on all agents at 650,000 (report section 21): the correction against the oracle at equal planning depth, and the
forms K1, K5 and M written before the computation. Reads run_e7_diagnostics.py --phase values --constrained <the
agent's constrained sweep> (scratch/values_650k/val_<agent>/...) read-only: per agent, the greedy directed share of the
exposed model's Q_k on the zone's cells for the variants oracle, model and model_corrected, and b at the root and along
the imagined steps (trace_b) for model and model_corrected.

Writes results/E6_phenotypes_latent/all_agents_650k/kdepth.csv (one row per agent and k in K_REPORT) and
kdepth_forms.json, and kdepth_trace.csv (description, not a form: along the first 5 imagined steps, the share of
traced particles decoded in the zone and their b in the zone). Run from the repo root:
  python experiments/E6_phenotypes_latent/kdepth_650k.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

from helplessness.config import REPO_ROOT
from helplessness.pipeline import frozen_setup as fs

AGENTS = ["p1-a_e4", "base_A", "p2-a_e4", "p2-b_e4", "base_B", "f1_A", "f1_E", "f1_F"]
SHUFFLE = ("f1_A", "f1_E", "f1_F")
RUNS = REPO_ROOT / "scratch" / "values_650k"
AN = REPO_ROOT / "results" / "E6_phenotypes_latent" / "all_agents_650k"
VARIANTS = ("oracle", "model", "model_corrected")
K_REPORT = (1, 2, 3, 5, 10)


def run_dir(agent: str) -> Path:
    found = [Path(d).parent for d in glob.glob(str(RUNS / f"val_{agent}" / "*" / f"e7diag__{agent}__values__*" / "status.json"))
             if json.loads(Path(d).read_text(encoding="utf-8"))["status"] == "done"]
    if len(found) != 1:
        raise SystemExit(f"{agent}: {len(found)} finished values runs")
    return found[0]


def main() -> int:
    setting5 = {a: r["setting"] for a, r in json.loads((AN / "forms.json").read_text(encoding="utf-8"))["5"].items() if a in AGENTS}
    rows, per_agent = [], {}
    for a in AGENTS:
        d = run_dir(a)
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if s["constrained_setting"]["setting"] != setting5[a]:
            raise SystemExit(f"{a}: values used {s['constrained_setting']['setting']}, form 5 {setting5[a]}")
        v = np.load(d / "values.npz")
        g = {var: {r["k"]: r["greedy_directed"] for r in s["variants"][f"experimental__{var}"]} for var in VARIANTS}
        b_root = {var: float(np.nanmean(v[f"experimental__{var}__b_root"])) for var in ("model", "model_corrected")}
        b_trace = {var: np.nanmean(v[f"experimental__{var}__trace_b"].reshape(v[f"experimental__{var}__trace_b"].shape[0], -1), axis=1)
                   for var in ("model", "model_corrected")}  # (k_max,): b after i + 1 imagined steps
        per_agent[a] = {"g": g, "b_root": b_root, "b_trace": b_trace}
        for k in K_REPORT:
            rows.append({"agent": a, "setting": setting5[a], "k": k, **{f"directed_{var}": g[var][k] for var in VARIANTS},
                         "b_model": float(b_trace["model"][k - 1]),
                         "b_corrected": float(b_trace["model_corrected"][k - 1]),
                         "b_model_root": b_root["model"], "b_corrected_root": b_root["model_corrected"]})
    # description (not a form): along the imagined steps, the share of traced particles decoded in the zone and their b
    zone = fs.zone_states("nw")  # the training image's zone (every agent's training image has the UD zone in slot nw)
    trace_rows = []
    for a in AGENTS:
        v = np.load(run_dir(a) / "values.npz")
        for var in ("model", "model_corrected"):
            dec, b = v[f"experimental__{var}__trace_decoded"], v[f"experimental__{var}__trace_b"]
            for i in range(5):
                inz = np.isin(dec[i], zone)
                trace_rows.append({"agent": a, "variant": var, "imagined_step": i + 1, "share_in_zone": float(inz.mean()),
                                   "b_in_zone": float(np.nanmean(b[i][inz])) if inz.any() else np.nan, "b_all": float(np.nanmean(b[i]))})
    with open(AN / "kdepth_trace.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0]))
        w.writeheader()
        w.writerows(trace_rows)
    with open(AN / "kdepth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    forms = {}
    for a in SHUFFLE:
        g, br, bt = per_agent[a]["g"], per_agent[a]["b_root"], per_agent[a]["b_trace"]
        gain = br["model_corrected"] - br["model"]
        lost = br["model_corrected"] - float(bt["model_corrected"][4])
        forms[a] = {"K1": {"corrected": g["model_corrected"][1], "oracle": g["oracle"][1],
                           "holds": bool(abs(g["model_corrected"][1] - g["oracle"][1]) <= 0.10)},
                    "K5": {"corrected": g["model_corrected"][5], "oracle": g["oracle"][5],
                           "holds": bool(g["model_corrected"][5] <= g["oracle"][5] - 0.10)},
                    "M": {"b_model_root": br["model"], "b_corrected_root": br["model_corrected"], "b_corrected_step5": float(bt["model_corrected"][4]),
                          "gain_at_root": gain, "lost_by_step5": lost, "holds": bool(lost >= 0.5 * gain)}}
    forms["verdict"] = {f: all(forms[a][f]["holds"] for a in SHUFFLE) for f in ("K1", "K5", "M")}
    (AN / "kdepth_forms.json").write_text(json.dumps(forms, indent=1), encoding="utf-8")
    for r in rows:
        print(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    print(json.dumps(forms["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
