"""Slides v3: E7_bars_650k -- E7_bars_critic on the campaign on all agents (shuffle at 650,000; docs/e6/DECISIONS.md (63),
report section 21): b at step 1, the critic's controllability C1 (D7, main z_bar = draw), localisation in the zone and
directed share (episodes, k = 1), training image, for exposed, latent fix (unconstrained stepwise, alpha = 1),
constrained fix (each agent's (48) setting), oracle (true transitions in the planner, k = 1) and reference. Two bars per
condition: the five baselines (E4's four pairs and baseline B, 300,000) and the three shuffle agents (A, E, F, 650,000),
each the mean of its agents; black dots = the agents (circles: baselines, triangles: shuffle). As in v2 / v3: the
oracle's b is the reference's (the true belief, not a measurement), its localisation the exposed model's (h is not
modified); no C1 for the latent fix and the oracle (not measured). Grey dashed: chance (2/5) on the directed share.

Reads the campaign's finished runs (scratch/all/<job>/, status "done" only) and its analysis
(results/E6_phenotypes_latent/all_agents_650k/: pairs.csv for C1, forms.json for each agent's constrained setting,
d7/d7_correction.csv for the corrected C1) read-only; nothing is recomputed. The extracted values are written to
slides/out_v3/data/E7_bars_650k.csv. Run from the repo root:
  python slides/figures_v3/E7_bars_650k.py"""
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "slides" / "out_v3"
ALL = REPO / "scratch" / "all"
AN = REPO / "results" / "E6_phenotypes_latent" / "all_agents_650k"
EXPOSED, REFERENCE, ORACLE, CONSTR, FREE = "#E8412C", "#1F6FEB", "#1DB954", "#6718AB", "#B98CE8"
BASELINES = ["p1-a_e4", "base_A", "p2-a_e4", "p2-b_e4", "base_B"]
SHUFFLE = ["f1_A", "f1_E", "f1_F"]
AGENTS = BASELINES + SHUFFLE
CONDS = ["exposed", "unconstrained", "constrained", "oracle", "reference"]
MEASURES = ("b", "C1", "loc", "dir")
NAN = float("nan")

plt.style.use("default")
plt.rcParams.update({"axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15, "lines.linewidth": 2.8,
                     "svg.fonttype": "none"})


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def run_dir(job: str) -> Path:
    """The job's one finished run directory."""
    found = [Path(d).parent for d in glob.glob(str(ALL / job / "*" / "*" / "status.json"))
             if json.loads(Path(d).read_text(encoding="utf-8"))["status"] == "done"]
    if len(found) != 1:
        raise SystemExit(f"{job}: {len(found)} finished runs (expected 1)")
    return found[0]


def collect() -> dict:
    """v[condition][measure][agent]; NaN when not measured."""
    v = {c: {m: {a: NAN for a in AGENTS} for m in MEASURES} for c in CONDS}
    forms = json.loads((AN / "forms.json").read_text(encoding="utf-8"))
    c1_pairs = {(r["agent"], r["role"]): num(r["C1"]) for r in rd(AN / "pairs.csv") if r["image"] == "training"}
    c1_corr = {(r["agent"], r["setting"]): num(r["C1_corr"]) for r in rd(AN / "d7" / "d7_correction.csv") if r["zbar"] == "draw"}
    for a in AGENTS:
        for r in json.loads((run_dir(f"e6_{a}_training") / "rows.json").read_text(encoding="utf-8")):
            c = r["role"]
            v[c]["b"][a], v[c]["loc"][a], v[c]["dir"][a] = r["b_step1"], r["loc_template_zone"], r["directed_share"]
            v[c]["C1"][a] = c1_pairs[(a, c)]
        free = [r for r in json.loads((run_dir(f"free_{a}") / "rows.json").read_text(encoding="utf-8"))
                if r["image"] == "training" and r["mode"] == "stepwise" and float(r["alpha"]) == 1.0]
        assert len(free) == 1, (a, len(free))
        v["unconstrained"]["b"][a], v["unconstrained"]["loc"][a] = free[0]["b_step1"], free[0]["loc_template_zone"]
        v["unconstrained"]["dir"][a] = free[0]["directed_share"]
        con = json.loads((run_dir(f"cor_{a}") / "summary.json").read_text(encoding="utf-8"))["rows"][0]
        assert con["setting"] == forms["5"][a]["setting"], (a, con["setting"])
        v["constrained"]["b"][a], v["constrained"]["loc"][a] = con["b_step1"], con["loc_template_zone"]
        v["constrained"]["dir"][a], v["constrained"]["C1"][a] = con["directed_share"], c1_corr[(a, con["setting"])]
        ora = [r for r in json.loads((run_dir(f"oracle_{a}") / "summary.json").read_text(encoding="utf-8"))["rows"]
               if r["model"] == "exposed" and r["k"] == 1]
        assert len(ora) == 1, (a, len(ora))
        v["oracle"]["dir"][a] = ora[0]["directed_share"]
        v["oracle"]["b"][a], v["oracle"]["loc"][a] = v["reference"]["b"][a], v["exposed"]["loc"][a]
    return v


def write_data(v: dict) -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    with open(OUT / "data" / "E7_bars_650k.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent", "group", "condition", *MEASURES])
        for a in AGENTS:
            for c in CONDS:
                w.writerow([a, "baseline" if a in BASELINES else "shuffle", c, *(v[c][m][a] for m in MEASURES)])


def figure(v: dict) -> None:
    cats = [("exposed", "exposed", EXPOSED), ("latent fix", "unconstrained", FREE),
            ("constrained fix", "constrained", CONSTR), ("oracle", "oracle", ORACLE), ("reference", "reference", REFERENCE)]
    panels = [("b", "b (bits)"), ("C1", "critic controllability"), ("loc", "localisation"), ("dir", "directed share")]
    groups = [(BASELINES, -0.2, None, "o"), (SHUFFLE, 0.2, "//", "^")]
    fig, axes = plt.subplots(1, 4, figsize=(13.33, 7.5))
    for ax, (m, lab) in zip(axes, panels):
        for i, (_, c, col) in enumerate(cats):
            for agents, dx, hatch, marker in groups:
                vals = np.array([v[c][m][a] for a in agents], float)
                ok = np.isfinite(vals)
                if not ok.any():
                    continue
                ax.bar(i + dx, vals[ok].mean(), width=0.38, color=col, edgecolor="black", lw=1, hatch=hatch)
                ax.scatter(i + dx + np.linspace(-0.1, 0.1, len(agents))[ok], vals[ok], s=22, color="black", marker=marker, zorder=4)
        if m == "dir":
            ax.axhline(0.4, color="#888888", ls="--", lw=2.2, zorder=1)
        ax.set_xticks(range(len(cats)), [c for c, _, _ in cats], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_ylabel(lab)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylim(-2.35, 0.45)
    for ax in axes[1:]:
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "E7_bars_650k.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    v = collect()
    write_data(v)
    figure(v)


if __name__ == "__main__":
    main()
