"""E7h (v2) -- Why the latent correction fails, one figure, the same four E4 agents everywhere (training image).
Four panels (b at step 1, critic C1, localisation in the zone, directed share) x four conditions:
  exposed        exposed model, no intervention                      E6 points.csv; C1: data_v2/e7h_c1_natural.csv
  unconstrained  stepwise correction, alpha = 1, no constraint       dev_e4_cpu/analysis/checks.csv; C1: not computed
  constrained    stepwise constrained correction, alpha = 1, the     D7 corrected run of the recorded setting
                 recorded setting of each agent (d7_identity.csv)    (summary.json); C1: d7_correction.csv C1_corr
  reference      reference model, no intervention                    E6 points.csv (b = 0 by construction); C1 as exposed
One marker per agent (same shape in every panel), short black line = mean of the available agents. Grey dashed: chance
(2/5) for C1 and the directed share. A missing value leaves the cell empty."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7h_correction_summary"
E6 = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
V1 = ("results", "E7_latent_manipulation", "dev_e4_cpu", "analysis", "checks.csv")
D7 = ("results", "E7_dissociation", "d7")
C1_NAT = ("slides", "data_v2", "e7h_c1_natural.csv")
AGENTS = ["p1-a-d1_e4", "p1-b-d1_e4", "p2-a-d1_e4", "p2-b-d1_e4"]
MARKERS = ["o", "s", "^", "D"]
CONDITIONS = [("exposed", P.EXPOSED), ("unconstrained", "#B98CE8"), ("constrained", P.CORRECTION),
              ("reference", P.REFERENCE)]
PANELS = [("b", "b (bits)"), ("C1", "critic C1"), ("loc", "localisation"), ("dir", "directed share")]


def collect() -> dict:
    """values[condition][measure][agent] (NaN when missing)."""
    nan = float("nan")
    v = {c: {m: {a: nan for a in AGENTS} for m, _ in PANELS} for c, _ in CONDITIONS}
    for r in C.read_csv(C.path(*E6)):
        if r["image"] == "training":
            c = r["role"]
            v[c]["b"][r["agent"]] = C.num(r["b_step1"])
            v[c]["loc"][r["agent"]] = C.num(r["loc_template_zone"])
            v[c]["dir"][r["agent"]] = C.num(r["directed_share"])
    for r in C.read_csv(C.path(*C1_NAT)):
        v[r["role"]]["C1"][r["agent"]] = C.num(r["C1"])
    for r in C.read_csv(C.path(*V1)):
        if r["image"] == "training" and r["mode"] == "stepwise" and C.num(r["alpha"]) == 1.0:
            v["unconstrained"]["b"][r["agent"]] = C.num(r["b_step1"])
            v["unconstrained"]["loc"][r["agent"]] = C.num(r["loc_template_zone"])
            v["unconstrained"]["dir"][r["agent"]] = C.num(r["directed_share"])
    recorded = {r["agent"]: r["setting"] for r in C.read_csv(C.path(*D7, "analysis", "d7_identity.csv"))
                if r["phase"] == "corrected" and C.truthy(r["recorded"])}
    corr = {(r["agent"], r["setting"]): r for r in C.read_csv(C.path(*D7, "analysis", "d7_correction.csv"))
            if r["zbar"] == "draw"}
    for d in sorted(C.path(*D7, "runs").iterdir()):
        if "__corrected__" not in d.name:
            continue
        r = json.loads((d / "summary.json").read_text(encoding="utf-8"))["rows"][0]
        if recorded.get(r["agent"]) != r["setting"]:
            continue
        v["constrained"]["b"][r["agent"]] = r["b_step1"]
        v["constrained"]["loc"][r["agent"]] = r["loc_template_zone"]
        v["constrained"]["dir"][r["agent"]] = r["directed_share"]
        v["constrained"]["C1"][r["agent"]] = C.num(corr[(r["agent"], r["setting"])]["C1_corr"])
    return v, recorded


def main():
    v, recorded = collect()
    fig, axes = C.full(1, 4)
    for ax, (m, label) in zip(axes, PANELS):
        for i, (cond, col) in enumerate(CONDITIONS):
            vals = np.array([v[cond][m][a] for a in AGENTS], float)
            for k, (a, mk) in enumerate(zip(AGENTS, MARKERS)):
                if np.isfinite(vals[k]):
                    ax.scatter(i + (k - 1.5) * 0.09, vals[k], marker=mk, s=95, color=col, edgecolor="black",
                               linewidth=0.7, zorder=4)
            if np.isfinite(vals).any():
                ax.hlines(np.nanmean(vals), i - 0.3, i + 0.3, color="black", lw=2.6, zorder=5)
        if m in ("C1", "dir"):
            C.chance(ax, 0.4)
        ax.set_xticks(range(len(CONDITIONS)), [c for c, _ in CONDITIONS], rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_ylabel(label)
    axes[0].set_ylim(-2.4, 0.15)
    for ax in axes[1:]:
        ax.set_ylim(0, 1.05)
    print("E7h recorded settings:", recorded)
    for cond, _ in CONDITIONS:
        print(cond, {m: round(float(np.nanmean([v[cond][m][a] for a in AGENTS])), 3)
                     if np.isfinite([v[cond][m][a] for a in AGENTS]).any() else None for m, _ in PANELS},
              {m: [round(float(v[cond][m][a]), 2) for a in AGENTS] for m, _ in PANELS})
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
