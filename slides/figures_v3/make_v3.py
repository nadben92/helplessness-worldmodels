"""Slides v3: E7_bars_critic (E7_bars + a critic-controllability panel) and E7_curve_b_vs_critic (directed share against
b and against the critic's controllability C1 of D7, one point per (agent, condition)). Same style as v2 (default
matplotlib, no title / legend / annotation). Reads results/ read-only; nothing is recomputed. The two small C1 tables
extracted for v2 (slides branch, data_v2/) are copied to slides/out_v3/data/. Run from the repo root:
  python slides/figures_v3/make_v3.py"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "slides" / "out_v3"
EXPOSED, REFERENCE, ORACLE, ANTI, CONSTR, FREE, INDUC = ("#E8412C", "#1F6FEB", "#1DB954", "#FF9F1C", "#6718AB",
                                                          "#B98CE8", "#FF2E88")
AGENTS = ["p1-a-d1_e4", "p1-b-d1_e4", "p2-a-d1_e4", "p2-b-d1_e4"]
R = REPO / "results"
E6 = R / "E6_phenotypes_latent/dev_e4_cpu_v2/analysis/points.csv"
V1 = R / "E7_latent_manipulation/dev_e4_cpu/analysis/checks.csv"
D4 = R / "E7_latent_manipulation/dev_e4_cpu_diag/analysis/d4_episodes.csv"
D7 = R / "E7_dissociation/d7"
T1 = R / "E7_dissociation/dev_e4_cpu/analysis/t1_runs.csv"
C1_NAT, C1_IND = OUT / "data/e7h_c1_natural.csv", OUT / "data/e7d_induction_runs.csv"
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


def d7_summaries():
    recorded = {r["agent"]: r["setting"] for r in rd(D7 / "analysis/d7_identity.csv")
                if r["phase"] == "corrected" and r["recorded"] == "True"}
    corr = {(r["agent"], r["setting"]): r for r in rd(D7 / "analysis/d7_correction.csv") if r["zbar"] == "draw"}
    nat, con = {}, {}
    for d in sorted((D7 / "runs").iterdir()):
        rows = json.loads((d / "summary.json").read_text(encoding="utf-8"))["rows"]
        if "__natural__" in d.name:
            for r in rows:
                nat[(r["agent"], r["model"], r["mode"], r["alpha"])] = r
        elif "__corrected__" in d.name and recorded.get(rows[0]["agent"]) == rows[0]["setting"]:
            r = rows[0]
            con[r["agent"]] = (r, num(corr[(r["agent"], r["setting"])]["C1_corr"]))
    return nat, con


def collect():
    """v[condition][measure][agent]: b (step 1), C1, loc (zone), dir; NaN when missing."""
    conds = ["exposed", "unconstrained", "constrained", "oracle", "reference"]
    v = {c: {m: {a: NAN for a in AGENTS} for m in ("b", "C1", "loc", "dir")} for c in conds}
    for r in rd(E6):
        if r["image"] == "training":
            for m, k in (("b", "b_step1"), ("loc", "loc_template_zone"), ("dir", "directed_share")):
                v[r["role"]][m][r["agent"]] = num(r[k])
    for r in rd(C1_NAT):
        v[r["role"]]["C1"][r["agent"]] = num(r["C1"])
    for r in rd(V1):
        if r["image"] == "training" and r["mode"] == "stepwise" and num(r["alpha"]) == 1.0:
            for m, k in (("b", "b_step1"), ("loc", "loc_template_zone"), ("dir", "directed_share")):
                v["unconstrained"][m][r["agent"]] = num(r[k])
    _, con = d7_summaries()
    for a, (r, c1) in con.items():
        v["constrained"]["b"][a], v["constrained"]["loc"][a] = r["b_step1"], r["loc_template_zone"]
        v["constrained"]["dir"][a], v["constrained"]["C1"][a] = r["directed_share"], c1
    od = {r["agent"]: num(r["directed_share"]) for r in rd(D4) if r["model"] == "exposed" and r["k"] == "1"}
    v["oracle"]["b"] = dict(v["reference"]["b"])
    v["oracle"]["loc"] = dict(v["exposed"]["loc"])
    v["oracle"]["dir"] = {a: od[a] for a in AGENTS}
    return v


def fig1(v):
    cats = [("exposed", "exposed", EXPOSED), ("latent fix", "unconstrained", FREE),
            ("constrained fix", "constrained", CONSTR), ("oracle", "oracle", ORACLE), ("reference", "reference", REFERENCE)]
    panels = [("b", "b (bits)"), ("C1", "critic controllability"), ("loc", "localisation"), ("dir", "directed share")]
    fig, axes = plt.subplots(1, 4, figsize=(13.33, 7.5))
    for ax, (m, lab) in zip(axes, panels):
        for i, (_, c, col) in enumerate(cats):
            vals = np.array([v[c][m][a] for a in AGENTS], float)
            ok = np.isfinite(vals)
            if not ok.any():
                continue
            ax.bar(i, vals[ok].mean(), width=0.75, color=col, edgecolor="black", lw=1)
            ax.scatter(i + np.linspace(-0.18, 0.18, 4)[ok], vals[ok], s=28, color="black", zorder=4)
        if m == "dir":
            ax.axhline(0.4, color="#888888", ls="--", lw=2.2, zorder=1)
        ax.set_xticks(range(len(cats)), [c for c, _, _ in cats], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_ylabel(lab)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_ylim(-2.35, 0.15)
    for ax in axes[1:]:
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "E7_bars_critic.png", dpi=300, facecolor="white")
    plt.close(fig)


def fig2(v):
    nat, _ = d7_summaries()
    P = []  # (condition, agent, b, C1, directed share)
    for c in ("exposed", "reference"):
        for a in AGENTS:
            P.append((c, a, v[c]["b"][a], v[c]["C1"][a], v[c]["dir"][a]))
    for a in AGENTS:
        P.append(("oracle", a, v["oracle"]["b"][a], NAN, v["oracle"]["dir"][a]))
        P.append(("free", a, v["unconstrained"]["b"][a], NAN, v["unconstrained"]["dir"][a]))
        P.append(("constrained", a, v["constrained"]["b"][a], v["constrained"]["C1"][a], v["constrained"]["dir"][a]))
    for r in rd(T1):
        if r["model"] == "reference" and r["h_mode"] == "mean" and r["k"] == "1":
            P.append(("anti", r["agent"], num(r["b_eff"]), NAN, num(r["directed_share"])))
    for r in rd(C1_IND):
        if r["mode"] == "reverse":
            s = nat[(r["agent"], "reference", "reverse", float(r["alpha"]))]
            P.append(("induction", r["agent"], s["b_step1"], num(r["C1"]), s["directed_share"]))
    col = {"exposed": EXPOSED, "reference": REFERENCE, "oracle": ORACLE, "anti": ANTI, "free": FREE,
           "constrained": CONSTR, "induction": INDUC}
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharey=True)
    for ax, j, lab in ((axes[0], 2, "b (bits)"), (axes[1], 3, "critic controllability")):
        for c in col:
            pts = [(p[j], p[4]) for p in P if p[0] == c and np.isfinite(p[j]) and np.isfinite(p[4])]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, s=95, color=col[c], edgecolor="black", linewidth=0.7, zorder=4)
        ax.set_xlabel(lab)
    axes[0].set_ylabel("directed share")
    axes[0].set_ylim(0.2, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / "E7_curve_b_vs_critic.png", dpi=300, facecolor="white")
    plt.close(fig)
    both = [p for p in P if np.isfinite(p[3]) and np.isfinite(p[4]) and np.isfinite(p[2])]
    out = {}
    for name, sel, j in (("left_all", [p for p in P if np.isfinite(p[2]) and np.isfinite(p[4])], 2),
                         ("right_all", both, 3), ("left_same_points_as_right", both, 2)):
        out[name] = (len(sel), float(spearmanr([p[j] for p in sel], [p[4] for p in sel])[0]))
    cnt = {c: [sum(1 for p in P if p[0] == c and np.isfinite(p[j]) and np.isfinite(p[4])) for j in (2, 3)] for c in col}
    return out, cnt


if __name__ == "__main__":
    v = collect()
    fig1(v)
    rho, cnt = fig2(v)
    for c in ("exposed", "unconstrained", "constrained", "oracle", "reference"):
        print(c, {m: [round(float(x), 2) for x in v[c][m].values()] for m in ("b", "C1", "dir")})
    print(rho)
    print(cnt)
