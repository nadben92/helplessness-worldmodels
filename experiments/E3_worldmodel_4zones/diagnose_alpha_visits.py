"""Exploratory, read-only complementary test of docs/e3/DECISIONS.md, 2026-10-14 (35): in CD, alpha (the action-independent
weight) and the prior's mass on the true successor against n(s), the visits of the state's least-visited action, for the
8 campaign runs at every measured checkpoint; is one run (default: seed 1 environment) on the other runs' relation?

Usage: python experiments/E3_worldmodel_4zones/diagnose_alpha_visits.py [--work results/E3_worldmodel_4zones/campaign] [--seed 1]
Outputs: <work>/diagnostic_seed<NN>_cd/alpha_visits.json and alpha_visits.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import action_independent_weight

ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "diag_av_analysis")
cf = load_experiment_module("E3_worldmodel_4zones", "campaign_figures.py", "diag_av_figures")
dsc = load_experiment_module("E3_worldmodel_4zones", "diagnose_seed_cd.py", "diag_av_seed_cd")
CD, CN = "controllable_deterministic", "controllable_noisy"
MIN_POINTS, MIN_OTHER_RUNS = 3, 4  # (35)


def points(run) -> dict:
    """Per (CD state, measured checkpoint): n(s), alpha(s), mass on the true successor."""
    rows = {"c": [], "n": [], "alpha": [], "mass": []}
    m = ca.zone_mask(run, CD)
    for c in sorted(run.measure_dirs):
        arr = run.measure(c)[1]
        a = action_independent_weight(arr["real_p_hat"], arr["P_true"])
        mass = dsc.successor_mass_per_state(arr["real_p_hat"], arr["P_true"])
        n = ca.min_action_visits(run.counts[f"N_{c}"])
        for s in np.flatnonzero(m):
            rows["c"].append(c); rows["n"].append(int(n[s])); rows["alpha"].append(float(a[s])); rows["mass"].append(float(mass[s]))
    return {k: np.asarray(v) for k, v in rows.items()}


def per_state_rule(pts: dict, me: str, quantity: str) -> dict:
    """(35), per state: bins of log2 n; the 7 other runs' [min, max] of their bin means; seed's bin mean against it."""
    bins = {}
    for key, p in pts.items():
        ok = (p["n"] >= 1) & np.isfinite(p[quantity])
        k = np.floor(np.log2(p["n"][ok])).astype(int)
        for b in np.unique(k):
            sel = p[quantity][ok][k == b]
            bins.setdefault(int(b), {})[key] = (float(sel.mean()), int(sel.size))
    table, qualifying = [], []
    for b in sorted(bins):
        d = bins[b]
        others = [v[0] for key, v in d.items() if key != me and v[1] >= MIN_POINTS]
        mine = d.get(me)
        row = {"n_range": [2 ** b, 2 ** (b + 1) - 1], "seed_mean": mine[0] if mine else None, "seed_points": mine[1] if mine else 0,
               "others_min": min(others) if others else None, "others_max": max(others) if others else None, "others_runs": len(others)}
        row["qualifies"] = bool(mine and mine[1] >= MIN_POINTS and len(others) >= MIN_OTHER_RUNS)
        if row["qualifies"]:
            row["position"] = "above" if mine[0] > row["others_max"] else "below" if mine[0] < row["others_min"] else "within"
            qualifying.append(row)
        table.append(row)
    return {"bins": table, **verdict([r["position"] for r in qualifying])}


def zone_rule(pts: dict, me: str, quantity: str) -> dict:
    """(35), zone means: per run and checkpoint, (mean log10 n, zone mean); the others interpolated at seed's mean log10 n."""
    curves = {}
    for key, p in pts.items():
        cs = np.unique(p["c"])
        x = np.array([np.log10(np.maximum(p["n"][p["c"] == c], 1)).mean() for c in cs])
        y = np.array([np.nanmean(p[quantity][p["c"] == c]) for c in cs])
        o = np.argsort(x)
        curves[key] = (x[o], y[o], cs[o])
    xs, ys, cs = curves[me]
    rows = []
    for x, y, c in zip(xs, ys, cs):
        vals = [float(np.interp(x, cx, cy)) for key, (cx, cy, _) in curves.items() if key != me and cx[0] <= x <= cx[-1]]
        row = {"checkpoint": int(c), "mean_log10_n": float(x), "seed": float(y), "others": len(vals)}
        if len(vals) >= MIN_OTHER_RUNS:
            row.update(others_min=min(vals), others_max=max(vals),
                       position="above" if y > max(vals) else "below" if y < min(vals) else "within")
        rows.append(row)
    return {"points": rows, **verdict([r["position"] for r in rows if "position" in r]), "curves": {
        k: {"mean_log10_n": v[0].tolist(), "value": v[1].tolist(), "checkpoints": v[2].tolist()} for k, v in curves.items()}}


def verdict(positions: list[str]) -> dict:
    n = len(positions)
    above = positions.count("above")
    if n == 0:
        v = "undetermined (no qualifying point)"
    elif all(p == "within" for p in positions):
        v = "learning delay due to the visits"
    elif above >= max(2, (n + 1) // 2):
        v = "effect proper to this model"
    else:
        v = "undetermined"
    return {"qualifying": n, "above": above, "within": positions.count("within"), "below": positions.count("below"), "verdict": v}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("results/E3_worldmodel_4zones/campaign"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    paths = sorted(p for p in (args.work / "runs").iterdir() if (p / "results.json").exists())
    runs = {f"seed{r.seed}_{r.condition}": r for r in (ca.Run(p) for p in paths)}
    me = f"seed{args.seed}_experimental"
    pts = {k: points(r) for k, r in runs.items()}
    out = {"exploratory": "DECISIONS (35)", "run": me,
           "alpha": {"per_state": per_state_rule(pts, me, "alpha"), "zone": zone_rule(pts, me, "alpha")},
           "mass_true_successor_no_rule": {"per_state": per_state_rule(pts, me, "mass"), "zone": zone_rule(pts, me, "mass")},
           "early_visit_shares_25119": {}}
    for k, r in runs.items():
        N = r.counts["N_25119"]
        out["early_visit_shares_25119"][k] = {z: float(N[ca.zone_mask(r, z)].sum() / N.sum()) for z in (CD, CN)}
    d = args.work / f"diagnostic_seed{args.seed:02d}_cd"
    d.mkdir(exist_ok=True)
    (d / "alpha_visits.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    figure(pts, out, me, d)
    print(json.dumps({q: {lvl: out[q][lvl]["verdict"] for lvl in ("per_state", "zone")} for q in ("alpha", "mass_true_successor_no_rule")}, indent=1))
    return 0


def figure(pts: dict, out: dict, me: str, d: Path) -> None:
    plt = cf._plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (q, qkey, label) in enumerate((("alpha", "alpha", "alpha (action-independent weight)"), ("mass", "mass_true_successor_no_rule", "mass on the true successor"))):
        ax = axes[0, col]
        for key, p in pts.items():
            mine = key == me
            ref1 = key == me.replace("experimental", "reference")
            color = "#e34948" if mine else "#2a78d6" if ref1 else cf.MUTED
            ok = p["n"] >= 1
            ax.scatter(p["n"][ok], p[q][ok], s=10 if (mine or ref1) else 4, c=color, alpha=0.8 if (mine or ref1) else 0.35,
                       label=key if (mine or ref1) else None, edgecolors="none")
        ax.set_xscale("log"); ax.set_xlabel("n(s): visits of the least-visited action"); ax.set_ylabel(label)
        ax.set_title(f"CD, per state and checkpoint (grey: the other 6 runs)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, col]
        for key, cv in out[qkey]["zone"]["curves"].items():
            mine = key == me
            ref1 = key == me.replace("experimental", "reference")
            color = "#e34948" if mine else "#2a78d6" if ref1 else cf.MUTED
            ax.plot(10 ** np.array(cv["mean_log10_n"]), cv["value"], color=color, lw=2 if (mine or ref1) else 0.9, marker="o", ms=4 if (mine or ref1) else 2,
                    label=key if (mine or ref1) else None)
        ax.set_xscale("log"); ax.set_xlabel("zone mean of n(s) (geometric)"); ax.set_ylabel(label)
        ax.set_title(f"CD zone mean per checkpoint ({'rule' if q == 'alpha' else 'no rule'}: {out[qkey]['zone']['verdict']})", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_title(f"CD, per state and checkpoint (rule: {out['alpha']['per_state']['verdict']})", fontsize=9)
    fig.suptitle("alpha and mass on the true successor against the visits, CD (exploratory; DECISIONS (35))", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "alpha_visits.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
