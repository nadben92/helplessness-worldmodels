"""Exploratory, read-only test of docs/e3/DECISIONS.md, 2026-10-14 (36): in CD, alpha per state regressed on log10 n(s) and
on the mean environment step of the state's visits; is seed 1 environment's gap explained by the timing of its visits?

Usage: python experiments/E3_worldmodel_4zones/diagnose_visit_timing.py [--work results/E3_worldmodel_4zones/campaign] [--seed 1]
Outputs: <work>/diagnostic_seed<NN>_cd/visit_timing.json and visit_timing.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import action_independent_weight

ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "diag_vt_analysis")
cf = load_experiment_module("E3_worldmodel_4zones", "campaign_figures.py", "diag_vt_figures")
CD = "controllable_deterministic"
N_BOOT, BOOT_SEED, T_UNIT = 2000, 20261014, 100_000.0  # (36)


def visit_times(run) -> tuple[np.ndarray, np.ndarray, int]:
    """For every transition of the replay: its state and its environment step (1-based); and the replay's last step."""
    d = np.load(run.path / "replay_final.npz")
    s, first = d["s"].astype(int), d["is_first"].astype(bool)
    has_next = np.zeros(len(s), dtype=bool)
    has_next[:-1] = ~first[1:]  # a transition out of observation i exists when i + 1 continues its episode
    states = s[has_next]
    steps = np.arange(1, states.size + 1)
    return states, steps, int(steps[-1]) if steps.size else 0


def points(run) -> dict:
    states, steps, last = visit_times(run)
    a_rows = []
    m = ca.zone_mask(run, CD)
    for c in sorted(run.measure_dirs):
        if c > last + 1:  # the pilot's 300,000 point: its replay stops at 290,000 (left out, (36))
            continue
        arr = run.measure(c)[1]
        alpha = action_independent_weight(arr["real_p_hat"], arr["P_true"])
        n = ca.min_action_visits(run.counts[f"N_{c}"])
        upto = steps <= c
        for s in np.flatnonzero(m):
            sel = upto & (states == s)
            if n[s] >= 1 and sel.any() and np.isfinite(alpha[s]):
                a_rows.append((s, c, np.log10(n[s]), steps[sel].mean() / T_UNIT, alpha[s]))
    return np.array(a_rows, dtype=float)


def fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("results/E3_worldmodel_4zones/campaign"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    paths = sorted(p for p in (args.work / "runs").iterdir() if (p / "results.json").exists())
    runs = {f"seed{r.seed}_{r.condition}": r for r in (ca.Run(p) for p in paths)}
    me = f"seed{args.seed}_experimental"
    keys = sorted(runs)
    data = {k: points(runs[k]) for k in keys}
    allp = np.vstack([np.column_stack([np.full(len(v), i), v]) for i, (k, v) in enumerate(data.items())])
    run_id, state, logn, t, alpha = allp[:, 0].astype(int), allp[:, 1].astype(int), allp[:, 3], allp[:, 4], allp[:, 5]
    XA = np.column_stack([np.ones(len(alpha)), logn])
    XB = np.column_stack([np.ones(len(alpha)), logn, t])
    bA, bB = fit(XA, alpha), fit(XB, alpha)
    out = {"exploratory": "DECISIONS (36)", "n_points": int(len(alpha)), "points_per_run": {k: int(len(v)) for k, v in data.items()},
           "model_A": {"b0": bA[0], "b1_log10n": bA[1]}, "model_B": {"b0": bB[0], "b1_log10n": bB[1], "b2_timing_per_100k": bB[2]}}
    # cluster bootstrap over (run, state)
    clusters = run_id * 1000 + state
    uniq = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([index[c] for c in rng.choice(uniq, size=uniq.size, replace=True)])
        boots.append(fit(XB[idx], alpha[idx]))
    boots = np.array(boots)
    out["model_B"]["ci95"] = {name: np.percentile(boots[:, j], [2.5, 97.5]).tolist() for j, name in enumerate(("b0", "b1_log10n", "b2_timing_per_100k"))}
    out["mean_timing_per_run"] = {k: float(np.mean(v[:, 2])) for k, v in data.items()}
    for name, X, b in (("A", XA, bA), ("B", XB, bB)):
        res = alpha - X @ b
        gaps = {k: float(res[run_id == i].mean()) for i, k in enumerate(keys)}
        others = [v for k, v in gaps.items() if k != me]
        out[f"model_{name}"]["residual_by_run"] = gaps
        out[f"model_{name}"]["seed_residual"] = gaps[me]
        out[f"model_{name}"]["others_range"] = [min(others), max(others)]
    b2, lo = out["model_B"]["b2_timing_per_100k"], out["model_B"]["ci95"]["b2_timing_per_100k"][0]
    inside = out["model_B"]["seed_residual"] <= out["model_B"]["others_range"][1]
    if b2 > 0 and lo > 0 and inside:
        v = "gap attributed to the timing of the visits"
    elif not inside:
        v = "gap persists: proper to the model, no explanation identified"
    else:
        v = "undetermined"
    out["verdict"] = v
    d = args.work / f"diagnostic_seed{args.seed:02d}_cd"
    d.mkdir(exist_ok=True)
    (d / "visit_timing.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    figure(data, keys, me, out, d)
    print(json.dumps({k: out[k] for k in ("model_A", "model_B", "verdict")}, indent=1, default=float))
    return 0


def figure(data: dict, keys: list[str], me: str, out: dict, d: Path) -> None:
    plt = cf._plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ref1 = me.replace("experimental", "reference")
    for k in keys:
        v = data[k]
        color = "#e34948" if k == me else "#2a78d6" if k == ref1 else cf.MUTED
        big = k in (me, ref1)
        axes[0].scatter(10 ** v[:, 2], v[:, 4], s=10 if big else 4, c=color, alpha=0.8 if big else 0.35, edgecolors="none", label=k if big else None)
        axes[0].set_xscale("log")
    axes[0].set_xlabel("n(s)"); axes[0].set_ylabel("alpha"); axes[0].set_title("CD: alpha against visits (grey: other runs)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    for k in keys:
        v = data[k]
        color = "#e34948" if k == me else "#2a78d6" if k == ref1 else cf.MUTED
        big = k in (me, ref1)
        axes[1].scatter(v[:, 3], v[:, 2], s=10 if big else 4, c=color, alpha=0.8 if big else 0.35, edgecolors="none")
    axes[1].set_xlabel("mean environment step of the visits (x 100,000)"); axes[1].set_ylabel("log10 n(s)")
    axes[1].set_title("CD: timing of the visits against their number", fontsize=9)
    x = np.arange(len(keys))
    for j, (m, off, color) in enumerate((("A", -0.18, "#eb6834"), ("B", 0.18, "#2a78d6"))):
        vals = [out[f"model_{m}"]["residual_by_run"][k] for k in keys]
        axes[2].bar(x + off, vals, width=0.34, color=color, label=f"model {m}" + (" (+ timing)" if m == "B" else " (log n only)"))
    axes[2].axhline(0, color="#c3c2b7", lw=1)
    axes[2].set_xticks(x, [k.replace("_experimental", " env").replace("_reference", " ref") for k in keys], rotation=45, fontsize=7)
    axes[2].set_ylabel("mean residual of alpha"); axes[2].set_title("mean residual of alpha by run", fontsize=9)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("CD: alpha against the number and the timing of the visits (exploratory; DECISIONS (36))", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "visit_timing.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
