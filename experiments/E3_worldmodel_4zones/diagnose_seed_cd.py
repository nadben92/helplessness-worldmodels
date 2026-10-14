"""Exploratory, post hoc diagnostic of one seed's CD zone (docs/e3/DECISIONS.md, 2026-10-14 (34)), read only on the
campaign's extracted runs (campaign_report.py's <work>/runs) and RSA outputs (<work>/rsa).

Reported: the visits of CN per seed at the early checkpoints (the reading correction of (34)); alpha in CD per checkpoint
(the seed's environment, the other seeds' environments, the seed's reference); per CD state at the budget: I_s, I_true,
the prior's mass on the true successor (mean over the state's deterministic actions), n(s), delta(s); Spearman rho of the
errors with delta and n; CD's learning curve; the seed's RSA confusion ratios; and the verdict of (34)'s rule.

Usage: python experiments/E3_worldmodel_4zones/diagnose_seed_cd.py [--work results/E3_worldmodel_4zones/campaign] [--seed 1]
Outputs: <work>/diagnostic_seed<NN>_cd/ (diagnostic.json, DIAGNOSTIC.md, figures).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import action_independent_weight

ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "diag_cd_analysis")
cf = load_experiment_module("E3_worldmodel_4zones", "campaign_figures.py", "diag_cd_figures")
CD, CN = "controllable_deterministic", "controllable_noisy"
BUDGET = ca.BUDGET_ENV_STEPS
# (34): thresholds fixed before computing
ALPHA_MARGIN, RHO_BOUNDARY, RHO_VISITS = 0.05, -0.3, -0.3


def successor_mass_per_state(p_hat: np.ndarray, P: np.ndarray) -> np.ndarray:
    det = P.max(-1) >= 1.0 - 1e-12
    mass = np.take_along_axis(p_hat, P.argmax(-1)[..., None], axis=-1)[..., 0]
    out = np.full(P.shape[0], np.nan)
    for s in range(P.shape[0]):
        if det[s].any():
            out[s] = float(np.nanmean(mass[s][det[s]]))
    return out


def alpha_cd(run, c: int) -> float:
    arr = run.measure(c)[1]
    a = action_independent_weight(arr["real_p_hat"], arr["P_true"])
    return float(np.nanmean(a[ca.zone_mask(run, CD)]))


def verdict(alpha_gap: float, rho_delta: float, rho_n: float) -> str:
    if alpha_gap > ALPHA_MARGIN and rho_delta <= RHO_BOUNDARY:
        return "helplessness signature"
    if alpha_gap <= ALPHA_MARGIN and (abs(rho_delta) < abs(RHO_BOUNDARY) or rho_n <= RHO_VISITS):
        return "failed-learning signature"
    return "undetermined"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path("results/E3_worldmodel_4zones/campaign"))
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    paths = sorted(p for p in (args.work / "runs").iterdir() if (p / "results.json").exists())
    runs = {(r.seed, r.condition): r for r in (ca.Run(p) for p in paths)}
    seeds = sorted({s for s, _ in runs})
    out_dir = args.work / f"diagnostic_seed{args.seed:02d}_cd"
    out_dir.mkdir(exist_ok=True)
    out: dict = {"seed": args.seed, "exploratory_post_hoc": "DECISIONS (34)"}

    # CN visits at the early checkpoints
    out["cn_visits"] = {}
    for c in (25_119, 39_811):
        out["cn_visits"][c] = {}
        for s in seeds:
            for cond in ("experimental", "reference"):
                r = runs[(s, cond)]
                N = r.counts[f"N_{c}"]
                m = ca.zone_mask(r, CN)
                out["cn_visits"][c][f"seed{s}_{cond}"] = {"visits": int(N[m].sum()), "share": float(N[m].sum() / N.sum()),
                                                          "mean_min_action_visits": float(ca.min_action_visits(N)[m].mean())}

    # alpha in CD per checkpoint
    out["alpha_cd"] = {f"seed{s}_{cond}": {c: alpha_cd(runs[(s, cond)], c) for c in sorted(runs[(s, cond)].measure_dirs)}
                       for s in seeds for cond in ("experimental", "reference")}

    # per CD state at the budget, every run
    per_state, rhos = {}, {}
    for (s, cond), r in runs.items():
        res, arr = r.measure(BUDGET)
        m = ca.zone_mask(r, CD)
        idx = np.flatnonzero(m)
        delta = ca.distance_to_uncontrollable(runs[(s, "experimental")])
        I, I_true = arr["real_I_s"], r.truth["I_true"]
        mass = successor_mass_per_state(arr["real_p_hat"], arr["P_true"])
        n = ca.min_action_visits(r.counts[f"N_{BUDGET}"])
        err, merr = np.abs(I - I_true), 1.0 - mass
        key = f"seed{s}_{cond}"
        per_state[key] = {"states": idx.tolist(), "I_s": I[idx].tolist(), "I_true": I_true[idx].tolist(), "mass_true_successor": mass[idx].tolist(),
                          "n": n[idx].tolist(), "delta": delta[idx].tolist()}
        rhos[key] = {"rho_error_delta": ca._spearman(delta[idx], err[idx]), "rho_error_n": ca._spearman(n[idx].astype(float), err[idx]),
                     "rho_mass_error_delta": ca._spearman(delta[idx], merr[idx]), "rho_mass_error_n": ca._spearman(n[idx].astype(float), merr[idx]),
                     "mean_error": float(err[idx].mean()), "mean_mass_error": float(merr[idx].mean()),
                     "mean_error_by_delta": {int(d): float(err[idx][delta[idx] == d].mean()) for d in np.unique(delta[idx])}}
    out["per_state_cd_budget"], out["rho"] = per_state, rhos

    # the rule
    me = f"seed{args.seed}_experimental"
    others = [out["alpha_cd"][f"seed{s}_experimental"][BUDGET] for s in seeds if s != args.seed] + [out["alpha_cd"][f"seed{args.seed}_reference"][BUDGET]]
    gap = out["alpha_cd"][me][BUDGET] - max(others)
    out["rule"] = {"alpha_seed_env": out["alpha_cd"][me][BUDGET], "alpha_largest_comparison": max(others), "alpha_gap": gap,
                   "rho_error_delta": rhos[me]["rho_error_delta"], "rho_error_n": rhos[me]["rho_error_n"],
                   "verdict": verdict(gap, rhos[me]["rho_error_delta"], rhos[me]["rho_error_n"])}

    # learning curve of CD (every measured checkpoint): mean mass on the true successor, I_s - I_true, C-b's L1 of the model
    out["learning_cd"] = {}
    for (s, cond), r in runs.items():
        rows = {}
        for c in sorted(r.measure_dirs):
            arr = r.measure(c)[1]
            m = ca.zone_mask(r, CD)
            mass = successor_mass_per_state(arr["real_p_hat"], arr["P_true"])
            l1 = np.abs(arr["real_p_hat"] - arr["P_true"]).sum(-1)
            rows[c] = {"mass_true_successor": float(np.nanmean(mass[m])), "I_s_minus_true": float(np.nanmean(arr["real_I_s"][m] - r.truth["I_true"][m])),
                       "l1_model": float(np.nanmean(l1[m])), "mean_min_action_visits": float(ca.min_action_visits(r.counts[f"N_{c}"])[m].mean())}
        out["learning_cd"][f"seed{s}_{cond}"] = rows

    # the seed's RSA confusion ratios
    out["rsa_ratios"] = {}
    for cond in ("experimental", "reference"):
        d = args.work / "rsa" / runs[(args.seed, cond)].path.name
        out["rsa_ratios"][cond] = {int(f.stem[3:]): {lat: json.loads(f.read_text())["states"]["latents"][lat]["ratios"] for lat in ("h", "z")}
                                   for f in sorted(d.glob("env*.json"))}

    (out_dir / "diagnostic.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    figures(out, runs, args.seed, out_dir)
    print(json.dumps(out["rule"], indent=1))
    return 0


def figures(out: dict, runs: dict, seed: int, out_dir: Path) -> None:
    plt = cf._plt()
    seq, div = cf._cmaps()
    # maps of the CD quadrant: I_s and mass on the true successor, seed env / seed ref / env - ref (the other seeds' CD
    # lies elsewhere on the torus -- the layout depends on the seed -- so they appear in the error plots only)
    zone = runs[(seed, "experimental")].zone_of_state
    others = [k for k in out["per_state_cd_budget"] if k.endswith("experimental") and k != f"seed{seed}_experimental"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for row, (key, label, cmap, lim) in enumerate((("I_s", "I_s (bit)", seq, (0, 2.4)), ("mass_true_successor", "mass on true successor", seq, (0, 1)))):
        panels = [(f"seed {seed} environment", out["per_state_cd_budget"][f"seed{seed}_experimental"]),
                  (f"seed {seed} reference", out["per_state_cd_budget"][f"seed{seed}_reference"]),
                  ("environment - reference", None)]
        for col, (title, d) in enumerate(panels):
            grid = np.full(144, np.nan)
            if d is None:
                e, r = out["per_state_cd_budget"][f"seed{seed}_experimental"], out["per_state_cd_budget"][f"seed{seed}_reference"]
                assert e["states"] == r["states"]
                grid[e["states"]] = np.array(e[key]) - np.array(r[key])
                dl = float(np.nanmax(np.abs(grid))) or 1e-6
                cm, vlim = div, (-dl, dl)
            else:
                grid[d["states"]] = d[key]
                cm, vlim = cmap, lim
            ax = axes[row, col]
            im = ax.imshow(grid.reshape(cf.GRID), cmap=cm, vmin=vlim[0], vmax=vlim[1])
            cf._zone_outline(ax, zone)
            ax.set_title(f"{title}: {label}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"CD zone at {BUDGET:,} (exploratory, post hoc; DECISIONS (34))", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "cd_maps.png", dpi=130)
    plt.close(fig)
    # errors against distance and visits
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, (_, color, marker) in ((f"seed{seed}_experimental", ("", "#e34948", "o")), (f"seed{seed}_reference", ("", "#2a78d6", "s"))):
        d = out["per_state_cd_budget"][key]
        err = np.abs(np.array(d["I_s"]) - np.array(d["I_true"]))
        axes[0].scatter(d["delta"], err, c=color, marker=marker, s=26, label=key, edgecolors=cf.SURFACE)
        axes[1].scatter(d["n"], err, c=color, marker=marker, s=26, label=key, edgecolors=cf.SURFACE)
    for k in others:
        d = out["per_state_cd_budget"][k]
        err = np.abs(np.array(d["I_s"]) - np.array(d["I_true"]))
        axes[0].scatter(d["delta"], err, c=cf.MUTED, s=10, alpha=0.6)
        axes[1].scatter(d["n"], err, c=cf.MUTED, s=10, alpha=0.6)
    axes[0].set_xlabel("distance to the nearest uncontrollable zone"); axes[1].set_xlabel("visits of the least-visited action n(s)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("|I_s - I_true| (bit)")
    axes[0].legend(frameon=False, fontsize=8, title="grey: other seeds' environments", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "cd_errors.png", dpi=130)
    plt.close(fig)
    # learning curve of CD and alpha
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, rows in out["learning_cd"].items():
        s, cond = key.split("_")
        color = "#e34948" if key == f"seed{seed}_experimental" else "#2a78d6" if key == f"seed{seed}_reference" else cf.MUTED
        lw = 2 if color != cf.MUTED else 0.8
        ls = "-" if cond == "experimental" else "--"
        cs = sorted(rows)
        axes[0].plot(cs, [rows[c]["mass_true_successor"] for c in cs], color=color, lw=lw, ls=ls, marker="o", ms=3, label=key)
        axes[1].plot(cs, [rows[c]["I_s_minus_true"] for c in cs], color=color, lw=lw, ls=ls, marker="o", ms=3)
        a = out["alpha_cd"][key]
        axes[2].plot(sorted(a), [a[c] for c in sorted(a)], color=color, lw=lw, ls=ls, marker="o", ms=3)
    for ax, t in zip(axes, ("CD: mass on the true successor", "CD: I_s - I_true (bit)", "CD: alpha (action-independent weight)")):
        ax.set_xscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("environment steps")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "cd_learning.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
