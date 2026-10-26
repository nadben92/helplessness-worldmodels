"""Analysis of E7's D5 (run_e7_d5.py; docs/e6/DECISIONS.md (58)-(60)). Reads the run directories read-only.

  task 1  d5_steps.csv is not written (large); d5_cdir_bins.csv (per group, C_dir and bin of b: directed share, 95% interval
          by episodes, steps, episodes; bins < 30 steps masked), d5_cdir_share.csv (per group and bin: the share of steps
          with C_dir = 1, argmax and mass >= 0.5), d5_regression.csv (M0, M1, M2 on every step and on the common b range),
          d5_njuste.csv (n_juste's distribution at b ~ -1.5), d5_identity.csv (the reruns against E6);
          figures/d5_cdir.png.
  task 2  d5_sampled.csv (per agent, variant, p: b_eff, directed share); figures/d5_sampled.png (expectation k = 1 and 5,
          by draws main and shared, the exposed's natural steps, the well-localised induced references).
  forms   d5_forms.json.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/analyse_e7_d5.py --runs "<dir>/E7_dissociation/e7d5__*" --dis-runs "<dir>/E7_dissociation/e7dis__*"
      --e6-runs "<dir>/E6_phenotypes_latent/e6__*" --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyse_e7_diagnostics as ad  # noqa: E402
import analyse_e7_dissociation as adis  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.measures import decision_rule as dr  # noqa: E402
from helplessness.measures import zone_phenotypes as zp  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

EDGES, MIN_STEPS = adis.EDGES, adis.MIN_STEPS
GROUPS = ("exposed", "induced references")


# -------------------------------------------------------------------------------------------------------------- task 1
def collect_steps(runs: list[dict], worlds: dict, e6_runs: list[str]) -> tuple[dict, list[dict]]:
    """Every measured zone step of the kernels runs, with its group (None: a reference run whose localisation broke)."""
    cols = {k: [] for k in ("agent", "group", "episode", "b", "directed", "greedy", "cdir", "cdir_mass", "njuste", "localised", "mass_dir")}
    identity = []
    for run in runs:
        agent, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        rows = run["summary"]["rows"]
        base = next(r for r in rows if r["mode"] == "reverse_none")["loc_template_zone"]
        for r in rows:
            tag = f"{r['mode']}__{r['alpha']:g}"
            if r["model"] == "exposed":
                group = "exposed"
            else:
                group = "induced references" if r["loc_template_zone"] >= base - 0.05 else None
            st, ln = ep[f"{tag}__states"], ep[f"{tag}__length"]
            inz = zp.in_zone_steps(st, ln, w.in_zone) & np.isfinite(ep[f"{tag}__I"])
            i, t = np.nonzero(inz)
            s = st[i, t]
            adir = w.directed[s]
            just = ep[f"{tag}__argmax_true"][i, t] == 1
            mass = ep[f"{tag}__mass_true_all"][i, t]
            cols["agent"] += [agent] * len(i)
            cols["group"] += [group] * len(i)
            cols["episode"] += [f"{agent}:{tag}:{e}" for e in i]
            cols["b"].append(ep[f"{tag}__b_local"][i, t])
            cols["directed"].append(zp.directed_steps(st, ln, w.dist)[i, t].astype(float))
            cols["greedy"].append(dr.greedy_directed_share(ep[f"{tag}__Q"][i, t], adir, ad.TIE_TOL))
            cols["cdir"].append((adir & just).any(axis=1).astype(float))
            cols["cdir_mass"].append((adir & (np.nan_to_num(mass) >= 0.5)).any(axis=1).astype(float))
            cols["njuste"].append(just.sum(axis=1))
            cols["localised"].append(ep[f"{tag}__loc_template"][i, t] == 1)
            cols["mass_dir"].append(np.nanmax(np.where(adir, mass, np.nan), axis=1))
            if r["mode"] in ("none", "reverse_none"):
                e6 = [d for p in e6_runs for d in glob.glob(p) if Path(d).name.startswith(f"e6__{agent}__")]
                if e6:
                    ref = np.load(Path(sorted(e6)[-1]) / "episodes.npz")
                    role = "exposed" if r["model"] == "exposed" else "reference"
                    identity.append({"agent": agent, "model": role, **{f: bool(np.array_equal(ep[f"{tag}__{f}"], ref[f"training__{role}__{f}"], equal_nan=True))
                                                                      for f in ("states", "actions", "Q", "I", "loc_template", "mass_played")}})
    out = {k: (np.concatenate(v) if k in ("b", "directed", "greedy", "cdir", "cdir_mass", "njuste", "localised", "mass_dir") else np.asarray(v, dtype=object))
           for k, v in cols.items()}
    return out, identity


def select(S: dict, group: str, exposed_all: bool = False) -> np.ndarray:
    m = S["group"] == group
    return m if (exposed_all and group == "exposed") else m & S["localised"]


def binned_share(S: dict, mask: np.ndarray, y: np.ndarray, rng) -> list[dict]:
    which = np.digitize(S["b"], EDGES) - 1
    out = []
    for j in range(len(EDGES) - 1):
        sel = mask & (which == j)
        n = int(sel.sum())
        row = {"b_lo": EDGES[j], "n_steps": n, "n_episodes": int(len(np.unique(S["episode"][sel]))), "share": float(y[sel].mean()) if n else np.nan,
               "lo": np.nan, "hi": np.nan, "masked": n < MIN_STEPS}
        if n >= MIN_STEPS:
            eps, inv = np.unique(S["episode"][sel], return_inverse=True)
            grp = np.array([e.split(":")[0] for e in eps])
            row["lo"], row["hi"] = adis.boot_share(np.bincount(inv, weights=y[sel]), np.bincount(inv).astype(float), grp, rng)
        out.append(row)
    return out


def irls(X: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by iteratively reweighted least squares: coefficients and their standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-8 * np.eye(X.shape[1]))
    return beta, np.sqrt(np.diag(cov))


def regressions(S: dict, mask: np.ndarray, label: str) -> list[dict]:
    agents = sorted(set(S["agent"][mask]))
    b, y = S["b"][mask], S["directed"][mask]
    grp = (S["group"][mask] == "exposed").astype(float)
    cdir = S["cdir"][mask]
    dummies = [(S["agent"][mask] == a).astype(float) for a in agents[1:]]
    base = [np.ones_like(b), b, grp, *dummies]
    names = ["intercept", "b", "group_exposed", *[f"agent_{a}" for a in agents[1:]]]
    specs = {"M0": (base, names), "M1": (base + [cdir], names + ["C_dir"]),
             "M2": (base + [cdir, cdir * grp, b * grp], names + ["C_dir", "C_dir_x_exposed", "b_x_exposed"])}
    out = []
    for model, (cols, nm) in specs.items():
        X = np.column_stack(cols)
        beta, se = irls(X, y)
        X1, X0 = X.copy(), X.copy()
        gi = nm.index("group_exposed")
        X1[:, gi], X0[:, gi] = 1, 0
        if "C_dir_x_exposed" in nm:
            X1[:, nm.index("C_dir_x_exposed")], X0[:, nm.index("C_dir_x_exposed")] = cdir, 0
            X1[:, nm.index("b_x_exposed")], X0[:, nm.index("b_x_exposed")] = b, 0
        gap = float(np.mean(1 / (1 + np.exp(-(X1 @ beta))) - 1 / (1 + np.exp(-(X0 @ beta)))))
        for name, bb, ss in zip(nm, beta, se):
            out.append({"range": label, "model": model, "term": name, "coef": bb, "se": ss, "n": int(mask.sum()), "group_gap_prob": gap})
    return out


def fig_task1(c_bins: list[dict], share_bins: list[dict], figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"exposed": "tab:red", "induced references": "tab:blue"}
    for r_group in GROUPS:
        for c, ls in ((1, "-"), (0, "--")):
            rr = [r for r in c_bins if r["group"] == r_group and r["cdir"] == c and r["outcome"] == "directed" and not r["masked"]]
            x = [r["b_lo"] + 0.125 for r in rr]
            axes[0].errorbar(x, [r["share"] for r in rr], yerr=[[r["share"] - r["lo"] for r in rr], [r["hi"] - r["share"] for r in rr]], fmt="o" + ls,
                             color=style[r_group], capsize=2, label=f"{r_group}, C_dir = {c}")
        rr = [r for r in share_bins if r["group"] == r_group and r["rule"] == "argmax" and not r["masked"]]
        axes[1].plot([r["b_lo"] + 0.125 for r in rr], [r["share"] for r in rr], "o-", color=style[r_group], label=f"{r_group} (argmax)")
        rr = [r for r in share_bins if r["group"] == r_group and r["rule"] == "mass>=0.5" and not r["masked"]]
        axes[1].plot([r["b_lo"] + 0.125 for r in rr], [r["share"] for r in rr], "s:", color=style[r_group], label=f"{r_group} (masse ≥ 0,5)")
    axes[0].axhline(0.39, color="gray", lw=0.6, ls="--"), axes[0].set_ylabel("part dirigée (ε-greedy)"), axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("part des pas avec C_dir = 1"), axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel("b local (bit), pas bien localisés"), ax.legend(fontsize=7)
    fig.suptitle("D5 — prédiction juste pour une action dirigée (C_dir) ; références induites : runs à localisation tenue", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d5_cdir.png", dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------------- task 2
def task2(runs: list[dict], dis_runs: list[dict], S: dict, rng, figdir: Path) -> list[dict]:
    rows = []
    for run in runs:
        for r in run["summary"]["rows"]:
            rows.append({"agent": r["agent"], "variant": "by draws, shared" if r["shared"] else "by draws", "k": 1, "p": r["p"], "b_eff": r["b_eff"],
                         "directed_share": r["directed_share"], "loc_zone": r["loc_template_zone"]})
    for run in dis_runs:
        if run["meta"]["role"] != "reference" or run["meta"]["h_mode"] != "mean":
            continue
        for r in run["summary"]["rows"]:
            rows.append({"agent": r["agent"], "variant": "expectation (section 17)" if r["k"] == 1 else "particles k = 5 (section 17)", "k": r["k"],
                         "p": r["p"], "b_eff": r["b_eff"], "directed_share": r["directed_share"], "loc_zone": r["loc_template_zone"]})
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for variant, style in (("expectation (section 17)", "s--"), ("by draws", "^-"), ("by draws, shared", "v:"), ("particles k = 5 (section 17)", "d-.")):
        rr = [r for r in rows if r["variant"] == variant]
        ps = sorted({r["p"] for r in rr})
        ax.plot([np.mean([r["b_eff"] for r in rr if r["p"] == p]) for p in ps], [np.mean([r["directed_share"] for r in rr if r["p"] == p]) for p in ps],
                style, label=f"références + anti-oracle, {variant} (b effectif)")
    for group, color in (("exposed", "tab:red"), ("induced references", "tab:blue")):
        bins = binned_share(S, select(S, group), S["directed"], rng)
        rr = [r for r in bins if not r["masked"]]
        ax.errorbar([r["b_lo"] + 0.125 for r in rr], [r["share"] for r in rr], yerr=[[r["share"] - r["lo"] for r in rr], [r["hi"] - r["share"] for r in rr]],
                    fmt="o-", color=color, capsize=2, lw=2, label=f"{'exposés naturels' if group == 'exposed' else 'références induites bien localisées'} (b naturel)")
    ax.axhline(0.39, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("b (bit) — effectif (anti-oracle) ou naturel (épisodes) : pas la même mesure"), ax.set_ylabel("part dirigée (ε-greedy)")
    ax.set_ylim(0, 1.02), ax.legend(fontsize=6.5)
    ax.set_title("D5 — anti-oracle en espérance, par tirages, à particules ; exposés et références induites", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d5_sampled.png", dpi=130)
    plt.close(fig)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--dis-runs", nargs="+", required=True)
    parser.add_argument("--e6-runs", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    rng = np.random.default_rng(7118)
    runs, dis = adis.runs_of(args.runs), adis.runs_of(args.dis_runs)
    worlds = {r["meta"]["agent"]: ad.World(cfg, r["meta"]) for r in runs}
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    kern = [r for r in runs if r["meta"]["phase"] == "kernels"]
    S, identity = collect_steps(kern, worlds, args.e6_runs)
    ad.write_csv(args.out / "d5_identity.csv", identity)
    c_bins, share_bins = [], []
    for group in GROUPS:
        for variant, exposed_all in (("well-localised", False), ("exposed: all steps", True)):
            if exposed_all and group != "exposed":
                continue
            m = select(S, group, exposed_all)
            for c in (1, 0):
                for outcome in ("directed", "greedy"):
                    for row in binned_share(S, m & (S["cdir"] == c), S[outcome], rng):
                        c_bins.append({"group": group, "steps": variant, "cdir": c, "outcome": outcome, **row})
            for rule, key in (("argmax", "cdir"), ("mass>=0.5", "cdir_mass")):
                for row in binned_share(S, m, S[key], rng):
                    share_bins.append({"group": group, "steps": variant, "rule": rule, **row})
    ad.write_csv(args.out / "d5_cdir_bins.csv", c_bins)
    ad.write_csv(args.out / "d5_cdir_share.csv", share_bins)
    fig_task1([r for r in c_bins if r["steps"] == "well-localised"], [r for r in share_bins if r["steps"] == "well-localised"], figdir)
    both = select(S, "exposed") | select(S, "induced references")
    which = np.digitize(S["b"], EDGES) - 1
    common = [j for j in range(len(EDGES) - 1) if all((select(S, g) & (which == j)).sum() >= MIN_STEPS for g in GROUPS)]
    reg = regressions(S, both, "all steps")
    if common:
        reg += regressions(S, both & np.isin(which, common), f"common b range [{EDGES[min(common)]:g}, {EDGES[max(common) + 1]:g})")
    ad.write_csv(args.out / "d5_regression.csv", reg)
    nj = []
    for group in GROUPS:
        m = select(S, group) & (S["b"] >= -1.75) & (S["b"] < -1.25)
        counts = np.bincount(S["njuste"][m].astype(int), minlength=6)
        nj.append({"group": group, "n_steps": int(m.sum()), **{f"n_juste={k}": float(counts[k] / max(m.sum(), 1)) for k in range(6)},
                   "mean_mass_best_directed": float(np.nanmean(S["mass_dir"][m])) if m.any() else np.nan})
    ad.write_csv(args.out / "d5_njuste.csv", nj)
    t2 = task2([r for r in runs if r["meta"]["phase"] == "sampled"], dis, S, rng, figdir)
    ad.write_csv(args.out / "d5_sampled.csv", t2)
    forms = {"common_bins": [float(EDGES[j]) for j in common]}
    f1 = {}
    for j in common:
        f1[f"{EDGES[j]:g}"] = {f"{g}, C_dir={c}": next((r["share"] for r in c_bins if r["group"] == g and r["steps"] == "well-localised" and r["cdir"] == c
                                                        and r["outcome"] == "directed" and abs(r["b_lo"] - EDGES[j]) < 1e-9 and not r["masked"]), None)
                               for g in GROUPS for c in (1, 0)}
    forms["1"] = f1
    forms["2"] = {f"{lo:g}": {g: next((r["share"] for r in share_bins if r["group"] == g and r["steps"] == "well-localised" and r["rule"] == "argmax"
                                       and abs(r["b_lo"] - lo) < 1e-9), None) for g in GROUPS} for lo in (-1.75, -1.5)}
    for rng_label in sorted({r["range"] for r in reg}):
        g0 = next(r for r in reg if r["range"] == rng_label and r["model"] == "M0" and r["term"] == "group_exposed")
        g1 = next(r for r in reg if r["range"] == rng_label and r["model"] == "M1" and r["term"] == "group_exposed")
        forms.setdefault("3", {})[rng_label] = {"beta_group_M0": g0["coef"], "beta_group_M1": g1["coef"], "absorbed_logodds": 1 - g1["coef"] / g0["coef"],
                                                "gap_prob_M0": g0["group_gap_prob"], "gap_prob_M1": g1["group_gap_prob"],
                                                "absorbed_prob": 1 - g1["group_gap_prob"] / g0["group_gap_prob"]}
    agents = sorted({r["agent"] for r in t2})
    f4, f5 = {}, {}
    for variant in ("by draws", "by draws, shared", "expectation (section 17)"):
        vv = [r for r in t2 if r["variant"] == variant]
        f4[variant] = {f"{p:g}": float(np.mean([r["directed_share"] for r in vv if r["p"] == p])) for p in sorted({r["p"] for r in vv})}
        vals = []
        for a in agents:
            rr = sorted([r for r in vv if r["agent"] == a], key=lambda r: r["b_eff"])
            if not rr:
                continue
            vals.append(float(np.interp(-1.5, [r["b_eff"] for r in rr], [r["directed_share"] for r in rr])))
        f5[variant] = float(np.mean(vals))
    exp_bins = binned_share(S, select(S, "exposed"), S["directed"], rng)
    forms["4"] = f4
    forms["5"] = {"anti_oracle_at_-1.5": f5, "exposed_natural_bin_-1.5": next(r["share"] for r in exp_bins if abs(r["b_lo"] + 1.5) < 1e-9)}
    weights_ok = all(r["meta"]["weights_hash_before"] == r["meta"]["weights_hash_after"] for r in runs)
    (args.out / "d5_forms.json").write_text(json.dumps({"forms": forms, "identity": identity, "weights_unchanged": weights_ok,
                                                        "seconds": {r["dir"].name: r["meta"]["seconds"] for r in runs}}, indent=1, default=float),
                                            encoding="utf-8")
    print(f"weights unchanged in every run: {weights_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
