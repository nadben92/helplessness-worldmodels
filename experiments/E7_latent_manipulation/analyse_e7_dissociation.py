"""Analysis of E7's belief x critic dissociation (docs/e6/DECISIONS.md (55)-(57)): the anti-oracle (task 1), E6 step by step
(task 2), D2's localisation control (task 3), the 2 x 2 table and the expected forms of the report's section 17. Reads the
run directories read-only.

  task 1  t1_runs.csv (per agent, model, h_mode, p, k: b_eff, the epsilon-greedy directed share and its 95% interval
          (episodes resampled), the strict argmax's and the tie set's directed shares at the zone steps, the share of zone
          steps where all 5 actions tie, the localisation, p_goal), t1_identity.csv (p = 0, k = 1 against E6);
          figures/t1_directed_vs_beff.png.
  task 2  t2_bins.csv (per model and bin of b_local: directed share, 95% interval, steps, episodes; bins < 30 steps
          masked); figures/t2_stepwise.png (the anti-oracle's k = 1 points overlaid: effective b, not natural b).
  task 3  t3_bins.csv (reference steps of D2 per bin: share localised, greedy directed share all steps / well-localised /
          run-level filter); figures/t3_localisation.png.
  2 x 2   table_2x2.csv; forms.json (the six forms' numbers).

Usage, from the repo root:
  python experiments/E7_latent_manipulation/analyse_e7_dissociation.py --runs "<dir>/E7_dissociation/e7dis__*"
      --diag-runs "<dir>/E7_latent_manipulation/e7diag__*" --e6-runs "<dir>/E6_phenotypes_latent/e6__*" --out <dir>
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
import yaml  # noqa: E402

import analyse_e7_diagnostics as ad  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.measures import decision_rule as dr  # noqa: E402
from helplessness.measures import zone_phenotypes as zp  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

N_BOOT, MIN_STEPS, BIN = 1000, 30, 0.25
EDGES = np.arange(-2.75, 0.5 + 1e-9, BIN)


def runs_of(patterns: list[str]) -> list[dict]:
    out = []
    for p in patterns:
        for d in sorted(Path(x) for x in glob.glob(p)):
            if (d / "status.json").is_file() and json.loads((d / "status.json").read_text())["status"] == "done":
                out.append({"dir": d, "meta": yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8")),
                            "summary": json.loads((d / "summary.json").read_text(encoding="utf-8"))})
    return out


def boot_share(n_dir: np.ndarray, n_steps: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval of sum(n_dir) / sum(n_steps), episodes resampled within each group (agent)."""
    stats = []
    idx_by = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(N_BOOT):
        pick = np.concatenate([rng.choice(i, size=len(i), replace=True) for i in idx_by])
        tot = n_steps[pick].sum()
        stats.append(n_dir[pick].sum() / tot if tot else np.nan)
    return tuple(np.nanpercentile(stats, [2.5, 97.5]))


# -------------------------------------------------------------------------------------------------------------- task 1
def task1(runs: list[dict], worlds: dict, e6_runs: list[str], rng) -> tuple[list[dict], list[dict]]:
    rows, identity = [], []
    for run in runs:
        meta, w = run["meta"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        for r in run["summary"]["rows"]:
            tag = f"p{r['p']:g}__k{r['k']}"
            st, ln = ep[f"{tag}__states"], ep[f"{tag}__length"]
            inz = zp.in_zone_steps(st, ln, w.in_zone)
            s = np.clip(st[:, :-1], 0, None)[inz]
            Q = ep[f"{tag}__Q"][inz]
            ds = zp.directed_share(st, ln, w.in_zone, w.dist)
            lo, hi = boot_share(ds["n_directed"], ds["n_zone_steps"], np.zeros(len(ln)), rng)
            rows.append({"agent": meta["agent"], "model": r["model"], "h_mode": meta["h_mode"], "p": r["p"], "k": r["k"], "b_eff": r["b_eff"],
                         "directed_share": r["directed_share"], "directed_lo": lo, "directed_hi": hi,
                         "argmax_directed": float(dr.argmax_directed(Q, w.directed[s]).mean()),
                         "tieset_directed": float(dr.greedy_directed_share(Q, w.directed[s], ad.TIE_TOL).mean()),
                         "all_tied": float((dr.tie_set(Q, ad.TIE_TOL).sum(axis=1) == Q.shape[1]).mean()),
                         "loc_zone": r["loc_template_zone"], "p_goal_zone": r["p_goal_zone"], "seconds": r["seconds"]})
            if r["p"] == 0 and r["k"] == 1:
                e6 = [d for p in e6_runs for d in glob.glob(p) if Path(d).name.startswith(f"e6__{meta['agent']}__")]
                if e6:
                    ref = np.load(Path(sorted(e6)[-1]) / "episodes.npz")
                    role = "exposed" if r["model"] == "exposed" else "reference"
                    same = {f: np.array_equal(ep[f"{tag}__{f}"], ref[f"training__{role}__{f}"], equal_nan=True) for f in ("states", "actions", "Q")}
                    I_e6, I_here = ref[f"training__{role}__I"], ep[f"{tag}__I"]
                    same["I_zone"] = I_here.shape == I_e6.shape and bool(np.array_equal(I_here[np.isfinite(I_here)], I_e6[np.isfinite(I_here)]))
                    identity.append({"agent": meta["agent"], "model": role, "h_mode": meta["h_mode"], **same,
                                     "I_kernel_p0_minus_E6_I": run["summary"]["b_eff"]["I_kernel_p0"] - run["summary"]["I_ref_step1"]})
    return rows, identity


def fig_task1(rows: list[dict], figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, k in zip(axes, (1, 5)):
        for agent in sorted({r["agent"] for r in rows}):
            rr = sorted([r for r in rows if r["agent"] == agent and r["k"] == k and r["model"] == "reference" and r["h_mode"] == "mean"], key=lambda r: r["p"])
            ax.errorbar([r["b_eff"] for r in rr], [r["directed_share"] for r in rr],
                        yerr=[[r["directed_share"] - r["directed_lo"] for r in rr], [r["directed_hi"] - r["directed_share"] for r in rr]],
                        fmt="^-", ms=4, lw=0.9, capsize=2, label=agent.replace("_e4", ""))
            for r in [r for r in rows if r["agent"] == agent and r["k"] == k and r["model"] == "exposed"]:
                ax.plot(r["b_eff"], r["directed_share"], "o", color="tab:red", ms=6, mfc="tab:red" if r["p"] == 1 else "none")
            for r in [r for r in rows if r["agent"] == agent and r["k"] == k and r["h_mode"] == "action"]:
                ax.plot(r["b_eff"], r["directed_share"], "x", color="black", ms=7)
        ax.axhline(0.4, color="gray", lw=0.6, ls="--")
        ax.set_title(f"k = {k}", fontsize=10), ax.set_xlabel("b effectif au pas 1 (bit)")
    axes[0].set_ylabel("part dirigée dans la zone (ε-greedy)"), axes[0].legend(fontsize=7)
    fig.suptitle("Anti-oracle chez les références (triangles, p = 0 → 1 de droite à gauche) ; ronds rouges : exposé (vide p = 0, plein p = 1) ;"
                 " croix : variante h'_a à p = 1", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "t1_directed_vs_beff.png", dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------------- task 2
def task2(e6_runs: list[str], worlds: dict, rng) -> list[dict]:
    per = {"exposed": [], "reference": []}
    for d in sorted({Path(x) for p in e6_runs for x in glob.glob(p)}):
        agent = d.name.split("__")[1]
        if agent not in worlds:
            continue
        w = worlds[agent]
        ep = np.load(d / "episodes.npz")
        for role in per:
            st, ln = ep[f"training__{role}__states"], ep[f"training__{role}__length"]
            inz = zp.in_zone_steps(st, ln, w.in_zone)
            dirs = zp.directed_steps(st, ln, w.dist)
            b = ep[f"training__{role}__b_local"]
            ok = inz & np.isfinite(b)
            i, t = np.nonzero(ok)
            per[role].append({"agent": agent, "b": b[i, t], "directed": dirs[i, t].astype(float), "episode": i})
    out = []
    for role, items in per.items():
        b = np.concatenate([x["b"] for x in items])
        y = np.concatenate([x["directed"] for x in items])
        ep_id = np.concatenate([[f"{x['agent']}:{e}" for e in x["episode"]] for x in items])
        agent = np.concatenate([[x["agent"]] * len(x["b"]) for x in items])
        which = np.digitize(b, EDGES) - 1
        for j in range(len(EDGES) - 1):
            sel = which == j
            n = int(sel.sum())
            row = {"model": role, "b_lo": EDGES[j], "b_hi": EDGES[j + 1], "n_steps": n, "n_episodes": int(len(np.unique(ep_id[sel]))),
                   "directed_share": float(y[sel].mean()) if n else np.nan, "lo": np.nan, "hi": np.nan, "masked": n < MIN_STEPS}
            if n >= MIN_STEPS:
                eps, inv = np.unique(ep_id[sel], return_inverse=True)
                n_dir, n_st = np.bincount(inv, weights=y[sel]), np.bincount(inv)
                grp = np.array([e.split(":")[0] for e in eps])
                row["lo"], row["hi"] = boot_share(n_dir, n_st.astype(float), grp, rng)
            out.append(row)
    return out


def fig_task2(bins: list[dict], t1: list[dict], figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for role, color in (("exposed", "tab:red"), ("reference", "tab:blue")):
        rr = [r for r in bins if r["model"] == role and not r["masked"]]
        x = [(r["b_lo"] + r["b_hi"]) / 2 for r in rr]
        ax.errorbar(x, [r["directed_share"] for r in rr], yerr=[[r["directed_share"] - r["lo"] for r in rr], [r["hi"] - r["directed_share"] for r in rr]],
                    fmt="o-", color=color, capsize=2, label=f"{'exposés' if role == 'exposed' else 'références'} E6, pas à pas (b naturel)")
    for k, style in ((1, "s--"), (5, "s:")):
        ao = [r for r in t1 if r["model"] == "reference" and r["k"] == k and r["h_mode"] == "mean"]
        ps = sorted({r["p"] for r in ao})
        ax.plot([np.mean([r["b_eff"] for r in ao if r["p"] == p]) for p in ps], [np.mean([r["directed_share"] for r in ao if r["p"] == p]) for p in ps],
                style, color="tab:green", alpha=1.0 if k == 1 else 0.6,
                label=f"références + anti-oracle k = {k} (b EFFECTIF, moyenne des 4 agents)")
    ax.axhline(0.4, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("b (bit) — naturel (E6) ou effectif (anti-oracle) : pas la même mesure"), ax.set_ylabel("part dirigée")
    ax.set_ylim(0, 1.02), ax.legend(fontsize=7)
    ax.set_title(f"E6 pas à pas (classes de {BIN} bit, ≥ {MIN_STEPS} pas, IC 95 % par épisodes) et anti-oracle", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "t2_stepwise.png", dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------------- task 3
def task3(diag_runs: list[dict], worlds: dict) -> tuple[list[dict], dict]:
    recs = []
    e6_loc = {}
    for run in diag_runs:
        if run["meta"]["phase"] != "induction":
            continue
        agent, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        base = next(r for r in run["summary"]["rows"] if r["mode"] == "reverse_none")
        e6_loc[agent] = base["loc_template_zone"]
        for r in run["summary"]["rows"]:
            if r["model"] != "reference":
                continue
            tag = f"{r['mode']}__{r['alpha']:g}"
            st, ln = ep[f"{tag}__states"], ep[f"{tag}__length"]
            inz = zp.in_zone_steps(st, ln, w.in_zone) & np.isfinite(ep[f"{tag}__I"])
            s = np.clip(st[:, :-1], 0, None)[inz]
            Q = ep[f"{tag}__Q"][inz]
            recs.append({"b": ep[f"{tag}__b_local"][inz], "loc": ep[f"{tag}__loc_template"][inz] == 1,
                         "greedy": dr.greedy_directed_share(Q, w.directed[s], ad.TIE_TOL),
                         "run_ok": np.full(int(inz.sum()), r["loc_template_zone"] >= base["loc_template_zone"] - 0.05)})
    b = np.concatenate([x["b"] for x in recs])
    loc = np.concatenate([x["loc"] for x in recs])
    g = np.concatenate([x["greedy"] for x in recs])
    run_ok = np.concatenate([x["run_ok"] for x in recs])
    level = float(np.mean(list(e6_loc.values())))
    which = np.digitize(b, EDGES) - 1
    out = []
    for j in range(len(EDGES) - 1):
        sel = which == j
        n = int(sel.sum())
        if n == 0:
            continue

        def share(m):
            return float(g[m].mean()) if m.sum() >= MIN_STEPS else np.nan
        out.append({"b_lo": EDGES[j], "b_hi": EDGES[j + 1], "n_all": n, "share_localised": float(loc[sel].mean()),
                    "flag_under_e6_minus_0.05": bool(loc[sel].mean() < level - 0.05),
                    "greedy_all_steps": share(sel), "greedy_well_localised_D2": share(sel & loc), "n_well_localised": int((sel & loc).sum()),
                    "greedy_run_filter": share(sel & run_ok), "n_run_filter": int((sel & run_ok).sum()),
                    "greedy_run_filter_and_localised": share(sel & run_ok & loc)})
    return out, {"e6_reference_loc_zone": e6_loc, "level": level}


def fig_task3(bins: list[dict], info: dict, figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [(r["b_lo"] + r["b_hi"]) / 2 for r in bins]
    for key, label, style in (("greedy_well_localised_D2", "D2 (pas bien localisés)", "o-"), ("greedy_all_steps", "tous les pas", "s--"),
                              ("greedy_run_filter", "runs à localisation ≥ E6 − 0,05", "^:")):
        axes[0].plot(x, [r[key] for r in bins], style, label=label)
    axes[0].set_ylabel("part dirigée du choix greedy (références)"), axes[0].legend(fontsize=8), axes[0].set_ylim(0, 1.02)
    axes[1].plot(x, [r["share_localised"] for r in bins], "o-", color="black")
    axes[1].axhline(info["level"] - 0.05, color="red", ls="--", lw=0.8, label="E6 référence − 0,05")
    axes[1].set_ylabel("part des pas bien localisés"), axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("b local (bit)")
    fig.suptitle("Tâche 3 — localisation des pas de référence de D2 (E6 + induction B)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "t3_localisation.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------- 2 x 2 and forms
def table_2x2(t1: list[dict], diag_runs: list[dict]) -> list[dict]:
    la = [r for run in diag_runs if run["meta"]["phase"] == "lookahead" for r in run["summary"]["rows"]]
    e6 = [r for run in diag_runs if run["meta"]["phase"] == "induction" for r in run["summary"]["rows"] if r["mode"] in ("none", "reverse_none")]
    agents = sorted({r["agent"] for r in t1})

    def pick(rows, **kw):
        v = [r["directed_share"] for r in rows if all(r.get(k) == val for k, val in kw.items())]
        return float(np.mean(v)) if v else np.nan

    out = []
    for agent in agents + ["mean"]:
        sel = (lambda r: True) if agent == "mean" else (lambda r, a=agent: r["agent"] == a)
        T, L, E = [r for r in t1 if sel(r) and r["h_mode"] == "mean"], [r for r in la if sel(r)], [r for r in e6 if sel(r)]
        for k in (1, 5):
            out.append({"agent": agent, "k": k,
                        "true_belief__ref_critic": pick(E, model="reference") if k == 1 else pick(L, model="reference", k=5),
                        "true_belief__ref_critic_own_model": pick(T, model="reference", p=0.0, k=k),
                        "true_belief__exp_critic (oracle D4)": pick(L, model="exposed", k=k),
                        "conveyor__ref_critic (anti-oracle p=1)": pick(T, model="reference", p=1.0, k=k),
                        "conveyor__exp_critic": pick(E, model="exposed") if k == 1 else pick(T, model="exposed", p=1.0, k=5),
                        "exp_own_model": pick(T, model="exposed", p=0.0, k=k), "exp_anti_oracle_p1": pick(T, model="exposed", p=1.0, k=k)})
    return out


def forms(t1: list[dict], t2: list[dict], t3: list[dict], diag_runs: list[dict]) -> dict:
    ref = [r for r in t1 if r["model"] == "reference" and r["h_mode"] == "mean"]
    agents = sorted({r["agent"] for r in ref})
    e6 = {(r["agent"], r["model"]): r["directed_share"] for run in diag_runs if run["meta"]["phase"] == "induction"
          for r in run["summary"]["rows"] if r["mode"] in ("none", "reverse_none")}
    f = {}
    viol, loc_dev = [], []
    for a in agents:
        for k in (1, 5):
            rr = sorted([r for r in ref if r["agent"] == a and r["k"] == k], key=lambda r: r["p"])
            for x, y in zip(rr, rr[1:]):
                if y["directed_lo"] > x["directed_hi"]:
                    viol.append((a, k, x["p"], y["p"]))
            loc_dev.append(max(abs(r["loc_zone"] - rr[0]["loc_zone"]) for r in rr))
    f["1"] = {"rises_beyond_ci": viol, "max_loc_deviation": max(loc_dev)}
    f["2"] = {a: next(r["directed_share"] for r in ref if r["agent"] == a and r["k"] == 1 and r["p"] == 1.0) for a in agents}
    f["3"] = {a: next(r["directed_share"] for r in ref if r["agent"] == a and r["k"] == 5 and r["p"] == 1.0) - f["2"][a] for a in agents}
    f["4"] = {a: next((r["directed_share"] for r in t1 if r["agent"] == a and r["model"] == "exposed" and r["k"] == 1 and r["p"] == 1.0), np.nan)
              - e6.get((a, "exposed"), np.nan) for a in agents}
    stepwise = {round(r["b_lo"], 2): r["directed_share"] for r in t2 if r["model"] == "exposed" and not r["masked"]}
    f5 = {}
    for target, lo in ((-1.5, -1.5), (-2.1, -2.25), (0.0, 0.0)):
        vals = []
        for a in agents:
            rr = sorted([r for r in ref if r["agent"] == a and r["k"] == 1], key=lambda r: r["b_eff"])
            vals.append(float(np.interp(target, [r["b_eff"] for r in rr], [r["directed_share"] for r in rr])))
        f5[f"{target:g}"] = {"anti_oracle_k1": float(np.mean(vals)), "exposed_stepwise_bin": stepwise.get(lo, np.nan)}
    f["5"] = f5
    d15 = next((r for r in t3 if abs(r["b_lo"] + 1.5) < 1e-9), None)
    f["6"] = {"bin_-1.5": d15}
    return f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--diag-runs", nargs="+", required=True)
    parser.add_argument("--e6-runs", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    rng = np.random.default_rng(7117)
    runs, diag = runs_of(args.runs), runs_of(args.diag_runs)
    worlds = {r["meta"]["agent"]: ad.World(cfg, r["meta"]) for r in runs + diag}
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    t1, identity = task1(runs, worlds, args.e6_runs, rng)
    ad.write_csv(args.out / "t1_runs.csv", t1)
    ad.write_csv(args.out / "t1_identity.csv", identity)
    fig_task1(t1, figdir)
    t2 = task2(args.e6_runs, worlds, rng)
    ad.write_csv(args.out / "t2_bins.csv", t2)
    fig_task2(t2, t1, figdir)
    t3, info = task3(diag, worlds)
    ad.write_csv(args.out / "t3_bins.csv", t3)
    fig_task3(t3, info, figdir)
    tab = table_2x2(t1, diag)
    ad.write_csv(args.out / "table_2x2.csv", tab)
    fo = forms(t1, t2, t3, diag)
    weights_ok = all(r["meta"]["weights_hash_before"] == r["meta"]["weights_hash_after"] for r in runs)
    (args.out / "forms.json").write_text(json.dumps({"forms": fo, "task3_info": info, "identity": identity, "weights_unchanged": weights_ok,
                                                     "seconds": {r["dir"].name: r["meta"]["seconds"] for r in runs}}, indent=1, default=float), encoding="utf-8")
    print(f"weights unchanged in every run: {weights_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
