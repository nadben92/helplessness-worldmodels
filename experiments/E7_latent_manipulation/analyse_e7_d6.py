"""Analysis of E7's D6 (run_e7_d6.py; docs/e6/DECISIONS.md (61)): the h'_a channel. Reads the run directories read-only.

  d6_identity.csv     the counterfactual runs against E6 (none, reverse_none) and D5's kernels runs (every plan-B run)
  d6_bins.csv         per group, variant (a normal, b common h, c common z) and bin of b: greedy, strict-argmax and
                      epsilon-greedy-expected directed shares with 95% intervals by episodes; steps, episodes
  d6_test2_bins.csv   per group and bin: test 2 (argmax of V(h'_a, z_bar) directed, best-minus-second gap, Spearman's rho
                      with minus the distance of a's true successor)
  d6_replay.csv       per run: the epsilon-greedy directed share with the policy on (b) against the original
  d6_forms.json, figures/d6_variants.png, figures/d6_test2.png

Usage, from the repo root:
  python experiments/E7_latent_manipulation/analyse_e7_d6.py --runs "<dir>/E7_dissociation/e7d6__*" --d5-runs "<dir>/E7_dissociation/e7d5__*"
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
from scipy.stats import rankdata  # noqa: E402

import analyse_e7_diagnostics as ad  # noqa: E402
import analyse_e7_dissociation as adis  # noqa: E402
from helplessness.agents.worldmodel.heads import AgentConfig  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.measures import decision_rule as dr  # noqa: E402
from helplessness.measures import zone_phenotypes as zp  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

EDGES, MIN_STEPS, EPS = adis.EDGES, adis.MIN_STEPS, AgentConfig().epsilon
GROUPS = ("exposed", "references, no intervention", "induced references")
VARIANTS = {"a": "Q_normal", "b": "Q_hbar", "c": "Q_zbar"}
COLORS = {"exposed": "tab:red", "references, no intervention": "tab:green", "induced references": "tab:blue"}


def group_of(row: dict, base_loc: float) -> str | None:
    if row["model"] == "exposed":
        return "exposed"
    if row["mode"] == "reverse_none":
        return "references, no intervention"
    return "induced references" if row["loc_template_zone"] >= base_loc - 0.05 else None


def collect(runs: list[dict], worlds: dict) -> dict:
    cols: dict[str, list] = {k: [] for k in ("agent", "group", "episode", "b")}
    for k in ("a", "b", "c"):
        for r in ("greedy", "argmax", "eps"):
            cols[f"{k}_{r}"] = []
    for k in ("t2_argmax", "t2_gap", "t2_rho"):
        cols[k] = []
    for run in runs:
        agent, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        rows = run["summary"]["rows"]
        base = next(r for r in rows if r["mode"] == "reverse_none")["loc_template_zone"]
        for r in rows:
            group = group_of(r, base)
            if group is None:
                continue
            tag = f"{r['mode']}__{r['alpha']:g}"
            st, ln = ep[f"{tag}__states"], ep[f"{tag}__length"]
            keep = zp.in_zone_steps(st, ln, w.in_zone) & np.isfinite(ep[f"{tag}__I"]) & (ep[f"{tag}__loc_template"] == 1)
            i, t = np.nonzero(keep)
            s = st[i, t]
            dirs = w.directed[s]
            cols["agent"] += [agent] * len(i)
            cols["group"] += [group] * len(i)
            cols["episode"] += [f"{agent}:{tag}:{e}" for e in i]
            cols["b"].append(ep[f"{tag}__b_local"][i, t])
            for k, key in VARIANTS.items():
                Q = ep[f"{tag}__{key}"][i, t]
                g = dr.greedy_directed_share(Q, dirs, ad.TIE_TOL)
                cols[f"{k}_greedy"].append(g)
                cols[f"{k}_argmax"].append(dr.argmax_directed(Q, dirs).astype(float))
                cols[f"{k}_eps"].append((1 - EPS) * g + EPS * dirs.sum(axis=1) / dirs.shape[1])
            V = ep[f"{tag}__V_zbar"][i, t]
            cols["t2_argmax"].append(dr.argmax_directed(V, dirs).astype(float))
            cols["t2_gap"].append(dr.best_second_gap(V))
            rv = rankdata(V, axis=1)
            rd = rankdata(-w.dist[w.succ[s]], axis=1)
            rv, rd = rv - rv.mean(1, keepdims=True), rd - rd.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols["t2_rho"].append((rv * rd).sum(1) / np.sqrt((rv ** 2).sum(1) * (rd ** 2).sum(1)))
    return {k: (np.concatenate(v) if k not in ("agent", "group", "episode") else np.asarray(v, dtype=object)) for k, v in cols.items()}


def bins_of(S: dict, mask: np.ndarray, y: np.ndarray, rng, ci: bool = True) -> list[dict]:
    which = np.digitize(S["b"], EDGES) - 1
    out = []
    for j in range(len(EDGES) - 1):
        sel = mask & (which == j) & np.isfinite(y)
        n = int(sel.sum())
        row = {"b_lo": EDGES[j], "n_steps": n, "n_episodes": int(len(np.unique(S["episode"][sel]))),
               "value": float(np.mean(y[sel])) if n else np.nan, "lo": np.nan, "hi": np.nan, "masked": n < MIN_STEPS}
        if n >= MIN_STEPS and ci:
            eps_, inv = np.unique(S["episode"][sel], return_inverse=True)
            grp = np.array([e.split(":")[0] for e in eps_])
            row["lo"], row["hi"] = adis.boot_share(np.bincount(inv, weights=y[sel]), np.bincount(inv).astype(float), grp, rng)
        out.append(row)
    return out


def pooled(S: dict, mask: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    sel = mask & np.isfinite(y)
    return (float(y[sel].mean()) if sel.any() else float("nan")), int(sel.sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--d5-runs", nargs="+", required=True)
    parser.add_argument("--e6-runs", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    rng = np.random.default_rng(7119)
    runs = adis.runs_of(args.runs)
    worlds = {r["meta"]["agent"]: ad.World(cfg, r["meta"]) for r in runs}
    cf = [r for r in runs if r["meta"]["phase"] == "counterfactual"]
    rp = [r for r in runs if r["meta"]["phase"] == "replay"]
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    identity = []
    d5 = {r["meta"]["agent"]: r for r in adis.runs_of(args.d5_runs) if r["meta"]["phase"] == "kernels"}
    for run in cf:
        agent = run["meta"]["agent"]
        ep = np.load(run["dir"] / "episodes.npz")
        tags = [f"{r['mode']}__{r['alpha']:g}" for r in run["summary"]["rows"]]
        same_q = all(np.array_equal(ep[f"{t}__Q_normal"], ep[f"{t}__Q"], equal_nan=True) for t in tags)
        ref5 = np.load(d5[agent]["dir"] / "episodes.npz") if agent in d5 else None
        same_d5 = ref5 is not None and all(np.array_equal(ep[f"{t}__{f}"], ref5[f"{t}__{f}"]) for t in tags for f in ("states", "actions"))
        e6 = [d for p in args.e6_runs for d in glob.glob(p) if Path(d).name.startswith(f"e6__{agent}__")]
        ref6 = np.load(Path(sorted(e6)[-1]) / "episodes.npz")
        same_e6 = all(np.array_equal(ep[f"{t}__{f}"], ref6[f"training__{role}__{f}"], equal_nan=True)
                      for t, role in (("none__0", "exposed"), ("reverse_none__0", "reference")) for f in ("states", "actions", "Q"))
        identity.append({"agent": agent, "a_equals_policy_Q": same_q, "states_actions_equal_D5_plan_B": same_d5, "E6_states_actions_Q": same_e6})
    ad.write_csv(args.out / "d6_identity.csv", identity)

    S = collect(cf, worlds)
    rows = []
    for g in GROUPS:
        m = S["group"] == g
        for k in VARIANTS:
            for r in ("greedy", "argmax", "eps"):
                for row in bins_of(S, m, S[f"{k}_{r}"], rng, ci=(r != "argmax")):
                    rows.append({"group": g, "variant": k, "reading": r, **row})
    ad.write_csv(args.out / "d6_bins.csv", rows)
    t2 = []
    for g in GROUPS:
        m = S["group"] == g
        for key in ("t2_argmax", "t2_gap", "t2_rho"):
            for row in bins_of(S, m, S[key], rng, ci=(key == "t2_argmax")):
                t2.append({"group": g, "quantity": key, **row})
    ad.write_csv(args.out / "d6_test2_bins.csv", t2)

    rep = []
    for run in rp:
        agent = run["meta"]["agent"]
        orig = next(r for r in cf if r["meta"]["agent"] == agent)["summary"]["rows"]
        base = next(r for r in orig if r["mode"] == "reverse_none")["loc_template_zone"]
        for r in run["summary"]["rows"]:
            o = next(x for x in orig if x["mode"] == r["mode"] and x["alpha"] == r["alpha"] and x["model"] == r["model"])
            rep.append({"agent": agent, "mode": r["mode"], "alpha": r["alpha"], "group": group_of(o, base), "b_step1": o["b_step1"],
                        "directed_original": o["directed_share"], "directed_replay_b": r["directed_share"],
                        "loc_zone_original": o["loc_template_zone"], "loc_zone_replay": r["loc_template_zone"]})
    ad.write_csv(args.out / "d6_replay.csv", rep)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (k, label) in zip(axes, (("a", "(a) normal"), ("b", "(b) h commun : seul z porte l'action"), ("c", "(c) z commun : seul h'_a porte l'action"))):
        for g in GROUPS:
            rr = [r for r in rows if r["group"] == g and r["variant"] == k and r["reading"] == "greedy" and not r["masked"]]
            ax.errorbar([r["b_lo"] + 0.125 for r in rr], [r["value"] for r in rr], yerr=[[r["value"] - r["lo"] for r in rr], [r["hi"] - r["value"] for r in rr]],
                        fmt="o-", color=COLORS[g], capsize=2, label=g)
        ax.axhline(0.4, color="gray", lw=0.6, ls="--"), ax.set_title(label, fontsize=9), ax.set_xlabel("b local (bit)"), ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("part greedy dirigée (contrefactuel pas à pas)"), axes[0].legend(fontsize=7)
    fig.suptitle("D6 — canaux de l'action vers le critic ; pas de zone bien localisés ; références induites : runs à localisation tenue", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d6_variants.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in ((axes[0], "t2_argmax", "argmax de V(h'_a, z̄) dirigé"), (axes[1], "t2_rho", "ρ de Spearman : V(h'_a, z̄) contre −distance de la case d'arrivée")):
        for g in GROUPS:
            rr = [r for r in t2 if r["group"] == g and r["quantity"] == key and not r["masked"]]
            ax.plot([r["b_lo"] + 0.125 for r in rr], [r["value"] for r in rr], "o-", color=COLORS[g], label=g)
        ax.set_ylabel(label, fontsize=8), ax.set_xlabel("b local (bit)")
    axes[0].axhline(0.4, color="gray", lw=0.6, ls="--"), axes[0].set_ylim(0, 1.02), axes[1].axhline(0, color="gray", lw=0.6), axes[0].legend(fontsize=7)
    fig.suptitle("D6, test 2 — la contrôlabilité vue par le critic à z commun", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d6_test2.png", dpi=130)
    plt.close(fig)

    forms = {"1": identity}
    ind, exp_, ref0 = (S["group"] == g for g in ("induced references", "exposed", "references, no intervention"))
    near = (S["b"] >= -1.75) & (S["b"] < -1.25)
    for label, m in (("induced_b~-1.5", ind & near), ("exposed_b~-1.5", exp_ & near), ("induced_b<-1", ind & (S["b"] < -1)),
                     ("exposed_all", exp_), ("ref_no_intervention_all", ref0), ("induced_all", ind)):
        forms[label] = {f"{k}_{r}": pooled(S, m, S[f"{k}_{r}"]) for k in VARIANTS for r in ("greedy", "eps")}
        forms[label].update({q: pooled(S, m, S[q]) for q in ("t2_argmax", "t2_gap", "t2_rho")})
    ia, ib = forms["induced_b~-1.5"]["a_greedy"][0], forms["induced_b~-1.5"]["b_greedy"][0]
    ea = forms["exposed_b~-1.5"]["a_greedy"][0]
    forms["share_of_gap_through_h"] = (ia - ib) / (ia - ea) if ia != ea else float("nan")
    for g in GROUPS:
        vals = [r["value"] for r in t2 if r["group"] == g and r["quantity"] == "t2_argmax" and not r["masked"]]
        forms[f"t2_range_{g}"] = [min(vals), max(vals)] if vals else None
    weights_ok = all(r["meta"]["weights_hash_before"] == r["meta"]["weights_hash_after"] for r in runs)
    (args.out / "d6_forms.json").write_text(json.dumps({"forms": forms, "weights_unchanged": weights_ok,
                                                        "seconds": {r["dir"].name: r["meta"]["seconds"] for r in runs}}, indent=1, default=float),
                                            encoding="utf-8")
    print(f"weights unchanged in every run: {weights_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
