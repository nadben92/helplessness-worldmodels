"""Analysis of E7's D7 (run_e7_d7.py; docs/e6/DECISIONS.md (62)): b against the controllability the critic reads (C1: the
argmax of V(h'_a, z_bar) is directed; C2: Spearman's rho of V(h'_a, z_bar) with minus the distance of a's true successor;
C3: variant (b)'s greedy directed share), for the natural agents and the correction A. Reads the run directories read-only.

  d7_identity.csv        natural runs against D6 (states, actions, Q, V_zbar) and E6; corrected runs against the recorded
                         plan-A rows
  d7_bins.csv            per z_bar, group and bin of b: C1, C2, C3 with 95% intervals by episodes, steps, episodes
  d7_corr.csv            Spearman's rho of b with C2, step by step, per group and pooled (natural agents)
  d7_recovery.csv        the exposed steps by index j of the first stay: b, C1, C2, n
  d7_correction.csv      per agent, setting and z_bar: delta b, C1, C2, C3 (corrected - uncorrected, same step) with 95%
                         intervals, the weighted C1 gap to the natural curve, and the outcome of form 4
  d7_forms.json, figures/d7_concordance.png

Usage, from the repo root:
  python experiments/E7_latent_manipulation/analyse_e7_d7.py --runs "<dir>/E7_dissociation/e7d7__*" --d6-runs "<dir>/E7_dissociation/e7d6__*"
      --apply-runs "<dir>/E7_latent_manipulation/e7apply__*" --e6-runs "<dir>/E6_phenotypes_latent/e6__*" --out <dir>
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
from scipy.stats import rankdata, spearmanr  # noqa: E402

import analyse_e7_diagnostics as ad  # noqa: E402
import analyse_e7_dissociation as adis  # noqa: E402
import analyse_e7_d6 as a6  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.measures import decision_rule as dr  # noqa: E402
from helplessness.measures import zone_phenotypes as zp  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

EDGES, MIN_STEPS = adis.EDGES, adis.MIN_STEPS
ZBAR = {"draw": "V_zbar", "mean": "V_zbar_mean"}
GROUPS = a6.GROUPS
NATURAL = ("exposed", "references, no intervention")


def c_measures(V: np.ndarray, Qh: np.ndarray, s: np.ndarray, w) -> dict[str, np.ndarray]:
    dirs = w.directed[s]
    rv = rankdata(V, axis=1)
    rd = rankdata(-w.dist[w.succ[s]], axis=1)
    rv, rd = rv - rv.mean(1, keepdims=True), rd - rd.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (rv * rd).sum(1) / np.sqrt((rv ** 2).sum(1) * (rd ** 2).sum(1))
    return {"C1": dr.argmax_directed(V, dirs).astype(float), "C2": rho, "C3": dr.greedy_directed_share(Qh, dirs, ad.TIE_TOL)}


def collect_natural(runs: list[dict], worlds: dict) -> dict:
    cols: dict[str, list] = {k: [] for k in ("agent", "group", "episode", "b", "localised", "j")}
    for z in ZBAR:
        for c in ("C1", "C2", "C3"):
            cols[f"{z}_{c}"] = []
    for run in runs:
        agent, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        rows = run["summary"]["rows"]
        base = next(r for r in rows if r["mode"] == "reverse_none")["loc_template_zone"]
        for r in rows:
            group = a6.group_of(r, base)
            if group is None:
                continue
            tag = f"{r['mode']}__{r['alpha']:g}"
            st, ln = ep[f"{tag}__states"], ep[f"{tag}__length"]
            keep = zp.in_zone_steps(st, ln, w.in_zone) & np.isfinite(ep[f"{tag}__I"])
            i, t = np.nonzero(keep)
            s = st[i, t]
            exit_t = zp.first_exit(st, ln, w.in_zone)["time"]
            cols["agent"] += [agent] * len(i)
            cols["group"] += [group] * len(i)
            cols["episode"] += [f"{agent}:{tag}:{e}" for e in i]
            cols["b"].append(ep[f"{tag}__b_local"][i, t])
            cols["localised"].append(ep[f"{tag}__loc_template"][i, t] == 1)
            cols["j"].append(np.where(t < exit_t[i], t + 1, 0))
            for z, key in ZBAR.items():
                m = c_measures(ep[f"{tag}__{key}"][i, t], ep[f"{tag}__Q_hbar"][i, t], s, w)
                for c in m:
                    cols[f"{z}_{c}"].append(m[c])
    return {k: (np.concatenate(v) if k not in ("agent", "group", "episode") else np.asarray(v, dtype=object)) for k, v in cols.items()}


def bins_of(b: np.ndarray, episode: np.ndarray, mask: np.ndarray, y: np.ndarray, rng) -> list[dict]:
    which = np.digitize(b, EDGES) - 1
    out = []
    for j in range(len(EDGES) - 1):
        sel = mask & (which == j) & np.isfinite(y)
        n = int(sel.sum())
        row = {"b_lo": EDGES[j], "n_steps": n, "n_episodes": int(len(np.unique(episode[sel]))), "value": float(y[sel].mean()) if n else np.nan,
               "lo": np.nan, "hi": np.nan, "masked": n < MIN_STEPS}
        if n >= MIN_STEPS:
            eps_, inv = np.unique(episode[sel], return_inverse=True)
            grp = np.array([e.split(":")[0] for e in eps_])
            row["lo"], row["hi"] = adis.boot_share(np.bincount(inv, weights=y[sel]), np.bincount(inv).astype(float), grp, rng)
        out.append(row)
    return out


def paired_ci(d: np.ndarray, episode: np.ndarray, rng) -> tuple[float, float, float]:
    ok = np.isfinite(d)
    d, episode = d[ok], episode[ok]
    eps_, inv = np.unique(episode, return_inverse=True)
    grp = np.zeros(len(eps_))
    lo, hi = adis.boot_share(np.bincount(inv, weights=d), np.bincount(inv).astype(float), grp, rng)
    return float(d.mean()), lo, hi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--d6-runs", nargs="+", required=True)
    parser.add_argument("--apply-runs", nargs="+", required=True)
    parser.add_argument("--e6-runs", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    rng = np.random.default_rng(7120)
    runs = adis.runs_of(args.runs)
    worlds = {r["meta"]["agent"]: ad.World(cfg, r["meta"]) for r in runs}
    nat = [r for r in runs if r["meta"]["phase"] == "natural"]
    cor = [r for r in runs if r["meta"]["phase"] == "corrected"]
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------------------------------- identity
    identity = []
    d6 = {r["meta"]["agent"]: r for r in adis.runs_of(args.d6_runs) if r["meta"]["phase"] == "counterfactual"}
    for run in nat:
        agent = run["meta"]["agent"]
        ep, ref6 = np.load(run["dir"] / "episodes.npz"), np.load(d6[agent]["dir"] / "episodes.npz")
        tags = [f"{r['mode']}__{r['alpha']:g}" for r in run["summary"]["rows"]]
        e6 = np.load(Path(sorted(d for p in args.e6_runs for d in glob.glob(p) if Path(d).name.startswith(f"e6__{agent}__"))[-1]) / "episodes.npz")
        identity.append({"agent": agent, "phase": "natural",
                         "equal_D6": all(np.array_equal(ep[f"{t}__{f}"], ref6[f"{t}__{f}"], equal_nan=True) for t in tags for f in ("states", "actions", "Q", "V_zbar", "Q_hbar")),
                         "equal_E6": all(np.array_equal(ep[f"{t}__{f}"], e6[f"training__{role}__{f}"], equal_nan=True)
                                         for t, role in (("none__0", "exposed"), ("reverse_none__0", "reference")) for f in ("states", "actions", "Q"))})
    applied = {}
    for d in sorted({Path(x) for p in args.apply_runs for x in glob.glob(p)}):
        if (d / "rows.json").is_file():
            for r in json.loads((d / "rows.json").read_text(encoding="utf-8")):
                if r["mode"] == "constrained_stepwise" and r["image"] == "training" and r["alpha"] == 1.0:
                    applied[(r["agent"], r["setting"])] = r
    for run in cor:
        r = run["summary"]["rows"][0]
        rec = applied.get((r["agent"], r["setting"]))
        identity.append({"agent": r["agent"], "phase": "corrected", "setting": r["setting"], "recorded": rec is not None,
                         "equal_recorded": None if rec is None else all(r[k] == rec[k] for k in ("directed_share", "b_step1", "loc_template_step1", "loc_template_zone"))})
    ad.write_csv(args.out / "d7_identity.csv", identity)

    # ------------------------------------------------------------------------------------------------------ task 1
    S = collect_natural(nat, worlds)
    loc = S["localised"]
    bins = []
    for z in ZBAR:
        for g in GROUPS:
            for c in ("C1", "C2", "C3"):
                if c == "C3" and z == "mean":
                    continue
                for row in bins_of(S["b"], S["episode"], (S["group"] == g) & loc, S[f"{z}_{c}"], rng):
                    bins.append({"zbar": z, "group": g, "measure": c, **row})
    ad.write_csv(args.out / "d7_bins.csv", bins)
    corr = []
    for z in ZBAR:
        for label, m in (("natural pooled", np.isin(S["group"], NATURAL)), *[(g, S["group"] == g) for g in GROUPS]):
            sel = m & loc & np.isfinite(S[f"{z}_C2"]) & np.isfinite(S["b"])
            res = spearmanr(S["b"][sel], S[f"{z}_C2"][sel])
            corr.append({"zbar": z, "group": label, "rho_b_C2": float(res.statistic), "n": int(sel.sum())})
    ad.write_csv(args.out / "d7_corr.csv", corr)
    rec = []
    for z in ZBAR:
        for jj in range(1, 13):
            sel = (S["group"] == "exposed") & loc & (S["j"] == jj)
            if sel.sum():
                rec.append({"zbar": z, "j": jj, "n": int(sel.sum()), "b": float(np.nanmean(S["b"][sel])), "C1": float(np.nanmean(S[f"{z}_C1"][sel])),
                            "C2": float(np.nanmean(S[f"{z}_C2"][sel]))})
    ad.write_csv(args.out / "d7_recovery.csv", rec)

    # ------------------------------------------------------------------------------------------------------ task 2
    natural_curve = {z: {round(r["b_lo"], 2): r for r in bins_of(S["b"], S["episode"], np.isin(S["group"], NATURAL) & loc, S[f"{z}_C1"], rng)}
                     for z in ZBAR}
    corr_rows, points = [], []
    for run in cor:
        agent, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        st, ln = ep["states"], ep["length"]
        keep = zp.in_zone_steps(st, ln, w.in_zone) & (ep["intervened"] == 1) & np.isfinite(ep["I"]) & np.isfinite(ep["I_raw"]) & (ep["loc_template"] == 1)
        i, t = np.nonzero(keep)
        s = st[i, t]
        episode = np.array([f"{agent}:{e}" for e in i], dtype=object)
        b_c, b_r = ep["b_local"][i, t], ep["b_raw"][i, t]
        for z, key in ZBAR.items():
            mc = c_measures(ep[f"corr_{key}"][i, t], ep["corr_Q_hbar"][i, t], s, w)
            mr = c_measures(ep[f"raw_{key}"][i, t], ep["raw_Q_hbar"][i, t], s, w)
            row = {"agent": agent, "setting": run["meta"]["tag"], "zbar": z, "n_steps": int(len(i)),
                   "b_raw": float(np.nanmean(b_r)), "b_corr": float(np.nanmean(b_c))}
            for name, d in (("db", b_c - b_r), *[(f"d{c}", mc[c] - mr[c]) for c in ("C1", "C2", "C3")]):
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = paired_ci(d, episode, rng)
            for c in ("C1", "C2", "C3"):
                row[f"{c}_raw"], row[f"{c}_corr"] = float(np.nanmean(mr[c])), float(np.nanmean(mc[c]))
            which = np.digitize(b_c, EDGES) - 1
            gaps, weights = [], []
            for j in np.unique(which):
                sel = which == j
                nb = natural_curve[z].get(round(EDGES[j], 2)) if 0 <= j < len(EDGES) - 1 else None
                if sel.sum() >= MIN_STEPS and nb is not None and not nb["masked"]:
                    gaps.append(mc["C1"][sel].mean() - nb["value"])
                    weights.append(sel.sum())
            row["c1_gap_to_natural"] = float(np.average(gaps, weights=weights)) if gaps else np.nan
            row["n_steps_in_compared_bins"] = int(np.sum(weights)) if weights else 0
            real = row["dC1_lo"] > 0 and row["dC2_lo"] > 0 and np.isfinite(row["c1_gap_to_natural"]) and abs(row["c1_gap_to_natural"]) <= 0.10
            fooled = row["db"] >= 1 and row["dC1"] <= 0.10 and np.isfinite(row["c1_gap_to_natural"]) and row["c1_gap_to_natural"] < -0.10
            row["outcome"] = "real" if real else ("measure fooled" if fooled else "intermediate")
            corr_rows.append(row)
            if z == "draw":
                points.append((row["b_raw"], row["C1_raw"], row["b_corr"], row["C1_corr"], row["C2_raw"], row["C2_corr"], agent, row["setting"]))
    ad.write_csv(args.out / "d7_correction.csv", corr_rows)

    # ------------------------------------------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, c in zip(axes, ("C1", "C2")):
        for g in GROUPS:
            rr = [r for r in bins if r["zbar"] == "draw" and r["group"] == g and r["measure"] == c and not r["masked"]]
            ax.errorbar([r["b_lo"] + 0.125 for r in rr], [r["value"] for r in rr], yerr=[[r["value"] - r["lo"] for r in rr], [r["hi"] - r["value"] for r in rr]],
                        fmt="o-" if g in NATURAL else "s--", color=a6.COLORS[g], capsize=2, label=g + ("" if g in NATURAL else " (D6)"))
        for br, c1r, bc, c1c, c2r, c2c, agent, setting in points:
            yr, yc = (c1r, c1c) if c == "C1" else (c2r, c2c)
            ax.annotate("", xy=(bc, yc), xytext=(br, yr), arrowprops={"arrowstyle": "->", "color": "tab:purple", "lw": 0.9})
            ax.plot(bc, yc, "D", color="tab:purple", ms=4)
        ax.set_xlabel("b local (bit)")
    axes[0].set_ylabel("C1 : argmax de V(h'_a, z̄) dirigé"), axes[0].axhline(0.4, color="gray", lw=0.6, ls="--"), axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("C2 : ρ(V(h'_a, z̄), −distance)"), axes[1].axhline(0, color="gray", lw=0.6)
    axes[0].plot([], [], "D", color="tab:purple", label="correction A : avant → après (moyenne par agent et réglage)")
    axes[0].legend(fontsize=7)
    fig.suptitle("D7 — b contre la contrôlabilité vue par le critic (z̄ principal) ; pas de zone bien localisés", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d7_concordance.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------------------------------------------- forms
    forms = {}
    for z in ZBAR:
        f = {"rho_natural_pooled": next(r["rho_b_C2"] for r in corr if r["zbar"] == z and r["group"] == "natural pooled"),
             "rho_exposed": next(r["rho_b_C2"] for r in corr if r["zbar"] == z and r["group"] == "exposed")}
        m = (S["group"] == "exposed") & loc & (S["b"] >= -2.25) & (S["b"] < -1.75)
        f["exposed_C1_b~-2.1"] = float(np.nanmean(S[f"{z}_C1"][m]))
        m = (S["group"] == "references, no intervention") & loc & (S["b"] >= -0.25) & (S["b"] < 0.25)
        f["references_C1_b~0"] = float(np.nanmean(S[f"{z}_C1"][m]))
        ex = [r for r in bins if r["zbar"] == z and r["group"] == "exposed" and r["measure"] == "C1" and not r["masked"]]
        f["exposed_max_adjacent_C1_jump"] = float(max(abs(a["value"] - b["value"]) for a, b in zip(ex, ex[1:]))) if len(ex) > 1 else None
        nc = {round(r["b_lo"], 2): r for r in bins_of(S["b"], S["episode"], np.isin(S["group"], NATURAL) & loc, S[f"{z}_C1"], rng)}
        ind = [r for r in bins if r["zbar"] == z and r["group"] == "induced references" and r["measure"] == "C1" and not r["masked"]]
        f["induced_minus_natural_C1"] = {f"{r['b_lo']:g}": r["value"] - nc[round(r["b_lo"], 2)]["value"] for r in ind
                                         if not nc[round(r["b_lo"], 2)]["masked"]}
        f["correction"] = [{k: r[k] for k in ("agent", "setting", "db", "dC1", "dC1_lo", "dC2", "dC2_lo", "dC3", "c1_gap_to_natural", "outcome")}
                           for r in corr_rows if r["zbar"] == z]
        forms[z] = f
    weights_ok = all(r["meta"]["weights_hash_before"] == r["meta"]["weights_hash_after"] for r in runs)
    (args.out / "d7_forms.json").write_text(json.dumps({"forms": forms, "identity": identity, "weights_unchanged": weights_ok,
                                                        "seconds": {r["dir"].name: r["meta"]["seconds"] for r in runs}}, indent=1, default=float),
                                            encoding="utf-8")
    print(f"weights unchanged in every run: {weights_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
