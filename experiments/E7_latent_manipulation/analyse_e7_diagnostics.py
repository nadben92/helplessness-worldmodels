"""Analysis of E7's diagnostics of the value and of the decision rule (run_e7_diagnostics.py; docs/e6/DECISIONS.md
(51)-(54)). Reads the run directories read-only; writes tables, figures and summary.json.

  D1  d1_values.csv: per agent and model, Spearman of V with minus the distance to the goal (all cells; zone AI), the
      greedy and argmax directed shares of the oracle and of the model's one-step Q at the zone's contexts.
      figures/d1_value_maps.png.
  D2  d2_steps.csv (every well-localised zone step: b_local, the best-minus-second gap, the greedy directed share),
      d2_bins.csv (per group and bin of b), the identity of the rerun E6 episodes with E6's (--e6-runs);
      figures/d2_gap_directed_vs_b.png; the tau grid of (53) printed and in summary.json.
  D3  d3_points.csv (per agent, tau, mode, alpha: b at step 1, the directed share, the localisation checks), Spearman of
      the directed share with b over B's points that keep the localisation; figures/d3_softmax.png.
  D4  d4_static.csv (per agent, model, variant, k: the greedy and argmax directed shares), d4_b_trace.csv (b along the
      imagined steps), d4_conveyor.csv (the imagined transitions from the zone: conveyor vs action), d4_episodes.csv;
      figures/d4_lookahead.png, figures/d4_b_trace.png.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/analyse_e7_diagnostics.py --runs "<dir>/e7diag__*" [--e6-runs "<dir>/e6__*"] --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from helplessness.agents.worldmodel.heads import AgentConfig  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.envs.layouts import two_zones  # noqa: E402
from helplessness.measures import decision_rule as dr  # noqa: E402
from helplessness.measures import zone_phenotypes as zp  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402
from helplessness.pipeline import two_zones_setup as setup  # noqa: E402

TIE_TOL = AgentConfig().tie_tol_abs  # the training's absolute tie tolerance, 0.0751 (every E4 checkpoint's agent_config)
VARIANT_STYLE = {"oracle": "tab:green", "model": "tab:red", "model_corrected": "tab:purple"}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    rx, ry = np.argsort(np.argsort(x[keep])), np.argsort(np.argsort(y[keep]))
    return float(np.corrcoef(rx, ry)[0, 1]) if keep.sum() > 2 else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class World:
    """The test world's successors, the conveyor and the directed actions of an agent's configuration (no model)."""

    def __init__(self, cfg: dict, meta: dict):
        conf = {**setup.configuration(cfg, meta["configuration"]), "seed": 0}
        self.succ = setup.successors(setup.probe_world(cfg, conf))
        self.conveyor = setup.successors(setup.world(cfg, conf, "experimental"))[:, 4]
        self.dist = fs.goal_distances()
        self.directed = dr.directed_action_mask(self.succ, self.dist)
        self.zone = fs.zone_states(meta["slot"])
        self.in_zone = np.zeros(two_zones.N_STATES, dtype=bool)
        self.in_zone[self.zone] = True


def load_runs(patterns: list[str]) -> dict[str, list[dict]]:
    """Newest run per (agent, phase) among the patterns: phase -> [{dir, meta, summary}]."""
    newest: dict[tuple, Path] = {}
    for p in patterns:
        for d in sorted(Path(x) for x in glob.glob(p)):
            if not (d / "meta.yaml").is_file() or json.loads((d / "status.json").read_text())["status"] != "done":
                continue
            meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
            newest[(meta["agent"], meta["phase"])] = d
    out: dict[str, list[dict]] = {}
    for (agent, phase), d in sorted(newest.items()):
        out.setdefault(phase, []).append({"dir": d, "meta": yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8")),
                                          "summary": json.loads((d / "summary.json").read_text(encoding="utf-8"))})
    return out


# ------------------------------------------------------------------------------------------------------------------ D1
def d1(runs: list[dict], worlds: dict, figdir: Path) -> list[dict]:
    rows = []
    fig, axes = plt.subplots(2, len(runs), figsize=(3.2 * len(runs), 6.4), squeeze=False)
    goal_rc = divmod(two_zones.GOAL_STATE, two_zones.GRID_SHAPE[1])
    for j, run in enumerate(runs):
        a, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        v = np.load(run["dir"] / "values.npz")
        cells = v["cells"]
        lim = [np.nanmin([v[f"{r}__V"] for r in ("experimental", "reference")]), np.nanmax([v[f"{r}__V"] for r in ("experimental", "reference")])]
        for i, role in enumerate(("experimental", "reference")):
            V = v[f"{role}__V"]
            row = {"agent": a, "model": "exposed" if role == "experimental" else "reference",
                   "spearman_V_all": spearman(V, -w.dist.astype(float)), "spearman_V_zone": spearman(V[w.zone], -w.dist[w.zone].astype(float)),
                   "V_zone_mean": float(np.nanmean(V[w.zone]))}
            for variant in ("oracle", "model"):
                Q = v[f"{role}__{variant}__Q"][..., 0]
                row[f"{variant}_greedy_directed"] = float(dr.greedy_directed_share(Q, w.directed[cells], TIE_TOL).mean())
                row[f"{variant}_argmax_directed"] = float(dr.argmax_directed(Q, w.directed[cells]).mean())
                row[f"{variant}_gap_median"] = float(np.median(dr.best_second_gap(Q)))
            rows.append(row)
            ax = axes[i, j]
            im = ax.imshow(V.reshape(two_zones.GRID_SHAPE), cmap="viridis", vmin=lim[0], vmax=lim[1])
            r0, c0 = np.divmod(w.zone, two_zones.GRID_SHAPE[1])
            ax.add_patch(plt.Rectangle((c0.min() - 0.5, r0.min() - 0.5), c0.max() - c0.min() + 1, r0.max() - r0.min() + 1, fill=False,
                                       ec="white", lw=2))
            ax.plot(goal_rc[1], goal_rc[0], marker="*", color="red", ms=14)
            ax.set_title(f"{a.replace('_e4', '')} {row['model']}\noracle dirigé {row['oracle_greedy_directed']:.2f}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("V du critic au pas 1 (départ sur la case), image d'entraînement ; cadre blanc : zone AI ; étoile : but", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "d1_value_maps.png", dpi=130)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------------------------------------------------------ D2
def zone_step_records(ep: dict, tag: str, w: World) -> dict[str, np.ndarray]:
    """The well-localised, measured steps taken in the zone of one set of episodes."""
    inz = zp.in_zone_steps(ep[f"{tag}__states"], ep[f"{tag}__length"], w.in_zone)
    keep = inz & np.isfinite(ep[f"{tag}__I"]) & (ep[f"{tag}__loc_template"] == 1)
    s = np.clip(ep[f"{tag}__states"][:, :-1], 0, None)[keep]
    Q = ep[f"{tag}__Q"][keep]
    return {"b": ep[f"{tag}__b_local"][keep], "gap": dr.best_second_gap(Q), "greedy_directed": dr.greedy_directed_share(Q, w.directed[s], TIE_TOL),
            "argmax_directed": dr.argmax_directed(Q, w.directed[s]).astype(float)}


def d2(runs: list[dict], worlds: dict, e6_runs: list[str], bin_width: float, figdir: Path) -> tuple[list[dict], list[dict], dict]:
    steps, identity = [], []
    for run in runs:
        a, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        ep = np.load(run["dir"] / "episodes.npz")
        for r in run["summary"]["rows"]:
            tag = f"{r['mode']}__{r['alpha']:g}"
            rec = zone_step_records(ep, tag, w)
            group = "exposed (E6)" if r["model"] == "exposed" else ("reference (B)" if r["mode"] != "reverse_none" else "reference (E6)")
            for i in range(len(rec["b"])):
                steps.append({"agent": a, "group": group, "mode": r["mode"], "alpha": r["alpha"], **{k: float(v[i]) for k, v in rec.items()}})
        e6 = [d for p in e6_runs for d in glob.glob(p) if Path(d).name.startswith(f"e6__{a}__")]
        if e6:
            ref = np.load(Path(sorted(e6)[-1]) / "episodes.npz")
            name = run["meta"]["image"]
            for tag, role in (("none__0", "exposed"), ("reverse_none__0", "reference")):
                identity.append({"agent": a, "model": role, "identical_to_e6": all(
                    np.array_equal(ep[f"{tag}__{f}"], ref[f"{name}__{role}__{f}"], equal_nan=True) for f in ("states", "actions", "I", "Q"))})
    edges = np.arange(-2.75, 0.5 + 1e-9, bin_width)
    bins = []
    groups = {"references (E6 + B)": lambda s: s["group"] != "exposed (E6)", "exposed (E6)": lambda s: s["group"] == "exposed (E6)",
              "all": lambda s: True}
    for gname, sel in groups.items():
        g = [s for s in steps if sel(s)]
        b = np.array([s["b"] for s in g])
        for q in ("gap", "greedy_directed"):
            res = dr.binned(b, np.array([s[q] for s in g]), edges)
            for i in range(len(res["n"])):
                bins.append({"group": gname, "quantity": q, "b_lo": res["lo"][i], "b_hi": res["hi"][i], "n": int(res["n"][i]),
                             "median": res["median"][i], "mean": res["mean"][i]})
    ref_gaps = np.array([s["gap"] for s in steps if s["group"] != "exposed (E6)"])
    g_med = float(np.median(ref_gaps)) if ref_gaps.size else float("nan")
    taus = [float(f"{g_med * m:.3g}") for m in (0.1, 10 ** -0.5, 1.0, 10 ** 0.5, 10.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for gname, color in (("references (E6 + B)", "tab:blue"), ("exposed (E6)", "tab:red"), ("all", "black")):
        for ax, q, stat in ((axes[0], "gap", "median"), (axes[1], "greedy_directed", "mean")):
            rr = [r for r in bins if r["group"] == gname and r["quantity"] == q and r["n"] >= 20]
            x = [(r["b_lo"] + r["b_hi"]) / 2 for r in rr]
            ax.plot(x, [r[stat] for r in rr], "o-", color=color, label=gname, lw=1.5 if gname != "all" else 1, alpha=1 if gname != "all" else 0.5)
    for agent in sorted({s["agent"] for s in steps}):
        g = [s for s in steps if s["agent"] == agent]
        res = dr.binned(np.array([s["b"] for s in g]), np.array([s["greedy_directed"] for s in g]), edges)
        ok = res["n"] >= 20
        axes[1].plot(((res["lo"] + res["hi"]) / 2)[ok], res["mean"][ok], ":", color="gray", lw=0.8)
    axes[0].set_ylabel("écart de Q médian (meilleure − deuxième)")
    axes[1].set_ylabel("part dirigée du choix greedy (tie set)"), axes[1].set_ylim(0, 1.02)
    axes[1].axhline(0.4, color="gray", lw=0.6, ls="--")
    for ax in axes:
        ax.set_xlabel("b local (bit), pas bien localisés dans la zone"), ax.legend(fontsize=8)
    fig.suptitle(f"D2 — tranches de {bin_width} bit, n ≥ 20 pas ; pointillés gris : chaque agent", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "d2_gap_directed_vs_b.png", dpi=130)
    plt.close(fig)
    return steps, bins, {"identity": identity, "ref_gap_median": g_med, "tau_grid": taus, "n_steps": len(steps)}


# ------------------------------------------------------------------------------------------------------------------ D3
def d3(runs_sm: list[dict], runs_ind: list[dict], figdir: Path) -> tuple[list[dict], dict]:
    pts = []
    for runs, rule in ((runs_ind, "epsilon-greedy"), (runs_sm, "softmax")):
        for run in runs:
            rows = run["summary"]["rows"]
            for r in rows:
                if r["mode"] == "reverse_random":
                    continue
                same = [x for x in rows if x["mode"] in ("none", "reverse_none") and x["model"] == r["model"] and x.get("tau") == r.get("tau")]
                base = same[0] if same else r
                pts.append({"agent": r["agent"], "rule": rule, "tau": r.get("tau"), "model": r["model"], "mode": r["mode"], "alpha": r["alpha"],
                            "b_step1": r["b_step1"], "directed_share": r["directed_share"], "loc_zone": r["loc_template_zone"],
                            "p_goal_zone": r["p_goal_zone"],
                            "passes": bool(r["loc_template_zone"] >= base["loc_template_zone"] - 0.02 and abs(r["p_goal_zone"] - base["p_goal_zone"]) <= 0.05)})
    rules = [("epsilon-greedy", None)] + [("softmax", t) for t in sorted({p["tau"] for p in pts if p["rule"] == "softmax"})]
    stats = {}
    fig, axes = plt.subplots(1, len(rules), figsize=(3.3 * len(rules), 3.8), sharey=True, squeeze=False)
    for ax, (rule, tau) in zip(axes[0], rules):
        sel = [p for p in pts if p["rule"] == rule and p["tau"] == tau]
        b_ref = [p for p in sel if p["model"] == "reference" and p["passes"]]
        key = "eps" if tau is None else f"{tau:g}"
        stats[key] = {"spearman_B_passing": spearman(np.array([p["b_step1"] for p in b_ref]), np.array([p["directed_share"] for p in b_ref])),
                      "n_B_passing": len(b_ref),
                      "reference_directed_alpha0": float(np.mean([p["directed_share"] for p in sel if p["mode"] == "reverse_none"])),
                      "exposed_directed": float(np.mean([p["directed_share"] for p in sel if p["model"] == "exposed"]))}
        for agent in sorted({p["agent"] for p in sel}):
            ref = sorted([p for p in sel if p["agent"] == agent and p["model"] == "reference"], key=lambda p: p["alpha"])
            ax.plot([p["b_step1"] for p in ref], [p["directed_share"] for p in ref], "-", color="tab:blue", lw=0.7, alpha=0.6)
            for p in ref:
                ax.plot(p["b_step1"], p["directed_share"], "^", color="tab:blue", mfc="tab:blue" if p["passes"] else "none", ms=5)
            for p in sel:
                if p["agent"] == agent and p["model"] == "exposed":
                    ax.plot(p["b_step1"], p["directed_share"], "o", color="tab:red", ms=6)
        ax.set_title(("ε-greedy (entraînement)" if tau is None else f"softmax τ = {tau:g}") + f"\nρ(B valides) = {stats[key]['spearman_B_passing']:.2f}", fontsize=9)
        ax.axhline(0.4, color="gray", lw=0.6, ls="--")
        ax.set_xlabel("b au pas 1 (bit)")
    axes[0][0].set_ylabel("part dirigée dans la zone")
    fig.suptitle("D3 — triangles : références sous induction B (pleins : localisation et p_goal tenus) ; ronds : exposés (E6)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "d3_softmax.png", dpi=130)
    plt.close(fig)
    return pts, stats


# ------------------------------------------------------------------------------------------------------------------ D4
def d4(runs_val: list[dict], runs_la: list[dict], runs_ind: list[dict], worlds: dict, figdir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    static, traces, conv = [], [], []
    for run in runs_val:
        a, w = run["meta"]["agent"], worlds[run["meta"]["agent"]]
        v = np.load(run["dir"] / "values.npz")
        cells = v["cells"]
        for key, rows in run["summary"]["variants"].items():
            role, variant = key.split("__")
            Q = v[f"{key}__Q"]
            for k in range(1, Q.shape[-1] + 1):
                static.append({"agent": a, "model": "exposed" if role == "experimental" else "reference", "variant": variant, "k": k,
                               "greedy_directed": float(dr.greedy_directed_share(Q[..., k - 1], w.directed[cells], TIE_TOL).mean()),
                               "argmax_directed": float(dr.argmax_directed(Q[..., k - 1], w.directed[cells]).mean())})
            if f"{key}__trace_b" in v.files:
                tb, dec, act = v[f"{key}__trace_b"], v[f"{key}__trace_decoded"], v[f"{key}__trace_actions"]
                traces.append({"agent": a, "model": static[-1]["model"], "variant": variant, "step": 0, "b_mean": float(np.nanmean(v[f"{key}__b_root"])),
                               "loc_root": float(np.mean(v[f"{key}__loc_root"]))})
                for i in range(tb.shape[0]):  # b_mean_in_zone: the particles decoded in the zone only (b rises outside it by location)
                    inz = w.in_zone[dec[i]]
                    traces.append({"agent": a, "model": static[-1]["model"], "variant": variant, "step": i + 1, "b_mean": float(np.nanmean(tb[i])),
                                   "b_mean_in_zone": float(np.nanmean(tb[i][inz])) if inz.any() else float("nan"), "share_in_zone": float(inz.mean())})
                frm = np.concatenate([np.broadcast_to(cells[:, None, None], dec.shape[1:])[None], dec[:-1]])  # the cell before each step
                ok = w.in_zone[frm]
                conv.append({"agent": a, "model": static[-1]["model"], "variant": variant, "n_transitions": int(ok.sum()),
                             "on_conveyor": float((dec[ok] == w.conveyor[frm[ok]]).mean()),
                             "on_action": float((dec[ok] == w.succ[frm[ok], act[ok]]).mean()),
                             "conveyor_differs_from_action": float((w.conveyor[frm[ok]] != w.succ[frm[ok], act[ok]]).mean()),
                             "on_conveyor_where_differs": float((dec[ok] == w.conveyor[frm[ok]])[w.conveyor[frm[ok]] != w.succ[frm[ok], act[ok]]].mean())})
    episodes = []
    for run in runs_la:
        for r in run["summary"]["rows"]:
            episodes.append({"agent": r["agent"], "model": r["model"], "k": r["k"], "directed_share": r["directed_share"], "success_rate": r["success_rate"],
                             "seconds": r["seconds"]})
    for run in runs_ind:  # E6's own episodes (k = 1, the training's Q) for comparison
        for r in run["summary"]["rows"]:
            if r["mode"] in ("none", "reverse_none"):
                episodes.append({"agent": r["agent"], "model": r["model"], "k": "E6", "directed_share": r["directed_share"], "success_rate": r["success_rate"],
                                 "seconds": r["seconds"]})
    agents = sorted({s["agent"] for s in static})
    fig, axes = plt.subplots(1, len(agents) + 1, figsize=(3.3 * (len(agents) + 1), 3.8), sharey=True, squeeze=False)
    for ax, agent in zip(axes[0], agents):
        for variant, color in VARIANT_STYLE.items():
            for model, ls in (("exposed", "-"), ("reference", "--")):
                rr = [s for s in static if s["agent"] == agent and s["variant"] == variant and s["model"] == model]
                if rr:
                    ax.plot([s["k"] for s in rr], [s["greedy_directed"] for s in rr], ls, color=color, marker="o", ms=3, label=f"{variant}, {model}")
        ax.set_title(agent.replace("_e4", ""), fontsize=9), ax.set_xlabel("k"), ax.set_xscale("log"), ax.set_xticks([1, 2, 3, 5, 10], ["1", "2", "3", "5", "10"])
    ax = axes[0][-1]
    for model, color in (("exposed", "tab:red"), ("reference", "tab:blue")):
        for agent in agents:
            rr = sorted([e for e in episodes if e["agent"] == agent and e["model"] == model and e["k"] != "E6"], key=lambda e: e["k"])
            ax.plot([e["k"] for e in rr], [e["directed_share"] for e in rr], "o-", color=color, lw=0.8, ms=3, label=model if agent == agents[0] else None)
    ax.set_title("épisodes, politique sur Q_k oracle", fontsize=9), ax.set_xlabel("k"), ax.set_xscale("log")
    ax.set_xticks([1, 5, 10], ["1", "5", "10"])
    axes[0][0].set_ylabel("part dirigée (choix greedy, zone AI)")
    axes[0][0].legend(fontsize=6), ax.legend(fontsize=7)
    for a_ in axes[0]:
        a_.axhline(0.4, color="gray", lw=0.6, ls="--"), a_.set_ylim(0, 1.02)
    fig.suptitle("D4 — anticipation à k pas : oracle (vert), modèle (rouge), modèle corrigé à la racine (violet) ; trait plein : exposé, tirets : référence", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d4_lookahead.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for agent in agents:
        for variant in ("model", "model_corrected"):
            for model, ls in (("exposed", "-"), ("reference", "--")):
                rr = [t for t in traces if t["agent"] == agent and t["variant"] == variant and t["model"] == model]
                if rr:
                    ax.plot([t["step"] for t in rr], [t["b_mean"] for t in rr], ls, color=VARIANT_STYLE[variant], marker="o", ms=3, lw=0.8)
    ax.set_xlabel("pas imaginé (0 = racine)"), ax.set_ylabel("b moyen (bit)")
    ax.set_title("D4 — b le long de l'imagination : modèle (rouge), corrigé à la racine (violet)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "d4_b_trace.png", dpi=130)
    plt.close(fig)
    return static, traces, conv, episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--e6-runs", nargs="+", default=[])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    runs = load_runs(args.runs)
    figdir = args.out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    worlds = {r["meta"]["agent"]: World(cfg, r["meta"]) for phase in runs.values() for r in phase}
    summary = {"runs": {phase: [str(r["dir"]) for r in rr] for phase, rr in runs.items()},
               "weights_unchanged": all(r["meta"]["weights_hash_before"] == r["meta"]["weights_hash_after"] for rr in runs.values() for r in rr),
               "seconds": {phase: {r["meta"]["agent"]: r["meta"]["seconds"] for r in rr} for phase, rr in runs.items()}}
    if "values" in runs:
        rows = d1(runs["values"], worlds, figdir)
        write_csv(args.out / "d1_values.csv", rows)
        summary["d1"] = rows
    if "induction" in runs:
        steps, bins, info = d2(runs["induction"], worlds, args.e6_runs, float(cfg["e7"]["diagnostics"]["bin_width"]), figdir)
        write_csv(args.out / "d2_steps.csv", steps)
        write_csv(args.out / "d2_bins.csv", bins)
        summary["d2"] = info
        print(f"D2: reference gap median {info['ref_gap_median']:.4g}; tau grid {info['tau_grid']}")
    if "softmax" in runs or "induction" in runs:
        pts, stats = d3(runs.get("softmax", []), runs.get("induction", []), figdir)
        write_csv(args.out / "d3_points.csv", pts)
        summary["d3"] = stats
    if "values" in runs:
        static, traces, conv, episodes = d4(runs["values"], runs.get("lookahead", []), runs.get("induction", []), worlds, figdir)
        write_csv(args.out / "d4_static.csv", static)
        write_csv(args.out / "d4_b_trace.csv", traces)
        write_csv(args.out / "d4_conveyor.csv", conv)
        write_csv(args.out / "d4_episodes.csv", episodes)
        summary["d4_conveyor"] = conv
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(f"weights unchanged in every run: {summary['weights_unchanged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
