"""E7's analysis (docs/E7_latent_manipulation.md, docs/e6/DECISIONS.md), read-only:

  directions  per agent and task (forward: exposed; reverse: reference): alignment of the corrections (leave-one-out
              cosines against the isotropic null), efficacy of the common direction on the test half, stability of the
              optimum -- from run_e7_directions.py's summary.json, tabulated, never judged;
  identity    alpha = 0 (mode none) against E6's rows of the same (agent, image): b at step 1 and every phenotype must be
              identical (the same function and seeds; on the same machine and device);
  checks      per intervention against alpha = 0: the belief moved (b), nothing else moved (the localisation at step 1 and in
              the zone, the mass on the true successor of the played action, p_goal), the random control, the reverse
              direction on the reference;
  curves      E6's main curve of every phenotype (refitted from E6's rows with E6's analysis settings and seed: the same
              curve as analyse_e6.py's) with E7's points over it: their residuals from it (points outside E6's range of b
              flagged), by mode and alpha.

Usage, from the repo root:
  python experiments/E7_latent_manipulation/analyse_e7.py --e6-runs <dirs/globs> --apply-runs <dirs/globs> --directions-runs <dirs/globs>
      --out <dir> [--config configs/e7/e7.yaml]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

from helplessness.config import REPO_ROOT
from helplessness.measures import monotone_fit as mf
from helplessness.measures.zone_phenotypes import PHENOTYPES
from helplessness.pipeline import frozen_setup as fs

MODE_COLOUR = {"direction": "#2a78d6", "random": "#eb6834", "stepwise": "#1baf7a", "reverse": "#eda100", "reverse_random": "#e87ba4"}
CHECKS = ("b_step1", "loc_template_step1", "loc_template_zone", "mass_played_step1", "mass_played_zone", "p_goal_step1", "p_goal_zone",
          "directed_share", "exit_median_km", "recovery_latency")


def read(patterns: list[str], name: str) -> list[tuple[Path, object]]:
    out = []
    for p in patterns:
        for d in sorted(glob.glob(p)) or [p]:
            f = Path(d) / name
            if f.is_file():
                out.append((Path(d), json.loads(f.read_text(encoding="utf-8"))))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r)) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])


def same(a, b) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return (np.isnan(fa) and np.isnan(fb)) or fa == fb


def induction_checks(rows: list[dict]) -> list[dict]:
    """Plan B, per (agent, image): b at step 1, the directed share, the localisation and p_goal along alpha for the reverse
    direction and its random control, against the reference without intervention (reverse_none); whether b decreases
    monotonically with alpha (no rise larger than 0.02 bit between consecutive alphas)."""
    out = []
    keys = sorted({(r["agent"], r["image"]) for r in rows if r.get("plan") == "B"})
    for agent, image in keys:
        base = next((r for r in rows if (r["agent"], r["image"], r["mode"]) == (agent, image, "reverse_none")), None)
        for mode in ("reverse", "reverse_random"):
            seq = sorted((r for r in rows if (r["agent"], r["image"], r["mode"], r.get("plan")) == (agent, image, mode, "B")), key=lambda r: r["alpha"])
            if not seq or base is None:
                continue
            seq = [base] + seq
            b = [float(r["b_step1"]) for r in seq]
            out.append({"agent": agent, "image": image, "mode": mode, "alphas": [float(r["alpha"]) for r in seq], "b_step1": b,
                        "b_monotone_decreasing": bool(all(b[i + 1] <= b[i] + 0.02 for i in range(len(b) - 1))),
                        "directed_share": [float(r["directed_share"]) for r in seq],
                        "loc_template_step1": [float(r["loc_template_step1"]) for r in seq], "loc_template_zone": [float(r["loc_template_zone"]) for r in seq],
                        "p_goal_zone": [float(r["p_goal_zone"]) for r in seq], "delta_norm_zone": [float(r["delta_norm_zone"]) for r in seq]})
    return out


def constrained_table(patterns: list[str], out: Path) -> list[dict]:
    """Manipulation A's sweep (run_e7_constrained.py), every agent: one row per setting, and the combined trade-off figure."""
    rows, lines = [], {}
    for d, s in read(patterns, "settings.json"):
        for r in s["settings"]:
            rows.append(r)
        for image, e in s["images"].items():
            lines[(s["agent"], image)] = e
    if not rows:
        return []
    write_csv(out / "constrained_settings.csv", rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = sorted({r["agent"] for r in rows})
    fig, axes = plt.subplots(1, len(agents), figsize=(4.2 * len(agents), 4.2), sharey=True, squeeze=False)
    for ax, agent in zip(axes[0], agents):
        ax.grid(color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        sel = [r for r in rows if r["agent"] == agent]
        for r in sel:
            ax.scatter(r["gain_b"], r["loc_after"], s=44, color="#2a78d6" if r["kept"] else "#7a7a75", zorder=3)
            ax.scatter(r["gain_b_random"], r["loc_random"], s=34, facecolors="white", edgecolors="#eb6834", linewidths=1.5, zorder=3)
            ax.annotate(f"{r['lam']:g}/{r['mu_pos']:g}", (r["gain_b"], r["loc_after"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(sel[0]["loc_before"], color="#2b2b29", ls="--", lw=1)
        ax.axhline(sel[0]["loc_reference"], color="#1baf7a", ls=":", lw=1.5)
        ax.set_title(agent, fontsize=9, loc="left")
        ax.set_xlabel("gain of b (bit)")
    axes[0, 0].set_ylabel("localisation under intervention")
    handles = [plt.Line2D([], [], marker="o", ls="", color="#2a78d6", label="kept"), plt.Line2D([], [], marker="o", ls="", color="#7a7a75", label="not kept"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#eb6834", label="random control"),
               plt.Line2D([], [], color="#2b2b29", ls="--", label="exposed, no intervention"), plt.Line2D([], [], color="#1baf7a", ls=":", label="reference")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "figures" / "constrained_tradeoff.png", dpi=130)
    plt.close(fig)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e6-runs", nargs="+", required=True)
    parser.add_argument("--apply-runs", nargs="+", required=True)
    parser.add_argument("--directions-runs", nargs="+", default=[])
    parser.add_argument("--constrained-runs", nargs="+", default=[], help="run_e7_constrained.py outputs (manipulation A's sweep)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e7" / "e7.yaml")
    parser.add_argument("--seed", type=int, default=6060, help="E6's analysis seed (analyse_e6.py --seed): the same curves")
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    an = cfg["e6"]["analysis"]
    e6_rows = [r for _, rows in read(args.e6_runs, "rows.json") for r in rows]
    e7_rows = [r for _, rows in read(args.apply_runs, "rows.json") for r in rows]
    if not e6_rows or not e7_rows:
        raise SystemExit("no E6 or no E7 rows")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    # directions
    dir_rows = []
    for d, s in read(args.directions_runs, "summary.json"):
        for task, t in s["tasks"].items():
            if "skipped" in t:
                dir_rows.append({"agent": s["agent"], "task": task, "skipped": t["skipped"]})
                continue
            al, ef, st = t["alignment"], t.get("efficacy_test", {}), t.get("stability", {})
            dir_rows.append({"agent": s["agent"], "condition": s["condition"], "task": task, "model": t["model"], "n_estimation": t["n_estimation"],
                             "mean_delta_norm": t["mean_delta_norm"], "v_norm": t["v_norm"], "cos_loo_mean": al["cos_mean"], "cos_loo_median": al["cos_median"],
                             "null_cos_mean": al["null_cos_mean"], "null_cos_p95": al["null_cos_p95"], "null_pair_sd": al["null_pair_sd_analytic"],
                             "efficacy_ratio_of_means": ef.get("ratio_of_means"), "efficacy_median_ratio": ef.get("median_ratio"),
                             "db_direction_mean": ef.get("db_direction_mean"), "db_optimal_mean": ef.get("db_optimal_mean"),
                             "random_db_mean": t.get("random_db_mean_test"), "stability_cos_mean": st.get("cos_mean"),
                             "stability_norm_sd": st.get("norm_sd"), "stability_norm_mean": st.get("norm_mean"),
                             "by_image": {k: {kk: v.get(kk) for kk in ("n_localised", "n_contexts", "I0_mean", "I_opt_mean", "cos_with_v", "steps", "converged")}
                                          for k, v in t["by_image"].items()}})
    write_csv(args.out / "directions.csv", dir_rows)

    # identity: alpha = 0 against E6
    e6_by = {(r["agent"], r["image"], r["role"]): r for r in e6_rows}
    ident = []
    for r in e7_rows:
        if r["mode"] not in ("none", "reverse_none"):
            continue
        ref = e6_by.get((r["agent"], r["image"], "exposed" if r["mode"] == "none" else "reference"))
        if ref is None:
            ident.append({"agent": r["agent"], "image": r["image"], "mode": r["mode"], "identical": None, "note": "no E6 row"})
            continue
        fields = ["b_step1", "I_step1", *PHENOTYPES, "exit_censored_share", "loc_template_step1", "success_rate"]
        diffs = [f for f in fields if f in ref and not same(r[f], ref[f])]
        ident.append({"agent": r["agent"], "image": r["image"], "mode": r["mode"], "identical": not diffs, "differs": diffs})
    write_csv(args.out / "identity_alpha0.csv", ident)

    # checks against alpha = 0
    base = {(r["agent"], r["image"], "exposed"): r for r in e7_rows if r["mode"] == "none"}
    base.update({(r["agent"], r["image"], "reference"): r for r in e7_rows if r["mode"] == "reverse_none"})
    checks = []
    for r in e7_rows:
        if r["mode"] in ("none", "reverse_none"):
            continue
        b0 = base.get((r["agent"], r["image"], r["model"]))
        if b0 is None:
            continue
        checks.append({"agent": r["agent"], "condition": r["condition"], "image": r["image"], "mode": r["mode"], "alpha": r["alpha"],
                       "delta_norm_zone": r["delta_norm_zone"], **{f"{c}": r.get(c) for c in CHECKS},
                       **{f"d_{c}": (float(r[c]) - float(b0[c])) if c in r and c in b0 else None for c in CHECKS}})
    write_csv(args.out / "checks.csv", checks)

    # E7 points on E6's curves, by group: A (the constrained correction's kept settings), B (the induction in the references,
    # reverse), v1 (the first report's modes)
    roles = an["phenotypes"]
    valid = {(r["agent"], r["image"]): fs._truthy(r["valid"]) for r in e6_rows if r["role"] == "exposed"}
    bs = [float(r["b_step1"]) for r in e6_rows] + [float(r["b_step1"]) for r in e7_rows]
    step = float(an["grid_step"])
    grid = np.arange(np.floor(min(bs + [0.0]) / step) * step - step, max(bs + [0.0]) + 2 * step, step)
    rng = np.random.default_rng(args.seed)
    resid_rows, summary = [], {"identity_all": all(i.get("identical") for i in ident) if ident else None, "phenotypes": {}}

    def group_of(r) -> str | None:
        plan = r.get("plan") or "v1"
        if r["mode"] in ("none", "reverse_none"):
            return None
        if plan == "A":
            return f"A {r['mode']}"
        if plan == "B":
            return "B" if r["mode"] == "reverse" else "B random"
        return f"v1 {r['mode']}"

    for name in [*roles["primary"], *roles["secondary"]]:
        pts = fs.curve_points(e6_rows, name, include_references=bool(an["include_references"]))
        if pts["b"].size < 3:
            continue
        main = mf.curve_summary(pts["b"], pts["y"], pts["agent"], grid, rng, int(an["n_boot"]), float(an["window"]), float(an["min_delta_share"]))
        for r in e7_rows:
            g = group_of(r)
            if g is None or not valid.get((r["agent"], r["image"]), False):
                continue
            y = float(r[name]) if not (name == "recovery_latency" and fs._truthy(r.get("recovery_censored"))) else float(r.get("recovery_max_index_read", np.nan)) + 1
            b = float(r["b_step1"])
            pred = float(mf.predict(main["fit"], [b])[0])
            resid_rows.append({"phenotype": name, "group": g, "agent": r["agent"], "image": r["image"], "mode": r["mode"], "setting": r.get("setting"),
                               "alpha": r["alpha"], "b_step1": b, "y": y, "e6_curve": pred, "residual": y - pred,
                               "loc_template_step1": r.get("loc_template_step1"), "loc_template_zone": r.get("loc_template_zone"),
                               "p_goal_zone": r.get("p_goal_zone"), "outside_e6_b_range": bool(b < pts["b"].min() or b > pts["b"].max())})
        sub = [x for x in resid_rows if x["phenotype"] == name]
        groups = sorted({x["group"] for x in sub})
        summary["phenotypes"][name] = {"primary": name in roles["primary"], "by_group": {
            g: {"n": len(v), "rmse_residual": float(np.sqrt(np.mean([x["residual"] ** 2 for x in v]))) if v else None,
                "mean_residual": float(np.mean([x["residual"] for x in v])) if v else None}
            for g in groups for v in [[x for x in sub if x["group"] == g and np.isfinite(x["residual"])]]},
            "e6_points_rmse_residual": float(np.sqrt(np.mean((pts["y"] - mf.predict(main["fit"], pts["b"])) ** 2)))}
        if name in roles["primary"]:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.grid(color="#e6e6e3", lw=0.6)
            ax.set_axisbelow(True)
            ax.fill_between(grid, main["band_lo"], main["band_hi"], color="#c3c2b7", alpha=0.5, lw=0, label="E6 band")
            ax.plot(grid, main["curve"], color="#2b2b29", lw=2, label="E6 curve")
            ax.scatter(pts["b"], pts["y"], s=30, color="#7a7a75", zorder=3, label="E6 points")
            style = {"A constrained_direction": ("s", "#2a78d6"), "A constrained_stepwise": ("s", "#1baf7a"), "B": ("^", "#eda100")}
            for g, (marker, colour) in style.items():
                v = [x for x in sub if x["group"] == g]
                if v:
                    ax.scatter([x["b_step1"] for x in v], [x["y"] for x in v], marker=marker, s=46, facecolors="white", edgecolors=colour,
                               linewidths=2, zorder=4, label=f"E7 {g}")
            ax.set_xlabel("b at step 1 (bit)")
            ax.set_ylabel(name)
            ax.legend(fontsize=7, frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(args.out / "figures" / f"e7_on_e6_{name}.png", dpi=130)
            plt.close(fig)
    summary["induction"] = induction_checks(e7_rows)
    summary["constrained"] = constrained_table(args.constrained_runs, args.out)
    write_csv(args.out / "residuals.csv", resid_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
