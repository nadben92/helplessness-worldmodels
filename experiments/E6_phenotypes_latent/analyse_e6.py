"""E6's analysis (docs/E6_phenotypes_latent.md, docs/e6/DECISIONS.md): every run of run_e6.py given, read-only.

For each phenotype of e6.analysis.phenotypes, primary then secondary (the logged ones -- the first exit time -- stay in
points.csv only, docs/e6/DECISIONS.md (44)):
  1. main cloud: one point per valid (agent, image) -- the exposed role, and the reference role (b = 0) unless
     e6.analysis.include_references is false; the isotonic fit, its smooth version, the band of a bootstrap over the agents;
  2. by condition: the same points coloured by condition, one curve per condition with enough points;
  3. held out: for each condition, then for each agent, the curve fitted without its points -- residuals, and the blind
     test (b predicted from the phenotype by inverting that curve, against the measured b);
  4. monotonicity: Spearman's rho (agent bootstrap) and the range of b where the curve is strictly monotone (invertible);
  5. localisation: the localisation of every point, the correlation of the main fit's residuals with it, and the curve
     refitted without the poorly localised points (below e6.analysis.localisation_min).
Also: the invalid (agent, image) pairs, the coherence of b at step 1 with E4's probe, the by-index curves (directed share
and b_local by step of the first stay in the zone).

Every computation is the library's (measures.monotone_fit, pipeline.frozen_setup.curve_points); this script reads and
writes. Usage, from the repo root:
  python experiments/E6_phenotypes_latent/analyse_e6.py --runs <e6 run dirs or globs> --out <dir> [--config configs/e6/e6.yaml]
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
from helplessness.measures import localisation_errors as le
from helplessness.measures.zone_phenotypes import PHENOTYPES
from helplessness.envs.layouts import two_zones
from helplessness.pipeline import two_zones_setup as setup
from helplessness.pipeline import frozen_setup as fs

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order (dataviz palette)
LABELS = {"exit_median_km": "first exit time (KM median, steps)", "directed_share": "directed share in the zone",
          "recovery_latency": "recovery latency j_theta (steps)", "zone_share": "share of steps in the zone",
          "stay_share": "stay share in the zone", "revisit_share": "revisit share (k = 4)"}


def read_rows(patterns: list[str]) -> tuple[list[dict], list[str]]:
    rows, dirs = [], []
    for p in patterns:
        for d in sorted(glob.glob(p)) or [p]:
            f = Path(d) / "rows.json"
            if f.is_file():
                rows += json.loads(f.read_text(encoding="utf-8"))
                dirs.append(str(d))
    if not rows:
        raise SystemExit(f"no rows.json under {patterns}")
    return rows, dirs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])


def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def analyse_phenotype(rows, name, an, grid, rng) -> dict:
    pts = fs.curve_points(rows, name, include_references=bool(an["include_references"]))
    out = {"phenotype": name, "n_points": int(pts["b"].size)}
    if pts["b"].size < 3 or np.unique(pts["b"]).size < 2:
        return {**out, "skipped": "fewer than three points"}
    main = mf.curve_summary(pts["b"], pts["y"], pts["agent"], grid, rng, int(an["n_boot"]), float(an["window"]), float(an["min_delta_share"]))
    resid = pts["y"] - mf.predict(main["fit"], pts["b"])
    by_condition = {}
    for c in np.unique(pts["condition"]):
        sel = pts["condition"] == c
        if sel.sum() >= int(an["min_points_per_condition"]) and np.unique(pts["b"][sel]).size >= 2:
            fit = mf.isotonic(pts["b"][sel], pts["y"][sel])
            curve = mf.smooth(fit, grid)
            curve[(grid < pts["b"][sel].min()) | (grid > pts["b"][sel].max())] = np.nan
            by_condition[str(c)] = {"n": int(sel.sum()), "increasing": fit["increasing"], "curve": curve}
        else:
            by_condition[str(c)] = {"n": int(sel.sum()), "skipped": "too few points"}
    held = {"condition": mf.held_out(pts["b"], pts["y"], pts["condition"], grid), "agent": mf.held_out(pts["b"], pts["y"], pts["agent"], grid)}
    loc_ok = pts["loc_step1"] >= float(an["localisation_min"])
    loc = {"threshold": float(an["localisation_min"]), "n_poor": int((~loc_ok).sum()),
           "corr_residual_loc_step1": _pearson(resid, pts["loc_step1"]), "corr_residual_loc_zone": _pearson(resid, pts["loc_zone"])}
    if (~loc_ok).any() and loc_ok.sum() >= 3 and np.unique(pts["b"][loc_ok]).size >= 2:
        fit = mf.isotonic(pts["b"][loc_ok], pts["y"][loc_ok])
        curve = mf.smooth(fit, grid)
        curve[(grid < pts["b"][loc_ok].min()) | (grid > pts["b"][loc_ok].max())] = np.nan
        both = np.isfinite(curve) & np.isfinite(main["curve"])
        loc.update(max_abs_change=float(np.max(np.abs(curve[both] - main["curve"][both]))) if both.any() else float("nan"),
                   spearman_without=mf.spearman(pts["b"][loc_ok], pts["y"][loc_ok]), curve_without=curve)
    return {**out, "points": pts, "main": main, "residual": resid, "by_condition": by_condition, "held": held, "localisation": loc}


def plot_phenotype(res: dict, grid, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, main = res["points"], res["main"]
    conds = sorted(np.unique(pts["condition"]).tolist())
    colour = {c: SERIES[i % len(SERIES)] for i, c in enumerate(conds)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, per_condition in zip(axes, (False, True)):
        ax.grid(color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
        if not per_condition:
            ax.fill_between(grid, main["band_lo"], main["band_hi"], color="#c3c2b7", alpha=0.5, lw=0, label="95% band (agents)")
            ax.plot(main["fit"]["x"], main["fit"]["fitted"], "o", color="#7a7a75", ms=3, label="isotonic fit (at the points)")
            # the smooth curve is solid where a point lies within 0.25 bit, dashed across the gaps (interpolation only)
            near = np.min(np.abs(grid[:, None] - pts["b"][None, :]), axis=1) <= 0.25
            ax.plot(grid, np.where(near, main["curve"], np.nan), color="#2b2b29", lw=2, label="smooth (data within 0.25 bit)")
            ax.plot(grid, np.where(~near, main["curve"], np.nan), color="#2b2b29", lw=1.2, ls="--", label="smooth (no data: interpolated)")
        else:
            for c, d in res["by_condition"].items():
                if "curve" in d:
                    ax.plot(grid, d["curve"], color=colour[c], lw=2)
        for c in conds:
            for role, face in (("exposed", True), ("reference", False)):
                sel = (pts["condition"] == c) & (pts["role"] == role)
                if sel.any():
                    ax.scatter(pts["b"][sel], pts["y"][sel], s=46, facecolors=colour[c] if face else "white", edgecolors=colour[c],
                               linewidths=2, zorder=3, label=f"{c} ({role})")
        ax.set_xlabel("b at step 1 (bit)")
        ax.set_title("main curve" if not per_condition else "by condition", fontsize=10, loc="left")
    axes[0].set_ylabel(LABELS.get(res["phenotype"], res["phenotype"]))
    handles, labels = axes[1].get_legend_handles_labels()
    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(h0[:4] + handles, l0[:4] + labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_by_index(rows: list[dict], path: Path, H: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = sorted({r["image"] for r in rows})
    fig, axes = plt.subplots(2, len(images), figsize=(3.2 * len(images), 5.6), sharex=True, squeeze=False)
    for j, image in enumerate(images):
        for r in (r for r in rows if r["image"] == image):
            ls = "-" if r["role"] == "exposed" else ":"
            n = np.asarray(r["by_index_n"][:H], float)
            share = np.asarray(r["by_index_share"][:H], float)
            share[n < 20] = np.nan
            bl = np.asarray([np.nan if v is None else v for v in r["b_local_by_index"][:H]], float)
            bl[np.asarray(r["b_local_by_index_n"][:H]) < 20] = np.nan
            label = f"{r['agent']} {r['role']}"
            axes[0, j].plot(np.arange(1, H + 1), share, ls, lw=1.5, label=label)
            axes[1, j].plot(np.arange(1, H + 1), bl, ls, lw=1.5, label=label)
        axes[0, j].set_title(image, fontsize=10, loc="left")
        axes[1, j].set_xlabel("step of the first stay in the zone")
        for ax in axes[:, j]:
            ax.grid(color="#e6e6e3", lw=0.6)
    axes[0, 0].set_ylabel("directed share")
    axes[1, 0].set_ylabel("b_local (bit)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def localisation_section(rows: list[dict], dirs: list[str], cfg: dict, out: Path) -> dict:
    """docs/e6/DECISIONS.md (45): the localisation of every (agent, image) next to b and the directed share; the directed
    share against the localisation across the exposed models (valid pairs), and step by step within each agent (steps
    well vs badly localised); the nature of the exposed models' errors at the former UD slot (on the conveyor's cell?).
    Reads each run's episodes.npz (decoded_template: the nearest-template rule, E4's localisation)."""
    exposed = [r for r in rows if r["role"] == "exposed"]
    table = [{k: r.get(k) for k in ("agent", "condition", "image", "valid", "b_step1", "directed_share", "loc_template_step1", "loc_template_zone",
                                    "loc_agent_zone")} for r in rows]
    write_csv(out / "localisation_by_pair.csv", table)
    valid = [r for r in exposed if fs._truthy(r["valid"])]
    res = {"n_valid_exposed": len(valid)}
    for name, sel in (("all_valid", valid), ("b_below_minus1", [r for r in valid if float(r["b_step1"]) < -1.0])):
        d = [float(r["directed_share"]) for r in sel]
        loc = [float(r["loc_template_zone"]) for r in sel]
        res[f"corr_directed_loc_zone_{name}"] = {"n": len(sel), "pearson": _pearson(d, loc), "spearman": mf.spearman(np.array(d), np.array(loc))}
    neighbours = le.torus_neighbours((12, 12))
    dist = fs.goal_distances()
    steps_rows, error_rows = [], []
    for d in dirs:
        npz = Path(d) / "episodes.npz"
        rows_d = json.loads((Path(d) / "rows.json").read_text(encoding="utf-8"))
        if not npz.is_file():
            continue
        arrays = np.load(npz)
        agent = rows_d[0]
        conf = {**setup.configuration(cfg, agent["configuration"]), "seed": agent["seed"]}
        conveyor = setup.successors(setup.world(cfg, conf, "experimental"))[:, 4]  # the conveyor ignores the action
        for r in (r for r in rows_d if r["role"] == "exposed"):
            key = f"{r['image']}__exposed__"
            if key + "decoded_template" not in arrays:
                continue
            zone = np.zeros(144, dtype=bool)
            zone[fs.zone_states(r["slot"])] = True
            st, length, dec = arrays[key + "states"], arrays[key + "length"], arrays[key + "decoded_template"]
            sb = le.steps_by_localisation(st, length, zone, dist, dec)
            steps_rows.append({"agent": r["agent"], "image": r["image"], "valid": r["valid"], "b_step1": r["b_step1"], **sb})
            if r["position"] == "ud":
                regions = np.array([two_zones.region_of_state(two_zones.build_layout(1, "reference"), q) for q in range(144)])
                en = le.error_nature(st, length, zone, dec, conveyor, neighbours, regions=regions)
                error_rows.append({"agent": r["agent"], "image": r["image"], "valid": r["valid"], "b_step1": r["b_step1"], **en})
    write_csv(out / "localisation_steps.csv", steps_rows)
    write_csv(out / "localisation_errors.csv", error_rows)
    res["steps"] = steps_rows
    res["errors"] = error_rows
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "e6" / "e6.yaml")
    parser.add_argument("--seed", type=int, default=6060, help="the bootstrap's generator")
    args = parser.parse_args(argv)
    cfg = fs.load_layered_config(args.config, ("e6", "e7"))
    an = cfg["e6"]["analysis"]
    rows, dirs = read_rows(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    bs = [float(r["b_step1"]) for r in rows]
    step = float(an["grid_step"])
    grid = np.arange(np.floor(min(bs + [0.0]) / step) * step - step, max(bs + [0.0]) + 2 * step, step)
    rng = np.random.default_rng(args.seed)
    exposed = [r for r in rows if r["role"] == "exposed"]
    write_csv(args.out / "invalid_pairs.csv", [{k: r[k] for k in ("agent", "condition", "image", "slot", "mass_ref_step1_probe", "I_ref_step1_probe")}
                                               for r in exposed if not fs._truthy(r["valid"])])
    write_csv(args.out / "coherence.csv", [{k: r[k] for k in ("agent", "condition", "image", "valid", "b_step1", "b_step1_probe", "coherence_gap",
                                                              "b_step1_fresh_ref", "loc_template_step1")} for r in exposed])
    write_csv(args.out / "points.csv", [{k: r[k] for k in ("agent", "condition", "image", "role", "valid", "b_step1", *PHENOTYPES,
                                                           *[f"{p}_lo" for p in PHENOTYPES if f"{p}_lo" in r], *[f"{p}_hi" for p in PHENOTYPES if f"{p}_hi" in r],
                                                           "exit_censored_share", "recovery_censored", "loc_template_step1", "loc_template_zone",
                                                           "success_rate") if k in r} for r in rows])
    summary = {"runs": dirs, "n_rows": len(rows), "n_valid_exposed": sum(fs._truthy(r["valid"]) for r in exposed), "n_exposed": len(exposed),
               "phenotypes": {}}
    curves, held_rows = {"b": grid}, []
    roles = an["phenotypes"]
    for name in [*roles["primary"], *roles["secondary"]]:
        res = analyse_phenotype(rows, name, an, grid, rng)
        if "skipped" in res:
            summary["phenotypes"][name] = res
            continue
        m = res["main"]
        curves[f"{name}__curve"], curves[f"{name}__lo"], curves[f"{name}__hi"] = m["curve"], m["band_lo"], m["band_hi"]
        for kind, items in res["held"].items():
            for it in items:
                if "skipped" in it:
                    held_rows.append({"phenotype": name, "held_out": kind, "group": it["group"], "n": it["n"], "skipped": it["skipped"]})
                    continue
                held_rows.append({"phenotype": name, "held_out": kind, "group": it["group"], "n": it["n"], "rmse_residual": it["rmse_residual"],
                                  "mae_b": it["mae_b"], "n_outside_y_range": int(it["outside_y_range"].sum()),
                                  "n_outside_b_range": int(it["outside_b_range"].sum()), "n_flat": int(it["flat"].sum())})
        loc = {k: v for k, v in res["localisation"].items() if k != "curve_without"}
        summary["phenotypes"][name] = {"primary": name in roles["primary"], "n_points": res["n_points"], "n_agents": m["n_clusters"],
                                       "increasing": m["increasing"], "spearman": m["spearman"], "spearman_ci": m["spearman_ci"],
                                       "invertible_range": m["strict"]["invertible_range"], "invertible_share": m["strict"]["invertible_share"],
                                       # the strict range is read on the smooth curves: across a gap without points it is interpolation
                                       "largest_gap_in_b": float(np.max(np.diff(np.unique(res["points"]["b"])))),
                                       "localisation": loc, "n_censored_points": int(res["points"]["censored"].sum()),
                                       "by_condition": {c: {k: v for k, v in d.items() if k != "curve"} for c, d in res["by_condition"].items()}}
        plot_phenotype(res, grid, args.out / "figures" / f"cloud_{name}.png")
    np.savez_compressed(args.out / "curves.npz", **curves)
    write_csv(args.out / "held_out.csv", held_rows)
    plot_by_index(rows, args.out / "figures" / "by_index.png")
    summary["logged_only"] = list(roles.get("logged", []))
    summary["localisation"] = localisation_section(rows, dirs, cfg, args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                                           encoding="utf-8")
    print(json.dumps({n: {k: v for k, v in s.items() if k in ("n_points", "spearman", "spearman_ci", "invertible_range", "increasing")}
                      for n, s in summary["phenotypes"].items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
