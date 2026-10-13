"""The E3 campaign's analysis pipeline, in one command (docs/e3/DECISIONS.md, 2026-10-13 (31); the analyses of (29),
(30) and (31), every criterion recorded before the campaign's data were read).

    python experiments/E3_worldmodel_4zones/campaign_report.py --handoff handoff/gpu_campaign
        [--work results/E3_worldmodel_4zones/campaign] [--no-rsa] [--append-report docs/E3_worldmodel_4zones.md]

1. integrity (campaign_integrity.py): the handoff is refused unless complete; its archives extracted into <work>/runs/;
2. analyses: main and secondary (campaign_analysis.py), propagation (D), learning and behaviour (B), measurement error
   in the noisy zones (C), convergence of the deterministic zones (E) (campaign_learning.py);
3. RSA of every run at every measured checkpoint on the CPU (rsa_latents.py, measures.rsa), unless --no-rsa;
4. figures (campaign_figures.py) into <work>/figures/; tables (CSV) into <work>/tables/; everything in <work>/analysis.json;
5. RESULTS_SECTION.md in <work>: the results section in English, with the verdict of every recorded rule;
   --append-report FILE appends it to FILE between markers (replaced if already there).

--runs-dir DIR instead of --handoff: run directories already laid out as campaign runs (development and tests: the
pilot's archives). The handoff is then NOT checked, and every output says so in its first line.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module

ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "campaign_report_analysis")
cl = load_experiment_module("E3_worldmodel_4zones", "campaign_learning.py", "campaign_report_learning")
ci = load_experiment_module("E3_worldmodel_4zones", "campaign_integrity.py", "campaign_report_integrity")
cf = load_experiment_module("E3_worldmodel_4zones", "campaign_figures.py", "campaign_report_figures")

MARK_START, MARK_END = "<!-- e3-campaign-results:start -->", "<!-- e3-campaign-results:end -->"
ZONE_SHORT = {"uncontrollable_deterministic": "UD", "controllable_noisy": "CN", "uncontrollable_noisy": "UN",
              "controllable_deterministic": "CD", "corridor": "corridor"}


def _f(x, nd: int = 4) -> str:
    try:
        return "NaN" if x is None or not np.isfinite(float(x)) else f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(d: dict, nd: int = 4) -> str:
    return f"{_f(d['mean'], nd)} [{_f(d['ci95'][0], nd)}, {_f(d['ci95'][1], nd)}]"


def run_rsa(pairs: list, work: Path) -> dict:
    """RSA of every run at its measured checkpoints (weights present), and the per-checkpoint summaries."""
    rl = load_experiment_module("E3_worldmodel_4zones", "rsa_latents.py", "campaign_report_rsa")
    out = {"runs": [], "seconds_per_checkpoint": []}
    for exp, ref in pairs:
        for run in (exp, ref):
            if not (run.path / "replay_final.npz").exists():
                out["runs"].append({"seed": run.seed, "condition": run.condition, "skipped": "no weights / replay in the run directory"})
                continue
            summaries = rl.rsa_run(run.path, sorted(run.measure_dirs), work / "rsa")
            out["seconds_per_checkpoint"] += [sum(s["seconds"].values()) for s in summaries]
            out["runs"].append({"seed": run.seed, "condition": run.condition, "checkpoints": summaries})
    return out


def rsa_over_seeds(rsa_out: dict) -> dict:
    """Per common checkpoint and latent: env, ref and env - ref over the seeds of R^2, the controllability unique share
    and the confusion ratios (descriptive)."""
    runs = [r for r in rsa_out["runs"] if "checkpoints" in r]
    if not runs:
        return {}
    by = {(r["seed"], r["condition"]): {s["env_step"]: s for s in r["checkpoints"]} for r in runs}
    seeds = sorted({s for s, _ in by if (s, "experimental") in by and (s, "reference") in by})
    if not seeds:
        return {}
    common = sorted(set.intersection(*[set(by[(s, c)]) for s in seeds for c in ("experimental", "reference")]))
    out = {}
    for c in common:
        out[c] = {}
        for lat in ("h", "z"):
            def q(summary, kind):
                L = summary["states"]["latents"][lat]
                if kind == "r2":
                    return L["regression"]["r2"]
                if kind == "unique_controllability":
                    return L["regression"]["unique_r2"].get("controllability", float("nan"))
                return L["ratios"][kind]["ratio"]
            out[c][lat] = {}
            for kind in ("r2", "unique_controllability", "controllable_vs_uncontrollable", "CD_vs_UD", "CN_vs_UN"):
                e = [q(by[(s, "experimental")][c], kind) for s in seeds]
                r = [q(by[(s, "reference")][c], kind) for s in seeds]
                out[c][lat][kind] = {"env": ca.seeds_ci(e), "ref": ca.seeds_ci(r), "difference": ca.seeds_ci([a - b for a, b in zip(e, r)])}
            # the partition of the explained variance and the predictors' collinearity, per condition (means over seeds)
            out[c][lat]["partition"] = {}
            for cond in ("experimental", "reference"):
                regs = [by[(s, cond)][c]["states"]["latents"][lat]["regression"] for s in seeds]
                names = sorted({k for g in regs for k in g.get("unique_r2", {})})
                out[c][lat]["partition"][cond] = {
                    "unique": {k: float(np.mean([g["unique_r2"].get(k, np.nan) for g in regs])) for k in names},
                    "shared": float(np.mean([g["variance_partition"]["shared"] for g in regs])),
                    "vif": {k: float(np.max([g["vif"].get(k, np.nan) for g in regs])) for k in names},
                    "correlations": {k: float(np.mean([g["predictor_correlations"][k] for g in regs]))
                                     for k in sorted({k for g in regs for k in g.get("predictor_correlations", {})})},
                    "removed": sorted({k for s in seeds for k in by[(s, cond)][c]["states"].get("removed_predictors", {})})}
    return out


def write_tables(res: dict, lr: dict, rsa_seeds: dict, tables: Path) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "b_alpha_by_checkpoint.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "env_step", "zone", "b_I_s", "b_I_s_projected", "alpha_env", "alpha_ref", "alpha_difference", "did"])
        for p in res["secondary"]:
            for c in p["checkpoints"]:
                for z in ca.ZONES:
                    w.writerow([p["seed"], c["env_step"], z, c["b_I_s_all"][z], c["b_I_s_projected"].get(z, ""), c["alpha"][z]["env"],
                                c["alpha"][z]["ref"], c["alpha"][z]["difference"], c["difference_of_differences"].get(z, {}).get("did", "")])
    with open(tables / "learning.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "condition", "env_step", "zone", "twin_l1", "c_b_l1_wm", "c_b_l1_twinB", "c_b_ok", "mass_true_successor",
                    "agent_delta", "c_a", "c_c", "success_interval", "success_lo", "success_hi", "success_last100", "episode_length_last100",
                    "median_goal_mass"])
        for r in lr["learning"]:
            for c in r["checkpoints"]:
                cal = c.get("calibration")
                for z in ca.ZONES:
                    w.writerow([r["seed"], r["condition"], c["env_step"], z, (c.get("twin_l1") or {}).get(z, ""),
                                cal["c_b"][z]["l1_wm"] if cal else "", cal["c_b"][z]["l1_b"] if cal else "", cal["c_b"][z]["ok"] if cal else "",
                                (c.get("mass_on_true_successor") or {}).get(z, ""), (c.get("agent_delta_by_zone") or {}).get(z, ""),
                                cal["c_a"]["value"] if cal else "", cal["c_c"]["value"] if cal else "", *c["interval"]["rate_ci95"],
                                c["success_last100"], c["episode_length_last100"], c["median_goal_mass"]])
    with open(tables / "noise_error.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "condition", "env_step", "zone", "I_s", "I_true", "excess"])
        for r in lr["noise_error"]["per_run"]:
            for c, zs in r["checkpoints"].items():
                for z, v in zs.items():
                    w.writerow([r["seed"], r["condition"], c, z, v["I_s"], v["I_true"], v["excess"]])
    with open(tables / "rsa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["env_step", "latent", "quantity", "env_mean", "ref_mean", "difference_mean", "difference_lo", "difference_hi"])
        for c, lats in rsa_seeds.items():
            for lat, qs in lats.items():
                for k, v in qs.items():
                    if k == "partition":  # in the section and analysis.json
                        continue
                    w.writerow([c, lat, k, v["env"]["mean"], v["ref"]["mean"], v["difference"]["mean"], *v["difference"]["ci95"]])


def section(res: dict, lr: dict, rsa_out: dict | None, rsa_seeds: dict, integrity_md: str, figures: list[Path], work: Path, mode: str) -> str:
    """The results section, in English, with every recorded rule's verdict."""
    L = [MARK_START, "", "## Results of the campaign (gpu_campaign) -- generated by campaign_report.py", ""]
    if mode != "handoff":
        L += [f"**NOT THE CAMPAIGN'S HANDOFF: {mode}. Development output; no verdict below is a campaign result.**", ""]
    L += [f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Rules and definitions: docs/e3/DECISIONS.md (16), (29), "
          "(30), (31), recorded before the data. With N = 4 everything is descriptive; a null result is a bound, not an absence.", ""]
    L += [integrity_md.replace("## Integrity", "### Integrity"), ""]
    # main
    agg = res["main"]["aggregate_b"]
    rule = agg.get("campaign_rule", {})
    L += ["### Main: b by zone at the budget (300,000), hybrid-code rule (29), aberration control (16)", "",
          f"Hybrid-code rule: **{rule.get('status')}**" + (f" (bound +-{rule['bound']:.4f} bit)." if rule.get("bound") is not None else "."), "",
          "| zone | b(I_s) - b(I_s projected), mean [95 %] | off-support env - ref, mean [95 %] | verdict |", "|---|---|---|---|"]
    for z, v in rule.get("zones", {}).items():
        L.append(f"| {ZONE_SHORT[z]} | {_ci(v['b_difference'])} | {_ci(v['off_support_difference'])} | {v.get('verdict', 'descriptive')} |")
    flags = [(p["seed"], z, f) for p in agg["pairs"] for z, v in p["bias_control"].items() for f in v.get("aberrations", [])]
    L += ["", f"Aberration flags (to be explained, never exclusions): {len(flags)}" + ("" if not flags else ":"), ""]
    L += [f"- seed {s}, {ZONE_SHORT[z]}: {f}" for s, z, f in flags]
    over = res["over_seeds"]
    if ca.BUDGET_ENV_STEPS in over.get("b_I_s", {}):
        L += ["", "| zone | b(I_s) at 300,000, mean [95 %] over the seeds | bound of a null | alpha_env - alpha_ref [95 %] |", "|---|---|---|---|"]
        for z in ca.ZONES:
            b = over["b_I_s"][ca.BUDGET_ENV_STEPS][z]
            a = over["alpha_difference"][ca.BUDGET_ENV_STEPS].get(z)
            L.append(f"| {ZONE_SHORT[z]} | {_ci(b)} | {_f(b['bound'])} | {_ci(a) if a and a['n'] else 'not defined'} |")
        L += ["", "**Expectation (29)/(31): in the controllable zones b ~ 0 and alpha_env - alpha_ref ~ 0 at the budget; a transient "
              "early effect possible.**", ""]
        for z, e in over.get("expectation_at_budget", {}).items():
            L.append(f"- {ZONE_SHORT[z]}: b {e['b']} (bound {_f(e['b_bound'])} bit); alpha difference {e['alpha_difference']}"
                     + (f" (bound {_f(e['alpha_bound'])})" if e['alpha_bound'] is not None else "")
                     + f"; earlier checkpoints whose interval excludes 0: {e['early_checkpoints_excluding_0'] or 'none'}")
    # b and alpha over checkpoints
    L += ["", "### b and alpha_env - alpha_ref by zone at every checkpoint measured in every pair (mean [95 %] over the seeds)", "",
          "| checkpoint | " + " | ".join(f"b {ZONE_SHORT[z]}" for z in ca.ZONES) + " | " + " | ".join(f"dalpha {ZONE_SHORT[z]}" for z in ca.CONTROLLABLE_ZONES) + " |",
          "|---|" + "---|" * (len(ca.ZONES) + len(ca.CONTROLLABLE_ZONES))]
    for c in over.get("common_checkpoints", []):
        L.append(f"| {c:,} | " + " | ".join(_ci(over["b_I_s"][c][z]) for z in ca.ZONES) + " | "
                 + " | ".join(_ci(over["alpha_difference"][c][z]) for z in ca.CONTROLLABLE_ZONES) + " |")
    # propagation
    prop = res.get("propagation") or {}
    s = prop.get("seeds", {})
    L += ["", "### Propagation (31) D", "", "**b at equal visits** (per state, interpolated in log n; regression d = b0 + b1 dtau + b2 log n, "
          "dtau the difference of training time at equal visits; b0: b at equal visits and equal training time):", "",
          "| zone | b0, mean [95 %] over seeds | b1 (dtau) | states used per seed |", "|---|---|---|---|"]
    for z in ca.CONTROLLABLE_ZONES:
        if z in s.get("equal_visits_b0", {}):
            used = [p["equal_visits"][z]["n_states"] for p in prop["pairs"]]
            L.append(f"| {ZONE_SHORT[z]} | {_ci(s['equal_visits_b0'][z])} | {_ci(s['equal_visits_b1_dtau'][z])} | {used} |")
    L += ["", f"**b against the distance to the nearest uncontrollable zone** (Spearman rho over the zone's states; main zone CD). "
          "Expected shape of a propagation by position, recorded before: |b| larger near the boundary, decreasing with the distance, "
          "in the environment only, rather early after the switch. The thresholds (95 % t intervals over 4 seeds) are descriptive.", "",
          "| checkpoint | zone | rho(abs b, distance) | rho(env error) | rho(ref error) | verdict |", "|---|---|---|---|---|---|"]
    for c, zs in s.get("distance", {}).items():
        for z in ca.CONTROLLABLE_ZONES:
            v = zs[z]
            L.append(f"| {c:,} | {ZONE_SHORT[z]}{' (main)' if z == ca.DISTANCE_MAIN_ZONE else ''} | {_ci(v['rho_abs_b'], 3)} | "
                     f"{_ci(v['rho_env_error'], 3)} | {_ci(v['rho_ref_error'], 3)} | {v['verdict']} |")
    L += ["", "**Useful generalisation** (median L1_WM - L1_twin against P over the pairs with n <= 10; rule: negative in every seed). "
          "Descriptive: \"negative in all 4 seeds\" is a sign test at p = 1/16 (one-sided).", "",
          "| checkpoint | condition | " + " | ".join(ZONE_SHORT[z] for z in ca.ZONES) + " |", "|---|---|" + "---|" * len(ca.ZONES)]
    for c, conds in s.get("generalisation", {}).items():
        for cond, zs in conds.items():
            L.append(f"| {c:,} | {cond} | " + " | ".join(zs[z] if isinstance(zs[z], str) else zs[z]["verdict"].split(" (")[0] for z in ca.ZONES) + " |")
    # noise error
    ne = lr["noise_error"]
    L += ["", "### Measurement error in the noisy zones (31) C", "",
          "| checkpoint | zone | excess env, mean (sd) | excess ref, mean (sd) | b, mean (sd over seeds) | resolution (95 % half-width, N = 4) |",
          "|---|---|---|---|---|---|"]
    for c, zs in ne["over_seeds"].items():
        for z, v in zs.items():
            L.append(f"| {c:,} | {ZONE_SHORT[z]} | {_f(v['excess_experimental']['mean'])} ({_f(v['excess_experimental']['sd'])}) | "
                     f"{_f(v['excess_reference']['mean'])} ({_f(v['excess_reference']['sd'])}) | {_f(v['b']['mean'])} ({_f(v['b']['sd_over_seeds'])}) | "
                     f"{_f(v['b']['resolution_half_width_95'])} |")
    L += ["", f"Hybrid-code rule (29): {ne['hybrid_rule']['status']}; " + "; ".join(f"{ZONE_SHORT[z]} {v.get('verdict') or 'descriptive'}"
                                                                                  for z, v in ne["hybrid_rule"]["zones"].items())]
    # convergence
    cv = lr["convergence"]
    L += ["", "### Convergence of the deterministic zones (31) E", "", f"**{cv.get('verdict', cv.get('status'))}**", ""]
    if cv.get("status") == "computed":
        L += [f"sigma by zone (x4): " + ", ".join(f"{ZONE_SHORT[z]} {v:.4f}" for z, v in cv["sigma"].items()) + f"; family-wise alpha {cv['family_alpha']}.", "",
              "| candidate | comparisons | k | thresholds (UD / CD / corridor) | (a) agent finished | (b) I_s stable | first failures |", "|---|---|---|---|---|---|---|"]
        for r in cv["table"]:
            th = " / ".join(f"{r['thresholds'][z]:.4f}" for z in cl.DETERMINISTIC_FOR_CONVERGENCE)
            L.append(f"| {r['checkpoint']:,} | {r['m']} | {r['k']:.2f} | {th} | {'yes' if r['a'] else 'no'} | {'yes' if r['b'] else 'no'} | "
                     f"{'; '.join(r['failures'][:3]) or '-'}{' ...' if len(r['failures']) > 3 else ''} |")
    # learning summary at the budget
    L += ["", "### Learning and behaviour at the budget (31) B (all checkpoints: tables/learning.csv, figures/curves_learning.png)", "",
          "| run | C-a | C-c | C-b ok by zone | twin L1 by zone | mass on true successor (UD / CD / corridor) | success last interval [95 %] | max Delta | median goal mass |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in lr["learning"]:
        last = r["checkpoints"][-1]
        cal = last.get("calibration")
        if cal is None:
            continue
        cb = " ".join(f"{ZONE_SHORT[z]}:{'ok' if cal['c_b'][z]['ok'] else 'FAIL'}" for z in ca.ZONES)
        tw = " ".join(f"{ZONE_SHORT[z]}:{_f(last['twin_l1'][z], 3)}" for z in ca.ZONES) if last.get("twin_l1") else "-"
        ms = " / ".join(_f(last["mass_on_true_successor"][z], 3) for z in cl.DETERMINISTIC_FOR_CONVERGENCE)
        rate, lo, hi = last["interval"]["rate_ci95"]
        dmax = max((last.get("agent_delta_by_zone") or {"-": float("nan")}).values())
        L.append(f"| seed {r['seed']} {r['condition']} | {cal['c_a']['value']:.3f} | {cal['c_c']['value']:.3f} | {cb} | {tw} | {ms} | "
                 f"{rate:.3f} [{lo:.3f}, {hi:.3f}] | {_f(dmax, 2)} | {_f(last['median_goal_mass'], 3)} |")
    # RSA
    L += ["", "### RSA of the latent (31) F", ""]
    if rsa_out is None:
        L.append("Not run (--no-rsa).")
    else:
        sp = rsa_out["seconds_per_checkpoint"]
        if sp:
            L.append(f"CPU time per checkpoint: median {np.median(sp):.0f} s ({len(sp)} checkpoints).")
        skipped = [r for r in rsa_out["runs"] if "skipped" in r]
        L += [f"Skipped: seed {r['seed']} {r['condition']} ({r['skipped']})" for r in skipped]
        if rsa_seeds:
            L += ["", "| checkpoint | latent | R^2 env / ref | controllability unique share env | confusion ratio (C vs U) env / ref, env - ref [95 %] | CD vs UD env - ref | CN vs UN env - ref |",
                  "|---|---|---|---|---|---|---|"]
            for c, lats in rsa_seeds.items():
                for lat, q in lats.items():
                    L.append(f"| {c:,} | {lat} | {_f(q['r2']['env']['mean'], 3)} / {_f(q['r2']['ref']['mean'], 3)} | {_f(q['unique_controllability']['env']['mean'], 3)} | "
                             f"{_f(q['controllable_vs_uncontrollable']['env']['mean'], 3)} / {_f(q['controllable_vs_uncontrollable']['ref']['mean'], 3)}, "
                             f"{_ci(q['controllable_vs_uncontrollable']['difference'], 3)} | {_ci(q['CD_vs_UD']['difference'], 3)} | {_ci(q['CN_vs_UN']['difference'], 3)} |")
            L += ["", "**Partition of the explained variance and collinearity of the predictors** (state level; unique share of each "
                  "theoretical matrix, shared share, largest variance inflation factor over the seeds, mean correlations between "
                  "predictors). No regression weight is read in isolation. In E3 theme, position and zone are confounded (each theme "
                  "occupies one fixed region of the torus): E3 cannot separate theme from position -- the object of the next step "
                  "(two-zone models on permuted themes and positions).", "",
                  "| checkpoint | latent | run | unique shares | shared | VIF (max) | predictor correlations | removed (constant) |",
                  "|---|---|---|---|---|---|---|---|"]
            for c, lats in rsa_seeds.items():
                for lat, q in lats.items():
                    for cond, pt in q["partition"].items():
                        L.append(f"| {c:,} | {lat} | {cond} | " + ", ".join(f"{k} {_f(v, 3)}" for k, v in pt["unique"].items()) + f" | {_f(pt['shared'], 3)} | "
                                 + ", ".join(f"{k} {v:.1f}" for k, v in pt["vif"].items()) + " | "
                                 + ", ".join(f"{k} {_f(v, 2)}" for k, v in pt["correlations"].items()) + f" | {', '.join(pt['removed']) or '-'} |")
            L += ["", "In the reference every zone is controllable: its controllability predictor is constant and removed explicitly "
                  "before the regression (DECISIONS (32)); its confusion ratios contrast appearances only (the control). The distance on z "
                  "is the sum over the 8 categorical variables of the per-variable Jensen-Shannon divergences, not the JS divergence of "
                  "the joint distribution."]
    L += ["", "### Figures", ""] + [f"- `{p.relative_to(work).as_posix()}`" for p in figures]
    L += ["", f"Tables: `{(work / 'tables').as_posix()}`; full output: `{(work / 'analysis.json').as_posix()}`; per-pair detail: "
          f"`{(work / 'CAMPAIGN_ANALYSIS.md').as_posix()}`.", "", MARK_END]
    return "\n".join(L) + "\n"


def append_report(report: Path, text: str) -> None:
    old = report.read_text(encoding="utf-8")
    if MARK_START in old and MARK_END in old:
        head, rest = old.split(MARK_START, 1)
        new = head + text.strip() + rest.split(MARK_END, 1)[1]
    else:
        new = old.rstrip() + "\n\n" + text
    report.write_text(new, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--handoff", type=Path)
    src.add_argument("--runs-dir", type=Path, help="development only: run directories already laid out as campaign runs")
    parser.add_argument("--work", type=Path, default=Path("results/E3_worldmodel_4zones/campaign"))
    parser.add_argument("--no-rsa", action="store_true")
    parser.add_argument("--append-report", type=Path, default=None)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    timings = {}
    if args.handoff is not None:
        try:
            report = ci.check_handoff(args.handoff)
        except ci.IntegrityError as err:
            print(f"REFUSED: {err}", flush=True)
            return 1
        integrity_md = ci.markdown(report)
        runs_root = work / "runs"
        if not all((runs_root / r["run_dir"]).exists() for r in report["runs"]):
            ci.extract(args.handoff, report, runs_root)
        paths = [runs_root / r["run_dir"] for r in report["runs"]]
        mode = "handoff"
    else:
        paths = sorted(p for p in args.runs_dir.iterdir() if (p / "results.json").exists())
        integrity_md = f"## Integrity\n\nNOT CHECKED: --runs-dir {args.runs_dir} (development mode).\n"
        mode = f"--runs-dir {args.runs_dir}, handoff not checked"
    timings["integrity_and_extraction"] = time.perf_counter() - t0
    pairs: list = []
    res = ca.analyse(paths, None, ca.aggregate_b.load_e30_control(), pairs_out=pairs)
    lr = cl.analyse_learning(pairs, res["main"])
    timings["analyses"] = time.perf_counter() - t0 - sum(timings.values())
    rsa_out = None if args.no_rsa else run_rsa(pairs, work)
    rsa_seeds = rsa_over_seeds(rsa_out) if rsa_out else {}
    timings["rsa"] = time.perf_counter() - t0 - sum(timings.values())
    fig_dir = work / "figures"
    fig_dir.mkdir(exist_ok=True)
    figures = []
    common = res["over_seeds"].get("common_checkpoints", [])
    if common and pairs:
        I_by_run = {(r.seed, r.condition): {c: r.measure(c)[1]["real_I_s"] for c in r.measure_dirs} for p in pairs for r in p}
        picks = sorted({common[0], common[len(common) // 2], common[-1]})
        figures += cf.maps(res["propagation"], pairs[0][0].zone_of_state, I_by_run, picks, fig_dir)
    p = cf.curves_b_alpha(res["over_seeds"], fig_dir)
    figures += [p] if p else []
    figures.append(cf.curves_learning(lr["learning"], fig_dir))
    if rsa_out:
        for r in rsa_out["runs"]:
            for s in r.get("checkpoints", []):
                if s["env_step"] in (common[:1] + common[-1:]):
                    figures.append(cf.rsa_mds(work / "rsa" / s["run"] / f"env{s['env_step']:06d}.npz", s, fig_dir))
    timings["figures"] = time.perf_counter() - t0 - sum(timings.values())
    write_tables(res, lr, rsa_seeds, work / "tables")
    (work / "CAMPAIGN_ANALYSIS.md").write_text(ca.markdown(res), encoding="utf-8")
    text = section(res, lr, rsa_out, rsa_seeds, integrity_md, figures, work, mode)
    (work / "RESULTS_SECTION.md").write_text(text, encoding="utf-8")
    timings["total"] = time.perf_counter() - t0
    (work / "analysis.json").write_text(json.dumps({"mode": mode, "timings_s": timings, "analysis": res, "learning": lr,
                                                    "rsa_over_seeds": rsa_seeds}, indent=1, default=float), encoding="utf-8")
    if args.append_report is not None:
        if mode != "handoff":
            print("--append-report refused outside the handoff mode (development output)", flush=True)
            return 1
        append_report(args.append_report, text)
    print(f"done in {timings['total']:.0f} s ({', '.join(f'{k} {v:.0f} s' for k, v in timings.items() if k != 'total')}); outputs in {work}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
