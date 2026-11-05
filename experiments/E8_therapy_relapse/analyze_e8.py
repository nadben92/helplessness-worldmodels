"""E8's analysis (docs/E8_therapy_relapse.md; definitions, thresholds and predictions: docs/e8/EXPECTED_FORMS.md, committed
before any computation). Reads read-only the probe jobs (probes_e8.py: rows.json) and the therapy runs (checkpoints.jsonl:
the treated-zone visits); writes light outputs only.

Curves: per pair and checkpoint, exposed and reference, of b (b_walks: E5's 12-assignment protocol, the main b), C1 and C3
(D7's readout over the zone steps measured and well localised) and the directed share D (E6), on the treated images and,
for the shuffle pairs, on the untreated ones (means over the images valid at that checkpoint, with their count); against
environment steps, gradient steps and treated-zone visits. Then the forms R and M, the recovery times (t½, sustained t½,
crossings, censoring), the predictions P1-P4 and the belief / cache decomposition at the oracle checkpoints, exactly as
EXPECTED_FORMS writes them. Form F0 is checked by check_identity.py; TW is blocked (DECISIONS (14)).

Outputs (--out): curves.csv, recovery.csv, decomposition.csv, transfer.csv, forms.json, figures/recovery_<pair>.png,
figures/b_vs_directed.png (over E6's 650,000 cloud when --e6-cloud is given).

Usage (from the repo root):
  python experiments/E8_therapy_relapse/analyze_e8.py --probes "<root>/E8_therapy_relapse/probes__*" \\
      --runs "<root>/E8_therapy_relapse/grid12x12__*" --out results/E8_therapy_relapse/<batch> [--e6-cloud "<campaign>/e6__*"]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

LOG2_5 = math.log2(5)
MEASURES = ("b", "C1", "C3", "D")
FIELD = {"b": "b_walks", "C1": "C1_loc", "C3": "C3_loc", "D": "directed_share"}
BASELINES, SHUFFLE_HEAVY, SHUFFLE_LIGHT = ("base_A", "base_B"), ("shuffle_A", "shuffle_E"), ("shuffle_F",)
R_FROM, R_TOL_I, R_TOL = 5000, 0.15, 0.05
M_DRAWDOWN = {"b": 0.5, "C3": 0.15, "D": 0.15}


# ------------------------------------------------------------------------------------------------------------- inputs
def load_rows(patterns: list[str]) -> list[dict]:
    """Every row of the done probe jobs; a checkpoint measured twice for a pair keeps the newest job."""
    newest: dict[tuple, tuple[str, list]] = {}
    for p in patterns:
        for d in sorted(glob.glob(p)):
            d = Path(d)
            status = d / "status.json"
            if not status.is_file() or json.loads(status.read_text(encoding="utf-8"))["status"] != "done":
                continue
            rows = json.loads((d / "rows.json").read_text(encoding="utf-8"))
            for pair in {r["pair"] for r in rows}:
                key = (pair, rows[0]["checkpoint"])
                if key not in newest or d.name > newest[key][0]:
                    newest[key] = (d.name, [r for r in rows if r["pair"] == pair])
    return [r for _, rs in newest.values() for r in rs]


def load_visits(patterns: list[str]) -> dict[str, dict[int, int]]:
    """model id -> checkpoint -> cumulative treated-zone visits (the run's records)."""
    out: dict[str, dict[int, int]] = {}
    for p in patterns:
        for d in sorted(glob.glob(p)):
            d = Path(d)
            if not (d / "checkpoints.jsonl").is_file():
                continue
            model = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))["e8"]["run"]["model"]
            recs = [json.loads(x) for x in (d / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            out[model] = {int(r["env_step"]): int(r["e8"]["visits"]["treated"]) for r in recs}
    return out


# ------------------------------------------------------------------------------------------------------------- curves
def mean_valid(rows: list[dict], field: str) -> tuple[float, int]:
    v = [float(r[field]) for r in rows if r["walks_valid"] and r.get(field) is not None and np.isfinite(float(r[field]))]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def curves(rows: list[dict], visits: dict) -> list[dict]:
    """One row per (pair, group, checkpoint): exposed and reference means of every measure over the valid images of the
    group (treated / untreated), the gaps g = reference - exposed, and the abscissae."""
    out = []
    for pair in sorted({r["pair"] for r in rows}):
        pr = [r for r in rows if r["pair"] == pair]
        for group in ("treated", "untreated"):
            gr = [r for r in pr if bool(r["treated"]) == (group == "treated")]
            for t in sorted({r["checkpoint"] for r in gr}):
                at = [r for r in gr if r["checkpoint"] == t]
                exp, ref = [r for r in at if r["role"] == "exposed"], [r for r in at if r["role"] == "reference"]
                if not exp:
                    continue
                rec = {"pair": pair, "group": group, "checkpoint": t, "grad_step": exp[0]["grad_step"], "therapy": exp[0]["therapy"],
                       "treated_visits": visits.get(exp[0]["model"], {}).get(t), "n_images": len(exp)}
                for m in MEASURES:
                    rec[f"{m}_exp"], rec[f"n_valid_{m}"] = mean_valid(exp, FIELD[m])
                    rec[f"{m}_ref"], _ = mean_valid(ref, FIELD[m])
                    rec[f"g_{m}"] = rec[f"{m}_ref"] - rec[f"{m}_exp"] if m != "b" else float("nan")
                rec["I_ref_walks"], _ = mean_valid(ref, "walks_I_step1")
                o = [float(r["oracle_directed_share"]) for r in exp if r.get("oracle_directed_share") is not None]
                rec["D_oracle1"] = float(np.mean(o)) if o else float("nan")
                out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------------------- times
def recovery_time(ts: list[int], ok: list[bool]) -> tuple[int | None, int | None]:
    """(t½, sustained t½): the first checkpoint t > 0 where ok holds; the first after which every later one holds."""
    first = next((t for t, o in zip(ts, ok) if t > 0 and o), None)
    sustained = None
    for i, t in enumerate(ts):
        if t > 0 and all(ok[i:]):
            sustained = t
            break
    return first, sustained


def times_for(series: list[dict]) -> dict:
    """EXPECTED_FORMS section 2 on one pair's treated (or untreated) curve."""
    ts = [s["checkpoint"] for s in series]
    T = ts[-1]
    out = {"T": T}
    b0 = series[0]["b_exp"]
    if np.isfinite(b0) and b0 <= -0.5:
        out["t_half_b"], out["t_half_b_sustained"] = recovery_time(ts, [s["b_exp"] >= b0 / 2 for s in series])
    else:
        out["t_half_b"] = out["t_half_b_sustained"] = "nothing to recover"
    for m in ("C1", "C3", "D"):
        g0 = series[0][f"g_{m}"]
        if np.isfinite(g0) and g0 >= 0.10:
            out[f"t_half_{m}"], out[f"t_half_{m}_sustained"] = recovery_time(ts, [s[f"g_{m}"] <= g0 / 2 for s in series])
        else:
            out[f"t_half_{m}"] = out[f"t_half_{m}_sustained"] = "nothing to recover"
    out["cross_b_in_control"], _ = recovery_time(ts, [s["b_exp"] > -0.5 for s in series])
    out["cross_D_at_reference"], _ = recovery_time(ts, [s["g_D"] <= 0.05 for s in series])
    for k, v in list(out.items()):
        if v is None:
            out[k] = f"censored at {T}"
    return out


def as_steps(v, T: int) -> float | None:
    """A t½ as a number for the lag: a censored one enters at T (a bound); undefined -> None."""
    if isinstance(v, (int, np.integer)):
        return float(v)
    if isinstance(v, str) and v.startswith("censored"):
        return float(T)
    return None


# ------------------------------------------------------------------------------------------------------------- forms
def form_R(rows: list[dict]) -> dict:
    """Every reference image valid at checkpoint 0: |I_ref - log2 5| <= 0.15 and C1, C3, D within 0.05 of checkpoint 0 at
    every t >= 5,000 (a transient before is reported)."""
    out = {}
    refs = [r for r in rows if r["role"] == "reference"]
    for model in sorted({r["model"] for r in refs}):
        mr = [r for r in refs if r["model"] == model]
        fails, transient = [], []
        for (pair, image) in sorted({(r["pair"], r["image"]) for r in mr}):
            ser = sorted((r for r in mr if r["pair"] == pair and r["image"] == image), key=lambda r: r["checkpoint"])
            if not ser or ser[0]["checkpoint"] != 0 or not ser[0]["walks_valid"]:
                continue
            base = {m: float(ser[0][FIELD[m]]) for m in ("C1", "C3", "D")}
            for r in ser[1:]:
                bad = [f"I {r['walks_I_step1']:.3f}"] if abs(float(r["walks_I_step1"]) - LOG2_5) > R_TOL_I else []
                bad += [f"{m} {float(r[FIELD[m]]):.3f} vs {base[m]:.3f}" for m in base if abs(float(r[FIELD[m]]) - base[m]) > R_TOL]
                if bad:
                    (fails if r["checkpoint"] >= R_FROM else transient).append({"pair": pair, "image": image, "checkpoint": r["checkpoint"], "what": bad})
        out[model] = {"held": not fails, "failures": fails, "transient_before_5000": transient}
    return {"held": all(v["held"] for v in out.values()), "by_model": out}


def drawdown(y: np.ndarray) -> float:
    run = np.maximum.accumulate(y)
    return float(np.max(run - y)) if y.size else float("nan")


def form_M(cv: list[dict]) -> dict:
    out = {}
    for pair in sorted({c["pair"] for c in cv}):
        ser = [c for c in cv if c["pair"] == pair and c["group"] == "treated"]
        res = {}
        for m in ("b", "C3", "D"):
            t = np.array([c["checkpoint"] for c in ser], dtype=float)
            y = np.array([c[f"{m}_exp"] for c in ser], dtype=float)
            ok = np.isfinite(y)
            rho = float(spearmanr(t[ok], y[ok]).statistic) if ok.sum() >= 3 else float("nan")
            dd = drawdown(y[ok])
            res[m] = {"rho": rho, "drawdown": dd, "held": bool(np.isfinite(rho) and rho >= 0 and dd <= M_DRAWDOWN[m])}
        out[pair] = {"held": all(v["held"] for v in res.values()), **res}
    return {"held": all(v["held"] for v in out.values()), "by_pair": out}


# ------------------------------------------------------------------------------------------------------------- predictions
def p1(series: list[dict], times: dict) -> dict:
    b = [s["b_exp"] for s in series]
    T = series[-1]["checkpoint"]
    i_cross = next((i for i, v in enumerate(b) if v > -0.5), None)
    stays = i_cross is not None and all(v >= -1.0 for v in b[i_cross:])
    conds = {"t_half_b_not_censored": isinstance(times["t_half_b"], (int, np.integer)), "b_T_in_control": bool(b[-1] > -0.5), "no_fall_back": bool(stays)}
    return {"held": all(conds.values()), **conds, "b_0": b[0], "b_T": b[-1], "T": T}


def p2(times: dict[str, dict]) -> dict:
    lags = {}
    for pair, t in times.items():
        tc, td = as_steps(t["t_half_C3"], t["T"]), as_steps(t["t_half_D"], t["T"])
        both_censored = all(isinstance(t[k], str) and t[k].startswith("censored") for k in ("t_half_C3", "t_half_D"))
        lags[pair] = None if (tc is None or td is None or both_censored) else td - tc
    exposed = [p for p in lags if lags[p] is not None]
    a = sum(lags[p] > 0 for p in exposed)
    heavy, light = [lags.get(p) for p in SHUFFLE_HEAVY], [lags.get(p) for p in (*BASELINES, *SHUFFLE_LIGHT)]
    b_ok = None if any(v is None for v in heavy + light) else min(heavy) > max(light)
    return {"lags": lags, "a_positive": a, "a_held": a >= 4, "b_held": b_ok, "held": bool(a >= 4 and b_ok)}


def p3(series: list[dict]) -> dict:
    out = {}
    for m, need in (("D", 0.20), ("b", 0.5)):
        y = np.array([s[f"{m}_exp"] for s in series], dtype=float)
        gain = y[-1] - y[0]
        steps = np.diff(y)
        out[f"gain_{m}"], out[f"s_{m}"] = float(gain), (float(np.nanmax(steps) / gain) if gain > 0 else float("nan"))
        out[f"enough_{m}"] = bool(gain >= need)
    if not (out["enough_D"] and out["enough_b"]):
        return {**out, "verdict": "gains too small: no verdict"}
    return {**out, "verdict": "holds" if (out["s_D"] >= 0.5 and out["s_b"] < 0.5) else "does not hold"}


def p4(rows: list[dict], pair: str) -> dict:
    """Transfer index at T (EXPECTED_FORMS P4) for one shuffle pair, per image then averaged by group."""
    pr = [r for r in rows if r["pair"] == pair]
    T = max(r["checkpoint"] for r in pr)
    out = {"T": T}
    for m in ("b", "D"):
        frac = {"treated": [], "untreated": []}
        for image in sorted({r["image"] for r in pr}):
            ser = {(r["checkpoint"], r["role"]): r for r in pr if r["image"] == image}
            if (0, "exposed") not in ser or (T, "exposed") not in ser or not ser[(0, "exposed")]["walks_valid"] or not ser[(T, "exposed")]["walks_valid"]:
                continue
            group = "treated" if ser[(0, "exposed")]["treated"] else "untreated"
            if m == "b":
                b0, bT = float(ser[(0, "exposed")]["b_walks"]), float(ser[(T, "exposed")]["b_walks"])
                if group == "untreated" and b0 > -1.0:
                    continue
                if b0 < 0:
                    frac[group].append(1 - bT / b0)
            else:
                g0 = float(ser[(0, "reference")]["directed_share"]) - float(ser[(0, "exposed")]["directed_share"])
                gT = float(ser[(T, "reference")]["directed_share"]) - float(ser[(T, "exposed")]["directed_share"])
                if g0 >= 0.10:
                    frac[group].append(1 - gT / g0)
        tr, un = (float(np.mean(frac[g])) if frac[g] else float("nan") for g in ("treated", "untreated"))
        ti = un / tr if np.isfinite(tr) and tr >= 0.25 and np.isfinite(un) else float("nan")
        out[m] = {"treated_mean_f": tr, "untreated_mean_f": un, "n_treated": len(frac["treated"]), "n_untreated": len(frac["untreated"]), "TI": ti}
    ti = out["D"]["TI"]
    out["verdict"] = ("undefined" if not np.isfinite(ti) else "transfer" if ti >= 0.5 else "no transfer" if ti <= 0.2 else "partial")
    return out


# ------------------------------------------------------------------------------------------------------------- outputs
def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows])


def figures(cv: list[dict], rows: list[dict], out: Path, e6_cloud: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "figures").mkdir(exist_ok=True)
    for pair in sorted({c["pair"] for c in cv}):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
        for ax, m in zip(axes, MEASURES):
            for group, ls in (("treated", "-"), ("untreated", "--")):
                ser = [c for c in cv if c["pair"] == pair and c["group"] == group]
                if not ser:
                    continue
                x = [c["checkpoint"] for c in ser]
                ax.plot(x, [c[f"{m}_exp"] for c in ser], ls, marker="o", ms=3, color="tab:red", label=f"exposed, {group}")
                if m != "b":
                    ax.plot(x, [c[f"{m}_ref"] for c in ser], ls, marker="o", ms=3, color="tab:blue", label=f"reference, {group}")
            ax.set_xscale("symlog", linthresh=500)
            ax.set_title(m), ax.set_xlabel("therapy environment steps")
        axes[0].legend(fontsize=7)
        fig.suptitle(pair)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"recovery_{pair}.png", dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if e6_cloud:
        ax.scatter([r["b_step1"] for r in e6_cloud], [r["directed_share"] for r in e6_cloud], s=8, color="0.75", label="E6 at 650,000")
    for role, color in (("exposed", "tab:red"), ("reference", "tab:blue")):
        sel = [r for r in rows if r["role"] == role and r["walks_valid"]]
        ax.scatter([r["b_step1"] for r in sel], [r["directed_share"] for r in sel], s=10, color=color, alpha=0.6, label=f"E8 {role}")
    ax.set_xlabel("b at step 1 (E6's episodes)"), ax.set_ylabel("directed share"), ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures" / "b_vs_directed.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--e6-cloud", nargs="+", default=[], help="E6's rows.json directories at 650,000 (the overlay)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    rows = load_rows(args.probes)
    if not rows:
        raise SystemExit("no done probe job")
    cv = curves(rows, load_visits(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "curves.csv", cv)
    times, recov = {}, []
    forms: dict = {"R": form_R(rows), "M": form_M(cv), "F0": "check_identity.py", "TW": "blocked (DECISIONS (14))", "P1": {}, "P3": {}, "P4": {}}
    for pair in sorted({c["pair"] for c in cv}):
        for group in ("treated", "untreated"):
            ser = [c for c in cv if c["pair"] == pair and c["group"] == group]
            if not ser or ser[0]["checkpoint"] != 0:
                continue
            t = times_for(ser)
            recov.append({"pair": pair, "group": group, **t})
            if group == "treated":
                times[pair] = t
                forms["P1"][pair] = p1(ser, t)
                forms["P3"][pair] = p3(ser)
        if pair.startswith("shuffle"):
            forms["P4"][pair] = p4(rows, pair)
    forms["P2"] = p2(times)
    verdicts = [v["verdict"] for v in forms["P4"].values()]
    forms["P4_overall"] = next((v for v in ("transfer", "no transfer") if verdicts.count(v) >= 2), "mixed") if len(verdicts) == 3 else "incomplete"
    write_csv(args.out / "recovery.csv", recov)
    deco = [{"pair": c["pair"], "checkpoint": c["checkpoint"], "D_exp": c["D_exp"], "D_oracle1": c["D_oracle1"], "D_ref": c["D_ref"],
             "belief_effect": c["D_oracle1"] - c["D_exp"], "cache_effect": c["D_ref"] - c["D_oracle1"]}
            for c in cv if c["group"] == "treated" and np.isfinite(c["D_oracle1"])]
    write_csv(args.out / "decomposition.csv", deco) if deco else None
    write_csv(args.out / "transfer.csv", [{"pair": p, **v} for p, v in forms["P4"].items()]) if forms["P4"] else None
    (args.out / "forms.json").write_text(json.dumps(forms, indent=1, default=float), encoding="utf-8")
    if not args.no_figures:
        cloud = [r for d in args.e6_cloud for g in glob.glob(d) for r in json.loads((Path(g) / "rows.json").read_text(encoding="utf-8"))
                 if r.get("valid") in (True, "True")]
        figures(cv, rows, args.out, cloud)
    print(f"[E8 analysis] {len(rows)} rows, {len(cv)} curve points -> {args.out}; R held {forms['R']['held']}, M held {forms['M']['held']}, "
          f"P2 {forms['P2']['held']}, P4 {forms['P4_overall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
