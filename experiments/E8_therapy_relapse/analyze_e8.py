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


H_FROM, H_SUCCESS, H_MASS, COLLAPSE_BITS = 5000, 0.90, 0.90, 1.0


def form_H(patterns: list[str]) -> dict:
    """EXPECTED_FORMS section 4, H, on every therapy run: done at max_env_steps, switch_step 0 in every record, the
    collapse guard's statistic above 1.0 bit at every evaluation, and from 5,000 on a success rate >= 0.90 and a prior
    mass on the true successor >= 0.90 in every region (the exposed run's nw apart: reported, no verdict)."""
    out = {}
    for p in patterns:
        for d in sorted(glob.glob(p)):
            d = Path(d)
            if not (d / "checkpoints.jsonl").is_file():
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8"))
            meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
            model, exposed = cfg["e8"]["run"]["model"], cfg["e8"]["run"]["role"] == "experimental"
            recs = [json.loads(x) for x in (d / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            budget = int(cfg["e8"]["max_env_steps"])
            stats = [e[1] for r in recs for e in (r.get("switch_pairs") or {}).get("evals", [])]
            a = {"status_done": meta["status"] == "done", "env_steps": int(meta["env_steps"]), "at_budget": int(meta["env_steps"]) == budget,
                 "switch_step_0_everywhere": all(r["switch_step"] == 0 for r in recs), "n_guard_evals": len(stats),
                 "guard_stat_min": float(min(stats)) if stats else float("nan"),
                 "guard_above_collapse": bool(stats) and min(stats) > COLLAPSE_BITS}
            fails, transient, nw, unmeasured = [], [], [], []
            for r in recs:
                t, h = int(r["env_step"]), r["e8"]["health"]
                sr = r["success_rate"]  # None before the first episode ends (checkpoint 0)
                bad = [f"success {sr:.3f}"] if sr is not None and sr < H_SUCCESS else []
                # baseline: one record; shuffle (EXPECTED_FORMS section 5): one per treated assignment, keyed by it
                parts = [] if h is None else [("", h)] if "mass_true_by_region" in h else [(f"assignment {k}: ", v) for k, v in h.items()]
                treated = r["e8"].get("treated_slot") or "nw"
                for label, part in parts:
                    for region, mass in part["mass_true_by_region"].items():
                        if mass is None:  # no real context of that region in the replay: not measured, reported
                            unmeasured.append({"checkpoint": t, "where": label + region})
                        elif exposed and region == treated:
                            nw.append({"checkpoint": t, "where": label + region, "mass_true": mass, "I_s": part["I_s_by_region"][region]})
                        elif mass < H_MASS:
                            bad.append(f"mass {label}{region} {mass:.3f}")
                if bad and t > 0:
                    (fails if t >= H_FROM else transient).append({"checkpoint": t, "what": bad})
            held = all(a[k] for k in ("status_done", "at_budget", "switch_step_0_everywhere", "guard_above_collapse")) and not fails
            out[model] = {"held": held, **a, "failures": fails, "transient_before_5000": transient,
                          "success_rate": {int(r["env_step"]): r["success_rate"] for r in recs},
                          "health": {int(r["env_step"]): r["e8"]["health"] for r in recs if r["e8"]["health"]},
                          "exposed_nw_recovery": nw, "regions_not_measured": unmeasured}
    return {"held": bool(out) and all(v["held"] for v in out.values()), "by_model": out}


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


def lag_interval(t: dict) -> tuple[float, float] | str:
    """EXPECTED_FORMS section 5 (corrected): the lag t½(D) - t½(C3) as an interval [lo, hi] -- exact when both t½ are
    reached, a lower bound T - t½(C3) when D is censored, an upper bound t½(D) - T when C3 is, (-inf, inf) when both are;
    "nothing to recover" when either gap was below 0.10 at checkpoint 0 (not a censoring)."""
    tc, td, T = t["t_half_C3"], t["t_half_D"], float(t["T"])
    if as_steps(tc, t["T"]) is None or as_steps(td, t["T"]) is None:
        return "nothing to recover"
    c_cens, d_cens = (isinstance(v, str) for v in (tc, td))
    if c_cens and d_cens:
        return (-np.inf, np.inf)
    if d_cens:
        return (T - float(tc), np.inf)
    if c_cens:
        return (-np.inf, float(td) - T)
    return (float(td) - float(tc),) * 2


def _lag_report(iv) -> float | str:
    """An exact lag as a number; a censored one as its bound, never as a value."""
    if isinstance(iv, str):
        return iv
    lo, hi = iv
    if lo == hi:
        return lo
    if np.isfinite(lo):
        return f">= {lo:g} (D censored)"
    if np.isfinite(hi):
        return f"<= {hi:g} (C3 censored)"
    return "not decidable (both censored)"


def p2(times: dict[str, dict]) -> dict:
    """P2 on the lag intervals (EXPECTED_FORMS section 5, corrected): (a) and (b) each hold, fail or are not decidable.
    (a) is read on shuffle A and E (DECISIONS (24)); the original rule (4 of the 5 positive) is reported, in no verdict."""
    iv = {pair: lag_interval(t) for pair, t in times.items()}
    positive = [p for p, v in iv.items() if not isinstance(v, str) and v[0] > 0]
    not_positive = [p for p, v in iv.items() if not isinstance(v, str) and v[1] <= 0]  # nothing to recover: undecided
    undecided = len(iv) - len(positive) - len(not_positive)
    a_original = True if len(positive) >= 4 else False if len(positive) + undecided < 4 else None
    a = False if any(p in not_positive for p in SHUFFLE_HEAVY) else True if all(p in positive for p in SHUFFLE_HEAVY) else None
    a_original_note = "reported, in no verdict"
    if all(p in not_positive for p in BASELINES):
        lags = ", ".join(f"{iv[p][1]:,.0f}" for p in BASELINES)
        a_original_note = f"failed by construction: wave 1's baseline lags ({lags}) made it impossible"
    heavy, light = [iv.get(p) for p in SHUFFLE_HEAVY], [iv.get(p) for p in (*BASELINES, *SHUFFLE_LIGHT)]
    if any(v is None or isinstance(v, str) for v in heavy + light):
        b = None
    elif min(v[0] for v in heavy) > max(v[1] for v in light):
        b = True
    elif min(v[1] for v in heavy) <= max(v[0] for v in light):
        b = False
    else:
        b = None
    decided = [x for x in (a, b) if x is not None]
    held = None if not decided else all(decided)
    on = {(True, True): "(a) and (b)", (True, False): "(a) only", (False, True): "(b) only", (False, False): "not decidable"}
    nd = lambda x: "not decidable" if x is None else x
    return {"lags": {p: _lag_report(v) for p, v in iv.items()}, "a_positive": len(positive), "a_held": nd(a), "b_held": nd(b),
            "held": nd(held), "held_on": on[(a is not None, b is not None)],
            "a_original_held": nd(a_original), "a_original_note": a_original_note}


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
            b0, bT = float(ser[(0, "exposed")]["b_walks"]), float(ser[(T, "exposed")]["b_walks"])
            if group == "untreated" and b0 > -1.0:  # EXPECTED_FORMS section 5: the untreated set, for f_b and f_D alike
                continue
            if m == "b":
                if b0 <= -0.5:  # treated: t½'s condition (untreated already <= -1.0)
                    frac[group].append(1 - bT / b0)
            else:
                g0 = float(ser[(0, "reference")]["directed_share"]) - float(ser[(0, "exposed")]["directed_share"])
                gT = float(ser[(T, "reference")]["directed_share"]) - float(ser[(T, "exposed")]["directed_share"])
                if g0 >= 0.10:
                    frac[group].append(1 - gT / g0)
        tr, un = (float(np.mean(frac[g])) if frac[g] else float("nan") for g in ("treated", "untreated"))
        ti = un / tr if np.isfinite(tr) and tr >= 0.25 and np.isfinite(un) else float("nan")  # both groups non-empty
        out[m] = {"treated_mean_f": tr, "untreated_mean_f": un, "n_treated": len(frac["treated"]), "n_untreated": len(frac["untreated"]), "TI": ti}
    ti = out["D"]["TI"]
    out["verdict"] = ("undefined" if not np.isfinite(ti) else "transfer" if ti >= 0.5 else "no transfer at T" if ti <= 0.2 else "partial")
    out["crossings"] = crossings(pr, T)
    return out


def crossings(pr: list[dict], T: int) -> dict:
    """EXPECTED_FORMS section 5 (revised): per image of P4's sets, the first checkpoint t > 0 with g_D(t) <= 0.05 (images
    with g_D(0) >= 0.10) and the first with b(t) > -0.5 (untreated: b(0) <= -1.0; treated: b(0) <= -0.5), read at the
    checkpoints where the image's point is valid; not reached by T: "> T". Untreated and treated, with the counts crossed."""
    out = {"untreated": {}, "treated": {}}
    for image in sorted({r["image"] for r in pr}):
        ser = {(r["checkpoint"], r["role"]): r for r in pr if r["image"] == image}
        if (0, "exposed") not in ser or (T, "exposed") not in ser or not ser[(0, "exposed")]["walks_valid"] or not ser[(T, "exposed")]["walks_valid"]:
            continue
        group = "treated" if ser[(0, "exposed")]["treated"] else "untreated"
        b0 = float(ser[(0, "exposed")]["b_walks"])
        if group == "untreated" and b0 > -1.0:  # P4's untreated set
            continue
        ts = sorted(t for (t, role) in ser if role == "exposed" and t > 0 and ser[(t, "exposed")]["walks_valid"] and (t, "reference") in ser)
        g = lambda t: float(ser[(t, "reference")]["directed_share"]) - float(ser[(t, "exposed")]["directed_share"])
        rec = {}
        if g(0) >= 0.10:
            rec["g_D<=0.05"] = next((t for t in ts if g(t) <= 0.05), f"> {T}")
        if b0 <= -0.5:
            rec["b>-0.5"] = next((t for t in ts if float(ser[(t, "exposed")]["b_walks"]) > -0.5), f"> {T}")
        if rec:
            out[group][image] = rec
    for group, recs in list(out.items()):
        out[f"{group}_crossed_by_T"] = {k: f"{sum(isinstance(r.get(k), (int, np.integer)) for r in recs.values())} / {sum(k in r for r in recs.values())}"
                                        for k in ("g_D<=0.05", "b>-0.5")}
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


ABSCISSAE = (("checkpoint", "therapy environment steps"), ("grad_step", "therapy gradient steps"), ("treated_visits", "treated-zone (nw) visits"))
LOC_QUADRANTS = ("nw", "ne", "se", "sw", "corridor")


def figures_wave(cv: list[dict], rows: list[dict], out: Path, twin: list[dict]) -> None:
    """Wave 1 figures: every measure against the three abscissae, the localisation, the decomposition and the twin's b_tab
    over the world model's b (EXPECTED_FORMS section 4: descriptive)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "figures").mkdir(exist_ok=True)
    pairs = sorted({c["pair"] for c in cv})
    for pair in pairs:
        ser = [c for c in cv if c["pair"] == pair and c["group"] == "treated"]
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for i, (key, label) in enumerate(ABSCISSAE):
            x = [c[key] if c[key] is not None else np.nan for c in ser]
            for j, m in enumerate(MEASURES):
                ax = axes[i, j]
                ax.plot(x, [c[f"{m}_exp"] for c in ser], "-", marker="o", ms=4, lw=2, color="tab:red", label="exposed")
                if m != "b":
                    ax.plot(x, [c[f"{m}_ref"] for c in ser], "--", marker="s", ms=4, lw=2, color="tab:blue", label="reference")
                else:
                    ax.axhline(-0.5, color="0.6", lw=1, ls=":")
                ax.set_xscale("symlog", linthresh=500 if key == "checkpoint" else 300 if key == "grad_step" else 50)
                ax.set_xlim(left=0)
                ax.set_xlabel(label), ax.set_title(f"{m}" if m != "b" else "b (walks, step 1)", fontsize=10)
                ax.grid(alpha=0.25)
        axes[0, 1].legend(fontsize=8)
        fig.suptitle(f"{pair}: recovery on the training image (zone nw)")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"recovery_3abscissae_{pair}.png", dpi=110)
        plt.close(fig)
        pr = sorted((r for r in rows if r["pair"] == pair), key=lambda r: r["checkpoint"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for role, color, ls in (("exposed", "tab:red", "-"), ("reference", "tab:blue", "--")):
            rr = [r for r in pr if r["role"] == role]
            x = [r["checkpoint"] for r in rr]
            axes[0].plot(x, [r["loc_template_zone"] for r in rr], ls, marker="o", ms=4, lw=2, color=color, label=role)
            axes[2].plot(x, [r["loc_directed_localised"] for r in rr], ls, marker="o", ms=4, lw=2, color=color, label=f"{role}, well localised")
            axes[2].plot(x, [r["loc_directed_mislocalised"] for r in rr], ls, marker="x", ms=6, lw=1, color=color, alpha=0.6,
                         label=f"{role}, mislocalised (true cell)")
        ex = [r for r in pr if r["role"] == "exposed"]
        x = np.array([r["checkpoint"] for r in ex])
        bottom = np.zeros(len(ex))
        greys = {"nw": "0.15", "ne": "0.35", "se": "0.55", "sw": "0.72", "corridor": "0.88"}
        for q in LOC_QUADRANTS:
            share = np.array([float(r.get(f"loc_mis_decoded_in_{q}") or 0.0) for r in ex])
            axes[1].bar(np.arange(len(ex)), share, bottom=bottom, color=greys[q], edgecolor="white", linewidth=1, label=q)
            bottom += share
        axes[1].set_xticks(np.arange(len(ex)), [f"{t / 1000:g}k" if t else "0" for t in x], rotation=60, fontsize=7)
        for i, r in enumerate(ex):  # how many mislocalised zone steps each bar rests on
            n_mis = int(round(r["loc_n_zone_steps"] * (1 - r["loc_loc_cell"])))
            axes[1].text(i, 1.01, str(n_mis), ha="center", va="bottom", fontsize=6, color="0.3")
        axes[1].set_ylim(0, 1.1)
        axes[0].set_title("decoder localisation in the zone (L_dec)", fontsize=10)
        axes[1].set_title("exposed, mislocalised zone steps: decoded quadrant", fontsize=10)
        axes[2].set_title("directed share from the true cell", fontsize=10)
        for ax in (axes[0], axes[2]):
            ax.set_xscale("symlog", linthresh=500), ax.set_xlabel("therapy environment steps"), ax.grid(alpha=0.25)
        axes[1].set_xlabel("therapy checkpoint")
        for ax in (axes[0], axes[2]):
            ax.legend(fontsize=7)
        axes[1].legend(fontsize=7, loc="center right", title="n above bars: steps", title_fontsize=6)
        fig.suptitle(f"{pair}: localisation over the therapy (descriptive)")
        fig.tight_layout()
        fig.savefig(out / "figures" / f"localisation_{pair}.png", dpi=110)
        plt.close(fig)
    deco = [c for c in cv if c["group"] == "treated" and np.isfinite(c["D_oracle1"])]
    if deco:
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
        for ax, pair in zip(axes[0], pairs):
            d = [c for c in deco if c["pair"] == pair]
            x = [c["checkpoint"] for c in d]
            ax.plot(x, [c["D_oracle1"] - c["D_exp"] for c in d], "-", marker="o", lw=2, color="tab:red", label="belief effect (oracle k=1 - exposed)")
            ax.plot(x, [c["D_ref"] - c["D_oracle1"] for c in d], "--", marker="s", lw=2, color="tab:purple", label="cache effect (reference - oracle k=1)")
            ax.axhline(0, color="0.6", lw=1)
            ax.set_xscale("symlog", linthresh=2000), ax.set_xlabel("therapy environment steps"), ax.set_ylabel("directed share")
            ax.set_title(pair), ax.grid(alpha=0.25), ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / "decomposition.png", dpi=110)
        plt.close(fig)
    if twin:
        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), squeeze=False)
        for ax, pair in zip(axes[0], pairs):
            ser = [c for c in cv if c["pair"] == pair and c["group"] == "treated"]
            ax.plot([c["treated_visits"] for c in ser], [c["b_exp"] for c in ser], "-", marker="o", lw=2, color="tab:red", label="world model, exposed (b, walks)")
            for role, color, ls in (("exposed", "0.25", "-"), ("reference", "0.6", "--")):
                tw = [t for t in twin if t["pair"] == pair and t["role"] == role]
                ax.plot([t["treated_visits"] for t in tw], [t["b_tab_nw"] for t in tw], ls, marker="^", lw=2, color=color,
                        label=f"tabular twin, {role} (start + therapy counts)")
            ax.axhline(-0.5, color="0.6", lw=1, ls=":")
            ax.set_xscale("symlog", linthresh=50), ax.set_xlim(left=0), ax.set_xlabel("treated-zone (nw) visits in the therapy"), ax.set_ylabel("b in nw (bits)")
            ax.set_title(pair), ax.grid(alpha=0.25), ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "twin_overlay.png", dpi=110)
        plt.close(fig)


COMPARE_FIELDS = ("b_walks", "walks_I_step1", "b_step1", "C1_loc", "C3_loc", "directed_share", "loc_template_zone", "oracle_directed_share")


def compare_devices(rows: list[dict], other: list[dict]) -> list[dict]:
    """EXPECTED_FORMS section 4, X (exploratory): the same jobs on another device, field by field."""
    key = lambda r: (r["pair"], r["checkpoint"], r["role"], r["image"])
    by = {key(r): r for r in other}
    out = []
    for r in rows:
        o = by.get(key(r))
        if o is None:
            continue
        rec = {"pair": r["pair"], "checkpoint": r["checkpoint"], "role": r["role"]}
        for f in COMPARE_FIELDS:
            a, b = r.get(f), o.get(f)
            rec[f"d_{f}"] = (float(b) - float(a)) if a is not None and b is not None else float("nan")
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--e6-cloud", nargs="+", default=[], help="E6's rows.json directories at 650,000 (the overlay)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--twin", nargs="+", default=[], help="therapy_counts.py's twin_<pair>.csv files (the twin overlay)")
    parser.add_argument("--compare-probes", nargs="+", default=[], help="the same probe jobs run on another device (form X)")
    args = parser.parse_args(argv)
    rows = load_rows(args.probes)
    if not rows:
        raise SystemExit("no done probe job")
    cv = curves(rows, load_visits(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "curves.csv", cv)
    times, recov = {}, []
    forms: dict = {"H": form_H(args.runs), "R": form_R(rows), "M": form_M(cv), "F0": "check_identity.py",
                   "TW": "therapy_counts.py (forms_therapy_<pair>.json)", "P1": {}, "P3": {}, "P4": {}}
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
    # EXPECTED_FORMS section 4: (b) and the original (a) need the 5 exposed agents; the lags alone otherwise
    forms["P2"]["decidable"] = all(p in times for p in (*BASELINES, *SHUFFLE_HEAVY, *SHUFFLE_LIGHT))
    if not forms["P2"]["decidable"]:
        for k in ("held", "a_held", "b_held", "held_on", "a_original_held", "a_original_note"):
            forms["P2"][k] = "not decidable on these agents"
    verdicts = [v["verdict"] for v in forms["P4"].values()]
    forms["P4_overall"] = next((v for v in ("transfer", "no transfer at T") if verdicts.count(v) >= 2), "mixed") if len(verdicts) == 3 else "incomplete"
    write_csv(args.out / "recovery.csv", recov)
    deco = [{"pair": c["pair"], "checkpoint": c["checkpoint"], "D_exp": c["D_exp"], "D_oracle1": c["D_oracle1"], "D_ref": c["D_ref"],
             "belief_effect": c["D_oracle1"] - c["D_exp"], "cache_effect": c["D_ref"] - c["D_oracle1"]}
            for c in cv if c["group"] == "treated" and np.isfinite(c["D_oracle1"])]
    write_csv(args.out / "decomposition.csv", deco) if deco else None
    write_csv(args.out / "transfer.csv", [{"pair": p, **v} for p, v in forms["P4"].items()]) if forms["P4"] else None
    if not args.no_figures:
        cloud = [r for d in args.e6_cloud for g in glob.glob(d) for r in json.loads((Path(g) / "rows.json").read_text(encoding="utf-8"))
                 if r.get("valid") in (True, "True")]
        figures(cv, rows, args.out, cloud)
        twin = [{k: (float(v) if k not in ("pair", "model", "role") else v) for k, v in r.items()}
                for f in args.twin for r in csv.DictReader(open(f, encoding="utf-8"))]
        figures_wave(cv, rows, args.out, twin)
    if args.compare_probes:
        cmp = compare_devices(rows, load_rows(args.compare_probes))
        write_csv(args.out / "devices.csv", cmp)
        forms["X"] = {f: float(np.nanmax(np.abs([c[f"d_{f}"] for c in cmp]))) if cmp else float("nan") for f in COMPARE_FIELDS}
    (args.out / "forms.json").write_text(json.dumps(forms, indent=1, default=float), encoding="utf-8")
    print(f"[E8 analysis] {len(rows)} rows, {len(cv)} curve points -> {args.out}; R held {forms['R']['held']}, M held {forms['M']['held']}, "
          f"P2 {forms['P2']['held']}, P4 {forms['P4_overall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
