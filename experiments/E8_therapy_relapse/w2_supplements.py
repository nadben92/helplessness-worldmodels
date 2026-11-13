"""E8 wave 2's supplements to analyze_e8.py (DECISIONS (26) and the user's reading of R and H, 2026-11-13). Descriptive
only: no rule of EXPECTED_FORMS is changed and no verdict is computed here. Reads read-only the probe jobs, the therapy runs
and analyze_e8.py's outputs (--analysis: curves.csv, recovery.csv, forms.json); writes light outputs.

  validity.csv         per pair, image and checkpoint: the reference's mass and I at step 1 (E4's criterion: mass >= 0.90,
                       |I_ref - log2 5| <= 0.15), the point's validity (walks_valid, the one every curve, t½ and P4 read)
  validity_counts.csv  per assignment and checkpoint, the number of invalid points over the 3 pairs
  drift.csv            per pair, untreated image and checkpoint: I, mass and D of the exposed model and of the reference
                       (R's drift of I_ref; the exposed model's I and mass beside it, exploratory)
  dose.csv             per pair and group, every t½ and crossing of recovery.csv and every P4 crossing, in environment steps,
                       gradient steps and treated-zone visits of the exposed run; flagged when inside F's H window
  visit_rates.csv      treated-zone visits per environment step between consecutive checkpoints, exposed runs and reference B
  cache_A.json         shuffle A's cache effect at checkpoint 0 (D_ref - D_oracle1) with reference B (E8) and with reference A
                       (the 650,000 campaign's pair), on the training image (assignment01) and the treated mean
  profile.csv          EXPLORATORY (user, 2026-11-13: transfer or forgetting?): per pair, untreated image and checkpoint, the
                       exposed model's mass on the true successor in the image's T_inc quadrant, C3, the decoder's
                       localisation (L_dec: decoded cell = true cell, zone steps), the directed share from the true cell on
                       the well-localised steps, I and D, beside the reference's
  profile_crossings.csv  EXPLORATORY: per pair and untreated image, the first valid checkpoint t > 0 at which each reaches the
                       reference's level (g_D <= 0.05 as P4; g_C3 <= 0.05; mass >= 0.90 and |I - log2 5| <= 0.15, E4's
                       levels; g_Ldec <= 0.05), not reached: "> T"; the mass and C3 at D's crossing; the largest fall of the
                       mass below its checkpoint-0 value (a forgetting would show there)
  figures/validity_drift.png, figures/dose_crossings.png, figures/transfer_profile.png

Usage (from the repo root):
  python experiments/E8_therapy_relapse/w2_supplements.py --probes "<wave 2 probes>/probes__*" --runs "<wave 2 runs>/grid12x12__*" \\
      --analysis results/E8_therapy_relapse/e8_w2_shuf_nw --campaign-e6-A <campaign e6__f1_A__... dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import analyze_e8 as analyze

PAIRS = ("shuffle_A", "shuffle_E", "shuffle_F")
REF_MODEL = "shuffle_B_ref"
H_WINDOW_F = (5000, 25000)  # form H of shuffle_F: mass below 0.90 in assignment 0 (ne, corridor) at these checkpoints
TOL_I, MASS = 0.15, 0.90
COLORS = {"treated": "#2a78d6", "drifting": "#eb6834", "untreated": "#9a9a94"}  # categorical slots 1, 2; muted gray
NW_CD = ("assignment03", "assignment06", "assignment09")  # the untreated assignments whose nw slot is CD (T_ctrl's colour)
PROFILE = {"mass": "walks_mass_step1", "C3": "C3_loc", "L_dec": "loc_loc_cell", "dir_loc": "loc_directed_localised",
           "I": "walks_I_step1", "D": "directed_share"}
SERIES = {"D": "#2a78d6", "C3": "#eb6834", "L_dec": "#1baf7a", "mass": "#eda100", "I / log2 5": "#e87ba4"}  # slots 1-5, fixed


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validity(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    out = []
    for r in sorted((r for r in rows if r["role"] == "reference" and r["pair"] in PAIRS), key=lambda r: (r["pair"], r["assignment"], r["checkpoint"])):
        mass, i = float(r["walks_mass_step1"]), float(r["walks_I_step1_s1"])
        out.append({"pair": r["pair"], "image": r["image"], "assignment": r["assignment"], "treated": r["treated"], "slot": r["slot"],
                    "checkpoint": r["checkpoint"], "ref_mass_step1": mass, "ref_I_step1": i,
                    "mass_ok": mass >= MASS, "I_ok": abs(i - math.log2(5)) <= TOL_I, "valid": r["walks_valid"]})
    counts = []
    for k in sorted({v["assignment"] for v in out}):
        for t in sorted({v["checkpoint"] for v in out}):
            sel = [v for v in out if v["assignment"] == k and v["checkpoint"] == t]
            counts.append({"assignment": k, "treated": sel[0]["treated"], "checkpoint": t, "n_pairs": len(sel),
                           "n_invalid": sum(not v["valid"] for v in sel), "n_mass_below": sum(not v["mass_ok"] for v in sel),
                           "n_I_out": sum(not v["I_ok"] for v in sel)})
    return out, counts


def drift(rows: list[dict]) -> list[dict]:
    key = {(r["pair"], r["image"], r["checkpoint"], r["role"]): r for r in rows if r["pair"] in PAIRS}
    out = []
    for (pair, image, t, role), r in sorted(key.items()):
        if role != "exposed" or r["treated"]:
            continue
        ref = key[(pair, image, t, "reference")]
        out.append({"pair": pair, "image": image, "assignment": r["assignment"], "slot": r["slot"], "checkpoint": t, "valid": r["walks_valid"],
                    "I_ref": ref["walks_I_step1"], "mass_ref": ref["walks_mass_step1"], "D_ref": ref["directed_share"],
                    "I_exp": r["walks_I_step1"], "mass_exp": r["walks_mass_step1"], "D_exp": r["directed_share"]})
    return out


def profile(rows: list[dict]) -> list[dict]:
    key = {(r["pair"], r["image"], r["checkpoint"], r["role"]): r for r in rows if r["pair"] in PAIRS}
    out = []
    for (pair, image, t, role), r in sorted(key.items()):
        if role != "exposed" or r["treated"]:
            continue
        ref = key[(pair, image, t, "reference")]
        rec = {"pair": pair, "image": image, "nw_role": "cd" if image in NW_CD else "neutral", "checkpoint": t, "valid": r["walks_valid"]}
        for name, field in PROFILE.items():
            rec[f"{name}_exp"], rec[f"{name}_ref"] = float(r[field]), float(ref[field])
        out.append(rec)
    return out


def profile_crossings(prof: list[dict]) -> list[dict]:
    out = []
    for pair in PAIRS:
        for image in sorted({p["image"] for p in prof if p["pair"] == pair}):
            ser = sorted((p for p in prof if p["pair"] == pair and p["image"] == image), key=lambda p: p["checkpoint"])
            T, ok = ser[-1]["checkpoint"], [p for p in ser[1:] if p["valid"]]
            first = lambda cond: next((p["checkpoint"] for p in ok if cond(p)), f"> {T}")
            rec = {"pair": pair, "image": image, "nw_role": ser[0]["nw_role"], "valid_at_0_and_T": ser[0]["valid"] and ser[-1]["valid"],
                   "D": first(lambda p: p["D_ref"] - p["D_exp"] <= 0.05), "C3": first(lambda p: p["C3_ref"] - p["C3_exp"] <= 0.05),
                   "L_dec": first(lambda p: p["L_dec_ref"] - p["L_dec_exp"] <= 0.05), "mass": first(lambda p: p["mass_exp"] >= MASS),
                   "I": first(lambda p: abs(p["I_exp"] - math.log2(5)) <= TOL_I)}
            at_d = next((p for p in ok if p["checkpoint"] == rec["D"]), None)
            rec.update({"mass_at_D": at_d["mass_exp"] if at_d else None, "mass_ref_at_D": at_d["mass_ref"] if at_d else None,
                        "C3_gap_at_D": (at_d["C3_ref"] - at_d["C3_exp"]) if at_d else None,
                        "mass_0": ser[0]["mass_exp"], "mass_max_fall_below_0": max(0.0, max(ser[0]["mass_exp"] - p["mass_exp"] for p in ser))})
            out.append(rec)
    return out


def in_window(t) -> bool:
    return isinstance(t, (int, np.integer)) and H_WINDOW_F[0] <= int(t) <= H_WINDOW_F[1]


def dose(cv: list[dict], recov: list[dict], forms: dict) -> list[dict]:
    axis = {(c["pair"], int(c["checkpoint"])): (int(c["grad_step"]), int(c["treated_visits"]) if c["treated_visits"] not in ("", "None") else None)
            for c in cv if c["group"] == "treated"}

    def entry(pair, group, what, t):
        t = int(t) if str(t).lstrip("-").isdigit() else t
        g, v = axis.get((pair, t), (None, None)) if isinstance(t, int) else (None, None)
        return {"pair": pair, "group": group, "what": what, "env_steps": t, "grad_steps": g, "treated_visits": v,
                "in_F_H_window": pair == "shuffle_F" and in_window(t)}

    out = []
    for r in recov:
        for k, v in r.items():
            if k.startswith(("t_half", "cross")) and v != "":
                out.append(entry(r["pair"], r["group"], k, v))
    for pair, p4 in forms["P4"].items():
        for group in ("treated", "untreated"):
            for image, rec in p4["crossings"][group].items():
                for k, t in rec.items():
                    out.append(entry(pair, f"{group} {image}", f"P4 {k}", t))
    return out


def visit_rates(visits: dict) -> list[dict]:
    out = []
    for model in (*(f"{p}_exp" for p in PAIRS), REF_MODEL):
        v = visits[model]
        ts = sorted(v)
        for a, b in zip(ts, ts[1:]):
            out.append({"model": model, "from": a, "to": b, "visits": v[b] - v[a], "rate": (v[b] - v[a]) / (b - a)})
    return out


def cache_A(rows: list[dict], campaign_e6: Path) -> dict:
    at0 = [r for r in rows if r["pair"] == "shuffle_A" and r["checkpoint"] == 0]
    exp = {r["image"]: r for r in at0 if r["role"] == "exposed"}
    ref = {r["image"]: r for r in at0 if r["role"] == "reference"}
    camp = json.loads((campaign_e6 / "rows.json").read_text(encoding="utf-8"))
    camp_ref = next(r for r in camp if r["image"] == "training" and r["role"] == "reference")
    o = float(exp["assignment01"]["oracle_directed_share"])  # the oracle runs on the training image only
    return {"image": "assignment01 (training image)", "D_oracle1": o,
            "D_ref_B (E8)": float(ref["assignment01"]["directed_share"]), "D_ref_A (campaign)": float(camp_ref["directed_share"]),
            "cache_effect_ref_B": float(ref["assignment01"]["directed_share"]) - o, "cache_effect_ref_A": float(camp_ref["directed_share"]) - o,
            "E7_cited": 0.24}


def figures(val: list[dict], dr: list[dict], ds: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "figures").mkdir(exist_ok=True)
    drifting = {"assignment03", "assignment06", "assignment09"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    panels = (("ref_I_step1", "I_ref at step 1 (bits)", "ref"), ("ref_mass_step1", "reference mass on the true successor", "ref"),
              ("I_exp", "I_exp at step 1 (bits), untreated", "exp"), ("mass_exp", "exposed mass on the true successor, untreated", "exp"))
    pa = "shuffle_A"
    for ax, (field, label, who) in zip(axes.flat, panels):
        src = [v for v in val if v["pair"] == pa] if who == "ref" else [d for d in dr if d["pair"] == pa]
        for image in sorted({v["image"] for v in src}):
            s = sorted((v for v in src if v["image"] == image), key=lambda v: v["checkpoint"])
            kind = "treated" if s[0].get("treated") else "drifting" if image in drifting else "untreated"
            ax.plot([v["checkpoint"] for v in s], [float(v[field]) for v in s], marker="o", ms=3, lw=2 if kind != "untreated" else 1.2,
                    color=COLORS[kind], label=kind, zorder=3 if kind != "untreated" else 2)
        if field == "ref_I_step1":
            ax.axhline(math.log2(5) - TOL_I, color="#555", lw=1, ls=":")
        if field == "ref_mass_step1":
            ax.axhline(MASS, color="#555", lw=1, ls=":")
        ax.set_xscale("symlog", linthresh=500)
        ax.set_xlim(left=0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    h, lab = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    axes[0, 0].legend([uniq[k] for k in ("treated", "drifting", "untreated") if k in uniq],
                      ["treated (0-2, nw = UD)", "untreated 3, 6, 9 (nw = CD)", "other untreated (nw neutral)"], fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("therapy environment steps (symlog)")
    fig.suptitle(f"{pa} pair (reference B's walks with {pa}'s seeds); dotted: E4's validity bounds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "validity_drift.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (abscissa, label) in zip(axes, (("env_steps", "environment steps"), ("treated_visits", "visits of nw, exposed run"))):
        for y, pair in enumerate(PAIRS):
            for group, color in (("treated", COLORS["treated"]), ("untreated", COLORS["drifting"])):
                pts = [d for d in ds if d["pair"] == pair and d["what"] == "P4 g_D<=0.05" and d["group"].startswith(group)
                       and isinstance(d[abscissa], int)]
                jitter = np.linspace(-0.12, 0.12, len(pts)) if pts else []
                ax.scatter([d[abscissa] for d in pts], [y + (0.18 if group == "untreated" else -0.18) + j for d, j in zip(pts, jitter)],
                           s=40, color=color, edgecolor="white", linewidth=1, label=group if y == 0 else None, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(f"first checkpoint with g_D <= 0.05 ({label})")
        ax.grid(alpha=0.25, axis="x")
    axes[0].axvspan(*H_WINDOW_F, ymin=0.67, ymax=1.0, color="#9a9a94", alpha=0.15, lw=0)
    axes[0].set_yticks(range(len(PAIRS)), PAIRS)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("P4 crossings per image (valid checkpoints only; no untreated point is valid at 25,000); shaded: F's H window", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "dose_crossings.png", dpi=120)
    plt.close(fig)


def figure_profile(prof: list[dict], out: Path) -> None:
    """Exposed model on the untreated images, means per checkpoint: top, the 6 images where nw is neutral (valid points
    only); bottom, the 3 where nw is CD (every point, all invalid from 15,000 on: exploratory). Dotted: the reference's mean."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
    get = {"D": "D", "C3": "C3", "L_dec": "L_dec", "mass": "mass", "I / log2 5": "I"}
    for col, pair in enumerate(PAIRS):
        for row, role in enumerate(("neutral", "cd")):
            ax = axes[row, col]
            sel = [p for p in prof if p["pair"] == pair and p["nw_role"] == role and (p["valid"] or role == "cd")]
            ts = sorted({p["checkpoint"] for p in sel})
            for name, key in get.items():
                scale = math.log2(5) if key == "I" else 1.0
                m = [np.mean([p[f"{key}_exp"] for p in sel if p["checkpoint"] == t]) / scale for t in ts]
                ax.plot(ts, m, marker="o", ms=3, lw=2, color=SERIES[name], label=name)
                if key in ("D", "mass"):
                    r = [np.mean([p[f"{key}_ref"] for p in sel if p["checkpoint"] == t]) / scale for t in ts]
                    ax.plot(ts, r, ls=":", lw=1.2, color=SERIES[name])
            ax.set_xscale("symlog", linthresh=500)
            ax.set_xlim(left=0)
            ax.grid(alpha=0.25)
            what = "neutral (6 images, valid points)" if role == "neutral" else "CD (3 images, all points)"
            ax.set_title(f"{pair}, nw {what}", fontsize=9)
        axes[1, col].set_xlabel("therapy environment steps (symlog)")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    for r in range(2):
        axes[r, 0].set_ylabel("exposed (dotted: reference)")
    fig.suptitle("EXPLORATORY: transfer or forgetting? Exposed model on the untreated images (no valid neutral point at 25,000)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "transfer_profile.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--analysis", type=Path, required=True, help="analyze_e8.py's --out of wave 2 (read; the outputs go there too)")
    parser.add_argument("--campaign-e6-A", type=Path, required=True, help="the 650,000 campaign's e6__f1_A__... job (reference A)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    rows = analyze.load_rows(args.probes)
    cv, recov = read_csv(args.analysis / "curves.csv"), read_csv(args.analysis / "recovery.csv")
    forms = json.loads((args.analysis / "forms.json").read_text(encoding="utf-8"))
    val, counts = validity(rows)
    dr, ds = drift(rows), dose(cv, [r for r in recov if r["pair"] in PAIRS], forms)
    analyze.write_csv(args.analysis / "validity.csv", val)
    analyze.write_csv(args.analysis / "validity_counts.csv", counts)
    analyze.write_csv(args.analysis / "drift.csv", dr)
    analyze.write_csv(args.analysis / "dose.csv", ds)
    analyze.write_csv(args.analysis / "visit_rates.csv", visit_rates(analyze.load_visits(args.runs)))
    prof = profile(rows)
    analyze.write_csv(args.analysis / "profile.csv", prof)
    analyze.write_csv(args.analysis / "profile_crossings.csv", profile_crossings(prof))
    (args.analysis / "cache_A.json").write_text(json.dumps(cache_A(rows, args.campaign_e6_A), indent=1), encoding="utf-8")
    if not args.no_figures:
        figures(val, dr, ds, args.analysis)
        figure_profile(prof, args.analysis)
    print(f"[E8 w2 supplements] {len(val)} reference points, {sum(not v['valid'] for v in val)} invalid -> {args.analysis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
