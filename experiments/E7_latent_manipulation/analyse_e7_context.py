"""Report section 22's analysis: the verdicts of A (persistence on real steps), B (decoder) and C (natural colour
direction) with the thresholds committed before the computation (c790003), from run_e7_context.py's runs (read-only).

  A  (and A in imagination, 807fcdc: |dh_t| / |dh_0| at t*, the first step where b's gap is back to <= 25 %: (i)
     transmitted but ignored >= 0.5, (ii) erased by the GRU <= 0.2)
     r(t) = [b_corr(t) - b_raw(t)] / [b_corr(0) - b_raw(0)] over the episodes still in their first stay in the zone at t;
     n(3) < 20: undetermined; else context if r(2) <= 0.25, transported if r(3) >= 0.50, else undetermined. C1, C3 and the
     mass on the played action's true successor: the same fraction, described.
  B  s = [D_inc(after) - D_inc(before)] / |T_inc - neutral| on the PREDICTED images, mean over contexts, 95 % bootstrap
     interval over contexts (2,000 draws); shuffle: s >= 0.10 and interval > 0; baselines: |s| <= 0.05 and the predicted
     agent cell's share drops by >= 0.10. The reconstruction, described.
  C  shuffle: split-half cos of d_col >= 0.80 else not interpretable; conditions 1-5 of the section; all five: holds; none
     or only condition 4: does not hold; otherwise partial. Described: cos of d_col between slots, with d_ctrl and d_rec.

Writes <out>/persist.csv, imagine.csv, decoder.csv, coldir.csv, directions.csv, forms.json and figures/context.png. Usage:
  python experiments/E7_latent_manipulation/analyse_e7_context.py --runs scratch/ctx_650k/E7_context --out results/E6_phenotypes_latent/all_agents_650k/context
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

AGENTS = ["p1-a_e4", "base_A", "p2-a_e4", "p2-b_e4", "base_B", "f1_A", "f1_E", "f1_F"]
SHUFFLE = ("f1_A", "f1_E", "f1_F")
AMPS = (0.25, 0.5, 0.75, 1.0, 1.5)
N_BOOT, N_NULL, SEED = 2000, 1000, 2208


def done_runs(root: Path, agent: str, phase: str) -> list[Path]:
    out = []
    for d in sorted(root.glob(f"e7ctx__{agent}__{phase}__*")):
        if json.loads((d / "status.json").read_text(encoding="utf-8"))["status"] == "done":
            out.append(d)
    return out


def one(root: Path, agent: str, phase: str) -> Path:
    found = done_runs(root, agent, phase)
    if len(found) != 1:
        raise SystemExit(f"{agent} {phase}: {len(found)} finished runs")
    return found[0]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def part_a(root: Path) -> tuple[list[dict], dict]:
    rows, forms = [], {}
    for a in AGENTS:
        s = json.loads((one(root, a, "persist") / "summary.json").read_text(encoding="utf-8"))
        by = {r["t"]: r for r in s["by_t"]}
        d0 = {k: by[0][f"{k}_corr"] - by[0][f"{k}_raw"] for k in ("b", "C1", "C3", "mass")}
        for t, r in by.items():
            rows.append({"agent": a, "t": t, "n_in_zone": r["n_in_zone"], **{k: r[k] for k in r if k not in ("t", "n_in_zone")},
                         **{f"r_{k}": (r[f"{k}_corr"] - r[f"{k}_raw"]) / d0[k] if d0[k] else np.nan for k in d0},
                         "directed_once": s["directed_by_index"]["once"][t], "directed_exposed": s["directed_by_index"]["exposed"][t]})
        r_b = {t: (by[t]["b_corr"] - by[t]["b_raw"]) / d0["b"] for t in by}
        n3 = by[3]["n_in_zone"] if 3 in by else 0
        if n3 < 20:
            verdict = "undetermined (n(3) < 20)"
        else:
            ctx, tr = r_b.get(2, np.nan) <= 0.25, r_b.get(3, np.nan) >= 0.50
            verdict = "context" if ctx and not tr else "transported" if tr and not ctx else "both" if ctx and tr else "undetermined"
        forms[a] = {"b_gap_t0": d0["b"], "r2": r_b.get(2), "r3": r_b.get(3), "n2": by.get(2, {}).get("n_in_zone", 0), "n3": n3, "verdict": verdict,
                    "dh": dh_reading([(t, r_b[t], by[t].get("dh_ratio"), by[t].get("dh_cos0")) for t in sorted(by) if by[t]["n_in_zone"] >= 20])}
    return rows, forms


def dh_reading(steps: list[tuple]) -> dict:
    """(t, r_b, dh_ratio, dh_cos0) per step: t* = the first t >= 1 with r_b <= 0.25; (i) transmitted but ignored if the
    ratio at t* >= 0.5, (ii) erased by the GRU if <= 0.2, else (or no t*) undetermined."""
    star = next((s for s in steps if s[0] >= 1 and s[1] <= 0.25), None)
    if star is None or star[2] is None:
        return {"t_star": None, "verdict": "undetermined (no t*)"}
    ratio = star[2]
    verdict = "(i) transmitted but ignored" if ratio >= 0.5 else "(ii) erased by the GRU" if ratio <= 0.2 else "undetermined"
    return {"t_star": star[0], "r_b_t_star": star[1], "dh_ratio_t_star": ratio, "dh_cos0_t_star": star[3], "verdict": verdict}


def part_imagine(root: Path) -> tuple[list[dict], dict]:
    rows, forms = [], {}
    for a in AGENTS:
        s = json.loads((one(root, a, "imagine") / "summary.json").read_text(encoding="utf-8"))
        rows += [{"agent": a, **r} for r in s["by_step"]]
        forms[a] = dh_reading([(r["step"], r["r_b"], r["dh_ratio_in_zone"], r["dh_cos0_all"]) for r in s["by_step"]])
    return rows, forms


def part_b(root: Path) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(SEED)
    rows, forms = [], {}
    for a in AGENTS:
        d = one(root, a, "decoder")
        z = np.load(d / "decoder.npz")
        norm = json.loads((d / "summary.json").read_text(encoding="utf-8"))["inc_minus_neutral"]
        s = (z["pred_D_inc_after"] - z["pred_D_inc_before"]) / norm
        s_rec = (z["rec_D_inc_after"] - z["rec_D_inc_before"]) / norm
        boot = np.array([s[rng.integers(0, len(s), len(s))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        loc_drop = float(z["pred_loc_before"].mean() - z["pred_loc_after"].mean())
        row = {"agent": a, "s_pred": float(s.mean()), "s_pred_lo": float(lo), "s_pred_hi": float(hi), "s_rec": float(s_rec.mean()),
               **{f"{p}_{m}_{t}": float(z[f"{p}_{m}_{t}"].mean()) for p in ("pred", "rec") for m in ("D_inc", "D_ctrl", "D_neu", "loc") for t in ("before", "after")},
               "pred_loc_drop": loc_drop, "delta_norm": float(np.linalg.norm(z["delta"], axis=1).mean())}
        rows.append(row)
        if a in SHUFFLE:
            holds = bool(row["s_pred"] >= 0.10 and lo > 0)
        else:
            holds = bool(abs(row["s_pred"]) <= 0.05 and loc_drop >= 0.10)
        forms[a] = {"s_pred": row["s_pred"], "ci": [float(lo), float(hi)], "pred_loc_drop": loc_drop, "holds": holds}
    return rows, forms


def part_c(root: Path) -> tuple[list[dict], list[dict], dict]:
    rows, drows, forms = [], [], {}
    rng = np.random.default_rng(SEED + 1)
    for a in SHUFFLE:
        parts = done_runs(root, a, "coldir")
        summ = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in parts]
        cond = {r["condition"]: r for s in summ for r in s["rows"]}
        slot = summ[0]["slot"]
        arr = np.load(parts[0] / "episodes.npz")
        dirs = {q: {k: arr[f"{q}__{k}"] for k in ("d_col", "d_ctrl", "d_rec", "d_col_half0", "d_col_half1")} for q in ("nw", "ne", "se", "sw")}
        for c in cond.values():
            rows.append({"agent": a, **c})
        for q, dq in dirs.items():
            drows.append({"agent": a, "slot": q, "norm_d_col": float(np.linalg.norm(dq["d_col"])), "split_half_cos": cos(dq["d_col_half0"], dq["d_col_half1"]),
                          "cos_d_col_d_ctrl": cos(dq["d_col"], dq["d_ctrl"]), "cos_d_col_d_rec": cos(dq["d_col"], dq["d_rec"]),
                          **{f"cos_d_col_{q2}": cos(dq["d_col"], dirs[q2]["d_col"]) for q2 in dirs}})
        reliable = cos(dirs[slot]["d_col_half0"], dirs[slot]["d_col_half1"]) >= 0.80
        e = cond["exposed"]
        bs = [e["b_step1"]] + [cond[f"col_a{x:g}"]["b_step1"] for x in AMPS]
        rho = float(spearmanr([0.0, *AMPS], bs).statistic)
        c1 = rho >= 0.9 and cond["col_a1"]["b_step1"] - e["b_step1"] >= 0.5
        c2 = cond["col_a1"]["C3"] - e["C3"] >= 0.10 and cond["col_a1"]["directed_share"] - e["directed_share"] >= 0.10
        c3 = all(cond[f"col_a{x:g}"]["loc_template_zone"] >= e["loc_template_zone"] - 0.10 for x in AMPS if x <= 1.0)
        rand = [cond[f"rand{j}_a1"] for j in range(5)]
        med = {k: float(np.median([abs(r[k] - e[k]) for r in rand])) for k in ("b_step1", "C3", "directed_share")}
        c4 = med["b_step1"] <= 0.20 and med["C3"] <= 0.05 and med["directed_share"] <= 0.05
        dec = np.load(one(root, a, "decoder") / "decoder.npz")["delta"]
        v = -dirs[slot]["d_col"]
        obs = float(np.mean([cos(x, v) for x in dec]))
        null = []
        for _ in range(N_NULL):
            u = rng.standard_normal(len(v))
            null.append(np.mean(dec @ u / (np.linalg.norm(dec, axis=1) * np.linalg.norm(u))))
        p95 = float(np.percentile(null, 95))
        c5 = obs > p95
        held = [c1, c2, c3, c4, c5]
        n_held = sum(held)
        verdict = ("not interpretable (split-half cos < 0.80)" if not reliable else "holds" if n_held == 5
                   else "does not hold" if n_held == 0 or held == [False, False, False, True, False] else "partial")
        forms[a] = {"slot": slot, "split_half_cos": cos(dirs[slot]["d_col_half0"], dirs[slot]["d_col_half1"]), "rho_b": rho,
                    "b_gain_a1": cond["col_a1"]["b_step1"] - e["b_step1"], "C3_gain_a1": cond["col_a1"]["C3"] - e["C3"],
                    "directed_gain_a1": cond["col_a1"]["directed_share"] - e["directed_share"],
                    "loc_by_a": {f"{x:g}": cond[f"col_a{x:g}"]["loc_template_zone"] for x in AMPS}, "loc_exposed": e["loc_template_zone"],
                    "random_median_abs_delta_a1": med, "cos_dh_minus_dcol": obs, "cos_null_p95": p95,
                    "conditions": dict(zip(("1", "2", "3", "4", "5"), map(bool, held))), "verdict": verdict}
    return rows, drows, forms


def figure(a_rows: list[dict], c_rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for a in AGENTS:
        rr = [r for r in a_rows if r["agent"] == a and r["n_in_zone"] >= 20]
        axes[0].plot([r["t"] for r in rr], [r["r_b"] for r in rr], marker="^" if a in SHUFFLE else "o", label=a)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].axhline(0.25, color="#888888", ls="--")
    axes[0].axhline(0.5, color="#888888", ls=":")
    axes[0].set_xlabel("step t after the one-shot correction")
    axes[0].set_ylabel("r(t), b")
    axes[0].legend(fontsize=7)
    for ax, key, lab in zip(axes[1:], ("b_step1", "C3", "directed_share"), ("b at step 1 (bits)", "C3", "directed share")):
        for a in SHUFFLE:
            cond = {r["condition"]: r for r in c_rows if r["agent"] == a}
            ax.plot([0.0, *AMPS], [cond["exposed"][key]] + [cond[f"col_a{x:g}"][key] for x in AMPS], marker="^", label=f"{a} -d_col")
            ax.scatter([1.0] * 5 + [1.5] * 5, [cond[f"rand{j}_a{x:g}"][key] for x in (1.0, 1.5) for j in range(5)], marker="x", color="grey")
        ax.set_xlabel("amplitude a (x |d_col|)")
        ax.set_ylabel(lab)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    a_rows, a_forms = part_a(args.runs)
    i_rows, i_forms = part_imagine(args.runs)
    write_csv(args.out / "imagine.csv", i_rows)
    b_rows, b_forms = part_b(args.runs)
    c_rows, d_rows, c_forms = part_c(args.runs)
    write_csv(args.out / "persist.csv", a_rows)
    write_csv(args.out / "decoder.csv", b_rows)
    write_csv(args.out / "coldir.csv", c_rows)
    write_csv(args.out / "directions.csv", d_rows)
    forms = {"A": a_forms, "A_imagine": i_forms, "B": b_forms, "C": c_forms}
    (args.out / "forms.json").write_text(json.dumps(forms, indent=1, default=float), encoding="utf-8")
    figure(a_rows, c_rows, args.out / "figures" / "context.png")
    print(json.dumps({p: {a: f.get("verdict", f.get("holds")) for a, f in forms[p].items()} for p in forms}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
