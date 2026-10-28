"""E6 / E7 on all agents (docs/e6/DECISIONS.md (63), report section 21): the tables and the expected forms. Reads the run
directories read-only (run_all_cpu.py's scratch/all/<job>/...) and E6/E7's development runs on E4's weights (form 1);
reuses E7's analysis functions (D7's C measures, the anti-oracle's task 1, the 2 x 2 table, D7's correction analysis).

Outputs in --out:
  pairs.csv            one row per (agent, image, model): E6's b at step 1, validity, localisation, directed share and its
                       interval, the probe's b, C1 / C2 (D7's per-step measures, main z_bar, mean over the zone steps
                       measured and well localised) and their steps
  identity_e4.csv      E4's agents against the development runs: E6 (training image), the oracle, the anti-oracle
  t1_runs.csv          the anti-oracle's runs (analyse_e7_dissociation.task1), every agent
  table_2x2.csv        the 2 x 2 per agent (analyse_e7_dissociation.table_2x2; E6's row from this campaign's E6 runs)
  d7/                  analyse_e7_d7.py --no-identity on the training image (natural) and the correction A runs
  forms.json           the five forms' numbers and verdicts

Usage, from the repo root: python experiments/E6_phenotypes_latent/analyse_all.py --dev <e6dev results dir> --out <dir>
                           [--parts e6 c12 e7]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E7_latent_manipulation"))

import analyse_e7_d7 as d7  # noqa: E402
import analyse_e7_diagnostics as ad  # noqa: E402
import analyse_e7_dissociation as adis  # noqa: E402
from helplessness.config import REPO_ROOT  # noqa: E402
from helplessness.pipeline import frozen_setup as fs  # noqa: E402

ALL = REPO_ROOT / "scratch" / "all"
DEV_ID = {"p1-a_e4": "p1-a-d1_e4", "base_A": "p1-b-d1_e4", "p2-a_e4": "p2-a-d1_e4", "p2-b_e4": "p2-b-d1_e4"}
ROLE_GROUP = {"exposed": "exposed", "reference": "references, no intervention"}
INTER = (-1.5, -0.5)          # intermediate b at step 1: (lo, hi]
C1_MID = (0.50, 0.85)         # intermediate C1 / directed share: open interval
LOC_MIN = 0.9


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def done(d: str) -> bool:
    return Path(d, "status.json").is_file() and json.loads(Path(d, "status.json").read_text())["status"] == "done"


def e6_runs(job_glob: str = "e6_*") -> list[str]:
    return sorted(d for d in glob.glob(str(ALL / job_glob / "E6_phenotypes_latent" / "e6__*")) if done(d))


def e6_rows() -> list[dict]:
    return [r for d in e6_runs() for r in json.loads(Path(d, "rows.json").read_text(encoding="utf-8"))]


def c12_rows(cfg: dict) -> list[dict]:
    """C1 / C2 per (agent, image, model): D7's per-step measures (main z_bar) over the zone steps measured and localised."""
    out = []
    for run in adis.runs_of([str(ALL / "c12_*" / "E7_dissociation" / "e7d7__*")]):
        meta = run["meta"]
        S = d7.collect_natural([run], {meta["agent"]: ad.World(cfg, meta)})
        for role, group in ROLE_GROUP.items():
            m = (S["group"] == group) & S["localised"]
            out.append({"agent": meta["agent"], "image": meta["image"], "role": role, "C1": float(np.nanmean(S["draw_C1"][m])) if m.any() else np.nan,
                        "C2": float(np.nanmean(S["draw_C2"][m])) if m.any() else np.nan, "n_steps_C": int(m.sum()),
                        "b_local_mean": float(np.nanmean(S["b"][m])) if m.any() else np.nan})
    return out


def pairs_table(agents: list[dict], selection: list[dict], e6: list[dict], c12: list[dict]) -> list[dict]:
    meta = {a["id"]: a for a in agents}
    probe_b = {(r["agent"], r["name"]): float(r["b_probe"]) for r in selection}
    c = {(r["agent"], r["image"], r["role"]): r for r in c12}
    out = []
    for r in e6:
        key = (r["agent"], r["image"], r["role"])
        cr = c.get(key, {})
        out.append({"agent": r["agent"], "condition": meta[r["agent"]]["condition"], "origin": meta[r["agent"]]["origin"], "image": r["image"],
                    "role": r["role"], "b_probe": probe_b.get((r["agent"], r["image"]), np.nan), "b_step1": r["b_step1"], "valid": r["valid"],
                    "loc_template_step1": r["loc_template_step1"], "directed_share": r["directed_share"],
                    "directed_lo": r["directed_share_lo"], "directed_hi": r["directed_share_hi"],
                    "C1": cr.get("C1", np.nan), "C2": cr.get("C2", np.nan), "n_steps_C": cr.get("n_steps_C", 0)})
    return sorted(out, key=lambda r: (r["agent"], r["image"], r["role"]))


def identity_e4(dev: Path) -> list[dict]:
    out = []
    for new, old in DEV_ID.items():
        nd = e6_runs(f"e6_{new}_training")
        od = sorted(glob.glob(str(dev / "E6_phenotypes_latent" / f"e6__{old}__*")))
        if not (nd and od):
            continue
        a, b = np.load(Path(nd[-1]) / "episodes.npz"), np.load(Path(od[-1]) / "episodes.npz")
        keys = [k for k in a.files if k.startswith("training__")]
        ra = {r["role"]: r for r in json.loads(Path(nd[-1], "rows.json").read_text())}
        rb = {r["role"]: r for r in json.loads(Path(od[-1], "rows.json").read_text()) if r["image"] == "training"}
        out.append({"agent": new, "part": "E6 training", "arrays_identical": sum(k in b.files and np.array_equal(a[k], b[k], equal_nan=True) for k in keys),
                    "arrays": len(keys), "rows_identical": all(ra[x][f] == rb[x][f] for x in ra for f in ("directed_share", "b_step1", "loc_template_step1"))})
    return out


def is_inter(b: float) -> bool:
    return INTER[0] < b <= INTER[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=Path, required=True, help="E6/E7's development runs on E4's weights (results root)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", default=["e6", "c12", "e7"])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = fs.load_layered_config(REPO_ROOT / "configs" / "e7" / "e7_all.yaml", ("e6", "e7"))
    agents = yaml.safe_load((REPO_ROOT / "configs" / "e6" / "agents_e5.yaml").read_text(encoding="utf-8"))["agents"]
    with open(REPO_ROOT / "experiments" / "E6_phenotypes_latent" / "selection_all" / "selected_images.csv", newline="", encoding="utf-8") as fh:
        selection = list(csv.DictReader(fh))
    forms: dict = {}
    identity = identity_e4(args.dev)
    c12 = c12_rows(cfg) if "c12" in args.parts else []
    pairs = pairs_table(agents, selection, e6_rows(), c12)
    write_csv(args.out / "pairs.csv", pairs)

    # form 2: intermediate pairs of the shuffle agents, their C1
    exp = [r for r in pairs if r["role"] == "exposed" and r["valid"] and r["loc_template_step1"] >= LOC_MIN]
    inter = [r for r in exp if is_inter(r["b_step1"])]
    by_agent = {a["id"]: sum(r["agent"] == a["id"] for r in inter) for a in agents}
    c1_inter = [r["C1"] for r in inter if np.isfinite(r["C1"])]
    rho = spearmanr([r["b_step1"] for r in inter if np.isfinite(r["C1"])], c1_inter).statistic if len(c1_inter) > 2 else np.nan
    share_mid = float(np.mean([C1_MID[0] < c < C1_MID[1] for c in c1_inter])) if c1_inter else np.nan
    forms["2"] = {"intermediate_pairs_by_agent": by_agent, "n_intermediate_with_C1": len(c1_inter), "share_C1_intermediate": share_mid,
                  "rho_b_C1_intermediate": float(rho),
                  "verdict": bool(all(by_agent[a] >= 3 for a in ("f1_E", "f1_F")) and share_mid >= 0.5 and rho >= 0.5)}

    if "e7" in args.parts:
        worlds = {a["id"]: ad.World(cfg, {"configuration": a["configuration"], "slot": "nw"}) for a in agents}  # training image: the UD slot
        dis = adis.runs_of([str(ALL / "anti_*" / "E7_dissociation" / "e7dis__*")])
        t1, t1_identity = adis.task1(dis, worlds, [str(ALL / "e6_*_training" / "E6_phenotypes_latent" / "e6__*")], np.random.default_rng(7110))
        write_csv(args.out / "t1_runs.csv", t1)
        write_csv(args.out / "t1_identity.csv", t1_identity)
        diag = adis.runs_of([str(ALL / "oracle_*" / "E7_latent_manipulation" / "e7diag__*")])
        e6_train = [{"agent": r["agent"], "model": "exposed" if r["role"] == "exposed" else "reference",
                     "mode": "none" if r["role"] == "exposed" else "reverse_none", "directed_share": r["directed_share"]}
                    for r in e6_rows() if r["image"] == "training"]
        table = adis.table_2x2(t1, diag + [{"meta": {"phase": "induction"}, "summary": {"rows": e6_train}}])
        write_csv(args.out / "table_2x2.csv", table)
        # form 1 (E7 parts): the oracle and the anti-oracle of E4's agents against the development runs
        dev_diag = adis.runs_of([str(args.dev / "E7_latent_manipulation" / "e7diag__*")])
        dev_dis = [r for r in adis.runs_of([str(args.dev / "E7_dissociation" / "e7dis__*")]) if r["meta"]["h_mode"] == "mean"]
        for new, old in DEV_ID.items():
            la_new = {(r["model"], r["k"]): r["directed_share"] for run in diag if run["meta"]["agent"] == new for r in run["summary"]["rows"]}
            la_old = {(r["model"], r["k"]): r["directed_share"] for run in dev_diag if run["meta"]["agent"] == old and run["meta"]["phase"] == "lookahead"
                      for r in run["summary"]["rows"]}
            an_new = {(r["model"], r["p"], r["k"]): r["directed_share"] for run in dis if run["meta"]["agent"] == new for r in run["summary"]["rows"]}
            an_old = {(r["model"], r["p"], r["k"]): r["directed_share"] for run in dev_dis if run["meta"]["agent"] == old for r in run["summary"]["rows"]}
            for part, a, b in (("oracle", la_new, la_old), ("anti-oracle", an_new, an_old)):
                common = sorted(set(a) & set(b), key=str)
                identity.append({"agent": new, "part": part, "arrays_identical": sum(a[k] == b[k] for k in common), "arrays": len(common),
                                 "rows_identical": bool(common) and all(a[k] == b[k] for k in common)})
        # form 3: intermediate pairs against their agent's anti-oracle k = 1 curve at equal b
        gaps = []
        for r in inter:
            cur = sorted((x["b_eff"], x["directed_share"]) for x in t1 if x["agent"] == r["agent"] and x["model"] == "reference"
                         and x["k"] == 1 and x["h_mode"] == "mean")
            if cur:
                bx, dy = zip(*cur)
                gaps.append({"agent": r["agent"], "image": r["image"], "b_step1": r["b_step1"], "directed": r["directed_share"],
                             "anti_oracle_at_b": float(np.interp(r["b_step1"], bx, dy)), "C1": r["C1"]})
        write_csv(args.out / "form3_gaps.csv", gaps)
        share_dir_mid = float(np.mean([C1_MID[0] < g["directed"] < C1_MID[1] for g in gaps])) if gaps else np.nan
        near = float(np.mean([abs(g["directed"] - g["anti_oracle_at_b"]) <= 0.10 for g in gaps])) if gaps else np.nan
        forms["3"] = {"n": len(gaps), "share_directed_intermediate": share_dir_mid, "share_within_0.10_of_anti_oracle": near,
                      "verdict": bool(gaps) and share_dir_mid >= 0.5 and near >= 0.5}
        # form 4: the 2 x 2 of the new agents
        f4 = {}
        for a in ("base_B", "f1_E", "f1_F"):
            row = next((t for t in table if t["agent"] == a and t["k"] == 1), None)
            if row is None:
                continue
            tb_r, tb_e = row["true_belief__ref_critic"], row["true_belief__exp_critic (oracle D4)"]
            cv_r, cv_e = row["conveyor__ref_critic (anti-oracle p=1)"], row["conveyor__exp_critic"]
            belief = float(np.mean([tb_r - cv_r, tb_e - cv_e]))
            critic = float(tb_r - tb_e)
            f4[a] = {"belief_effect": belief, "critic_effect": critic, "holds": bool(belief >= 0.40 and critic <= 0.20 and critic <= belief / 2)}
        forms["4"] = {**f4, "verdict": bool(f4) and all(v["holds"] for v in f4.values())}
        # form 5: the correction A (D7's analysis without its identity checks)
        nat_train = [d for d in glob.glob(str(ALL / "c12_*_training" / "E7_dissociation" / "e7d7__*")) if done(d)]
        cor = [d for d in glob.glob(str(ALL / "cor_*" / "E7_dissociation" / "e7d7__*")) if done(d)]
        if nat_train and cor:
            d7.main(["--runs", *nat_train, *cor, "--no-identity", "--config", str(REPO_ROOT / "configs" / "e7" / "e7_all.yaml"),
                     "--out", str(args.out / "d7")])
            with open(args.out / "d7" / "d7_correction.csv", newline="", encoding="utf-8") as fh:
                corr = [r for r in csv.DictReader(fh) if r["zbar"] == "draw"]
            c1_ref = {r["agent"]: r["C1"] for r in pairs if r["image"] == "training" and r["role"] == "reference"}
            f5 = {}
            for r in corr:
                b0, b1, c0, c1 = (float(r[k]) for k in ("b_raw", "b_corr", "C1_raw", "C1_corr"))
                fb = (b1 - b0) / (0.0 - b0)
                fc = (c1 - c0) / (c1_ref[r["agent"]] - c0) if c1_ref.get(r["agent"]) is not None and c1_ref[r["agent"]] != c0 else np.nan
                f5[r["agent"]] = {"setting": r["setting"], "b_before": b0, "b_after": b1, "C1_before": c0, "C1_after": c1, "C1_reference": c1_ref.get(r["agent"]),
                                  "gap_closed_b": fb, "gap_closed_C1": fc, "outcome_D7": r["outcome"], "holds": bool(fb > fc)}
            forms["5"] = {**f5, "verdict": all(f5[a]["holds"] for a in ("base_B", "f1_E", "f1_F") if a in f5)}
    write_csv(args.out / "identity_e4.csv", identity)
    forms["1"] = {"rows": identity, "verdict": bool(identity) and all(r["rows_identical"] and r["arrays_identical"] == r["arrays"] for r in identity)}
    (args.out / "forms.json").write_text(json.dumps(forms, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v.get("verdict") for k, v in forms.items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
