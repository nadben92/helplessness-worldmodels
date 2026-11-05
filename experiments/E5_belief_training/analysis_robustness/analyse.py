"""The robustness diagnostic's tables and verdicts (PREDICTIONS.md, rules committed before the measurements), from the
cluster localisation (tables/localisation.csv), mechanism.py's per-walk outputs (--raw, the newest directory of each
model and set), the training records (tables/history_summary.csv).

Every ratio of a cell (image, zone, part) is sum / n over its walks; a curve point is the mean over the retained swaps
of the mean over the zones concerned (all four, the moved slot m, the other three). Uncertainty: 200 bootstrap
replicates of the 100 walks, the same resampled indices for every image, zone and model (paired); SE = their standard
deviation. The latent MI is the plug-in MI of latent_ceiling.mutual_information, with walk weights for the replicates.

Outputs in --out: scores_restricted.csv, test0.csv, latent.csv (per variable), latent_summary.csv, sensitivity.csv,
step4.csv, models_summary.csv, reading.json (the verdicts, rule by rule).

Usage, from the repo root:
  python experiments/E5_belief_training/analysis_robustness/analyse.py --raw <mechanism.py out-dir>
      --out experiments/E5_belief_training/analysis_robustness/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # this experiment's latent_ceiling.py

import common  # noqa: E402
import latent_ceiling  # noqa: E402

N_BOOT = 200
BOOT_SEED = 5005
LOCALISED, LOST = 0.95, 0.85  # PREDICTIONS.md, band rule
ACTIVE_BITS, RATIO = 0.3, 0.25
MOVES_LITTLE = 0.10
L_MIN_DENOMINATOR = 0.02
PART = {"corridor": 0, "zone": 1, "step1": 2}
MECHANISTIC = ("baseA_exp_k300k", "baseA_ref_k300k", "baseB_exp_k300k", "baseB_ref_k300k", "A", "ref_A", "E", "F", "ref_B")
PAIRS = (("baseA_exp_k300k", "baseA_ref_k300k"), ("baseB_exp_k300k", "baseB_ref_k300k"), ("A", "ref_A"), ("E", "ref_B"), ("F", "ref_B"))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def band(v: float) -> str:
    return "nan" if not np.isfinite(v) else "localised" if v >= LOCALISED else "lost" if v < LOST else "indeterminate"


# ---------------------------------------------------------------------------------------------- step 1: restricted scores
def restricted_scores(loc_csv: Path) -> tuple[list[dict], dict, list[dict]]:
    """Per model and axis: the scores on the swaps localised at t = 0 (PREDICTIONS.md section 0), and the se swap at
    t = 0 apart. Returns (rows, {model: retained swaps}, se rows)."""
    rows = [{**r, "t": float(r["t"]), "loc": float(r["loc"])} for r in csv.DictReader(open(loc_csv, encoding="utf-8"))]
    out, retained, se_rows = [], {}, []
    for mid in dict.fromkeys(r["id"] for r in rows):
        mine = [r for r in rows if r["id"] == mid]
        t0 = {s: np.mean([r["loc"] for r in mine if r["axis"] == "arc_ctrl" and r["t"] == 0.0 and r["swap"] == s]) for s in ("ne", "se", "sw")}
        keep = [s for s, v in t0.items() if v >= common.LOCALISATION_MIN]
        retained[mid] = keep
        se_rows.append({"id": mid, "se_t0_mean_over_zones": round(float(t0["se"]), 4), "se_retained": "se" in keep,
                        **{f"se_t0_{r['zone']}": r["loc"] for r in mine if r["axis"] == "arc_ctrl" and r["t"] == 0.0 and r["swap"] == "se"}})
        for axis in common.AXES:
            sub = [r for r in mine if r["swap"] in keep]
            if not sub:
                out.append({"id": mid, "axis": axis, "swaps": "", "first_t_below": float("nan"), "auc": float("nan")})
                continue
            t, loc = common.curve(sub, axis)
            out.append({"id": mid, "axis": axis, "swaps": "+".join(keep), **common.scores(t, loc),
                        **{f"t{round(x * 10):02d}": round(float(v), 4) for x, v in zip(t, loc)}})
    return out, retained, se_rows


# ---------------------------------------------------------------------------------------------- raw outputs
def load_raw(raw: Path, model: str, set_name: str) -> dict | None:
    dirs = sorted(raw.glob(f"{model}__{set_name}__w*"), key=lambda p: p.name.rsplit("__", 1)[1])
    dirs = [d for d in dirs if (d / "walk_sums.npz").is_file()]
    if not dirs:
        return None
    d = dirs[-1]
    out = {"dir": d, "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
           "images": json.loads((d / "images.json").read_text(encoding="utf-8")), "sums": dict(np.load(d / "walk_sums.npz"))}
    if (d / "frames.npz").is_file():
        out["frames"] = dict(np.load(d / "frames.npz"))
    return out


def boot_weights(n_walks: int) -> np.ndarray:
    """(1 + N_BOOT, n_walks) walk multiplicities: row 0 the estimate (all ones), then the replicates."""
    rng = np.random.default_rng(np.random.SeedSequence(BOOT_SEED))
    w = np.ones((1 + N_BOOT, n_walks))
    for b in range(N_BOOT):
        w[1 + b] = np.bincount(rng.integers(0, n_walks, n_walks), minlength=n_walks)
    return w


class Sums:
    """The (1 + N_BOOT) resampled totals of every (image key, slot), field by field."""

    def __init__(self, raw: dict, weights: np.ndarray):
        self.fi = {k: (slice(*v) if isinstance(v, list) else v) for k, v in raw["meta"]["fields"].items()}
        self.tot = {key: np.einsum("bw,wpf->bpf", weights, arr) for key, arr in raw["sums"].items()}
        self.images = {e["key"]: e for e in raw["images"]}

    def ratio(self, key: str, slot: str, part: str, num: str, den: str = "n") -> np.ndarray:
        a = self.tot[f"{key}_{slot}"][:, PART[part]]
        with np.errstate(invalid="ignore", divide="ignore"):
            return a[:, self.fi[num]] / (a[:, self.fi[den]] if np.ndim(self.fi[den]) == 0 else a[:, self.fi[den]])

    def pooled(self, keys_slots: list[tuple[str, str]], part: str, num: str, den: str = "n") -> np.ndarray:
        """sum num / sum den over several cells (used for the per-variable TV means)."""
        num_t = sum(self.tot[f"{k}_{s}"][:, PART[part], self.fi[num]] for k, s in keys_slots)
        den_t = sum(self.tot[f"{k}_{s}"][:, PART[part], self.fi[den]] for k, s in keys_slots)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num_t / (den_t[:, None] if np.ndim(num_t) == 2 else den_t)


def point(values: list[np.ndarray]) -> np.ndarray:
    return np.nanmean(np.stack(values), axis=0)


def est(x: np.ndarray) -> tuple[float, float]:
    return float(x[0]), float(np.nanstd(x[1:]))


# ---------------------------------------------------------------------------------------------- step 2: test 0
def arc_points(S: Sums, swaps: list[str]) -> dict[float, dict[str, list[str]]]:
    """t -> swap -> image key (arc_ctrl entries of the retained swaps)."""
    out: dict[float, dict[str, str]] = {}
    for key, e in S.images.items():
        info = e["info"]
        if info.get("axis") == "arc_ctrl" and info["swap"] in swaps:
            out.setdefault(float(info["t"]), {})[info["swap"]] = key
    return dict(sorted(out.items()))


ZONES = ("nw", "ne", "se", "sw")


def where_slots(swap: str, where: str) -> list[str]:
    return list(ZONES) if where == "all" else [swap] if where == "moved" else [z for z in ZONES if z != swap]


def curve_of(S: Sums, swaps: list[str], num: str, den: str = "n", part: str = "step1", where: str = "all") -> dict[float, np.ndarray]:
    out = {}
    for t, by_swap in arc_points(S, swaps).items():
        out[t] = point([point([S.ratio(key, z, part, num, den) for z in where_slots(swap, where)]) for swap, key in by_swap.items()])
    return out


def test0_rows(mid: str, S: Sums, swaps: list[str]) -> list[dict]:
    rows = []
    for where in ("all", "moved", "other"):
        curves = {name: curve_of(S, swaps, num, den, "step1", where) for name, num, den in
                  (("a1", "a1", "n"), ("a2", "a2", "n"), ("o1", "o1", "n"), ("b_cond", "b_ok", "b_n"), ("b_all", "b_all_ok", "b_all_n"))}
        curves["b_cond_zone"] = curve_of(S, swaps, "b_ok", "b_n", "zone", where)
        for t in curves["a1"]:
            row = {"id": mid, "where": where, "t": t}
            for name, c in curves.items():
                v, se = est(c[t])
                row[name], row[f"{name}_se"] = round(v, 4), round(se, 4)
            row["a1_band"], row["a2_band"] = band(row["a1"]), band(row["a2"])
            rows.append(row)
    return rows


def se_canonical_rows(mid: str, S: Sums) -> list[dict]:
    keys = [k for k, e in S.images.items() if e["info"].get("axis") == "se_t0_canonical"]
    rows = []
    for key in keys:
        for z in ZONES:
            row = {"id": mid, "image": key, "zone": z}
            for name, num, den in (("a1", "a1", "n"), ("a2", "a2", "n"), ("o1", "o1", "n"), ("b_cond", "b_ok", "b_n"), ("b_all", "b_all_ok", "b_all_n")):
                v, se = est(S.ratio(key, z, "step1", num, den))
                row[name], row[f"{name}_se"] = round(v, 4), round(se, 4)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------- step 3: latent
def weighted_mi(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Plug-in MI (bits) of two label arrays with frame weights; w = 1 gives latent_ceiling.mutual_information."""
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    nx, ny = xi.max() + 1, yi.max() + 1
    joint = np.bincount(xi * ny + yi, weights=w, minlength=nx * ny).reshape(nx, ny)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def latent_stats(train: dict, weights: np.ndarray) -> dict:
    """I_pos, I_asg per variable (1 + N_BOOT, 8), and the sharpness / margin per variable (1 + N_BOOT, 8)."""
    fr, images = train["frames"], train["images"]
    cls, state, asg, walk = [], [], [], []
    for e in images:
        for z in ZONES:
            k = f"{e['key']}_{z}"
            cls.append(fr[f"{k}_cls"]); state.append(fr[f"{k}_state"]); walk.append(fr[f"{k}_walk"])
            asg.append(np.full(len(fr[f"{k}_state"]), e["info"]["assignment"]))
    cls, state, asg, walk = np.concatenate(cls).astype(np.int64), np.concatenate(state), np.concatenate(asg), np.concatenate(walk)
    V = cls.shape[1]
    i_pos, i_asg = np.zeros((weights.shape[0], V)), np.zeros((weights.shape[0], V))
    masks = [asg == a for a in np.unique(asg)]
    for b in range(weights.shape[0]):
        w = weights[b][walk]
        for v in range(V):
            i_asg[b, v] = weighted_mi(cls[:, v], asg, w)
            i_pos[b, v] = np.mean([weighted_mi(cls[m, v], state[m], w[m]) for m in masks])
    S = Sums(train, weights)
    keys = [(e["key"], z) for e in images for z in ZONES]
    sharp = sum(S.tot[f"{k}_{z}"][:, PART[p], S.fi["sharp"]] for k, z in keys for p in ("corridor", "zone"))
    margin = sum(S.tot[f"{k}_{z}"][:, PART[p], S.fi["margin"]] for k, z in keys for p in ("corridor", "zone"))
    n = sum(S.tot[f"{k}_{z}"][:, PART[p], S.fi["n"]] for k, z in keys for p in ("corridor", "zone"))
    return {"i_pos": i_pos, "i_asg": i_asg, "sharp": sharp / n[:, None], "margin": margin / n[:, None],
            "check_unweighted": [latent_ceiling.mutual_information(cls[:, v], asg) for v in range(V)]}


def classify(i_pos: np.ndarray, i_asg: np.ndarray) -> list[str]:
    out = []
    for p, a in zip(i_pos, i_asg):
        if max(p, a) < ACTIVE_BITS:
            out.append("inactive")
        elif a <= RATIO * p:
            out.append("position")
        elif p <= RATIO * a:
            out.append("background")
        else:
            out.append("mixed")
    return out


def position_set(classes: list[str], i_pos: np.ndarray) -> tuple[list[int], str]:
    pos = [v for v, c in enumerate(classes) if c == "position"]
    if pos:
        return pos, "position"
    pos = [v for v, c in enumerate(classes) if c in ("position", "mixed")]
    if pos:
        return pos, "position+mixed"
    return [int(v) for v in np.argsort(-i_pos)[:2]], "top-2 I_pos"


def separation(i_pos: np.ndarray, i_asg: np.ndarray) -> np.ndarray:
    num = (i_pos * i_asg).sum(-1)
    den = np.linalg.norm(i_pos, axis=-1) * np.linalg.norm(i_asg, axis=-1)
    return 1.0 - num / den


def sensitivity(S: Sums, swaps: list[str], tv: str) -> dict[str, dict[float, np.ndarray]]:
    """where -> t -> (1 + N_BOOT, 8): per variable, the mean TV over the frames of that place, mean over the swaps."""
    out = {"moved": {}, "other": {}, "corridor": {}}
    for t, by_swap in arc_points(S, swaps).items():
        for where in out:
            per_swap = []
            for swap, key in by_swap.items():
                if where == "corridor":
                    per_swap.append(S.pooled([(key, z) for z in ZONES], "corridor", tv))
                else:
                    per_swap.append(S.pooled([(key, z) for z in where_slots(swap, where)], "zone", tv))
            out[where][t] = point(per_swap)
    return out


def locality(sens: dict, pos: list[int]) -> dict[str, np.ndarray]:
    ts = [t for t in sens["moved"] if t > 0]
    mean = {w: np.mean([sens[w][t][:, pos].sum(-1) for t in ts], axis=0) for w in sens}
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.where(mean["moved"] >= L_MIN_DENOMINATOR, mean["other"] / mean["moved"], np.nan)
        Lc = np.where(mean["moved"] >= L_MIN_DENOMINATOR, mean["corridor"] / mean["moved"], np.nan)
    return {"L": L, "L_corridor": Lc, "S_moved": mean["moved"], "S_other": mean["other"], "S_corridor": mean["corridor"]}


# ---------------------------------------------------------------------------------------------- step 4
def step4_rows(mid: str, S: Sums) -> tuple[list[dict], dict]:
    fam: dict[str, list[str]] = {}
    for key, e in S.images.items():
        fam.setdefault(e["info"]["family"], []).append(key)
    fam["4ab"] = fam["4a"] + fam["4b"]
    rows, vals = [], {}
    for name, keys in fam.items():
        row = {"id": mid, "family": name, "n_images": len(keys)}
        for m in ("a1", "a2", "o1") + (("a1x", "a2x") if name == "4c" else ()):
            x = point([point([S.ratio(k, z, "step1", m) for z in ZONES]) for k in keys])
            vals[(name, m)] = x
            row[m], row[f"{m}_se"] = round(est(x)[0], 4), round(est(x)[1], 4)
        row["a2_band"] = band(row["a2"])
        rows.append(row)
        if name in ("4a", "4b", "4c"):
            for k in keys:
                x1 = point([S.ratio(k, z, "step1", "a1") for z in ZONES])
                x2 = point([S.ratio(k, z, "step1", "a2") for z in ZONES])
                rows.append({"id": mid, "family": f"{name}:{S.images[k]['info']['name']}", "n_images": 1, "a1": round(est(x1)[0], 4),
                             "a1_se": round(est(x1)[1], 4), "a2": round(est(x2)[0], 4), "a2_se": round(est(x2)[1], 4), "a2_band": band(est(x2)[0])})
    vals["drop"] = vals[("canonical", "a2")] - vals[("4c", "a2")]
    return rows, vals


# ---------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=HERE / "tables")
    parser.add_argument("--out", type=Path, default=HERE / "tables")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    score_rows, retained, se_rows = restricted_scores(args.tables / "localisation.csv")
    write_csv(args.out / "scores_restricted.csv", score_rows)
    write_csv(args.out / "se_swap_t0_cluster.csv", se_rows)
    area = {r["id"]: r["auc"] for r in score_rows if r["axis"] == "arc_ctrl"}
    history = {r["id"]: r for r in csv.DictReader(open(args.tables / "history_summary.csv", encoding="utf-8"))}
    test0, se_canon, latent_rows, latent_sum, sens_rows, step4, summary = [], [], [], [], [], [], {}
    step4_vals, t0_curves, loc_res = {}, {}, {}
    for mid in MECHANISTIC:
        arc, train, s4 = (load_raw(args.raw, mid, s) for s in ("arc_ctrl", "training", "step4"))
        if arc is None or train is None or s4 is None:
            print(f"{mid}: missing raw outputs ({[x is None for x in (arc, train, s4)]})")
            continue
        weights = boot_weights(arc["meta"]["n_walks"])
        S = Sums(arc, weights)
        rows = test0_rows(mid, S, retained[mid])
        test0 += rows
        se_canon += se_canonical_rows(mid, S)
        t0_curves[mid] = {(r["where"], r["t"]): r for r in rows}
        lat = latent_stats(train, weights)
        classes = classify(lat["i_pos"][0], lat["i_asg"][0])
        pos, pos_rule = position_set(classes, lat["i_pos"][0])
        for v in range(8):
            latent_rows.append({"id": mid, "var": v, "I_pos": round(lat["i_pos"][0, v], 4), "I_pos_se": round(float(np.std(lat["i_pos"][1:, v])), 4),
                                "I_asg": round(lat["i_asg"][0, v], 4), "I_asg_se": round(float(np.std(lat["i_asg"][1:, v])), 4),
                                "I_asg_unweighted_check": round(lat["check_unweighted"][v], 4), "class": classes[v], "in_pos": v in pos,
                                "sharpness": round(lat["sharp"][0, v], 4), "margin": round(lat["margin"][0, v], 4)})
        sep = separation(lat["i_pos"], lat["i_asg"])
        res = {"pos": pos, "pos_rule": pos_rule, "classes": classes}
        for tv in ("tv_a2", "tv_a1"):
            sens = sensitivity(S, retained[mid], tv)
            loc = locality(sens, pos)
            res[tv] = loc
            for where, by_t in sens.items():
                for t, x in by_t.items():
                    sens_rows.append({"id": mid, "tv": tv, "where": where, "t": t,
                                      **{f"v{v}": round(float(x[0, v]), 4) for v in range(8)},
                                      "pos_mean": round(float(x[0, pos].mean()), 4), "pos_mean_se": round(float(np.std(x[1:, pos].mean(-1))), 4)})
        loc_res[mid] = res
        s4rows, s4vals = step4_rows(mid, Sums(s4, weights))
        step4 += s4rows
        step4_vals[mid] = s4vals
        latent_sum.append({"id": mid, "pos_vars": "+".join(map(str, pos)), "pos_rule": pos_rule,
                           "n_position": classes.count("position"), "n_background": classes.count("background"),
                           "n_mixed": classes.count("mixed"), "n_inactive": classes.count("inactive"),
                           "separation": round(est(sep)[0], 4), "separation_se": round(est(sep)[1], 4),
                           "sharpness_all": round(float(lat["sharp"][0].mean()), 4), "sharpness_all_se": round(float(np.std(lat["sharp"][1:].mean(-1))), 4),
                           "margin_all": round(float(lat["margin"][0].mean()), 4), "margin_all_se": round(float(np.std(lat["margin"][1:].mean(-1))), 4),
                           "sharpness_pos": round(float(lat["sharp"][0, pos].mean()), 4), "sharpness_pos_se": round(float(np.std(lat["sharp"][1:, pos].mean(-1))), 4),
                           "margin_pos": round(float(lat["margin"][0, pos].mean()), 4), "margin_pos_se": round(float(np.std(lat["margin"][1:, pos].mean(-1))), 4),
                           **{f"{k}_{tv}": round(est(res[tv][k])[0], 4) for tv in ("tv_a2", "tv_a1") for k in ("L", "L_corridor", "S_moved", "S_other", "S_corridor")},
                           **{f"{k}_{tv}_se": round(est(res[tv][k])[1], 4) for tv in ("tv_a2", "tv_a1") for k in ("L", "L_corridor")}})
        summary[mid] = {"sep": sep, "margin_pos": lat["margin"][:, pos].mean(-1), "margin_all": lat["margin"].mean(-1), "L": res["tv_a2"]["L"]}
        print(f"{mid}: done", flush=True)
    write_csv(args.out / "test0.csv", test0)
    if se_canon:
        write_csv(args.out / "se_swap_t0_test0.csv", se_canon)
    write_csv(args.out / "latent.csv", latent_rows)
    write_csv(args.out / "latent_summary.csv", latent_sum)
    write_csv(args.out / "sensitivity.csv", sens_rows)
    write_csv(args.out / "step4.csv", step4)
    reading = verdicts(t0_curves, summary, step4_vals, area, history, latent_sum)
    rows = []
    for ls in latent_sum:
        mid = ls["id"]
        rows.append({"id": mid, "area_arc_ctrl": round(area[mid], 4), "swaps": "+".join(retained[mid]),
                     "mass_min_quadrant": history.get(mid, {}).get("mass_min_quadrant_at_checkpoint"),
                     "L": ls["L_tv_a2"], "L_se": ls["L_tv_a2_se"], "separation": ls["separation"], "margin_pos": ls["margin_pos"],
                     **{f"{f}_{m}": next((r[m] for r in step4 if r["id"] == mid and r["family"] == f), None)
                        for f in ("canonical", "4a", "4b", "4c") for m in ("a1", "a2")}})
    write_csv(args.out / "models_summary.csv", rows)
    (args.out / "reading.json").write_text(json.dumps(reading, indent=1, default=float), encoding="utf-8")
    print(json.dumps(reading, indent=1, default=float))
    return 0


def diff_se(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    d = x - y
    return float(d[0]), float(np.nanstd(d[1:]))


def verdicts(t0: dict, summary: dict, s4: dict, area: dict, history: dict, latent_sum: list[dict]) -> dict:
    out: dict = {"rules": "PREDICTIONS.md sections 4-5", "band": [LOST, LOCALISED]}

    def lost_points(mid: str) -> list[float]:
        return [t for (w, t), r in t0[mid].items() if w == "all" and r["a1"] < LOST]

    def a2_at(mid: str, ts: list[float]) -> dict:
        return {t: t0[mid][("all", t)]["a2"] for t in ts}

    # P1 / H6
    for mid in ("E", "A"):
        ts = lost_points(mid)
        a2 = a2_at(mid, ts)
        out[f"lost_points_{mid}"] = {"t": ts, "a1": {t: t0[mid][("all", t)]["a1"] for t in ts}, "a2": a2,
                                     "b_cond": {t: t0[mid][("all", t)]["b_cond"] for t in ts}}
    ts_e = lost_points("E")
    a2_e = a2_at("E", ts_e)
    n_lost2 = sum(v < LOST for v in a2_e.values())
    n_loc2 = sum(v >= LOCALISED for v in a2_e.values())
    out["P1"] = {"verdict": "holds" if ts_e and n_lost2 > len(ts_e) / 2 else "fails" if ts_e and n_loc2 > len(ts_e) / 2 else "indeterminate",
                 "E_lost_points": len(ts_e), "a2_lost": n_lost2, "a2_localised": n_loc2}
    out["H6"] = {"verdict": "supported" if ts_e and n_loc2 > len(ts_e) / 2 else "not supported", "a2_localised_at_E_lost_points": n_loc2}
    # test 0 stop rule
    stop = [t for (w, t), r in t0["E"].items() if w == "all" and t >= 0.5 and r["a1"] >= LOCALISED and r["b_cond"] < LOST]
    out["test0_stop"] = {"triggered": bool(stop), "points": stop}
    # P2
    LE, LF = summary["E"]["L"], summary["F"]["L"]
    d, se = diff_se(LE, LF)
    p2_ok = LF[0] < 0.5 and LE[0] >= 0.75 and d > 2 * se
    p2_fail = (d < -2 * se) or LF[0] >= 0.75
    out["P2"] = {"verdict": "holds" if p2_ok else "fails" if p2_fail else "indeterminate", "L_E": float(LE[0]), "L_F": float(LF[0]),
                 "L_E_minus_L_F": d, "se": se, "L_A": float(summary["A"]["L"][0])}
    # P3
    p3 = {}
    for fam in ("4a", "4b"):
        p3[fam] = {"F": float(s4["F"][(fam, "a2")][0]), "E": float(s4["E"][(fam, "a2")][0]),
                   "F_a1": float(s4["F"][(fam, "a1")][0]), "E_a1": float(s4["E"][(fam, "a1")][0])}
    d, se = diff_se(s4["F"][("4ab", "a2")], s4["E"][("4ab", "a2")])
    holds = all(p3[f]["F"] >= LOCALISED and p3[f]["E"] < LOST for f in ("4a", "4b"))
    out["P3"] = {"verdict": "holds" if holds else "fails (H1 refuted)" if d <= 2 * se else "partial", "by_family": p3,
                 "F_minus_E_4ab": d, "se": se}
    d1, se1 = diff_se(s4["F"][("4ab", "a1")], s4["E"][("4ab", "a1")])
    out["P3_a1"] = {"F_minus_E_4ab": d1, "se": se1}
    # P4
    dd, sed = diff_se(s4["F"]["drop"], s4["E"]["drop"])
    gap4c = s4["F"][("4c", "a2")] - s4["E"][("4c", "a2")]
    gap4ab = s4["F"][("4ab", "a2")] - s4["E"][("4ab", "a2")]
    dm, sem = diff_se(gap4ab, gap4c)
    out["P4"] = {"verdict": "holds" if dd > 2 * sed else "holds (minimal form)" if dm > 2 * sem else "fails",
                 "drop_F": float(s4["F"]["drop"][0]), "drop_E": float(s4["E"]["drop"][0]), "drop_F_minus_drop_E": dd, "se": sed,
                 "gap_F_minus_E_4c": float(gap4c[0]), "gap_F_minus_E_4ab": float(gap4ab[0]), "gap_shrink": dm, "gap_shrink_se": sem}
    # A's direction
    ts_a = lost_points("A")
    a2_a = a2_at("A", ts_a)
    dla, sela = diff_se(summary["A"]["L"], summary["F"]["L"])
    d4a, se4a = diff_se(s4["F"][("4ab", "a2")], s4["A"][("4ab", "a2")])
    out["A_direction"] = {"a2_lost_at_A_lost_points": sum(v < LOST for v in a2_a.values()), "A_lost_points": len(ts_a),
                          "L_A_minus_L_F": dla, "se_L": sela, "F_minus_A_4ab_a2": d4a, "se_4ab": se4a}
    # P5, H2, H3 (descriptive)
    ids = [r["id"] for r in latent_sum]
    ar = np.array([area[i] for i in ids])
    L = np.array([summary[i]["L"][0] for i in ids])
    mass = np.array([float(history[i]["mass_min_quadrant_at_checkpoint"]) for i in ids])
    sep = np.array([summary[i]["sep"][0] for i in ids])
    mar = np.array([summary[i]["margin_pos"][0] for i in ids])
    ok = np.isfinite(L)
    out["P5"] = {"n": int(ok.sum()), "rho_area_minus_L": float(spearmanr(ar[ok], -L[ok])[0]), "rho_area_mass": float(spearmanr(ar, mass)[0]),
                 "ties_area_ge_0.97": int((ar >= 0.97).sum()), "descriptive": True}
    out["P5"]["reading"] = "consistent" if out["P5"]["rho_area_minus_L"] >= 0.6 and abs(out["P5"]["rho_area_mass"]) < 0.3 else "not consistent"
    ds, ses = diff_se(summary["F"]["sep"], summary["E"]["sep"])
    out["H2"] = {"Sep_F_minus_Sep_E": ds, "se": ses, "rho_area_sep": float(spearmanr(ar, sep)[0]),
                 "verdict": "consistent" if ds > 2 * ses and spearmanr(ar, sep)[0] >= 0.6 else "not consistent"}
    dmg, semg = diff_se(summary["F"]["margin_pos"], summary["E"]["margin_pos"])
    dma, sema = diff_se(summary["F"]["margin_all"], summary["E"]["margin_all"])
    out["H3"] = {"margin_pos_F_minus_E": dmg, "se": semg, "margin_all_F_minus_E": dma, "se_all": sema, "rho_area_margin_pos": float(spearmanr(ar, mar)[0]),
                 "verdict": "consistent" if dmg > 2 * semg and dma > 2 * sema and spearmanr(ar, mar)[0] >= 0.6 else "not consistent"}
    out["H5"] = {f"{e}_vs_{r}": {"area_exposed": area[e], "area_reference": area[r]} for e, r in PAIRS}
    # decision
    h1 = (out["P1"]["verdict"] == "holds" and out["P2"]["verdict"] == "holds" and out["P3"]["verdict"] == "holds"
          and out["A_direction"]["a2_lost_at_A_lost_points"] > len(ts_a) / 2 and dla > 0 and d4a > 2 * se4a)
    out["decision"] = ("H1 supported" if h1 else "H1 refuted" if out["P3"]["verdict"].startswith("fails")
                       else "H6 supported" if out["H6"]["verdict"] == "supported" else "not decided")
    return out


if __name__ == "__main__":
    sys.exit(main())
