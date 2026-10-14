"""Where the E1 report's figures and numbers read their data, and how. Data access only: no new analysis.

Every E1 source is under experiments/E1_offline_4zones/artifacts/ (read-only):
  - measurements.csv, permutation_null.csv (Step B), read with E1's own artifact.py loaders;
  - calibration_report.csv, gradient_curves.csv (Step A).

The quantities report.py only prints (zone means over the 20 seeds, mean p95 over seeds, bias of I_hat) are
recomputed here with EXACTLY report.py's filters (environment, region, zone, budget -- each function below names the
report.py function whose filter it copies); everything report.py returns as a number is called from report.py itself
(predicted_b, across_seed_sd, fit_inverse_sqrt_n, fit_power_law_slope_ci, out_of_tolerance_rate). `step_b()` then
checks every recomputed value against artifacts/etape_b_report.txt, formatted as that report prints it, and stops on
any difference: the report's figures and numbers can never drift from E1's own report.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

import common

from helplessness.config import load_yaml_block
from helplessness.envs.zones import interior_mask
from helplessness.experiment_loader import load_experiment_module

E1_DIR = common.REPO_ROOT / "experiments" / "E1_offline_4zones"
ARTIFACTS = E1_DIR / "artifacts"
ZONES = ("uncontrollable_deterministic", "controllable_noisy", "uncontrollable_noisy", "controllable_deterministic")
ZONE_ABBR = {"uncontrollable_deterministic": "UD", "controllable_noisy": "CN",
             "uncontrollable_noisy": "UN", "controllable_deterministic": "CD"}


def e1_module(filename: str):
    """An E1 module, imported as the experiment imports it (its bare sibling imports resolved in its own directory)."""
    return load_experiment_module("E1_offline_4zones", filename, f"paper_e1_{Path(filename).stem}")


def read_csv(name: str) -> list[dict]:
    path = common.require_file(ARTIFACTS / name)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def measurements() -> list[dict]:
    return e1_module("artifact.py").load_measurement_artifact(common.require_file(ARTIFACTS / "measurements.csv"))


@lru_cache(maxsize=None)
def permutation_rows() -> list[dict]:
    return e1_module("artifact.py").load_permutation_artifact(common.require_file(ARTIFACTS / "permutation_null.csv"))


@lru_cache(maxsize=None)
def report():
    return e1_module("report.py")


# --- layout ----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def e1_config() -> dict:
    import yaml
    return yaml.safe_load(common.require_file(E1_DIR / "config.yaml").read_text(encoding="utf-8"))


def zones_config() -> dict:
    return load_yaml_block(e1_config()["zones_block"])


def layout(direction: str | None = None, control: bool = False):
    env = e1_module("env.py")
    direction = direction or e1_config()["direction"]
    return (env.build_control_layout if control else env.build_layout)(zones_config(), direction)


def region_sizes() -> dict:
    """{"corridor": cells, "zone": cells per block, "interior": interior cells per block} (envs.zones.interior_mask),
    checked to be the same for every block."""
    lay = layout()
    grid = tuple(zones_config()["grid_shape"])
    sizes = {}
    for zone in lay.zones:
        category = zone.appearance.category
        if category == "corridor":
            sizes["corridor"] = int(zone.mask.sum())
        else:
            sizes.setdefault("blocks", set()).add((int(zone.mask.sum()), int(interior_mask(zone, grid).sum())))
    if len(sizes["blocks"]) != 1:
        raise SystemExit(f"E1 blocks differ in size or interior: {sizes['blocks']}")
    zone_cells, interior_cells = sizes.pop("blocks").pop()
    return {**sizes, "zone": zone_cells, "interior": interior_cells}


# --- Step B, with report.py's filters ----------------------------------------------------------------------------------

def _rows(zone: str, n: int, env: str, region: str) -> list[dict]:
    return [r for r in measurements() if r["zone"] == zone and r["n_budget"] == n and r["env"] == env and r["region"] == region]


def zone_mean(zone: str, n: int, column: str, env: str = "exp", region: str = "all") -> float:
    """Mean over the seeds of a zone-level column (report.py: observed_b_in_true_h0_zones / bias_cancellation_factor
    use region "all"; delta_l1_reading uses "interior" for delta and "all" for l1; delta_at_n10_table "interior")."""
    vals = [r[column] for r in _rows(zone, n, env, region)]
    if not vals:
        raise SystemExit(f"measurements.csv: no row for zone={zone}, n={n}, env={env}, region={region}")
    return float(np.mean(vals))


def zone_values(zone: str, n: int, column: str, env: str = "exp", region: str = "all") -> list[float]:
    return [r[column] for r in _rows(zone, n, env, region)]


def mean_p95(zone: str, n: int, method: str = "bootstrap", quantity: str = "b") -> float:
    """report.py headline_detectable_b_table / detectable_b_table_inflated: mean p95 over seeds."""
    vals = [r["p95"] for r in permutation_rows()
            if r["zone"] == zone and r["quantity"] == quantity and r["method"] == method and r["n_budget"] == n]
    if not vals:
        raise SystemExit(f"permutation_null.csv: no row for zone={zone}, n={n}, method={method}, quantity={quantity}")
    return float(np.mean(vals))


def bias(zone: str, n: int) -> dict:
    """report.py bias_cancellation_factor: biases of I_hat (env and its control), residual |b - predicted|, factor."""
    rep = report()
    bias_exp = zone_mean(zone, n, "I_hat", "exp") - zone_mean(zone, n, "I_star", "exp")
    bias_ctrl = zone_mean(zone, n, "I_hat", "ctrl") - zone_mean(zone, n, "I_star", "ctrl")
    residual = abs(zone_mean(zone, n, "b") - rep.predicted_b(zone))
    factor = ((abs(bias_exp) + abs(bias_ctrl)) / 2) / residual if residual > 0 else float("inf")
    return {"exp": bias_exp, "ctrl": bias_ctrl, "residual": residual, "factor": factor}


@lru_cache(maxsize=None)
def step_b() -> dict:
    """Every Step B quantity the report quotes or plots, recomputed, then checked against etape_b_report.txt."""
    rep = report()
    budgets = list(rep.BUDGETS)
    rows = measurements()
    q = {"budgets": budgets, "seeds": sorted({r["seed"] for r in rows}), "n_rows": len(rows),
         "delta_violations": sum(1 for r in rows if r["delta"] < 0)}
    q["b"] = {z: {n: {"mean": zone_mean(z, n, "b"), "sd": rep.across_seed_sd(rows, z, n)} for n in budgets} for z in ZONES}
    q["predicted_b"] = {z: rep.predicted_b(z) for z in ZONES}
    q["I_hat_exp"] = {z: {n: zone_mean(z, n, "I_hat") for n in budgets} for z in ZONES}
    q["I_star_exp"] = {z: zone_mean(z, budgets[0], "I_star") for z in ZONES}
    q["bias"] = {n: bias("uncontrollable_noisy", n) for n in budgets}
    q["delta"] = {z: {n: zone_mean(z, n, "delta", region="interior") for n in budgets} for z in ZONES}
    q["delta_boundary"] = {z: {n: zone_mean(z, n, "delta", region="boundary") for n in budgets} for z in ZONES}
    q["l1"] = {z: {n: zone_mean(z, n, "l1") for n in budgets} for z in ZONES}
    q["p95"] = {z: {n: mean_p95(z, n) for n in budgets} for z in ZONES}
    cn = [q["p95"]["controllable_noisy"][n] for n in budgets]
    q["slope"] = rep.fit_power_law_slope_ci(budgets, cn)
    q["slope_no10"] = rep.fit_power_law_slope_ci(budgets[1:], cn[1:])
    q["precision"] = {}
    for z in rep.NULL_ZONES:
        sds = [rep.across_seed_sd(rows, z, n) for n in budgets]
        a = rep.fit_inverse_sqrt_n(budgets, sds)
        q["precision"][z] = {"sd": dict(zip(budgets, sds)), "A": a,
                             "n_needed": {k: (k * a / rep.PRECISION_TOL) ** 2 for k in rep.SAFETY_MARGIN_SIGMAS}}
    q["n_needed"] = {k: max(q["precision"][z]["n_needed"][k] for z in rep.NULL_ZONES) for k in rep.SAFETY_MARGIN_SIGMAS}
    q["n_tested"] = {k: next((n for n in budgets if n >= q["n_needed"][k]), budgets[-1]) for k in rep.SAFETY_MARGIN_SIGMAS}
    q["out_of_tol"] = {n: rep.out_of_tolerance_rate(rows, "uncontrollable_noisy", n) for n in budgets}
    q["tolerance"] = rep.PRECISION_TOL
    q["sigmas"] = rep.SAFETY_MARGIN_SIGMAS
    _check_against_report(q)
    return q


def _check_against_report(q: dict) -> None:
    """Each recomputed value, formatted as etape_b_report.txt prints it, must appear there."""
    text = common.require_file(ARTIFACTS / "etape_b_report.txt").read_text(encoding="utf-8")
    b10 = q["bias"][10]
    expected = [
        f"Ihat bias = {b10['exp']:+.4f} bits", f"bias = {b10['ctrl']:+.4f} bits", f"= {b10['residual']:.4f} bits",
        f"factor of ~{b10['factor']:.1f}x",
        "controllable_noisy".ljust(30) + "".join(f"{q['p95']['controllable_noisy'][n]:<12.5f}" for n in q["budgets"]),
        f"log-log slope = {q['slope'][0]:.3f}, 95% CI [{q['slope'][1]:.3f}, {q['slope'][2]:.3f}]",
        f"slope = {q['slope_no10'][0]:.3f}, 95% CI [{q['slope_no10'][1]:.3f}, {q['slope_no10'][2]:.3f}]",
        "controllable_noisy: " + ", ".join(f"n={n}: {q['b']['controllable_noisy'][n]['mean']:+.5f} "
                                          f"(sd={q['b']['controllable_noisy'][n]['sd']:.5f})" for n in q["budgets"]),
        f"fitted A (sd = A/sqrt(n)): {q['precision']['uncontrollable_noisy']['A']:.4f}",
    ]
    for zone in ("controllable_noisy", "uncontrollable_noisy"):
        for n in q["budgets"]:
            expected.append(f"n={n:5d}  Delta={q['delta'][zone][n]:.4f}  L1={q['l1'][zone][n]:.4f}")
    for k in q["sigmas"]:
        rate, over, total = q["out_of_tol"][q["n_tested"][k]]
        expected.append(f"{k}-sigma margin: fitted crossing ~{q['n_needed'][k]:.0f} -> smallest tested budget "
                        f"n={q['n_tested'][k]} (actual out-of-tolerance rate there: {over}/{total} = {rate:.0%})")
    rate, over, total = q["out_of_tol"][30]
    expected.append(f"{over}/{total} seeds ({rate:.0%})")
    missing = [e for e in expected if e not in text]
    if missing:
        raise SystemExit("Recomputed E1 values differ from artifacts/etape_b_report.txt (not found there):\n  "
                         + "\n  ".join(missing))


# --- Step A -----------------------------------------------------------------------------------------------------------

def calibration_rows() -> list[dict]:
    rows = read_csv("calibration_report.csv")
    for r in rows:
        for k, v in r.items():
            if k != "direction" and not k.startswith("well_shape") and v not in ("True", "False"):
                r[k] = float(v)
    return rows


@lru_cache(maxsize=None)
def step_a() -> dict:
    """Step A quantities from calibration_report.csv, checked against artifacts/recommendation.txt."""
    rows = calibration_rows()
    q = {
        "n_combinations": len(rows),
        "gammas": sorted({r["gamma"] for r in rows}), "rewards": sorted({r["R"] for r in rows}),
        "directions": sorted({r["direction"] for r in rows}),
        "exterior_vs_ctrl": max(r["exterior_max_abs_diff_exp_vs_ctrl"] for r in rows),
        "exterior_vs_direction": max(r["exterior_max_abs_diff_away_vs_toward"] for r in rows),
        "interior_gap_max": max(max(abs(r["interior_gap_nw"]), abs(r["interior_gap_se"])) for r in rows),
        "away_wells_both": sum(1 for r in rows if r["direction"] == "away_from_goal"
                               and r["well_shape_nw"] == "well" and r["well_shape_se"] == "well"),
        "toward_nw_negligible": sum(1 for r in rows if r["direction"] == "toward_goal" and r["well_shape_nw"] == "negligible"),
        "per_direction": {d: sum(1 for r in rows if r["direction"] == d) for d in ("away_from_goal", "toward_goal")},
    }
    text = common.require_file(ARTIFACTS / "recommendation.txt").read_text(encoding="utf-8")
    expected = [f"by at most {q['exterior_vs_ctrl']:.2e}", f"by at most {q['exterior_vs_direction']:.2e}",
                f"reaches {q['interior_gap_max']:.2f} in absolute value",
                f"{q['away_wells_both']}/{q['per_direction']['away_from_goal']} away_from_goal combinations produce a real interior well",
                f"{q['toward_nw_negligible']}/{q['per_direction']['toward_goal']} toward_goal combinations are negligible"]
    missing = [e for e in expected if e not in text]
    if missing:
        raise SystemExit("Recomputed E1 Step A values differ from artifacts/recommendation.txt:\n  " + "\n  ".join(missing))
    return q


def gradient_curves() -> list[dict]:
    rows = read_csv("gradient_curves.csv")
    for r in rows:
        for k in ("gamma", "R", "V_mean"):
            r[k] = float(r[k])
        for k in ("distance", "n"):
            r[k] = int(r[k])
    return rows


def exterior_profile_differences() -> dict:
    """Largest |NW - SE| (same env) and |exp - ctrl| (same zone) of the exterior profiles, over the whole sweep."""
    rows = gradient_curves()
    by = {}
    for r in rows:
        by[(r["gamma"], r["R"], r["direction"], r["env"], r["zone"], r["distance"])] = r["V_mean"]
    zones = sorted({r["zone"] for r in rows})
    envs = sorted({r["env"] for r in rows})
    if len(zones) != 2 or envs != ["ctrl", "exp"]:
        raise SystemExit(f"gradient_curves.csv: unexpected zones {zones} or envs {envs}")
    nw_se = max(abs(v - by[k[:4] + (zones[1],) + k[5:]]) for k, v in by.items() if k[4] == zones[0])
    exp_ctrl = max(abs(v - by[k[:3] + ("ctrl",) + k[4:]]) for k, v in by.items() if k[3] == "exp")
    return {"nw_vs_se": nw_se, "exp_vs_ctrl": exp_ctrl}

