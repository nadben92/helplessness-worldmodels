"""The E1 report's generated numbers (paper/reports/e1_report.tex). Each entry reads one source:
  - a config key (configs/zones/e1_4zones.yaml, experiments/E1_offline_4zones/config.yaml), the layout E1's env.py
    builds from them, or a plain arithmetic combination of config keys, stated in its src;
  - E1's artifacts (experiments/E1_offline_4zones/artifacts/), through paper/figures/e1_sources.py, which recomputes
    with report.py's filters and functions and checks every value against etape_b_report.txt / recommendation.txt.
Protocol parameters without a config key (budgets, tolerance, permutation count, sigma margins: code constants) and
numbers stated only in E1's README / docs or only in the notes are written in the report as they are, with their
% src: and % num-ok.
"""

from __future__ import annotations

import sys

from registry import REPO_ROOT, Number, SourceError

sys.path.insert(0, str(REPO_ROOT / "paper" / "figures"))
import e1_sources as src  # noqa: E402

ZONES_CFG = "configs/zones/e1_4zones.yaml"
E1_CFG = "experiments/E1_offline_4zones/config.yaml"
ART = "experiments/E1_offline_4zones/artifacts"
MEAS = f"{ART}/measurements.csv"
PERM = f"{ART}/permutation_null.csv"
CALIB = f"{ART}/calibration_report.csv"
CURVES = f"{ART}/gradient_curves.csv"
CHECKED = "(e1_sources.step_b: report.py's filters and functions; checked against etape_b_report.txt)"
BUDGET_WORDS = {10: "Ten", 30: "Thirty", 100: "Hundred", 300: "ThreeHundred", 1000: "Thousand"}


def cfg(key: str):
    c = src.e1_config()
    if key not in c:
        raise SourceError(f"{E1_CFG}: no key {key!r}")
    return c[key]


def zcfg(key: str):
    c = src.zones_config()
    if key not in c:
        raise SourceError(f"{ZONES_CFG}: no key {key!r}")
    return c[key]


def square(values, what: str):
    """The common value of `values`, required to be all equal (a square grid, equal counts, ...)."""
    if len(set(values)) != 1:
        raise SourceError(f"{what}: values differ {values}")
    return values[0]


def block_side() -> int:
    sides = {(b["rows"][1] - b["rows"][0] + 1, b["cols"][1] - b["cols"][0] + 1) for b in zcfg("blocks").values()}
    if len(sides) != 1:
        raise SourceError(f"{ZONES_CFG}: blocks of different shapes {sides}")
    return square(list(sides.pop()), "a block")


def noise_p() -> float:
    ps = {b["p"] for b in zcfg("blocks").values() if "p" in b}
    if len(ps) != 1:
        raise SourceError(f"{ZONES_CFG}: noisy blocks with different p {ps}")
    return ps.pop()


def n_actions() -> int:
    from helplessness.envs.dynamics import N_ACTIONS
    return N_ACTIONS


def b_at(zone: str, n: int, key: str) -> float:
    return src.step_b()["b"][zone][n][key]


def delta_zero_budget(zone: str) -> int:
    """report.py delta_l1_reading: the first budget whose interior Delta is exactly 0."""
    q = src.step_b()
    hit = next((n for n in q["budgets"] if q["delta"][zone][n] == 0.0), None)
    if hit is None:
        raise SourceError(f"Delta never reaches 0 in {zone}")
    return hit


def step_cost_horizon() -> float:
    return 1 / (1 - cfg("gamma"))


NUMBERS = [
    # environment
    Number("eOneGridSize", f"{ZONES_CFG} :: grid_shape (square)", lambda: square(list(zcfg("grid_shape")), "grid_shape"), 0),
    Number("eOneZoneSize", f"{ZONES_CFG} :: blocks.*.rows / cols (side of every block)", block_side, 0),
    Number("eOneNZones", f"{ZONES_CFG} :: number of blocks", lambda: len(zcfg("blocks")), 0),
    Number("eOneCorridorCells", f"{ZONES_CFG} :: corridor cells of experiments/E1_offline_4zones/env.py::build_layout",
           lambda: src.region_sizes()["corridor"], 0),
    Number("eOneZoneCells", f"{ZONES_CFG} :: cells of a block (env.py::build_layout)", lambda: src.region_sizes()["zone"], 0),
    Number("eOneInteriorCells", f"{ZONES_CFG} :: interior cells of a block (envs.zones.interior_mask)",
           lambda: src.region_sizes()["interior"], 0),
    Number("eOneNActions", "src/helplessness/envs/dynamics/__init__.py :: N_ACTIONS", n_actions, 0),
    Number("eOneNoiseP", f"{ZONES_CFG} :: blocks.ne.p = blocks.se.p", noise_p, 3),
    Number("eOneNoiseMain", f"{ZONES_CFG} :: 1 - p + p / N_ACTIONS", lambda: 1 - noise_p() + noise_p() / n_actions(), 1),
    Number("eOneNoiseMinor", f"{ZONES_CFG} :: p / N_ACTIONS", lambda: noise_p() / n_actions(), 3),
    # frozen parameters
    Number("eOneGamma", f"{E1_CFG} :: gamma", lambda: cfg("gamma"), 1),
    Number("eOneGoalReward", f"{E1_CFG} :: goal_reward", lambda: cfg("goal_reward"), 0),
    Number("eOneStepReward", f"{E1_CFG} :: step_reward", lambda: cfg("step_reward"), 0, signed=True),
    Number("eOneStepCostHorizon", f"{E1_CFG} :: 1 / (1 - gamma)", step_cost_horizon, 0),
    Number("eOneStepCostShare", f"{E1_CFG} :: (1 / (1 - gamma)) / goal_reward",
           lambda: step_cost_horizon() / cfg("goal_reward"), 0, style="percent"),
    # Step A
    Number("eOneCalibCombos", f"{CALIB} :: number of rows", lambda: src.step_a()["n_combinations"], 0),
    Number("eOneCalibGammas", f"{CALIB} :: distinct gamma", lambda: len(src.step_a()["gammas"]), 0),
    Number("eOneCalibRewards", f"{CALIB} :: distinct R", lambda: len(src.step_a()["rewards"]), 0),
    Number("eOneCalibDirections", f"{CALIB} :: distinct direction", lambda: len(src.step_a()["directions"]), 0),
    Number("eOneCalibPerDirection", f"{CALIB} :: rows per direction (equal for both)",
           lambda: square(list(src.step_a()["per_direction"].values()), "rows per direction"), 0),
    Number("eOneExteriorDiffCtrl", f"{CALIB} :: max exterior_max_abs_diff_exp_vs_ctrl (checked against recommendation.txt)",
           lambda: src.step_a()["exterior_vs_ctrl"], 2, style="sci"),
    Number("eOneExteriorDiffDirection", f"{CALIB} :: max exterior_max_abs_diff_away_vs_toward (checked against recommendation.txt)",
           lambda: src.step_a()["exterior_vs_direction"], 2, style="sci"),
    Number("eOneInteriorGapMax", f"{CALIB} :: max |interior_gap_nw|, |interior_gap_se| (checked against recommendation.txt)",
           lambda: src.step_a()["interior_gap_max"], 2),
    Number("eOneAwayWells", f"{CALIB} :: away_from_goal rows with well_shape_nw = well_shape_se = well",
           lambda: src.step_a()["away_wells_both"], 0),
    Number("eOneTowardNegligible", f"{CALIB} :: toward_goal rows with well_shape_nw = negligible",
           lambda: src.step_a()["toward_nw_negligible"], 0),
    Number("eOneProfileDiffNWSE", f"{CURVES} :: max |V_mean(NW) - V_mean(SE)|, same env, over the whole sweep",
           lambda: src.exterior_profile_differences()["nw_vs_se"], 1, style="sci"),
    # Step B: accuracy
    Number("eOneNSeeds", f"{MEAS} :: distinct seed", lambda: len(src.step_b()["seeds"]), 0),
    Number("eOneNRows", f"{MEAS} :: number of rows", lambda: src.step_b()["n_rows"], 0),
    Number("eOneDeltaViolations", f"{MEAS} :: rows with delta < 0", lambda: src.step_b()["delta_violations"], 0),
    Number("eOneBUDMean", f"{MEAS} :: b, UD, n = 1000, mean over seeds {CHECKED}",
           lambda: b_at("uncontrollable_deterministic", 1000, "mean"), 5, signed=True),
    Number("eOneBUDSd", f"{MEAS} :: b, UD, n = 1000, sd over seeds {CHECKED}",
           lambda: b_at("uncontrollable_deterministic", 1000, "sd"), 0),
    Number("eOneBUNMean", f"{MEAS} :: b, UN, n = 1000, mean over seeds {CHECKED}",
           lambda: b_at("uncontrollable_noisy", 1000, "mean"), 4, signed=True),
    Number("eOneBUNSd", f"{MEAS} :: b, UN, n = 1000, sd over seeds {CHECKED}",
           lambda: b_at("uncontrollable_noisy", 1000, "sd"), 4),
    Number("eOneBUNError", f"{MEAS} :: |mean b - predicted b|, UN, n = 1000 {CHECKED}",
           lambda: abs(b_at("uncontrollable_noisy", 1000, "mean") - src.step_b()["predicted_b"]["uncontrollable_noisy"]), 4),
    Number("eOneBCNMean", f"{MEAS} :: b, CN, n = 1000, mean over seeds {CHECKED}",
           lambda: b_at("controllable_noisy", 1000, "mean"), 4, signed=True),
    Number("eOneBCNSd", f"{MEAS} :: b, CN, n = 1000, sd over seeds {CHECKED}", lambda: b_at("controllable_noisy", 1000, "sd"), 4),
    Number("eOneBCDMean", f"{MEAS} :: b, CD, n = 1000, mean over seeds {CHECKED}",
           lambda: b_at("controllable_deterministic", 1000, "mean"), 0),
    Number("eOnePredBUN", "experiments/E1_offline_4zones/report.py :: predicted_b(uncontrollable_noisy)",
           lambda: src.step_b()["predicted_b"]["uncontrollable_noisy"], 4, signed=True),
    Number("eOnePredBUNShort", "experiments/E1_offline_4zones/report.py :: predicted_b(uncontrollable_noisy)",
           lambda: src.step_b()["predicted_b"]["uncontrollable_noisy"], 2, signed=True),
    Number("eOnePredBUD", "experiments/E1_offline_4zones/report.py :: predicted_b(uncontrollable_deterministic)",
           lambda: src.step_b()["predicted_b"]["uncontrollable_deterministic"], 2, signed=True),
    # Step B: bias cancellation (UN, n = 10)
    Number("eOneBiasIhat", f"{MEAS} :: I_hat - I_star, UN, n = 10, env {CHECKED}", lambda: src.step_b()["bias"][10]["exp"], 3, signed=True),
    Number("eOneBiasIref", f"{MEAS} :: I_hat - I_star, UN, n = 10, control {CHECKED}", lambda: src.step_b()["bias"][10]["ctrl"], 3, signed=True),
    Number("eOneBiasResidual", f"{MEAS} :: |b - predicted b|, UN, n = 10 {CHECKED}", lambda: src.step_b()["bias"][10]["residual"], 4),
    Number("eOneBiasFactor", f"{MEAS} :: cancellation factor (report.py bias_cancellation_factor), UN, n = 10 {CHECKED}",
           lambda: src.step_b()["bias"][10]["factor"], 1),
    # Step B: Delta and L1
    *[Number(f"eOneDeltaCN{BUDGET_WORDS[n]}", f"{MEAS} :: delta, CN, interior, env, n = {n}, mean over seeds {CHECKED}",
             (lambda n=n: src.step_b()["delta"]["controllable_noisy"][n]), d)
      for n, d in ((10, 2), (30, 2), (100, 3), (300, 3), (1000, 0))],
    Number("eOneLoneCNTen", f"{MEAS} :: l1, CN, whole zone, n = 10 {CHECKED}", lambda: src.step_b()["l1"]["controllable_noisy"][10], 2),
    Number("eOneLoneCNThousand", f"{MEAS} :: l1, CN, whole zone, n = 1000 {CHECKED}",
           lambda: src.step_b()["l1"]["controllable_noisy"][1000], 3),
    Number("eOneDeltaUNZeroAt", f"{MEAS} :: first n with delta = 0, UN, interior {CHECKED}",
           lambda: delta_zero_budget("uncontrollable_noisy"), 0),
    Number("eOneLoneUNAtZero", f"{MEAS} :: l1, UN, whole zone, at that n {CHECKED}",
           lambda: src.step_b()["l1"]["uncontrollable_noisy"][delta_zero_budget("uncontrollable_noisy")], 3),
    Number("eOneLoneUNRatio", f"{MEAS} :: l1 at that n / l1 at n = 1000, UN {CHECKED}",
           lambda: src.step_b()["l1"]["uncontrollable_noisy"][delta_zero_budget("uncontrollable_noisy")]
           / src.step_b()["l1"]["uncontrollable_noisy"][1000], 1),
    # Step B: detectability (CN, bootstrap null)
    *[Number(f"eOneThreshold{BUDGET_WORDS[n]}", f"{PERM} :: mean p95, b, bootstrap, CN, n = {n} {CHECKED}",
             (lambda n=n: src.step_b()["p95"]["controllable_noisy"][n]), 3)
      for n in (10, 30, 100, 300, 1000)],
    Number("eOneThresholdCD", f"{PERM} :: max over n of mean p95, b, bootstrap, CD {CHECKED}",
           lambda: max(src.step_b()["p95"]["controllable_deterministic"].values()), 0),
    Number("eOneBSdCNTen", f"{MEAS} :: sd of b over seeds, CN, n = 10 {CHECKED}", lambda: b_at("controllable_noisy", 10, "sd"), 3),
    Number("eOneBSdCNThousand", f"{MEAS} :: sd of b over seeds, CN, n = 1000 {CHECKED}",
           lambda: b_at("controllable_noisy", 1000, "sd"), 4),
    Number("eOneSlope", f"{PERM} :: report.fit_power_law_slope_ci, CN thresholds, slope {CHECKED}", lambda: src.step_b()["slope"][0], 3, signed=True),
    Number("eOneSlopeLow", f"{PERM} :: report.fit_power_law_slope_ci, CN thresholds, CI low {CHECKED}", lambda: src.step_b()["slope"][1], 3, signed=True),
    Number("eOneSlopeHigh", f"{PERM} :: report.fit_power_law_slope_ci, CN thresholds, CI high {CHECKED}", lambda: src.step_b()["slope"][2], 3, signed=True),
    Number("eOneSlopeNoTen", f"{PERM} :: same fit without n = 10, slope {CHECKED}", lambda: src.step_b()["slope_no10"][0], 3, signed=True),
    Number("eOneSlopeNoTenLow", f"{PERM} :: same fit without n = 10, CI low {CHECKED}", lambda: src.step_b()["slope_no10"][1], 3, signed=True),
    Number("eOneSlopeNoTenHigh", f"{PERM} :: same fit without n = 10, CI high {CHECKED}", lambda: src.step_b()["slope_no10"][2], 3, signed=True),
    # Step B: recommended n (UN)
    Number("eOneOutOfTolThirty", f"{MEAS} :: report.out_of_tolerance_rate(UN, n = 30), seeds over {CHECKED}",
           lambda: src.step_b()["out_of_tol"][30][1], 0),
    Number("eOneOutOfTolThirtyShare", f"{MEAS} :: report.out_of_tolerance_rate(UN, n = 30), rate {CHECKED}",
           lambda: src.step_b()["out_of_tol"][30][0], 0, style="percent"),
    Number("eOneCrossingTwoSigma", f"{MEAS} :: (2 A / tolerance)^2, A = report.fit_inverse_sqrt_n {CHECKED}",
           lambda: src.step_b()["n_needed"][2], 0, style="rounded"),
    Number("eOneCrossingThreeSigma", f"{MEAS} :: (3 A / tolerance)^2 {CHECKED}", lambda: src.step_b()["n_needed"][3], 0, style="rounded"),
    Number("eOneRecommendedN", f"{MEAS} :: smallest tested budget past the 2-sigma crossing {CHECKED}",
           lambda: src.step_b()["n_tested"][2], 0),
    Number("eOneRecommendedNThree", f"{MEAS} :: smallest tested budget past the 3-sigma crossing {CHECKED}",
           lambda: src.step_b()["n_tested"][3], 0),
    Number("eOneOutOfTolRecommended", f"{MEAS} :: report.out_of_tolerance_rate(UN, recommended n), seeds over {CHECKED}",
           lambda: src.step_b()["out_of_tol"][src.step_b()["n_tested"][2]][1], 0),
]
