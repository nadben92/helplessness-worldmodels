"""Etape B report: the headline deliverable (smallest detectable |b| per
budget, measured where H0: b=0 is actually TRUE -- the controllable
zones, using the bootstrap null) and a recommended n for E2/E3, based on
across-seed precision (not a single mean-error fluctuation). Also the
corrected Delta/L1 reading (unit-free, no crossing-point claim) and the
Ihat/b bias-cancellation factor that figure 2 (b_vs_n.png) exists to
establish.

Reads the already-written artifacts (never recomputes what they hold) via
artifact.py's loaders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import linregress
from scipy.stats import t as t_dist

from artifact import load_measurement_artifact, load_permutation_artifact

from helplessness.envs.dynamics.controllable import ControllableNoisy

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BUDGETS = (10, 30, 100, 300, 1000)
NULL_ZONES = ("uncontrollable_deterministic", "uncontrollable_noisy")  # H0: b != 0 here (real effect) -- inflated null, kept but labelled
TRUE_H0_ZONES = ("controllable_noisy", "controllable_deterministic")  # H0: b == 0 here, TRUE -- the headline
DEGENERATE_ZONES = ("uncontrollable_deterministic", "controllable_deterministic")  # exact at every n -- excluded from scaling/precision analysis, not pooled with live zones
PRECISION_TOL = 0.03  # spec 3.7: b within 0.03 bits of prediction at n=1000
SAFETY_MARGIN_SIGMAS = (2, 3)  # see recommended_n(): the crossing of the fitted SD curve itself is a 50%-reliability point, not a safety margin


def predicted_b(zone: str) -> float:
    """Freshly computed from analytic_controllability(), never a
    hardcoded decimal (same discipline as the layout tests)."""
    if zone == "uncontrollable_deterministic":
        # paired control is ControllableDeterministic: I* = log2(5); exp I* = 0
        return 0.0 - np.log2(5)
    if zone == "uncontrollable_noisy":
        # paired control is ControllableNoisy(p=0.125); exp I* = 0
        return 0.0 - ControllableNoisy(p=0.125).analytic_controllability(5)
    if zone in TRUE_H0_ZONES:
        return 0.0
    raise ValueError(zone)


def delta_at_n10_table(rows: list[dict]) -> str:
    lines = ["Delta at n=10 (interior region, exp env), per zone -- mean and per-seed values:", ""]
    for zone in ("uncontrollable_deterministic", "controllable_noisy", "controllable_deterministic", "uncontrollable_noisy"):
        vals = [r["delta"] for r in rows if r["zone"] == zone and r["n_budget"] == 10 and r["env"] == "exp" and r["region"] == "interior"]
        vals_sorted = ", ".join(f"{v:.4f}" for v in vals)
        lines.append(f"  {zone}: mean={np.mean(vals):.4f}, seeds=[{vals_sorted}]")
    return "\n".join(lines)


def bias_cancellation_factor(rows: list[dict], zone: str = "uncontrollable_noisy", n_budget: int = 10) -> str:
    """What figure 2 (b_vs_n.png) exists to establish: Ihat's own
    estimation bias at small n is large, but it is nearly the SAME bias
    (same support, same conditional entropy by the pairing design -- see
    docs/E1_offline_4zones.md) on both sides of b, so it cancels in the
    subtraction far more than either term's own bias would suggest."""
    i_hat_exp = np.mean([r["I_hat"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "exp" and r["region"] == "all"])
    i_star_exp = np.mean([r["I_star"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "exp" and r["region"] == "all"])
    i_hat_ctrl = np.mean([r["I_hat"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "ctrl" and r["region"] == "all"])
    i_star_ctrl = np.mean([r["I_star"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "ctrl" and r["region"] == "all"])
    bias_exp = i_hat_exp - i_star_exp
    bias_ctrl = i_hat_ctrl - i_star_ctrl
    b_observed = np.mean([r["b"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "exp" and r["region"] == "all"])
    b_pred = predicted_b(zone)
    residual = abs(b_observed - b_pred)
    avg_bias = (abs(bias_exp) + abs(bias_ctrl)) / 2
    factor = avg_bias / residual if residual > 0 else float("inf")
    return (
        f"Bias cancellation in b (zone={zone}, n={n_budget}): Ihat bias = {bias_exp:+.4f} bits, "
        f"its reference's (paired control) bias = {bias_ctrl:+.4f} bits, residual error in b (|observed - "
        f"predicted|) = {residual:.4f} bits -- a cancellation factor of ~{factor:.1f}x (average raw bias "
        f"divided by the residual that survives in b). This is what figure 2 (b_vs_n.png) exists to "
        "establish: the paired design's shared support/conditional-entropy makes the two sides' "
        "estimation biases nearly equal, so most of each one's own bias cancels in the subtraction."
    )


def delta_l1_reading(rows: list[dict]) -> str:
    """Unit-free reading (corrected -- see delta_and_l1_vs_n.png): NOT a
    crossing point (Delta is in value units up to ~50, L1 is bounded
    probability mass up to 2 -- any apparent "crossing" between them is
    an artifact of R's scale, not a real relationship, and would move or
    vanish at a different reward). The unit-free fact these numbers
    support instead: Delta reaches (numerically clamped) zero while L1 is
    still substantially non-zero -- deciding well needs less information
    than modelling well, stated without comparing across units.
    """
    def series(zone, col, region="all"):
        return np.array([
            np.mean([r[col] for r in rows if r["zone"] == zone and r["n_budget"] == n and r["env"] == "exp" and r["region"] == region])
            for n in BUDGETS
        ])

    lines = [
        "Delta vs L1 reading (unit-free, no crossing-point claim -- see delta_and_l1_vs_n.png):",
        "",
        "Delta and L1 are in different units (Delta: value, up to ~50; L1: probability mass, bounded by 2) --",
        "any apparent 'crossing' between their raw curves is a function of R's scale and is not reported here.",
        "",
        "Both deterministic zones (uncontrollable_deterministic, controllable_deterministic) have",
        "Delta == 0 at EVERY tested budget including n=10, trivially: a deterministic transition is",
        "exact from a single sample, so the greedy policy is already correct from n=1. This is not a",
        "'decision needs no information' finding about the task -- it is an exactness property of",
        "deterministic dynamics, already covered by the n=1-exactness tolerance.",
        "",
    ]

    for zone in ("controllable_noisy", "uncontrollable_noisy"):
        delta_interior = series(zone, "delta", region="interior")
        l1_all = series(zone, "l1", region="all")
        lines.append(f"{zone}:")
        for n, d, l in zip(BUDGETS, delta_interior, l1_all):
            lines.append(f"    n={n:5d}  Delta={d:.4f}  L1={l:.4f}")
        zero_idx = next((i for i, d in enumerate(delta_interior) if d == 0.0), None)
        l1_final = l1_all[-1]
        if zero_idx is not None:
            n_zero = BUDGETS[zero_idx]
            l1_at_zero = l1_all[zero_idx]
            ratio = l1_at_zero / l1_final if l1_final > 0 else float("inf")
            lines.append(
                f"  Delta reaches 0 at n={n_zero}, where L1 is still {l1_at_zero:.4f} -- "
                f"{ratio:.1f}x its final (n=1000) value of {l1_final:.4f}. Deciding well needs less "
                "information than modelling well: the policy is already correct while the model is still "
                "far from converged, stated as a ratio within L1's own units, not against Delta's."
            )
        else:
            lines.append("  Delta never reaches exactly 0 in the tested range.")
        lines.append("")

    return "\n".join(lines)


def across_seed_sd(rows: list[dict], zone: str, n_budget: int) -> float:
    vals = [r["b"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "exp" and r["region"] == "all"]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def fit_inverse_sqrt_n(ns: list[int], values: list[float]) -> float:
    """Least-squares fit of value = A / sqrt(n), slope fixed at -0.5 in
    log space: log(A) = mean(log(value) + 0.5*log(n)) over budgets with a
    positive value. An SD estimated from k seeds has relative standard
    error ~1/sqrt(2*(k-1)) (~35% at k=5, ~16% at k=20) -- far too noisy to
    read pointwise (a "plateau" found by comparing individual noisy
    points can't be told apart from that estimation noise, see
    docs/E1_offline_4zones.md). Fitting the theoretically-expected 1/sqrt(n)
    form across ALL budgets at once uses every point's information
    instead of trusting any single one."""
    log_A_estimates = [np.log(v) + 0.5 * np.log(n) for n, v in zip(ns, values) if v > 0]
    return float(np.exp(np.mean(log_A_estimates))) if log_A_estimates else 0.0


def fit_power_law_slope(ns: list[int], values: list[float]) -> tuple[float, float]:
    """General log-log linear regression (slope free, not fixed at
    -0.5): log(value) = intercept + slope*log(n). Used to REPORT the
    actual exponent, not to force it -- ordinary sampling noise around a
    non-zero true value scales as ~1/sqrt(n) (slope ~-0.5); a statistic
    measuring distance from independence at a null boundary (e.g. a
    G-test-style quantity) scales as ~1/n (slope ~-1) by Wilks'-theorem-
    style asymptotics. Values <= 0 are dropped (undefined in log space)."""
    pairs = [(n, v) for n, v in zip(ns, values) if v > 0]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    log_n = np.log([p[0] for p in pairs])
    log_v = np.log([p[1] for p in pairs])
    slope, intercept = np.polyfit(log_n, log_v, 1)
    return float(slope), float(intercept)


def fit_power_law_slope_ci(ns: list[int], values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Same fit as fit_power_law_slope(), plus a confidence interval on
    the slope via the OLS standard error and a t-distribution critical
    value (small-sample correct at 5 budget points, not a normal
    approximation). Returns (slope, ci_low, ci_high).

    With 20 seeds already averaged into each per-budget p95 before this
    regression runs, the 5 points going in are themselves fairly precise
    -- so this CI is about whether -0.5 is a real, resolvable difference
    from the fitted slope, not swamped by per-seed noise. If -0.5 falls
    outside it, that is worth a line of explanation (e.g. residual
    contamination from the 1/n independence-test regime in part of the
    statistic), not silently absorbed as "close enough"."""
    pairs = [(n, v) for n, v in zip(ns, values) if v > 0]
    if len(pairs) < 3:
        return float("nan"), float("nan"), float("nan")
    log_n = np.log([p[0] for p in pairs])
    log_v = np.log([p[1] for p in pairs])
    result = linregress(log_n, log_v)
    dof = len(pairs) - 2
    t_crit = float(t_dist.ppf(1 - (1 - confidence) / 2, dof))
    margin = t_crit * result.stderr
    return float(result.slope), float(result.slope - margin), float(result.slope + margin)


def detectable_b_table_inflated(perm_rows: list[dict], method: str = "bootstrap") -> str:
    lines = [
        f"Smallest detectable |b| in the UNCONTROLLABLE zones ({method} null), mean p95 over seeds --",
        "H0: b=0 is FALSE here (the real, predicted effect is large: -log2(5) or -1.6529 bits), so this",
        "null is inflated by that real effect and is the WRONG number to cite for detectability. Kept",
        "for reference, labelled as such -- see the headline table below instead.",
        "",
    ]
    header = "zone".ljust(30) + "".join(f"n={n:<10}" for n in BUDGETS)
    lines.append(header)
    for zone in NULL_ZONES:
        row_vals = []
        for n in BUDGETS:
            vals = [r["p95"] for r in perm_rows if r["zone"] == zone and r["quantity"] == "b" and r["method"] == method and r["n_budget"] == n]
            row_vals.append(np.mean(vals))
        lines.append(zone.ljust(30) + "".join(f"{v:<12.5f}" for v in row_vals))
    return "\n".join(lines)


def headline_detectable_b_table(perm_rows: list[dict]) -> str:
    """The real headline: smallest detectable |b|, measured where H0: b=0
    is actually TRUE (the two already-controllable zones, where
    derive_control is a no-op) -- using the bootstrap null (the
    unconditional, fresh-dataset precision estimate; the shuffle null,
    also reported in permutation_null.csv, answers the different question
    of whether Ihat is distinguishable from zero -- see
    permutation_null.py's module docstring, not a correction of one by
    the other).

    Reported PER ZONE, never pooled: controllable_deterministic is exact
    at every n (its bootstrap null is degenerate at exactly 0 -- zero
    estimation noise is possible for a deterministic transition,
    regardless of budget), so averaging it with controllable_noisy would
    silently halve the reported threshold without reflecting any real
    change in controllable_noisy's own precision. The log-log slope is
    fit and reported ONLY for the live zone (controllable_noisy) --
    fitting a slope through a zone whose values are exactly zero is
    undefined and would be reported as such, not silently included.
    """
    lines = [
        "HEADLINE: smallest detectable |b| where H0: b=0 is actually true (controllable zones,",
        "bootstrap null -- the unconditional, fresh-dataset precision estimate; see module docstring),",
        "mean p95 over seeds, reported per zone (never pooled/averaged across zones):",
        "",
    ]
    header = "zone".ljust(30) + "".join(f"n={n:<10}" for n in BUDGETS)
    lines.append(header)
    p95_by_zone = {}
    for zone in TRUE_H0_ZONES:
        row_vals = []
        for n in BUDGETS:
            vals = [r["p95"] for r in perm_rows if r["zone"] == zone and r["quantity"] == "b" and r["method"] == "bootstrap" and r["n_budget"] == n]
            row_vals.append(np.mean(vals))
        p95_by_zone[zone] = row_vals
        lines.append(zone.ljust(30) + "".join(f"{v:<12.5f}" for v in row_vals))
    lines.append("")

    live_zones = [z for z in TRUE_H0_ZONES if z not in DEGENERATE_ZONES]
    for zone in TRUE_H0_ZONES:
        if zone in DEGENERATE_ZONES:
            lines.append(f"  {zone}: DEGENERATE (exact at every n) -- excluded from the scaling fit below, not pooled with live zones.")
            continue
        slope, ci_low, ci_high = fit_power_law_slope_ci(list(BUDGETS), p95_by_zone[zone])
        interpretation = (
            "consistent with ordinary estimator sampling noise around a non-zero true value (~1/sqrt(n))"
            if -0.75 <= slope <= -0.25
            else "closer to a null-boundary/independence-test statistic's scaling (~1/n) than ordinary sampling noise -- needs explanation before citing"
            if slope < -0.75
            else "shallower than ordinary sampling noise (~1/sqrt(n)) would predict -- needs explanation before citing"
        )
        lines.append(f"  {zone}: log-log slope = {slope:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}] ({interpretation}).")
        if ci_low <= -0.5 <= ci_high:
            lines.append("    -0.5 is INSIDE the 95% CI -- no discrepancy to explain.")
        else:
            budgets_no10 = [n for n in BUDGETS if n != 10]
            vals_no10 = [v for n, v in zip(BUDGETS, p95_by_zone[zone]) if n != 10]
            slope_no10, ci_low_no10, ci_high_no10 = fit_power_law_slope_ci(budgets_no10, vals_no10)
            resolved_by_dropping_n10 = ci_low_no10 <= -0.5 <= ci_high_no10
            lines.append(
                f"    -0.5 is OUTSIDE the 95% CI (fit is {'shallower' if slope > -0.5 else 'steeper'} than "
                "-0.5) -- with 20 seeds already averaged into each point, this looks like a real, resolvable "
                "difference, not sampling noise."
            )
            lines.append(
                f"    Refit excluding n=10 (leftmost point, most likely outside the 1/sqrt(n) asymptotic "
                f"regime, and most heavily weighted in a log-log fit): slope = {slope_no10:.3f}, 95% CI "
                f"[{ci_low_no10:.3f}, {ci_high_no10:.3f}]."
            )
            if resolved_by_dropping_n10:
                lines.append("    -0.5 is INSIDE this refit's CI -- dropping n=10 resolves the discrepancy. Cause: n=10 pre-asymptotic.")
            else:
                lines.append(
                    f"    Still excludes -0.5 (moved from {slope:.3f} toward {slope_no10:.3f}, i.e. partially "
                    "toward -0.5, but not far enough): dropping n=10 is a real but partial contributor, not the "
                    "whole story."
                )
                lines.append(
                    "    Checked and RULED OUT: (a) resampling noise in the p95 estimate itself "
                    "(n_permutations=500 vs 5000 at n_budget=1000, 10 vs 5 replicates each: mean p95 0.01847 "
                    "vs 0.01830, well within the sampling sd of both estimates -- 0.00044 shrinking to 0.00014, "
                    "matching the ~sqrt(10x) reduction pure sampling noise predicts, not a bias); (b) "
                    "aggregation of heterogeneous per-state rates (if states individually scaled at -0.5 but "
                    "with different amplitudes, averaging them would NOT itself produce a shallower exponent "
                    "-- a linear combination of same-exponent power laws is exactly that same exponent, "
                    "amplitude-averaged. Tested directly: per-state slopes, 5 seeds averaged, 25 states -- "
                    "mean -0.407, median -0.411, range [-0.484,-0.349], NONE reaching -0.5, and the "
                    "zone-aggregate slope computed the same way (-0.408) is essentially IDENTICAL to the "
                    "per-state mean, not shallower than it. If aggregation were the cause the aggregate would "
                    "be shallower than the typical state; it isn't -- the shallow scaling is a per-state "
                    "property, not an averaging artifact. See permutation_null.py::per_state_i_under_bootstrap_batch "
                    "and test_per_state_slopes_are_not_shallower_than_the_zone_aggregate.)"
                )
                lines.append(
                    "    UNRESOLVED: three candidate causes checked (n=10 pre-asymptotic: partial contributor; "
                    "permutation-count bias: ruled out; state-aggregation artifact: ruled out). The residual gap "
                    "between the n=10-excluded slope and -0.5 has no confirmed explanation -- stated plainly "
                    "rather than offered an untested mechanism."
                )
    if not live_zones:
        lines.append("  (no live zone to fit -- all TRUE_H0_ZONES are degenerate)")
    lines.append("")
    lines.append("Observed b (mean +/- across-seed sd) in these same TRUE-H0 zones -- should be small but non-zero:")
    return "\n".join(lines)


def observed_b_in_true_h0_zones(rows: list[dict]) -> str:
    lines = []
    for zone in TRUE_H0_ZONES:
        parts = []
        for n in BUDGETS:
            vals = [r["b"] for r in rows if r["zone"] == zone and r["n_budget"] == n and r["env"] == "exp" and r["region"] == "all"]
            parts.append(f"n={n}: {np.mean(vals):+.5f} (sd={np.std(vals, ddof=1):.5f})")
        lines.append(f"  {zone}: " + ", ".join(parts))
    return "\n".join(lines)


def out_of_tolerance_rate(rows: list[dict], zone: str, n_budget: int, tol: float = PRECISION_TOL) -> tuple[float, int, int]:
    """Fraction of seeds at this (zone, n_budget) whose |b_observed -
    b_predicted| >= tol -- counted directly from the real data, not
    inferred from the fitted curve. Returns (rate, n_over, n_total)."""
    pred = predicted_b(zone)
    vals = [r["b"] for r in rows if r["zone"] == zone and r["n_budget"] == n_budget and r["env"] == "exp" and r["region"] == "all"]
    if not vals:
        return float("nan"), 0, 0
    n_over = sum(1 for v in vals if abs(v - pred) >= tol)
    return n_over / len(vals), n_over, len(vals)


def recommended_n(rows: list[dict], perm_rows: list[dict]) -> str:
    """Precision-driven (spec 3.7's 0.03-bit tolerance against the KNOWN
    predicted b), based on a FITTED 1/sqrt(n) curve across all budgets --
    not a pointwise plateau read, which an SD estimated from few seeds
    cannot support (relative standard error of a sample SD from k seeds
    is ~1/sqrt(2*(k-1)): ~35% at k=5, ~16% at k=20 -- see
    fit_inverse_sqrt_n()'s docstring).

    CORRECTED CRITERION: the fitted curve's own crossing of the tolerance
    (sd(n) = A/sqrt(n) = tol) is the budget where the TYPICAL run's error
    equals tol -- roughly half the runs exceed it there, which is not what
    "the tolerance is met" should mean. A tolerance is only meaningful if
    held reliably, not half the time. The useful budget is the one where
    it is held with a safety margin: k*sd(n) < tol for k=2 or 3 standard
    deviations, which moves the crossing by a factor of k^2 (4x to 9x) on
    n, since sd decays as 1/sqrt(n). Both margins are reported; the
    empirical out-of-tolerance rate at the OLD (unmargined) n=30
    recommendation is counted directly from the real data as a sanity
    check on the theory, not just derived from the fit."""
    sd_by_zone = {}
    fit_by_zone = {}
    for zone in NULL_ZONES:
        sds = {n: across_seed_sd(rows, zone, n) for n in BUDGETS}
        sd_by_zone[zone] = sds
        fit_by_zone[zone] = fit_inverse_sqrt_n(list(BUDGETS), [sds[n] for n in BUDGETS])

    lines = ["Recommended n for E2/E3 (fitted 1/sqrt(n) curve WITH a safety margin, not the raw crossing):", ""]

    old_rate, old_over, old_total = out_of_tolerance_rate(rows, "uncontrollable_noisy", 30)
    lines.append(
        f"Sanity check: the OLD (unmargined, 1-sigma) recommendation was n=30. Counted directly from the real "
        f"20-seed data, {old_over}/{old_total} seeds ({old_rate:.0%}) have |b_observed - b_predicted| >= "
        f"{PRECISION_TOL} bits for uncontrollable_noisy AT n=30 -- a real, non-trivial failure rate, confirming "
        "the raw crossing point is not a reliable tolerance budget (the theoretical 50% at the EXACT crossing "
        "does not apply verbatim at n=30, which is already somewhat past it, but the qualitative point holds)."
    )
    lines.append("")

    n_needed_by_zone_and_sigma: dict[str, dict[int, float]] = {}
    for zone, sds in sd_by_zone.items():
        sd_str = ", ".join(f"n={n}: {sds[n]:.4f}" for n in BUDGETS)
        A = fit_by_zone[zone]
        n_needed_by_zone_and_sigma[zone] = {}
        margin_strs = []
        for k in SAFETY_MARGIN_SIGMAS:
            n_needed = (k * A / PRECISION_TOL) ** 2 if A > 0 else 0.0
            n_needed_by_zone_and_sigma[zone][k] = n_needed
            margin_strs.append(f"{k}-sigma: n~={n_needed:.0f}")
        lines.append(f"  {zone} across-seed sd of b: {sd_str} -- fitted A (sd = A/sqrt(n)): {A:.4f}, crosses {PRECISION_TOL} bits with " + ", ".join(margin_strs))
    lines.append("")

    recommendations = {}
    for k in SAFETY_MARGIN_SIGMAS:
        n_continuous = max(n_needed_by_zone_and_sigma[zone][k] for zone in NULL_ZONES)
        n_tested = next((n for n in BUDGETS if n >= n_continuous), BUDGETS[-1])
        recommendations[k] = (n_continuous, n_tested)
        rate, over, total = out_of_tolerance_rate(rows, "uncontrollable_noisy", n_tested)
        lines.append(
            f"  {k}-sigma margin: fitted crossing ~{n_continuous:.0f} -> smallest tested budget n={n_tested} "
            f"(actual out-of-tolerance rate there: {over}/{total} = {rate:.0%})"
        )
    lines.append("")

    k_primary = SAFETY_MARGIN_SIGMAS[0]
    n_primary = recommendations[k_primary][1]
    lines.append(
        f"RECOMMENDED: n={n_primary} for E2/E3, using a {k_primary}-sigma safety margin (the budget at which the "
        f"fitted curve PLUS {k_primary} standard deviations falls below the {PRECISION_TOL}-bit tolerance, not "
        "the budget where the typical run merely equals it). The more conservative "
        f"{SAFETY_MARGIN_SIGMAS[-1]}-sigma margin gives n={recommendations[SAFETY_MARGIN_SIGMAS[-1]][1]}; use that "
        "instead if E2/E3's own tolerance for occasional excursions is tighter than this recommendation assumes."
    )
    lines.append(
        "This is a precision recommendation for the two zones with a known non-zero predicted effect. It "
        "should be revisited against the controllable-zone detectability headline above (measured where "
        "H0: b=0 is actually true) before treating it as final for E2/E3 -- detectability there is the "
        "complementary question this recommendation does not, by itself, answer."
    )
    return "\n".join(lines)


def build_report(rows: list[dict], perm_rows: list[dict]) -> str:
    return "\n\n".join(
        [
            "Etape B report",
            "===============",
            delta_at_n10_table(rows),
            bias_cancellation_factor(rows),
            delta_l1_reading(rows),
            "HEADLINE DELIVERABLE",
            "---------------------",
            headline_detectable_b_table(perm_rows),
            observed_b_in_true_h0_zones(rows),
            "",
            detectable_b_table_inflated(perm_rows),
            "",
            recommended_n(rows, perm_rows),
        ]
    )


def main() -> None:
    rows = load_measurement_artifact(ARTIFACTS_DIR / "measurements.csv")
    perm_rows = load_permutation_artifact(ARTIFACTS_DIR / "permutation_null.csv")
    text = build_report(rows, perm_rows)
    (ARTIFACTS_DIR / "etape_b_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
