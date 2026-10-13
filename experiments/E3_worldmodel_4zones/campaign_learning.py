"""The E3 campaign: the model's learning and the agent's behaviour, the measurement error in the noisy zones, and the
convergence of the deterministic zones (docs/e3/DECISIONS.md, 2026-10-13 (31), B, C and E -- recorded before any
campaign value). Read-only on the runs; works on campaign_analysis.Run pairs.

B. Learning and behaviour, per run, zone (UD, CN, UN, CD, corridor) and checkpoint -- descriptive, no rule:
   at every checkpoint (checkpoints.jsonl, counts.npz, twin.npz): the twin's L1 against P (mean over the zone's
   (s, a) pairs); the success over the last 100 episodes and over the interval since the previous checkpoint (95 %
   Wilson interval), the episode length (last 100 episodes); the agent's Delta under P by zone (real contexts; the four
   zones, as recorded); the prior's mass on the goal for the 4 pairs into it and their median; the Q gap (into the
   goal - best other) at the 4 states next to the goal, against the tie tolerance;
   at every measured checkpoint (measures/env<NNNNNN>/): C-a (replay template accuracy >= 0.99), C-c (z-only probe >=
   0.99), C-b by zone (pairs with N[s, a] >= 100: L1_WM <= L1_B + 0.05); the prior's mean mass on the true successor
   over the deterministic (s, a) pairs of the zone (the deterministic zones).
C. Measurement error in the noisy zones: excess = I_s - I_true (zone means; I_true from the run's own kernel) in CN and
   UN, per run and measured checkpoint; over the seeds, at the checkpoints measured in every pair: the mean and sd of
   the excess per condition, and of b; the resolution an extension to the noisy zones would have with N seeds, the
   half-width of b's 95 % Student-t interval (t(0.975, N - 1) sd(b) / sqrt(N)); the hybrid-code rule's verdict ((29)).
E. Convergence of the deterministic zones: the first candidate checkpoint (measured in all 8 runs, before the budget)
   from which, in every run, (a) the agent has finished learning ((27) a) and (b) I_s of UD, CD and the corridor does not
   change by more than the measurement noise, with a Bonferroni correction (convergence() below).
"""

from __future__ import annotations

import math

import numpy as np

from helplessness.experiment_loader import load_experiment_module

ca = load_experiment_module("E3_worldmodel_4zones", "campaign_analysis.py", "campaign_learning_analysis")
aggregate_b = ca.aggregate_b
ZONES, CATEGORIES = ca.ZONES, ca.CATEGORIES
NOISY = ("controllable_noisy", "uncontrollable_noisy")
DETERMINISTIC_FOR_CONVERGENCE = ("uncontrollable_deterministic", "controllable_deterministic", "corridor")
WILSON_Z = 1.959963984540054

# (27) a: the agent has finished learning
GOAL_MASS_MIN, DELTA_MAX = 0.9, 1.0
# E: the measurement noise of a zone's I_s at x4 (sd due to the contexts, DECISIONS (28) task a, final weights of the
# pilot's four runs EXP 0, EXP 5, REF 0, REF 5); one sigma per zone, the root mean square over the four runs. The corridor
# was not measured in (28): CD's sigma scaled to the corridor's 43 non-terminal states (sd of a zone mean ~ 1 / sqrt(states);
# CD has 25) -- an assumption, recorded in (31).
NOISE_X4_PILOT = {"uncontrollable_deterministic": (0.0005, 0.0016, 0.0024, 0.0026),
                  "controllable_deterministic": (0.0026, 0.0023, 0.0010, 0.0022)}
CORRIDOR_STATES, CD_STATES = 43, 25
FAMILY_ALPHA = 0.05


def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float, float]:
    """Rate and 95 % Wilson interval of k successes in n trials (NaN without a trial)."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, centre - half, centre + half


def zone_sigma() -> dict[str, float]:
    sig = {z: float(np.sqrt(np.mean(np.square(v)))) for z, v in NOISE_X4_PILOT.items()}
    sig["corridor"] = sig["controllable_deterministic"] * math.sqrt(CD_STATES / CORRIDOR_STATES)
    return sig


def _zone_pairs(run, zone: str) -> np.ndarray:
    return ca.zone_mask(run, zone)


# --- B -----------------------------------------------------------------------------------------------------------------
def learning_run(run) -> dict:
    """B for one run: a row per checkpoint."""
    steps = sorted(run.records)
    P = run.P_true
    rows = []
    prev = None
    for c in steps:
        r = run.records[c]
        row = {"env_step": c, "measured": c in run.measure_dirs, "success_last100": r.get("success_rate"),
               "episode_length_last100": r.get("episode_length")}
        k = r["successes"] - (prev["successes"] if prev else 0)
        n = r["episodes"] - (prev["episodes"] if prev else 0)
        row["interval"] = {"from": prev["env_step"] if prev else 0, "episodes": int(n), "successes": int(k),
                           "rate_ci95": list(wilson(int(k), int(n)))}
        prev = r
        row["agent_delta_by_zone"] = (r.get("agent_delta") or {}).get("real_contexts", {}).get("delta_by_zone")
        g = r.get("goal_diagnostics") or {}
        row["goal_masses"] = [p.get("mass_on_goal") for p in g.get("goal_pairs", [])]
        row["median_goal_mass"] = g.get("median_mass_on_goal_pairs")
        row["goal_draws"] = g.get("goal_draws", g.get("draws"))
        row["q_gaps"] = [{"s": q["s"], "gap": q["gap_into_goal_minus_best_other"], "tolerance": q["tie_tolerance"]}
                         for q in g.get("goal_neighbours", [])] or None
        row["V_by_zone"] = g.get("V_by_zone")
        if run.twin is not None and f"T_hat_{c}" in run.twin:
            l1 = np.abs(run.twin[f"T_hat_{c}"].astype(float) - P).sum(-1)
            row["twin_l1"] = {z: float(l1[_zone_pairs(run, z)].mean()) for z in ZONES}
        if c in run.measure_dirs:
            res, arr = run.measure(c)
            row["calibration"] = calibration_all_zones(res, arr, run)
            row["mass_on_true_successor"] = successor_mass(arr["real_p_hat"], P, run)
        rows.append(row)
    return {"seed": run.seed, "condition": run.condition, "switch_env_step": run.switch_env_step, "checkpoints": rows}


def calibration_all_zones(res: dict, arr: dict, run) -> dict:
    """C-a, C-c and C-b (aggregate_b's definitions), C-b for the corridor too."""
    N_sa = arr["N"].sum(-1)
    l1_wm = np.abs(arr["real_p_hat"] - arr["P_true"]).sum(-1)
    l1_b = np.abs(arr["T_hat_B"] - arr["P_true"]).sum(-1)
    c_b = {}
    for z in ZONES:
        m = _zone_pairs(run, z)[:, None] & (N_sa >= aggregate_b.C_B_MIN_COUNT) & np.isfinite(l1_wm)
        wm, b = (float(l1_wm[m].mean()), float(l1_b[m].mean())) if m.any() else (float("nan"), float("nan"))
        c_b[z] = {"l1_wm": wm, "l1_b": b, "n_pairs": int(m.sum()), "ok": bool(m.any() and wm <= b + aggregate_b.C_B_MARGIN)}
    acc, zonly = res["accuracies"]["acc_replay_hard"], res["z_only_probe"]["test_accuracy"]
    return {"c_a": {"value": acc, "ok": bool(acc >= aggregate_b.C_A)}, "c_c": {"value": zonly, "ok": bool(zonly >= aggregate_b.C_C)}, "c_b": c_b}


def successor_mass(p_hat: np.ndarray, P: np.ndarray, run) -> dict:
    """The prior's mean mass on the true successor over the zone's deterministic (s, a) pairs (NaN if none)."""
    det = P.max(-1) >= 1.0 - 1e-12
    succ = P.argmax(-1)
    mass = np.take_along_axis(p_hat, succ[..., None], axis=-1)[..., 0]
    out = {}
    for z in ZONES:
        m = _zone_pairs(run, z)[:, None] & det & np.isfinite(mass)
        out[z] = float(mass[m].mean()) if m.any() else float("nan")
    return out


# --- C -----------------------------------------------------------------------------------------------------------------
def noise_error(pairs: list, main: dict) -> dict:
    """C over the campaign's pairs (campaign_analysis.Run pairs) and the main analysis (its hybrid-code verdict)."""
    per_run, common = [], None
    for exp, ref in pairs:
        cs = sorted(set(exp.measure_dirs) & set(ref.measure_dirs))
        common = set(cs) if common is None else common & set(cs)
        for run in (exp, ref):
            rows = {}
            for c in sorted(run.measure_dirs):
                I = run.measure(c)[1]["real_I_s"]
                m_true = ca.zone_means_all(run.truth["I_true"], run.zone_of_state)
                m_I = ca.zone_means_all(I, run.zone_of_state)
                rows[c] = {z: {"I_s": m_I[z], "I_true": m_true[z], "excess": m_I[z] - m_true[z]} for z in NOISY}
            per_run.append({"seed": run.seed, "condition": run.condition, "checkpoints": rows})
    common = sorted(common or [])
    t = {}
    for c in common:
        t[c] = {}
        for z in NOISY:
            ent = {}
            for cond in ("experimental", "reference"):
                v = [r["checkpoints"][c][z]["excess"] for r in per_run if r["condition"] == cond]
                ent[f"excess_{cond}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), "values": v}
            bs = []
            for exp, ref in pairs:
                e = next(r for r in per_run if r["seed"] == exp.seed and r["condition"] == "experimental")["checkpoints"][c][z]["I_s"]
                f = next(r for r in per_run if r["seed"] == ref.seed and r["condition"] == "reference")["checkpoints"][c][z]["I_s"]
                bs.append(e - f)
            n = len(bs)
            sd = float(np.std(bs, ddof=1)) if n > 1 else float("nan")
            from scipy.stats import t as student

            ent["b"] = {"mean": float(np.mean(bs)), "sd_over_seeds": sd, "values": bs,
                        "resolution_half_width_95": float(student.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")}
            t[c][z] = ent
    rule = (main.get("aggregate_b") or {}).get("campaign_rule", {})
    return {"per_run": per_run, "common_checkpoints": common, "over_seeds": t,
            "hybrid_rule": {"status": rule.get("status"), "zones": {z: {k: rule["zones"][z].get(k) for k in ("verdict", "b_difference", "off_support_difference")}
                                                                    for z in rule.get("zones", {})}}}


# --- E -----------------------------------------------------------------------------------------------------------------
def agent_finished(run, c: int, last_interval: tuple[float, float]) -> dict:
    """(27) a at checkpoint c: median goal mass >= 0.9; the success over the next interval compatible with the last
    interval's (95 % Wilson intervals overlap); the agent's Delta < 1.0 in every zone."""
    steps = sorted(run.records)
    r = run.records[c]
    g = r.get("goal_diagnostics") or {}
    mass = g.get("median_mass_on_goal_pairs")
    i = steps.index(c)
    if i + 1 < len(steps):
        a, b = run.records[steps[i]], run.records[steps[i + 1]]
        _, lo, hi = wilson(b["successes"] - a["successes"], b["episodes"] - a["episodes"])
        overlap = bool(hi >= last_interval[0] and last_interval[1] >= lo)
    else:
        overlap = True  # the budget's checkpoint: no next interval
    delta = (r.get("agent_delta") or {}).get("real_contexts", {}).get("delta_by_zone") or {}
    dmax = max(delta.values()) if delta else float("nan")
    ok = bool(mass is not None and mass >= GOAL_MASS_MIN and overlap and np.isfinite(dmax) and dmax < DELTA_MAX)
    return {"median_goal_mass": mass, "next_interval_overlaps_last": overlap, "max_delta": dmax, "ok": ok}


def convergence(runs: list) -> dict:
    """E. Candidates: the checkpoints measured in all runs, before the budget. For a candidate c, in every run:
    (a) agent_finished at c and at every later candidate;
    (b) for each zone Z of DETERMINISTIC_FOR_CONVERGENCE, |I_s(next) - I_s(this)| over every pair of consecutive
        candidates from c on, and |I_s(budget) - I_s(c)|, all below T_Z(c) = k(m) sqrt(2) sigma_Z, sigma_Z = zone_sigma(),
        k(m) = Phi^-1(1 - FAMILY_ALPHA / (2 m)), m the number of differences tested for c over all runs and zones
        (Bonferroni: under no change and Gaussian noise of sd sigma_Z per measure, the probability that some difference
        exceeds its threshold is at most FAMILY_ALPHA).
    The first candidate where both hold is the budget for the next runs; none: the budget stays at 300,000."""
    from scipy.stats import norm

    budget = ca.BUDGET_ENV_STEPS
    measured = [set(r.measure_dirs) for r in runs]
    common = sorted(set.intersection(*measured)) if runs else []
    if budget not in common:
        return {"status": f"not computable: {budget} not measured in every run", "budget": budget}
    cands = [c for c in common if c < budget]
    sig = zone_sigma()
    series = {}
    for r in runs:
        key = f"seed{r.seed}_{r.condition}"
        series[key] = {c: ca.zone_means_all(r.measure(c)[1]["real_I_s"], r.zone_of_state) for c in common}
    last = {}
    for r in runs:
        steps = sorted(r.records)
        a, b = r.records[steps[-2]], r.records[steps[-1]]
        _, lo, hi = wilson(b["successes"] - a["successes"], b["episodes"] - a["episodes"])
        last[f"seed{r.seed}_{r.condition}"] = (lo, hi)
    table = []
    first = None
    for c in cands:
        later = [x for x in common if x >= c]
        comparisons = list(zip(later[:-1], later[1:]))
        if (c, budget) not in comparisons:
            comparisons.append((c, budget))
        m = len(runs) * len(DETERMINISTIC_FOR_CONVERGENCE) * len(comparisons)
        k = float(norm.ppf(1 - FAMILY_ALPHA / (2 * m)))
        thr = {z: k * math.sqrt(2) * sig[z] for z in DETERMINISTIC_FOR_CONVERGENCE}
        a_ok, b_ok, failures = True, True, []
        for r in runs:
            key = f"seed{r.seed}_{r.condition}"
            for x in [x for x in cands if x >= c]:
                fin = agent_finished(r, x, last[key])
                if not fin["ok"]:
                    a_ok = False
                    failures.append(f"{key} (a) at {x}: goal mass {fin['median_goal_mass']}, next interval overlaps "
                                    f"{fin['next_interval_overlaps_last']}, max Delta {fin['max_delta']:.3f}")
            for (u, v) in comparisons:
                for z in DETERMINISTIC_FOR_CONVERGENCE:
                    d = abs(series[key][v][z] - series[key][u][z])
                    if not d < thr[z]:
                        b_ok = False
                        failures.append(f"{key} (b) {z} {u} -> {v}: {d:.4f} >= {thr[z]:.4f}")
        row = {"checkpoint": c, "comparisons_per_run_zone": len(comparisons), "m": m, "k": k, "thresholds": thr,
               "a": a_ok, "b": b_ok, "failures": failures}
        table.append(row)
        if first is None and a_ok and b_ok:
            first = c
    return {"status": "computed", "candidates": cands, "sigma": sig, "family_alpha": FAMILY_ALPHA, "table": table,
            "first_checkpoint": first, "budget_next": first if first is not None else budget,
            "verdict": (f"budget for the next runs: {first:,} environment steps" if first is not None
                        else f"no candidate satisfies (a) and (b): the budget stays at {budget:,}")}


def analyse_learning(pairs: list, main: dict) -> dict:
    runs = [r for p in pairs for r in p]
    return {"learning": [learning_run(r) for r in runs], "noise_error": noise_error(pairs, main), "convergence": convergence(runs)}
