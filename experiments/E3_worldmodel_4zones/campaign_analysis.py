"""The E3 campaign's analyses (docs/e3/DECISIONS.md, 2026-10-07 (29), recorded before the launch). Read-only.

MAIN, at the budget's checkpoint (300,000 environment steps, the runs' final measures): b by zone on I_s with the
campaign rule for the hybrid codes (aggregate_b.campaign_rule: N = 4, FRACTION = 1/2, REFERENCE EFFECT = 0.02 bit, E2's
effect in CN) and E3-0's aberration control (aggregate_b, configs/e3/e30_aberration_control.yaml). With N = 4 the rule
may declare zones NOT interpretable; reported as it is.

SECONDARY, at every checkpoint measured in both runs of a pair (environment / reference, same seed):
  - b by zone on I_s and on I_s projected;
  - b by visit class: each state's visits are those of its least-visited action, n(s) = min_a N[s, a] (the counts at
    the checkpoint, counts.npz); classes VISIT_CLASSES; in each zone and class, the mean I_s over the environment run's
    states of that class minus the mean over the reference run's states of that class (each run classes its own states),
    with the number of states on each side;
  - difference of differences, by zone: [I_WM - I_twin]_env - [I_WM - I_twin]_ref, I_WM the zone's I_s (real contexts)
    and I_twin the tabular twin's I on the same run's counts at the checkpoint (checkpoints.jsonl twin_by_zone I_hat);
    DESCRIPTIVE (2026-10-07 (30)), reported with its parts: b_WM, b_twin, and each run's I_WM - I_true and
    I_twin - I_true (I_true: the run's true kernel);
  - alpha, the weight of an action-independent component in the predicted kernel (measures.kernel_probe.
    action_independent_weight on the real-context kernel against the run's true kernel), mean over each zone's states,
    environment, reference and their difference (expected ~0 in the controllable zones);
  - the zones' separation in the latent: recorded in the run at every checkpoint (checkpoints.jsonl
    goal_diagnostics.latent_zone_separation, on h and z) -- or, for runs measured without it (the pilot), computed here
    from the measure's per-context posteriors (z only; arrays real_post_probs, real_context_states);
  - from the runs' records, when present: V by zone and the Q gaps next to the goal (reported, no rule).

Inputs: run directories (results.json, checkpoints.jsonl, counts.npz; measures/env<NNNNNN>/ for the campaign). For runs
whose checkpoints were measured elsewhere (the pilot's gpu_pilot_remeasure), --measures-root DIR reads DIR/<run name>/
env<NNNNNN>/ instead.
Usage: python experiments/E3_worldmodel_4zones/campaign_analysis.py <run dir> ... [--measures-root DIR] [--out DIR]
Outputs: CAMPAIGN_ANALYSIS.md and campaign_analysis.json in --out (default: printed only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.kernel_probe import action_independent_weight

aggregate_b = load_experiment_module("E3_worldmodel_4zones", "aggregate_b.py", "campaign_analysis_aggregate_b")
e3_env = aggregate_b.e3_env
CATEGORIES = aggregate_b.CATEGORIES
# the campaign rule's parameters (user decision, 2026-10-07 (29))
N_SEEDS, FRACTION, REFERENCE_EFFECT = 4, 0.5, 0.02
BUDGET_ENV_STEPS = 300_000
VISIT_CLASSES = ((0, 9), (10, 29), (30, 99), (100, 299), (300, 999), (1000, None))
PROBE_TEST_FRAC, PROBE_SEED = 0.2, 20261007


@lru_cache(maxsize=1)
def _run_mini():
    return load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "campaign_analysis_run_mini")


def controllability_of_kernel(P: np.ndarray) -> np.ndarray:
    """Per-state I of a kernel, the one implementation (measures.controllability via run_mini)."""
    return _run_mini().controllability_of_kernel(P)


def class_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def min_action_visits(N: np.ndarray) -> np.ndarray:
    """n(s) = min_a N[s, a]: the visits of the state's least-visited action (N: (S, A, S) counts)."""
    return N.sum(axis=-1).min(axis=-1)


def b_by_visit_class(I_env: np.ndarray, n_env: np.ndarray, I_ref: np.ndarray, n_ref: np.ndarray, zone: np.ndarray) -> dict:
    """In each zone and visit class: mean I over the environment run's states of the class minus the mean over the
    reference run's states of the class (NaN states left out), with the counts of states on each side."""
    out = {}
    for cat in CATEGORIES:
        out[cat] = {}
        for lo, hi in VISIT_CLASSES:
            def sel(I, n):
                m = (zone == cat) & (n >= lo) & ((n <= hi) if hi is not None else True) & np.isfinite(I)
                return I[m]
            e, r = sel(I_env, n_env), sel(I_ref, n_ref)
            out[cat][class_label(lo, hi)] = {"b": float(e.mean() - r.mean()) if e.size and r.size else float("nan"),
                                             "n_states_env": int(e.size), "n_states_ref": int(r.size)}
    return out


def zone_means_all(values: np.ndarray, zone: np.ndarray) -> dict:
    """Mean over each zone's states, the corridor included (the goal, terminal, left out of it)."""
    out = {}
    for cat in (*CATEGORIES, "corridor"):
        m = zone == cat
        m[goal_state()] = False
        out[cat] = float(np.nanmean(values[m])) if np.isfinite(values[m]).any() else float("nan")
    return out


def zone_mean(values: np.ndarray, zone: np.ndarray) -> dict:
    return {cat: float(np.nanmean(values[zone == cat])) if np.isfinite(values[zone == cat]).any() else float("nan") for cat in CATEGORIES}


def separation_from_arrays(arrays, zone_of_state: list[str]) -> dict | None:
    """The zones' separation on the measure's per-context posteriors (z only), for runs without the in-run record."""
    if "real_post_probs" not in arrays:
        return None
    probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "campaign_analysis_probe")
    x = arrays["real_post_probs"].reshape(arrays["real_post_probs"].shape[0], -1)
    out = probe.latent_zone_separation(x, arrays["real_context_states"], zone_of_state, list(CATEGORIES), PROBE_TEST_FRAC,
                                       np.random.default_rng(PROBE_SEED))
    return {**out, "latent": "z only (posterior probabilities, from the measure's arrays)"}


class Run:
    def __init__(self, path: Path, measures_root: Path | None = None):
        self.path = Path(path)
        self.results = json.loads((self.path / "results.json").read_text(encoding="utf-8"))
        self.seed, self.condition = int(self.results["seed"]), self.results["condition"]
        self.zone_class = e3_env.block_class_for_seed(self.seed)
        self.config = aggregate_b.config_of(str(self.path))
        self.records = {r["env_step"]: r for r in (json.loads(line) for line in
                                                   (self.path / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines())}
        self.counts = np.load(self.path / "counts.npz")
        base = (measures_root / self.path.name) if measures_root is not None else (self.path / "measures")
        self.measure_dirs = {int(d.name[3:]): d for d in sorted(base.glob("env*")) if (d / "arrays.npz").exists()}
        self._measures: dict[int, tuple[dict, dict]] = {}
        self.twin = dict(np.load(self.path / "twin.npz")) if (self.path / "twin.npz").exists() else None
        self.switch_env_step = self.results.get("switch_env_step") or switch_from_log(self.path / "console.log")
        if self.measure_dirs:
            res, arr = self.measure(max(self.measure_dirs))
            self.zone_of_state = np.array(res["zone_of_state"])
            self.P_true = arr["P_true"]
            self.truth = state_truth(self.P_true)

    def measure(self, c: int) -> tuple[dict, dict]:
        """results.json and arrays.npz of the measures at checkpoint c (loaded once)."""
        if c not in self._measures:
            d = self.measure_dirs[c]
            self._measures[c] = (json.loads((d / "results.json").read_text(encoding="utf-8")), dict(np.load(d / "arrays.npz")))
        return self._measures[c]


def switch_from_log(path: Path) -> int | None:
    """The switch's environment step from a run's console.log (run_agent: '>>> SWITCH ... environment step N')."""
    if not path.exists():
        return None
    m = re.search(r">>> SWITCH.*?environment step (\d+)", path.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def state_truth(P: np.ndarray) -> dict:
    """Per state, from the run's own true kernel: I_true (the one implementation), controllable (I_true > 0) and
    stochastic (some action without a single successor)."""
    I_true = controllability_of_kernel(P)
    return {"I_true": I_true, "controllable": I_true > 1e-9, "stochastic": (P.max(axis=-1) < 1.0 - 1e-12).any(axis=-1)}


def checkpoint_pair(exp: Run, ref: Run, c: int) -> dict:
    (res_e, arr_e), (res_r, arr_r) = exp.measure(c), ref.measure(c)
    zone = np.array(res_e["zone_of_state"])
    rc_e, rc_r = res_e["real_context"], res_r["real_context"]
    out = {"env_step": c, "replay_env_steps": {"experimental": res_e.get("replay_env_steps", c), "reference": res_r.get("replay_env_steps", c)}}
    v = aggregate_b._nan  # NaN for a zone without a measured state (null in results.json)
    for m in ("I_s", "I_s_projected"):
        out[f"b_{m}"] = {cat: v(rc_e[f"{m}_by_zone"][cat]["mean"]) - v(rc_r[f"{m}_by_zone"][cat]["mean"]) for cat in CATEGORIES}
    N_e, N_r = exp.counts[f"N_{c}"], ref.counts[f"N_{c}"]
    out["b_by_visit_class"] = b_by_visit_class(arr_e["real_I_s"], min_action_visits(N_e), arr_r["real_I_s"], min_action_visits(N_r), zone)
    twin_e, twin_r = exp.records[c]["twin_by_zone"], ref.records[c]["twin_by_zone"]
    did = {}
    for cat in CATEGORIES:
        e = v(rc_e["I_s_by_zone"][cat]["mean"]) - v(twin_e[cat]["I_hat"])
        r = v(rc_r["I_s_by_zone"][cat]["mean"]) - v(twin_r[cat]["I_hat"])
        did[cat] = {"wm_minus_twin_env": e, "wm_minus_twin_ref": r, "did": e - r}
    out["difference_of_differences"] = did
    # its decomposition (2026-10-07 (30)): did = b_WM - b_twin; each run's I_WM and I_twin against the true kernel's I
    zone_e, zone_r = np.array(res_e["zone_of_state"]), np.array(res_r["zone_of_state"])
    true_e = zone_mean(controllability_of_kernel(arr_e["P_true"]), zone_e)
    true_r = zone_mean(controllability_of_kernel(arr_r["P_true"]), zone_r)
    out["decomposition"] = {cat: {"b_wm": out["b_I_s"][cat], "b_twin": v(twin_e[cat]["I_hat"]) - v(twin_r[cat]["I_hat"]),
                                  "wm_minus_true_env": v(rc_e["I_s_by_zone"][cat]["mean"]) - true_e[cat],
                                  "wm_minus_true_ref": v(rc_r["I_s_by_zone"][cat]["mean"]) - true_r[cat],
                                  "twin_minus_true_env": v(twin_e[cat]["I_hat"]) - true_e[cat],
                                  "twin_minus_true_ref": v(twin_r[cat]["I_hat"]) - true_r[cat]} for cat in CATEGORIES}
    a_e = zone_means_all(action_independent_weight(arr_e["real_p_hat"], arr_e["P_true"]), zone)
    a_r = zone_means_all(action_independent_weight(arr_r["real_p_hat"], arr_r["P_true"]), zone)
    out["alpha"] = {cat: {"env": a_e[cat], "ref": a_r[cat], "difference": a_e[cat] - a_r[cat]} for cat in ZONES}
    # b on I_s with the corridor (its states' mean, the goal left out; the four zones: the measures' own zone means)
    corridor = zone_means_all(arr_e["real_I_s"], zone)["corridor"] - zone_means_all(arr_r["real_I_s"], zone)["corridor"]
    out["b_I_s_all"] = {**out["b_I_s"], "corridor": corridor}
    sep = {}
    for role, run, arr, res in (("experimental", exp, arr_e, res_e), ("reference", ref, arr_r, res_r)):
        g = run.records[c].get("goal_diagnostics", {})
        sep[role] = ({**g["latent_zone_separation"], "latent": "h and z (recorded in the run)"} if "latent_zone_separation" in g
                     else separation_from_arrays(arr, res["zone_of_state"]))
    out["latent_zone_separation"] = sep
    out["agent"] = {role: {k: run.records[c]["goal_diagnostics"][k] for k in ("V_by_zone", "goal_neighbours")
                           if k in run.records[c].get("goal_diagnostics", {})} for role, run in (("experimental", exp), ("reference", ref))}
    return out


# --- propagation (docs/e3/DECISIONS.md, 2026-10-13 (31), D): definitions recorded before any campaign value ---------------
ZONES = (*CATEGORIES, "corridor")
CONTROLLABLE_ZONES = ("controllable_noisy", "controllable_deterministic", "corridor")  # controllable in both runs of a pair
DISTANCE_MAIN_ZONE = "controllable_deterministic"
N_GRID_EQUAL_VISITS = 8
L1_VISIT_BINS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 30), (31, 100), (101, 300), (301, None))
LOW_N, LOW_N_MIN_PAIRS = 10, 5
GRID_SHAPE = (12, 12)


def goal_state() -> int:
    return int(e3_env.goal_state())


def zone_mask(run: Run, zone: str) -> np.ndarray:
    """The zone's states, the goal (terminal: no transition out of it) left out of the per-state analyses."""
    m = run.zone_of_state == zone
    m[goal_state()] = False
    return m


def _curve(n: np.ndarray, I: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One state's (log10 n, I_s, log10 checkpoint) over the measured checkpoints, n >= 1 only, ties in n averaged."""
    ok = (n >= 1) & np.isfinite(I)
    x, y, t = np.log10(n[ok].astype(float)), I[ok], np.log10(c[ok].astype(float))
    ux = np.unique(x)
    return ux, np.array([y[x == u].mean() for u in ux]), np.array([t[x == u].mean() for u in ux])


def visit_series(run: Run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured checkpoints, n(s) = min_a N[s, a] and I_s(s) at each: (K,), (K, S), (K, S)."""
    cs = np.array(sorted(run.measure_dirs))
    n = np.array([min_action_visits(run.counts[f"N_{c}"]) for c in cs])
    I = np.array([run.measure(int(c))[1]["real_I_s"] for c in cs])
    return cs, n, I


def equal_visits_pair(exp: Run, ref: Run, n_grid: int = N_GRID_EQUAL_VISITS) -> dict:
    """D3: per state of the controllable zones, I_s interpolated linearly in log10 n between the measured checkpoints, in
    each run; on n_grid log-spaced visit counts within the range both runs cover, d(s, n) = I_env(s, n) - I_ref(s, n)
    and dtau(s, n) = log10 c_env(s, n) - log10 c_ref(s, n), the training time (checkpoint, interpolated the same way)
    at which each run reached n. Per zone, least squares d = b0 + b1 dtau + b2 (log10 n - mean): b0 is b at equal visits
    and equal training time. The points of one state are not independent: no standard error here, the seeds are the
    replicates."""
    ce, ne, Ie = visit_series(exp)
    cr, nr, Ir = visit_series(ref)
    out = {}
    for zone in CONTROLLABLE_ZONES:
        rows, excluded = [], 0
        for s in np.flatnonzero(zone_mask(exp, zone)):
            xe, ye, te = _curve(ne[:, s], Ie[:, s], ce)
            xr, yr, tr = _curve(nr[:, s], Ir[:, s], cr)
            if xe.size < 2 or xr.size < 2 or not min(xe[-1], xr[-1]) > max(xe[0], xr[0]):
                excluded += 1
                continue
            g = np.linspace(max(xe[0], xr[0]), min(xe[-1], xr[-1]), n_grid)
            d = np.interp(g, xe, ye) - np.interp(g, xr, yr)
            dtau = np.interp(g, xe, te) - np.interp(g, xr, tr)
            rows += [(s, gi, di, ti) for gi, di, ti in zip(g, d, dtau)]
        z = {"n_states": len({r[0] for r in rows}), "n_states_excluded": excluded, "n_points": len(rows),
             "b0": float("nan"), "b1_dtau": float("nan"), "b2_logn": float("nan"), "mean_d": float("nan")}
        if len(rows) >= 4:
            arr = np.array(rows, dtype=float)
            X = np.column_stack([np.ones(len(arr)), arr[:, 3], arr[:, 1] - arr[:, 1].mean()])
            beta, *_ = np.linalg.lstsq(X, arr[:, 2], rcond=None)
            z.update(b0=float(beta[0]), b1_dtau=float(beta[1]), b2_logn=float(beta[2]), mean_d=float(arr[:, 2].mean()),
                     log10_n_range=[float(arr[:, 1].min()), float(arr[:, 1].max())])
        out[zone] = z
    return out


def distance_to_uncontrollable(exp: Run) -> np.ndarray:
    """delta(s): torus (4-neighbour, Manhattan) distance from s to the nearest state of an uncontrollable zone of the
    experimental environment (its own true kernel); the same geometry serves the reference's states."""
    from helplessness.measures.rsa import torus_distances

    pos = np.array([divmod(s, GRID_SHAPE[1]) for s in range(GRID_SHAPE[0] * GRID_SHAPE[1])])
    d = torus_distances(pos, GRID_SHAPE)
    unc = ~exp.truth["controllable"]
    unc[goal_state()] = False  # the goal is absorbing (I_true 0), not an uncontrollable zone
    return d[:, unc].min(axis=1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def distance_profile_pair(exp: Run, ref: Run, c: int, delta: np.ndarray) -> dict:
    """D5 at checkpoint c: per controllable zone, Spearman rho of |b(s)| = |I_env(s) - I_ref(s)| with delta(s), and of
    each run's |I_s(s) - I_true(s)| with delta(s); mean |b| by distance."""
    Ie, Ir = exp.measure(c)[1]["real_I_s"], ref.measure(c)[1]["real_I_s"]
    out = {}
    for zone in CONTROLLABLE_ZONES:
        m = zone_mask(exp, zone)
        b = np.abs(Ie[m] - Ir[m])
        e_env, e_ref = np.abs(Ie[m] - exp.truth["I_true"][m]), np.abs(Ir[m] - ref.truth["I_true"][m])
        dm = delta[m]
        out[zone] = {"rho_abs_b": _spearman(dm, b), "rho_env_error": _spearman(dm, e_env), "rho_ref_error": _spearman(dm, e_ref),
                     "mean_abs_b_by_distance": {int(k): float(b[dm == k].mean()) for k in np.unique(dm)},
                     "n_states_by_distance": {int(k): int((dm == k).sum()) for k in np.unique(dm)}}
    return out


def _bin_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}")


def generalisation_run(run: Run, c: int) -> dict:
    """D6 at checkpoint c, one run: per (s, a) pair, L1 of the world model's kernel (real contexts, pooled) and of the
    twin's (T_hat at c) against the run's true kernel, by n(s, a) = N[s, a] at c; per zone and visit bin, the medians and
    the median of the paired difference L1_WM - L1_twin; and over the pairs with n <= LOW_N, the median difference."""
    if run.twin is None or f"T_hat_{c}" not in run.twin:
        return {"available": False}
    P = run.P_true
    l1_wm = np.abs(run.measure(c)[1]["real_p_hat"] - P).sum(-1)
    l1_tw = np.abs(run.twin[f"T_hat_{c}"].astype(float) - P).sum(-1)
    n = run.counts[f"N_{c}"].sum(-1)
    out = {"available": True, "zones": {}}
    for zone in ZONES:
        m = zone_mask(run, zone)[:, None] & np.isfinite(l1_wm)
        z = {"bins": {}}
        for lo, hi in L1_VISIT_BINS:
            sel = m & (n >= lo) & ((n <= hi) if hi is not None else True)
            k = _bin_label(lo, hi)
            z["bins"][k] = {"n_pairs": int(sel.sum()),
                            **({"l1_wm": float(np.median(l1_wm[sel])), "l1_twin": float(np.median(l1_tw[sel])),
                                "diff": float(np.median(l1_wm[sel] - l1_tw[sel]))} if sel.any() else {})}
        low = m & (n <= LOW_N)
        z["low_n"] = {"n_pairs": int(low.sum()), "diff": float(np.median(l1_wm[low] - l1_tw[low])) if low.sum() >= LOW_N_MIN_PAIRS else float("nan")}
        out["zones"][zone] = z
    return out


def generalisation_verdict(diffs: list[float]) -> str:
    """D6's rule over the seeds' median differences at low n (one condition, one zone, one checkpoint)."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0 or not np.isfinite(d).all():
        return "not testable (fewer than 5 pairs with n <= 10 in some run)"
    if (d < 0).all():
        return "generalises (L1_WM < L1_twin at low n in every seed)"
    if (d >= 0).all():
        return "does not generalise (L1_WM >= L1_twin at low n in every seed)"
    return "mixed"


def seeds_ci(values: list[float]) -> dict:
    """Mean over the seeds and its 95 % Student-t interval; with the bound of a null result (the largest |end|)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")], "bound": float("nan"), "values": []}
    ci = aggregate_b.mean_ci95(v)
    lo, hi = ci["ci95"]
    return {**ci, "bound": float(max(abs(lo), abs(hi))) if np.isfinite(lo) else float("nan"), "values": v.tolist(),
            "excludes_0": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}


def distance_shape_verdict(rho_b: dict, rho_env: dict, rho_ref: dict) -> str:
    """D5's rule at one checkpoint, from the seeds' summaries (seeds_ci) of the three Spearman rhos."""
    if not all(np.isfinite(x["ci95"][0]) for x in (rho_b, rho_env, rho_ref)):
        return "undetermined (fewer than 2 seeds with a defined rho)"
    if rho_b["ci95"][1] < 0 and rho_env["ci95"][1] < 0 and rho_ref["ci95"][0] <= 0 <= rho_ref["ci95"][1]:
        return "propagation shape present"
    if rho_b["ci95"][0] <= 0 <= rho_b["ci95"][1]:
        return "absent (rho(abs b, distance) compatible with 0)"
    return "undetermined"


def propagation(pairs: list[tuple[Run, Run]]) -> dict:
    """D2 to D6 over the campaign's pairs: per pair, then over the seeds at the checkpoints measured in every pair."""
    per_pair, common = [], None
    for exp, ref in pairs:
        cs = sorted(set(exp.measure_dirs) & set(ref.measure_dirs))
        common = set(cs) if common is None else common & set(cs)
        delta = distance_to_uncontrollable(exp)
        per_pair.append({"seed": exp.seed, "zone_class": exp.zone_class, "checkpoints": cs,
                         "equal_visits": equal_visits_pair(exp, ref),
                         "distance": {c: distance_profile_pair(exp, ref, c, delta) for c in cs},
                         "generalisation": {c: {"experimental": generalisation_run(exp, c), "reference": generalisation_run(ref, c)} for c in cs},
                         "b_by_state": {c: (exp.measure(c)[1]["real_I_s"] - ref.measure(c)[1]["real_I_s"]).tolist() for c in cs},
                         "distance_to_uncontrollable": delta.tolist()})
    common = sorted(common or [])
    out = {"pairs": per_pair, "common_checkpoints": common, "seeds": {}}
    s = out["seeds"]
    s["equal_visits_b0"] = {z: seeds_ci([p["equal_visits"][z]["b0"] for p in per_pair]) for z in CONTROLLABLE_ZONES}
    s["equal_visits_b1_dtau"] = {z: seeds_ci([p["equal_visits"][z]["b1_dtau"] for p in per_pair]) for z in CONTROLLABLE_ZONES}
    s["distance"] = {}
    for c in common:
        s["distance"][c] = {}
        for z in CONTROLLABLE_ZONES:
            rb, re_, rr = (seeds_ci([p["distance"][c][z][k] for p in per_pair]) for k in ("rho_abs_b", "rho_env_error", "rho_ref_error"))
            s["distance"][c][z] = {"rho_abs_b": rb, "rho_env_error": re_, "rho_ref_error": rr, "verdict": distance_shape_verdict(rb, re_, rr)}
    s["generalisation"] = {}
    for c in common:
        s["generalisation"][c] = {}
        for cond in ("experimental", "reference"):
            g = [p["generalisation"][c][cond] for p in per_pair]
            if not all(x["available"] for x in g):
                s["generalisation"][c][cond] = {z: "twin kernels not available" for z in ZONES}
                continue
            s["generalisation"][c][cond] = {z: {"verdict": generalisation_verdict([x["zones"][z]["low_n"]["diff"] for x in g]),
                                                "diffs": [x["zones"][z]["low_n"]["diff"] for x in g],
                                                "n_pairs": [x["zones"][z]["low_n"]["n_pairs"] for x in g]} for z in ZONES}
    return out


def seeds_over_checkpoints(secondary: list[dict]) -> dict:
    """D2 and D4 over the seeds at the checkpoints measured in every pair: b by zone (I_s; the corridor from the
    per-state values) and alpha_env - alpha_ref, mean and 95 % interval; the expectation's reading at the budget."""
    common = sorted(set.intersection(*[{c["env_step"] for c in p["checkpoints"]} for p in secondary])) if secondary else []
    out = {"common_checkpoints": common, "b_I_s": {}, "alpha_difference": {}}
    for c in common:
        rows = [next(x for x in p["checkpoints"] if x["env_step"] == c) for p in secondary]
        out["b_I_s"][c] = {z: seeds_ci([r["b_I_s_all"][z] for r in rows]) for z in ZONES}
        out["alpha_difference"][c] = {z: seeds_ci([r["alpha"][z]["difference"] for r in rows]) for z in ZONES}
    if BUDGET_ENV_STEPS in common:
        exp = {}
        for z in CONTROLLABLE_ZONES:
            b, a = out["b_I_s"][BUDGET_ENV_STEPS][z], out["alpha_difference"][BUDGET_ENV_STEPS].get(z)
            exp[z] = {"b": "compatible with 0" if not b["excludes_0"] else "NOT compatible with 0", "b_bound": b["bound"],
                      "alpha_difference": ("not defined" if a is None else "compatible with 0" if not a["excludes_0"] else "NOT compatible with 0"),
                      "alpha_bound": None if a is None else a["bound"],
                      "early_checkpoints_excluding_0": [c for c in common if c < BUDGET_ENV_STEPS and out["b_I_s"][c][z]["excludes_0"]]}
        out["expectation_at_budget"] = exp
    return out


def analyse(paths: list[Path], measures_root: Path | None = None, e30_control: dict | None = None,
            pairs_out: list | None = None) -> dict:
    """Main, secondary and propagation analyses; pairs_out, if a list, receives the loaded (experimental, reference)
    Run pairs (for the report's other stages)."""
    runs = [Run(p, measures_root) for p in paths]
    pair_runs: list[tuple[Run, Run]] = []
    pairs: dict[tuple, dict] = {}
    for r in runs:
        slot = pairs.setdefault((r.config, r.seed), {})
        if r.condition in slot:
            raise ValueError(f"two {r.condition} runs for {r.config} seed {r.seed}")
        slot[r.condition] = r
    main_runs = [aggregate_b.load_run(r.path) for r in runs if (r.path / "arrays.npz").exists()]
    main = aggregate_b.aggregate(main_runs, e30_control, {"n_seeds": N_SEEDS, "fraction": FRACTION, "reference_effect": REFERENCE_EFFECT})
    out = {"main": {"budget_env_steps": BUDGET_ENV_STEPS, "aggregate_b": main,
                    "final_env_steps": sorted({int(r.results["env_steps"]) for r in runs})},
           "secondary": [], "unpaired": []}
    for (config, seed), slot in sorted(pairs.items()):
        if set(slot) != {"experimental", "reference"}:
            out["unpaired"] += [str(r.path) for r in slot.values()]
            continue
        exp, ref = slot["experimental"], slot["reference"]
        common = sorted(set(exp.measure_dirs) & set(ref.measure_dirs))
        out["secondary"].append({"config": config, "seed": seed, "zone_class": exp.zone_class,
                                 "checkpoints": [checkpoint_pair(exp, ref, c) for c in common]})
        pair_runs.append((exp, ref))
    if pairs_out is not None:
        pairs_out.extend(pair_runs)
    out["propagation"] = propagation(pair_runs) if pair_runs else None
    out["over_seeds"] = seeds_over_checkpoints(out["secondary"])
    return out



def _f(x) -> str:
    return "NaN" if x is None or not np.isfinite(x) else f"{x:+.4f}"


def markdown(res: dict) -> str:
    cats = list(CATEGORIES)
    head = " | ".join(cats)
    lines = ["# E3 campaign: analyses (docs/e3/DECISIONS.md, 2026-10-07 (29))", "",
             f"## Main: b by zone at the budget ({BUDGET_ENV_STEPS:,} environment steps), campaign rule and aberration control", "",
             f"Final environment steps of the runs read: {res['main']['final_env_steps']}.", "",
             aggregate_b.markdown(res["main"]["aggregate_b"]), "## Secondary analyses, by pair and checkpoint", ""]
    for p in res["secondary"]:
        lines += [f"### {p['config']}, seed {p['seed']} (class {p['zone_class']})", ""]
        lines += [f"| checkpoint | quantity | {head} |", "|---|---|" + "---|" * len(cats)]
        for c in p["checkpoints"]:
            e = c["env_step"]
            lines.append(f"| {e} | b (I_s) | " + " | ".join(_f(c["b_I_s"][k]) for k in cats) + " |")
            lines.append(f"| {e} | b (I_s projected) | " + " | ".join(_f(c["b_I_s_projected"][k]) for k in cats) + " |")
            lines.append(f"| {e} | difference of differences (descriptive) | " + " | ".join(_f(c["difference_of_differences"][k]["did"]) for k in cats) + " |")
            lines.append(f"| {e} | its parts: b_twin; WM - true env / ref | " + " | ".join(
                f"{_f(c['decomposition'][k]['b_twin'])}; {_f(c['decomposition'][k]['wm_minus_true_env'])} / {_f(c['decomposition'][k]['wm_minus_true_ref'])}"
                for k in cats) + " |")
            lines.append(f"| {e} | alpha env / ref | " + " | ".join(f"{_f(c['alpha'][k]['env'])} / {_f(c['alpha'][k]['ref'])}" for k in cats) + " |")
        lines += ["", "b by visit class (n(s) = visits of the state's least-visited action; b [states env / ref]):", "",
                  f"| checkpoint | class | {head} |", "|---|---|" + "---|" * len(cats)]
        for c in p["checkpoints"]:
            for lo, hi in VISIT_CLASSES:
                k = class_label(lo, hi)
                cells = [c["b_by_visit_class"][cat][k] for cat in cats]
                if all(x["n_states_env"] == 0 and x["n_states_ref"] == 0 for x in cells):
                    continue
                lines.append(f"| {c['env_step']} | {k} | " + " | ".join(f"{_f(x['b'])} [{x['n_states_env']}/{x['n_states_ref']}]" for x in cells) + " |")
        lines += ["", "Zones' separation in the latent (held-out accuracy of the zone probe / controllability probe; "
                  "centroid distance controllable vs uncontrollable categories, in within-zone RMS units):", "",
                  "| checkpoint | run | latent | zone probe | controllability probe | centroid distance ratio |", "|---|---|---|---|---|---|"]
        for c in p["checkpoints"]:
            for role, s in c["latent_zone_separation"].items():
                if s is None:
                    lines.append(f"| {c['env_step']} | {role} | not available | - | - | - |")
                    continue
                lines.append(f"| {c['env_step']} | {role} | {s['latent']} | {s['probe_zone']['test_accuracy']:.3f} | "
                             f"{s['probe_controllability']['test_accuracy']:.3f} | {s['centroid_distance_ratio']:.3f} |")
        lines.append("")
    if res["unpaired"]:
        lines.append("Runs without their pair (left out): " + ", ".join(f"`{Path(u).name}`" for u in res["unpaired"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--measures-root", type=Path, default=None, help="checkpoint measures under DIR/<run name>/env<NNNNNN>/")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    res = analyse(args.runs, args.measures_root, aggregate_b.load_e30_control())
    text = markdown(res)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "CAMPAIGN_ANALYSIS.md").write_text(text, encoding="utf-8")
        (args.out / "campaign_analysis.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
