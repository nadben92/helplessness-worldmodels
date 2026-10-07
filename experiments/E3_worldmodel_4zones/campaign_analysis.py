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
import sys
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


def zone_mean(values: np.ndarray, zone: np.ndarray) -> dict:
    return {cat: float(np.nanmean(values[zone == cat])) if np.isfinite(values[zone == cat]).any() else float("nan") for cat in CATEGORIES}


def separation_from_arrays(arrays, zone_of_state: list[str]) -> dict | None:
    """The zones' separation on the measure's per-context posteriors (z only), for runs without the in-run record."""
    if "real_post_probs" not in arrays.files:
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

    def measure(self, c: int) -> tuple[dict, object]:
        d = self.measure_dirs[c]
        return json.loads((d / "results.json").read_text(encoding="utf-8")), np.load(d / "arrays.npz")


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
    a_e = zone_mean(action_independent_weight(arr_e["real_p_hat"], arr_e["P_true"]), zone)
    a_r = zone_mean(action_independent_weight(arr_r["real_p_hat"], arr_r["P_true"]), zone)
    out["alpha"] = {cat: {"env": a_e[cat], "ref": a_r[cat], "difference": a_e[cat] - a_r[cat]} for cat in CATEGORIES}
    sep = {}
    for role, run, arr, res in (("experimental", exp, arr_e, res_e), ("reference", ref, arr_r, res_r)):
        g = run.records[c].get("goal_diagnostics", {})
        sep[role] = ({**g["latent_zone_separation"], "latent": "h and z (recorded in the run)"} if "latent_zone_separation" in g
                     else separation_from_arrays(arr, res["zone_of_state"]))
    out["latent_zone_separation"] = sep
    out["agent"] = {role: {k: run.records[c]["goal_diagnostics"][k] for k in ("V_by_zone", "goal_neighbours")
                           if k in run.records[c].get("goal_diagnostics", {})} for role, run in (("experimental", exp), ("reference", ref))}
    return out


def analyse(paths: list[Path], measures_root: Path | None = None, e30_control: dict | None = None) -> dict:
    runs = [Run(p, measures_root) for p in paths]
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
            lines.append(f"| {e} | difference of differences | " + " | ".join(_f(c["difference_of_differences"][k]["did"]) for k in cats) + " |")
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
