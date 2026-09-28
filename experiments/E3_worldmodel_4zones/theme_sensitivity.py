"""How much the separation of the probabilities phase depends on the colour
draw (docs/e3/DECISIONS.md, 2026-09-25) -- a characterisation, not a test
of the suite. Report: docs/e3/THEME_SENSITIVITY.md.

Design, fixed before the runs:
- the autoencoder isolation test (ae_isolation.py), lr 1e-3, batch 32, the
  current procedure (probabilities until the label-free JS criterion, then
  straight-through), k = 6;
- ONLY the colours vary: the themes of theme draw i are those of seed i's
  theme stream (rng_streams.py), i = 0..9. Everything else is seed 0's
  (zone layout, network initialisation, minibatches, switch statistic), so
  a difference between draws is a colour effect;
- 10,000 steps, switch deadline 8,000 (80%); a draw that does not switch by
  then is CENSORED: reported as "> 8,000", never dropped and never counted
  as 8,000; final hard-sample template accuracy at step 10,000 (mean of 5
  draws);
- colour statistics per draw, in CIELAB: agent marker to zone themes
  (min, mean), between zone themes (min, mean), marker to the corridor and
  goal colours (min), zone-theme lightness L* (mean, min, max);
- Spearman correlation of the switch step with each statistic (n = 10,
  descriptive), censored draws ranked as the slowest (maximal rank, tied
  among themselves); the report gives their number.

Intrinsic variability (added before its runs, after the 10 draws):
- 2 colour draws: draw 0, and the slowest draw that switched -- or the
  first censored draw if there is one;
- 3 replicates each (replicates 1, 2, 3): same colours, layout and network
  initialisation, different seeds for the latent draws and the minibatches
  (ae_isolation.run(replicate=r));
- compared: the spread of the switch step between replicates at fixed
  colours, against its spread between the 10 colour draws. Criterion,
  fixed before the replicate runs: if the pooled within-draw standard
  deviation (the 3 replicates and the original run of each replicated draw)
  is at least HALF the between-draw standard deviation (the 10 draws), the
  two are "of the same order" and the report states that the Spearman
  correlations are not interpretable with this design. A censored run makes
  a standard deviation undefined; the report then says so and compares
  ranges instead.

    python experiments/E3_worldmodel_4zones/theme_sensitivity.py --draws 0 1 2 3 4 --threads 2
    python experiments/E3_worldmodel_4zones/theme_sensitivity.py --report
Raw results: results/E3_worldmodel_4zones/theme_sensitivity/draw_NN.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from helplessness.envs.appearance.cielab import lab_distance, srgb_to_lab
from helplessness.envs.appearance.theme_sampling import build_themes
from helplessness.experiment_loader import load_experiment_module

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "E3_worldmodel_4zones" / "theme_sensitivity"
REPORT = REPO_ROOT / "docs" / "e3" / "THEME_SENSITIVITY.md"
FIGURE = REPO_ROOT / "docs" / "e3" / "theme_sensitivity_figures" / "switch_vs_colour.png"

TRAIN_SEED = 0
DRAWS = tuple(range(10))
N_STEPS, DEADLINE, LR, BATCH = 10_000, 8_000, 1e-3, 32
STATS = ("marker_theme_min", "marker_theme_mean", "theme_theme_min", "theme_theme_mean", "marker_constant_min",
         "theme_L_mean", "theme_L_min", "theme_L_max")


def themes_of_draw(draw: int):
    run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "theme_sens_run_mini")
    rng_streams = load_experiment_module("E3_worldmodel_4zones", "rng_streams.py", "theme_sens_rng")
    cfg = run_mini.load_mini_config()
    return build_themes(run_mini.e3_env.ZONE_CATEGORIES, goal_state=run_mini.e3_env.goal_state(),
                        rng=rng_streams.spawn_streams(draw)["theme"], min_dist=cfg["render"]["cielab_min_dist"],
                        marker_min_dist=cfg["render"]["cielab_marker_min_dist"])


def colour_stats(themes) -> dict[str, float]:
    lab = lambda rgb: srgb_to_lab(np.asarray(rgb, dtype=np.float64) / 255.0)  # noqa: E731  (sRGB in [0, 1], as theme_sampling._lab)
    zones = [lab(c) for c in themes.zone_colors.values()]
    marker = lab(themes.agent_marker_color)
    m_t = [lab_distance(marker, z) for z in zones]
    t_t = [lab_distance(a, b) for a, b in itertools.combinations(zones, 2)]
    return {
        "marker_theme_min": float(min(m_t)), "marker_theme_mean": float(np.mean(m_t)),
        "theme_theme_min": float(min(t_t)), "theme_theme_mean": float(np.mean(t_t)),
        "marker_constant_min": float(min(lab_distance(marker, lab(themes.corridor_color)), lab_distance(marker, lab(themes.goal_color)))),
        "theme_L_mean": float(np.mean([z[0] for z in zones])), "theme_L_min": float(min(z[0] for z in zones)),
        "theme_L_max": float(max(z[0] for z in zones)),
    }


def _already_done(path: Path) -> bool:
    """A finished result file: present and a complete JSON record (a run
    interrupted before its write leaves no file; a partial one is redone)."""
    try:
        return "switch_step" in json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False


def run_draws(draws: tuple[int, ...], threads: int) -> None:
    ae = load_experiment_module("E3_worldmodel_4zones", "ae_isolation.py", "theme_sens_ae")
    torch.set_num_threads(threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for draw in draws:
        if _already_done(RESULTS_DIR / f"draw_{draw:02d}.json"):
            print(f"draw {draw}: already done, skipped", flush=True)
            continue
        themes = themes_of_draw(draw)
        result = ae.run(N_STEPS, LR, BATCH, TRAIN_SEED, "warmup_probs_until_separated", k=6, switch_max_steps=DEADLINE,
                        raise_on_deadline=False, themes=themes)
        record = {
            "draw": draw, "train_seed": TRAIN_SEED, "status": result["status"], "switch_step": result["switch_step"],
            "final_acc_hard": result["final"]["acc_hard"] if result["status"] == "ok" else None,
            "final_acc_mode": result["final"]["acc_mode"] if result["status"] == "ok" else None,
            "themes": {"zone_colors": {k: list(v) for k, v in themes.zone_colors.items()},
                       "agent_marker_color": list(themes.agent_marker_color)},
            "colour_stats": colour_stats(themes), "curve": result["curve"], "num_threads": torch.get_num_threads(),
        }
        (RESULTS_DIR / f"draw_{draw:02d}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"draw {draw}: {record['status']} switch {record['switch_step']} final acc {record['final_acc_hard']}", flush=True)


STAT_GROUPS = (
    ("agent-theme contrast", ("marker_theme_min", "marker_theme_mean", "marker_constant_min")),
    ("distances between themes", ("theme_theme_min", "theme_theme_mean")),
    ("theme lightness", ("theme_L_mean", "theme_L_min", "theme_L_max")),
)

# mini.yaml's deadline rule (docs/e3/DECISIONS.md, 2026-09-25, fixed before the results).
MINI_BUDGET_GRAD_STEPS = (5000 + 256) * 2
MINI_DEADLINE_CAP = int(0.8 * MINI_BUDGET_GRAD_STEPS)  # 8,409
DEADLINE_MARGIN = 1.25
# RSSM / autoencoder pairs, seed 0: (label, RSSM switch step, autoencoder switch step). None = from mini_long.
FACTOR_PAIRS = (("seed 0, old themes (mini run / estimator comparison)", 4801, 2301),
                ("seed 0, new themes (mini_long / isolation test)", None, 4901))


def mini_long_switch_step() -> tuple[int | None, str]:
    """mini_long's effective switch step, seed 0, from its latest run
    directory: meta.yaml once the seed is done, else the ">>> SWITCH" line
    of console.log while it runs."""
    import re

    import yaml

    runs = sorted((REPO_ROOT / "results" / "E3_worldmodel_4zones").glob("grid12x12__*worldmodel-mini_long__seed00__*"))
    for run in reversed(runs):
        meta_path, log_path = run / "meta.yaml", run / "console.log"
        if meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            if meta.get("switch_step") is not None:
                return int(meta["switch_step"]), run.name
        if log_path.exists():
            match = re.search(r"SWITCH to straight-through effective at step (\d+)", log_path.read_text(encoding="utf-8"))
            if match:
                return int(match.group(1)), run.name
    return None, "no mini_long run with a switch yet"


CONCLUSION = [
    "", "## Conclusion (criterion fixed in advance; recorded as decided by the user, 2026-09-28)", "",
    "- The variability between replicates (same colours, other latent draws and minibatches) is of the same order as the "
    "variability between colour draws. The Spearman correlations with the colour statistics are therefore **not interpretable** "
    "with this design, and the switch step depends mostly on the randomness of training.",
    "- The RSSM / autoencoder factor is **not stable**: 2.09 (seed 0, old themes: 4,801 / 2,301) and 0.85 (seed 0, new themes: "
    "mini_long 4,151 / 4,901).",
    "- **1 run out of 16** (draw 1, replicate 3) did not separate before 8,000 steps: the start-up procedure is **not fully "
    "robust**.",
    "- No further investigation for E3. The switch step stays a covariate of every analysis (DECISIONS.md, 2026-09-25), and "
    "every GPU run has a large absolute deadline; a run that does not separate is reported as a failure.",
]


def deadline_section(steps: list[float | None], n_censored: int) -> list[str]:
    """RSSM / autoencoder factor; mini.yaml's deadline rule is closed without
    object (mini.yaml retired, docs/e3/DECISIONS.md 2026-09-28)."""
    rssm_long, source = mini_long_switch_step()
    lines = ["", "## RSSM / autoencoder factor and mini.yaml's switch deadline", "",
             "Rule fixed before these results (docs/e3/DECISIONS.md, 2026-09-25).", "",
             "| pair | RSSM switch step | autoencoder switch step | factor |", "|---|---|---|---|"]
    factors = []
    for label, rssm, ae in FACTOR_PAIRS:
        rssm = rssm if rssm is not None else rssm_long
        if rssm is None:
            lines.append(f"| {label} | not available yet ({source}) | {ae} | - |")
            continue
        factors.append(rssm / ae)
        lines.append(f"| {label} | {rssm}{' (' + source + ')' if label.startswith('seed 0, new') else ''} | {ae} | {rssm / ae:.2f} |")
    lines.append("")
    lines.append("**mini.yaml's deadline rule is closed without object**: mini.yaml is retired from current use and its budget is "
                 "not increased (docs/e3/DECISIONS.md, 2026-09-28). What the rule would have given is kept below for the record.")
    lines.append("")
    if n_censored:
        lines += [f"**{n_censored} draw(s) censored (> {DEADLINE:,}): rule 2 applies.** mini.yaml's deadline is set to the cap, "
                  f"{MINI_DEADLINE_CAP:,} (80% of {MINI_BUDGET_GRAD_STEPS:,}), with no other adjustment. This is a robustness "
                  "problem of the start-up procedure, not a deadline setting; no remedy is proposed here (correlations above, "
                  "agent-theme contrast kept apart from the distances between themes)."]
        return lines
    observed_max = max(v for v in steps if v is not None)
    if not factors:
        lines.append(f"Rule 1 applies (no censored draw; maximum observed {observed_max:.0f}), but no RSSM / autoencoder factor "
                     "is available yet: the deadline is computed once mini_long has switched.")
        return lines
    factor = max(factors)
    deadline = observed_max * factor * DEADLINE_MARGIN
    lines.append(f"Rule 1 (no censored draw): {observed_max:.0f} (max observed) x {factor:.2f} (larger factor) x {DEADLINE_MARGIN} "
                 f"= **{deadline:.0f}** steps; cap {MINI_DEADLINE_CAP:,}.")
    if len(factors) < len(FACTOR_PAIRS):
        lines.append("Provisional: only one of the two factor pairs is available (mini_long has not switched yet).")
    if deadline > MINI_DEADLINE_CAP:
        lines.append(f"Above the cap: the rule would have increased mini's budget (to at least {int(np.ceil(deadline / 0.8)):,} "
                     "gradient steps); not applied (closed without object).")
    return lines


REPLICATES = (1, 2, 3)
SAME_ORDER_RATIO = 0.5


def replicated_draws() -> tuple[int, int]:
    """Draw 0 and the slowest draw that switched, or the first censored draw."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("draw_*.json"))]
    censored = [r["draw"] for r in records if r["switch_step"] is None]
    if censored:
        return 0, censored[0] if censored[0] != 0 else (censored[1] if len(censored) > 1 else max(
            (r for r in records if r["switch_step"] is not None), key=lambda r: r["switch_step"])["draw"])
    slowest = max((r for r in records if r["draw"] != 0), key=lambda r: r["switch_step"])["draw"]
    return 0, slowest


def run_replicates(draws: tuple[int, ...], threads: int) -> None:
    ae = load_experiment_module("E3_worldmodel_4zones", "ae_isolation.py", "theme_sens_ae_rep")
    torch.set_num_threads(threads)
    for draw in draws:
        themes = themes_of_draw(draw)
        for r in REPLICATES:
            if _already_done(RESULTS_DIR / f"replicate_d{draw:02d}_r{r}.json"):
                print(f"draw {draw} replicate {r}: already done, skipped", flush=True)
                continue
            result = ae.run(N_STEPS, LR, BATCH, TRAIN_SEED, "warmup_probs_until_separated", k=6, switch_max_steps=DEADLINE,
                            raise_on_deadline=False, themes=themes, replicate=r)
            record = {"draw": draw, "replicate": r, "train_seed": TRAIN_SEED, "status": result["status"],
                      "switch_step": result["switch_step"],
                      "final_acc_hard": result["final"]["acc_hard"] if result["status"] == "ok" else None,
                      "num_threads": torch.get_num_threads()}
            (RESULTS_DIR / f"replicate_d{draw:02d}_r{r}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"draw {draw} replicate {r}: {record['status']} switch {record['switch_step']}", flush=True)


def variability_section(records: list[dict]) -> list[str]:
    reps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("replicate_d*_r*.json"))]
    lines = ["", "## Intrinsic variability: replicates at fixed colours", ""]
    if not reps:
        return lines + ["Replicates not run yet: the correlations above cannot be judged against the run-to-run spread."]
    fmt = lambda v: f"> {DEADLINE:,}" if v is None else f"{v:.0f}"  # noqa: E731
    by_draw: dict[int, list] = {}
    for r in records:
        by_draw.setdefault(r["draw"], []).append(r["switch_step"])  # the original run (historical streams)
    lines += ["Same colours, layout and initialisation; different seeds for the latent draws and the minibatches. "
              "\"original\" is the draw's run above (historical streams).", "",
              "| draw | original | " + " | ".join(f"replicate {r}" for r in REPLICATES) + " | range | sd |",
              "|---|---|" + "---|" * len(REPLICATES) + "---|---|"]
    within_sds, any_censored = [], False
    for draw in sorted({r["draw"] for r in reps}):
        vals = by_draw.get(draw, [None])[:1] + [next((x["switch_step"] for x in reps if x["draw"] == draw and x["replicate"] == rr),
                                                    "n/a") for rr in REPLICATES]
        observed = [v for v in vals if isinstance(v, (int, float))]
        censored = any(v is None for v in vals)
        any_censored |= censored
        rng_txt = f"{min(observed):.0f} - {fmt(None) if censored else f'{max(observed):.0f}'}" if observed else "-"
        sd = None if censored or len(observed) < 2 else float(np.std(observed, ddof=1))
        if sd is not None:
            within_sds.append(sd)
        lines.append(f"| {draw} | " + " | ".join(fmt(v) if v != "n/a" else "n/a" for v in vals) + f" | {rng_txt} | "
                     + ("undefined (censored)" if censored else f"{sd:.0f}") + " |")
    between_obs = [r["switch_step"] for r in records if r["switch_step"] is not None]
    between_censored = len(between_obs) < len(records)
    lines.append("")
    if any_censored or between_censored or not within_sds:
        lines += [f"A censored run makes a standard deviation undefined, so the pre-declared ratio cannot be computed. "
                  f"Between colour draws: range {min(between_obs):.0f} - "
                  f"{fmt(None) if between_censored else f'{max(between_obs):.0f}'}; compare with the ranges above."]
        return lines
    pooled = float(np.sqrt(np.mean(np.square(within_sds))))
    between = float(np.std(between_obs, ddof=1))
    ratio = pooled / between
    lines.append(f"Pooled within-draw sd **{pooled:.0f}** steps; between-draw sd ({len(between_obs)} colour draws) **{between:.0f}** steps; "
                 f"ratio **{ratio:.2f}** (criterion fixed before the replicates: same order if >= {SAME_ORDER_RATIO}).")
    lines.append("")
    if ratio >= SAME_ORDER_RATIO:
        lines.append("**The run-to-run variability at fixed colours is of the same order as the variability between colour "
                     "draws: the Spearman correlations above are not interpretable with this design.**")
    else:
        lines.append("The run-to-run variability at fixed colours is smaller than the variability between colour draws "
                     "(ratio below the pre-declared threshold): the colour effect exceeds the run-to-run noise. The "
                     "correlations remain descriptive (10 draws, no correction).")
    return lines


def _censored_quantile(values: list[float | None], q: float) -> str:
    """Empirical quantile of switch steps where None = censored (> DEADLINE),
    ordered after every observed value: "> 8,000" when the quantile falls
    among the censored draws."""
    ordered = sorted(values, key=lambda v: (v is None, v if v is not None else 0.0))
    v = ordered[int(np.ceil(q * len(ordered))) - 1] if q > 0 else ordered[0]
    return f"> {DEADLINE:,}" if v is None else f"{v:.0f}"


def write_report() -> None:
    from scipy.stats import rankdata, spearmanr

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("draw_*.json"))]
    steps = [r["switch_step"] for r in records]
    n_censored = sum(v is None for v in steps)
    # Ranking only: a censored draw is slower than every observed one (maximal rank, tied among censored draws).
    switch_rank = rankdata([v if v is not None else np.inf for v in steps])
    lines = [
        "# E3 — theme sensitivity of the probabilities phase", "",
        "How long the probabilities phase takes to separate the 144 states depends on the colour draw. Written for "
        "whoever analyses E3/E4/E5 runs: the switch step is a covariate (docs/e3/DECISIONS.md, 2026-09-25).", "",
        "**Design (fixed before the runs, `experiments/E3_worldmodel_4zones/theme_sensitivity.py`).** Autoencoder isolation "
        f"test, lr {LR:g}, batch {BATCH}, current procedure, k = 6, {N_STEPS:,} steps, switch deadline {DEADLINE:,}. Only the "
        f"colours vary (theme stream of seeds {list(DRAWS)}); layout, initialisation, minibatches and switch statistic are "
        f"seed {TRAIN_SEED}'s. Final accuracy: hard sample, step {N_STEPS:,}. A draw that does not switch before "
        f"{DEADLINE:,} steps is **censored**: reported as \"> {DEADLINE:,}\", never dropped and never counted as {DEADLINE:,}.", "",
        f"![switch step vs colour statistics](theme_sensitivity_figures/{FIGURE.name})", "",
        "## Switch step", "",
        f"- Draws: {len(records)}; switched before {DEADLINE:,} steps: {len(records) - n_censored}; **censored (> {DEADLINE:,}): "
        f"{n_censored}**.",
        f"- Distribution over all {len(records)} draws (censored draws ordered after every observed one): min "
        f"{_censored_quantile(steps, 0)}, lower quartile {_censored_quantile(steps, 0.25)}, median {_censored_quantile(steps, 0.5)}, "
        f"upper quartile {_censored_quantile(steps, 0.75)}, max {_censored_quantile(steps, 1.0)}.",
        "", "| draw | switch step | final hard-sample acc. | " + " | ".join(STATS) + " |",
        "|---|---|---|" + "---|" * len(STATS),
    ]
    for r in records:
        sw = f"> {DEADLINE:,} (censored)" if r["switch_step"] is None else str(r["switch_step"])
        acc = "-" if r["final_acc_hard"] is None else f"{r['final_acc_hard']:.3f}"
        lines.append(f"| {r['draw']} | {sw} | {acc} | " + " | ".join(f"{r['colour_stats'][k]:.1f}" for k in STATS) + " |")
    lines += ["", "## Spearman correlation of the switch step with the colour statistics", "",
              f"n = {len(records)}, of which {n_censored} censored, ranked as the slowest draws (maximal rank, tied among "
              f"themselves). Descriptive only: {len(records)} draws, {len(STATS)} statistics, no correction for multiple comparisons.", "",
              "| group | statistic | rho | p |", "|---|---|---|---|"]
    rows = []
    for group, keys in STAT_GROUPS:
        for k in keys:
            x = np.array([r["colour_stats"][k] for r in records])
            rho, p = spearmanr(x, switch_rank) if len(records) > 2 else (float("nan"), float("nan"))
            rows.append((k, rho, p))
            lines.append(f"| {group} | {k} | {rho:.2f} | {p:.3f} |")
    lines += variability_section(records)
    lines += deadline_section(steps, n_censored)
    lines += CONCLUSION
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    top = DEADLINE * 1.08  # censored draws drawn above the deadline line, as "> deadline"
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (k, rho, _) in zip(axes.flat, rows):
        for r in records:
            x, censored = r["colour_stats"][k], r["switch_step"] is None
            y = top if censored else r["switch_step"]
            ax.scatter([x], [y], c="tab:red" if censored else "tab:blue", marker="^" if censored else "o")
            ax.annotate(str(r["draw"]), (x, y), fontsize=7)
        ax.axhline(DEADLINE, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"{k} (Spearman rho {rho:.2f})", fontsize=9)
        ax.set_ylabel(f"switch step (red triangles: censored, > {DEADLINE:,})")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=100)
    plt.close(fig)
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, nargs="+")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--replicates", type=str, nargs="+", default=None,
                        help="draw numbers, or 'auto' for draw 0 and the slowest (or first censored) draw")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        write_report()
    elif args.replicates:
        draws = replicated_draws() if args.replicates == ["auto"] else tuple(int(d) for d in args.replicates)
        print(f"replicating draws {draws}", flush=True)
        run_replicates(tuple(draws), args.threads)
    elif args.draws:
        run_draws(tuple(args.draws), args.threads)
    else:
        parser.error("pass --draws, --replicates or --report")


if __name__ == "__main__":
    main()
