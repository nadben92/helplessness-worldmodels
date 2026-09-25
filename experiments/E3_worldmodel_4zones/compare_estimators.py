"""Estimator comparison in the autoencoder isolation test
(docs/e3/DECISIONS.md, 2026-09-24): (a) straight_through, (b) gumbel_st,
(c) warmup_probs -- no KL, k=6, 3 seeds each, pre-declared schedules
(ae_isolation.py's DEFAULT_* constants). Criterion fixed in advance:
hard-sample template accuracy >= 0.95 on all 3 seeds. Selection rule
fixed in advance: the simplest option that passes -- (a) if it passes;
otherwise, of (b) and (c), prefer (c) whenever it passes.

Run one estimator per process (they are independent), then aggregate:
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_estimators.py --estimator straight_through
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_estimators.py --estimator gumbel_st
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_estimators.py --estimator warmup_probs
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_estimators.py --report

Raw curves: results/E3_worldmodel_4zones/estimator_comparison/<estimator>_seedNN.json.
Report: docs/e3/ESTIMATOR_COMPARISON.md + docs/e3/estimator_comparison_figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from helplessness.experiment_loader import load_experiment_module

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "E3_worldmodel_4zones" / "estimator_comparison"
DOCS_DIR = REPO_ROOT / "docs" / "e3"
FIGURES_DIR = DOCS_DIR / "estimator_comparison_figures"

COMPARED = ("straight_through", "gumbel_st", "warmup_probs")
CRITERION_SWITCH = "warmup_probs_until_separated"  # the retained procedure, added after the selection
LABELS = {
    "straight_through": "(a) straight-through", "gumbel_st": "(b) Gumbel-softmax ST, tau 1 -> 0.1",
    "warmup_probs": "(c) probabilities warm-up, then ST",
    CRITERION_SWITCH: "(c') probabilities until the label-free JS criterion, then ST",
}
SEEDS = (0, 1, 2)
K = 6


def select_estimator(passes: dict[str, bool]) -> str | None:
    """The pre-declared rule: (a) if it passes (simplest: no schedule at
    all); otherwise (c) if it passes (preferred on a tie with (b));
    otherwise (b) if it passes; otherwise None."""
    for estimator in ("straight_through", "warmup_probs", "gumbel_st"):
        if passes.get(estimator):
            return estimator
    return None


def run_estimator(estimator: str, seeds: tuple[int, ...], threads: int) -> None:
    from helplessness.agents.worldmodel.loop import SeparationNotReachedError

    ae = load_experiment_module("E3_worldmodel_4zones", "ae_isolation.py", "compare_estimators_ae_isolation")
    torch.set_num_threads(threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        try:
            result = ae.run(ae.DEFAULT_N_STEPS, ae.DEFAULT_LR, ae.DEFAULT_BATCH, seed, estimator, k=K)
        except SeparationNotReachedError as err:
            result = {"estimator": estimator, "seed": seed, "k": K, "accuracy": 0.0, "status": "separation_not_reached",
                      "error": str(err), "final": {"acc_hard": 0.0, "acc_mode": 0.0, "posterior_entropy": float("nan"),
                                                   "mean_js": float("nan")}, "curve": []}
        path = RESULTS_DIR / f"{estimator}_seed{seed:02d}.json"
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        f = result["final"]
        print(f"{estimator} seed={seed}: acc_hard={f['acc_hard']:.4f} acc_mode={f['acc_mode']:.4f} "
              f"entropy={f['posterior_entropy']:.2f} js={f['mean_js']:.3f}", flush=True)


def _load_all() -> dict[str, list[dict]]:
    out = {}
    for estimator in COMPARED:
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob(f"{estimator}_seed*.json"))]
        out[estimator] = runs
    return out


def _save_curves(all_runs: dict[str, list[dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("acc_hard", "hard-sample template accuracy"), ("posterior_entropy", "mean posterior entropy (bits, max 24)"),
               ("mean_js", "mean JS between states' posteriors (bits)"))
    colors = {"straight_through": "tab:red", "gumbel_st": "tab:blue", "warmup_probs": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        for estimator, runs in all_runs.items():
            for i, r in enumerate(runs):
                steps = [c["step"] for c in r["curve"]]
                ax.plot(steps, [c[key] for c in r["curve"]], color=colors[estimator], alpha=0.8,
                        label=LABELS[estimator] if i == 0 else None)
        if key == "acc_hard":
            ax.axhline(0.95, color="black", linestyle=":", linewidth=1, label="criterion 0.95")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _criterion_section(ae) -> list[str]:
    """Added after the selection (DECISIONS.md, 2026-09-24): (1) does the
    label-free batch-pair JS track the labelled between-state JS? --
    correlation over every evaluation point of the (c) curves, re-run with
    the statistic recorded (the original runs never logged it); (2) the
    retained criterion-triggered switch, same 3 seeds."""
    from scipy.stats import pearsonr, spearmanr

    lines = [
        "", "## Label-free switch criterion (added after the selection)", "",
        "The original (c) runs never logged the batch statistic, so they were re-run with the same seeds and "
        "settings: every point of the 3 curves (61 each) reproduced the stored values exactly, and the batch-pair "
        "JS was added alongside. Scope of the correlation: in this isolation test the batch images are distinct "
        "states and the posterior is always taken at the same initial h, so the batch statistic is essentially a "
        "subsample of the labelled quantity. In the full RSSM the posteriors come from real contexts h_t and a "
        "pair can repeat a state -- run_mini.py plots both statistics along training to check the tracking there.",
        "",
    ]
    for estimator in ("warmup_probs", CRITERION_SWITCH):
        runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob(f"{estimator}_seed*.json"))]
        points = [(c["batch_js"], c["mean_js"]) for r in runs for c in r["curve"] if "batch_js" in c]
        if len(points) > 2:
            x, y = np.array(points).T
            lines.append(f"- {LABELS[estimator]}: label-free batch-pair JS vs labelled between-state JS over "
                         f"{len(points)} evaluation points (3 seeds): Pearson r = {pearsonr(x, y)[0]:.3f}, "
                         f"Spearman rho = {spearmanr(x, y)[0]:.3f}.")
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob(f"{CRITERION_SWITCH}_seed*.json"))]
    if runs:
        lines += ["", f"{LABELS[CRITERION_SWITCH]} -- retained defaults: 3 bits, 5 consecutive evaluations every 50 "
                  f"steps, not before step 500, deadline 60% of {ae.DEFAULT_N_STEPS} steps:", "",
                  "| seed | effective switch step | hard-sample acc. | mode acc. | pass |", "|---|---|---|---|---|"]
        for r in runs:
            f = r["final"]
            status = r.get("status", "ok")
            lines.append(f"| {r['seed']} | {r.get('switch_step') if status == 'ok' else status} | {f['acc_hard']:.4f} | "
                         f"{f['acc_mode']:.4f} | {'yes' if r['accuracy'] >= ae.CRITERION else 'no'} |")
    return lines


def write_report() -> None:
    ae = load_experiment_module("E3_worldmodel_4zones", "ae_isolation.py", "compare_estimators_report_ae")
    all_runs = _load_all()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_curves(all_runs, FIGURES_DIR / "curves.png")

    passes = {e: len(runs) == len(SEEDS) and all(r["accuracy"] >= ae.CRITERION for r in runs) for e, runs in all_runs.items()}
    selected = select_estimator(passes)

    lines = [
        "# E3 -- estimator comparison (autoencoder isolation, no KL)",
        "",
        f"k={K}, {ae.DEFAULT_N_STEPS} steps, lr {ae.DEFAULT_LR:g}, batch {ae.DEFAULT_BATCH}, seeds {list(SEEDS)}. "
        f"Schedules fixed before the runs: (b) tau 1 -> 0.1 exponentially over the first "
        f"{int(round(ae.DEFAULT_N_STEPS * ae.DEFAULT_ANNEAL_FRACTION))} steps, then held; (c) posterior probabilities "
        f"for the first {int(round(ae.DEFAULT_N_STEPS * ae.DEFAULT_WARMUP_FRACTION))} steps, then plain straight-through. "
        f"Evaluation: discrete forward pass with a hard one-hot sample, template accuracy averaged over 5 draws. "
        f"Criterion (fixed in advance): >= {ae.CRITERION} on all 3 seeds.",
        "",
        "![curves](estimator_comparison_figures/curves.png)",
        "",
        "| estimator | seed | hard-sample acc. | mode acc. | posterior entropy (bits) | mean JS between states (bits) | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for estimator, runs in all_runs.items():
        for r in runs:
            f = r["final"]
            lines.append(f"| {LABELS[estimator]} | {r['seed']} | {f['acc_hard']:.4f} | {f['acc_mode']:.4f} | "
                         f"{f['posterior_entropy']:.2f} | {f['mean_js']:.3f} | {'yes' if f['acc_hard'] >= ae.CRITERION else 'no'} |")
    lines += ["", "| estimator | passes on all 3 seeds |", "|---|---|"]
    lines += [f"| {LABELS[e]} | {'yes' if passes[e] else 'no'} |" for e in COMPARED]
    lines += ["", f"**Selected (pre-declared rule): {LABELS[selected] if selected else 'none -- no option passes'}**", ""]
    init = {e: runs[0]["curve"][0] for e, runs in all_runs.items() if runs}
    if init:
        lines.append("At initialisation (step 0, seed 0): " + "; ".join(
            f"{e}: entropy {c['posterior_entropy']:.2f} bits, JS {c['mean_js']:.4f}" for e, c in init.items()))
    lines += _criterion_section(ae)
    aids = load_experiment_module("E3_worldmodel_4zones", "compare_probs_phase_aids.py", "compare_estimators_report_aids")
    lines += aids.report_section(ae.CRITERION, FIGURES_DIR)
    (DOCS_DIR / "ESTIMATOR_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", choices=COMPARED + (CRITERION_SWITCH,))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        write_report()
    elif args.estimator:
        run_estimator(args.estimator, tuple(args.seeds), args.threads)
    else:
        parser.error("pass --estimator or --report")


if __name__ == "__main__":
    main()
