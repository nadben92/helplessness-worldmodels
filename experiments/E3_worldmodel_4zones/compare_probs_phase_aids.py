"""The lr 1e-4 separation failure, in the autoencoder isolation test
(docs/e3/DECISIONS.md, 2026-09-24). The mini run at lr 1e-4 hit the switch
deadline (SeparationNotReachedError at step 6307) with the retained
procedure (c'); this script checks whether the isolation test reproduces
it and compares three fixes. No KL, k=6, 3 seeds, 10,000-step budget,
switch deadline 60% of the budget (6000), batch 32 -- all variants:

  reproduction: (c') unchanged, lr 1e-4.
  (i)   lr 1e-3, (c') unchanged.
  (ii)  lr 1e-4, entropy penalty eta(t) * H(q) during the probabilities
        phase, eta ramping linearly 0 -> ETA_MAX over the deadline,
        removed at the switch (loop.ProbsPhaseAid "entropy_penalty").
  (iii) lr 1e-4, progressive samples: a fraction rho(t) of the latent
        variables gets the straight-through sample, rho ramping 0 -> 1 over
        the deadline; the switch stays governed by the JS criterion
        (loop.ProbsPhaseAid "progressive_samples").

Criterion fixed in advance: hard-sample template accuracy >= 0.95 on all
3 seeds, reached within the 10,000 steps (the run's final evaluation).
Selection rule fixed in advance: the simplest option that passes, in the
order (ii), (iii), (i).

    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_probs_phase_aids.py --variant reproduction
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_probs_phase_aids.py --variant ii_entropy_penalty
    ...
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\compare_estimators.py --report   (includes these results)

Verification of the fix for (i)'s late collapse (DECISIONS.md,
2026-09-24): (i) + lr_after_switch = 1e-4 from the effective switch +
no gradient clipping, same criterion, and no collapse after the
switch (CollapseAfterSwitchError guard active). Two lots x 3 seeds:
  i_fixed: the current code's RNG stream (3 threads, as the retained (i) re-run);
  i_fixed_legacy_stream: the discarded (i) batch's RNG stream (at-switch
      snapshot drawing from the global torch RNG, 1 thread) -- the stream on
      which seed 0 collapsed.

Raw results: results/E3_worldmodel_4zones/probs_phase_aids/<variant>_seedNN.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from helplessness.experiment_loader import load_experiment_module

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "E3_worldmodel_4zones" / "probs_phase_aids"

N_STEPS = 10_000
BATCH = 32
K = 6
SEEDS = (0, 1, 2)
SLOW_LR = 1e-4
FAST_LR = 1e-3
# Fixed before the (ii) runs, from the reproduction runs' reconstruction
# scale -- justification in DECISIONS.md (2026-09-24).
ETA_MAX = 1.0
LR_AFTER_SWITCH = 1e-4
# No global gradient clipping (DECISIONS.md, 2026-09-24): a threshold of 500
# broke the probabilities phase and would not have acted on the collapse
# mechanism. Kept as a named constant for the record.
CLIP_NORM = None

VARIANTS = {
    "reproduction": {"lr": SLOW_LR, "aid": None},
    "i_lr_1e-3": {"lr": FAST_LR, "aid": None},
    "ii_entropy_penalty": {"lr": SLOW_LR, "aid": "entropy_penalty"},
    "iii_progressive_samples": {"lr": SLOW_LR, "aid": {"kind": "progressive_samples"}},
    "i_fixed": {"lr": FAST_LR, "aid": None, "fix": True, "legacy": False, "threads": 3},
    "i_fixed_legacy_stream": {"lr": FAST_LR, "aid": None, "fix": True, "legacy": True, "threads": 1},
}
LABELS = {
    "reproduction": "(c') at lr 1e-4 (reproduction)",
    "i_lr_1e-3": "(i) lr 1e-3, (c') unchanged",
    "ii_entropy_penalty": "(ii) entropy penalty during the probabilities phase",
    "iii_progressive_samples": "(iii) progressive samples during the probabilities phase",
    "i_fixed": "(i) + lr 1e-4 after the switch",
    "i_fixed_legacy_stream": "(i) + lr 1e-4 after the switch, discarded batch's RNG stream",
}
SELECTION_ORDER = ("ii_entropy_penalty", "iii_progressive_samples", "i_lr_1e-3")


def _aid(variant: str) -> dict | None:
    aid = VARIANTS[variant]["aid"]
    if aid == "entropy_penalty":
        if ETA_MAX is None:
            raise RuntimeError("ETA_MAX is not fixed yet -- fix it (and log it in DECISIONS.md) before running (ii)")
        return {"kind": "entropy_penalty", "eta_max": ETA_MAX}
    return aid


def run_variant(variant: str, seeds: tuple[int, ...], threads: int) -> None:
    ae = load_experiment_module("E3_worldmodel_4zones", "ae_isolation.py", "compare_aids_ae_isolation")
    torch.set_num_threads(threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    spec = VARIANTS[variant]
    fix = {}
    if spec.get("fix"):
        fix = {"lr_after_switch": LR_AFTER_SWITCH, "clip_norm": CLIP_NORM, "legacy_snapshot_rng": spec["legacy"]}
        torch.set_num_threads(spec["threads"])
    for seed in seeds:
        result = ae.run(N_STEPS, spec["lr"], BATCH, seed, "warmup_probs_until_separated", k=K,
                        aid=_aid(variant), raise_on_deadline=False, **fix)
        result["variant"] = variant
        (RESULTS_DIR / f"{variant}_seed{seed:02d}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        f = result["final"]
        print(f"{variant} seed={seed}: status={result['status']} switch={result['switch_step']} "
              f"acc_hard={f['acc_hard']:.4f} entropy={f['posterior_entropy']:.2f} js={f['mean_js']:.3f}", flush=True)


def load_all() -> dict[str, list[dict]]:
    return {v: [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob(f"{v}_seed*.json"))]
            for v in VARIANTS}


def passes(runs: list[dict], criterion: float) -> bool:
    return len(runs) == len(SEEDS) and all(r["status"] == "ok" and r["final"]["acc_hard"] >= criterion for r in runs)


def select(all_runs: dict[str, list[dict]], criterion: float) -> str | None:
    for variant in SELECTION_ORDER:
        if passes(all_runs[variant], criterion):
            return variant
    return None


def first_step_at(curve: list[dict], criterion: float) -> int | None:
    return next((c["step"] for c in curve if c["acc_hard"] >= criterion), None)


def save_curves(all_runs: dict[str, list[dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("acc_hard", "hard-sample template accuracy"), ("posterior_entropy", "mean posterior entropy (bits, max 24)"),
               ("mean_js", "mean JS between states' posteriors (bits)"))
    colors = {"reproduction": "tab:red", "i_lr_1e-3": "tab:purple", "ii_entropy_penalty": "tab:orange",
              "iii_progressive_samples": "tab:cyan", "i_fixed": "tab:green", "i_fixed_legacy_stream": "tab:olive"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        for variant, runs in all_runs.items():
            for i, r in enumerate(runs):
                steps = [c["step"] for c in r["curve"]]
                ax.plot(steps, [c[key] for c in r["curve"]], color=colors[variant], alpha=0.8,
                        label=LABELS[variant] if i == 0 else None)
                if r.get("switch_step") is not None:
                    ax.axvline(r["switch_step"], color=colors[variant], linestyle=":", linewidth=0.8, alpha=0.6)
        if key == "acc_hard":
            ax.axhline(0.95, color="black", linestyle=":", linewidth=1, label="criterion 0.95")
        ax.axvline(int(round(0.6 * N_STEPS)), color="grey", linestyle="--", linewidth=1, label="switch deadline")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


COLLAPSED_REFERENCE = RESULTS_DIR / "discarded_rng_perturbed_snapshot" / "i_lr_1e-3_seed00_instrumented_step_log.json"


def trajectory_identity_lines(all_runs: dict[str, list[dict]]) -> list[str]:
    """Controlled-experiment check: the discarded-stream lot's seed 0 (lr
    1e-3 -> 1e-4 at the switch, no clipping, 1 thread, the discarded batch's
    RNG stream) must be IDENTICAL, step by step, to the run that collapsed
    (same lr 1e-3, same stream, same threads) up to the step the lr
    changes: the lr acts only from switch_step's update on, so the loss and
    global gradient norm of steps 0..switch_step are still those of the
    collapsed run, and they must be bit-identical."""
    runs = [r for r in all_runs["i_fixed_legacy_stream"] if r["seed"] == 0]
    if not runs or not COLLAPSED_REFERENCE.exists():
        return []
    new, ref = runs[0], json.loads(COLLAPSED_REFERENCE.read_text(encoding="utf-8"))
    sw = new["switch_step"]
    ref_loss = [e["loss"] for e in ref["step_log"]]
    ref_grad = [e["grad_norm"] for e in ref["step_log"]]
    n = (sw + 1) if sw is not None else 0
    same_switch = sw == ref["switch_step"]
    same_loss = new.get("losses", [])[:n] == ref_loss[:n]
    same_grad = new["grad_norms"][:n] == ref_grad[:n]
    div = next((i for i in range(min(len(ref_loss), len(new.get("losses", [])))) if new["losses"][i] != ref_loss[i]), None)
    return ["", f"Trajectory identity, discarded-stream lot seed 0 vs the run that collapsed (lr 1e-3 throughout): switch step "
            f"{sw} vs {ref['switch_step']} ({'same' if same_switch else 'DIFFERENT'}); per-step loss and global gradient norm "
            f"identical over steps 0..{n - 1}: {'yes' if same_loss and same_grad else 'NO'}; first step where the losses differ: "
            f"{div if div is not None else 'none'} (the lr change applies to the update of step {sw if sw is not None else '-'}, so the first loss it can change is step {sw + 1 if sw is not None else '-'})."]


def report_section(criterion: float, figures_dir: Path) -> list[str]:
    """The section compare_estimators.write_report appends to
    ESTIMATOR_COMPARISON.md (empty when nothing has run yet)."""
    all_runs = load_all()
    if not any(all_runs.values()):
        return []
    save_curves(all_runs, figures_dir / "probs_phase_aids_curves.png")
    lines = [
        "", "## The lr 1e-4 separation failure (added after the mini run's SeparationNotReachedError)", "",
        f"k={K}, no KL, {N_STEPS} steps, batch {BATCH}, seeds {list(SEEDS)}, procedure (c') with its switch deadline at 60% "
        f"of the budget ({int(round(0.6 * N_STEPS))}). Criterion fixed in advance: hard-sample accuracy >= {criterion} on all 3 "
        "seeds within the budget. Selection rule fixed in advance: the simplest option that passes, in the order (ii), (iii), (i)."
        + (f" (ii): eta_max = {ETA_MAX:g}, ramped linearly over the deadline." if ETA_MAX is not None else ""), "",
        "![curves](estimator_comparison_figures/probs_phase_aids_curves.png)", "",
        "| variant | seed | status | switch step | first step >= 0.95 | final hard-sample acc. | at switch/deadline: acc. probs | acc. hard sample | labelled JS (bits) | posterior entropy (bits) | I(z;s) (bits) | centred logit norm |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, runs in all_runs.items():
        for r in runs:
            snap = r.get("at_switch") or r.get("at_deadline") or {}
            where = "switch" if r.get("at_switch") else "deadline"
            first = first_step_at(r["curve"], criterion)

            def g(key, fmt="{:.3f}"):
                return fmt.format(snap[key]) if key in snap else "-"

            lines.append(
                f"| {LABELS[variant]} | {r['seed']} | {r['status']} | {r.get('switch_step') or '-'} | {first if first is not None else '-'} | "
                f"{r['final']['acc_hard']:.4f} | {g('acc_probs')} ({where}, step {snap.get('step', '-')}) | {g('acc_hard')} | "
                f"{g('mean_js')} | {g('posterior_entropy', '{:.2f}')} | {g('I_zs', '{:.2f}')} | {g('logit_norm_centered', '{:.3f}')} |"
            )
    lines += ["", "| variant | passes on all 3 seeds |", "|---|---|"]
    selectable = [v for v in VARIANTS if v not in ("i_fixed", "i_fixed_legacy_stream")]
    lines += [f"| {LABELS[v]} | {'yes' if passes(all_runs[v], criterion) else ('no' if all_runs[v] else 'not run')} |" for v in selectable]
    selected = select(all_runs, criterion)
    lines += ["", f"**Selected (pre-declared rule): {LABELS[selected] if selected else 'none -- no option passes'}**"]
    lines += ["", "Discarded first batch of (i) (its at-switch snapshot shifted the torch RNG stream of every later "
              "straight-through draw; fixed with torch.random.fork_rng, (i) re-run): seed 0 reached 1.000 and then "
              "collapsed in a single step -- loss 1.1-2.1 up to step 9981, 40.1 at step 9982, never recovering; final "
              "posterior entropy 0.7756 bits = 8 x a unimix'd one-hot, the same saturated code for every state. The "
              "straight-through phase at lr 1e-3 can blow up late, depending on the sampling stream. Raw files: "
              "results/E3_worldmodel_4zones/probs_phase_aids/discarded_rng_perturbed_snapshot/; DECISIONS.md (2026-09-24)."]
    fixed = [(v, r) for v in ("i_fixed", "i_fixed_legacy_stream") for r in all_runs[v]]
    if fixed:
        lines += [
            "", "### Fix for the late collapse: lr 1e-4 after the switch", "",
            f"Cause (per-step diagnostic of the collapsed run, DECISIONS.md 2026-09-24): with a saturated posterior the encoder "
            f"receives tiny gradients (median 0.38), so its Adam second moment is tiny; a rare 100x gradient (step 9980) then "
            f"produces a large coordinated Adam step at lr 1e-3 that throws the posterior into a saturated state-independent "
            f"code, from which the straight-through gradient cannot recover. Fix: lr {FAST_LR:g} until the effective switch, "
            f"then {LR_AFTER_SWITCH:g}; no gradient clipping. "
            "Guard: CollapseAfterSwitchError if the batch-pair JS falls below 1 bit on 5 consecutive evaluations after the "
            "switch. Pass = criterion met AND no collapse after the switch. Two lots x 3 seeds: the current RNG stream, and "
            "the discarded batch's stream (seed 0 = the deterministic reproduction of the collapse).", "",
            "| lot | seed | status | switch step | lr change step | final hard-sample acc. | min batch-pair JS after the switch (bits) | pass |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for v, r in fixed:
            after = [c["batch_js"] for c in r["curve"] if r.get("switch_step") and c["step"] >= r["switch_step"] and "batch_js" in c]
            ok = r["status"] == "ok" and r["final"]["acc_hard"] >= criterion
            lines.append(f"| {LABELS[v]} | {r['seed']} | {r['status']} | {r.get('switch_step') or '-'} | {r.get('lr_switch_step') or '-'} | "
                         f"{r['final']['acc_hard']:.4f} | {min(after):.2f} | {'yes' if ok else 'no'} |"
                         if after else
                         f"| {LABELS[v]} | {r['seed']} | {r['status']} | {r.get('switch_step') or '-'} | {r.get('lr_switch_step') or '-'} | "
                         f"{r['final']['acc_hard']:.4f} | - | {'yes' if ok else 'no'} |")
        all_ok = len(fixed) == 2 * len(SEEDS) and all(r["status"] == "ok" and r["final"]["acc_hard"] >= criterion for _, r in fixed)
        lines += ["", f"**Fix verified on all {2 * len(SEEDS)} runs: {'yes' if all_ok else 'no'}**"]
        lines += trajectory_identity_lines(all_runs)
    lost = [(v, r) for v, runs in all_runs.items() for r in runs
            if first_step_at(r["curve"], criterion) is not None and r["final"]["acc_hard"] < criterion]
    if lost:
        lines += ["", "Reading of the criterion: 'within the budget' is scored on the FINAL evaluation (step 10,000). "
                  "Runs that reached the criterion and then lost it (they would pass under the reading 'reached at some "
                  "point before 10,000 steps'):"]
        lines += [f"- {LABELS[v]}, seed {r['seed']}: >= {criterion} from step {first_step_at(r['curve'], criterion)}, final "
                  f"{r['final']['acc_hard']:.4f} (posterior entropy {r['final']['posterior_entropy']:.4f} bits, JS "
                  f"{r['final']['mean_js']:.3f})." for v, r in lost]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    run_variant(args.variant, tuple(args.seeds), args.threads)


if __name__ == "__main__":
    main()
