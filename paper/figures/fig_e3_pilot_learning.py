"""E3, closed-loop pilot (pilot data: seeds 0 and 5, environment and reference, 300,000 environment steps). (a) success
rate over each interval between two checkpoints, with its 95 % Wilson interval, against the uniform policy's exact
rate in each world; (b) median prior mass on the goal over the four pairs into it, at every checkpoint. Writes
paper/figures/out/fig_e3_pilot_learning.pdf.

Source: the pilot's run directories (results/E3_worldmodel_4zones/*gpu_pilot_closed_loop__seed*, extracted from
handoff/gpu_pilot_cl/), checkpoints.jsonl: cumulative episodes and successes, goal_diagnostics.median_mass_on_goal_pairs.
The interval's Wilson interval is experiments/E3_worldmodel_4zones/diagnose_agent_run.py's own `wilson`.
UNIFORM_SUCCESS: the uniform policy's success rate computed exactly under P (uniform starts over the non-terminal
states, H = 36): docs/e3/DECISIONS.md, 2026-09-30 (7), expected shape 2 -- no results file holds it.
"""

from __future__ import annotations

import numpy as np

import common
import e3_sources as src

NAME = "fig_e3_pilot_learning"
UNIFORM_SUCCESS = {"experimental": 0.0459, "reference": 0.1173}  # docs/e3/DECISIONS.md, 2026-09-30 (7)
STYLE = {("experimental", 0): ("-", "o"), ("experimental", 5): ("-", "s"),
         ("reference", 0): ("--", "o"), ("reference", 5): ("--", "s")}
COLOR = {"experimental": common.TOL_BRIGHT["red"], "reference": common.TOL_BRIGHT["blue"]}


def main() -> None:
    wilson = src.e3_module("diagnose_agent_run.py").wilson
    common.apply_style()
    fig, (ax_s, ax_m) = common.figure(aspect=0.42, ncols=2)
    for run in src.pilot_runs():
        seed, condition = src.run_identity(run)
        records = src.read_jsonl(run / "checkpoints.jsonl")
        steps = np.array([r["env_step"] for r in records], dtype=float)
        episodes = np.array([0] + [r["episodes"] for r in records])
        successes = np.array([0] + [r["successes"] for r in records])
        k, n = np.diff(successes), np.diff(episodes)
        rate = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        ci = np.array([wilson(int(a), int(b)) if b > 0 else (np.nan, np.nan) for a, b in zip(k, n)])
        ls, marker = STYLE[(condition, seed)]
        label = f"{src.CONDITION_ABBR[condition]} {seed}"
        ax_s.plot(steps, rate, ls=ls, marker=marker, ms=2.5, lw=0.9, color=COLOR[condition], label=label)
        ax_s.fill_between(steps, ci[:, 0], ci[:, 1], color=COLOR[condition], alpha=0.12, lw=0)
        mass = [r["goal_diagnostics"]["median_mass_on_goal_pairs"] for r in records]
        ax_m.plot(steps, [np.nan if m is None else m for m in mass], ls=ls, marker=marker, ms=2.5, lw=0.9,
                  color=COLOR[condition], label=label)
    for condition, rate in UNIFORM_SUCCESS.items():
        ax_s.axhline(rate, color=COLOR[condition], lw=0.6, ls=":")
    for ax in (ax_s, ax_m):
        ax.set_xscale("log")
        ax.set_xlabel("environment steps")
        ax.set_ylim(-0.03, 1.03)
    ax_s.set_ylabel("success rate over the interval")
    ax_m.set_ylabel("median prior mass on the goal")
    ax_s.set_title("(a)", fontsize=common.SMALL_FONT_SIZE)
    ax_m.set_title("(b)", fontsize=common.SMALL_FONT_SIZE)
    ax_m.legend(loc="center right", fontsize=common.SMALL_FONT_SIZE - 1)
    fig.tight_layout(pad=0.2)
    out = common.save_pdf(fig, NAME)
    print(f"wrote {out.relative_to(common.REPO_ROOT)}")


if __name__ == "__main__":
    main()
