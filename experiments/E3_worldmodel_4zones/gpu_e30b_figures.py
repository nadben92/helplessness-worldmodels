"""Figures of the gpu_e30b analysis (read-only on the extracted runs). Usage, from the repo root:
python experiments/E3_worldmodel_4zones/gpu_e30b_figures.py   ->  reports/figures/gpu_e30b_trajectories.png, gpu_e30b_b_by_zone.png
(in reports/ until 2026-10-14; moved here, output path adjusted to the new depth, logic unchanged)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

spec = importlib.util.spec_from_file_location("gpu_e30b_analysis", Path(__file__).with_name("gpu_e30b_analysis.py"))
an = importlib.util.module_from_spec(spec)
sys.modules["gpu_e30b_analysis"] = an
spec.loader.exec_module(an)

OUT = Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(exist_ok=True)
COLOR = {"win64": "#1f5fa8", "win16": "#d9730d"}  # colour-blind safe pair: window length
STYLE = {"exp": "-", "ref": "--"}  # environment / reference
R = an.runs()


def line(ax, run, xs, ys, label=None):
    ax.plot(xs, ys, STYLE[run["cond"][:3]], color=COLOR[run["win"]], marker="o", ms=3, lw=1.3, label=label)


def legend(ax):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=COLOR["win64"], label="win64 (16 x 64)"), Line2D([], [], color=COLOR["win16"], label="win16 (64 x 16)"),
                       Line2D([], [], color="k", ls="-", label="environment"), Line2D([], [], color="k", ls="--", label="reference")],
              fontsize=7, loc="best")


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
ax = axes[0, 0]
for k, run in R.items():
    h = np.array(run["res"]["training_procedure"]["batch_pair_js"], dtype=float)
    ax.plot(h[:, 0], h[:, 1], STYLE[run["cond"][:3]], color=COLOR[run["win"]], lw=0.9, alpha=0.85)
    sw = run["res"]["training_procedure"]["switch_step"]
    ax.plot([sw], [3.0], marker="v", color=COLOR[run["win"]], ms=6)
ax.axhline(3.0, color="grey", lw=0.8, ls=":")
ax.set_xlim(0, 10000)
ax.set_title("batch-pair JS (bits); triangle: switch step;\ndotted: 3-bit threshold", fontsize=8)
ax.set_xlabel("gradient step")
legend(ax)

zones = {"controllable_noisy": "controllable-noisy zone", "uncontrollable_noisy": "uncontrollable-noisy zone"}
for ax, (zone, title) in zip((axes[0, 1], axes[0, 2]), zones.items()):
    for k, run in R.items():
        steps = [d["step"] for d in run["diag"]]
        line(ax, run, steps, [d["real_context"]["l1_vs_p_by_zone"][zone] for d in run["diag"]])
    ax.axhline(0.11, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"L1(WM, P), {title}\n(zone mean; dotted: about the C-b bound; K = 256, 1,358 at 60,000)", fontsize=8)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("L1")

for ax, (zone, title) in zip((axes[1, 0], axes[1, 1]), zones.items()):
    for k, run in R.items():
        steps = [d["step"] for d in run["diag"]]
        line(ax, run, steps, [d["real_context"]["I_s_by_zone"][zone]["mean"] for d in run["diag"]])
    if zone == "controllable_noisy":
        ax.axhline(1.653, color="k", lw=0.8, ls=":")
    else:
        ax.axhline(1.653, color="grey", lw=0.8, ls=":")  # the reference's truth (controllable-noisy dynamics)
        ax.axhline(0.0, color="k", lw=0.8, ls=":")  # the environment's truth
    ax.set_xscale("log")
    ax.set_title(f"I_s (MAIN), {title}\n(dotted: truth" + (" 1.653)" if zone == "controllable_noisy" else ": 0 environment, 1.653 reference)"), fontsize=8)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("bits")

ax = axes[1, 2]
for k, run in R.items():
    steps = [d["step"] for d in run["diag"]]
    line(ax, run, steps, [d["real_context"]["state_probe_h"] for d in run["diag"]])
    ax.plot(steps, [d["real_context"]["state_probe_z"] for d in run["diag"]], ":", color=COLOR[run["win"]], lw=0.9)
ax.set_xscale("log")
ax.set_title("real-context probes: h -> s (lines), z -> s (dotted); chance 0.007", fontsize=8)
ax.set_xlabel("gradient step")
ax.set_ylabel("test accuracy")
fig.tight_layout()
fig.savefig(OUT / "gpu_e30b_trajectories.png", dpi=110)
plt.close(fig)

# b by zone: measured vs theory
fig, ax = plt.subplots(figsize=(9, 4.2))
cats = an.CATS
theory = {"uncontrollable_deterministic": -2.322, "controllable_noisy": 0.0, "uncontrollable_noisy": -1.653, "controllable_deterministic": 0.0}
x = np.arange(len(cats))
pairs = [(w, s) for w in ("win64", "win16") for s in (0, 5)]
width = 0.8 / len(pairs)
for i, (w, s) in enumerate(pairs):
    e, r = R[f"{w} s{s} exp"], R[f"{w} s{s} ref"]
    b = [e["res"]["real_context"]["I_s_by_zone"][c]["mean"] - r["res"]["real_context"]["I_s_by_zone"][c]["mean"] for c in cats]
    ax.bar(x + i * width, b, width, color=COLOR[w], alpha=1.0 if s == 0 else 0.55, label=f"{w} seed {s} (class {'A' if s == 0 else 'B'})")
ax.scatter(x + 0.4 - width / 2, [theory[c] for c in cats], marker="_", color="k", s=400, zorder=3, label="theory")
ax.set_xticks(x + 0.4 - width / 2)
ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
ax.set_ylabel("b = I_s(environment) - I_s(reference), bits")
ax.set_title("b by zone (I_s, real contexts, K = 1,358); C-b fails in both noisy zones of all 8 runs", fontsize=9)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "gpu_e30b_b_by_zone.png", dpi=110)
print("ok")
