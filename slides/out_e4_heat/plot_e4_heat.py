"""E4 controllability heatmaps (main figure p1-b-d1, appendix grid) and the manifest, from data_<config>.npz
(measure_e4_heat.py). Style: no title, legend, annotation or text in cells."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from helplessness.envs.layouts import two_zones  # noqa: E402
from helplessness.pipeline import two_zones_setup as setup  # noqa: E402

OUT = Path(__file__).parent
CONFIGS = ["p1-a-d1", "p1-b-d1", "p2-a-d1", "p2-b-d1"]
MAX = float(np.log2(5))
GOAL = two_zones.GOAL_STATE


def regions(conf):
    r = two_zones.region_states(two_zones.build_layout(conf["diagonal"], "experimental"))
    ro = conf["roles"]
    out = {"UD": r[ro["ud"]], "CD": r[ro["cd"]], "neutral": np.concatenate([r[ro["neutral"]], r[ro["other"]]]),
           "corridor": r["corridor"][r["corridor"] != GOAL]}
    return out, r


def load(c):
    d = np.load(OUT / f"data_{c}.npz")
    Ie, Ir = d["I_experimental"].copy(), d["I_control"].copy()
    for a in (Ie, Ir):
        a[GOAL] = np.nan
    return d, Ie, Ir, Ie - Ir


def draw(axs, cax, Ie, Ir, b, conf, cb=True):
    masks = [two_zones.quadrant_mask(q) for q in two_zones.QUADRANT_ORDER]
    ims = []
    for ax, v, cmap, lim in ((axs[0], Ie, "viridis", (0, MAX)), (axs[1], Ir, "viridis", (0, MAX)), (axs[2], b, "coolwarm", (-MAX, MAX))):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
        ims.append(ax.imshow(v.reshape(12, 12), cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest"))
        for m in masks:
            ax.contour(m.astype(float), levels=[0.5], colors="gray", linewidths=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if cb:
        plt.colorbar(ims[0], cax=cax[0]); plt.colorbar(ims[2], cax=cax[1])


def main():
    cfg = setup.load_config(REPO / "configs/e4/e4.yaml")
    confs = {c: setup.configuration(cfg, c) for c in CONFIGS}
    data = {c: load(c) for c in CONFIGS if (OUT / f"data_{c}.npz").exists()}
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.05, 1, 0.05], "wspace": 0.35})
    _, Ie, Ir, b = data["p1-b-d1"][0], *data["p1-b-d1"][1:]
    draw([axs[0], axs[1], axs[3]], [axs[2], axs[4]], Ie, Ir, b, confs["p1-b-d1"])
    fig.savefig(OUT / "E4_heat_p1b.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, axs = plt.subplots(len(data), 5, figsize=(14, 3.4 * len(data)), gridspec_kw={"width_ratios": [1, 1, 0.05, 1, 0.05], "wspace": 0.35})
    for i, c in enumerate(data):
        _, Ie, Ir, b = data[c]
        draw([axs[i, 0], axs[i, 1], axs[i, 3]], [axs[i, 2], axs[i, 4]], Ie, Ir, b, confs[c])
    fig.savefig(OUT / "E4_heat_all.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    rows, valid = [], []
    for c, (d, Ie, Ir, b) in data.items():
        reg, _ = regions(confs[c])
        ms = d["mass_control"]
        def f(x): return f"{np.nanmean(x):+.2f} ± {np.nanstd(x):.2f}"
        nonud = np.concatenate([reg[k] for k in ("CD", "neutral", "corridor")])
        rows.append(f"| {c} | " + " | ".join(f(b[reg[k]]) for k in ("UD", "CD", "neutral", "corridor")) + f" | {np.nanmax(np.abs(b[nonud])):.2f} |")
        valid.append(f"| {c} | " + " | ".join(f"{np.nanmean(ms[reg[k]]):.3f}" for k in ("UD", "CD", "neutral", "corridor")) + " |")
    c0 = confs["p1-b-d1"]; reg0, r0 = regions(c0)
    def rc(s):
        return f"rows {s // 12 and 0}"
    def box(states):
        rr, cc = np.divmod(states, 12); return f"rows {rr.min()}-{rr.max()}, cols {cc.min()}-{cc.max()}"
    lines = []
    for c in data:
        cf = confs[c]; rg, r = regions(cf); ro = cf["roles"]
        lines.append(f"- {c} (seed {cf['seed']:02d}, diagonal {cf['diagonal']}): UD = slot {ro['ud']} ({box(r[ro['ud']])}); CD = slot {ro['cd']} ({box(r[ro['cd']])}); "
                     f"neutral quadrants = {ro['neutral']} ({box(r[ro['neutral']])}) and {ro['other']} ({box(r[ro['other']])}); "
                     f"corridor = the cross, rows/cols {cfg and ''}{box(r['corridor'])} (bounding box; goal excluded); goal = state {GOAL} (row, col) = {divmod(GOAL, 12)} (masked)")
    d = next(iter(data.values()))[0]
    md = f"""# MANIFEST e4_heat

Quantity: I(a;s'|s) in bits (prior, 5 actions, max log2(5) = {MAX:.3f}), per state of the TRAINING environment (training image), final checkpoint
(env step 300000), frozen weights, CPU. b = I_exposed - I_reference. Goal state masked (white). Contours: thin grey, the 4 quadrants.
Geometry per configuration (not realigned between configurations; array row = image row, top to bottom):
{chr(10).join(lines)}

## Figures
- `E4_heat_p1b.png` (p1-b-d1, seed 02, n = 1 seed): panel 1 I exposed, panel 2 I reference (viridis, fixed [0, {MAX:.3f}], shared colorbar), panel 3 b (coolwarm, fixed [-{MAX:.3f}, +{MAX:.3f}], own colorbar).
- `E4_heat_all.png`: 4 rows (p1-a-d1 seed 01, p1-b-d1 seed 02, p2-a-d1 seed 05, p2-b-d1 seed 06; n = 1 seed each) x the same 3 panels and scales.

## Source and estimator
- Weights: `agent_env300000.pt` of the E4 wave-1 runs `grid12x12__2zones-<config>__pair-{{experimental,control}}__worldmodel-agent-e4__seed<NN>__*` (extract at /c/nrw/e6all/E4_generalization_gradient).
- No per-state map existed: E4's records hold region means only, and the replay buffers (real-context probe) are not in the exports. Recomputed with `slides/out_e4_heat/measure_e4_heat.py`
  (data: `data_<config>.npz`), using the existing E4 walk estimator `probing.fixed_trajectory` (`fixed_trajectory_context` + `step_controllability`, I via `controllability_of_kernel`), with E4's settings
  (`configs/e4/e4.yaml`): fixed walks of `walks_for_slot` for each of the 4 slots (n_pre 32, n_in 25, seed 2026), probe environment = reference dynamics on the training image, measurement seed `measurement_seed(..., 'probe')`.
  Deviation from E4's reported quantity: instead of the zone steps only (entry / curve), I_s is taken at EVERY walk step (corridor, path, zone) after a burn-in of {8} steps, and averaged per state.
  Walks per slot: {int(d['n_walks'])} (E4's full set is 300); draws per action: {int(d['draws'])} (E4's probe setting); contexts per state: min {int(min(np.nanmin(x[0]['count_experimental']) for x in data.values()))} (exposed).
- Context = filtered walk history (not replay contexts); the exposed model's I is read at the reference's (all controllable) true successors for the mass only.

## b by region (mean ± sd over states; goal excluded)
| config | UD | CD | neutral | corridor | max abs b outside UD |
|---|---|---|---|---|---|
{chr(10).join(rows)}

## Validity: reference mass on the true successor (mean over states, criterion >= 0.90)
| config | UD | CD | neutral | corridor |
|---|---|---|---|---|
{chr(10).join(valid)}
"""
    (OUT / "MANIFEST_e4_heat.md").write_text(md, encoding="utf-8")
    print("\n".join(rows)); print("\n".join(valid))


if __name__ == "__main__":
    main()
