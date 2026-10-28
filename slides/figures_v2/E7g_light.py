"""E7g_light (v2) -- E7g with fewer points: localisation (left) and directed share (right) against Delta b, training
image, E4's four agents. One point = one (agent, setting), averaged over its episodes:
  light violet  unconstrained correction, stepwise mode only, 4 agents x alpha in {0.25, 0.5, 0.75, 1, 1.5} (20 points;
                dev_e4_cpu/analysis/checks.csv)
  dark violet   constrained stepwise correction, alpha = 1: exactly the 10 corrected runs of D7 used by E7g_critic
                (results/E7_dissociation/d7/runs/*corrected*/summary.json)
Delta b = b_step1 of the run - b_step1 of the same agent without intervention (E7g's baselines). Exposed point, star,
reference and chance lines exactly as in E7g."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import E7g_correction_fails as G  # noqa: E402

FIG_ID = "E7g_light"
D7_RUNS = ("results", "E7_dissociation", "d7", "runs")


def main():
    t = G.targets()
    base = t["base"]
    light, _ = G.episode_points(C.read_csv(C.path(*G.V1)), ("stepwise",))
    dark = []
    for d in sorted(C.path(*D7_RUNS).iterdir()):
        if "__corrected__" in d.name:
            r = json.loads((d / "summary.json").read_text(encoding="utf-8"))["rows"][0]
            dark.append((r["b_step1"] - base[r["agent"]]["b"], r["loc_template_zone"], r["directed_share"]))
    dark = np.array(dark)
    fig, (al, ar) = C.full(1, 2, sharex=True)
    for ax, key, col_dark in ((al, "loc", 1), (ar, "dir", 2)):
        ax.scatter(light["db"], light[key], s=130, color=G.LIGHT, edgecolor="black", linewidth=0.8, zorder=4)
        ax.scatter(dark[:, 0], dark[:, col_dark], s=130, color=P.CORRECTION, edgecolor="black", linewidth=0.8,
                   zorder=4)
    al.scatter([0], [t["loc0"]], s=170, color=P.EXPOSED, edgecolor="black", linewidth=0.8, zorder=5)
    ar.scatter([0], [t["dir0"]], s=170, color=P.EXPOSED, edgecolor="black", linewidth=0.8, zorder=5)
    G.star(al, t["full_b"], 1.0)
    G.star(ar, t["full_b"], t["oracle"])
    G.decorate((al, ar), t, t["full_b"] + 0.25, "Δb (bits)")
    print(f"E7g_light: {len(light['db'])} light, {len(dark)} dark; dark Delta b {dark[:, 0].min():.2f}-"
          f"{dark[:, 0].max():.2f}, loc {dark[:, 1].min():.2f}-{dark[:, 1].max():.2f}, "
          f"directed {dark[:, 2].min():.2f}-{dark[:, 2].max():.2f}; light loc {light['loc'].min():.2f}-"
          f"{light['loc'].max():.2f}")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
