"""E7g_critic (v2) -- Same two panels as E7g, abscissa = gain in the controllability read by the critic (Delta C1, D7,
main z-bar), for the constrained correction runs of D7 only (10 (agent, setting) pairs, stepwise, alpha = 1, training
image). One point = one D7 corrected run: Delta C1 from d7_correction.csv (corrected - uncorrected on the same steps),
localisation in the zone and directed share from the run's summary.json (scratch-free: results/E7_dissociation/d7/runs).
Full correction = C1 of the natural references - C1 of the natural exposed agents (d7_bins.csv, weighted by steps).
Vermilion: exposed, no intervention (D7's natural runs, mean of 4). Star, reference and chance lines as in E7g."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style_v2 import common as C, palette as P  # noqa: E402
import E7g_correction_fails as G  # noqa: E402

FIG_ID = "E7g_critic"
D7 = ("results", "E7_dissociation", "d7")
ZBAR = "draw"


def natural_c1():
    bins = C.read_csv(C.path(*D7, "analysis", "d7_bins.csv"))

    def grp(name):
        sel = [r for r in bins if r["zbar"] == ZBAR and r["group"] == name and r["measure"] == "C1"
               and not C.truthy(r["masked"])]
        return np.average([C.num(r["value"]) for r in sel], weights=[C.num(r["n_steps"]) for r in sel])
    return grp("references, no intervention") - grp("exposed")


def main():
    t = G.targets()
    corr = {(r["agent"], r["setting"]): r for r in C.read_csv(C.path(*D7, "analysis", "d7_correction.csv"))
            if r["zbar"] == ZBAR}
    runs = C.path(*D7, "runs")
    pts, base = [], []
    for d in sorted(runs.iterdir()):
        rows = json.loads((d / "summary.json").read_text(encoding="utf-8"))["rows"]
        if "__corrected__" in d.name:
            r = rows[0]
            pts.append((C.num(corr[(r["agent"], r["setting"])]["dC1"]), r["loc_template_zone"], r["directed_share"]))
        elif "__natural__" in d.name:
            r = next(x for x in rows if x["model"] == "exposed" and x["mode"] == "none")
            base.append((r["loc_template_zone"], r["directed_share"]))
    pts, base = np.array(pts), np.array(base)
    full = natural_c1()
    fig, (al, ar) = C.full(1, 2, sharex=True)
    al.scatter(pts[:, 0], pts[:, 1], s=130, color=P.CORRECTION, edgecolor="black", linewidth=0.8, zorder=4)
    ar.scatter(pts[:, 0], pts[:, 2], s=130, color=P.CORRECTION, edgecolor="black", linewidth=0.8, zorder=4)
    al.scatter([0], [base[:, 0].mean()], s=170, color=P.EXPOSED, edgecolor="black", linewidth=0.8, zorder=5)
    ar.scatter([0], [base[:, 1].mean()], s=170, color=P.EXPOSED, edgecolor="black", linewidth=0.8, zorder=5)
    G.star(al, full, 1.0)
    G.star(ar, full, t["oracle"])
    t = {**t, "loc0": base[:, 0].mean()}
    G.decorate((al, ar), t, full + 0.08, "ΔC1")
    for ax in (al, ar):
        ax.set_xlim(-0.25, full + 0.08)
    print(f"E7g_critic: {len(pts)} runs, full C1 gain {full:.3f}")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
