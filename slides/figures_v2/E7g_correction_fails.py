"""E7g (v2) -- Why the latent correction of h fails: two panels, abscissa = measured belief gain Delta b at step 1
(training image, E4's four exposed agents). Left: localisation in the zone; right: directed share.

One point = one (agent, setting), averaged over its episodes (or over its 100 situations for the sweep):
  light violet circles   unconstrained correction applied in episodes (E7 first report: modes `direction` and
                         `stepwise`, alpha in {0.25, 0.5, 0.75, 1, 1.5}; dev_e4_cpu/analysis/checks.csv)
  dark violet circles    constrained correction applied in episodes (alpha = 1, `constrained_direction` and
                         `constrained_stepwise`, the retained setting per agent; dev_e4_cpu_v2/analysis/checks.csv)
  light / dark squares   the static sweep (100 situations per agent; constrained_settings.csv), lam = 0.01 only:
                         mu_pos = 0 light, mu_pos in {1, 10, 100} dark; gain_b and the localisation after correction --
                         left panel only (no episodes, no behaviour)
Delta b of episodes = b_step1 under intervention - b_step1 without (the d_ columns). Full correction = mean over the
four agents of -b_step1 without intervention (the reference's b is 0 by construction). Star (oracle colour): the target,
full correction with localisation 1 (left) and the oracle's directed share at k = 1 in episodes (mean of 4, d4_episodes)
(right). Grey dashed (left): the exposed agents' zone localisation without intervention (mean of 4). Blue dashed
(right): the references' directed share (E6, mean of 4); grey dashed: chance 2/5. Vermilion: exposed, no intervention.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_v2 import common as C, palette as P  # noqa: E402

FIG_ID = "E7g_correction_fails"
M = ("results", "E7_latent_manipulation")
V1 = (*M, "dev_e4_cpu", "analysis", "checks.csv")
V2 = (*M, "dev_e4_cpu_v2", "analysis", "checks.csv")
SWEEP = (*M, "dev_e4_cpu_v2", "analysis", "constrained_settings.csv")
D4 = (*M, "dev_e4_cpu_diag", "analysis", "d4_episodes.csv")
E6 = ("results", "E6_phenotypes_latent", "dev_e4_cpu_v2", "analysis", "points.csv")
LIGHT = "#B98CE8"  # the correction's violet, lighter (unconstrained)
SWEEP_LAM = 0.01  # the sweep's settings shown: lam = 0.01 with mu_pos = 0 (unconstrained) vs 1, 10, 100 (constrained)
UNCONSTRAINED, CONSTRAINED =("direction", "stepwise"), ("constrained_direction", "constrained_stepwise")


def episode_points(rows, modes):
    sel = [r for r in rows if r["image"] == "training" and r["mode"] in modes]
    return {k: np.array([C.num(r[c]) for r in sel]) for k, c in
            (("db", "d_b_step1"), ("loc", "loc_template_zone"), ("dir", "directed_share"))}, sel


def baselines(rows):
    """Per agent, the no-intervention values, recovered as value - d_value from the training-image rows."""
    out = {}
    for r in rows:
        if r["image"] == "training" and r["agent"] not in out:
            out[r["agent"]] = {"b": C.num(r["b_step1"]) - C.num(r["d_b_step1"]),
                               "loc": C.num(r["loc_template_zone"]) - C.num(r["d_loc_template_zone"]),
                               "dir": C.num(r["directed_share"]) - C.num(r["d_directed_share"])}
    return out


def targets():
    v1 = C.read_csv(C.path(*V1))
    base = baselines(v1)
    oracle = np.mean([C.num(r["directed_share"]) for r in C.read_csv(C.path(*D4))
                      if r["model"] == "exposed" and r["k"] == "1"])
    ref = np.mean([C.num(r["directed_share"]) for r in C.read_csv(C.path(*E6))
                   if r["image"] == "training" and r["role"] == "reference"])
    return {"full_b": -np.mean([v["b"] for v in base.values()]), "loc0": np.mean([v["loc"] for v in base.values()]),
            "dir0": np.mean([v["dir"] for v in base.values()]), "oracle": oracle, "ref": ref, "base": base}


def decorate(axes, t, xmax, xlabel):
    (al, ar) = axes
    al.axhline(t["loc0"], color=P.CHANCE, ls=P.CHANCE_LS, lw=2.2, zorder=1)
    ar.axhline(t["ref"], color=P.REFERENCE, ls="--", lw=2.2, zorder=1)
    ar.axhline(0.4, color=P.CHANCE, ls=P.CHANCE_LS, lw=2.2, zorder=1)
    al.set_ylim(-0.03, 1.05)
    ar.set_ylim(0.2, 1.0)
    al.set_ylabel("localisation")
    ar.set_ylabel("directed share")
    for ax in axes:
        ax.set_xlim(-0.08, xmax)
        ax.set_xlabel(xlabel)


def star(ax, x, y):
    ax.scatter([x], [y], marker="*", s=900, color=P.ORACLE, edgecolor="black", linewidth=1.2, zorder=6, clip_on=False)


def main():
    t = targets()
    v1, v2 = C.read_csv(C.path(*V1)), C.read_csv(C.path(*V2))
    sweep = [r for r in C.read_csv(C.path(*SWEEP)) if r["image"] == "training" and C.num(r["lam"]) == SWEEP_LAM]
    fig, (al, ar) = C.full(1, 2, sharex=True)
    for r in sweep:
        con = C.num(r["mu_pos"]) > 0
        al.scatter(C.num(r["gain_b"]), C.num(r["loc_after"]), marker="s", s=70, color=P.CORRECTION if con else LIGHT,
                   alpha=0.8, zorder=3)
    for modes, col in ((UNCONSTRAINED, LIGHT), (CONSTRAINED, P.CORRECTION)):
        pts, _ = episode_points(v1 if modes == UNCONSTRAINED else v2, modes)
        al.scatter(pts["db"], pts["loc"], s=130, color=col, edgecolor="black", linewidth=0.8, zorder=4)
        ar.scatter(pts["db"], pts["dir"], s=130, color=col, edgecolor="black", linewidth=0.8, zorder=4)
    ar.scatter([0], [t["dir0"]], s=170, color=P.EXPOSED, edgecolor="black", linewidth=0.8, zorder=5)
    al.scatter([0], [t["loc0"]], s=170, color=P.EXPOSED, edgecolor="black", linewidth=0.8, zorder=5)
    star(al, t["full_b"], 1.0)
    star(ar, t["full_b"], t["oracle"])
    decorate((al, ar), t, t["full_b"] + 0.25, "Δb (bits)")
    return C.save(fig, FIG_ID)


if __name__ == "__main__":
    main()
