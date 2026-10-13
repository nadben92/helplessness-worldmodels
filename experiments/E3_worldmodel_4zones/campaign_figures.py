"""Figures of the E3 campaign's analysis (docs/e3/DECISIONS.md, 2026-10-13 (31), G). Read-only: everything is drawn from
the analyses' outputs (campaign_analysis.analyse, campaign_learning.analyse_learning, rsa_latents), nothing recomputed.

  maps_<checkpoint>.png     I_s by state on the 12 x 12 torus, mean over the seeds: environment, reference, difference
  curves_b_alpha.png        b by zone (I_s) and alpha_env - alpha_ref against the checkpoint, mean over the seeds (95 %
                            Student-t band) and each seed
  curves_learning.png       per zone and condition, mean over the seeds: C-b's L1 of the world model, the twin's L1, the
                            prior's mass on the true successor, the agent's Delta; success over each interval
  rsa_<run>_env<c>.png      the RSA's 2D MDS of the states' mean latents (h and z), coloured by zone, theme, controllability
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GRID = (12, 12)
SURFACE, INK, INK2, MUTED, GRIDLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# fixed zone order and colours (dataviz reference palette, categorical slots 1-5, validated; markers as secondary encoding)
ZONE_ORDER = ("uncontrollable_deterministic", "controllable_noisy", "uncontrollable_noisy", "controllable_deterministic", "corridor")
ZONE_STYLE = {"uncontrollable_deterministic": ("UD", "#2a78d6", "o"), "controllable_noisy": ("CN", "#eb6834", "s"),
              "uncontrollable_noisy": ("UN", "#1baf7a", "^"), "controllable_deterministic": ("CD", "#eda100", "D"),
              "corridor": ("corridor", "#e87ba4", "v")}
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
SEQUENTIAL = ("#cde2fb", "#0d366b")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRIDLINE,
                         "grid.linewidth": 0.6, "font.size": 9, "lines.linewidth": 2, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap

    return (LinearSegmentedColormap.from_list("seq", SEQUENTIAL), LinearSegmentedColormap.from_list("div", DIVERGING))


def _zone_outline(ax, zone_of_state):
    """Zone boundaries drawn as thin lines between cells of different zones."""
    z = np.asarray(zone_of_state).reshape(GRID)
    for r in range(GRID[0]):
        for c in range(GRID[1]):
            if c + 1 < GRID[1] and z[r, c] != z[r, c + 1]:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5], color=INK2, lw=0.8)
            if r + 1 < GRID[0] and z[r, c] != z[r + 1, c]:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5], color=INK2, lw=0.8)


def maps(propagation: dict, zone_of_state, I_by_run: dict, checkpoints: list[int], out: Path) -> list[Path]:
    """I_by_run: {(seed, condition): {checkpoint: I_s per state}}. One figure per checkpoint, mean over the seeds."""
    plt = _plt()
    seq, div = _cmaps()
    paths = []
    for c in checkpoints:
        env = [v[c] for (s, cond), v in I_by_run.items() if cond == "experimental" and c in v]
        ref = [v[c] for (s, cond), v in I_by_run.items() if cond == "reference" and c in v]
        if not env or not ref:
            continue
        e, r = np.nanmean(env, axis=0), np.nanmean(ref, axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
        vmax = float(np.nanmax([e, r]))
        dmax = float(np.nanmax(np.abs(e - r))) if np.isfinite(e - r).any() else 1e-6
        dmax = dmax or 1e-6
        for ax, data, title, cmap, lim in ((axes[0], e, f"environment (n = {len(env)})", seq, (0, vmax)),
                                           (axes[1], r, f"reference (n = {len(ref)})", seq, (0, vmax)),
                                           (axes[2], e - r, "difference b(s) = env - ref", div, (-dmax, dmax))):
            im = ax.imshow(data.reshape(GRID), cmap=cmap, vmin=lim[0], vmax=lim[1])
            _zone_outline(ax, zone_of_state)
            ax.set_title(title, fontsize=9, color=INK)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="bit")
        fig.suptitle(f"I_s by state at {c:,} environment steps (mean over seeds; lines: zone boundaries)", fontsize=10)
        fig.tight_layout()
        p = out / f"maps_{c:06d}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def _series_panel(ax, over: dict, key: str, zones, ylabel: str):
    cs = sorted(over)
    for z in zones:
        label, color, marker = ZONE_STYLE[z]
        m = np.array([over[c][z]["mean"] for c in cs], dtype=float)
        lo = np.array([over[c][z]["ci95"][0] for c in cs], dtype=float)
        hi = np.array([over[c][z]["ci95"][1] for c in cs], dtype=float)
        if not np.isfinite(m).any():
            continue
        for i in range(max((len(over[c][z]["values"]) for c in cs), default=0)):
            v = [over[c][z]["values"][i] if i < len(over[c][z]["values"]) else np.nan for c in cs]
            ax.plot(cs, v, color=color, lw=0.6, alpha=0.45)
        ax.fill_between(cs, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(cs, m, color=color, marker=marker, ms=5, label=label)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)


def curves_b_alpha(over_seeds: dict, out: Path) -> Path | None:
    if not over_seeds.get("common_checkpoints"):
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _series_panel(axes[0], over_seeds["b_I_s"], "b", ZONE_ORDER, "b = I_s(env) - I_s(ref)  [bit]")
    axes[0].set_title("b by zone (mean over seeds, 95 % t band; thin: each seed)", fontsize=9)
    _series_panel(axes[1], over_seeds["alpha_difference"], "alpha", [z for z in ZONE_ORDER if z in next(iter(over_seeds["alpha_difference"].values()))],
                  "alpha_env - alpha_ref")
    axes[1].set_title("action-independent weight, env - ref (defined zones only)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=len(labels), loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = out / "curves_b_alpha.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def curves_learning(learning: list[dict], out: Path) -> Path:
    """Means over the seeds of each condition (thin lines: runs), per zone."""
    plt = _plt()
    quantities = (("calibration", "C-b: L1 of the world model (N >= 100)"), ("twin_l1", "twin's L1 vs P"),
                  ("mass_on_true_successor", "prior mass on the true successor"), ("agent_delta_by_zone", "agent's Delta under P"))
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
    for row, cond in enumerate(("experimental", "reference")):
        runs = [r for r in learning if r["condition"] == cond]
        for col, (key, title) in enumerate(quantities):
            ax = axes[row, col]
            for z in ZONE_ORDER:
                label, color, marker = ZONE_STYLE[z]
                per_run = []
                for r in runs:
                    xs, ys = [], []
                    for c in r["checkpoints"]:
                        v = c.get(key)
                        if v is None:
                            continue
                        y = v["c_b"][z]["l1_wm"] if key == "calibration" else v.get(z)
                        if y is not None and np.isfinite(y):
                            xs.append(c["env_step"]); ys.append(y)
                    if xs:
                        ax.plot(xs, ys, color=color, lw=0.6, alpha=0.45)
                        per_run.append(dict(zip(xs, ys)))
                if per_run:
                    cs = sorted(set.intersection(*[set(p) for p in per_run]))
                    ax.plot(cs, [np.mean([p[c] for p in per_run]) for c in cs], color=color, marker=marker, ms=4, label=label)
            ax.set_xscale("log")
            ax.set_title(f"{cond}: {title}", fontsize=8)
            if key == "agent_delta_by_zone":
                ax.set_yscale("symlog", linthresh=1.0)
        ax = axes[row, 4]
        for k, r in enumerate(runs):
            xs = [c["env_step"] for c in r["checkpoints"]]
            ys = [c["interval"]["rate_ci95"][0] for c in r["checkpoints"]]
            ax.plot(xs, ys, color=INK2, lw=1, marker="osD^v<>p"[k % 8], ms=3, alpha=0.7, label=f"seed {r['seed']}")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{cond}: success over each interval", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("environment steps")
    fig.tight_layout()
    p = out / "curves_learning.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def rsa_mds(npz_path: Path, summary: dict, out: Path) -> Path:
    """MDS of the states' latents (h and z) coloured by zone, theme and controllability."""
    plt = _plt()
    data = np.load(npz_path)
    attrs = summary["attributes"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for row, lat in enumerate(("h", "z")):
        xy = data[f"state_{lat}_mds"]
        for col, key in enumerate(("zone", "theme", "controllability")):
            ax = axes[row, col]
            labels = np.asarray(attrs[key])
            uniq = list(dict.fromkeys(labels))
            for k, lab in enumerate(uniq):
                m = (labels == lab) & np.isfinite(xy[:, 0])
                if key == "zone" and lab in ZONE_STYLE:
                    name, color, marker = ZONE_STYLE[lab]
                elif key == "theme":
                    zone_of = next(z for z, t in zip(attrs["zone"], attrs["theme"]) if t == lab)
                    name, color, marker = lab, "#" + "".join(f"{int(v):02x}" for v in lab[3:].split(",")), ZONE_STYLE.get(zone_of, ("", "", "o"))[2]
                else:
                    name, color, marker = lab, ("#2a78d6", "#eb6834")[k % 2], ("o", "s")[k % 2]
                ax.scatter(xy[m, 0], xy[m, 1], s=22, c=color, marker=marker, edgecolors=SURFACE, linewidths=0.8, label=name)
            ax.set_title(f"{lat}: coloured by {key}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(frameon=False, fontsize=6, loc="best")
    reg = summary["states"]["latents"]
    fig.suptitle(f"{summary['condition']} seed {summary['seed']} at {summary['env_step']:,}: classical MDS of the states' mean latents "
                 f"(R^2 h {reg['h']['regression']['r2']:.2f}, z {reg['z']['regression']['r2']:.2f})", fontsize=10)
    fig.tight_layout()
    p = out / f"rsa_{summary['condition']}_seed{summary['seed']}_env{summary['env_step']:06d}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p
