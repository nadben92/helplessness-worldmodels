"""Per-state I(a;s'|s) of E4's exposed and reference models on the TRAINING image, at the final checkpoint (frozen weights).
Existing estimator only (probing.fixed_trajectory: fixed_trajectory_context + step_controllability, the E4 walk probe),
applied to every step of the fixed walks of the four slots (corridor, path and zone steps; burn-in steps dropped) and
averaged per state.
  python slides/out_e4_heat/measure_e4_heat.py --config p1-b-d1 --runs-root /c/nrw/e6all/E4_generalization_gradient"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from helplessness.envs.layouts import two_zones  # noqa: E402
from helplessness.pipeline import two_zones_setup as setup  # noqa: E402
from helplessness.probing.fixed_trajectory import fixed_trajectory_context, step_controllability  # noqa: E402
from helplessness.probing.frozen_episodes import load_frozen_agent  # noqa: E402

BURN_IN = 8  # steps dropped at the start of each walk (the filter starts from the learned initial state)


def measure(model, world, succ, cfg, conf, n_walks, draws, device):
    N = two_zones.N_STATES
    sum_I, sum_m, cnt = np.zeros(N), np.zeros(N), np.zeros(N)
    thr = cfg["measure"]["kernel_margin_threshold"]
    for slot in two_zones.QUADRANT_ORDER:
        walks = setup.walks_for_slot(cfg, slot, n_walks)
        g = torch.Generator(device=device)
        g.manual_seed(setup.measurement_seed(cfg, conf, 0, slot, "probe"))
        ctx = fixed_trajectory_context(model.rssm, world.templates_np, walks, generator=g, chunk=cfg["e4"]["walks"]["chunk"],
                                       max_images=cfg["measure"]["max_images_per_forward"])
        mask = (walks["phase"] >= 0) & (walks["steps_since_start"] >= BURN_IN)
        mask_t = torch.as_tensor(mask, device=device)
        states = walks["states"][mask]
        res = step_controllability(model.rssm, ctx["h"][mask_t], ctx["z"][mask_t], world.templates_t.to(device), draws, thr,
                                   generator=g, true_successor=succ[states])
        mass = np.nanmean(res["mass_true"], axis=1)
        ok = np.isfinite(res["I"])
        np.add.at(sum_I, states[ok], res["I"][ok]); np.add.at(sum_m, states[ok], mass[ok]); np.add.at(cnt, states[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sum_I / cnt, sum_m / cnt, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--n-walks", type=int, default=None)
    ap.add_argument("--draws", type=int, default=None)
    a = ap.parse_args()
    cfg = setup.load_config(REPO / "configs/e4/e4.yaml")
    conf = {**setup.configuration(cfg, a.config)}
    n_walks = a.n_walks or cfg["e4"]["walks"]["n_walks"]
    draws = a.draws or cfg["e4"]["probe"]["draws"]
    device = torch.device("cpu")
    torch.set_num_threads(4)
    world = setup.probe_world(cfg, conf)
    succ = setup.successors(world)
    out = {"n_walks": n_walks, "draws": draws}
    for role in ("experimental", "control"):
        pat = f"{a.runs_root}/grid12x12__2zones-{a.config}__pair-{role}__worldmodel-agent-e4__seed{conf['seed']:02d}__*/checkpoints/agent_env300000.pt"
        (path,) = glob.glob(pat)
        t0 = time.perf_counter()
        model = load_frozen_agent(path, device)
        I, m, c = measure(model, world, succ, cfg, conf, n_walks, draws, device)
        out[f"I_{role}"], out[f"mass_{role}"], out[f"count_{role}"], out[f"path_{role}"] = I, m, c, path
        print(role, f"{time.perf_counter() - t0:.0f}s", "states measured", int(np.isfinite(I).sum()), flush=True)
    np.savez(Path(__file__).parent / f"data_{a.config}.npz", **out)


if __name__ == "__main__":
    main()
