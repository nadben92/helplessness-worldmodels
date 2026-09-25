"""Phase 2 throughput profiling (spec section 3.5 / phase table P2):
environment steps/second and gradient steps/second, at the spec's
default full-scale config (base_depth=32, embed_dim=h_dim=256,
n_vars=n_classes=8, batch=16, seq_len=64, k=6 -> 72x72 images).

CPU-only on this machine (docs/e3/DECISIONS.md, point 7) -- these
numbers are INDICATIVE ONLY, not the calibration-grade profile Phase 4
will need; that gets redone on the actual GPU training machine. Prints a
plain-text report; no results/ artifact is written (this is a Phase 2
sanity/profiling script, not a run).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\profile_throughput.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from env import build_pixel_env, build_layout, goal_state, quadrant_assignment_for_rotation, rotation_for_seed_index, ZONE_CATEGORIES  # noqa: E402

from helplessness.agents.worldmodel.loop import WorldModelLossConfig, collect_fixed_policy_data, train_world_model_step
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, resolve_device
from helplessness.envs.appearance.theme_sampling import build_themes
from helplessness.envs.observation.allocentric_pixel_backend import render_background


def profile(n_env_steps: int, n_train_steps: int, batch_size: int, seq_len: int, k: int, device_str: str) -> None:
    device = resolve_device(device_str)
    print(f"device: {device}")

    rotation = rotation_for_seed_index(0)
    assignment = quadrant_assignment_for_rotation(rotation)
    layout = build_layout(assignment)
    themes = build_themes(ZONE_CATEGORIES, goal_state=goal_state(), rng=np.random.default_rng(0), min_dist=30.0)
    env = build_pixel_env(layout, themes, goal_reward=50.0, seed=0, k=k)
    background = render_background(layout, themes, k=k)
    background_t = torch.as_tensor(background, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0

    image_side = 12 * k
    buf = ReplayBuffer(capacity=max(n_env_steps, batch_size * seq_len + 1), image_shape=(image_side, image_side, 3))
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    collect_fixed_policy_data(env, buf, n_steps=n_env_steps, rng=rng, start_rng=np.random.default_rng([1, 7]), epsilon=1.0)
    env_elapsed = time.perf_counter() - t0
    env_sps = n_env_steps / env_elapsed
    print(f"env: {n_env_steps} steps in {env_elapsed:.2f}s -> {env_sps:.1f} steps/s")

    cfg = RSSMConfig(image_size=(image_side, image_side), base_depth=32, embed_dim=256, h_dim=256, n_vars=8, n_classes=8, n_actions=5)
    rssm = RSSM(cfg).to(device)
    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-4, eps=1e-8)
    loss_cfg = WorldModelLossConfig(k=k, w_patch=10.0)

    # warm-up (excluded from timing: first call pays one-time CPU cache/allocator cost)
    model_batch, _ = buf.sample(batch_size=batch_size, seq_len=seq_len, rng=rng)
    train_world_model_step(rssm, optimizer, model_batch, background_t, loss_cfg)

    t0 = time.perf_counter()
    for _ in range(n_train_steps):
        model_batch, _ = buf.sample(batch_size=batch_size, seq_len=seq_len, rng=rng)
        train_world_model_step(rssm, optimizer, model_batch, background_t, loss_cfg)
    train_elapsed = time.perf_counter() - t0
    grad_sps = n_train_steps / train_elapsed
    env_steps_per_grad_step = batch_size * seq_len
    print(f"train: {n_train_steps} gradient steps in {train_elapsed:.2f}s -> {grad_sps:.3f} steps/s "
          f"(batch={batch_size}, seq_len={seq_len}, {env_steps_per_grad_step} transitions/step)")

    print()
    print("These numbers are CPU-indicative only (docs/e3/DECISIONS.md, point 7) -- "
          "Phase 4's cost/replay_ratio estimate is redone on the GPU training machine.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-env-steps", type=int, default=500)
    parser.add_argument("--n-train-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    profile(args.n_env_steps, args.n_train_steps, args.batch_size, args.seq_len, args.k, args.device)


if __name__ == "__main__":
    main()
