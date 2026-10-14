"""E3's run pipeline: checkpoints and run directories -- the checkpoint grid, atomic weight saves, random-generator
states, the failure diagnostic, the run directory, the RSSM built from a config, and reloading a finished run's
data. Moved verbatim from experiments/E3_worldmodel_4zones/run_mini.py on 2026-10-14 (run_mini.py re-exports every
name); the module-level names those functions read are imports here.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from helplessness.agents.worldmodel.loop import CollapseAfterSwitchError, SeparationNotReachedError, SeparationSwitch
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig
from helplessness.pipeline.measure import SLOT_FAILURE
from helplessness.pipeline.runtime import _jsonable, RunLog
from helplessness.pipeline.world import (
    build_world,
    EXPERIMENT_CONFIG_PATH,
    EXPERIMENT_ID,
    GRID_SHAPE,
    N_ACTIONS,
    N_STATES,
    PAIR_ROLE,
    run_identity,
    themes_from_dict,
    World,
    ZONE_PRESET,
)
from helplessness.probing import probe as e3_probe
from helplessness.runs import make_run_dir, make_run_name
from helplessness.streams import worldmodel as e3_rng


def e2_grid_env_steps() -> list[int]:
    """E2's 16-point checkpoint grid in environment steps
    (experiments/E3_worldmodel_4zones/config.yaml, = E2's sweep.CHECKPOINTS)."""
    with open(EXPERIMENT_CONFIG_PATH, encoding="utf-8") as f:
        return [int(c) for c in yaml.safe_load(f)["budget"]["checkpoints"]]


def checkpoint_steps(cfg: dict, planned_steps: int) -> list[int]:
    """Gradient steps at which the weights are saved (and, with
    measure.diagnostics_at_checkpoints, measured). checkpoints.grid "e2":
    E2's grid points c <= the run's environment steps, mapped to gradient
    step round(c * replay_ratio) -- in E3-0 all the data is collected
    first, so checkpoint c is "the training a run at replay_ratio would
    have done after c environment steps" (docs/e3/DECISIONS.md,
    2026-09-28). A list: explicit gradient steps. null: none. The final
    step is always saved (weights_final.pt), not listed here."""
    grid = (cfg.get("checkpoints") or {}).get("grid")
    if grid is None:
        return []
    if grid == "e2":
        total_env = cfg["budget"]["total_steps"] + cfg["budget"]["prefill_steps"]
        steps = {int(round(c * cfg["train"]["replay_ratio"])) for c in e2_grid_env_steps() if c <= total_env}
    else:
        steps = {int(g) for g in grid}
    return sorted(g for g in steps if 0 < g < planned_steps)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_weights(run_dir: Path, rssm: RSSM, step: int, name: str | None = None, **extra) -> Path:
    path = run_dir / "checkpoints" / (name or f"weights_step{step:08d}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save({"rssm": rssm.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg), "step": step, **extra}, path)
    return path


def _generator_state(g) -> object:
    return g.bit_generator.state if isinstance(g, np.random.Generator) else g.get_state()


def _set_generator_state(g, state) -> None:
    if isinstance(g, np.random.Generator):
        g.bit_generator.state = state
    else:
        g.set_state(state)


def save_failure_diagnostic(fail_dir: Path, cfg: dict, seed: int, rssm: RSSM, optimizer: torch.optim.Optimizer,
                            templates_t: torch.Tensor, step: int, switch: SeparationSwitch,
                            error: SeparationNotReachedError | CollapseAfterSwitchError, meta: dict, log: RunLog,
                            condition: str = "experimental") -> Path:
    """Called when SeparationNotReachedError or CollapseAfterSwitchError is
    raised (docs/e3/DECISIONS.md, 2026-09-24): saves the checkpoint (model +
    optimizer state) and the latent-usage diagnostic (probe.latent_usage_diagnostic:
    template accuracy with the decoder fed probabilities / mode / hard
    sample, posterior entropy, I(z;s), logit norms) into the run's failure/."""
    kind = "collapse_after_switch" if isinstance(error, CollapseAfterSwitchError) else "separation_not_reached"
    fail_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"rssm": rssm.state_dict(), "optimizer": optimizer.state_dict(), "rssm_config": dataclasses.asdict(rssm.cfg),
                "step": step}, fail_dir / "checkpoint.pt")
    rssm.eval()
    diag = e3_probe.latent_usage_diagnostic(rssm, templates_t, M=cfg["measure"]["iz_mc_samples_final"],
                                            rng=e3_rng.measure_streams(seed, SLOT_FAILURE, next(rssm.parameters()).device, condition)[0])
    record = {"seed": seed, "condition": condition, "steps_done": step, "kind": kind, "error": str(error), **meta, "config": cfg,
              "switch_history": [list(map(float, h)) for h in switch.history], "diagnostic": diag}
    (fail_dir / "diagnostic.json").write_text(json.dumps(_jsonable(record), indent=1), encoding="utf-8")
    log(f"FAILURE DIAGNOSTIC (saved to {fail_dir}):")
    for key, value in diag.items():
        log(f"    {key:22s} {value:.4f}")
    return fail_dir


def make_seed_run_dir(out_root: Path, config_name: str, seed: int, condition: str = "experimental") -> Path:
    name = make_run_name(GRID_SHAPE, ZONE_PRESET, PAIR_ROLE[condition], f"worldmodel-{config_name}", seed)
    return make_run_dir(out_root, EXPERIMENT_ID, name)


def build_rssm(cfg: dict) -> RSSM:
    image_side = 12 * cfg["render"]["k"]
    return RSSM(RSSMConfig(
        image_size=(image_side, image_side), base_depth=cfg["model"]["base_depth"], embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"], n_vars=cfg["model"]["n_vars"], n_classes=cfg["model"]["n_classes"],
        n_actions=N_ACTIONS, unimix=cfg["model"]["unimix"],
    ))


def world_record(world: World, cfg: dict) -> dict:
    return {"seed": world.seed, "condition": world.condition, **run_identity(world.seed, world.themes, world.condition),
            "zone_kinds": world.zone_kinds, "env_dynamics_seed": world.env_seed, "render": cfg["render"],
            "templates_sha256": world.templates_sha256(), "n_states": N_STATES, "n_actions": N_ACTIONS}


def load_run_data(run_dir: Path, cfg: dict, seed: int, condition: str) -> tuple[World, ReplayBuffer, np.ndarray, np.ndarray]:
    """The world, replay, N and episode starts of an existing run directory
    (resume, recompute): the world is rebuilt from the seed and condition
    with the SAVED themes, and must reproduce the saved hash of the
    pre-rendered images; the replay is rebuilt from replay.npz."""
    saved = json.loads((run_dir / "world.json").read_text(encoding="utf-8"))
    if (saved["seed"], saved["condition"]) != (seed, condition):
        raise ValueError(f"world.json is for seed {saved['seed']} / {saved['condition']}, not {seed} / {condition}")
    world = build_world(cfg, seed, condition, themes=themes_from_dict(saved["themes"]))
    if world.templates_sha256() != saved["templates_sha256"]:
        raise ValueError("the rebuilt pre-rendered images differ from the run's (templates_sha256): rendering changed since the run")
    data = np.load(run_dir / "replay.npz")
    replay = ReplayBuffer.from_compact({key: data[key] for key in ("s", "a", "r", "terminal", "truncated", "is_first")},
                                       world.templates_np)
    return world, replay, data["N"], data["episode_starts"]
