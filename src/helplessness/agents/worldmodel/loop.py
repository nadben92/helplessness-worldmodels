"""Phase 2 scope: fixed-epsilon=1 data collection into a ReplayBuffer,
and the world-model-only training step (reconstruction + KL, spec 3.2)
run against that FIXED data ("entraine sur des donnees fixes a
epsilon=1" -- no live interleaved collect/train loop yet, no agent, no
goal head, no critic: those are Phase 5).

No state label reaches any training function here: world_model_loss and
train_world_model_step take a replay.ModelBatch only (see
docs/e3/DECISIONS.md, adjustment D) -- the type itself has no `s` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.losses import dyn_rep_kl, patch_only_reconstruction_error, weighted_reconstruction_loss
from helplessness.agents.worldmodel.replay import ModelBatch, ReplayBuffer
from helplessness.agents.worldmodel.rssm import RSSM, LatentEstimator


def non_terminal_states(env) -> np.ndarray:
    """Every state that is not terminal (duck-typed: env.terminal_mask()) --
    the support of the uniform start distribution, as E2's own
    NON_TERMINAL_STATES (experiments/E2_online_4zones/sweep.py)."""
    return np.flatnonzero(~np.asarray(env.terminal_mask(), dtype=bool))


def draw_start_state(start_states: np.ndarray, start_rng: np.random.Generator) -> int:
    """One uniform draw among start_states from the dedicated start_state
    stream -- the same draw as E2's loop.py::_draw_start."""
    return int(start_states[start_rng.integers(len(start_states))])


def collect_fixed_policy_data(
    env, replay: ReplayBuffer, n_steps: int, rng: np.random.Generator, start_rng: np.random.Generator,
    epsilon: float = 1.0, start_states: np.ndarray | None = None,
) -> None:
    """Runs n_steps environment transitions under an epsilon-greedy
    policy that is UNIFORM RANDOM whenever it explores (no policy
    exists yet at this phase, so epsilon=1.0 always explores -- E3-0's
    own condition, spec section 6). Never counts a reset as a step.
    Stores every transition into replay, including the true label s
    (log-only -- see replay.py's own docstring).

    Every episode, the first included, starts in a state drawn uniformly
    among the non-terminal states (start_states, default
    non_terminal_states(env)) from the dedicated start_rng stream -- the
    E2 protocol (spec 4.6 and 6; docs/e3/DECISIONS.md, 2026-09-25).
    start_rng is required: no caller can fall back to the fixed start."""
    start_states = non_terminal_states(env) if start_states is None else np.asarray(start_states)
    obs, info = env.reset(start_state=draw_start_state(start_states, start_rng))
    s = info["state"]
    is_first = True
    for _ in range(n_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(env.n_actions))
        else:
            raise NotImplementedError("epsilon < 1 requires a policy, which does not exist before Phase 5")

        obs_next, r, terminated, truncated, info = env.step(a)
        replay.add(x=obs, a=a, r=r, terminal=terminated, truncated=truncated, is_first=is_first, s=s)

        is_first = terminated or truncated
        if is_first:
            obs_next, info = env.reset(start_state=draw_start_state(start_states, start_rng))
        s = info["state"]
        obs = obs_next


def to_chw_float(x, device: torch.device) -> torch.Tensor:
    """(B, T, H, W, C) uint8 (numpy array or torch tensor) -> (B, T, C, H,
    W) float32 in [0, 1] on `device`. The uint8 values cross to the device
    first and are converted there (4x fewer bytes than converting on the
    host -- audit 1.8); x / 255 of an exact integer is correctly rounded
    on every device, so the values are the same bits either way."""
    t = torch.as_tensor(x).to(device).to(torch.float32) / 255.0
    return t.permute(0, 1, 4, 2, 3)


@dataclass(frozen=True)
class EstimatorSchedule:
    """Which LatentEstimator the posterior uses at each training step --
    a training-procedure choice only (docs/e3/DECISIONS.md, 2026-09-24):

      "straight_through": plain straight-through throughout (DreamerV3).
      "gumbel_st": straight-through Gumbel-softmax, temperature annealed
          exponentially from tau_start to tau_end over anneal_steps, then
          held at tau_end.
      "warmup_probs": the decoder receives the posterior PROBABILITIES
          (no sampling) for the first warmup_steps steps, then plain
          straight-through samples.
    """

    kind: str = "straight_through"
    tau_start: float = 1.0
    tau_end: float = 0.1
    anneal_steps: int = 1
    warmup_steps: int = 0

    def __post_init__(self):
        if self.kind not in ("straight_through", "gumbel_st", "warmup_probs"):
            raise ValueError(f"unknown EstimatorSchedule kind {self.kind!r}")

    def at(self, step: int) -> LatentEstimator:
        if self.kind == "gumbel_st":
            frac = min(step, self.anneal_steps) / max(self.anneal_steps, 1)
            return LatentEstimator("gumbel_st", tau=self.tau_start * (self.tau_end / self.tau_start) ** frac)
        if self.kind == "warmup_probs" and step < self.warmup_steps:
            return LatentEstimator("probs")
        return LatentEstimator("straight_through")


class CollapseAfterSwitchError(RuntimeError):
    """Raised when, after the switch to straight-through, the label-free
    batch-pair JS falls back below collapse_threshold_bits on
    collapse_consecutive consecutive evaluations: the posterior has lost
    the separation it had at the switch (docs/e3/DECISIONS.md, 2026-09-24
    -- the late single-step collapse seen at lr 1e-3)."""


class SeparationNotReachedError(RuntimeError):
    """Raised when the posterior has not separated the states by
    switch_max_steps: the run stops instead of switching to
    straight-through silently (docs/e3/DECISIONS.md, 2026-09-24)."""


@dataclass
class SeparationSwitch:
    """Criterion-triggered version of the "warmup_probs" schedule
    (docs/e3/DECISIONS.md, 2026-09-24): the decoder receives the posterior
    PROBABILITIES until the posteriors of random image pairs in the
    current batch are separated, then plain straight-through samples.

    Every `eval_every` steps the caller computes the label-free statistic
    (measures.diagnostics.mean_random_pair_jensen_shannon on the batch's
    posteriors) and passes it to update(). The switch fires once the
    statistic exceeds `threshold_bits` on `consecutive` consecutive
    evaluations taken at step >= `min_steps`; it takes effect from the
    NEXT step (`switch_step`). If it has not fired by `max_steps`,
    check_deadline() raises SeparationNotReachedError -- never a silent
    switch. After the switch, update() keeps watching the same statistic
    and raises CollapseAfterSwitchError once it is below
    `collapse_threshold_bits` on `collapse_consecutive` consecutive
    evaluations. Stateful; one instance per training run.
    """

    max_steps: int
    threshold_bits: float = 3.0
    consecutive: int = 5
    eval_every: int = 50
    min_steps: int = 500
    collapse_threshold_bits: float = 1.0
    collapse_consecutive: int = 5
    switch_step: int | None = None
    history: list[tuple[int, float]] = field(default_factory=list)
    _streak: int = 0
    _collapse_streak: int = 0

    def __post_init__(self):
        first_eval = -(-self.min_steps // self.eval_every) * self.eval_every  # first evaluation at step >= min_steps
        earliest_switch = first_eval + (self.consecutive - 1) * self.eval_every + 1
        if self.max_steps < earliest_switch:
            raise ValueError(
                f"switch_max_steps={self.max_steps} is before the earliest possible switch ({earliest_switch}, "
                f"given min_steps={self.min_steps}, consecutive={self.consecutive}, eval_every={self.eval_every}) "
                "-- this configuration can never switch"
            )

    @property
    def switched(self) -> bool:
        return self.switch_step is not None

    def estimator(self) -> LatentEstimator:
        return LatentEstimator("straight_through") if self.switched else LatentEstimator("probs")

    def is_eval_step(self, step: int) -> bool:
        return step % self.eval_every == 0

    def update(self, step: int, statistic: float) -> None:
        self.history.append((step, float(statistic)))
        if self.switched:
            self._collapse_streak = self._collapse_streak + 1 if statistic < self.collapse_threshold_bits else 0
            if self._collapse_streak >= self.collapse_consecutive:
                raise CollapseAfterSwitchError(
                    f"effondrement apres la bascule / collapse after the switch: the batch-pair JS statistic fell below "
                    f"{self.collapse_threshold_bits} bits on {self.collapse_consecutive} consecutive evaluations after the "
                    f"switch at step {self.switch_step} (last value {float(statistic):.3f} bits at step {step})"
                )
            return
        self._streak = self._streak + 1 if (step >= self.min_steps and statistic > self.threshold_bits) else 0
        if self._streak >= self.consecutive:
            self.switch_step = step + 1

    def check_deadline(self, steps_done: int) -> None:
        if not self.switched and steps_done >= self.max_steps:
            last = self.history[-1][1] if self.history else float("nan")
            raise SeparationNotReachedError(
                f"separation non atteinte / separation not reached: the batch-pair JS statistic did not exceed "
                f"{self.threshold_bits} bits on {self.consecutive} consecutive evaluations (every {self.eval_every} "
                f"steps, from step {self.min_steps}) within switch_max_steps={self.max_steps} (last value {last:.3f} "
                "bits) -- stopping instead of switching to straight-through silently"
            )


PROBS_PHASE_AIDS = ("none", "entropy_penalty", "progressive_samples")


@dataclass(frozen=True)
class ProbsPhaseAid:
    """What happens DURING the probabilities phase of a SeparationSwitch
    (docs/e3/DECISIONS.md, 2026-09-24 -- the lr 1e-4 separation failure).
    Both aids ramp linearly from 0 at step 0 to their maximum at
    ramp_steps (the switch deadline: the only length of the phase known in
    advance) and are removed at the switch:

      "none": the retained procedure (c') unchanged.
      "entropy_penalty": option (ii) -- eta(t) * H(q) is added to the loss
          (losses.mean_categorical_entropy), eta(t) = eta_max * t / ramp_steps.
      "progressive_samples": option (iii) -- the decoder receives
          rssm.partial_straight_through with rho(t) = t / ramp_steps.

    The switch itself stays governed by the JS criterion."""

    kind: str = "none"
    eta_max: float = 0.0
    ramp_steps: int = 1

    def __post_init__(self):
        if self.kind not in PROBS_PHASE_AIDS:
            raise ValueError(f"ProbsPhaseAid kind must be one of {PROBS_PHASE_AIDS}, got {self.kind!r}")
        if self.ramp_steps <= 0:
            raise ValueError(f"ramp_steps must be > 0, got {self.ramp_steps}")

    def _ramp(self, step: int) -> float:
        return min(1.0, step / self.ramp_steps)

    def entropy_weight(self, step: int, switched: bool) -> float:
        if switched or self.kind != "entropy_penalty":
            return 0.0
        return self.eta_max * self._ramp(step)

    def estimator(self, step: int, switch: "SeparationSwitch") -> LatentEstimator:
        """The switch's own estimator, except during the probabilities
        phase of "progressive_samples"."""
        if switch.switched or self.kind != "progressive_samples":
            return switch.estimator()
        return LatentEstimator("partial_straight_through", rho=self._ramp(step))


def phase_learning_rate(step: int, switch_step: int | None, lr: float, lr_after_switch: float | None) -> float:
    """Two-phase learning rate (docs/e3/DECISIONS.md, 2026-09-24): lr until
    the EFFECTIVE switch, lr_after_switch from switch_step on (None: lr
    throughout)."""
    if lr_after_switch is None or switch_step is None or step < switch_step:
        return lr
    return lr_after_switch


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> bool:
    """Sets lr on every parameter group; True if it changed."""
    changed = False
    for group in optimizer.param_groups:
        if group["lr"] != lr:
            group["lr"] = lr
            changed = True
    return changed


def module_grad_norms(modules: dict[str, torch.nn.Module]) -> dict[str, float]:
    """L2 norm of the current .grad of each named module's parameters (0.0
    for a module none of whose parameters has a gradient), read-only --
    never modifies the gradients."""
    out = {}
    for name, module in modules.items():
        grads = [p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]
        out[name] = float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0
    return out


def rep_beta(step: int, switch_step: int | None, rep_warmup_steps: int, beta_rep: float) -> float:
    """KL weight of the representation term (docs/e3/DECISIONS.md,
    2026-09-24): 0 until the estimator switch, then a linear ramp to
    beta_rep over rep_warmup_steps, then held. beta_dyn is NOT scheduled:
    KL(sg(q) || p) only trains the prior, never pulls the posterior, so it
    runs at full weight from step 0 while the posterior separates."""
    if switch_step is None or step < switch_step:
        return 0.0
    if rep_warmup_steps <= 0:
        return beta_rep
    return beta_rep * min(1.0, (step - switch_step) / rep_warmup_steps)


@dataclass(frozen=True)
class WorldModelLossConfig:
    k: int = 6
    w_patch: float = 10.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    free_bits: float = 0.0
    free_bits_mode: str = "per_step"


def unroll_observation(
    rssm: RSSM, x_seq: torch.Tensor, a_seq: torch.Tensor, is_first_seq: torch.Tensor, estimator: LatentEstimator | None = None,
    generator: torch.Generator | None = None, decode: bool = True,
) -> dict[str, torch.Tensor]:
    """x_seq: (B, T, C, H, W) float in [0,1]. a_seq: (B, T) int action
    indices (a_seq[:, t] is the action taken AT step t, i.e. a_{t-1} for
    the transition INTO step t is a_seq[:, t-1] -- see the shift below).
    is_first_seq: (B, T) bool. Runs RSSM.observation_step across the
    window, returning stacked h, z, prior_probs, post_probs, decoded
    images -- each (B, T, ...); decode=False skips the decoder (measures
    that only need the latents; no "decoded" key then).

    The encoder (a function of x_t alone) runs once on the B*T images
    before the recurrence, and the decoder (a function of (h_t, z_t)) once
    after it: the same computation as encoding/decoding step by step, with
    two large calls instead of 2T small ones (audit 7.1 -- on a GPU the
    step-by-step version is bound by kernel launches). The latent draws
    happen in the same order, so a given generator gives the same samples;
    only the floating-point reduction order of the convolutions can
    differ from the step-by-step version."""
    batch, T = x_seq.shape[0], x_seq.shape[1]
    device = x_seq.device
    n_actions = rssm.cfg.n_actions
    a_onehot_seq = F.one_hot(a_seq.clamp(min=0), num_classes=n_actions).to(x_seq.dtype)
    embeds = rssm.encoder(x_seq.reshape(batch * T, *x_seq.shape[2:])).reshape(batch, T, -1)

    h, z = rssm.initial_state(batch, device)
    a_prev = torch.zeros(batch, n_actions, device=device, dtype=x_seq.dtype)

    hs, zs, priors, posts = [], [], [], []
    for t in range(T):
        out = rssm.observation_step(h, z, a_prev, None, is_first_seq[:, t], estimator=estimator, generator=generator,
                                    embed=embeds[:, t])
        h, z = out["h"], out["z"]
        hs.append(h)
        zs.append(z)
        priors.append(out["prior_probs"])
        posts.append(out["post_probs"])
        a_prev = a_onehot_seq[:, t]

    result = {
        "h": torch.stack(hs, dim=1),
        "z": torch.stack(zs, dim=1),
        "prior_probs": torch.stack(priors, dim=1),
        "post_probs": torch.stack(posts, dim=1),
    }
    if decode:
        decoded = rssm.decode(result["h"].reshape(batch * T, -1), result["z"].reshape(batch * T, *result["z"].shape[2:]))
        result["decoded"] = decoded.reshape(batch, T, *decoded.shape[1:])
    return result


def world_model_loss(
    rssm: RSSM, model_batch: ModelBatch, background: torch.Tensor, cfg: WorldModelLossConfig, estimator: LatentEstimator | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    return _world_model_loss_and_unroll(rssm, model_batch, background, cfg, estimator, generator)[0]


def _world_model_loss_and_unroll(
    rssm: RSSM, model_batch: ModelBatch, background: torch.Tensor, cfg: WorldModelLossConfig, estimator: LatentEstimator | None = None,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """model_batch: replay.ModelBatch ONLY -- no label ever passed in.
    background: (C, H, W) float in [0, 1], the run's single fixed
    render_background() image (constant across the whole run, see
    module docstring)."""
    device = next(rssm.parameters()).device
    x_seq = to_chw_float(model_batch.x, device)
    a_seq = torch.as_tensor(model_batch.a, dtype=torch.long, device=device)
    is_first_seq = torch.as_tensor(model_batch.is_first, dtype=torch.bool, device=device)

    out = unroll_observation(rssm, x_seq, a_seq, is_first_seq, estimator=estimator, generator=generator)

    B, T = x_seq.shape[0], x_seq.shape[1]
    bg = background.to(device).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
    x_flat = x_seq.reshape(B * T, *x_seq.shape[2:])
    decoded_flat = out["decoded"].reshape(B * T, *out["decoded"].shape[2:])
    bg_flat = bg.reshape(B * T, *bg.shape[2:])
    recon_loss = weighted_reconstruction_loss(x_flat, decoded_flat, bg_flat, k=cfg.k, w_patch=cfg.w_patch)
    # Diagnostic only (never added to the training loss): reconstruction
    # error restricted to the masked patch alone, reused by
    # experiments/E3_worldmodel_4zones/run_mini.py's report to separate
    # "error on the patch" from "error overall".
    recon_patch = patch_only_reconstruction_error(x_flat, decoded_flat, bg_flat, k=cfg.k)

    kl = dyn_rep_kl(
        out["prior_probs"].reshape(B * T, *out["prior_probs"].shape[2:]),
        out["post_probs"].reshape(B * T, *out["post_probs"].shape[2:]),
        beta_dyn=cfg.beta_dyn,
        beta_rep=cfg.beta_rep,
        free_bits=cfg.free_bits,
        free_bits_mode=cfg.free_bits_mode,
    )

    total = recon_loss + kl["total"]
    return {
        "total": total,
        "recon": recon_loss,
        "recon_patch": recon_patch,
        "kl_dyn": kl["dyn"],
        "kl_rep": kl["rep"],
        "kl_dyn_raw_mean": kl["dyn_raw_mean"],
        "kl_rep_raw_mean": kl["rep_raw_mean"],
    }, out


def train_world_model_step(
    rssm: RSSM,
    optimizer: torch.optim.Optimizer,
    model_batch: ModelBatch,
    background: torch.Tensor,
    cfg: WorldModelLossConfig,
    clip_norm: float | None = 1000.0,
    estimator: LatentEstimator | None = None,
    return_posteriors: bool = False,
    generator: torch.Generator | None = None,
    sync: bool = True,
) -> dict:
    """One Adam step (lr/eps set on the optimizer by the caller, spec
    3.3: lr=1e-4, eps=1e-8) with gradient-norm clipping (clip_norm=None:
    no clipping -- the global norm is still measured and returned as
    "grad_norm", always the pre-clip value). Returns
    detached scalar losses for logging; with return_posteriors=True, also
    "post_probs": this batch's posterior probabilities as a numpy array
    (B*T, n_vars, n_classes), computed by the same forward pass (for the
    label-free separation statistic -- no label involved). generator: the
    dedicated stream for the latent draws of this step (None: the global
    torch RNG; see rssm.sample_categorical_index). sync=False returns the
    scalars as detached 0-d tensors on the model's device instead of
    Python floats: each float() forces a device synchronisation, so a
    caller that logs every step converts them in batches (audit 7.2;
    same values)."""
    optimizer.zero_grad()
    losses, out = _world_model_loss_and_unroll(rssm, model_batch, background, cfg, estimator=estimator, generator=generator)
    losses["total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(rssm.parameters(), max_norm=float("inf") if clip_norm is None else clip_norm)
    optimizer.step()
    if sync:
        result = {k: float(v.detach()) for k, v in losses.items()} | {"grad_norm": float(grad_norm)}
    else:
        result = {k: v.detach() for k, v in losses.items()} | {"grad_norm": grad_norm.detach()}
    if return_posteriors:
        post = out["post_probs"]
        result["post_probs"] = post.reshape(-1, *post.shape[2:]).detach().cpu().numpy()
    return result
