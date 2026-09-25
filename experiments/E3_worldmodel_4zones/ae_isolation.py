"""Autoencoder isolation test (docs/e3/DECISIONS.md, 2026-09-24): can
the rendering + reconstruction mask + encoder/decoder + categorical
bottleneck represent the 144 states, with every other RSSM mechanism
removed? No KL, no prior, no GRU, no dynamics: image -> RSSM.encoder ->
RSSM.posterior_probs at the fixed learned initial h -> z ->
RSSM.decode(h0, z), trained on the 144 isolated state images with the
SAME weighted reconstruction loss the world model uses. Reuses the
RSSM's own modules and the training loop's own EstimatorSchedule --
never a separate architecture or estimator implementation.

Estimators (training only -- the model's categorical/unimix definition
is unchanged, see rssm.LatentEstimator):
  "straight_through": plain straight-through (DreamerV3's; the reference).
  "gumbel_st": straight-through Gumbel-softmax, tau annealed 1 -> 0.1.
  "warmup_probs": decoder fed the posterior probabilities for the first
      warmup_steps, then plain straight-through samples.
  "warmup_probs_until_separated": same, but the switch is triggered by the
      label-free batch-pair JS criterion (loop.SeparationSwitch) instead of
      a fixed step -- the retained procedure.
  "deterministic": z = posterior probabilities throughout -- isolates
      rendering + mask + encoder/decoder only (no discrete bottleneck).

Evaluation after training, for every categorical estimator: a DISCRETE
forward pass with a hard sample (one-hot draw from the posterior),
nearest-template accuracy averaged over n_eval_draws draws. Criterion:
>= 0.95. Also recorded along training (every eval_every steps, starting
at initialisation): hard-sample and mode accuracy, mean posterior
entropy, mean pairwise Jensen-Shannon divergence between the 144
states' posteriors (measures.diagnostics.mean_pairwise_jensen_shannon).

Usage (PowerShell, from the repo root):
    .venv\\Scripts\\python.exe experiments\\E3_worldmodel_4zones\\ae_isolation.py --estimator warmup_probs
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch

from helplessness.agents.worldmodel.losses import mean_categorical_entropy, weighted_reconstruction_loss
from helplessness.agents.worldmodel.loop import (
    CollapseAfterSwitchError,
    EstimatorSchedule,
    ProbsPhaseAid,
    SeparationNotReachedError,
    SeparationSwitch,
    module_grad_norms,
    phase_learning_rate,
    set_learning_rate,
)
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig, LatentEstimator, sample_posterior
from helplessness.experiment_loader import load_experiment_module
from helplessness.measures.diagnostics import mean_random_pair_jensen_shannon

ESTIMATORS = ("straight_through", "gumbel_st", "warmup_probs", "warmup_probs_until_separated", "deterministic")
N_PAIRS = 64
CRITERION = 0.95

# Schedule defaults, fixed BEFORE the comparison was run (DECISIONS.md,
# 2026-09-24). 6000 steps total; the first half (3000) is the budget the
# deterministic path was already known to need, at this lr, to separate
# the 144 states (0.986 at k=2, 0.993 at k=6 after 3000 steps; 0.40 after
# 1500) -- so the warm-up never switches before separation could exist,
# and gumbel_st gets the same horizon to reach its final temperature.
DEFAULT_N_STEPS = 6000
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 32
DEFAULT_ANNEAL_FRACTION = 1 / 2  # gumbel_st: tau 1 -> 0.1 over the first half, then held at 0.1
DEFAULT_WARMUP_FRACTION = 1 / 2  # warmup_probs: probabilities for the first half, then straight-through


def make_schedule(
    estimator: str, n_steps: int, anneal_steps: int | None = None, warmup_steps: int | None = None,
    switch_max_steps: int | None = None,
) -> EstimatorSchedule | SeparationSwitch | None:
    """None for "deterministic" (z = probs throughout, not a schedule). The
    criterion switch uses the retained defaults (3 bits, 5 consecutive
    evaluations every 50 steps, not before step 500, deadline 60% of
    n_steps unless switch_max_steps is given)."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    if estimator == "deterministic":
        return None
    if estimator == "warmup_probs_until_separated":
        max_steps = switch_max_steps if switch_max_steps is not None else int(round(0.6 * n_steps))
        return SeparationSwitch(max_steps=max_steps)
    if estimator == "gumbel_st":
        steps = anneal_steps if anneal_steps is not None else int(round(n_steps * DEFAULT_ANNEAL_FRACTION))
        return EstimatorSchedule(kind="gumbel_st", tau_start=1.0, tau_end=0.1, anneal_steps=steps)
    if estimator == "warmup_probs":
        steps = warmup_steps if warmup_steps is not None else int(round(n_steps * DEFAULT_WARMUP_FRACTION))
        return EstimatorSchedule(kind="warmup_probs", warmup_steps=steps)
    return EstimatorSchedule(kind="straight_through")


def _posterior(rssm: RSSM, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h0, _ = rssm.initial_state(x.shape[0], x.device)
    return h0, rssm.posterior_probs(h0, rssm.encoder(x))


def _probe_module():
    return load_experiment_module("E3_worldmodel_4zones", "probe.py", "e3_ae_isolation_probe")


def evaluate(rssm: RSSM, templates: torch.Tensor, n_eval_draws: int, deterministic: bool) -> dict[str, float]:
    """probe.canonical_separation -- one implementation, shared with
    run_mini.py's training-time separation curve."""
    return _probe_module().canonical_separation(rssm, templates, n_eval_draws=n_eval_draws, deterministic=deterministic)


def usage_snapshot(rssm: RSSM, templates: torch.Tensor, step: int, n_eval_draws: int, seed: int,
                   isolate_rng: bool = True) -> dict[str, float]:
    """probe.latent_usage_diagnostic at one step (the switch, or the
    deadline): accuracy with probabilities / mode / hard sample, labelled
    JS between states, posterior entropy, I(z;s), logit norms. Runs under
    torch.random.fork_rng: its hard-sample draws must not shift the torch
    RNG stream the training samples come from (taken mid-run at the
    switch, it would otherwise change every later straight-through draw).
    isolate_rng=False reproduces the discarded (i) batch's stream exactly
    (DECISIONS.md, 2026-09-24) -- reproduction only."""
    def diag():
        return _probe_module().latent_usage_diagnostic(rssm, templates, M=2000, rng=np.random.default_rng([seed, 5]),
                                                       n_eval_draws=n_eval_draws)

    if not isolate_rng:
        return {"step": step, **diag()}
    with torch.random.fork_rng():
        return {"step": step, **diag()}


def train_autoencoder(
    rssm: RSSM,
    templates: torch.Tensor,
    background: torch.Tensor,
    k: int,
    w_patch: float,
    n_steps: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
    schedule: EstimatorSchedule | SeparationSwitch | None,
    eval_every: int = 100,
    n_eval_draws: int = 5,
    pair_rng: np.random.Generator | None = None,
    stop_after_switch: int | None = None,
    aid: ProbsPhaseAid | None = None,
    raise_on_deadline: bool = True,
    seed: int = 0,
    lr_after_switch: float | None = None,
    clip_norm: float | None = None,
    log_step_diagnostics: bool = False,
    legacy_snapshot_rng: bool = False,
) -> dict:
    """Minibatches of isolated state images. schedule=None means the
    deterministic path (z = probs). Returns the loss history and the
    evaluation curve (starting at initialisation, step 0); every curve
    entry also carries the label-free batch-pair JS of that step's
    training batch ("batch_js", same parameters as the labelled
    evaluation). A SeparationSwitch raises SeparationNotReachedError past
    its deadline. stop_after_switch (SeparationSwitch only): end training
    that many steps after the effective switch, instead of at n_steps.

    aid (SeparationSwitch only): loop.ProbsPhaseAid applied during the
    probabilities phase (entropy penalty / progressive samples). Every
    curve entry also carries the training batch's reconstruction loss
    ("recon_loss") and its posterior entropy in nats ("batch_entropy_nats").
    With a SeparationSwitch, a latent-usage snapshot is taken at the
    effective switch ("at_switch") or, when the deadline is hit, at the
    deadline ("at_deadline"); raise_on_deadline=False then returns the
    partial curve with status "separation_not_reached" instead of
    raising. The same applies to the post-switch collapse guard
    (CollapseAfterSwitchError -> "at_collapse", status
    "collapse_after_switch").

    lr_after_switch (SeparationSwitch only): the learning rate from the
    effective switch on (loop.phase_learning_rate); the step it changed is
    returned as "lr_switch_step". clip_norm: global gradient-norm clipping
    (None: no clipping). The global gradient norm (before clipping) of every
    step is returned as "grad_norms". log_step_diagnostics: also return,
    per step, the gradient norm per module (encoder, posterior_net,
    decoder, h0 -- the isolation loop has no GRU and no prior), the mean
    centred posterior logit norm and the batch posterior entropy (bits),
    and the lr ("step_log"). legacy_snapshot_rng: reproduction only, see
    usage_snapshot."""
    aid = aid if aid is not None else ProbsPhaseAid()
    at_switch = at_deadline = None
    status, error = "ok", None
    pair_rng = pair_rng if pair_rng is not None else np.random.default_rng(0)
    params = list(rssm.encoder.parameters()) + list(rssm.posterior_net.parameters()) + list(rssm.decoder.parameters()) + [rssm.h0]
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-8)
    at_collapse, lr_switch_step, grad_norms, step_log = None, None, [], []
    captured: dict[str, torch.Tensor] = {}
    hook = rssm.posterior_net.register_forward_hook(lambda _m, _i, out: captured.__setitem__("logits", out)) if log_step_diagnostics else None
    modules = {"encoder": rssm.encoder, "posterior_net": rssm.posterior_net, "decoder": rssm.decoder}
    deterministic = schedule is None
    n = templates.shape[0]
    losses, curve = [], []
    stop_step = n_steps
    for step in range(n_steps + 1):
        if step % eval_every == 0 or step == stop_step:
            curve.append({"step": step, **evaluate(rssm, templates, n_eval_draws, deterministic)})
        if step == stop_step:
            break
        if isinstance(schedule, SeparationSwitch) and lr_after_switch is not None:
            if set_learning_rate(optimizer, phase_learning_rate(step, schedule.switch_step, lr, lr_after_switch)):
                lr_switch_step = step
        if deterministic:
            estimator = LatentEstimator("probs")
        elif isinstance(schedule, SeparationSwitch):
            estimator = aid.estimator(step, schedule)
        else:
            estimator = schedule.at(step)
        idx = torch.as_tensor(rng.choice(n, size=min(batch_size, n), replace=False))
        x = templates[idx]
        h0, post = _posterior(rssm, x)
        batch_logits = captured.get("logits")
        is_switch_eval = isinstance(schedule, SeparationSwitch) and schedule.is_eval_step(step)
        if step % eval_every == 0 or is_switch_eval:
            batch_js = mean_random_pair_jensen_shannon(post.detach().cpu().numpy(), pair_rng, N_PAIRS)
            if step % eval_every == 0:
                curve[-1]["batch_js"] = batch_js
            if is_switch_eval:
                was_switched = schedule.switched
                try:
                    schedule.update(step, batch_js)
                except CollapseAfterSwitchError as err:
                    at_collapse = usage_snapshot(rssm, templates, step, n_eval_draws, seed)
                    if raise_on_deadline:
                        raise
                    status, error = "collapse_after_switch", str(err)
                    curve.append({"step": step, **evaluate(rssm, templates, n_eval_draws, deterministic)})
                    break
                if schedule.switched and not was_switched:
                    at_switch = usage_snapshot(rssm, templates, step, n_eval_draws, seed,  # the triggering evaluation
                                               isolate_rng=not legacy_snapshot_rng)
        recon = rssm.decode(h0, sample_posterior(post, estimator))
        recon_loss = weighted_reconstruction_loss(x, recon, background.unsqueeze(0).expand_as(x), k=k, w_patch=w_patch)
        entropy = mean_categorical_entropy(post)
        eta = aid.entropy_weight(step, schedule.switched) if isinstance(schedule, SeparationSwitch) else 0.0
        loss = recon_loss + eta * entropy if eta > 0 else recon_loss
        if step % eval_every == 0:
            curve[-1]["recon_loss"] = float(recon_loss.detach())
            curve[-1]["batch_entropy_nats"] = float(entropy.detach())
        optimizer.zero_grad()
        loss.backward()
        if clip_norm is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm))
        else:
            grad_norm = float(torch.linalg.vector_norm(torch.cat([q.grad.detach().reshape(-1) for q in params if q.grad is not None])))
        grad_norms.append(grad_norm)
        if log_step_diagnostics:
            # Read after clipping when clip_norm is set: per-module norms of the APPLIED update direction;
            # grad_norm above is always the pre-clip global norm.
            centred = batch_logits.detach().reshape(-1, rssm.cfg.n_vars, rssm.cfg.n_classes)
            centred = centred - centred.mean(dim=-1, keepdim=True)
            step_log.append({
                "step": step, "loss": float(loss.detach()), "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"],
                **{f"grad_{name}": v for name, v in module_grad_norms(modules).items()},
                "grad_h0": float(rssm.h0.grad.norm()) if rssm.h0.grad is not None else 0.0,
                "logit_norm_centered": float(centred.norm(dim=-1).mean()),
                "batch_entropy_bits": float(entropy.detach()) / float(np.log(2)),
            })
        optimizer.step()
        losses.append(float(loss.detach()))
        if isinstance(schedule, SeparationSwitch):
            try:
                schedule.check_deadline(step + 1)
            except SeparationNotReachedError as err:
                at_deadline = usage_snapshot(rssm, templates, step + 1, n_eval_draws, seed)
                if raise_on_deadline:
                    raise
                status, error = "separation_not_reached", str(err)
                curve.append({"step": step + 1, **evaluate(rssm, templates, n_eval_draws, deterministic)})
                break
            if stop_after_switch is not None and schedule.switched:
                stop_step = min(n_steps, schedule.switch_step + stop_after_switch)
    if hook is not None:
        hook.remove()
    return {"losses": losses, "curve": curve, "at_switch": at_switch, "at_deadline": at_deadline, "at_collapse": at_collapse,
            "status": status, "error": error, "lr_switch_step": lr_switch_step, "grad_norms": grad_norms, "step_log": step_log}


def run(
    n_steps: int,
    lr: float,
    batch_size: int,
    seed: int,
    estimator: str,
    k: int | None = None,
    anneal_steps: int | None = None,
    warmup_steps: int | None = None,
    eval_every: int = 100,
    n_eval_draws: int = 5,
    switch_max_steps: int | None = None,
    stop_after_switch: int | None = None,
    aid: dict | None = None,
    raise_on_deadline: bool = True,
    lr_after_switch: float | None = None,
    clip_norm: float | None = None,
    log_step_diagnostics: bool = False,
    legacy_snapshot_rng: bool = False,
) -> dict:
    """k=None uses configs/e3/mini.yaml's own render scale. aid: keyword
    arguments of loop.ProbsPhaseAid (kind, eta_max); its ramp_steps is
    always the switch deadline."""
    # Loaded through the experiment loader, never a bare `import run_mini`:
    # run_mini does `import env`, and a bare import would bind whichever
    # `env` module is already cached (E1's or E2's, in a full test session).
    run_mini = load_experiment_module("E3_worldmodel_4zones", "run_mini.py", "e3_ae_isolation_run_mini")

    cfg = run_mini.load_mini_config()
    if k is not None:
        cfg["render"]["k"] = k
    k = cfg["render"]["k"]
    torch.manual_seed(seed)
    _layout, _themes, _env, background, templates = run_mini.build_run(cfg, seed)
    side = 12 * k
    rssm = RSSM(RSSMConfig(
        image_size=(side, side), base_depth=cfg["model"]["base_depth"], embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"], n_vars=cfg["model"]["n_vars"], n_classes=cfg["model"]["n_classes"],
        n_actions=5, unimix=cfg["model"]["unimix"],
    ))
    schedule = make_schedule(estimator, n_steps, anneal_steps, warmup_steps, switch_max_steps)
    probs_aid = None
    if aid is not None:
        if not isinstance(schedule, SeparationSwitch):
            raise ValueError("a probabilities-phase aid needs the criterion switch (warmup_probs_until_separated)")
        probs_aid = ProbsPhaseAid(**aid, ramp_steps=schedule.max_steps)
    out = train_autoencoder(
        rssm, templates, background, k=k, w_patch=cfg["train"]["w_patch"], n_steps=n_steps, lr=lr,
        batch_size=batch_size, rng=np.random.default_rng(seed), schedule=schedule, eval_every=eval_every,
        n_eval_draws=n_eval_draws, pair_rng=np.random.default_rng([seed, 4]), stop_after_switch=stop_after_switch,
        aid=probs_aid, raise_on_deadline=raise_on_deadline, seed=seed, lr_after_switch=lr_after_switch,
        clip_norm=clip_norm, log_step_diagnostics=log_step_diagnostics, legacy_snapshot_rng=legacy_snapshot_rng,
    )
    final = out["curve"][-1]
    return {
        "estimator": estimator, "seed": seed, "k": k, "n_steps": n_steps, "lr": lr, "batch_size": batch_size,
        "schedule": None if schedule is None else {k: v for k, v in schedule.__dict__.items() if k != "history"},
        "switch_step": schedule.switch_step if isinstance(schedule, SeparationSwitch) else None,
        "aid": None if probs_aid is None else dataclasses.asdict(probs_aid),
        "status": out["status"], "error": out["error"], "at_switch": out["at_switch"], "at_deadline": out["at_deadline"],
        "at_collapse": out["at_collapse"], "lr_after_switch": lr_after_switch, "lr_switch_step": out["lr_switch_step"],
        "clip_norm": clip_norm, "legacy_snapshot_rng": legacy_snapshot_rng, "num_threads": torch.get_num_threads(),
        "grad_norms": out["grad_norms"], "losses": out["losses"], "step_log": out["step_log"],
        "accuracy": final["acc_hard"] if out["status"] == "ok" else 0.0, "final": final, "curve": out["curve"],
        "loss_first": out["losses"][0], "loss_last": out["losses"][-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", choices=ESTIMATORS, default="warmup_probs")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    result = run(args.n_steps, args.lr, args.batch_size, args.seed, args.estimator, k=args.k)
    verdict = "PASS" if result["accuracy"] >= CRITERION else "FAIL"
    f = result["final"]
    print(f"{args.estimator} seed={args.seed} k={result['k']}: hard-sample accuracy = {f['acc_hard']:.4f} "
          f"(criterion >= {CRITERION}): {verdict}; mode accuracy {f['acc_mode']:.4f}; "
          f"posterior entropy {f['posterior_entropy']:.2f} bits; mean JS {f['mean_js']:.3f} bits")


if __name__ == "__main__":
    main()
