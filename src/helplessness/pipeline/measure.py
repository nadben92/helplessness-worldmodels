"""E3's run pipeline: the measures of a run -- real-context and canonical measures, training diagnostics, D1/D2,
the final measurement, their random-stream slots, and the collapse figure. Moved verbatim from
experiments/E3_worldmodel_4zones/run_mini.py on 2026-10-14 (run_mini.py re-exports every name); the module-level
names those functions read are imports here (E3's probes from helplessness.probing.probe, the rest from
helplessness.pipeline.runtime and helplessness.pipeline.world).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from helplessness.agents.worldmodel.agent import categorical_mode
from helplessness.agents.worldmodel.losses import reconstruction_mask
from helplessness.agents.worldmodel.replay import ReplayBuffer
from helplessness.agents.worldmodel.rssm import categorical_kl, RSSM
from helplessness.envs.layouts import four_zones_pixel as e3_env
from helplessness.envs.zones import geographic_support_mask
from helplessness.measures.context_summary import per_state_context_summary, zone_mean_with_se
from helplessness.measures.controllability import controllability_bootstrap, controllability_per_state
from helplessness.measures.diagnostics import (
    alpha_app_with_se,
    d1_contrast,
    jensen_shannon_categorical,
    summarize_alpha_app,
)
from helplessness.measures.kernel_probe import expected_l1_sampling_floor, off_support_mass
from helplessness.measures.latent_controllability import latent_state_information
from helplessness.pipeline.runtime import gpu_peak_end, gpu_peak_start, max_images_per_forward, measure_streams, RunLog
from helplessness.pipeline.world import (
    build_tabular_twin,
    GRID_SHAPE,
    N_ACTIONS,
    N_STATES,
    NOISY_ZONES,
    World,
    zone_of_state,
    zone_states,
)
from helplessness.probing import probe as e3_probe


N_SE = 3.0  # every MC-based verdict in the report uses a 3-standard-error margin


ACC_LABELS = {
    "acc_replay_hard": "replay windows (real contexts h_t), hard sample",
    "acc_real_context": "real contexts of the main measure (after the burn-in, no is_first step), hard sample",
    "acc_canonical_hard": "canonical probe (fresh h0), hard sample, mean of 5 draws",
    "acc_canonical_mode": "canonical probe (fresh h0), posterior mode",
}
# Measurement slots (rng_streams.measure_streams): each measurement owns one.
SLOT_SEPARATION, SLOT_DIAG, SLOT_IZS, SLOT_ZONLY, SLOT_D1_REAL, SLOT_FAILURE, SLOT_FINAL, SLOT_D1_CANONICAL = 1, 2, 3, 4, 6, 7, 8, 9
SLOT_HONLY = 11  # the h-only probe (2026-09-30); slot 10 is action_in_latent.py's
SLOT_PER_CONTEXT = 12  # the per-context I_s check (2026-09-30, also is_pooling_check.py)


def zone_mean_mc_se(se_per_state: np.ndarray) -> float:
    """MC standard error of a zone MEAN of per-state I_z estimates:
    sqrt(sum se_s^2) / n. Only Monte-Carlo noise counts -- the states are
    a fixed, complete set, not a sample, so their spread is signal."""
    se = np.asarray(se_per_state, dtype=np.float64)
    return float(np.sqrt(np.sum(se**2)) / len(se))


def labelled_between_state_js(post_probs: np.ndarray, labels: np.ndarray) -> float:
    """Console monitoring only -- never an input to the switch or to
    training. Mean JS (bits, summed over latent variables) between the
    posteriors of every pair of batch samples whose true states DIFFER;
    post_probs (n, n_vars, n_classes) and labels (n,) in the same order.
    NaN if every sample shows the same state."""
    pair_js = jensen_shannon_categorical(post_probs[:, None], post_probs[None, :]).sum(axis=-1)
    different = labels[:, None] != labels[None, :]
    return float(pair_js[different].mean()) if different.any() else float("nan")


def kl_post_prior_by_zone(probe: dict, layout) -> dict[str, float]:
    """Canonical-probe "KL by zone type", in BITS: mean_a KL(post_probs[s]
    || prior_probs[s,a]), averaged over states of each zone category.
    CAUTION (docs/e3/DECISIONS.md, 2026-09-28): post_probs[s] is the
    posterior of the CURRENT state s (at h0) and prior_probs[s,a] the
    prediction of the NEXT state after a -- the two describe different
    states whenever s' != s, so this quantity is large even for a perfect
    model and is not the training loss's KL. Kept, labelled, as the
    secondary record; the model-fit KL is the real-context one
    (probe.real_context_probe's kl_next_bits). rssm.categorical_kl works in
    nats; converted here."""
    post = torch.as_tensor(probe["posterior_probs"])  # (144, n_vars, n_classes)
    prior = torch.as_tensor(probe["prior_probs"])  # (144, n_actions, n_vars, n_classes)
    n_states, n_actions = prior.shape[0], prior.shape[1]
    post_expanded = post.unsqueeze(1).expand(-1, n_actions, -1, -1)
    kl = categorical_kl(post_expanded.reshape(-1, *post.shape[1:]), prior.reshape(-1, *prior.shape[2:]))
    kl = kl.reshape(n_states, n_actions).mean(dim=1).numpy() / np.log(2.0)  # nats -> bits, (144,)

    by_zone: dict[str, list[float]] = {}
    for s in range(n_states):
        by_zone.setdefault(zone_of_state(layout, s), []).append(float(kl[s]))
    return {zone: float(np.mean(vals)) for zone, vals in by_zone.items()}


def frequent_outcome_preview(p_hat: np.ndarray, layout) -> dict[str, float]:
    """T4 preview (spec section 9): mean, over (s, a) pairs in each
    noisy zone, of the world model's own s-space kernel's max
    probability (the "frequent outcome"); states not measured (NaN rows)
    are left out. Not the full T4 statistical test -- a shape preview
    only."""
    out = {}
    for zone in NOISY_ZONES:
        rows = p_hat[zone_states(layout, zone)]  # (n_states_in_zone, n_actions, n_states)
        max_probs = np.where(np.isnan(rows).any(axis=-1), np.nan, np.nan_to_num(rows).max(axis=-1))
        out[zone] = float(np.nanmean(max_probs)) if np.isfinite(max_probs).any() else float("nan")
    return out


def save_collapse_grid(rssm: RSSM, templates: torch.Tensor, background: torch.Tensor, k: int, states: list[int], path: Path) -> None:
    """Collapse diagnostic figure: for each state, the input image, the
    reconstruction-loss mask computed from it, and the POSTERIOR
    reconstruction (canonical path: fresh h0, z = posterior mode). A
    collapsed posterior shows the same reconstruction in every column,
    with no agent marker."""
    plt = _plt()
    device = next(rssm.parameters()).device
    x = templates[states].to(device)
    with torch.no_grad():
        h0, _ = rssm.initial_state(len(states), device)
        z0 = categorical_mode(rssm.posterior_probs(h0, rssm.encoder(x)))
        recon = rssm.decode(h0, z0).clamp(0.0, 1.0)
        mask = reconstruction_mask(x, background.to(device).unsqueeze(0).expand_as(x), k)

    fig, axes = plt.subplots(3, len(states), figsize=(1.6 * len(states), 5))
    for j, s in enumerate(states):
        for row, (img, title) in enumerate(((x[j], "input"), (mask[j].expand(3, -1, -1), "mask"), (recon[j], "posterior recon"))):
            ax = axes[row, j]
            ax.imshow(img.permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(title, fontsize=8)
        axes[0, j].set_title(f"s={s}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def real_context_settings(cfg: dict) -> dict:
    """measure.real_context, with window_len null = the training seq_len."""
    rc = dict(cfg["measure"]["real_context"])
    if rc.get("window_len") is None:
        rc["window_len"] = cfg["train"]["seq_len"]
    if rc["burn_in"] >= rc["window_len"]:
        raise ValueError(f"measure.real_context.burn_in ({rc['burn_in']}) must be < window_len ({rc['window_len']})")
    return rc


def kernel_draws(cfg: dict, final: bool) -> tuple[int, int]:
    """(K for the real-context kernel, K_s for the canonical kernel): the
    final values (measure.real_context.kernel_draws, measure.kernel_k_s),
    or during training the reduced *_training values when the config sets
    them (docs/e3/DECISIONS.md, 2026-09-30)."""
    m, rc = cfg["measure"], cfg["measure"]["real_context"]
    if final:
        return rc["kernel_draws"], m["kernel_k_s"]
    return rc.get("kernel_draws_training", rc["kernel_draws"]), m.get("kernel_k_s_training", m["kernel_k_s"])


def controllability_of_kernel(p_hat: np.ndarray) -> np.ndarray:
    """I_s per state: measures.controllability.controllability_per_state of
    the world model's s-space kernel p_hat (n_states, n_actions, n_states);
    NaN for a state with a row that is not a law (not measured, or every
    draw of an action a ghost)."""
    valid = np.isfinite(p_hat).all(axis=(1, 2)) & np.isclose(np.nan_to_num(p_hat).sum(axis=-1), 1.0).all(axis=1)
    out = np.full(p_hat.shape[0], np.nan)
    if valid.any():
        out[valid] = controllability_per_state(p_hat[valid])
    return out


def per_context_is_check(rssm: RSSM, world: World, contexts: dict, cfg: dict, rng: np.random.Generator,
                         generator: torch.Generator, states_per_zone: int, n_contexts: int, draws: int) -> dict:
    """I_s per context vs pooled over the same contexts (docs/e3/DECISIONS.md,
    2026-09-30): the SECONDARY diagnostic reported with every final
    measurement. The main I_s pools the decoded draws of a state's contexts
    into one kernel (as E2's per-state kernel and the tabular twin B: one
    kernel per state, histories merged); here, for states_per_zone states per
    zone (drawn from `rng` among those with >= n_contexts contexts) and
    n_contexts of their contexts, `draws` prior draws per context and action
    (probe.real_context_action_samples) give each context its own kernel:
      per_context   mean over the contexts of I_s(context's kernel);
      pooled        I_s of the kernel pooling the same contexts' draws;
      gap           pooled - per_context (<= 0 up to sampling: mutual
                    information is convex in the channel);
    with the bootstrap bias of each, per state and by zone."""
    layout = world.layout
    chosen_states, chosen_ctx = [], []
    for cat in e3_env.ZONE_CATEGORIES:
        candidates = [s for s in zone_states(layout, cat) if (contexts["states"] == s).sum() >= n_contexts]
        for s in sorted(rng.choice(candidates, size=min(states_per_zone, len(candidates)), replace=False)) if candidates else []:
            chosen_states.append(int(s))
            chosen_ctx.append(np.sort(rng.choice(np.flatnonzero(contexts["states"] == s), size=n_contexts, replace=False)))
    if not chosen_states:
        return {"states": [], "by_zone": {}, "settings": {"states_per_zone": states_per_zone, "contexts": n_contexts, "draws": draws}}
    idx = np.concatenate(chosen_ctx)
    idx_t = torch.as_tensor(idx, device=contexts["h"].device)
    sub = {"positions": contexts["positions"][idx], "states": contexts["states"][idx],
           "prior_probs_t": contexts["prior_probs_t"][idx_t], "h_next": contexts["h_next"][idx_t]}
    samples = e3_probe.real_context_action_samples(rssm, sub, world.templates_t, draws=draws,
                                                   margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=generator)
    keep = ~samples["is_ghost"]
    counts = np.zeros((idx.size, N_ACTIONS, N_STATES))
    np.add.at(counts, (samples["context"][keep], samples["a"][keep], samples["s_next"][keep]), 1.0)
    with np.errstate(invalid="ignore"):
        kernels = counts / counts.sum(axis=-1, keepdims=True)
    rows = []
    for k, s in enumerate(chosen_states):
        block = kernels[k * n_contexts:(k + 1) * n_contexts]
        valid = np.isfinite(block).all(axis=(1, 2))
        I_ctx = controllability_per_state(block[valid])
        pooled = block[valid].mean(axis=0, keepdims=True)  # the same number of draws per context
        I_pool = float(controllability_per_state(pooled)[0])
        rows.append({"s": s, "zone": zone_of_state(layout, s), "n_contexts": int(valid.sum()),
                     "per_context": float(I_ctx.mean()), "per_context_sd": float(I_ctx.std(ddof=1)) if I_ctx.size > 1 else float("nan"),
                     "per_context_bias": float(np.mean(controllability_bootstrap(block[valid], draws, 200, rng)["bias"])),
                     "pooled": I_pool, "pooled_bias": float(controllability_bootstrap(pooled, draws * int(valid.sum()), 200, rng)["bias"][0]),
                     "gap": I_pool - float(I_ctx.mean())})
    by_zone = {}
    for cat in e3_env.ZONE_CATEGORIES:
        zr = [r for r in rows if r["zone"] == cat]
        if not zr:
            continue
        gaps = np.array([r["gap"] for r in zr])
        by_zone[cat] = {**{key: float(np.mean([r[key] for r in zr]))
                           for key in ("per_context", "per_context_bias", "pooled", "pooled_bias", "gap")},
                        "gap_se_over_states": float(gaps.std(ddof=1) / np.sqrt(gaps.size)) if gaps.size > 1 else float("nan"),
                        "n_states": len(zr)}
    return {"states": rows, "by_zone": by_zone, "ghost_fraction": float(samples["is_ghost"].mean()),
            "settings": {"states_per_zone": states_per_zone, "contexts": n_contexts, "draws": draws}}


def is_by_zone(I_s: np.ndarray, boot: dict, layout) -> dict[str, dict]:
    """Zone summaries of I_s: mean over the measured states with the se of
    the mean (zone_mean_with_se, per-state bootstrap se), and the mean
    bootstrap estimate of the plug-in bias."""
    out = {}
    for cat in e3_env.ZONE_CATEGORIES:
        states = zone_states(layout, cat)
        bias = boot["bias"][states]
        out[cat] = {**zone_mean_with_se(I_s, boot["se"], states),
                    "bias": float(np.nanmean(bias)) if np.isfinite(bias).any() else float("nan")}
    return out


def real_context_measures(
    rssm: RSSM, world: World, replay: ReplayBuffer, P_true: np.ndarray, T_hat_B: np.ndarray, cfg: dict,
    rng: np.random.Generator, generator: torch.Generator, M: int, final: bool = True,
) -> tuple[dict, dict]:
    """The main measure, in real contexts: per state, the s-space kernel
    decoded from prior draws of the state's contexts and its
    controllability I_s = I(a ; s' | s) (THE MAIN MEASURE since
    docs/e3/DECISIONS.md 2026-09-30: measures.controllability on the
    decoded arrival states), with a parametric-bootstrap se and bias; I_z
    (mean over the contexts, SECONDARY since 2026-09-30: an upper bound),
    the one-step KL, the linear probes h -> s and z -> s on the contexts;
    aggregated by zone over the MEASURED states (a state without a context
    is "non mesuré"). final: the final kernel sizes, else the reduced
    training ones (kernel_draws). Returns a JSON-able record and the arrays
    (contexts included)."""
    rc = real_context_settings(cfg)
    draws, _ = kernel_draws(cfg, final)
    layout = world.layout
    ctx = e3_probe.real_context_probe(rssm, replay, window_len=rc["window_len"], burn_in=rc["burn_in"],
                                      max_per_state=rc["max_per_state"], n_states=N_STATES, rng=rng, generator=generator,
                                      max_images=max_images_per_forward(cfg))
    iz = e3_probe.real_context_iz(ctx, M=M, rng=rng)
    kl = per_state_context_summary(ctx["kl_next_bits"], ctx["states"], N_STATES)
    # The reachable successors of each state: the support of the tabular twin's prior (build_tabular_twin).
    support = geographic_support_mask(GRID_SHAPE)
    kernel = e3_probe.real_context_kernel(rssm, ctx, world.templates_t, draws=draws,
                                          margin_threshold=cfg["measure"]["kernel_margin_threshold"], generator=generator,
                                          support=support)
    acc = e3_probe.real_context_template_accuracy(rssm, ctx, world.templates_t)
    l1_vs_p = np.abs(kernel["p_hat"] - P_true).sum(axis=-1)  # (144, 5), NaN for unmeasured states
    l1_vs_b = np.abs(kernel["p_hat"] - T_hat_B).sum(axis=-1)
    l1_b_vs_p = np.abs(T_hat_B - P_true).sum(axis=-1)
    # C-b's sampling floor (docs/e3/DECISIONS.md, 2026-09-29): the L1 an exact kernel scores with this many draws
    l1_floor = np.where(np.isnan(l1_vs_p), np.nan, expected_l1_sampling_floor(P_true, draws))
    I_s = controllability_of_kernel(kernel["p_hat"])
    boot = controllability_bootstrap(kernel["p_hat"], draws, cfg["measure"].get("is_bootstrap", 200), rng)
    probes = e3_probe.context_state_probes(ctx, cfg["measure"]["z_only_test_frac"], rng) if ctx["positions"].size else None
    # docs/e3/DECISIONS.md, 2026-09-30 (5): I_s above is computed on the kernel as it is (the model's actual belief,
    # off-support mass included). Reported next to it: the share of that mass outside the reachable successors, and
    # I_s of the kernel projected on the support (SECONDARY: for the comparison with the tabular twin, whose prior
    # is restricted to the same support). The projected bootstrap draws last, so everything above is unchanged.
    off_support = off_support_mass(kernel["p_hat"], support)
    I_s_projected = controllability_of_kernel(kernel["p_hat_projected"])
    boot_projected = controllability_bootstrap(kernel["p_hat_projected"], draws, cfg["measure"].get("is_bootstrap", 200), rng)
    cats = list(e3_env.ZONE_CATEGORIES)

    def by_zone(arr: np.ndarray) -> dict[str, float]:
        out = {}
        for cat in cats:
            vals = arr[zone_states(layout, cat)]
            out[cat] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        return out

    n = iz["n_contexts"]
    record = {
        "kl_by_zone": {cat: zone_mean_with_se(kl["mean"], kl["se"], zone_states(layout, cat))["mean"] for cat in cats},
        "l1_vs_p_by_zone": by_zone(l1_vs_p),
        "l1_vs_b_by_zone": by_zone(l1_vs_b),
        "l1_b_vs_p_by_zone": by_zone(l1_b_vs_p),
        "l1_floor_by_zone": by_zone(l1_floor),
        "kernel_draws": draws,
        "I_s_by_zone": is_by_zone(I_s, boot, layout),
        "I_s_projected_by_zone": is_by_zone(I_s_projected, boot_projected, layout),
        "off_support_mass_by_zone": by_zone(off_support),
        "ghost_fraction_projected_mean": (float(np.nanmean(kernel["ghost_fraction_projected"]))
                                          if np.isfinite(kernel["ghost_fraction_projected"]).any() else float("nan")),
        "I_z_by_zone": {cat: zone_mean_with_se(iz["I_z"], iz["se"], zone_states(layout, cat)) for cat in cats},
        "state_probe_h": None if probes is None else probes["h"]["test_accuracy"],
        "state_probe_z": None if probes is None else probes["z"]["test_accuracy"],
        "frequent_outcome": frequent_outcome_preview(kernel["p_hat"], layout),
        "posterior_entropy": float(e3_probe.per_variable_entropy(ctx["post_probs"]).mean()) if n.sum() else float("nan"),
        "acc_real_context": acc,
        "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])) if np.isfinite(kernel["ghost_fraction"]).any() else float("nan"),
        "n_contexts_total": int(n.sum()),
        "n_states_measured": int((n > 0).sum()),
        "contexts_per_state": {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())},
        "not_measured_states": [int(s) for s in np.flatnonzero(n == 0)],
        "settings": rc,
    }
    arrays = {"contexts": ctx, "iz": iz, "kl": kl, "kernel": kernel, "l1_vs_p": l1_vs_p, "l1_vs_b": l1_vs_b,
              "l1_b_vs_p": l1_b_vs_p, "I_s": I_s, "I_s_se": boot["se"], "I_s_bias": boot["bias"],
              "I_s_projected": I_s_projected, "I_s_projected_se": boot_projected["se"], "I_s_projected_bias": boot_projected["bias"],
              "off_support": off_support}
    return record, arrays


def training_diagnostics(
    rssm: RSSM, world: World, P_true: np.ndarray, T_hat_B: np.ndarray, replay: ReplayBuffer, cfg: dict,
    rng: np.random.Generator, generator: torch.Generator, M: int, final: bool = True,
) -> tuple[dict, dict]:
    """One point of the training diagnostics (docs/e3/DECISIONS.md,
    2026-09-25 and 2026-09-28), also used for the final measurement.
    Top-level keys: the canonical probe (secondary since 2026-09-28) and the
    template accuracies; "real_context": the main measure. Returns a
    JSON-able record and the full arrays it was computed from (reused by
    the final report). All draws come from (rng, generator), owned by the
    caller. M: Monte-Carlo samples per I_z (measure.iz_mc_samples_training
    during training, measure.iz_mc_samples_final at the end). final: the
    final kernel sizes, else the reduced training ones (kernel_draws)."""
    m = cfg["measure"]
    _, k_s = kernel_draws(cfg, final)
    layout, templates_t = world.layout, world.templates_t
    was_training = rssm.training
    rssm.eval()
    t0 = time.perf_counter()
    iz = e3_probe.canonical_iz(rssm, templates_t, M=M, rng=rng, generator=generator)
    probe = iz["probe"]
    kernel = e3_probe.canonical_kernel_probe(rssm, templates_t, K_s=k_s, margin_threshold=m["kernel_margin_threshold"],
                                             probe=probe, generator=generator)
    separation = e3_probe.canonical_separation(rssm, templates_t, n_eval_draws=5, generator=generator)
    batch, labels = replay.sample(32, 16, rng)
    acc_replay = e3_probe.template_match_accuracy(rssm, batch, labels, templates_t, margin_threshold=m["kernel_margin_threshold"],
                                                  generator=generator)
    real_record, real_arrays = real_context_measures(rssm, world, replay, P_true, T_hat_B, cfg, rng, generator, M, final=final)
    rssm.train(was_training)

    l1_vs_p = np.abs(kernel["p_hat"] - P_true).sum(axis=-1)  # (144, 5)
    l1_vs_b = np.abs(kernel["p_hat"] - T_hat_B).sum(axis=-1)
    l1_b_vs_p = np.abs(T_hat_B - P_true).sum(axis=-1)
    l1_floor = expected_l1_sampling_floor(P_true, k_s)
    I_s_canonical = controllability_of_kernel(kernel["p_hat"])  # secondary: the canonical kernel (h0 contexts)
    boot_canonical = controllability_bootstrap(kernel["p_hat"], k_s, m.get("is_bootstrap", 200), rng)
    cats = list(e3_env.ZONE_CATEGORIES)

    def by_zone(arr: np.ndarray) -> dict[str, float]:
        return {cat: float(np.nanmean(arr[zone_states(layout, cat)])) for cat in cats}

    record = {
        "kl_by_zone": kl_post_prior_by_zone(probe, layout),
        "l1_vs_p_by_zone": by_zone(l1_vs_p),
        "l1_vs_b_by_zone": by_zone(l1_vs_b),
        "l1_b_vs_p_by_zone": by_zone(l1_b_vs_p),
        "l1_floor_by_zone": by_zone(l1_floor),
        "kernel_k_s": k_s,
        "I_s_by_zone": is_by_zone(I_s_canonical, boot_canonical, layout),
        "I_z_by_zone": {cat: {"mean": float(iz["I_z"][zone_states(layout, cat)].mean()),
                              "mc_se": zone_mean_mc_se(iz["se"][zone_states(layout, cat)])} for cat in cats},
        "frequent_outcome": frequent_outcome_preview(kernel["p_hat"], layout),
        "posterior_entropy_canonical": float(probe["posterior_entropy"].mean()),
        "acc_replay_hard": acc_replay,
        "acc_real_context": real_record["acc_real_context"],
        "acc_canonical_hard": separation["acc_hard"],
        "acc_canonical_mode": separation["acc_mode"],
        "ghost_fraction_mean": float(np.nanmean(kernel["ghost_fraction"])),
        "real_context": real_record,
        "iz_mc_samples": M,
        "seconds": time.perf_counter() - t0,
    }
    return record, {"iz": iz, "probe": probe, "kernel": kernel, "l1_vs_p": l1_vs_p, "l1_vs_b": l1_vs_b, "l1_b_vs_p": l1_b_vs_p,
                    "real": real_arrays}


def _d1_pairs(cats: list[str], stats: dict, cfg: dict, per_state) -> list[dict]:
    """The D1 table (spec section 8, criteria of docs/e3/DECISIONS.md
    2026-09-25): for every ordered pair of zones (own L, target L'), the
    contrast is accepted only if |I_L' - I_L| > max(d1_contrast_min_bits,
    3 x combined se). For an accepted pair, alpha_app is computed for every
    (measured) state of L, local and global, with its standard error;
    per-state values outside [0, 1] beyond 3 se are anomalies (listed,
    never averaged); the others are summarised by their mean over the
    zone's states +/- 3 sd/sqrt(n). per_state(own, target, global_swap) ->
    (states, I_swap, I_swap_se)."""
    results = []
    for own in cats:
        for target in cats:
            if own == target:
                continue
            (I_L, se_L), (I_Lp, se_Lp) = stats[own], stats[target]
            if not (np.isfinite(I_L) and np.isfinite(I_Lp)):
                results.append({"own_category": own, "target_category": target, "I_L": I_L, "se_L": se_L, "I_Lprime": I_Lp,
                                "se_Lprime": se_Lp, "contrast": float("nan"), "threshold": float("nan"), "combined_se": float("nan"),
                                "accepted": False, "not_measured": True})
                continue
            entry = {"own_category": own, "target_category": target, "I_L": I_L, "se_L": se_L, "I_Lprime": I_Lp,
                     "se_Lprime": se_Lp, **d1_contrast(I_L, se_L, I_Lp, se_Lp, min_bits=cfg["measure"]["d1_contrast_min_bits"], n_se=N_SE)}
            if entry["accepted"]:
                for kind, global_swap in (("local", False), ("global", True)):
                    states, I_swap, I_swap_se = per_state(own, target, global_swap)
                    alpha, se = alpha_app_with_se(I_swap, I_swap_se, I_L, se_L, I_Lp, se_Lp)
                    summary = summarize_alpha_app(alpha, se, n_se=N_SE)
                    for anomaly in summary["anomalies"]:
                        anomaly["s"] = states[anomaly.pop("index")]
                    summary["per_state"] = [{"s": s, "alpha": float(a), "se": float(e)} for s, a, e in zip(states, alpha, se)]
                    entry[kind] = summary
            results.append(entry)
    return results


def d1_analysis(rssm: RSSM, layout, themes, I_z: np.ndarray, I_z_se: np.ndarray, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """D1 from the canonical probe (secondary since 2026-09-28): I_swap from
    a fresh h0 on the counterfactual render of each state of L."""
    cats = list(e3_env.ZONE_CATEGORIES)
    stats = {cat: (float(I_z[zone_states(layout, cat)].mean()), zone_mean_mc_se(I_z_se[zone_states(layout, cat)])) for cat in cats}
    I_by_category = {cat: v[0] for cat, v in stats.items()}

    def per_state(own, target, global_swap):
        states = zone_states(layout, own)
        swaps = [e3_probe.d1_alpha_app_for_state(rssm, layout, themes, s, target, I_by_category, M=cfg["measure"]["iz_mc_samples_final"],
                                                 rng=rng, k=cfg["render"]["k"], contrast_threshold=0.0, global_swap=global_swap)
                 for s in states]
        return states, np.array([d["I_swap"] for d in swaps]), np.array([d["I_swap_se"] for d in swaps])

    return _d1_pairs(cats, stats, cfg, per_state)


def d1_analysis_real(rssm: RSSM, world: World, real_iz: dict, contexts: dict, cfg: dict, rng: np.random.Generator,
                     generator: torch.Generator) -> list[dict]:
    """D1 in real contexts (the main D1 since 2026-09-28): zone means and
    their standard errors from the real-context I_z of the measured states;
    I_swap(s) = mean over the contexts of s of I_z with x_t replaced by the
    counterfactual render (probe.real_context_d1_iswap). Only measured
    states of L enter."""
    layout, cats = world.layout, list(e3_env.ZONE_CATEGORIES)
    zone_stats = {cat: zone_mean_with_se(real_iz["I_z"], real_iz["se"], zone_states(layout, cat)) for cat in cats}
    stats = {cat: (v["mean"], v["se"]) for cat, v in zone_stats.items()}

    def per_state(own, target, global_swap):
        states = [s for s in zone_states(layout, own) if real_iz["n_contexts"][s] > 0]
        swaps = [e3_probe.real_context_d1_iswap(rssm, contexts, layout, world.themes, s, target, k=cfg["render"]["k"],
                                                global_swap=global_swap, M=cfg["measure"]["iz_mc_samples_final"], rng=rng,
                                                generator=generator) for s in states]
        return states, np.array([d["I_swap"] for d in swaps]), np.array([d["I_swap_se"] for d in swaps])

    return _d1_pairs(cats, stats, cfg, per_state)


def measure_final(rssm: RSSM, world: World, replay: ReplayBuffer, N: np.ndarray, cfg: dict, streams: dict, run_dir: Path,
                  log: RunLog, step: int) -> tuple[dict, dict]:
    """Every final measure of a trained model, in real contexts (main) and
    from the canonical probe (secondary). Used by run_one_seed at the end
    of training and by recompute_measures.py on any saved checkpoint:
    same measurement streams, so the same weights and data give the same
    numbers on the same device. Returns (report fields, arrays)."""
    seed = streams["seed"]
    layout, themes, k = world.layout, world.themes, cfg["render"]["k"]
    T_hat_B = build_tabular_twin(N, N_STATES, N_ACTIONS)
    P_true = world.env.transition_tensor()
    rssm.eval()
    device = next(rssm.parameters()).device
    gpu_peak_start(device)
    log("final measurements (real contexts, canonical probe, s-space kernels, template accuracies) ...")
    rng, gen = measure_streams(seed, SLOT_FINAL, streams)
    final, extras = training_diagnostics(rssm, world, P_true, T_hat_B, replay, cfg, rng, gen, cfg["measure"]["iz_mc_samples_final"])
    final = {"step": step, **final}
    iz_result, probe, kernel_result, real = extras["iz"], extras["probe"], extras["kernel"], extras["real"]

    log("collapse diagnostics (I(z;s), posterior entropy, grid) ...")
    rng_izs, _ = measure_streams(seed, SLOT_IZS, streams)
    izs = latent_state_information(probe["posterior_probs"], M=cfg["measure"]["iz_mc_samples_final"], rng=rng_izs)
    grid_states = [s for cat in e3_env.ZONE_CATEGORIES for s in zone_states(layout, cat)[:2]]
    grid_name = f"collapse_grid_seed{seed:02d}.png"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    save_collapse_grid(rssm, world.templates_t, world.background_t, k, grid_states, run_dir / "artifacts" / grid_name)

    log("z-only probe (every replay posterior, in chunks) ...")
    rng_z, gen_z = measure_streams(seed, SLOT_ZONLY, streams)
    z_seq_len = cfg["measure"].get("z_only_seq_len", 64)
    z_probe_result = e3_probe.z_only_probe_chunked(rssm, replay.iter_chronological_windows(z_seq_len, chunk=256), n_states=N_STATES,
                                                   test_frac=cfg["measure"]["z_only_test_frac"], rng=rng_z, generator=gen_z,
                                                   max_images=max_images_per_forward(cfg))

    log("h-only probe (every replay deterministic state h_t, in chunks) ...")
    rng_h, gen_h = measure_streams(seed, SLOT_HONLY, streams)
    h_probe_result = e3_probe.z_only_probe_chunked(rssm, replay.iter_chronological_windows(z_seq_len, chunk=256), n_states=N_STATES,
                                                   test_frac=cfg["measure"]["z_only_test_frac"], rng=rng_h, generator=gen_h,
                                                   max_images=max_images_per_forward(cfg), latent="h")

    pc = cfg["measure"].get("is_per_context")
    per_context = None
    if pc:
        log(f"I_s per context vs pooled (secondary; {pc['states_per_zone']} states per zone, {pc['contexts']} contexts, "
            f"{pc['draws']} draws) ...")
        rng_pc, gen_pc = measure_streams(seed, SLOT_PER_CONTEXT, streams)
        per_context = per_context_is_check(rssm, world, real["contexts"], cfg, rng_pc, gen_pc, pc["states_per_zone"], pc["contexts"],
                                           pc["draws"])

    log("D1 in real contexts (every measured state of every zone with an accepted contrast) ...")
    rng_d1, gen_d1 = measure_streams(seed, SLOT_D1_REAL, streams)
    d1_real = d1_analysis_real(rssm, world, real["iz"], real["contexts"], cfg, rng_d1, gen_d1)
    d2_real = e3_probe.d2_neighbor_distances_contexts(real["contexts"], layout)
    log("D1 / D2 from the canonical probe (secondary) ...")
    rng_d1c, _ = measure_streams(seed, SLOT_D1_CANONICAL, streams)
    d1_canonical = d1_analysis(rssm, layout, themes, iz_result["I_z"], iz_result["se"], cfg, rng_d1c)
    d2_canonical = e3_probe.d2_neighbor_distances(probe["posterior_probs"], layout)

    # Context dispersion (spec 5.5), from the real contexts: per state, the gap between the real-context
    # mean and the canonical value, and the spread of I_z over the state's own contexts.
    gap = np.abs(real["iz"]["I_z"] - iz_result["I_z"])
    dispersion = {cat: {"mean_abs_gap_real_vs_canonical": float(np.nanmean(gap[zone_states(layout, cat)]))
                        if np.isfinite(gap[zone_states(layout, cat)]).any() else float("nan"),
                        "mean_sd_over_contexts": float(np.nanmean(real["iz"]["sd"][zone_states(layout, cat)]))
                        if np.isfinite(real["iz"]["sd"][zone_states(layout, cat)]).any() else float("nan")}
                  for cat in e3_env.ZONE_CATEGORIES}

    fields = {
        "final_step": step,
        "diagnostics_final": final,
        "zone_categories": list(e3_env.ZONE_CATEGORIES),
        "zone_kinds": world.zone_kinds,
        "zone_of_state": [zone_of_state(layout, s) for s in range(N_STATES)],
        "real_context": final["real_context"],
        "I_s_real": real["I_s"], "I_s_real_se": real["I_s_se"], "I_s_real_bias": real["I_s_bias"],
        "I_z_real": real["iz"]["I_z"], "I_z_real_se": real["iz"]["se"], "I_z_real_sd": real["iz"]["sd"],
        "n_contexts": real["iz"]["n_contexts"],
        "I_z": iz_result["I_z"], "I_z_se": iz_result["se"],
        "I_zs": izs["I_zs"], "I_zs_se": izs["se"],
        "canonical_posterior_entropy_mean": float(probe["posterior_entropy"].mean()),
        "posterior_entropy_ceiling": float(rssm.cfg.n_vars * np.log2(rssm.cfg.n_classes)),
        "collapse_grid": grid_name,
        "I_by_category": {cat: float(iz_result["I_z"][zone_states(layout, cat)].mean()) for cat in e3_env.ZONE_CATEGORIES},
        "I_by_category_real": {cat: final["real_context"]["I_z_by_zone"][cat]["mean"] for cat in e3_env.ZONE_CATEGORIES},
        "I_s_by_category_real": {cat: final["real_context"]["I_s_by_zone"][cat]["mean"] for cat in e3_env.ZONE_CATEGORIES},
        "I_s_projected_by_category_real": {cat: final["real_context"]["I_s_projected_by_zone"][cat]["mean"]
                                           for cat in e3_env.ZONE_CATEGORIES},
        "accuracies": {key: final[key] for key in ACC_LABELS},
        "ghost_fraction_mean": final["ghost_fraction_mean"],
        "z_only_probe": z_probe_result,
        "h_only_probe": h_probe_result,
        "is_per_context": per_context,
        "context_dispersion": dispersion,
        "d1": d1_real,
        "d1_canonical": d1_canonical,
        "d2_boundary_mean": float(d2_real["boundary"].mean()) if d2_real["boundary"].size else float("nan"),
        "d2_within_zone_mean": float(d2_real["within_zone"].mean()) if d2_real["within_zone"].size else float("nan"),
        "d2_canonical_boundary_mean": float(d2_canonical["boundary"].mean()),
        "d2_canonical_within_zone_mean": float(d2_canonical["within_zone"].mean()),
        "kl_by_zone": final["kl_by_zone"],
        "l1_vs_p_by_zone": final["l1_vs_p_by_zone"],
        "l1_vs_b_by_zone": final["l1_vs_b_by_zone"],
        "l1_b_vs_p_by_zone": final["l1_b_vs_p_by_zone"],
        "l1_floor_by_zone": final["l1_floor_by_zone"],
        "frequent_outcome_preview": final["frequent_outcome"],
        "gpu_memory_final": gpu_peak_end(device),
    }
    if fields["gpu_memory_final"] is not None:
        log(f"final measures: GPU peak {fields['gpu_memory_final']['peak_allocated_mib']:.0f} MiB allocated, "
            f"{fields['gpu_memory_final']['peak_reserved_mib']:.0f} MiB reserved")
    ctx = real["contexts"]
    arrays = {
        "N": N, "T_hat_B": T_hat_B, "P_true": P_true,
        "p_hat": kernel_result["p_hat"], "ghost_fraction": kernel_result["ghost_fraction"],
        "I_z": iz_result["I_z"], "I_z_se": iz_result["se"],
        "posterior_probs": probe["posterior_probs"], "prior_probs": probe["prior_probs"],
        "l1_vs_p": extras["l1_vs_p"], "l1_vs_b": extras["l1_vs_b"], "l1_b_vs_p": extras["l1_b_vs_p"],
        "real_I_s": real["I_s"], "real_I_s_se": real["I_s_se"], "real_I_s_bias": real["I_s_bias"],
        "real_I_s_projected": real["I_s_projected"], "real_I_s_projected_se": real["I_s_projected_se"],
        "real_I_s_projected_bias": real["I_s_projected_bias"], "real_p_hat_projected": real["kernel"]["p_hat_projected"],
        "real_ghost_fraction_projected": real["kernel"]["ghost_fraction_projected"], "real_off_support": real["off_support"],
        "real_I_z": real["iz"]["I_z"], "real_I_z_se": real["iz"]["se"], "real_I_z_sd": real["iz"]["sd"],
        "real_n_contexts": real["iz"]["n_contexts"], "real_I_z_context": real["iz"]["I_z_context"],
        "real_I_z_context_se": real["iz"]["se_context"], "real_kl_per_state": real["kl"]["mean"],
        "real_kl_next_bits": ctx["kl_next_bits"], "real_p_hat": real["kernel"]["p_hat"],
        "real_ghost_fraction": real["kernel"]["ghost_fraction"], "real_l1_vs_p": real["l1_vs_p"], "real_l1_vs_b": real["l1_vs_b"],
        "real_context_positions": ctx["positions"], "real_context_states": ctx["states"], "real_n_eligible": ctx["n_eligible"],
        "real_post_probs": ctx["post_probs"], "real_prior_probs": ctx["prior_probs"],
    }
    return fields, arrays


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
