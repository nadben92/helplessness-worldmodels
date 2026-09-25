"""The canonical probe and everything built on it (spec sections 5 and
8): the one place that does real forward passes through a frozen RSSM
for measurement purposes. Torch-dependent -- lives here, not in
measures/ (adjustment B, docs/e3/DECISIONS.md): every function below
calls the RSSM's own public methods (encoder, posterior_probs,
gru_step, imagination_step, decode) and NEVER reimplements the forward
pass; latent_hook is exercised exactly as the agent exercises it
(imagination_step/decode both route through it already).

Every actual MEASURE (entropy, KL, alpha_app, JS distance, ghost
classification) is computed by calling into measures/latent_controllability.py,
measures/kernel_probe.py, measures/diagnostics.py on the plain numpy
arrays this module produces -- never recomputed here.

Device contract (docs/e3/DECISIONS.md, 2026-09-25): every function takes
its device from the model's parameters and moves its own tensor inputs
there (templates, images); callers may pass CPU tensors. Randomness:
every function that draws accepts an optional torch.Generator
(`generator`, on the model's device) so measurements never consume the
training stream; None falls back to the global torch RNG.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from helplessness.agents.worldmodel.agent import categorical_mode, sample_categorical_batch
from helplessness.agents.worldmodel.loop import to_chw_float, unroll_observation
from helplessness.agents.worldmodel.replay import ModelBatch, LabelBatch
from helplessness.agents.worldmodel.rssm import RSSM, straight_through_sample
from helplessness.measures.diagnostics import alpha_app as alpha_app_fn
from helplessness.measures.diagnostics import jensen_shannon_product, mean_pairwise_jensen_shannon
from helplessness.measures.kernel_probe import kernel_probe
from helplessness.measures.latent_controllability import latent_controllability, latent_state_information, per_variable_entropy


# ---------------------------------------------------------------------
# 5.1 -- canonical probe
# ---------------------------------------------------------------------


def canonical_probe(rssm: RSSM, templates: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, np.ndarray]:
    """For every one of the n_states templates (spec 5.1): h0 (fresh
    episode start), z0 = mode of q(.|h0, enc(x)) (never a stochastic
    sample -- see agent.categorical_mode), posterior entropy, and for
    every action: h1_a = GRU(h0, z0, a), p_a = prior(h1_a) (already
    unimix'd, hook applied -- via RSSM.imagination_step)."""
    device = next(rssm.parameters()).device
    n_states = templates.shape[0]
    n_actions = rssm.cfg.n_actions

    with torch.no_grad():
        x = templates.to(device)
        h0, _ = rssm.initial_state(n_states, device)
        e = rssm.encoder(x)
        post_probs = rssm.posterior_probs(h0, e)
        z0 = categorical_mode(post_probs)

        h1_per_action, prior_per_action = [], []
        for a in range(n_actions):
            a_onehot = F.one_hot(torch.full((n_states,), a, device=device, dtype=torch.long), num_classes=n_actions).to(z0.dtype)
            out = rssm.imagination_step(h0, z0, a_onehot, generator=generator)
            h1_per_action.append(out["h"])
            prior_per_action.append(out["prior_probs"])
        h1 = torch.stack(h1_per_action, dim=1)  # (n_states, n_actions, h_dim)
        prior_probs = torch.stack(prior_per_action, dim=1)  # (n_states, n_actions, n_vars, n_classes)

    post_probs_np = post_probs.detach().cpu().numpy()
    return {
        "z0_mode": z0.detach().cpu().numpy(),
        "posterior_probs": post_probs_np,
        "posterior_entropy": per_variable_entropy(post_probs_np),
        "h0": h0.detach().cpu().numpy(),
        "h1": h1.detach().cpu().numpy(),
        "prior_probs": prior_probs.detach().cpu().numpy(),
    }


def single_state_probe(rssm: RSSM, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Same as canonical_probe, for ONE arbitrary image x (1, C, H, W)
    -- used by the D1 diagnostics on counterfactual renders, which are
    never part of `templates` (spec section 8: "ces images ne servent
    jamais a l'entrainement" -- nor to the canonical probe's own
    template gallery)."""
    return canonical_probe(rssm, x)


# ---------------------------------------------------------------------
# 5.2 -- I_z(s) for every state
# ---------------------------------------------------------------------


def canonical_iz(
    rssm: RSSM, templates: torch.Tensor, M: int, rng: np.random.Generator, generator: torch.Generator | None = None
) -> dict[str, np.ndarray]:
    probe = canonical_probe(rssm, templates, generator=generator)
    n_states = probe["prior_probs"].shape[0]
    I_z = np.empty(n_states)
    se = np.empty(n_states)
    H_marg = np.empty(n_states)
    H_cond = np.empty(n_states)
    for s in range(n_states):
        result = latent_controllability(probe["prior_probs"][s], M=M, rng=rng)
        I_z[s], se[s], H_marg[s], H_cond[s] = result["I_z"], result["se"], result["H_marginal"], result["H_conditional_mean"]
    return {"I_z": I_z, "se": se, "H_marginal": H_marg, "H_conditional_mean": H_cond, "probe": probe}


# ---------------------------------------------------------------------
# 5.3 -- kernel in s-space, for every (s, a)
# ---------------------------------------------------------------------


def canonical_kernel_probe(
    rssm: RSSM, templates: torch.Tensor, K_s: int, margin_threshold: float, probe: dict[str, np.ndarray] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, np.ndarray]:
    device = next(rssm.parameters()).device
    if probe is None:
        probe = canonical_probe(rssm, templates, generator=generator)
    n_states, n_actions = probe["h1"].shape[0], probe["h1"].shape[1]
    templates_np = templates.detach().cpu().numpy()

    h1_t = torch.as_tensor(probe["h1"], device=device)
    prior_t = torch.as_tensor(probe["prior_probs"], device=device)
    p_hat = np.zeros((n_states, n_actions, n_states))
    ghost_fraction = np.zeros((n_states, n_actions))

    with torch.no_grad():
        for s in range(n_states):
            for a in range(n_actions):
                h1_sa = h1_t[s, a].unsqueeze(0).expand(K_s, -1)
                samples = sample_categorical_batch(prior_t[s, a], K_s, generator=generator)
                decoded = rssm.decode(h1_sa, samples).detach().cpu().numpy()
                result = kernel_probe(decoded, templates_np, margin_threshold)
                p_hat[s, a] = result["p_hat"]
                ghost_fraction[s, a] = result["ghost_fraction"]

    return {"p_hat": p_hat, "ghost_fraction": ghost_fraction}


def template_match_accuracy(
    rssm: RSSM, model_batch: ModelBatch, label_batch: LabelBatch, templates: torch.Tensor, margin_threshold: float,
    generator: torch.Generator | None = None,
) -> float:
    """Calibration criterion C-a (spec section 7): fraction of VISITED
    states whose POSTERIOR reconstruction (decode(h_t, z_t), z_t from
    the real posterior, not the prior) matches its own true template
    under nearest-L2 matching.

    What exactly this measures (docs/e3/DECISIONS.md, 2026-09-25): REPLAY
    windows (real contexts h_t, unrolled from each window's start), z_t a
    HARD straight-through sample of the posterior (one draw per step).
    Not the canonical probe: see canonical_separation for the fresh-h0
    path (hard sample and mode)."""
    device = next(rssm.parameters()).device
    x_seq = to_chw_float(model_batch.x, device)
    a_seq = torch.as_tensor(model_batch.a, dtype=torch.long, device=device)
    is_first_seq = torch.as_tensor(model_batch.is_first, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = unroll_observation(rssm, x_seq, a_seq, is_first_seq, generator=generator)
        decoded = rssm.decoder(out["h"].reshape(-1, rssm.cfg.h_dim), out["z"].reshape(-1, rssm.cfg.latent_dim))
    decoded_np = decoded.detach().cpu().numpy()
    templates_np = templates.detach().cpu().numpy()
    matched = kernel_probe(decoded_np, templates_np, margin_threshold=-1e18)  # margin_threshold<0 -> never a ghost
    predicted_s = matched["nearest_state"]
    true_s = label_batch.s.reshape(-1)
    return float((predicted_s == true_s).mean())


# ---------------------------------------------------------------------
# 5.4 -- z-only linear probe
# ---------------------------------------------------------------------


def z_only_probe(
    rssm: RSSM, model_batch: ModelBatch, label_batch: LabelBatch, n_states: int, test_frac: float, rng: np.random.Generator, epochs: int = 300, lr: float = 0.05
) -> dict[str, float]:
    """A linear classifier z -> s, trained on REAL posterior samples
    from the replay (spec 5.4), evaluated on a held-out split -- checks
    that z "carries" the state, without claiming this proves the state
    is USED (see CLAUDE.md's own caveat, docs/E3_worldmodel_4zones.md).
    Pass replay.chronological_windows(...) to train on every available
    posterior, not a small random subsample. Also returns the mean
    posterior entropy (bits) over the same samples -- a collapse
    diagnostic computed at no extra cost from the same unroll."""
    device = next(rssm.parameters()).device
    x_seq = to_chw_float(model_batch.x, device)
    a_seq = torch.as_tensor(model_batch.a, dtype=torch.long, device=device)
    is_first_seq = torch.as_tensor(model_batch.is_first, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = unroll_observation(rssm, x_seq, a_seq, is_first_seq)
    post_probs = out["post_probs"].reshape(-1, rssm.cfg.n_vars, rssm.cfg.n_classes).detach().cpu().numpy()
    mean_posterior_entropy = float(per_variable_entropy(post_probs).mean())
    z = out["z"].reshape(-1, rssm.cfg.latent_dim).to(device)
    s = torch.as_tensor(label_batch.s.reshape(-1), dtype=torch.long, device=device)

    n = z.shape[0]
    perm = torch.as_tensor(rng.permutation(n), device=device)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    classifier = torch.nn.Linear(rssm.cfg.latent_dim, n_states).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = classifier(z[train_idx])
        loss = F.cross_entropy(logits, s[train_idx])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_acc = (classifier(z[train_idx]).argmax(-1) == s[train_idx]).float().mean().item()
        test_acc = (classifier(z[test_idx]).argmax(-1) == s[test_idx]).float().mean().item() if n_test > 0 else float("nan")

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "n_train": int(len(train_idx)),
        "n_test": n_test,
        "mean_posterior_entropy": mean_posterior_entropy,
    }


# ---------------------------------------------------------------------
# 5.5 -- context dispersion
# ---------------------------------------------------------------------


def context_dispersion(
    rssm: RSSM,
    model_batch: ModelBatch,
    label_batch: LabelBatch,
    canonical_I_z: np.ndarray,
    M: int,
    rng: np.random.Generator,
    n_samples: int = 100,
) -> dict[str, np.ndarray]:
    """For n_samples real (h, z, true-label-s) triples drawn from the
    replay (via the SAME unroll_observation the training loop uses),
    computes I_z from that REAL context and compares it to the
    canonical probe's own I_z(s) for the same true state -- spec 5.5."""
    device = next(rssm.parameters()).device
    x_seq = to_chw_float(model_batch.x, device)
    a_seq = torch.as_tensor(model_batch.a, dtype=torch.long, device=device)
    is_first_seq = torch.as_tensor(model_batch.is_first, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = unroll_observation(rssm, x_seq, a_seq, is_first_seq)
    h_all = out["h"].reshape(-1, rssm.cfg.h_dim)
    z_all = out["z"].reshape(-1, rssm.cfg.n_vars, rssm.cfg.n_classes)
    s_all = label_batch.s.reshape(-1)

    n = h_all.shape[0]
    idx = rng.choice(n, size=min(n_samples, n), replace=False)

    context_I_z, canonical_I_z_matched, true_states = [], [], []
    n_actions = rssm.cfg.n_actions
    with torch.no_grad():
        for i in idx:
            h_i = h_all[i : i + 1]
            z_i = z_all[i : i + 1]
            s_i = int(s_all[i])
            priors = []
            for a in range(n_actions):
                a_onehot = F.one_hot(torch.tensor([a], device=device), num_classes=n_actions).to(z_i.dtype)
                out_a = rssm.imagination_step(h_i, z_i, a_onehot)
                priors.append(out_a["prior_probs"][0].detach().cpu().numpy())
            prior_probs_per_action = np.stack(priors, axis=0)
            result = latent_controllability(prior_probs_per_action, M=M, rng=rng)
            context_I_z.append(result["I_z"])
            canonical_I_z_matched.append(canonical_I_z[s_i])
            true_states.append(s_i)

    context_I_z = np.array(context_I_z)
    canonical_I_z_matched = np.array(canonical_I_z_matched)
    return {
        "context_I_z": context_I_z,
        "canonical_I_z": canonical_I_z_matched,
        "true_states": np.array(true_states),
        "abs_discrepancy": np.abs(context_I_z - canonical_I_z_matched),
    }


# ---------------------------------------------------------------------
# section 8 -- D1 (alpha_app) and D2 (neighbour JS distances)
# ---------------------------------------------------------------------


def d1_alpha_app_for_state(
    rssm: RSSM,
    layout,
    themes,
    s: int,
    target_category: str,
    I_by_category: dict[str, float],
    M: int,
    rng: np.random.Generator,
    k: int,
    contrast_threshold: float,
    global_swap: bool,
) -> dict[str, float | None]:
    """One state's D1-local (global_swap=False) or D1-global
    (global_swap=True) alpha_app, spec section 8. I_by_category: mean
    canonical I_z per zone appearance category (normal images)."""
    from helplessness.envs.observation.allocentric_pixel_backend import render_counterfactual

    own_category = layout.zone_at(divmod(s, layout.grid_shape[1])).appearance.category
    if global_swap:
        img = render_counterfactual(s, layout, themes, k=k, global_category_swap=(own_category, target_category))
    else:
        img = render_counterfactual(s, layout, themes, k=k, cell_category_override=target_category)

    device = next(rssm.parameters()).device
    x = torch.as_tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
    probe = single_state_probe(rssm, x)
    result = latent_controllability(probe["prior_probs"][0], M=M, rng=rng)
    I_swap = result["I_z"]

    I_L = I_by_category[own_category]
    I_Lprime = I_by_category[target_category]
    alpha = alpha_app_fn(I_swap, I_L, I_Lprime, contrast_threshold)
    return {"I_swap": I_swap, "I_swap_se": result["se"], "alpha_app": alpha, "I_L": I_L, "I_Lprime": I_Lprime}


def d2_neighbor_distances(posterior_probs: np.ndarray, layout) -> dict[str, np.ndarray]:
    """JS distance (measures.diagnostics.jensen_shannon_product) between
    canonical-probe posterior probabilities at neighbouring cells,
    split into "boundary" (different appearance category) and
    "within_zone" (same category) pairs -- spec section 8, D2."""
    n_rows, n_cols = layout.grid_shape
    boundary, within_zone = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            cat_s = layout.zone_at((r, c)).appearance.category
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = (r + dr) % n_rows, (c + dc) % n_cols
                s2 = r2 * n_cols + c2
                cat_s2 = layout.zone_at((r2, c2)).appearance.category
                js = jensen_shannon_product(posterior_probs[s], posterior_probs[s2])
                (within_zone if cat_s == cat_s2 else boundary).append(js)
    return {"boundary": np.array(boundary), "within_zone": np.array(within_zone)}


# ---------------------------------------------------------------------
# posterior separation between states (collapse diagnostic, used during
# training by run_mini.py and by the autoencoder isolation test)
# ---------------------------------------------------------------------


def canonical_separation(
    rssm: RSSM, templates: torch.Tensor, n_eval_draws: int = 5, deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Canonical path (fresh h0, z from the posterior of render(s)) for
    the n_states templates: nearest-template accuracy of the DISCRETE
    reconstruction with a hard one-hot sample (averaged over
    n_eval_draws draws) and with the posterior mode, mean posterior
    entropy, and mean pairwise Jensen-Shannon divergence between the
    states' posteriors. deterministic=True evaluates z = the posterior
    probabilities instead of a hard sample (the no-bottleneck reference).

    Labels (docs/e3/DECISIONS.md, 2026-09-25): "acc_hard" = CANONICAL
    PROBE, HARD SAMPLE; "acc_mode" = CANONICAL PROBE, MODE. The canonical
    path uses the learned initial state h0 for every state, a context the
    model sees only at window starts, so it is expected to read lower
    than the replay accuracy (template_match_accuracy)."""
    device = next(rssm.parameters()).device
    templates = templates.to(device)
    templates_np = templates.detach().cpu().numpy()

    def accuracy(h0: torch.Tensor, z: torch.Tensor) -> float:
        recon = rssm.decode(h0, z).detach().cpu().numpy()
        matched = kernel_probe(recon, templates_np, margin_threshold=-np.inf)
        return float((matched["nearest_state"] == np.arange(templates_np.shape[0])).mean())

    with torch.no_grad():
        h0, _ = rssm.initial_state(templates.shape[0], templates.device)
        post = rssm.posterior_probs(h0, rssm.encoder(templates))
        if deterministic:
            acc_hard = accuracy(h0, post)
        else:
            acc_hard = float(np.mean([accuracy(h0, straight_through_sample(post, generator)) for _ in range(n_eval_draws)]))
        acc_mode = accuracy(h0, categorical_mode(post))
    post_np = post.detach().cpu().numpy()
    return {
        "acc_hard": acc_hard,
        "acc_mode": acc_mode,
        "posterior_entropy": float(per_variable_entropy(post_np).mean()),
        "mean_js": mean_pairwise_jensen_shannon(post_np),
    }


def latent_usage_diagnostic(rssm: RSSM, templates: torch.Tensor, M: int, rng: np.random.Generator, n_eval_draws: int = 5) -> dict[str, float]:
    """How the model is using its latent (docs/e3/DECISIONS.md,
    2026-09-24 -- the diagnostic saved when SeparationNotReachedError is
    raised). Canonical path, 144 templates:
      acc_probs  -- template accuracy with the decoder fed the posterior
                    PROBABILITIES (what it sees during the warm-up phase);
      acc_mode   -- fed the one-hot MODE;
      acc_hard   -- fed a hard one-hot SAMPLE (mean over n_eval_draws);
      posterior_entropy (bits), mean_js between states (bits),
      I_zs = I(z; s) over the canonical posteriors (existing estimator),
      logit_norm_centered -- mean L2 norm of the posterior logits per
                    variable after removing their mean over classes (the
                    softmax-relevant magnitude; ~0 = near-uniform
                    posteriors), and logit_norm_raw (without centring).
    A gap acc_probs >> acc_hard with near-uniform posteriors is the
    signature of state information carried by tiny probability
    differences that a discrete sample cannot transmit."""
    templates = templates.to(next(rssm.parameters()).device)
    hard = canonical_separation(rssm, templates, n_eval_draws=n_eval_draws)
    probs = canonical_separation(rssm, templates, deterministic=True)
    with torch.no_grad():
        h0, _ = rssm.initial_state(templates.shape[0], templates.device)
        e = rssm.encoder(templates)
        logits = rssm.posterior_net(torch.cat([h0, e], dim=-1)).reshape(-1, rssm.cfg.n_vars, rssm.cfg.n_classes)
        centered = logits - logits.mean(dim=-1, keepdim=True)
        post = rssm.posterior_probs(h0, e).detach().cpu().numpy()
    izs = latent_state_information(post, M=M, rng=rng)
    return {
        "acc_probs": probs["acc_hard"],
        "acc_mode": hard["acc_mode"],
        "acc_hard": hard["acc_hard"],
        "posterior_entropy": hard["posterior_entropy"],
        "mean_js": hard["mean_js"],
        "I_zs": izs["I_zs"],
        "I_zs_se": izs["se"],
        "logit_norm_centered": float(centered.norm(dim=-1).mean()),
        "logit_norm_raw": float(logits.norm(dim=-1).mean()),
    }
