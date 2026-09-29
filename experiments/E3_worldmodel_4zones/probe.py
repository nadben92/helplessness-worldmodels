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
from helplessness.agents.worldmodel.loop import encode_windows, to_chw_float, unroll_observation
from helplessness.agents.worldmodel.replay import ModelBatch, LabelBatch
from helplessness.agents.worldmodel.rssm import RSSM, categorical_kl, sample_categorical_index, straight_through_sample
from helplessness.measures.context_summary import per_state_context_summary
from helplessness.measures.diagnostics import alpha_app as alpha_app_fn
from helplessness.measures.diagnostics import jensen_shannon_product, mean_pairwise_jensen_shannon
from helplessness.measures.kernel_probe import kernel_from_matches, kernel_probe, match_templates
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
        out = unroll_observation(rssm, x_seq, a_seq, is_first_seq, generator=generator, decode=False)
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
    posterior, not a small random subsample (or use z_only_probe_chunked
    with replay.iter_chronological_windows for a large buffer). Also
    returns the mean posterior entropy (bits) over the same samples -- a
    collapse diagnostic computed at no extra cost from the same unroll."""
    return z_only_probe_chunked(rssm, [(model_batch, label_batch)], n_states, test_frac, rng, epochs=epochs, lr=lr)


def linear_state_probe(x: torch.Tensor, s: torch.Tensor, n_states: int, test_frac: float, rng: np.random.Generator,
                       epochs: int = 300, lr: float = 0.05) -> dict[str, float]:
    """A linear classifier x -> s (softmax regression, full batch, Adam),
    trained on a random (1 - test_frac) split drawn from `rng` and scored on
    the rest. The classifier's initial weights come from `rng` too (a forked
    CPU torch RNG seeded from it), never from the global torch RNG: the
    probe gives the same numbers whatever ran before it. x: (n, d) on the
    device, s: (n,) long on the same device."""
    device = x.device
    n = x.shape[0]
    perm = torch.as_tensor(rng.permutation(n), device=device)
    n_test = int(n * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(rng.integers(2**63 - 1)))
        classifier = torch.nn.Linear(x.shape[1], n_states).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = classifier(x[train_idx])
        loss = F.cross_entropy(logits, s[train_idx])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_acc = (classifier(x[train_idx]).argmax(-1) == s[train_idx]).float().mean().item()
        test_acc = (classifier(x[test_idx]).argmax(-1) == s[test_idx]).float().mean().item() if n_test > 0 else float("nan")
    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "n_train": int(len(train_idx)), "n_test": n_test}


def z_only_probe_chunked(
    rssm: RSSM, chunks, n_states: int, test_frac: float, rng: np.random.Generator, epochs: int = 300, lr: float = 0.05,
    generator: torch.Generator | None = None, max_images: int | None = None, latent: str = "z",
) -> dict[str, float]:
    """z_only_probe over an iterable of (ModelBatch, LabelBatch) chunks
    (audit B4): each chunk is unrolled without decoding and only z (and
    the posterior entropies) are kept, so the whole buffer never sits in
    one image tensor. The same windows in the same chunks give the same
    samples. max_images (measure.max_images_per_forward): the images of a
    chunk are encoded at most max_images at a time (loop.encode_windows);
    the latent draws, taken step by step over the whole chunk, are
    unchanged. latent="h" probes the deterministic state h_t instead of
    z_t (docs/e3/DECISIONS.md, 2026-09-30: how much of the state
    information travels through h)."""
    device = next(rssm.parameters()).device
    zs, labels, entropies = [], [], []
    with torch.no_grad():
        for model_batch, label_batch in chunks:
            embeds = encode_windows(rssm, model_batch.x, device, max_images)
            a_seq = torch.as_tensor(model_batch.a, dtype=torch.long, device=device)
            is_first_seq = torch.as_tensor(model_batch.is_first, dtype=torch.bool, device=device)
            out = unroll_observation(rssm, None, a_seq, is_first_seq, generator=generator, decode=False, embeds=embeds)
            post = out["post_probs"].reshape(-1, rssm.cfg.n_vars, rssm.cfg.n_classes).detach().cpu().numpy()
            entropies.append(per_variable_entropy(post))
            zs.append(out[latent].reshape(out[latent].shape[0] * out[latent].shape[1], -1))
            labels.append(np.asarray(label_batch.s).reshape(-1))
    mean_posterior_entropy = float(np.concatenate(entropies).mean())
    x = torch.cat(zs).to(device)
    s = torch.as_tensor(np.concatenate(labels), dtype=torch.long, device=device)
    return {**linear_state_probe(x, s, n_states, test_frac, rng, epochs=epochs, lr=lr),
            "mean_posterior_entropy": mean_posterior_entropy, "latent": latent}


def context_state_probes(contexts: dict, test_frac: float, rng: np.random.Generator, epochs: int = 300) -> dict[str, dict]:
    """Linear probes h_t -> s_t and z_t -> s_t on the real contexts
    themselves (docs/e3/DECISIONS.md, 2026-09-30): cheap enough for every
    training-diagnostic point, and read in the regime of the main measure
    (after the burn-in). One split per probe, both from `rng` (h first)."""
    s = torch.as_tensor(contexts["states"], dtype=torch.long, device=contexts["h"].device)
    n_states = int(contexts["n_states"])
    return {key: linear_state_probe(contexts[key].reshape(contexts[key].shape[0], -1), s, n_states, test_frac, rng, epochs=epochs)
            for key in ("h", "z")}


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
        out = unroll_observation(rssm, x_seq, a_seq, is_first_seq, decode=False)
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


# ---------------------------------------------------------------------
# the main measure in REAL contexts (docs/e3/DECISIONS.md, 2026-09-28):
# the contexts (h_t, z_t) of replay windows unrolled with their true
# history, instead of the learned initial state h0 of the canonical probe
# (which the model rarely sees with an arbitrary state, and from which
# the template accuracy was 0.40 against 0.98 in real contexts)
# ---------------------------------------------------------------------


def select_real_contexts(
    s_all: np.ndarray, is_first_all: np.ndarray, window_len: int, burn_in: int, max_per_state: int, n_states: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Which replay steps serve as contexts (labels in, positions out --
    no model involved). The buffer (logical order) is cut into consecutive
    windows of window_len steps, each followed by at least one more stored
    step (so a context at a window's last step still has its next step):
    n_windows = (n_steps - 1) // window_len. A step is eligible if it is
    not among the first `burn_in` steps of its window and is not an
    is_first step (both are the h0 regime). For each state, up to
    max_per_state eligible steps are drawn without replacement from rng
    (states in increasing order); every eligible step is kept when there
    are fewer. Returns positions (n,) sorted by state then position,
    states (n,), window (n,), t (n,), and n_eligible (n_states,)."""
    s_all, is_first_all = np.asarray(s_all), np.asarray(is_first_all, dtype=bool)
    n_windows = (len(s_all) - 1) // window_len
    if n_windows <= 0:
        raise ValueError(f"need more than window_len={window_len} stored steps, got {len(s_all)}")
    positions = np.arange(n_windows * window_len)
    t = positions % window_len
    eligible = positions[(t >= burn_in) & ~is_first_all[: n_windows * window_len]]
    eligible_states = s_all[eligible]
    order = np.argsort(eligible_states, kind="stable")
    eligible, eligible_states = eligible[order], eligible_states[order]
    bounds = np.searchsorted(eligible_states, np.arange(n_states + 1))
    chosen, n_eligible = [], np.zeros(n_states, dtype=np.int64)
    for s in range(n_states):
        cand = eligible[bounds[s]:bounds[s + 1]]
        n_eligible[s] = cand.size
        if cand.size > max_per_state:
            cand = np.sort(rng.choice(cand, size=max_per_state, replace=False))
        chosen.append(cand)
    positions = np.concatenate(chosen).astype(np.int64)
    return {"positions": positions, "states": s_all[positions].astype(np.int64), "window": positions // window_len,
            "t": positions % window_len, "n_eligible": n_eligible}


def real_context_probe(
    rssm: RSSM, replay, window_len: int, burn_in: int, max_per_state: int, n_states: int, rng: np.random.Generator,
    generator: torch.Generator | None = None, chunk: int = 256, max_images: int | None = None,
) -> dict:
    """The contexts of select_real_contexts, computed by unrolling their
    windows (window_len + 1 steps, from the learned initial state at the
    window's start, exactly as training unrolls a window) with hard
    straight-through posterior samples drawn from `generator`, `chunk`
    windows at a time; max_images (measure.max_images_per_forward): each
    chunk's images are encoded at most max_images at a time
    (loop.encode_windows), the draws unchanged. For each
    context (step t of a window, true state s_t):
      h, z          the context itself (torch, on the model's device): h_t
                    and the posterior sample z_t of the unroll;
      post_probs    q(z_t | h_t, e(x_t));
      prior_probs   for every action a, p(z_{t+1} | GRU(h_t, z_t, a)) --
                    the one-step predictions from the context
                    (imagination_step, as the canonical probe);
      h_next        GRU(h_t, z_t, a) for every action (torch);
      kl_next_bits  KL(q_{t+1} || p_{t+1}) in bits for the transition the
                    data actually took from the context (action a_t): the
                    model's one-step prediction error from state s_t, the
                    same pairing as the training loss; NaN when step t+1
                    starts a new episode (no transition).
    Labels select the contexts; they never enter the model."""
    device = next(rssm.parameters()).device
    s_all, is_first_all = replay.labels_and_flags()
    sel = select_real_contexts(s_all, is_first_all, window_len, burn_in, max_per_state, n_states, rng)
    n_ctx = sel["positions"].size
    V, C, A, H = rssm.cfg.n_vars, rssm.cfg.n_classes, rssm.cfg.n_actions, rssm.cfg.h_dim
    h_ctx = torch.zeros(n_ctx, H, device=device)
    z_ctx = torch.zeros(n_ctx, V, C, device=device)
    post_ctx = np.zeros((n_ctx, V, C))
    kl_next = np.full(n_ctx, np.nan)
    windows = np.unique(sel["window"])
    with torch.no_grad():
        for first in range(0, windows.size, chunk):
            block = windows[first:first + chunk]
            model_batch, _ = replay.windows(block * window_len, window_len + 1)
            embeds = encode_windows(rssm, model_batch.x, device, max_images)
            a_seq = torch.as_tensor(model_batch.a, dtype=torch.long, device=device)
            is_first_seq = torch.as_tensor(model_batch.is_first, dtype=torch.bool, device=device)
            out = unroll_observation(rssm, None, a_seq, is_first_seq, generator=generator, decode=False, embeds=embeds)
            members = np.flatnonzero(np.isin(sel["window"], block))
            row_np, t_np = np.searchsorted(block, sel["window"][members]), sel["t"][members]
            row, t = torch.as_tensor(row_np, device=device), torch.as_tensor(t_np, device=device)
            member_t = torch.as_tensor(members, device=device)
            h_ctx[member_t] = out["h"][row, t]
            z_ctx[member_t] = out["z"][row, t]
            post_ctx[members] = out["post_probs"][row, t].cpu().numpy()
            kl = categorical_kl(out["post_probs"][row, t + 1], out["prior_probs"][row, t + 1]).cpu().numpy() / np.log(2.0)
            kl_next[members] = np.where(np.asarray(model_batch.is_first)[row_np, t_np + 1], np.nan, kl)
        prior_ctx = torch.zeros(n_ctx, A, V, C, device=device)
        h_next = torch.zeros(n_ctx, A, H, device=device)
        step = chunk * 16
        for first in range(0, n_ctx, step):
            sl = slice(first, min(first + step, n_ctx))
            for a in range(A):
                a_onehot = F.one_hot(torch.full((sl.stop - sl.start,), a, device=device, dtype=torch.long),
                                     num_classes=A).to(z_ctx.dtype)
                out_a = rssm.imagination_step(h_ctx[sl], z_ctx[sl], a_onehot, generator=generator)
                prior_ctx[sl, a] = out_a["prior_probs"]
                h_next[sl, a] = out_a["h"]
    return {**sel, "h": h_ctx, "z": z_ctx, "post_probs": post_ctx, "prior_probs": prior_ctx.cpu().numpy(),
            "prior_probs_t": prior_ctx, "h_next": h_next, "kl_next_bits": kl_next, "n_states": n_states,
            "window_len": window_len, "burn_in": burn_in, "max_per_state": max_per_state}


def real_context_iz(contexts: dict, M: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """I_z of every context (the same exact-density estimator as the
    canonical probe, measures.latent_controllability), then per state
    (measures.context_summary): mean over the state's contexts, sd, number
    of contexts and the standard error of the mean. A state without a
    context is NaN ("non mesuré")."""
    n_ctx = contexts["positions"].size
    I_ctx, se_ctx = np.empty(n_ctx), np.empty(n_ctx)
    for i in range(n_ctx):
        result = latent_controllability(contexts["prior_probs"][i], M=M, rng=rng)
        I_ctx[i], se_ctx[i] = result["I_z"], result["se"]
    summary = per_state_context_summary(I_ctx, contexts["states"], contexts["n_states"], mc_se=se_ctx)
    return {"I_z_context": I_ctx, "se_context": se_ctx, "I_z": summary["mean"], "se": summary["se"], "sd": summary["sd"],
            "n_contexts": summary["n"]}


def real_context_kernel(
    rssm: RSSM, contexts: dict, templates: torch.Tensor, draws: int, margin_threshold: float,
    generator: torch.Generator | None = None, chunk: int = 1024,
) -> dict[str, np.ndarray]:
    """The s-space kernel (spec 5.3) from the real contexts: for every
    measured state s and action a, `draws` samples z' ~ p(. | h_{t+1,a}),
    the i-th draw from the state's (i mod n_s)-th context, decoded as
    decode(h_{t+1,a}, z') and matched to the n_states templates
    (measures.kernel_probe). p_hat (n_states, n_actions, n_states) and
    ghost_fraction (n_states, n_actions) are NaN for a state without a
    context."""
    device = next(rssm.parameters()).device
    n_states, A = contexts["n_states"], rssm.cfg.n_actions
    templates_np = templates.detach().cpu().numpy()
    p_hat = np.full((n_states, A, n_states), np.nan)
    ghost = np.full((n_states, A), np.nan)
    rows_ctx, rows_a, rows_owner = [], [], []
    for s in range(n_states):
        ctx_s = np.flatnonzero(contexts["states"] == s)
        if ctx_s.size == 0:
            continue
        ctx = ctx_s[np.arange(draws) % ctx_s.size]
        for a in range(A):
            rows_ctx.append(ctx)
            rows_a.append(np.full(draws, a))
            rows_owner.append(np.full(draws, s * A + a))
    if not rows_ctx:
        return {"p_hat": p_hat, "ghost_fraction": ghost}
    rows_ctx, rows_a, rows_owner = np.concatenate(rows_ctx), np.concatenate(rows_a), np.concatenate(rows_owner)
    nearest = np.empty(rows_ctx.size, dtype=np.int64)
    is_ghost = np.empty(rows_ctx.size, dtype=bool)
    with torch.no_grad():
        for first in range(0, rows_ctx.size, chunk):
            sl = slice(first, min(first + chunk, rows_ctx.size))
            c = torch.as_tensor(rows_ctx[sl], device=device)
            a = torch.as_tensor(rows_a[sl], device=device)
            probs = contexts["prior_probs_t"][c, a]
            z = F.one_hot(sample_categorical_index(probs, generator), num_classes=probs.shape[-1]).to(probs.dtype)
            decoded = rssm.decode(contexts["h_next"][c, a], z).cpu().numpy()
            matched = match_templates(decoded, templates_np, margin_threshold)
            nearest[sl], is_ghost[sl] = matched["nearest_state"], matched["is_ghost"]
    for owner in np.unique(rows_owner):
        idx = rows_owner == owner
        result = kernel_from_matches(nearest[idx], is_ghost[idx], n_states)
        s, a = divmod(int(owner), A)
        p_hat[s, a], ghost[s, a] = result["p_hat"], result["ghost_fraction"]
    return {"p_hat": p_hat, "ghost_fraction": ghost}


def real_context_action_samples(
    rssm: RSSM, contexts: dict, templates: torch.Tensor, draws: int, margin_threshold: float,
    generator: torch.Generator | None = None, chunk: int = 1024,
) -> dict[str, np.ndarray]:
    """Action-labelled prior draws from the real contexts (docs/e3/DECISIONS.md,
    2026-09-29, test of whether z' encodes the action beyond the arrival
    state): for every context and every action a, `draws` samples
    z' ~ p(. | h_{t+1,a}) (the same one-step predictions as
    real_context_kernel), each decoded as decode(h_{t+1,a}, z') and matched to
    the templates (measures.kernel_probe). Returns, one row per draw: context
    (index into the contexts), s (the context's true state), a, z_index
    (n_vars ints, the drawn class of each latent variable), s_next (the
    decoded arrival state) and is_ghost."""
    device = next(rssm.parameters()).device
    A = rssm.cfg.n_actions
    templates_np = templates.detach().cpu().numpy()
    n_ctx = contexts["positions"].size
    ctx = np.repeat(np.arange(n_ctx), A * draws)
    act = np.tile(np.repeat(np.arange(A), draws), n_ctx)
    z_index = np.empty((ctx.size, rssm.cfg.n_vars), dtype=np.int64)
    nearest = np.empty(ctx.size, dtype=np.int64)
    is_ghost = np.empty(ctx.size, dtype=bool)
    with torch.no_grad():
        for first in range(0, ctx.size, chunk):
            sl = slice(first, min(first + chunk, ctx.size))
            c = torch.as_tensor(ctx[sl], device=device)
            a = torch.as_tensor(act[sl], device=device)
            probs = contexts["prior_probs_t"][c, a]
            idx = sample_categorical_index(probs, generator)
            z = F.one_hot(idx, num_classes=probs.shape[-1]).to(probs.dtype)
            decoded = rssm.decode(contexts["h_next"][c, a], z).cpu().numpy()
            matched = match_templates(decoded, templates_np, margin_threshold)
            z_index[sl], nearest[sl], is_ghost[sl] = idx.cpu().numpy(), matched["nearest_state"], matched["is_ghost"]
    return {"context": ctx, "s": contexts["states"][ctx], "a": act, "z_index": z_index, "s_next": nearest, "is_ghost": is_ghost}


def real_context_template_accuracy(rssm: RSSM, contexts: dict, templates: torch.Tensor, chunk: int = 1024) -> float:
    """Template accuracy of the posterior reconstruction decode(h_t, z_t)
    at the real contexts themselves (after the burn-in, no h0 step)."""
    templates_np = templates.detach().cpu().numpy()
    n_ctx = contexts["positions"].size
    nearest = np.empty(n_ctx, dtype=np.int64)
    with torch.no_grad():
        for first in range(0, n_ctx, chunk):
            sl = slice(first, min(first + chunk, n_ctx))
            decoded = rssm.decode(contexts["h"][sl], contexts["z"][sl]).cpu().numpy()
            nearest[sl] = match_templates(decoded, templates_np, margin_threshold=-np.inf)["nearest_state"]
    return float((nearest == contexts["states"]).mean()) if n_ctx else float("nan")


def real_context_d1_iswap(
    rssm: RSSM, contexts: dict, layout, themes, s: int, target_category: str, k: int, global_swap: bool, M: int,
    rng: np.random.Generator, generator: torch.Generator | None = None,
) -> dict[str, float]:
    """D1 in real contexts: in every context of state s, the observation
    x_t is replaced by the counterfactual render (the agent's cell
    re-themed as target_category, or -- global -- the two zones' themes
    swapped on the whole map); the history h_t is kept. z_t is a hard
    sample of q(. | h_t, e(x_cf)) and I_z is computed from the one-step
    predictions, as for the real image. Returns I_swap (mean over the
    contexts), its standard error (sd / sqrt(n); the MC se for a single
    context) and n; NaN when s has no context."""
    from helplessness.envs.observation.allocentric_pixel_backend import render_counterfactual

    idx = np.flatnonzero(contexts["states"] == s)
    if idx.size == 0:
        return {"I_swap": float("nan"), "I_swap_se": float("nan"), "n": 0}
    own_category = layout.zone_at(divmod(s, layout.grid_shape[1])).appearance.category
    if global_swap:
        img = render_counterfactual(s, layout, themes, k=k, global_category_swap=(own_category, target_category))
    else:
        img = render_counterfactual(s, layout, themes, k=k, cell_category_override=target_category)
    device = next(rssm.parameters()).device
    x = torch.as_tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
    A = rssm.cfg.n_actions
    with torch.no_grad():
        h = contexts["h"][torch.as_tensor(idx, device=device)]
        e = rssm.encoder(x).expand(idx.size, -1)
        z = straight_through_sample(rssm.posterior_probs(h, e), generator)
        priors = []
        for a in range(A):
            a_onehot = F.one_hot(torch.full((idx.size,), a, device=device, dtype=torch.long), num_classes=A).to(z.dtype)
            priors.append(rssm.imagination_step(h, z, a_onehot, generator=generator)["prior_probs"])
        priors = torch.stack(priors, dim=1).cpu().numpy()  # (n, A, V, C)
    values, mc_se = np.empty(idx.size), np.empty(idx.size)
    for i in range(idx.size):
        result = latent_controllability(priors[i], M=M, rng=rng)
        values[i], mc_se[i] = result["I_z"], result["se"]
    se = float(values.std(ddof=1) / np.sqrt(idx.size)) if idx.size >= 2 else float(mc_se[0])
    return {"I_swap": float(values.mean()), "I_swap_se": se, "n": int(idx.size)}


def d2_neighbor_distances_contexts(contexts: dict, layout) -> dict[str, np.ndarray]:
    """D2 in real contexts: for each pair of neighbouring cells (s, s2),
    the mean over i < min(n_s, n_s2) of the JS distance
    (measures.diagnostics.jensen_shannon_product) between the posterior of
    the i-th context of s and that of the i-th context of s2; pairs with an
    unmeasured cell are left out. Split into "boundary" (different
    appearance category) and "within_zone" pairs, as d2_neighbor_distances."""
    n_rows, n_cols = layout.grid_shape
    post = contexts["post_probs"]
    ctx_of_state = [np.flatnonzero(contexts["states"] == s) for s in range(n_rows * n_cols)]
    boundary, within_zone = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            s = r * n_cols + c
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = (r + dr) % n_rows, (c + dc) % n_cols
                s2 = r2 * n_cols + c2
                n = min(ctx_of_state[s].size, ctx_of_state[s2].size)
                if n == 0:
                    continue
                js = float(np.mean([jensen_shannon_product(post[ctx_of_state[s][i]], post[ctx_of_state[s2][i]]) for i in range(n)]))
                same = layout.zone_at((r, c)).appearance.category == layout.zone_at((r2, c2)).appearance.category
                (within_zone if same else boundary).append(js)
    return {"boundary": np.array(boundary), "within_zone": np.array(within_zone)}
