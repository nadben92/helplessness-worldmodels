"""Two optional accelerations of the world model's training step
(docs/e3/DECISIONS.md, 2026-09-30 (8)). Neither is on by default; each is
tested for equivalence with the current step (tests/agents/worldmodel/
test_acceleration.py: the same losses and weights to a floating-point
tolerance over a few steps) and timed against it by the preflight's speed
bench (experiments/E3_worldmodel_4zones/preflight.py --bench-accel), which
only measures the gain -- the decision to use either is taken on those
numbers, not here.

1. compile_rssm: torch.compile of the step's three dense pieces -- the
   encoder, the decoder and RSSM.observation_core (one recurrent step
   without its posterior draw, so the latents' random stream stays
   outside the compiled graph and is consumed exactly as before). The
   module is compiled in place: its parameters, state_dict and every
   caller are unchanged. mode="reduce-overhead" is inductor's CUDA-graph
   mode (the recurrence is bound by kernel launches on a GPU).
   Config: runtime.compile = {backend: ..., mode: ...}; absent or null =
   off (compile_from_config).

2. StackedRSSM / stacked_train_step: several seeds' models trained in one
   process, their parameters stacked (torch.func.stack_module_state) and
   every dense call vectorised over the seeds (torch.func.vmap +
   functional_call). What must stay per seed stays per seed, outside the
   vectorised calls: each seed's own minibatch, its latent draws from its
   own generator (the same calls, in the same order, as its sequential
   step: its stream is consumed identically), its loss configuration
   (beta_rep), its estimator phase and learning rate (seeds switch at
   different steps), its gradient-norm clipping. The optimiser is Adam
   written for stacked parameters with one learning rate per seed
   (StackedAdam; Adam is elementwise, so this is each seed's own Adam).
   Not wired into run_mini.py / run_agent.py: the runners train one seed
   per process, and a multi-seed runner is only worth writing if the bench
   shows the gain.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from helplessness.agents.worldmodel.loop import WorldModelLossConfig, losses_from_unroll, to_chw_float
from helplessness.agents.worldmodel.replay import ModelBatch
from helplessness.agents.worldmodel.rssm import RSSM, LatentEstimator, sample_posterior


# ---------------------------------------------------------------------
# 1. torch.compile
# ---------------------------------------------------------------------


def compile_rssm(rssm: RSSM, backend: str = "inductor", mode: str | None = None, dynamic: bool | None = None) -> RSSM:
    """Compiles, in place, the encoder, the decoder and observation_core.
    Same parameters, same state_dict; the posterior draws stay eager (see
    the module docstring). mode is inductor's (None, "default",
    "reduce-overhead" = CUDA graphs, "max-autotune"); other backends take
    no mode ("aot_eager": tracing only, for the tests)."""
    kwargs = {"backend": backend, "dynamic": dynamic}
    if mode is not None:
        kwargs["mode"] = mode
    rssm.encoder.compile(**kwargs)
    rssm.decoder.compile(**kwargs)
    rssm.observation_core = torch.compile(rssm.observation_core, **kwargs)
    rssm.compiled_with = {"backend": backend, "mode": mode}
    return rssm


def compile_from_config(rssm: RSSM, cfg: dict) -> RSSM:
    """runtime.compile ({backend, mode}) -- absent or null: the model is
    returned untouched (every config before 2026-09-30 (8))."""
    opts = (cfg.get("runtime") or {}).get("compile")
    if not opts:
        return rssm
    return compile_rssm(rssm, backend=opts.get("backend", "inductor"), mode=opts.get("mode"))


# ---------------------------------------------------------------------
# 2. several seeds in one process
# ---------------------------------------------------------------------


class _Call(nn.Module):
    """functional_call calls forward(); this forward dispatches to any RSSM method."""

    def __init__(self, rssm: RSSM):
        super().__init__()
        self.rssm = rssm

    def forward(self, method: str, *args):
        return getattr(self.rssm, method)(*args)


class _GRUCellByGates(nn.GRUCell):
    """nn.GRUCell's own equations written with linear layers (same
    parameters): vmap has a batching rule for these and none for
    aten::gru_cell, which it would otherwise run seed by seed."""

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r_i, z_i, n_i = F.linear(x, self.weight_ih, self.bias_ih).chunk(3, dim=-1)
        r_h, z_h, n_h = F.linear(h, self.weight_hh, self.bias_hh).chunk(3, dim=-1)
        r, z = torch.sigmoid(r_i + r_h), torch.sigmoid(z_i + z_h)
        n = torch.tanh(n_i + r * n_h)
        return (1.0 - z) * n + z * h


class StackedRSSM:
    """n models of one architecture, their parameters stacked along a new
    first dimension (leaf tensors: the gradients of a summed loss land on
    them, each seed's slice receiving its own model's gradient)."""

    def __init__(self, models: list[RSSM]):
        if len({repr(m.cfg) for m in models}) != 1:
            raise ValueError("StackedRSSM: every model must have the same RSSMConfig")
        self.n = len(models)
        self.cfg = models[0].cfg
        params, buffers = torch.func.stack_module_state(models)
        self.params = {f"rssm.{k}": v for k, v in params.items()}
        self.buffers = {f"rssm.{k}": v for k, v in buffers.items()}
        skeleton = RSSM(self.cfg).to("meta")
        skeleton.gru = _GRUCellByGates(self.cfg.h_dim, self.cfg.h_dim, device="meta")
        self._call = _Call(skeleton)

    @property
    def device(self) -> torch.device:
        return next(iter(self.params.values())).device

    def vcall(self, method: str, *args, in_dims: tuple):
        """rssm.<method>(*args) for every seed at once; in_dims: 0 for an
        argument stacked over the seeds, None for one shared by all."""
        def one(p, b, *a):
            return torch.func.functional_call(self._call, {**p, **b}, (method, *a))
        return torch.func.vmap(one, in_dims=(0, 0, *in_dims))(self.params, self.buffers, *args)

    @torch.no_grad()
    def write_back(self, models: list[RSSM]) -> None:
        """Copies each seed's slice into its own model (checkpoints, measures)."""
        for i, model in enumerate(models):
            for name, p in model.named_parameters():
                p.copy_(self.params[f"rssm.{name}"][i])

    def parameters(self):
        return list(self.params.values())


class StackedAdam:
    """torch.optim.Adam (no weight decay, no amsgrad), on stacked
    parameters, with one learning rate per seed: the same elementwise
    update as each seed's own Adam."""

    def __init__(self, params: list[torch.Tensor], n: int, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params, self.n = params, n
        self.lr = [float(lr)] * n
        self.betas, self.eps = betas, eps
        self.exp_avg = [torch.zeros_like(p) for p in params]
        self.exp_avg_sq = [torch.zeros_like(p) for p in params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2_sqrt = 1.0 - b1 ** self.t, math.sqrt(1.0 - b2 ** self.t)
        for p, m, v in zip(self.params, self.exp_avg, self.exp_avg_sq):
            g = p.grad
            m.lerp_(g, 1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
            denom = (v.sqrt() / bc2_sqrt).add_(self.eps)
            step_size = torch.tensor([lr / bc1 for lr in self.lr], dtype=p.dtype, device=p.device).view(-1, *([1] * (p.dim() - 1)))
            p.sub_(step_size * (m / denom))


def _per_seed_clip(params: list[torch.Tensor], n: int, clip_norm: float | None) -> torch.Tensor:
    """clip_grad_norm_ applied to each seed's slice separately; returns the
    pre-clip global norm of every seed (n,)."""
    norms = torch.stack([torch.linalg.vector_norm(p.grad.reshape(n, -1), dim=1) for p in params], dim=1)  # (n, n_params)
    total = torch.linalg.vector_norm(norms, dim=1)
    max_norm = float("inf") if clip_norm is None else clip_norm
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for p in params:
        p.grad.mul_(coef.view(-1, *([1] * (p.dim() - 1))).to(p.grad.dtype))
    return total


def stacked_train_step(
    stack: StackedRSSM,
    optimizer: StackedAdam,
    batches: list[ModelBatch],
    backgrounds: list[torch.Tensor],
    loss_cfgs: list[WorldModelLossConfig],
    estimators: list[LatentEstimator],
    generators: list[torch.Generator | None],
    clip_norm: float | None = 1000.0,
    return_posteriors: bool = False,
) -> list[dict]:
    """One training step of every seed: loop.train_world_model_step's
    computation, the seeds vectorised. Returns one dict per seed, as
    train_world_model_step(sync=True) returns (plus "post_probs" when
    asked). The learning rates are the optimizer's (StackedAdam.lr)."""
    n, device = stack.n, stack.device
    optimizer.zero_grad()
    x = torch.stack([to_chw_float(mb.x, device) for mb in batches])                      # (n, B, T, C, H, W)
    a = torch.stack([torch.as_tensor(mb.a, dtype=torch.long, device=device) for mb in batches])
    first = torch.stack([torch.as_tensor(mb.is_first, dtype=torch.bool, device=device) for mb in batches])
    B, T = x.shape[1], x.shape[2]
    emb = stack.vcall("encoder", x.reshape(n, B * T, *x.shape[3:]), in_dims=(0,)).reshape(n, B, T, -1)
    a_onehot = F.one_hot(a.clamp(min=0), num_classes=stack.cfg.n_actions).to(emb.dtype)

    h, z = stack.vcall("initial_state", B, device, in_dims=(None, None))
    a_prev = torch.zeros(n, B, stack.cfg.n_actions, device=device, dtype=emb.dtype)
    hs, zs, priors, posts = [], [], [], []
    for t in range(T):
        h, prior, post, _ = stack.vcall("observation_core", h, z, a_prev, None, first[:, :, t], None, emb[:, :, t],
                                        in_dims=(0, 0, 0, None, 0, None, 0))
        # each seed's draw from its own stream, the same call as in its sequential step
        z = torch.stack([sample_posterior(post[i], estimators[i], generators[i]) for i in range(n)])
        hs.append(h)
        zs.append(z)
        priors.append(prior)
        posts.append(post)
        a_prev = a_onehot[:, :, t]
    h_seq, z_seq = torch.stack(hs, dim=2), torch.stack(zs, dim=2)
    decoded = stack.vcall("decode", h_seq.reshape(n, B * T, -1), z_seq.reshape(n, B * T, *z_seq.shape[3:]), in_dims=(0, 0))
    decoded = decoded.reshape(n, B, T, *decoded.shape[2:])
    prior_seq, post_seq = torch.stack(priors, dim=2), torch.stack(posts, dim=2)

    losses = [losses_from_unroll(x[i], {"decoded": decoded[i], "prior_probs": prior_seq[i], "post_probs": post_seq[i]},
                                 backgrounds[i], loss_cfgs[i]) for i in range(n)]
    torch.stack([loss["total"] for loss in losses]).sum().backward()
    grad_norm = _per_seed_clip(stack.parameters(), n, clip_norm)
    optimizer.step()
    out = []
    for i in range(n):
        result = {k: float(v.detach()) for k, v in losses[i].items()} | {"grad_norm": float(grad_norm[i])}
        if return_posteriors:
            result["post_probs"] = post_seq[i].reshape(-1, *post_seq.shape[3:]).detach().cpu().numpy()
        out.append(result)
    return out


def max_relative_difference(a: list[dict], b: list[dict], keys=("total", "recon", "kl_dyn", "kl_rep"), floor: float = 1e-3) -> float:
    """Largest |a - b| / max(|b|, floor) over the loss terms of two lists of
    step results (the floor: a KL term near 0 early in training has no
    meaningful relative difference)."""
    return float(max(abs(x[k] - y[k]) / max(abs(y[k]), floor) for x, y in zip(a, b) for k in keys)) if a else float("nan")



# ---------------------------------------------------------------------
# what a campaign may mix, and the gate in front of runtime.compile (DECISIONS 2026-09-30 (9))
# ---------------------------------------------------------------------


def execution_mode(cfg: dict) -> dict:
    """The execution mode of a config: compile (None, or {backend, mode}) and stacked seeds (None, or the number of
    seeds in one process). The modes are not bit-equivalent to each other, so every run of a campaign
    (environment, reference, E3-0 if rerun) must have the same one: mixed_modes() is what the preflight refuses on."""
    runtime = cfg.get("runtime") or {}
    opts = runtime.get("compile")
    stacked = runtime.get("stacked_seeds")
    return {"compile": None if not opts else {"backend": opts.get("backend", "inductor"), "mode": opts.get("mode")},
            "stacked_seeds": None if not stacked or int(stacked) <= 1 else int(stacked)}


def mixed_modes(modes: dict[str, dict]) -> list[str]:
    """{config name: execution_mode} -> the problems (empty when every config has the same mode)."""
    distinct = {repr(sorted(m.items(), key=lambda kv: kv[0])) for m in modes.values()}
    if len(distinct) <= 1:
        return []
    return ["the campaign mixes execution modes (runs are not bit-equivalent across compile / stacked seeds): "
            + "; ".join(f"{name}: {mode}" for name, mode in modes.items())]


def tf32_flags() -> dict:
    return {"matmul": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def check_compile_equivalence(make_model, batch_shape: tuple[int, int], device: torch.device, backend: str = "inductor",
                              mode: str | None = None, n_steps: int = 4, tol: float = 1e-4, k: int = 6, seed: int = 0) -> dict:
    """The gate of runtime.compile. make_model() builds the run's model (same weights every call); the eager and
    the compiled model take the same n_steps training steps (half per phase) from the same weights, batches and
    generators, on `device`. passed: the largest relative loss difference is <= tol AND TF32 is off (matmul,
    cudnn, float32_matmul_precision "highest") after the compiled steps. Any exception of the compilation itself
    is reported as a failure (error), never swallowed into a pass."""
    import numpy as np

    result = {"backend": backend, "mode": mode, "tol": tol, "passed": False, "max_rel_loss_diff": float("nan"), "tf32_after": None,
              "error": None}
    try:
        rng = np.random.default_rng(seed)
        cfg = make_model().cfg
        H, W = cfg.image_size
        B, T = batch_shape
        batches = []
        for _ in range(n_steps):
            first = np.zeros((B, T), dtype=bool)
            first[:, 0] = True
            z = np.zeros((B, T))
            batches.append(ModelBatch(x=torch.as_tensor(rng.integers(0, 256, size=(B, T, H, W, cfg.in_channels), dtype=np.uint8)),
                                      a=rng.integers(0, cfg.n_actions, size=(B, T)), r=z, terminal=z.astype(bool),
                                      truncated=z.astype(bool), is_first=first))
        background = torch.rand(cfg.in_channels, H, W, generator=torch.Generator().manual_seed(seed))
        estimators = [LatentEstimator("probs")] * (n_steps // 2) + [LatentEstimator("straight_through")] * (n_steps - n_steps // 2)
        loss_cfg = WorldModelLossConfig(k=k)
        from helplessness.agents.worldmodel.loop import train_world_model_step

        losses = {}
        for name in ("eager", "compiled"):
            model = make_model().to(device)
            if name == "compiled":
                torch._dynamo.reset()
                compile_rssm(model, backend=backend, mode=mode)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)
            gen = torch.Generator(device=device).manual_seed(seed)
            losses[name] = [train_world_model_step(model, opt, mb, background.to(device), loss_cfg, clip_norm=1000.0, estimator=est,
                                                   generator=gen) for mb, est in zip(batches, estimators)]
        result["max_rel_loss_diff"] = max_relative_difference(losses["compiled"], losses["eager"])
        result["tf32_after"] = tf32_flags()
        tf32_off = not (result["tf32_after"]["matmul"] or result["tf32_after"]["cudnn"]) and \
            result["tf32_after"]["float32_matmul_precision"] == "highest"
        result["passed"] = bool(result["max_rel_loss_diff"] <= tol and tf32_off)
        if not tf32_off:
            result["error"] = f"TF32 is on after the compiled steps: {result['tf32_after']}"
        elif not result["passed"]:
            result["error"] = f"compiled losses differ from eager by {result['max_rel_loss_diff']:.2e} (> {tol:g})"
    except Exception as err:  # the compilation's own failure: a refusal, not a pass
        result["error"] = f"{type(err).__name__}: {str(err).splitlines()[0][:300] if str(err) else ''}"
    return result
