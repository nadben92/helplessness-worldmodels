"""The two optional accelerations of the training step (docs/e3/DECISIONS.md,
2026-09-30 (8)) against the current step: the same losses and weights to a
floating-point tolerance over a few steps, in both estimator phases."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch

from helplessness.agents.worldmodel.acceleration import (
    StackedAdam,
    StackedRSSM,
    compile_from_config,
    compile_rssm,
    max_relative_difference,
    stacked_train_step,
)
from helplessness.agents.worldmodel.loop import WorldModelLossConfig, set_learning_rate, train_world_model_step
from helplessness.agents.worldmodel.replay import ModelBatch
from helplessness.agents.worldmodel.rssm import RSSM, LatentEstimator, RSSMConfig

CFG = RSSMConfig(image_size=(24, 24), base_depth=4, embed_dim=16, h_dim=16, n_vars=3, n_classes=4)
PROBS, ST = LatentEstimator("probs"), LatentEstimator("straight_through")


def _model(seed: int) -> RSSM:
    torch.manual_seed(seed)
    return RSSM(CFG)


def _batch(rng: np.random.Generator, B: int = 3, T: int = 6) -> ModelBatch:
    first = np.zeros((B, T), dtype=bool)
    first[:, 0] = True
    first[0, 3] = True  # an episode start inside a window
    z = np.zeros((B, T))
    return ModelBatch(x=rng.integers(0, 256, size=(B, T, 24, 24, 3), dtype=np.uint8), a=rng.integers(0, 5, size=(B, T)),
                      r=z, terminal=z.astype(bool), truncated=z.astype(bool), is_first=first)


def _background(seed: int) -> torch.Tensor:
    return torch.rand(3, 24, 24, generator=torch.Generator().manual_seed(100 + seed))


def _sequential(models, batches, schedule, lrs, loss_cfgs, clip_norm):
    """The current step, one seed after the other, each with its own Adam and generator."""
    opts = [torch.optim.Adam(m.parameters(), lr=1e-3, eps=1e-8) for m in models]
    gens = [torch.Generator().manual_seed(7 + i) for i in range(len(models))]
    out = []
    for step, ests in enumerate(schedule):
        row = []
        for i, m in enumerate(models):
            set_learning_rate(opts[i], lrs[step][i])
            row.append(train_world_model_step(m, opts[i], batches[step][i], _background(i), loss_cfgs[i], clip_norm=clip_norm,
                                              estimator=ests[i], generator=gens[i]))
        out.append(row)
    return out


def test_stacked_seeds_match_the_sequential_step_in_both_phases():
    """Two seeds, different weights, batches, backgrounds, beta_rep and learning rates; the second switches to
    straight-through one step before the first; each draws from its own generator. Losses and final weights
    agree with the per-seed sequential step."""
    rng = np.random.default_rng(0)
    n_steps = 5
    batches = [[_batch(rng) for _ in range(2)] for _ in range(n_steps)]
    schedule = [(PROBS, PROBS), (PROBS, PROBS), (PROBS, ST), (ST, ST), (ST, ST)]
    lrs = [(1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 3e-4), (3e-4, 3e-4), (3e-4, 3e-4)]
    loss_cfgs = [WorldModelLossConfig(k=2, beta_rep=0.1), WorldModelLossConfig(k=2, beta_rep=0.05)]
    clip = 0.5  # small enough to bind: the per-seed clipping is exercised

    seq_models = [_model(0), _model(1)]
    stacked_models = copy.deepcopy(seq_models)
    ref = _sequential(seq_models, batches, schedule, lrs, loss_cfgs, clip)

    stack = StackedRSSM(stacked_models)
    opt = StackedAdam(stack.parameters(), stack.n, lr=1e-3, eps=1e-8)
    gens = [torch.Generator().manual_seed(7 + i) for i in range(2)]
    got = []
    for step, ests in enumerate(schedule):
        opt.lr = list(lrs[step])
        got.append(stacked_train_step(stack, opt, batches[step], [_background(0), _background(1)], loss_cfgs, list(ests), gens,
                                      clip_norm=clip))
    for r, g in zip(ref, got):
        assert max_relative_difference(g, r) < 1e-4
        assert all(abs(x["grad_norm"] - y["grad_norm"]) <= 1e-4 * y["grad_norm"] for x, y in zip(g, r))
    assert any(r[0]["grad_norm"] > clip for r in ref)
    stack.write_back(stacked_models)
    for a, b in zip(seq_models, stacked_models):
        for (name, p), q in zip(a.named_parameters(), b.parameters()):
            assert torch.allclose(p, q, atol=2e-5, rtol=1e-4), name


def test_stacked_step_consumes_each_seed_s_stream_as_its_sequential_step():
    rng = np.random.default_rng(1)
    batches = [_batch(rng), _batch(rng)]
    models = [_model(0), _model(1)]
    g_seq = [torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)]
    for i, m in enumerate(copy.deepcopy(models)):
        train_world_model_step(m, torch.optim.Adam(m.parameters(), lr=1e-3), batches[i], _background(i), WorldModelLossConfig(k=2),
                               estimator=ST, generator=g_seq[i])
    stack = StackedRSSM(models)
    g_st = [torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)]
    stacked_train_step(stack, StackedAdam(stack.parameters(), 2, lr=1e-3), batches, [_background(0), _background(1)],
                       [WorldModelLossConfig(k=2)] * 2, [ST, ST], g_st)
    for a, b in zip(g_seq, g_st):
        assert torch.equal(a.get_state(), b.get_state())


def test_models_must_share_the_architecture():
    with pytest.raises(ValueError, match="same RSSMConfig"):
        StackedRSSM([_model(0), RSSM(dataclasses.replace(CFG, h_dim=8))])


def _compiled_equivalence(backend: str, mode: str | None):
    rng = np.random.default_rng(2)
    batches = [_batch(rng) for _ in range(4)]
    eager, compiled = _model(0), _model(0)
    compile_rssm(compiled, backend=backend, mode=mode)
    losses = {}
    for name, m in (("eager", eager), ("compiled", compiled)):
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(5)
        losses[name] = [train_world_model_step(m, opt, mb, _background(0), WorldModelLossConfig(k=2), estimator=est, generator=gen)
                        for mb, est in zip(batches, (PROBS, PROBS, ST, ST))]
    assert max_relative_difference(losses["compiled"], losses["eager"]) < 1e-4
    assert set(compiled.state_dict()) == set(eager.state_dict())  # compiled in place: checkpoints unchanged
    for (name, p), q in zip(eager.named_parameters(), compiled.parameters()):
        assert torch.allclose(p, q, atol=2e-5, rtol=1e-4), name


def test_compiled_step_matches_the_eager_step_traced():
    """Dynamo tracing and AOT autograd, without code generation (runs on any machine)."""
    torch._dynamo.reset()
    _compiled_equivalence("aot_eager", None)


def test_compiled_step_matches_the_eager_step_inductor():
    """Inductor's generated code; skipped where it cannot compile (no C++ compiler, e.g. a bare Windows CPU)."""
    torch._dynamo.reset()
    try:
        _compiled_equivalence("inductor", None)
    except Exception as err:  # the backend's own failure to build, not a numerical difference
        if isinstance(err, AssertionError):
            raise
        pytest.skip(f"inductor unavailable here: {type(err).__name__}: {str(err).splitlines()[0][:200]}")


def test_compile_is_off_unless_the_config_asks():
    m = _model(0)
    assert compile_from_config(m, {"runtime": {"device": "cpu"}}) is m and not hasattr(m, "compiled_with")
    assert compile_from_config(m, {"runtime": {"compile": None}}) is m and not hasattr(m, "compiled_with")
    compile_from_config(m, {"runtime": {"compile": {"backend": "aot_eager"}}})
    assert m.compiled_with == {"backend": "aot_eager", "mode": None}


def test_compile_gate_passes_on_equivalence_and_refuses_otherwise(monkeypatch):
    """DECISIONS 2026-09-30 (9): the gate in front of runtime.compile: losses within tolerance and TF32 off once compiled;
    a backend that fails to build is a refusal (error), never a pass."""
    from helplessness.agents.worldmodel.acceleration import check_compile_equivalence

    torch._dynamo.reset()
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)  # the state run_mini.configure_runtime leaves (cudnn defaults to True)
    ok = check_compile_equivalence(lambda: _model(0), (3, 6), torch.device("cpu"), backend="aot_eager", k=2)
    assert ok["passed"] and ok["error"] is None and ok["max_rel_loss_diff"] < 1e-4
    assert ok["tf32_after"] == {"matmul": False, "cudnn": False, "float32_matmul_precision": "highest"}
    bad = check_compile_equivalence(lambda: _model(0), (3, 6), torch.device("cpu"), backend="no_such_backend", k=2)
    assert not bad["passed"] and bad["error"]
    strict = check_compile_equivalence(lambda: _model(0), (3, 6), torch.device("cpu"), backend="aot_eager", k=2, tol=-1.0)
    assert not strict["passed"] and "differ" in strict["error"]


def test_tf32_on_after_compile_fails_the_gate(monkeypatch):
    from helplessness.agents.worldmodel import acceleration as acc

    torch._dynamo.reset()
    monkeypatch.setattr(acc, "tf32_flags", lambda: {"matmul": True, "cudnn": False, "float32_matmul_precision": "highest"})
    result = acc.check_compile_equivalence(lambda: _model(0), (3, 6), torch.device("cpu"), backend="aot_eager", k=2)
    assert not result["passed"] and "TF32" in result["error"]


def test_a_campaign_may_not_mix_execution_modes():
    from helplessness.agents.worldmodel.acceleration import execution_mode, mixed_modes

    eager, compiled = {"runtime": {"device": "cuda"}}, {"runtime": {"compile": {"backend": "inductor", "mode": None}}}
    assert execution_mode(eager) == {"compile": None, "stacked_seeds": None}
    assert execution_mode({"runtime": {"compile": None, "stacked_seeds": 1}}) == execution_mode(eager)
    assert mixed_modes({"a": execution_mode(eager), "b": execution_mode({"runtime": {}})}) == []
    assert mixed_modes({"a": execution_mode(compiled), "b": execution_mode(compiled)}) == []
    problems = mixed_modes({"env": execution_mode(compiled), "ref": execution_mode(eager)})
    assert len(problems) == 1 and "env" in problems[0] and "ref" in problems[0]
    assert mixed_modes({"a": execution_mode(eager), "b": execution_mode({"runtime": {"stacked_seeds": 2}})})
