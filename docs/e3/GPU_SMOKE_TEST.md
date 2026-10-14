# E3 — GPU smoke test (Google Colab, T4)

**Superseded (2026-09-28): the Colab test was cancelled. The GPU check is now `experiments/E3_worldmodel_4zones/preflight.py` (see the E3 README, "First GPU run on the A100"). Kept as the record of the protocol prepared on 2026-09-25; its commands use the retired mini.yaml.**

Goal: in about 15 minutes, check that the E3 code runs on CUDA, that
everything is on the GPU, and that memory use and the first curves are
plausible. **Bit-for-bit equality with the CPU is not expected**: the
random draws and the floating-point reductions differ between devices.

Taken from the GPU readiness audit (`docs/e3/AUDIT_GPU_2026-09-25.md`, §5; at the repo root as `audit_gpu.md` until 2026-10-14), updated after the
device support was added (docs/e3/DECISIONS.md, 2026-09-25).

## Step 0 — Install

Runtime → Change runtime type → T4 GPU. Then, in notebook cells:

```bash
!nvidia-smi
!python --version
!git clone https://github.com/nadben92/helplessness-worldmodels.git
%cd helplessness-worldmodels
!git checkout e3/worldmodel
!pip install -q -e ".[e3,dev]"     # Colab already ships a CUDA torch >= 2.2; pip keeps it
!python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name(0))"
!pip freeze | grep -iE "^(torch|numpy|scipy|matplotlib|pyyaml)=="
```

Record the `pip freeze` line: it is the version set the future lock file
will pin.

Optional (a few minutes, CPU):
`!python -m pytest tests/agents/worldmodel tests/measures tests/e3/test_e3_mini_report.py -q`.

## Step 1 — A training step and a probe on CUDA (≈ 1 minute)

**1a.** Paste into a Python cell (checked on CPU before publication):

```python
import time
import numpy as np
import torch
from helplessness.experiment_loader import load_experiment_module
from helplessness.agents.worldmodel.loop import WorldModelLossConfig, train_world_model_step
from helplessness.agents.worldmodel.replay import ModelBatch
from helplessness.agents.worldmodel.rssm import RSSM, RSSMConfig

torch.backends.cuda.matmul.allow_tf32 = False   # as run_mini.py does
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.get_device_name(0) if device.type == "cuda" else "-")
torch.manual_seed(0)
k, B, T = 6, 8, 16
side = 12 * k
rssm = RSSM(RSSMConfig(image_size=(side, side), base_depth=8, embed_dim=64, h_dim=64,
                       n_vars=8, n_classes=8, n_actions=5)).to(device)
assert all(p.device.type == device.type for p in rssm.parameters()), "parameters not on the target device"
opt = torch.optim.Adam(rssm.parameters(), lr=1e-3, eps=1e-8)
gen = torch.Generator(device=device)
gen.manual_seed(0)
rng = np.random.default_rng(0)
mb = ModelBatch(
    x=rng.integers(0, 255, (B, T, side, side, 3), dtype=np.uint8), a=rng.integers(0, 5, (B, T)),
    r=np.zeros((B, T), np.float32), terminal=np.zeros((B, T), bool), truncated=np.zeros((B, T), bool),
    is_first=np.zeros((B, T), bool),
)
bg = torch.rand(3, side, side).to(device)
cfg = WorldModelLossConfig(k=k, w_patch=10.0)
train_world_model_step(rssm, opt, mb, bg, cfg, clip_norm=None, generator=gen)   # warm-up
if device.type == "cuda":
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
for _ in range(20):
    out = train_world_model_step(rssm, opt, mb, bg, cfg, clip_norm=None, generator=gen)
if device.type == "cuda":
    torch.cuda.synchronize()
print(f"20 train steps: {20 / (time.perf_counter() - t0):.1f} steps/s | total {out['total']:.2f} | grad_norm {out['grad_norm']:.1f}")
assert all(np.isfinite(v) for v in out.values()), "non-finite loss"
if device.type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

probe = load_experiment_module("E3_worldmodel_4zones", "probe.py", "gpu_smoke_probe")
templates = torch.rand(144, 3, side, side)   # CPU on purpose: the probes move their inputs to the model's device
print("canonical_separation:", {kk: round(v, 4) for kk, v in probe.canonical_separation(rssm, templates, n_eval_draws=1, generator=gen).items()})
print("OK: everything ran on", device)
```

The images are random noise, so the loss is large (≈ 2·10⁴). That is
expected: this step checks devices, not learning. Reference on the
development CPU: `20 train steps: 2.2 steps/s`; the last line reads
`OK: everything ran on cpu`.

**1b.** Throughput and memory at the full model size (batch 16×64,
`base_depth=32, h_dim=256`) — this also gives the training VRAM:

```bash
!python experiments/E3_worldmodel_4zones/profile_throughput.py --device cuda --n-env-steps 500 --n-train-steps 20 --batch-size 16 --seq-len 64
!nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

The first line printed must be `device: cuda`.

## Step 2 — The real entry point, one seed, truncated (≈ 10 minutes)

```bash
!python experiments/E3_worldmodel_4zones/run_mini.py --config configs/e3/mini.yaml --seeds 0 --device cuda --max-steps 600 --out-dir results 2>&1 | tee run_mini_colab.log
```

While it runs, in another cell: `!nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 5`.

`--max-steps 600` stops training after 600 gradient steps. The final
measurements still run: canonical probe, s-space kernel, D1, z-only probe.
The report is marked "Truncated by --max-steps". The run directory is
`results/E3_worldmodel_4zones/grid12x12__4zones__pair-experimental__worldmodel-mini__seed00__<UTC>/`.

The same command on the development CPU (`--device cpu`, 6 threads, torch
2.14.0+cpu, commit of 2026-09-25) is the reference for the curves below:

```
[seed=0] step      0/600 | batch-pair JS  0.000 bits | labelled JS  0.000 bits | phase probabilities    | beta_rep 0.0000 | recon_patch 67.4189 | grad_norm 1.98e+04 | lr 0.001
[seed=0] step    500/600 | batch-pair JS  0.052 bits | labelled JS  0.068 bits | phase probabilities    | beta_rep 0.0000 | recon_patch 2.9019 | grad_norm 140 | lr 0.001
[seed=0] training done in 210.2s
[seed=0] diagnostics @ 600: KL det [0.03, 0.03] bits | acc replay/canonical-hard/canonical-mode 0.072/0.019/0.035 | 149s
```

At step 0 nothing has been trained yet and the model is built on the CPU
before being moved, so `recon_patch` at step 0 should match 67.42 up to
float differences (a few units in the last digits). A large gap points to
a problem with the initialisation or the data, not to GPU arithmetic.

## What to check

| # | Check | Expected |
|---|---|---|
| 1 | No crash | No `Expected all tensors to be on the same device`, no CUDA error. Exit code 0. |
| 2 | Everything on `cuda` | 1a: the assertions pass and the last line says `cuda`. 1b: first line `device: cuda`. Step 2: the first log line says `device cuda`, `meta.yaml` has `device: cuda`, and `nvidia-smi` shows the Python process using memory. |
| 3 | TF32 off | `meta.yaml`: `allow_tf32_matmul: false`, `allow_tf32_cudnn: false`. |
| 4 | Memory | 1a (mini model): a few hundred MB at most. 1b: note the peak; it must fit in the T4's 15 GB, and therefore in an A100. |
| 5 | Shape of the curves | Compare with the CPU reference lines. `recon_patch` drops by an order of magnitude within the first 500 steps. The batch-pair JS stays near 0 while the probabilities phase has not separated. In 600 steps the switch has not happened (the isolation test puts it around steps 1700-3100), so the report says `Effective switch step: None`. Exact values are not expected to match. |
| 6 | Speed | The CPU does ≈ 2.9 steps/s on the mini model (600 steps in 210 s above). Note the GPU rate (1a and step 2's log timestamps). A GPU rate of the same order as the CPU points to kernel-launch overhead (audit, 7.1), not to a device problem. |
| 7 | Outputs complete | The run directory holds `config.yaml`, `meta.yaml` (`status: done`), `console.log`, `diagnostics.jsonl` (1 line: the final point), `results.json`, `arrays.npz`, `REPORT.md` and `artifacts/*.png`. |

Colab limits: its ~12 GB of RAM is enough for the mini runs (replay of
82 MB for `mini.yaml`, 311 MB for `mini_long.yaml`), not for a
1e6-step buffer (15.6 GB — Phase 4a, B6). Sessions expire, so there is no
long run on Colab until periodic checkpoints and `--resume` exist
(Phase 4a, B5).
