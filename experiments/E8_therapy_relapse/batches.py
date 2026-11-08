"""E8's batch table (docs/E8_therapy_relapse.md, docs/e8/DECISIONS.md (15)). Standard library only. Every batch trains four
therapy runs in parallel on one pod (3 CPUs, one GPU: E5's pod); the probes go by job = the pairs that share a reference
at one checkpoint (DECISIONS (18)).

  e8_w1_base     base_A and base_B, exposed and references, the full exposure (the pilot: it sets wave 2's budget and
                 checkpoint grid)
  e8_w2_shuf_nw  shuffle A, E, F and reference B, the partial therapy on nw (validated after wave 1: DECISIONS (23); its
                 own config, configs/e8/e8_w2.yaml: budget 250,000, 16 checkpoints, 8 oracle points)
  e8_w3_shuf_ne, e8_w4_shuf_sw, e8_w5_shuf_se   the same four runs on another treated quadrant: generated, never launched
                 without the user

LAUNCHABLE lists the batches the submission accepts; a batch joins it only by the user's decision (wave 2 after wave 1's
validation, with its budget set in configs/e8/; the later quadrants on request). BATCH_CONFIG names a batch's config when
it is not CONFIG; the scripts pass it as --config (train, probes), so the default batches' command lines are unchanged.
"""

from __future__ import annotations

CONFIG = "configs/e8/e8.yaml"
SHUFFLE_MODELS = ("shuffle_A_exp", "shuffle_E_exp", "shuffle_F_exp", "shuffle_B_ref")
SHUFFLE_PAIRS = ("shuffle_A", "shuffle_E", "shuffle_F")


def _shuffle(quadrant: str) -> dict:
    return {"runs": {m: quadrant for m in SHUFFLE_MODELS}, "probe_jobs": [list(SHUFFLE_PAIRS)]}


# batch -> {"runs": {model id: treated quadrant or None (baseline)}, "probe_jobs": [[pairs sharing a reference], ...]}
BATCHES: dict[str, dict] = {
    "e8_w1_base": {"runs": {"base_A_exp": None, "base_A_ref": None, "base_B_exp": None, "base_B_ref": None},
                   "probe_jobs": [["base_A"], ["base_B"]]},
    "e8_w2_shuf_nw": _shuffle("nw"),
    "e8_w3_shuf_ne": _shuffle("ne"),
    "e8_w4_shuf_sw": _shuffle("sw"),
    "e8_w5_shuf_se": _shuffle("se"),
}
LAUNCHABLE = ("e8_w1_base", "e8_w2_shuf_nw")
BATCH_CONFIG = {"e8_w2_shuf_nw": "configs/e8/e8_w2.yaml"}  # DECISIONS (23); every other batch: CONFIG
BATCH_CPUS = {name: 3 for name in BATCHES}  # the quota: one pod of 3 CPUs (E5's)
PROBES_MAX_PARALLEL = 3


def train_args(batch: str) -> dict[str, list[str]]:
    """job name -> run_e8.py's train arguments (the runs root and device are added by the submission)."""
    out = {}
    config = config_args(batch)
    for model, q in BATCHES[batch]["runs"].items():
        out[model] = ["--phase", "train", "--model", model] + ([] if q is None else ["--treated-quadrant", q]) + config
    return out


def config_args(batch: str) -> list[str]:
    """["--config", <path>] for a batch with its own config (BATCH_CONFIG), [] for the default one (CONFIG)."""
    return ["--config", BATCH_CONFIG[batch]] if batch in BATCH_CONFIG else []
