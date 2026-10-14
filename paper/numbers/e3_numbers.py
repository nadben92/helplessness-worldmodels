"""The E3 report's generated numbers (paper/reports/e3_report.tex). Each entry reads one source:
  - a config key (configs/e3/*.yaml, configs/zones/e1_4zones.yaml, experiments/E3_worldmodel_4zones/config.yaml), or a
    plain arithmetic combination of config keys, stated in its src;
  - a results file of an E3 run or diagnostic (results/E3_worldmodel_4zones/..., reports/...), through the readers of
    paper/figures/e3_sources.py and, where a quantity needs computing, the experiment's own functions
    (aggregate_b, measures.controllability, measures.kernel_probe);
  - the final campaign (pending=True): PendingSource until it is extracted (e3_sources.CAMPAIGN_ROOT).
Numbers stated only in docs/e3/DECISIONS.md, or only in the notes, are not generated: they are written in the report
as they are, with their % src: and % num-ok.
"""

from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import yaml

from registry import REPO_ROOT, Number, PendingSource, SourceError

sys.path.insert(0, str(REPO_ROOT / "paper" / "figures"))
import e3_sources as src  # noqa: E402

from helplessness.measures.controllability import controllability_per_state  # noqa: E402
from helplessness.measures.kernel_probe import draws_for_l1_floor, expected_l1_sampling_floor  # noqa: E402

ZONES_CFG = "configs/zones/e1_4zones.yaml"
EPISODE_CFG = "experiments/E3_worldmodel_4zones/config.yaml"
CAMPAIGN_CFG = "configs/e3/gpu_campaign.yaml"
PILOT_CFG = "configs/e3/gpu_pilot_closed_loop.yaml"
MINI_CFG = "configs/e3/agent_mini.yaml"
E30_CFG = "configs/e3/gpu_e30.yaml"
E30B64_CFG = "configs/e3/gpu_e30b_win64.yaml"
E30B16_CFG = "configs/e3/gpu_e30b_win16.yaml"
E3_RES = "results/E3_worldmodel_4zones"
AGENT_MINI_RUN = "grid12x12__4zones__pair-experimental__worldmodel-agent-agent_mini__seed00__20260930T193212Z"
OLD_K_S = 32          # C-b's draws per (s, a) before the correction (docs/e3/DECISIONS.md, 2026-09-29 (3))
L1_FLOOR_TARGET = 0.02  # the floor K_s was chosen for (docs/e3/DECISIONS.md, 2026-09-29 (3))


# --- configs -----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cfg(rel: str) -> dict:
    path = REPO_ROOT / rel
    if not path.is_file():
        raise SourceError(f"Missing source file: {rel}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def key(rel: str, dotted: str):
    node = _cfg(rel)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SourceError(f"{rel}: no key {dotted!r}")
        node = node[part]
    return node


def zone_size() -> int:
    blocks = key(ZONES_CFG, "blocks")
    sizes = {(b["rows"][1] - b["rows"][0] + 1, b["cols"][1] - b["cols"][0] + 1) for b in blocks.values()}
    if len(sizes) != 1 or len({*next(iter(sizes))}) != 1:
        raise SourceError(f"{ZONES_CFG}: the blocks are not all square and of one size: {sizes}")
    return next(iter(sizes))[0]


def noise_p() -> float:
    ps = {b["p"] for b in key(ZONES_CFG, "blocks").values() if "p" in b}
    if len(ps) != 1:
        raise SourceError(f"{ZONES_CFG}: the noisy blocks do not share one p: {ps}")
    return ps.pop()


def n_actions() -> int:
    from helplessness.envs.dynamics import N_ACTIONS
    return N_ACTIONS


def budget(rel: str) -> int:
    return key(rel, "budget.total_steps") + key(rel, "budget.prefill_steps")


def same(values: list, what: str):
    if len(set(values)) != 1:
        raise SourceError(f"{what} differs between its sources: {values}")
    return values[0]


# --- E3-0 (gpu_e30b) -------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def e30b() -> list[dict]:
    """(seed, condition, results.json, arrays) of the 8 gpu_e30b runs."""
    out = []
    for shape in ("win64", "win16"):
        for run in src.e30b_runs(shape):
            res = src.read_json(run / "results.json")
            out.append({"shape": shape, "seed": int(res["seed"]), "condition": res["condition"], "results": res,
                        "arrays": np.load(run / "arrays.npz"), "path": run})
    return out


def true_zone_mean(run: dict, zone: str) -> float:
    per_state = controllability_per_state(run["arrays"]["P_true"])
    return float(per_state[np.array(run["results"]["zone_of_state"]) == zone].mean())


@lru_cache(maxsize=None)
def e30b_pairs() -> list[dict]:
    aggregate_b = src.e3_module("aggregate_b.py")
    pairs = []
    for shape in ("win64", "win16"):
        agg = aggregate_b.aggregate([aggregate_b.load_run(r) for r in src.e30b_runs(shape)])
        if agg["unpaired"] or len(agg["pairs"]) != 2:
            raise SourceError(f"gpu_e30b {shape}: expected 2 complete pairs")
        pairs += agg["pairs"]
    return pairs


def e30b_b(zones: tuple[str, ...]) -> list[float]:
    return [p["I_s"][z]["b"] for p in e30b_pairs() for z in zones]


def true_I(zone: str) -> float:
    return same([round(true_zone_mean(r, zone), 12) for r in e30b() if r["condition"] == "experimental"], f"true I of {zone}")


def true_b(zone: str) -> float:
    by = {}
    for r in e30b():
        by.setdefault(r["condition"], []).append(round(true_zone_mean(r, zone), 12))
    return same(by["experimental"], "true I (env)") - same(by["reference"], "true I (ref)")


def noisy_rows() -> np.ndarray:
    """The distinct outcome laws of the true kernel's rows in the controllable-noisy zone (gpu_e30b win64 seed 0,
    environment): each row reduced to its non-zero probabilities, sorted. Exact for the sampling floor: an outcome of
    probability 0 is never drawn and adds exactly 0 to it; the reduction only spares computing those zeros."""
    run = next(r for r in e30b() if (r["shape"], r["seed"], r["condition"]) == ("win64", 0, "experimental"))
    zone = np.array(run["results"]["zone_of_state"]) == "controllable_noisy"
    rows = run["arrays"]["P_true"][zone].reshape(-1, run["arrays"]["P_true"].shape[-1])
    laws = {tuple(np.sort(row[row > 0]).round(12)) for row in rows}
    if len({len(law) for law in laws}) != 1:
        raise SourceError("the controllable-noisy rows do not all have the same number of outcomes")
    return np.array(sorted(laws))


def k_s() -> int:
    configured = same([key(CAMPAIGN_CFG, "measure.kernel_k_s"), key(CAMPAIGN_CFG, "measure.real_context.kernel_draws")], "K_s")
    derived = draws_for_l1_floor(noisy_rows(), L1_FLOOR_TARGET)
    if derived != configured:
        raise SourceError(f"K_s: config {configured}, draws_for_l1_floor gives {derived}")
    return configured


# --- diagnostics and pilot -------------------------------------------------------------------------------------------

def theme_runs() -> list[dict]:
    root = src.E3_RESULTS / "theme_sensitivity"
    files = sorted(p for p in root.glob("*.json"))
    if len(files) != 16:
        raise SourceError(f"{root}: expected 16 records (10 draws, 6 replicates), found {len(files)}")
    return [src.read_json(p) for p in files]


def theme_switch(stat) -> float:
    draws = [src.read_json(src.E3_RESULTS / "theme_sensitivity" / f"draw_{d:02d}.json") for d in range(10)]
    if any(d["status"] != "ok" for d in draws):
        raise SourceError("a colour draw did not switch: the min / max of switch steps would be censored")
    return stat([d["switch_step"] for d in draws])


def mini_diag() -> dict:
    return src.read_json(REPO_ROOT / "reports" / f"diagnostics_{AGENT_MINI_RUN}" / "diagnostics.json")


def mini_grads() -> dict:
    return src.read_json(REPO_ROOT / "reports" / f"gradients_vs_data_{AGENT_MINI_RUN}" / "result.json")


@lru_cache(maxsize=None)
def pilot_records() -> list[list[dict]]:
    return [src.read_jsonl(r / "checkpoints.jsonl") for r in src.pilot_runs()]


def pilot_final(field) -> list[float]:
    return [field(records[-1]) for records in pilot_records()]


def pilot_final_deltas() -> list[float]:
    return [v for records in pilot_records() for v in records[-1]["agent_delta"]["real_contexts"]["delta_by_zone"].values()]


def pilot_meta_tie_tol() -> float:
    return same([round(yaml.safe_load((r / "meta.yaml").read_text(encoding="utf-8"))["agent"]["tie_tol_abs"], 15)
                 for r in src.pilot_runs()], "tie_tol_abs")


# --- final campaign (pending) ----------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def campaign_b() -> dict:
    """{zone: mean_ci95 of b on I_s over the campaign's pairs at the budget} (main analysis)."""
    if not src.CAMPAIGN_ROOT.is_dir():
        raise PendingSource(f"{src.CAMPAIGN_ROOT} does not exist")
    analysis, aggregate_b = src.e3_module("campaign_analysis.py"), src.e3_module("aggregate_b.py")
    result = analysis.analyse(src.campaign_runs(), None, aggregate_b.load_e30_control())
    if result["main"]["final_env_steps"] != [analysis.BUDGET_ENV_STEPS]:
        raise SourceError(f"campaign runs end at {result['main']['final_env_steps']}, not at the budget")
    pairs = result["main"]["aggregate_b"]["pairs"]
    return {z: aggregate_b.mean_ci95([p["I_s"][z]["b"] for p in pairs]) for z in src.ZONES}


# --- declarations ----------------------------------------------------------------------------------------------------

def _c(macro: str, rel: str, dotted: str, decimals: int = 0, **kw) -> Number:
    return Number(macro, f"{rel} :: {dotted}", lambda: key(rel, dotted), decimals, **kw)


E30B_RUNS = f"{E3_RES}/*gpu_e30b_win{{64,16}}__seed{{00,05}}*/"
PILOT_RUNS = f"{E3_RES}/*gpu_pilot_closed_loop__seed{{00,05}}*/"
BRIDGE = f"{E3_RES}/*gpu_e30_bridge*/recomputed/{src.BRIDGE_RECOMPUTATION}/results.json"

NUMBERS = [
    # environment and episode
    Number("eThreeGridSize", f"{ZONES_CFG} :: grid_shape (square)",
           lambda: same(list(key(ZONES_CFG, "grid_shape")), "grid_shape"), 0),
    Number("eThreeZoneSize", f"{ZONES_CFG} :: blocks.*.rows / cols (side of every block)", zone_size, 0),
    Number("eThreeNZones", f"{ZONES_CFG} :: number of blocks", lambda: len(key(ZONES_CFG, "blocks")), 0),
    Number("eThreeNActions", "src/helplessness/envs/dynamics/__init__.py :: N_ACTIONS", n_actions, 0),
    Number("eThreeNoiseP", f"{ZONES_CFG} :: blocks.ne.p = blocks.se.p", noise_p, 3),
    Number("eThreeNoiseMain", f"{ZONES_CFG} :: 1 - p + p / N_ACTIONS (blocks.ne.p; envs/dynamics)",
           lambda: 1 - noise_p() + noise_p() / n_actions(), 1),
    Number("eThreeNoiseMinor", f"{ZONES_CFG} :: p / N_ACTIONS (blocks.ne.p; envs/dynamics)", lambda: noise_p() / n_actions(), 3),
    _c("eThreeGoalReward", EPISODE_CFG, "episode.goal_reward", signed=True),
    _c("eThreeStepReward", EPISODE_CFG, "episode.step_reward", signed=True),
    _c("eThreeEpisodeHorizon", EPISODE_CFG, "episode.max_steps"),
    _c("eThreeGamma", EPISODE_CFG, "episode.gamma", 1),
    Number("eThreeVNoGoal", f"{EPISODE_CFG} :: episode.step_reward / (1 - episode.gamma)",
           lambda: key(EPISODE_CFG, "episode.step_reward") / (1 - key(EPISODE_CFG, "episode.gamma")), 0, signed=True),
    _c("eThreeRenderScale", CAMPAIGN_CFG, "render.k"),
    _c("eThreeMarkerMinDist", CAMPAIGN_CFG, "render.cielab_marker_min_dist"),
    # world model and training
    _c("eThreeLatentVars", CAMPAIGN_CFG, "model.n_vars"),
    _c("eThreeLatentClasses", CAMPAIGN_CFG, "model.n_classes"),
    _c("eThreeFreeBits", CAMPAIGN_CFG, "train.free_bits"),
    _c("eThreeLr", CAMPAIGN_CFG, "train.lr", 0, style="sci"),
    _c("eThreeLrAfterSwitch", CAMPAIGN_CFG, "train.lr_after_switch", 0, style="sci"),
    _c("eThreeSwitchPairs", CAMPAIGN_CFG, "train.latent_estimator.switch_n_pairs"),
    _c("eThreeSwitchEvalEvery", CAMPAIGN_CFG, "train.latent_estimator.switch_eval_every"),
    _c("eThreeSwitchJS", CAMPAIGN_CFG, "train.latent_estimator.switch_js_threshold"),
    _c("eThreeSwitchConsecutive", CAMPAIGN_CFG, "train.latent_estimator.switch_consecutive"),
    _c("eThreeCollapseJS", CAMPAIGN_CFG, "train.latent_estimator.collapse_js_threshold"),
    _c("eThreeBatchWindows", CAMPAIGN_CFG, "train.batch_size"),
    _c("eThreeWindowLength", CAMPAIGN_CFG, "train.seq_len"),
    Number("eThreeFramesPerUpdate", f"{E30B64_CFG} :: train.batch_size x train.seq_len",
           lambda: key(E30B64_CFG, "train.batch_size") * key(E30B64_CFG, "train.seq_len"), 0),
    _c("eThreeShortBatchWindows", E30B16_CFG, "train.batch_size"),
    _c("eThreeShortWindowLength", E30B16_CFG, "train.seq_len"),
    _c("eThreeReplayRatio", CAMPAIGN_CFG, "train.replay_ratio", 1),
    _c("eThreeReplayRatioMini", MINI_CFG, "train.replay_ratio"),
    # agent
    _c("eThreeEpsilon", CAMPAIGN_CFG, "agent.epsilon", 1),
    _c("eThreeEpsilonOffline", E30_CFG, "collection.epsilon"),
    _c("eThreeKAct", CAMPAIGN_CFG, "agent.k_act"),
    _c("eThreeImagHorizon", CAMPAIGN_CFG, "agent.horizon"),
    Number("eThreeTieTol", f"{PILOT_RUNS}meta.yaml :: agent.tie_tol_abs (the 4 runs agree)", pilot_meta_tie_tol, 4),
    # measures
    Number("eThreeKs", f"{CAMPAIGN_CFG} :: measure.kernel_k_s (= real_context.kernel_draws; checked against "
           f"measures.kernel_probe.draws_for_l1_floor(CN kernel, {L1_FLOOR_TARGET}))", k_s, 0),
    Number("eThreeFloorOldK", f"{E3_RES}/*gpu_e30b_win64__seed00* (experimental) arrays.npz P_true :: max over the CN rows "
           f"of measures.kernel_probe.expected_l1_sampling_floor(row, {OLD_K_S})",
           lambda: float(max(expected_l1_sampling_floor(row, OLD_K_S) for row in noisy_rows())), 2),
    _c("eThreeContextBurnIn", CAMPAIGN_CFG, "measure.real_context.burn_in"),
    _c("eThreeContextsPilot", PILOT_CFG, "measure.real_context.max_per_state"),
    _c("eThreeContextsFinal", CAMPAIGN_CFG, "measure.real_context.max_per_state"),
    _c("eThreeIzMcTraining", CAMPAIGN_CFG, "measure.iz_mc_samples_training"),
    _c("eThreeIzMcFinal", CAMPAIGN_CFG, "measure.iz_mc_samples_final"),
    _c("eThreeGoalDrawsPilot", PILOT_CFG, "measure.checkpoint_goal_kernel.draws"),
    _c("eThreeGoalDraws", CAMPAIGN_CFG, "measure.checkpoint_goal_kernel.goal_draws"),
    # runs and seeds
    Number("eThreeZeroSeedA", f"{E30_CFG} :: seeds[0]", lambda: key(E30_CFG, "seeds")[0], 0),
    Number("eThreeZeroSeedB", f"{E30_CFG} :: seeds[1]", lambda: key(E30_CFG, "seeds")[1], 0),
    Number("eThreeZeroEnvSteps", f"{E30_CFG} :: budget.total_steps + budget.prefill_steps", lambda: budget(E30_CFG), 0),
    Number("eThreeZeroGradSteps", f"{E30_CFG} :: (budget.total_steps + budget.prefill_steps) x train.replay_ratio",
           lambda: round(budget(E30_CFG) * key(E30_CFG, "train.replay_ratio")), 0),
    Number("eThreeBudget", f"{CAMPAIGN_CFG} :: budget.total_steps + budget.prefill_steps", lambda: budget(CAMPAIGN_CFG), 0),
    Number("eThreePilotBudget", f"{PILOT_CFG} :: budget.total_steps + budget.prefill_steps", lambda: budget(PILOT_CFG), 0),
    Number("eThreePilotSeedA", f"{PILOT_CFG} :: seeds[0]", lambda: key(PILOT_CFG, "seeds")[0], 0),
    Number("eThreePilotSeedB", f"{PILOT_CFG} :: seeds[1]", lambda: key(PILOT_CFG, "seeds")[1], 0),
    Number("eThreeCampaignSeedA", f"{CAMPAIGN_CFG} :: seeds[0]", lambda: key(CAMPAIGN_CFG, "seeds")[0], 0),
    Number("eThreeCampaignSeedB", f"{CAMPAIGN_CFG} :: seeds[1]", lambda: key(CAMPAIGN_CFG, "seeds")[1], 0),
    # results: E3-0 (gpu_e30b, 8 runs) and the bridge
    Number("eThreeTrueICN", f"{E30B_RUNS}arrays.npz P_true :: CN zone mean of controllability_per_state",
           lambda: true_I("controllable_noisy"), 2),
    Number("eThreeTrueICD", f"{E30B_RUNS}arrays.npz P_true :: CD zone mean of controllability_per_state",
           lambda: true_I("controllable_deterministic"), 2),
    Number("eThreeTrueBUD", f"{E30B_RUNS}arrays.npz P_true :: UD zone mean of controllability_per_state, env - ref",
           lambda: true_b("uncontrollable_deterministic"), 2, signed=True),
    Number("eThreeZeroIsCNMin", f"{E30B_RUNS}results.json :: min of I_s_by_category_real.controllable_noisy (8 runs)",
           lambda: min(r["results"]["I_s_by_category_real"]["controllable_noisy"] for r in e30b()), 2),
    Number("eThreeZeroIsCNMax", f"{E30B_RUNS}results.json :: max of I_s_by_category_real.controllable_noisy (8 runs)",
           lambda: max(r["results"]["I_s_by_category_real"]["controllable_noisy"] for r in e30b()), 2),
    Number("eThreeZeroProbeHMin", f"{E30B_RUNS}results.json :: min of h_only_probe.test_accuracy (8 runs)",
           lambda: min(r["results"]["h_only_probe"]["test_accuracy"] for r in e30b()), 3),
    Number("eThreeZeroProbeHMax", f"{E30B_RUNS}results.json :: max of h_only_probe.test_accuracy (8 runs)",
           lambda: max(r["results"]["h_only_probe"]["test_accuracy"] for r in e30b()), 3),
    Number("eThreeZeroProbeZMin", f"{E30B_RUNS}results.json :: min of z_only_probe.test_accuracy (8 runs)",
           lambda: min(r["results"]["z_only_probe"]["test_accuracy"] for r in e30b()), 3),
    Number("eThreeZeroProbeZMax", f"{E30B_RUNS}results.json :: max of z_only_probe.test_accuracy (8 runs)",
           lambda: max(r["results"]["z_only_probe"]["test_accuracy"] for r in e30b()), 3),
    Number("eThreeZeroFloorUDMin", f"{E30B_RUNS} :: min over the 4 pairs of aggregate_b b(I_s), UD",
           lambda: min(e30b_b(("uncontrollable_deterministic",))), 2, signed=True),
    Number("eThreeZeroFloorUDMax", f"{E30B_RUNS} :: max over the 4 pairs of aggregate_b b(I_s), UD",
           lambda: max(e30b_b(("uncontrollable_deterministic",))), 2, signed=True),
    Number("eThreeZeroFloorCtrlMin", f"{E30B_RUNS} :: min over the 4 pairs of aggregate_b b(I_s), CN and CD",
           lambda: min(e30b_b(("controllable_noisy", "controllable_deterministic"))), 3, signed=True),
    Number("eThreeZeroFloorCtrlMax", f"{E30B_RUNS} :: max over the 4 pairs of aggregate_b b(I_s), CN and CD",
           lambda: max(e30b_b(("controllable_noisy", "controllable_deterministic"))), 3, signed=True),
    Number("eThreeTFourCNMin", f"{E30B_RUNS}results.json :: min of real_context.frequent_outcome.controllable_noisy (8 runs)",
           lambda: min(r["results"]["real_context"]["frequent_outcome"]["controllable_noisy"] for r in e30b()), 2),
    Number("eThreeTFourCNMax", f"{E30B_RUNS}results.json :: max of real_context.frequent_outcome.controllable_noisy (8 runs)",
           lambda: max(r["results"]["real_context"]["frequent_outcome"]["controllable_noisy"] for r in e30b()), 2),
    Number("eThreeTFourUNMin", f"{E30B_RUNS}results.json :: min of real_context.frequent_outcome.uncontrollable_noisy (4 env runs)",
           lambda: min(r["results"]["real_context"]["frequent_outcome"]["uncontrollable_noisy"] for r in e30b()
                       if r["condition"] == "experimental"), 2),
    Number("eThreeTFourUNMax", f"{E30B_RUNS}results.json :: max of real_context.frequent_outcome.uncontrollable_noisy (4 env runs)",
           lambda: max(r["results"]["real_context"]["frequent_outcome"]["uncontrollable_noisy"] for r in e30b()
                       if r["condition"] == "experimental"), 2),
    Number("eThreeBridgeDTwoBoundary", f"{BRIDGE} :: d2_boundary_mean",
           lambda: src.read_json(src.bridge_recomputation() / "results.json")["d2_boundary_mean"], 1),
    Number("eThreeBridgeDTwoWithin", f"{BRIDGE} :: d2_within_zone_mean",
           lambda: src.read_json(src.bridge_recomputation() / "results.json")["d2_within_zone_mean"], 1),
    # results: theme sensitivity, agent_mini diagnostics
    Number("eThreeThemeSwitchMin", f"{E3_RES}/theme_sensitivity/draw_NN.json :: min of switch_step (10 draws)",
           lambda: theme_switch(min), 0),
    Number("eThreeThemeSwitchMax", f"{E3_RES}/theme_sensitivity/draw_NN.json :: max of switch_step (10 draws)",
           lambda: theme_switch(max), 0),
    Number("eThreeThemeRuns", f"{E3_RES}/theme_sensitivity/*.json :: number of records", lambda: len(theme_runs()), 0),
    Number("eThreeThemeCensored", f"{E3_RES}/theme_sensitivity/*.json :: records with status separation_not_reached",
           lambda: sum(r["status"] == "separation_not_reached" for r in theme_runs()), 0),
    Number("eThreeMiniGoalFrames", f"reports/diagnostics_{AGENT_MINI_RUN}/diagnostics.json :: a.goal_frames_total_at_resume_point",
           lambda: mini_diag()["a"]["goal_frames_total_at_resume_point"], 0),
    Number("eThreeMiniNoGoalAtSwitch", f"reports/diagnostics_{AGENT_MINI_RUN}/diagnostics.json :: a.rows[first].P_batch_without_positive (the switch)",
           lambda: mini_diag()["a"]["rows"][0]["P_batch_without_positive"], 0, style="percent"),
    Number("eThreeMiniNoGoalAtResume", f"reports/diagnostics_{AGENT_MINI_RUN}/diagnostics.json :: a.rows[last].P_batch_without_positive (18,000)",
           lambda: mini_diag()["a"]["rows"][-1]["P_batch_without_positive"], 0, style="percent"),
    Number("eThreeMiniMassStart", f"reports/gradients_vs_data_{AGENT_MINI_RUN}/result.json :: criterion.start",
           lambda: mini_grads()["criterion"]["start"], 2),
    Number("eThreeMiniMassEnd", f"reports/gradients_vs_data_{AGENT_MINI_RUN}/result.json :: criterion.end",
           lambda: mini_grads()["criterion"]["end"], 2),
    Number("eThreeMiniExtraGradSteps", f"reports/gradients_vs_data_{AGENT_MINI_RUN}/result.json :: steps_done",
           lambda: mini_grads()["steps_done"], 0),
    # results: closed-loop pilot (pilot data, 2 seeds)
    Number("eThreePilotGoalMassFinal", f"{PILOT_RUNS}checkpoints.jsonl :: min over the 4 runs of the last record's "
           "goal_diagnostics.median_mass_on_goal_pairs",
           lambda: min(pilot_final(lambda r: r["goal_diagnostics"]["median_mass_on_goal_pairs"])), 2),
    Number("eThreePilotSuccessFinal", f"{PILOT_RUNS}checkpoints.jsonl :: min over the 4 runs of the last record's "
           "success_rate (last 100 episodes)", lambda: min(pilot_final(lambda r: r["success_rate"])), 2),
    Number("eThreePilotDeltaMin", f"{PILOT_RUNS}checkpoints.jsonl :: min over runs and zones of the last record's "
           "agent_delta.real_contexts.delta_by_zone", lambda: min(pilot_final_deltas()), 2),
    Number("eThreePilotDeltaMax", f"{PILOT_RUNS}checkpoints.jsonl :: max over runs and zones of the last record's "
           "agent_delta.real_contexts.delta_by_zone", lambda: max(pilot_final_deltas()), 2),
]

# The final campaign's main result, by zone: b on I_s at the budget, mean over the seeds and 95 % Student-t interval.
for _zone in src.ZONES:
    _abbr = src.ZONE_ABBR[_zone]
    for _part, _get in (("Mean", lambda z=_zone: campaign_b()[z]["mean"]),
                        ("Low", lambda z=_zone: campaign_b()[z]["ci95"][0]),
                        ("High", lambda z=_zone: campaign_b()[z]["ci95"][1])):
        NUMBERS.append(Number(f"eThreeCampaignB{_abbr}{_part}",
                              f"{E3_RES}/gpu_campaign/ (campaign_analysis.analyse main pairs; aggregate_b.mean_ci95) :: "
                              f"b(I_s) at the budget, {_abbr}, {_part.lower()}",
                              _get, 3, signed=True, pending=True))
