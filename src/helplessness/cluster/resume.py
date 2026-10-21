"""Resuming the runs of a GPU batch (E4 on): which runs to resume and a read-only check of their resume points, before
anything is submitted. Moved from experiments/E4_generalization_gradient/submit_kubeflow.py on 2026-10-21 (bodies
unchanged; the run directories of a batch are now an argument, so that E4 and E5 share it); each experiment's submit
script finds its batch's run directories and calls these.

  --phase resume --resume <run>   that run only (refused if it is done);
  --phase resume (no --resume)    every run of the batch whose meta.yaml status is not "done" (and that holds no
                                  checkpoints/agent_final.pt), all in one pod, each with the single run's command.

The check (resume_check): checkpoints/latest.json names resume.pt at an environment step; resume.pt is a complete zip
archive (every member's CRC-32, zip_problem); it loads entirely on the CPU (load_for_check: the notebook has no GPU;
torch.load with map_location="cpu" and the same unpickler for the pickles inside it -- the replay with its CUDA copy of
its table, the environment); its step is latest.json's and below the run's last one. Standard library at import time:
the notebook's Python runs the submit scripts without the package's dependencies installed (numpy, torch and yaml are
imported inside the functions that need them).
"""

from __future__ import annotations

import json
from pathlib import Path


RESUME_KEYS = ("progress", "rssm", "goal_head", "critic", "slow_critic", "optimizer", "goal_optimizer", "critic_optimizer",
               "switch", "replay", "env", "collector", "streams", "n_env_steps", "planned_env_steps")


def load_for_check(path: Path) -> tuple[dict, object, object]:
    """resume.pt loaded entirely on the CPU (returns the state, its replay and its environment), for the check only (the resume itself loads it with run_e4.py on the GPU):
    torch.load with map_location="cpu", and the same unpickler for the pickles inside it (the replay and the
    environment), whose torch tensors -- the replay's CUDA copy of its table -- would otherwise be restored on the device
    they were saved from. Written by torch 2.10 / numpy 2 on the cluster, read by the notebook's older versions (torch
    2.3.1, tested with numpy 1.26): when numpy < 2 is installed, the unpickler maps numpy 2's module numpy._core to numpy
    1's numpy.core, and numpy 2's random-generator pickles (which pass the bit generator's class or instance where numpy 1
    expects its name) to numpy 1's constructors."""
    import io
    import pickle
    import types

    import numpy
    import torch

    numpy_1 = int(numpy.__version__.split(".")[0]) < 2
    numpy_1_random = {}
    if numpy_1:
        from numpy.random import Generator, RandomState
        from numpy.random.bit_generator import BitGenerator

        def bit_generator_ctor(bit_generator="MT19937"):
            cls = bit_generator if isinstance(bit_generator, type) else numpy.random._pickle.BitGenerators[bit_generator]

            def setstate(self, state):  # numpy 2 pickles (state, seed sequence), numpy 1 the state alone
                cls.__setstate__(self, state[0] if isinstance(state, tuple) else state)

            return type(cls.__name__, (cls,), {"__setstate__": setstate})()

        def generator_ctor(bit_generator_name="MT19937", bit_generator_ctor=bit_generator_ctor):
            return Generator(bit_generator_name if isinstance(bit_generator_name, BitGenerator) else bit_generator_ctor(bit_generator_name))

        def randomstate_ctor(bit_generator_name="MT19937", bit_generator_ctor=bit_generator_ctor):
            return RandomState(bit_generator_name if isinstance(bit_generator_name, BitGenerator) else bit_generator_ctor(bit_generator_name))

        numpy_1_random = {"__bit_generator_ctor": bit_generator_ctor, "__generator_ctor": generator_ctor,
                          "__randomstate_ctor": randomstate_ctor}

    class CheckUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False, pickle_module=check_pickle)
            if numpy_1 and (module == "numpy._core" or module.startswith("numpy._core.")):
                module = "numpy.core" + module[len("numpy._core"):]
            if module == "numpy.random._pickle" and name in numpy_1_random:
                return numpy_1_random[name]
            return super().find_class(module, name)

    check_pickle = types.ModuleType("check_pickle")
    check_pickle.Unpickler = CheckUnpickler
    check_pickle.load = lambda f, **kw: CheckUnpickler(f, **kw).load()
    check_pickle.loads = lambda b, **kw: CheckUnpickler(io.BytesIO(b), **kw).load()
    state = torch.load(path, map_location="cpu", weights_only=False, pickle_module=check_pickle)
    missing = [k for k in RESUME_KEYS if k not in state]
    if missing:
        raise KeyError(f"keys missing: {missing}")
    return state, check_pickle.loads(state["replay"]), check_pickle.loads(state["env"])


def zip_problem(path: Path) -> str | None:
    """None if resume.pt is a complete torch zip archive whose every member matches its CRC-32 (independent of the torch
    and numpy versions); else what is wrong."""
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return f"member {bad} fails its CRC check"
            if not any(n.endswith("/data.pkl") or n == "data.pkl" for n in z.namelist()):
                return "no data.pkl in the archive"
    except Exception as err:
        return f"not a complete zip archive: {type(err).__name__}: {err}"
    return None


def resume_check(run_dir: Path) -> dict:
    """Where a resume of run_dir would start, checked read-only: done (meta.yaml status "done" or agent_final.pt: never
    relaunched), else checkpoints/latest.json must name resume.pt at a step, resume.pt must load entirely (torch.load,
    every key the resume reads, its pickled replay and environment, on the CPU: load_for_check) at that environment step,
    below the run's last one; its zip archive is first checked member by member (zip_problem). Returns {"run", "status", "done", "resume_from_env_step", "latest_json_step", "n_env_steps", "last_weights_env_step",
    "problems"}."""
    ck = run_dir / "checkpoints"
    out = {"run": str(run_dir), "status": None, "done": False, "resume_from_env_step": None, "latest_json_step": None,
           "n_env_steps": None, "last_weights_env_step": None, "problems": []}
    problems = out["problems"]
    try:
        import yaml

        out["status"] = (yaml.safe_load((run_dir / "meta.yaml").read_text(encoding="utf-8")) or {}).get("status")
    except Exception as err:
        problems.append(f"meta.yaml unreadable: {type(err).__name__}: {err}")
    weights = sorted(int(f.stem[len("agent_env"):]) for f in ck.glob("agent_env*.pt") if f.stem[len("agent_env"):].isdigit())
    out["last_weights_env_step"] = weights[-1] if weights else None
    if out["status"] == "done" or (ck / "agent_final.pt").exists():
        out["done"] = True
        return out
    try:
        latest = json.loads((ck / "latest.json").read_text(encoding="utf-8"))
        out["latest_json_step"] = int(latest["step"])
        if latest.get("filename") != "resume.pt":
            problems.append(f"latest.json names {latest.get('filename')!r}, not resume.pt")
    except Exception as err:
        problems.append(f"checkpoints/latest.json missing or unreadable: {type(err).__name__}: {err}")
    if (ck / "resume.pt.tmp").exists():
        problems.append(f"checkpoints/resume.pt.tmp exists ({(ck / 'resume.pt.tmp').stat().st_size} bytes): a resume point "
                        "was being written when the run stopped (it is not used; resume.pt is the previous one)")
    path = ck / "resume.pt"
    if not path.is_file():
        problems.append("checkpoints/resume.pt missing")
        return out
    damaged = zip_problem(path)
    if damaged:
        problems.append(f"resume.pt ({path.stat().st_size} bytes) is damaged: {damaged}")
        return out
    try:
        import torch  # noqa: F401  (load_for_check needs it)
    except ImportError as err:
        problems.append(f"cannot check resume.pt here ({err}): run the dry-run with a Python that has torch")
        return out
    try:
        state, _, _ = load_for_check(path)
        step = int(state["progress"]["env_step"])
        n_env_steps = int(state["progress"]["schedule"][0]) if "schedule" in state["progress"] else int(state["n_env_steps"])
        out["resume_from_env_step"], out["n_env_steps"] = step, n_env_steps
        del state
    except Exception as err:
        problems.append(f"resume.pt ({path.stat().st_size} bytes, its zip archive intact) does not load entirely here: "
                        f"{type(err).__name__}: {err}")
        return out
    if out["latest_json_step"] is not None and out["latest_json_step"] != step:
        problems.append(f"resume.pt is at environment step {step}, latest.json says {out['latest_json_step']}")
    if step >= n_env_steps:
        problems.append(f"resume.pt is at environment step {step}, not below the run's last ({n_env_steps}): nothing to resume")
    return out


def resume_plan(repo: Path, run_dirs_by_job: dict[str, Path | None], resume: str, batch: str = "") -> tuple[dict[str, str] | None, list[dict]]:
    """The runs to resume and their checks. resume: one run directory (that run only), or "" (every run of
    run_dirs_by_job -- job name -> its run directory, None if it has none -- that is not done). Returns ({job name: run
    directory relative to repo} or None for a single run, every check)."""
    if resume:
        d = Path(resume) if Path(resume).is_absolute() else repo / resume
        check = {"job": "resume", **resume_check(d)}
        if check["done"]:
            check["problems"].append("the run is done (meta.yaml status or agent_final.pt): never relaunched")
        return None, [check]
    checks, runs = [], {}
    for name, d in run_dirs_by_job.items():
        if d is None:
            checks.append({"job": name, "run": None, "done": False, "problems": ["no run directory for this job"]})
            continue
        check = {"job": name, **resume_check(d)}
        checks.append(check)
        if not check["done"]:
            runs[name] = d.relative_to(repo).as_posix()
    if not runs and not any(c["problems"] for c in checks):
        checks.append({"job": "batch", "problems": [f"every run of {batch or 'the batch'} is done: nothing to resume"]})
    return runs, checks


def report_checks(batch: str, checks: list[dict], log=print) -> bool:
    """Prints the checks (JSON, then one line per run: the step it resumes from, or done); False, with every problem,
    when one fails -- the caller then submits nothing."""
    log(json.dumps({"batch": batch, "phase": "resume", "resume_checks": checks}, indent=1))
    bad = [c for c in checks if c["problems"]]
    if bad:
        log("RESUME CHECK FAILED -- nothing submitted:")
        for c in bad:
            for problem in c["problems"]:
                log(f"  {c['job']}: {problem}")
        return False
    for c in checks:
        log(f"  {c['job']}: {'done, not relaunched' if c['done'] else 'resumes from environment step ' + str(c['resume_from_env_step'])}"
            f" (status {c['status']}, last weights {c['last_weights_env_step']})")
    return True


def encode_resume_runs(runs: dict[str, str] | None) -> str:
    return ";".join(f"{name}={d}" for name, d in (runs or {}).items())


def decode_resume_runs(text: str) -> dict[str, str] | None:
    return dict(item.split("=", 1) for item in text.split(";")) if text else None
