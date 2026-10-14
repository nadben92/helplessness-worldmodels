"""Optional Weights & Biases tracking of an E3 run (docs/e3/DECISIONS.md,
2026-09-28, W&B entry). Never part of the measurement and never a reason for
a run to stop: every call is guarded, and any failure (wandb not installed,
no key, no network, an API error) prints one warning in the console and
turns the logger into a no-op for the rest of the run.

When it is on (config `logging.wandb`, CLI --wandb):
  enabled: true | false | auto -- auto (the default) means on if the
      WANDB_API_KEY environment variable is set, off otherwise; --wandb
      forces it on (e.g. offline, without a key);
  project, entity: where the runs go;
  mode: online | offline | disabled -- the WANDB_MODE environment variable,
      if set, takes precedence (offline runs are uploaded later with
      `wandb sync <run dir>/wandb/offline-run-*`).
The W&B run is named after the run directory; its config is the resolved
config plus meta.yaml. It logs only at points where run_mini.py already
synchronises (no extra device synchronisation): the console line every
`measure.monitor_every` steps (losses, gradient norm, lr, phase), the
switch, every diagnostic point (the fields of diagnostics.jsonl), and at the
end the scalars of results.json, the figures and REPORT.md. The x axis is
`grad_step` (W&B's own step is not used), so a resumed run can log again
from its resume point. `--wandb-artifact` uploads the run directory as an
artifact at the end (weights included, checkpoints/resume.pt and the local
wandb/ directory excluded).

Moved from experiments/E3_worldmodel_4zones/wandb_logger.py on 2026-10-14 (a
re-export shim stays there); body unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

LOG_PREFIX = "[wandb]"


def _flatten(obj, prefix: str = "") -> dict:
    """Nested dicts -> {"a/b/c": scalar}; lists and non-numeric values are left out."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, bool):
        out[prefix] = int(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = obj
    elif obj is None:
        pass
    else:
        try:  # numpy scalars
            out[prefix] = float(obj)
        except (TypeError, ValueError):
            pass
    return out


def wandb_settings(cfg: dict) -> dict:
    block = ((cfg.get("logging") or {}).get("wandb") or {})
    return {"enabled": block.get("enabled", "auto"), "project": block.get("project", "helplessness-e3"),
            "entity": block.get("entity"), "mode": block.get("mode", "online")}


def wandb_requested(cfg: dict, force: bool = False) -> bool:
    enabled = wandb_settings(cfg)["enabled"]
    if force or enabled is True:
        return True
    if enabled == "auto":
        return bool(os.environ.get("WANDB_API_KEY"))
    return False


class WandbLogger:
    """No-op unless start() succeeds. Every method swallows its own errors."""

    def __init__(self, log=print):
        self._log = log
        self._run = None
        self._wandb = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def _warn(self, what: str, err: Exception) -> None:
        self._log(f"{LOG_PREFIX} WARNING: {what} failed ({type(err).__name__}: {err}); W&B tracking is off for the rest of this "
                  "run, the run itself continues normally")
        self._run = None

    def start(self, cfg: dict, run_dir: Path, meta: dict, force: bool = False, resume_id: str | None = None) -> str | None:
        """Starts the W&B run if requested; returns its id (for meta.yaml) or None."""
        if not wandb_requested(cfg, force):
            return None
        s = wandb_settings(cfg)
        try:
            import wandb

            self._wandb = wandb
            mode = os.environ.get("WANDB_MODE") or s["mode"]
            self._run = wandb.init(
                project=s["project"], entity=s["entity"], mode=mode, name=Path(run_dir).name, dir=str(run_dir),
                config={"config": cfg, "meta": meta}, id=resume_id, resume="allow" if resume_id else None,
            )
            self._run.define_metric("grad_step")
            self._run.define_metric("*", step_metric="grad_step")
            self._log(f"{LOG_PREFIX} tracking on (mode {mode}, project {s['project']}): {getattr(self._run, 'url', None) or 'offline'}")
            return self._run.id
        except Exception as err:  # noqa: BLE001 -- W&B must never stop a run
            self._warn("initialisation", err)
            return None

    def log(self, step: int, values: dict, prefix: str = "") -> None:
        if not self.active:
            return
        try:
            data = _flatten(values, prefix)
            data["grad_step"] = int(step)
            self._run.log(data)
        except Exception as err:  # noqa: BLE001
            self._warn("log", err)

    def finish(self, results: dict | None, run_dir: Path, artifact: bool = False) -> None:
        if not self.active:
            return
        run_dir = Path(run_dir)
        try:
            if results is not None:
                for key, value in _flatten(results).items():
                    self._run.summary[f"results/{key}"] = value
            for png in sorted((run_dir / "artifacts").glob("*.png")):
                self._run.log({f"figures/{png.stem}": self._wandb.Image(str(png))})
            if (run_dir / "REPORT.md").exists():
                self._run.save(str(run_dir / "REPORT.md"), base_path=str(run_dir), policy="now")
            if artifact:
                art = self._wandb.Artifact(name=run_dir.name, type="e3-run",
                                           description="E3 run directory (weights included, resume.pt excluded)")
                for path in sorted(run_dir.rglob("*")):
                    rel = path.relative_to(run_dir)
                    if path.is_file() and rel.parts[0] != "wandb" and rel.as_posix() != "checkpoints/resume.pt":
                        art.add_file(str(path), name=rel.as_posix())
                self._run.log_artifact(art)
                self._log(f"{LOG_PREFIX} run directory logged as artifact {run_dir.name}")
            self._run.finish()
            self._run = None
        except Exception as err:  # noqa: BLE001
            self._warn("finish", err)
