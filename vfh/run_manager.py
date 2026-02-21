"""Run manager: create and track VFH run directories.

Each training run gets its own directory:
  logs/VerlRun/{MM}/{DD}/{description}_{HH}_{mm}_{run_id}/

The run_id doubles as the wandb run ID so the two are linked.
"""

from __future__ import annotations

import dataclasses
import os
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import dacite
import pyjson5
import wandb.util

from vfh.vfh_types import ForkReason, RunMetadata, RunOrigin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_run_dir(
    logs_root: str,
    description: str,
    origin: RunOrigin,
    project_name: str,
    base_config_path: str,
    overrides_path: Optional[str],
    resolved_hydra_overrides: list[str],
) -> RunMetadata:
    """Create a new run directory and write run_metadata.json5.

    Args:
        logs_root: Root directory for all runs (e.g. "logs").
        description: Human-readable label for this run.
        origin: How this run relates to previous runs.
        project_name: WandB project name (from trainer.project_name in config).
        base_config_path: Path to the base JSON5 config.
        overrides_path: Path to the override JSON5 (or None).
        resolved_hydra_overrides: Full list of Hydra overrides to pass to verl.

    Returns:
        RunMetadata written to disk.
    """
    wandb_entity = os.environ.get("WANDB_ENTITY", "")
    if not wandb_entity:
        raise EnvironmentError(
            "$WANDB_ENTITY is not set. Export it before launching VFH."
        )

    run_id: str = wandb.util.generate_id()  # 8-char alphanumeric, same as wandb internals
    wandb_url = f"https://wandb.ai/{wandb_entity}/{project_name}/runs/{run_id}"

    now = datetime.now(tz=timezone.utc)
    run_dir = _build_run_dir(
        logs_root=logs_root,
        description=description,
        now=now,
        run_id=run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = RunMetadata(
        run_id=run_id,
        run_dir=str(run_dir.resolve()),
        wandb_run_id=run_id,
        wandb_url=wandb_url,
        project_name=project_name,
        description=description,
        origin=origin,
        created_at=now.isoformat(),
        base_config_path=str(Path(base_config_path).resolve()),
        overrides_path=str(Path(overrides_path).resolve()) if overrides_path else None,
        resolved_hydra_overrides=resolved_hydra_overrides,
        child_run_ids=[],
    )
    _write_metadata(run_dir=run_dir, metadata=metadata)

    return metadata


def create_run_subdirs(run_dir: str) -> None:
    """Create sub-directories expected by verl and VFH.

    Called AFTER validation so that validate_env doesn't see
    freshly-created empty dirs and spuriously auto-remove them.
    """
    d = Path(run_dir)
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "rollouts" / "train").mkdir(parents=True, exist_ok=True)
    (d / "rollouts" / "val").mkdir(parents=True, exist_ok=True)
    (d / "daemon_logs").mkdir(exist_ok=True)
    # Read by ray_trainer.py to force checkpoint save
    (d / "checkpoints" / "should_save_asap.txt").touch()


def register_child_run(
    parent_run_dir: str,
    child_run_id: str,
) -> None:
    """Append child_run_id to the parent run's metadata."""
    parent_path = Path(parent_run_dir)
    metadata = load_run_metadata(run_dir=parent_run_dir)
    metadata.child_run_ids.append(child_run_id)
    _write_metadata(run_dir=parent_path, metadata=metadata)


def load_run_metadata(run_dir: str) -> RunMetadata:
    """Load and deserialize run_metadata.json5 from a run directory."""
    meta_path = Path(run_dir) / "run_metadata.json5"
    if not meta_path.exists():
        raise FileNotFoundError(f"No run_metadata.json5 found in: {run_dir}")
    with open(meta_path) as f:
        raw: dict[str, Any] = pyjson5.load(f)
    return dacite.from_dict(
        data_class=RunMetadata,
        data=raw,
        config=dacite.Config(cast=[ForkReason]),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_run_dir(
    logs_root: str,
    description: str,
    now: datetime,
    run_id: str,
) -> Path:
    safe_desc = description.replace(" ", "_").replace("/", "-")[:40] or "run"
    dir_name = f"{safe_desc}_{now.strftime('%H_%M')}_{run_id}"
    return Path(logs_root) / "VerlRun" / now.strftime("%m") / now.strftime("%d") / dir_name


def _serialize_for_json(obj: Any) -> Any:
    """Recursively convert Enum values to their .value for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _serialize_for_json(obj=v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(obj=v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


def _write_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    meta_path = run_dir / "run_metadata.json5"
    raw = _serialize_for_json(obj=dataclasses.asdict(metadata))
    with open(meta_path, "w") as f:
        pyjson5.dump(obj=raw, fp=f, indent=2)
