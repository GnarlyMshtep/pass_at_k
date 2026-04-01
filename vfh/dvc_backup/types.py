"""Data types for DVC backup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BackupTarget:
    """One item to be backed up: a checkpoint dir or a rollout dir."""

    path: Path               # e.g. .../checkpoints/global_step_40
    dvc_file: Path           # e.g. .../checkpoints/global_step_40.dvc
    run_id: str
    kind: str                # "checkpoint" or "rollout"
    size_bytes: int = 0
    already_added: bool = False   # .dvc exists (from a prior interrupted run)
    already_synced: bool = False  # .dvc exists AND is synced with remote


@dataclass
class RunSummary:
    """Summary of what needs to be backed up for one run."""

    run_id: str
    run_dir: Path
    targets: list[BackupTarget] = field(default_factory=list)
