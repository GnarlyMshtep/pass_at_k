"""Data types for DVC backup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


TargetKind = Literal["checkpoint", "rollout", "post_hoc_eval"]


@dataclass
class BackupTarget:
    """One item to be backed up: a checkpoint dir, rollout dir, or post-hoc-eval dir."""

    path: Path               # e.g. .../checkpoints/global_step_40
    dvc_file: Path           # e.g. .../checkpoints/global_step_40.dvc
    run_id: str
    kind: TargetKind
    size_bytes: int = 0
    already_added: bool = False   # .dvc exists (from a prior interrupted run)
    already_synced: bool = False  # .dvc exists AND is synced with remote

    @property
    def label(self) -> str:
        return f"{self.run_id}/{self.path.name}"


@dataclass
class RunSummary:
    """Summary of what needs to be backed up for one run."""

    run_id: str
    run_dir: Path
    targets: list[BackupTarget] = field(default_factory=list)
