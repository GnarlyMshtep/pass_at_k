"""Display helpers for DVC backup."""

from __future__ import annotations

from vfh.dvc_backup.backup_log import log
from vfh.dvc_backup.types import BackupTarget, RunSummary


def human_size(size_bytes: int | float) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def print_summary(summaries: list[RunSummary]) -> None:
    """Print a table of what will be backed up."""
    total_size = 0
    total_targets = 0
    total_already_added = 0

    log("")
    log("=== DVC Backup Summary ===")
    log("")
    for s in summaries:
        n_ckpts = sum(1 for t in s.targets if t.kind == "checkpoint")
        n_rolls = sum(1 for t in s.targets if t.kind == "rollout")
        n_added = sum(1 for t in s.targets if t.already_added)
        run_size = sum(t.size_bytes for t in s.targets)
        total_size += run_size
        total_targets += len(s.targets)
        total_already_added += n_added

        parts: list[str] = []
        if n_ckpts:
            parts.append(f"{n_ckpts} checkpoint(s)")
        if n_rolls:
            parts.append(f"{n_rolls} rollout dir(s)")
        desc = ", ".join(parts)

        recovery_note = f"  [{n_added} previously added]" if n_added else ""
        log(f"  {s.run_id}  {desc:30s}  {human_size(run_size):>10s}  {s.run_dir.name}{recovery_note}")

    log(f"")
    log(f"  Total: {total_targets} target(s), {human_size(total_size)}")
    if total_already_added:
        log(f"  Recovery: {total_already_added} target(s) have existing .dvc files (will check remote status)")
