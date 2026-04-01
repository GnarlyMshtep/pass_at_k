"""Discovery of runs and backup targets."""

from __future__ import annotations

import json
from pathlib import Path

from vfh.dvc_backup.backup_log import vprint
from vfh.dvc_backup.types import BackupTarget, RunSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dir_size_bytes(path: Path) -> int:
    """Recursively compute total size of a directory."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_runs(logs_root: Path) -> list[Path]:
    """Find all run directories (dirs containing run_metadata.json5)."""
    run_dirs: list[Path] = []
    for metadata_file in sorted(logs_root.rglob("run_metadata.json5")):
        run_dirs.append(metadata_file.parent)
    return run_dirs


def discover_targets(run_dir: Path, run_id: str) -> list[BackupTarget]:
    """Find checkpoint dirs and rollout dirs that need backup.

    Also discovers targets that were previously dvc-added (have .dvc file)
    but may not have been pushed yet (for recovery after interruption).
    """
    targets: list[BackupTarget] = []

    # --- Checkpoints: global_step_N dirs ---
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        for step_dir in sorted(checkpoints_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            if not step_dir.name.startswith("global_step_"):
                continue
            dvc_file = checkpoints_dir / f"{step_dir.name}.dvc"
            # Skip if cleaned by daemon (only has .cleaned_by_daemon marker)
            if (step_dir / ".cleaned_by_daemon").exists():
                vprint(f"Skipping cleaned: {step_dir.name}")
                continue
            # Skip if empty
            contents = [f for f in step_dir.iterdir() if f.name != ".cleaned_by_daemon"]
            if not contents:
                vprint(f"Skipping empty: {step_dir.name}")
                continue

            already_added = dvc_file.exists()
            if already_added:
                vprint(f"Found existing .dvc: {dvc_file.name} (will check remote status)")

            targets.append(BackupTarget(
                path=step_dir,
                dvc_file=dvc_file,
                run_id=run_id,
                kind="checkpoint",
                size_bytes=dir_size_bytes(path=step_dir),
                already_added=already_added,
            ))

    # --- Rollouts: entire rollouts/ dir as one target ---
    rollouts_dir = run_dir / "rollouts"
    if rollouts_dir.is_dir():
        dvc_file = run_dir / "rollouts.dvc"
        # Skip if no actual data files inside
        has_data = any(rollouts_dir.rglob("*.jsonl"))
        if not has_data:
            vprint(f"Skipping empty rollouts dir")
        else:
            already_added = dvc_file.exists()
            # Also check for legacy per-subdir .dvc files
            if not already_added:
                for legacy in ("train.dvc", "val.dvc"):
                    if (rollouts_dir / legacy).exists():
                        vprint(f"Found legacy {legacy} — will re-add as rollouts.dvc")
                        break

            if already_added:
                vprint(f"Found existing .dvc: {dvc_file.name} (will check remote status)")

            targets.append(BackupTarget(
                path=rollouts_dir,
                dvc_file=dvc_file,
                run_id=run_id,
                kind="rollout",
                size_bytes=dir_size_bytes(path=rollouts_dir),
                already_added=already_added,
            ))

    return targets


def discover_all(logs_root: Path) -> list[RunSummary]:
    """Discover all runs and their backup targets."""
    summaries: list[RunSummary] = []
    for run_dir in discover_runs(logs_root=logs_root):
        # Extract run_id from dir name: {desc}_{HH}_{mm}_{run_id}
        parts = run_dir.name.split("_")
        run_id = parts[-1] if len(parts) >= 3 else run_dir.name

        targets = discover_targets(run_dir=run_dir, run_id=run_id)
        if targets:
            summaries.append(RunSummary(
                run_id=run_id,
                run_dir=run_dir,
                targets=targets,
            ))
    return summaries


def discover_single(run_dir: Path) -> list[RunSummary]:
    """Discover targets for a single run directory."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run_metadata.json5").exists():
        raise ValueError(f"No run_metadata.json5 in {run_dir}")

    # Extract run_id from metadata
    with open(run_dir / "run_metadata.json5") as f:
        meta = json.load(f)
    run_id = meta.get("run_id", run_dir.name.split("_")[-1])

    targets = discover_targets(run_dir=run_dir, run_id=run_id)
    if not targets:
        return []
    return [RunSummary(run_id=run_id, run_dir=run_dir, targets=targets)]
