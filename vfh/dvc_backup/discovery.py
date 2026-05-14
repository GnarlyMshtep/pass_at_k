"""Discovery of runs and backup targets.

Framework-specific rules (which subdirs of a run dir are worth backing up)
live in `DiscoveryStrategy` subclasses. Two are provided:

- `VFHDiscovery` — pass_at_k / verl runs: per-step `checkpoints/global_step_N/`
  + whole `rollouts/` dir (post-hoc-val rides along inside rollouts/).
- `TFHDiscovery` — tinker-cookbook runs: whole `rollouts/` dir + top-level
  `post-hoc-evals/` dir (Tinker server owns model checkpoints).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
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


def _has_any_file(path: Path) -> bool:
    for _ in path.rglob("*"):
        if _.is_file():
            return True
    return False


def newest_mtime_in_tree(path: Path) -> float:
    """Max mtime across all files in `path`. Returns 0.0 if no files."""
    newest = 0.0
    for f in path.rglob("*"):
        if f.is_file():
            m = f.stat().st_mtime
            if m > newest:
                newest = m
    return newest


def _is_recent(path: Path, skip_recent_seconds: float) -> bool:
    """True if any file under `path` was touched within the last N seconds."""
    if skip_recent_seconds <= 0:
        return False
    newest = newest_mtime_in_tree(path=path)
    return newest > 0 and (time.time() - newest) < skip_recent_seconds


# ---------------------------------------------------------------------------
# Discovery strategies
# ---------------------------------------------------------------------------


class DiscoveryStrategy(ABC):
    """Pluggable per-run-dir discovery of backup targets.

    Subclasses encode framework-specific knowledge about which subdirs
    (checkpoints, rollouts, etc.) are worth backing up.
    """

    name: str  # short tag used in logs ("vfh" / "tfh")

    @abstractmethod
    def discover_targets(
        self,
        run_dir: Path,
        run_id: str,
        *,
        skip_recent_seconds: float = 0,
    ) -> list[BackupTarget]:
        """Return the list of backup targets for a single run dir.

        `skip_recent_seconds`: if >0, skip any target whose newest file
        was touched within the last N seconds (race-safe vs. live writers).
        """


def _rollouts_target(
    run_dir: Path, run_id: str, skip_recent_seconds: float = 0,
) -> BackupTarget | None:
    """Shared helper: whole `rollouts/` dir tracked as `{run_dir}/rollouts.dvc`."""
    rollouts_dir = run_dir / "rollouts"
    if not rollouts_dir.is_dir():
        return None
    dvc_file = run_dir / "rollouts.dvc"
    if not any(rollouts_dir.rglob("*.jsonl")):
        vprint(f"Skipping empty rollouts dir at {run_dir}")
        return None
    if _is_recent(path=rollouts_dir, skip_recent_seconds=skip_recent_seconds):
        vprint(f"Skipping recently-written rollouts at {run_dir} "
               f"(< {skip_recent_seconds/60:.0f} min since last write)")
        return None
    already_added = dvc_file.exists()
    if not already_added:
        for legacy in ("train.dvc", "val.dvc"):
            if (rollouts_dir / legacy).exists():
                vprint(f"Found legacy {legacy} — will re-add as rollouts.dvc")
                break
    if already_added:
        vprint(f"Found existing .dvc: {dvc_file.name} (will check remote status)")
    return BackupTarget(
        path=rollouts_dir,
        dvc_file=dvc_file,
        run_id=run_id,
        kind="rollout",
        size_bytes=dir_size_bytes(path=rollouts_dir),
        already_added=already_added,
    )


class VFHDiscovery(DiscoveryStrategy):
    """Verl runs: per-step checkpoint dirs + whole rollouts dir."""

    name = "vfh"

    def discover_targets(
        self,
        run_dir: Path,
        run_id: str,
        *,
        skip_recent_seconds: float = 0,
    ) -> list[BackupTarget]:
        targets: list[BackupTarget] = []

        # --- Checkpoints: per-step global_step_N dirs ---
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.is_dir():
            for step_dir in sorted(checkpoints_dir.iterdir()):
                if not step_dir.is_dir():
                    continue
                if not step_dir.name.startswith("global_step_"):
                    continue
                dvc_file = checkpoints_dir / f"{step_dir.name}.dvc"
                if (step_dir / ".cleaned_by_daemon").exists():
                    vprint(f"Skipping cleaned: {step_dir.name}")
                    continue
                contents = [f for f in step_dir.iterdir() if f.name != ".cleaned_by_daemon"]
                if not contents:
                    vprint(f"Skipping empty: {step_dir.name}")
                    continue
                if _is_recent(path=step_dir, skip_recent_seconds=skip_recent_seconds):
                    vprint(f"Skipping recently-written checkpoint: "
                           f"{run_id}/{step_dir.name} "
                           f"(< {skip_recent_seconds/60:.0f} min since last write)")
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

        # --- Rollouts: entire dir as one target (post-hoc-val rides along) ---
        rollouts = _rollouts_target(
            run_dir=run_dir, run_id=run_id,
            skip_recent_seconds=skip_recent_seconds,
        )
        if rollouts is not None:
            targets.append(rollouts)

        return targets


class TFHDiscovery(DiscoveryStrategy):
    """Tinker-cookbook runs: rollouts + top-level post-hoc-evals. No checkpoints
    (Tinker server owns those)."""

    name = "tfh"

    def discover_targets(
        self,
        run_dir: Path,
        run_id: str,
        *,
        skip_recent_seconds: float = 0,
    ) -> list[BackupTarget]:
        targets: list[BackupTarget] = []

        rollouts = _rollouts_target(
            run_dir=run_dir, run_id=run_id,
            skip_recent_seconds=skip_recent_seconds,
        )
        if rollouts is not None:
            targets.append(rollouts)

        # --- Post-hoc evals: top-level dir tracked as {run_dir}/post-hoc-evals.dvc ---
        phe_dir = run_dir / "post-hoc-evals"
        if phe_dir.is_dir():
            if not _has_any_file(phe_dir):
                vprint(f"Skipping empty post-hoc-evals dir at {run_dir}")
            elif _is_recent(path=phe_dir, skip_recent_seconds=skip_recent_seconds):
                vprint(f"Skipping recently-written post-hoc-evals at {run_dir} "
                       f"(< {skip_recent_seconds/60:.0f} min since last write)")
            else:
                dvc_file = run_dir / "post-hoc-evals.dvc"
                already_added = dvc_file.exists()
                if already_added:
                    vprint(f"Found existing .dvc: {dvc_file.name}")
                targets.append(BackupTarget(
                    path=phe_dir,
                    dvc_file=dvc_file,
                    run_id=run_id,
                    kind="post_hoc_eval",
                    size_bytes=dir_size_bytes(path=phe_dir),
                    already_added=already_added,
                ))

        return targets


# Back-compat free function used by existing imports. Defaults to VFH behavior.
def discover_targets(run_dir: Path, run_id: str) -> list[BackupTarget]:
    return VFHDiscovery().discover_targets(run_dir=run_dir, run_id=run_id)


# ---------------------------------------------------------------------------
# Top-level entrypoints (run dir iteration)
# ---------------------------------------------------------------------------


def discover_runs(logs_root: Path) -> list[Path]:
    """Find all run directories (dirs containing run_metadata.json5)."""
    run_dirs: list[Path] = []
    for metadata_file in sorted(logs_root.rglob("run_metadata.json5")):
        run_dirs.append(metadata_file.parent)
    return run_dirs


def discover_all(
    logs_root: Path,
    strategy: DiscoveryStrategy | None = None,
    skip_recent_seconds: float = 0,
) -> list[RunSummary]:
    """Discover all runs and their backup targets."""
    if strategy is None:
        strategy = VFHDiscovery()
    summaries: list[RunSummary] = []
    for run_dir in discover_runs(logs_root=logs_root):
        parts = run_dir.name.split("_")
        run_id = parts[-1] if len(parts) >= 2 else run_dir.name
        targets = strategy.discover_targets(
            run_dir=run_dir, run_id=run_id,
            skip_recent_seconds=skip_recent_seconds,
        )
        if targets:
            summaries.append(RunSummary(
                run_id=run_id,
                run_dir=run_dir,
                targets=targets,
            ))
    return summaries


def discover_single(
    run_dir: Path,
    strategy: DiscoveryStrategy | None = None,
    skip_recent_seconds: float = 0,
) -> list[RunSummary]:
    """Discover targets for a single run directory."""
    if strategy is None:
        strategy = VFHDiscovery()
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run_metadata.json5").exists():
        raise ValueError(f"No run_metadata.json5 in {run_dir}")
    with open(run_dir / "run_metadata.json5") as f:
        meta = json.load(f)
    run_id = meta.get("run_id", run_dir.name.split("_")[-1])
    targets = strategy.discover_targets(
        run_dir=run_dir, run_id=run_id,
        skip_recent_seconds=skip_recent_seconds,
    )
    if not targets:
        return []
    return [RunSummary(run_id=run_id, run_dir=run_dir, targets=targets)]
