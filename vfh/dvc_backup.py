"""DVC backup script for VerlRun checkpoints and rollouts.

Crawls logs/VerlRun/ for all run directories, discovers untracked checkpoints
and rollouts, backs them up via DVC (add + push), verifies the push, and
optionally deletes local data after confirmed backup.

Recovery: if interrupted mid-run and re-launched, the script detects targets
that were already dvc-added (have .dvc file) and checks their remote status
via `dvc status --cloud`. Already-synced targets are skipped; un-pushed
targets are included in the push step.

Usage:
    python -m vfh.dvc_backup
    python -m vfh.dvc_backup --dry-run
    python -m vfh.dvc_backup --verbose
    python -m vfh.dvc_backup --setup-test-dir    # create fake dir for testing
    python -m vfh.dvc_backup --logs-root /tmp/dvc_backup_test/VerlRun --dry-run
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DvcBackupConfig:
    """Configuration for DVC backup of VerlRun checkpoints and rollouts."""
    logs_root: str = "logs/VerlRun"
    dry_run: bool = False
    verbose: bool = False
    setup_test_dir: bool = False  # create a fake VerlRun dir for testing


# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------

_DVC_ADD_TIMEOUT = 1800   # 30 min
_DVC_PUSH_TIMEOUT = 3600  # 60 min
_DVC_GC_TIMEOUT = 600     # 10 min
_DVC_STATUS_TIMEOUT = 120 # 2 min

# Module-level verbose flag, set in main()
_VERBOSE = False


def _vprint(msg: str) -> None:
    """Print only if verbose mode is on."""
    if _VERBOSE:
        print(f"  [verbose] {msg}")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test directory setup
# ---------------------------------------------------------------------------


def _setup_test_dir() -> Path:
    """Create a fake VerlRun directory for testing the backup script."""
    test_root = Path("/tmp/dvc_backup_test/VerlRun/01/01")
    base = Path("/tmp/dvc_backup_test")
    if base.exists():
        shutil.rmtree(base)

    # Run 1: has checkpoints + rollouts
    run1 = test_root / "test_run_12_00_abc12345"
    (run1 / "checkpoints" / "global_step_10").mkdir(parents=True)
    (run1 / "checkpoints" / "global_step_10" / "dummy_model.bin").write_text("model_data_step10")
    (run1 / "checkpoints" / "global_step_20").mkdir(parents=True)
    (run1 / "checkpoints" / "global_step_20" / "dummy_model.bin").write_text("model_data_step20")
    (run1 / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("20")
    (run1 / "rollouts" / "train").mkdir(parents=True)
    (run1 / "rollouts" / "train" / "1.jsonl").write_text('{"prompt": "test1"}\n')
    (run1 / "rollouts" / "train" / "2.jsonl").write_text('{"prompt": "test2"}\n')
    (run1 / "rollouts" / "val").mkdir(parents=True)
    (run1 / "rollouts" / "val" / "1.jsonl").write_text('{"prompt": "val1"}\n')
    meta1 = {
        "run_id": "abc12345",
        "run_dir": str(run1),
        "resolved_hydra_overrides": [],
    }
    (run1 / "run_metadata.json5").write_text(json.dumps(meta1, indent=2))

    # Run 2: has a cleaned checkpoint (should skip) + one good checkpoint
    run2 = test_root / "test_run2_13_00_def67890"
    (run2 / "checkpoints" / "global_step_5").mkdir(parents=True)
    (run2 / "checkpoints" / "global_step_5" / ".cleaned_by_daemon").write_text("cleaned")
    (run2 / "checkpoints" / "global_step_10").mkdir(parents=True)
    (run2 / "checkpoints" / "global_step_10" / "dummy_model.bin").write_text("model_data")
    (run2 / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("10")
    meta2 = {
        "run_id": "def67890",
        "run_dir": str(run2),
        "resolved_hydra_overrides": [],
    }
    (run2 / "run_metadata.json5").write_text(json.dumps(meta2, indent=2))

    # Run 3: empty rollouts (should skip)
    run3 = test_root / "test_empty_14_00_ghi11111"
    (run3 / "checkpoints").mkdir(parents=True)
    (run3 / "rollouts" / "train").mkdir(parents=True)
    (run3 / "rollouts" / "val").mkdir(parents=True)
    meta3 = {
        "run_id": "ghi11111",
        "run_dir": str(run3),
        "resolved_hydra_overrides": [],
    }
    (run3 / "run_metadata.json5").write_text(json.dumps(meta3, indent=2))

    root = base / "VerlRun"
    print(f"Test directory created: {root}")
    print(f"  Run 1 (abc12345): 2 checkpoints + train + val rollouts")
    print(f"  Run 2 (def67890): 1 cleaned (skip) + 1 good checkpoint, no rollouts")
    print(f"  Run 3 (ghi11111): empty checkpoints + empty rollouts (skip entirely)")
    print(f"\nTest with:")
    print(f"  python -m vfh.dvc_backup --logs-root {root} --dry-run --verbose")
    return root


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    """Recursively compute total size of a directory."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _human_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _discover_runs(logs_root: Path) -> list[Path]:
    """Find all run directories (dirs containing run_metadata.json5)."""
    run_dirs: list[Path] = []
    for metadata_file in sorted(logs_root.rglob("run_metadata.json5")):
        run_dirs.append(metadata_file.parent)
    return run_dirs


def _discover_targets(run_dir: Path, run_id: str) -> list[BackupTarget]:
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
                _vprint(f"Skipping cleaned: {step_dir.name}")
                continue
            # Skip if empty
            contents = [f for f in step_dir.iterdir() if f.name != ".cleaned_by_daemon"]
            if not contents:
                _vprint(f"Skipping empty: {step_dir.name}")
                continue

            already_added = dvc_file.exists()
            if already_added:
                _vprint(f"Found existing .dvc: {dvc_file.name} (will check remote status)")

            targets.append(BackupTarget(
                path=step_dir,
                dvc_file=dvc_file,
                run_id=run_id,
                kind="checkpoint",
                size_bytes=_dir_size_bytes(step_dir),
                already_added=already_added,
            ))

    # --- Rollouts: train/ and val/ dirs ---
    rollouts_dir = run_dir / "rollouts"
    if rollouts_dir.is_dir():
        for subdir_name in ("train", "val"):
            subdir = rollouts_dir / subdir_name
            if not subdir.is_dir():
                continue
            dvc_file = rollouts_dir / f"{subdir_name}.dvc"
            # Skip if empty
            contents = list(subdir.iterdir())
            if not contents:
                _vprint(f"Skipping empty rollout: {subdir_name}")
                continue

            already_added = dvc_file.exists()
            if already_added:
                _vprint(f"Found existing .dvc: {dvc_file.name} (will check remote status)")

            targets.append(BackupTarget(
                path=subdir,
                dvc_file=dvc_file,
                run_id=run_id,
                kind="rollout",
                size_bytes=_dir_size_bytes(subdir),
                already_added=already_added,
            ))

    return targets


def discover_all(logs_root: Path) -> list[RunSummary]:
    """Discover all runs and their backup targets."""
    summaries: list[RunSummary] = []
    for run_dir in _discover_runs(logs_root):
        # Extract run_id from dir name: {desc}_{HH}_{mm}_{run_id}
        parts = run_dir.name.split("_")
        run_id = parts[-1] if len(parts) >= 3 else run_dir.name

        targets = _discover_targets(run_dir=run_dir, run_id=run_id)
        if targets:
            summaries.append(RunSummary(
                run_id=run_id,
                run_dir=run_dir,
                targets=targets,
            ))
    return summaries


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def print_summary(summaries: list[RunSummary]) -> None:
    """Print a table of what will be backed up."""
    total_size = 0
    total_targets = 0
    total_already_added = 0

    print("\n=== DVC Backup Summary ===\n")
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
        print(f"  {s.run_id}  {desc:30s}  {_human_size(run_size):>10s}  {s.run_dir.name}{recovery_note}")

    print(f"\n  Total: {total_targets} target(s), {_human_size(total_size)}")
    if total_already_added:
        print(f"  Recovery: {total_already_added} target(s) have existing .dvc files (will check remote status)")


# ---------------------------------------------------------------------------
# Lock management
# ---------------------------------------------------------------------------


def _clear_stale_lock() -> bool:
    """Check for and clear stale DVC lock. Returns True if a lock was cleared."""
    lock_path = Path(".dvc/tmp/lock")
    if not lock_path.exists():
        return False
    # Check if the process holding the lock is still alive
    try:
        lock_pid = int(lock_path.read_text().strip())
        import os
        os.kill(lock_pid, 0)  # check if alive (doesn't actually send signal)
        # Process is alive — lock is NOT stale
        print(f"  WARNING: DVC lock held by PID {lock_pid} (still running).")
        print(f"  Another DVC process is active. Wait for it or kill it manually.")
        return False
    except (ValueError, ProcessLookupError, PermissionError):
        # Process is dead or lock file is not a valid PID — stale lock
        lock_path.unlink()
        print(f"  Cleared stale DVC lock (dead process).")
        return True


# ---------------------------------------------------------------------------
# DVC operations
# ---------------------------------------------------------------------------


def _run_dvc(args: list[str], timeout: int, label: str) -> subprocess.CompletedProcess[str]:
    """Run a DVC command with timeout. Handles stale locks."""
    cmd = ["dvc"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Retry once if lock error and lock is stale
        if result.returncode != 0 and "Unable to acquire lock" in result.stderr:
            if _clear_stale_lock():
                _vprint(f"Retrying after stale lock clear: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

        if result.returncode != 0:
            print(f"  ERROR [{label}]: exit {result.returncode}")
            if result.stdout.strip():
                print(f"    stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"    stderr: {result.stderr.strip()}")
        else:
            _vprint(f"OK: {label}")
        return result
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{label}]: {timeout}s exceeded for: {' '.join(cmd)}")
        raise


def dvc_add(target: BackupTarget) -> bool:
    """Run dvc add on a target. Returns True on success."""
    result = _run_dvc(
        args=["add", str(target.path)],
        timeout=_DVC_ADD_TIMEOUT,
        label=f"dvc add {target.path.name}",
    )
    return result.returncode == 0


def dvc_push(dvc_file: Path) -> bool:
    """Run dvc push on a .dvc file. Returns True on success."""
    result = _run_dvc(
        args=["push", str(dvc_file)],
        timeout=_DVC_PUSH_TIMEOUT,
        label=f"dvc push {dvc_file.name}",
    )
    return result.returncode == 0


def dvc_check_remote_status(dvc_file: Path) -> str:
    """Check remote status of a .dvc file.

    Returns:
        "synced"  — cache and remote are in sync
        "new"     — exists locally but not on remote (needs push)
        "deleted" — ambiguous: either pushed+cache-cleared, or never pushed+cache-cleared
        "error"   — command failed
    """
    result = _run_dvc(
        args=["status", "--cloud", str(dvc_file)],
        timeout=_DVC_STATUS_TIMEOUT,
        label=f"dvc status --cloud {dvc_file.name}",
    )
    if result.returncode != 0:
        return "error"
    combined = (result.stdout + result.stderr).lower()
    if "in sync" in combined:
        return "synced"
    if "new:" in combined:
        return "new"
    if "deleted:" in combined:
        return "deleted"
    # Unexpected output
    _vprint(f"Unexpected dvc status output: {result.stdout.strip()}")
    return "error"


def dvc_verify_on_remote(dvc_file: Path) -> bool:
    """Verify a .dvc file's data exists on remote by attempting dvc push.

    dvc push does a cheap object_exists hash check (~2-5s) and reports
    "Everything is up to date" if all blobs exist on remote. This is more
    reliable than dvc status --cloud, which reports ambiguous "deleted"
    when the local cache has been cleared.
    """
    result = _run_dvc(
        args=["push", str(dvc_file)],
        timeout=_DVC_PUSH_TIMEOUT,
        label=f"dvc push (verify) {dvc_file.name}",
    )
    if result.returncode != 0:
        return False
    combined = (result.stdout + result.stderr).lower()
    # "Everything is up to date" or "N files pushed" both mean success
    return True


def dvc_gc() -> None:
    """Run dvc gc to clean local cache."""
    print("\nCleaning local DVC cache (dvc gc --not-in-remote -w -f)...")
    _run_dvc(
        args=["gc", "--not-in-remote", "-w", "-f"],
        timeout=_DVC_GC_TIMEOUT,
        label="dvc gc",
    )


def git_add_and_commit(logs_root: Path) -> bool:
    """Git add all .dvc and .gitignore files under logs_root, then commit."""
    dvc_files = list(logs_root.rglob("*.dvc"))
    gitignore_files = list(logs_root.rglob(".gitignore"))
    files_to_stage = [str(f) for f in dvc_files + gitignore_files]

    if not files_to_stage:
        print("  No .dvc or .gitignore files to commit.")
        return True

    result = subprocess.run(
        ["git", "add"] + files_to_stage,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR [git add]: {result.stderr.strip()}")
        return False

    result = subprocess.run(
        ["git", "commit", "-m", "dvc: backup checkpoints and rollouts"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "nothing to commit" in result.stdout:
            print("  Nothing new to commit.")
            return True
        print(f"  ERROR [git commit]: {result.stderr.strip()}")
        return False

    print(f"  Committed DVC tracking files.")
    return True


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def delete_targets(targets: list[BackupTarget]) -> None:
    """Delete local data for verified targets."""
    for t in targets:
        if t.path.is_dir():
            shutil.rmtree(t.path)
            print(f"  Deleted: {t.path}")
        elif t.path.is_file():
            t.path.unlink()
            print(f"  Deleted: {t.path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = tyro.cli(DvcBackupConfig)

    global _VERBOSE
    _VERBOSE = config.verbose

    # --- Test mode ---
    if config.setup_test_dir:
        _setup_test_dir()
        return

    logs_root = Path(config.logs_root)
    if not logs_root.is_dir():
        print(f"ERROR: logs_root not found: {logs_root}")
        sys.exit(1)

    # Step 1: Discover
    print("Discovering targets...")
    summaries = discover_all(logs_root=logs_root)
    if not summaries:
        print("No untracked checkpoints or rollouts found. Nothing to do.")
        return

    # Step 2: Present summary
    print_summary(summaries)

    if config.dry_run:
        print("\n(dry-run mode — no changes made)")
        return

    # Step 3: Prompt for confirmation
    response = input("\nProceed with backup? [y/N] ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    # Step 4: Separate targets into needs_add (fresh) vs needs_push_only (already added)
    # Already-added targets go straight to push — dvc push is a cheap no-op if already
    # synced (hash check only, ~2-5s), so no need to check status first.
    all_targets = [t for s in summaries for t in s.targets]
    needs_add: list[BackupTarget] = [t for t in all_targets if not t.already_added]
    needs_push_only: list[BackupTarget] = [t for t in all_targets if t.already_added]

    if needs_push_only:
        print(f"\n  {len(needs_push_only)} target(s) already have .dvc files — will push directly (skipping add)")
    if not needs_add and not needs_push_only:
        print("\nNothing to do.")
        return

    # Step 5: dvc add fresh targets only
    added_targets: list[BackupTarget] = list(needs_push_only)
    if needs_add:
        print(f"\n--- DVC Add ({len(needs_add)} target(s)) ---")
        for t in needs_add:
            print(f"  Adding {t.kind}: {t.path.name} ({_human_size(t.size_bytes)})...")
            if dvc_add(target=t):
                added_targets.append(t)
            else:
                print(f"    FAILED — skipping {t.path}")

    if not added_targets:
        print("All operations failed. Aborting.")
        return

    # Step 6: Git commit .dvc files (only if we added new ones)
    if needs_add:
        print("\n--- Git Commit ---")
        git_add_and_commit(logs_root=logs_root)

    # Step 7: dvc push all targets
    print(f"\n--- DVC Push ({len(added_targets)} target(s)) ---")
    for t in added_targets:
        print(f"  Pushing: {t.dvc_file.name}...")
        dvc_push(dvc_file=t.dvc_file)

    # Step 8: Verify each target via dvc push (cheap hash check).
    # We use dvc push instead of dvc status --cloud because the latter
    # reports ambiguous "deleted" when local cache has been cleared (after gc).
    # dvc push does an object_exists check against S3 (~2-5s) and is reliable.
    all_to_verify = added_targets
    print(f"\n--- Verification ({len(all_to_verify)} target(s)) ---")
    verified: list[BackupTarget] = []
    failed: list[BackupTarget] = []
    for t in all_to_verify:
        ok = dvc_verify_on_remote(dvc_file=t.dvc_file)
        if ok:
            print(f"  OK: {t.dvc_file.name}")
            verified.append(t)
        else:
            print(f"  FAILED: {t.dvc_file.name}")
            failed.append(t)

    print(f"\n  Verified: {len(verified)}/{len(all_to_verify)}")
    if failed:
        print(f"  Failed:   {len(failed)} (will NOT be deleted)")
        for t in failed:
            print(f"    - {t.path}")

    # Step 9: Prompt for deletion
    if not verified:
        print("No targets verified. Skipping deletion.")
        return

    total_verified_size = sum(t.size_bytes for t in verified)
    response = input(
        f"\nDelete {len(verified)} verified target(s) "
        f"({_human_size(total_verified_size)}) from local disk? [y/N] "
    ).strip().lower()
    if response != "y":
        print("Keeping local data.")
        return

    # Step 10: Delete
    print("\n--- Deleting verified targets ---")
    delete_targets(targets=verified)

    # Step 11: Clean DVC cache
    dvc_gc()

    print("\nDone.")


if __name__ == "__main__":
    main()
