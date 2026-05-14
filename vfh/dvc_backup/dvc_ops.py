"""DVC and git subprocess wrappers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from vfh.dvc_backup.backup_log import log, vprint
from vfh.dvc_backup.config import (
    DVC_ADD_TIMEOUT,
    DVC_GC_TIMEOUT,
    DVC_PULL_TIMEOUT,
    DVC_PUSH_TIMEOUT,
    DVC_STATUS_TIMEOUT,
)
from vfh.dvc_backup.types import BackupTarget


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
        log(f"  WARNING: DVC lock held by PID {lock_pid} (still running).")
        log(f"  Another DVC process is active. Wait for it or kill it manually.")
        return False
    except (ValueError, ProcessLookupError, PermissionError):
        # Process is dead or lock file is not a valid PID — stale lock
        lock_path.unlink()
        log(f"  Cleared stale DVC lock (dead process).")
        return True


# ---------------------------------------------------------------------------
# DVC commands
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
                vprint(f"Retrying after stale lock clear: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

        if result.returncode != 0:
            log(f"  ERROR [{label}]: exit {result.returncode}")
            if result.stdout.strip():
                log(f"    stdout: {result.stdout.strip()}")
            if result.stderr.strip():
                log(f"    stderr: {result.stderr.strip()}")
        else:
            vprint(f"OK: {label}")
        return result
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT [{label}]: {timeout}s exceeded for: {' '.join(cmd)}")
        raise


def dvc_add(target: BackupTarget) -> bool:
    """Run dvc add on a target. Returns True on success.

    Handles two common blockers automatically:
    - Target tracked by git → git rm -r --cached, then retry
    - Legacy per-subdir .dvc files overlap → dvc remove them, then retry
    """
    # Pre-check: remove legacy per-subdir .dvc files (e.g. rollouts/train.dvc)
    # that would conflict with adding the parent dir (rollouts/)
    if target.path.is_dir():
        for legacy_dvc in target.path.glob("*.dvc"):
            log(f"    Removing legacy {legacy_dvc.name} (overlaps with {target.path.name})...")
            _run_dvc(
                args=["remove", str(legacy_dvc)],
                timeout=60,
                label=f"dvc remove {legacy_dvc.name}",
            )

    result = _run_dvc(
        args=["add", str(target.path)],
        timeout=DVC_ADD_TIMEOUT,
        label=f"dvc add {target.path.name}",
    )
    # Handle "already tracked by SCM" error: un-track from git, then retry
    if result.returncode != 0 and "already tracked by SCM" in result.stderr:
        log(f"    Target tracked by git — running git rm -r --cached...")
        git_result = subprocess.run(
            ["git", "rm", "-r", "--cached", str(target.path)],
            capture_output=True,
            text=True,
        )
        if git_result.returncode != 0:
            log(f"    ERROR [git rm --cached]: {git_result.stderr.strip()}")
            return False
        # Retry dvc add
        result = _run_dvc(
            args=["add", str(target.path)],
            timeout=DVC_ADD_TIMEOUT,
            label=f"dvc add {target.path.name} (retry after git rm)",
        )
    # Handle "overlapping with other DVC tracked output" — remove + retry
    if result.returncode != 0 and "overlapping with other DVC tracked output" in result.stderr:
        log(f"    Overlapping DVC output detected — removing conflicting .dvc files...")
        for legacy_dvc in target.path.glob("**/*.dvc"):
            _run_dvc(
                args=["remove", str(legacy_dvc)],
                timeout=60,
                label=f"dvc remove {legacy_dvc.name}",
            )
        result = _run_dvc(
            args=["add", str(target.path)],
            timeout=DVC_ADD_TIMEOUT,
            label=f"dvc add {target.path.name} (retry after dvc remove)",
        )
    return result.returncode == 0


def dvc_push(dvc_file: Path) -> bool:
    """Run dvc push on a .dvc file. Returns True on success."""
    result = _run_dvc(
        args=["push", str(dvc_file)],
        timeout=DVC_PUSH_TIMEOUT,
        label=f"dvc push {dvc_file.name}",
    )
    return result.returncode == 0


def dvc_add_batch(targets: list[BackupTarget]) -> list[BackupTarget]:
    """Run dvc add on multiple targets at once. Returns list of successfully added targets.

    Handles legacy .dvc cleanup and git rm --cached before the batch add.
    Falls back to per-target add if the batch fails.
    """
    if not targets:
        return []

    # Pre-cleanup: remove legacy per-subdir .dvc files + git rm --cached
    for t in targets:
        if t.path.is_dir():
            for legacy_dvc in t.path.glob("*.dvc"):
                log(f"    Removing legacy {legacy_dvc.name} ({t.run_id})...")
                _run_dvc(args=["remove", str(legacy_dvc)], timeout=60,
                         label=f"dvc remove {legacy_dvc.name}")

    # Try git rm --cached for all targets at once (in case any are git-tracked)
    git_result = subprocess.run(
        ["git", "rm", "-r", "--cached", "--ignore-unmatch"] + [str(t.path) for t in targets],
        capture_output=True, text=True,
    )
    if git_result.returncode != 0:
        vprint(f"git rm --cached warning: {git_result.stderr.strip()}")

    # Batch dvc add
    paths = [str(t.path) for t in targets]
    log(f"  Running dvc add on {len(targets)} targets at once...")
    result = _run_dvc(
        args=["add"] + paths,
        timeout=DVC_ADD_TIMEOUT * 2,  # batch timeout (2h)
        label=f"dvc add (batch of {len(targets)})",
    )

    if result.returncode == 0:
        return list(targets)

    # Batch failed — fall back to per-target add
    log(f"  Batch dvc add failed — falling back to per-target add...")
    added: list[BackupTarget] = []
    for t in targets:
        if dvc_add(target=t):
            added.append(t)
        else:
            log(f"    FAILED — skipping {t.label}")
    return added


def dvc_push_batch(targets: list[BackupTarget]) -> list[BackupTarget]:
    """Run dvc push on multiple .dvc files at once. Returns list of successfully pushed targets.

    Falls back to per-target push if the batch fails.
    """
    if not targets:
        return []

    dvc_files = [str(t.dvc_file) for t in targets]
    log(f"  Running dvc push on {len(targets)} targets at once...")
    result = _run_dvc(
        args=["push"] + dvc_files,
        timeout=DVC_PUSH_TIMEOUT * 2,  # batch timeout (4h)
        label=f"dvc push (batch of {len(targets)})",
    )

    if result.returncode == 0:
        return list(targets)

    # Batch failed — fall back to per-target push
    log(f"  Batch dvc push failed — falling back to per-target push...")
    pushed: list[BackupTarget] = []
    for t in targets:
        if dvc_push(dvc_file=t.dvc_file):
            pushed.append(t)
        else:
            log(f"    PUSH FAILED — skipping {t.label}")
    return pushed


def dvc_pull(dvc_file: Path) -> bool:
    """Run dvc pull on a .dvc file. Returns True on success."""
    result = _run_dvc(
        args=["pull", str(dvc_file)],
        timeout=DVC_PULL_TIMEOUT,
        label=f"dvc pull {dvc_file.name}",
    )
    return result.returncode == 0


def dvc_checkout_to(dvc_file: Path, dest: Path) -> bool:
    """Pull a .dvc file's contents into a specific destination directory.

    Workflow: dvc pull (populates cache + checks out to original path),
    then move the checked-out data to dest.
    """
    import shutil

    original_path = dvc_file.parent / dvc_file.stem
    if not dvc_pull(dvc_file=dvc_file):
        return False
    if not original_path.exists():
        log(f"  WARNING: dvc pull succeeded but {original_path} doesn't exist")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(src=str(original_path), dst=str(dest))
    return True


def dvc_pull_batch(targets: list[BackupTarget]) -> list[BackupTarget]:
    """Run dvc pull on multiple .dvc files at once. Returns list of successfully pulled targets.

    Falls back to per-target pull if the batch fails.
    """
    if not targets:
        return []

    dvc_files = [str(t.dvc_file) for t in targets]
    log(f"  Running dvc pull on {len(targets)} targets at once...")
    result = _run_dvc(
        args=["pull"] + dvc_files,
        timeout=DVC_PULL_TIMEOUT * 2,
        label=f"dvc pull (batch of {len(targets)})",
    )

    if result.returncode == 0:
        return list(targets)

    log(f"  Batch dvc pull failed — falling back to per-target pull...")
    pulled: list[BackupTarget] = []
    for t in targets:
        if dvc_pull(dvc_file=t.dvc_file):
            pulled.append(t)
        else:
            log(f"    PULL FAILED — skipping {t.label}")
    return pulled


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
        timeout=DVC_STATUS_TIMEOUT,
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
    vprint(f"Unexpected dvc status output: {result.stdout.strip()}")
    return "error"


def dvc_gc() -> None:
    """Run dvc gc to clean local cache."""
    log("Cleaning local DVC cache (dvc gc --not-in-remote -w -f)...")
    _run_dvc(
        args=["gc", "--not-in-remote", "-w", "-f"],
        timeout=DVC_GC_TIMEOUT,
        label="dvc gc",
    )


def clear_cache_fast() -> None:
    """Nuke the entire DVC cache directory. Much faster than dvc gc.

    Use this when all data has been verified on remote — the cache is
    purely redundant and safe to delete entirely.
    """
    import shutil
    cache_dir = Path(".dvc/cache/files/md5")
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        log("  Cleared DVC cache (fast rm).")
    else:
        log("  No DVC cache to clear.")


def dvc_remove_cache_for(dvc_file: Path) -> bool:
    """Remove cached data for a specific .dvc file by parsing the hash and deleting from cache."""
    import re

    try:
        content = dvc_file.read_text()
    except FileNotFoundError:
        return False

    # Extract md5 hash from .dvc file (e.g. "md5: abc123def456.dir")
    match = re.search(r"md5:\s*([a-f0-9]+)(?:\.dir)?", content)
    if not match:
        vprint(f"Could not parse md5 from {dvc_file}")
        return False

    md5_hash = match.group(1)
    prefix = md5_hash[:2]
    rest = md5_hash[2:]

    cache_base = Path(".dvc/cache/files/md5")

    # Remove the .dir manifest
    dir_cache = cache_base / prefix / f"{rest}.dir"
    if dir_cache.exists():
        dir_cache.unlink()
        vprint(f"Removed cache .dir: {dir_cache}")

    # Remove the hash entry itself (for non-dir files)
    hash_cache = cache_base / prefix / rest
    if hash_cache.exists():
        hash_cache.unlink()
        vprint(f"Removed cache entry: {hash_cache}")

    # For directory targets, we also need to remove all individual file caches
    # referenced in the .dir manifest. But the manifest might already be gone.
    # Fall back to dvc gc which handles this properly.
    return True


def git_add_and_commit(logs_root: Path) -> bool:
    """Git add all .dvc and .gitignore files under logs_root, then commit."""
    dvc_files = list(logs_root.rglob("*.dvc"))
    gitignore_files = list(logs_root.rglob(".gitignore"))
    files_to_stage = [str(f) for f in dvc_files + gitignore_files]

    if not files_to_stage:
        log("  No .dvc or .gitignore files to commit.")
        return True

    result = subprocess.run(
        ["git", "add"] + files_to_stage,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"  ERROR [git add]: {result.stderr.strip()}")
        return False

    result = subprocess.run(
        ["git", "commit", "-m", "dvc: backup checkpoints and rollouts"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        combined = result.stdout + result.stderr
        if "nothing to commit" in combined:
            log("  Nothing new to commit.")
            return True
        log(f"  ERROR [git commit]: {result.stderr.strip()}")
        return False

    log("  Committed DVC tracking files.")
    return True
