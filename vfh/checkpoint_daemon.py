"""Checkpoint daemon — backs up and optionally cleans checkpoints during training.

Spawned by the orchestrator as a subprocess. Watches the verl PID and:
  1. Periodically polls for new complete checkpoints
  2. Runs `dvc add` + `dvc push` per global_step (each step gets its own .dvc file)
  3. Optionally removes checkpoint contents after confirmed backup (keeps empty dir)
  4. When the verl process dies, does one final backup pass and exits

Logs all actions with timestamps to {run_dir}/daemon_logs/checkpoint_daemon.log

Usage (called by orchestrator, not directly):
    python -m vfh.checkpoint_daemon \\
        --run-dir logs/VerlRun/02/21/my_run/ \\
        --verl-pid 12345 \\
        --poll-interval 300 \\
        --clean-after-backup
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _eastern_time(*args: object) -> time.struct_time:
    """Converter for logging formatters: returns current time in US Eastern."""
    return datetime.datetime.now(tz=_EASTERN).timetuple()


def _setup_logging(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "daemon_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "checkpoint_daemon.log"

    logger = logging.getLogger("checkpoint_daemon")
    logger.setLevel(logging.DEBUG)

    # File handler — everything
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fmt.converter = _eastern_time
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stderr handler — INFO+
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[checkpoint_daemon] %(message)s"))
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# PID monitoring
# ---------------------------------------------------------------------------


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # signal 0 = no-op, just checks existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission — still alive
        return True


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _get_complete_step(checkpoints_dir: Path) -> Optional[int]:
    """Read latest_checkpointed_iteration.txt to find the last complete step."""
    tracker = checkpoints_dir / "latest_checkpointed_iteration.txt"
    if not tracker.exists():
        return None
    content = tracker.read_text().strip()
    if not content:
        return None
    return int(content)


def _list_checkpoint_steps(checkpoints_dir: Path) -> list[int]:
    """List all global_step_N directories, sorted ascending."""
    steps: list[int] = []
    if not checkpoints_dir.exists():
        return steps
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.startswith("global_step_"):
            try:
                steps.append(int(d.name.split("_")[-1]))
            except ValueError:
                pass
    steps.sort()
    return steps


def _is_checkpoint_non_empty(checkpoints_dir: Path, step: int) -> bool:
    """Check if a checkpoint dir has actual content (not already cleaned)."""
    step_dir = checkpoints_dir / f"global_step_{step}"
    if not step_dir.exists():
        return False
    # Non-empty = has files other than the .cleaned_by_daemon marker
    return any(item.name != ".cleaned_by_daemon" for item in step_dir.iterdir())


# ---------------------------------------------------------------------------
# DVC backup
# ---------------------------------------------------------------------------


def _find_dvc_root(start: Path) -> Optional[Path]:
    """Walk up to find the DVC root (directory containing .dvc/)."""
    current = start.resolve()
    while True:
        if (current / ".dvc").is_dir():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _clean_dvc_cache(dvc_root: Path, logger: logging.Logger) -> None:
    """Run dvc gc to clean cache of files already pushed to remote."""
    logger.info("Running: dvc gc --not-in-remote -w -f -v")
    try:
        result = subprocess.run(
            ["dvc", "gc", "--not-in-remote", "-w", "-f", "-v"],
            capture_output=True,
            text=True,
            cwd=str(dvc_root),
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.warning("dvc gc timed out after 600s — will retry next poll")
        return
    if result.returncode != 0:
        logger.warning(f"dvc gc failed: {result.stderr.strip()}")
    else:
        logger.info(f"dvc gc complete: {result.stdout.strip()}")


def _dvc_add_and_push_step(
    step_dir: Path,
    dvc_root: Path,
    logger: logging.Logger,
) -> bool:
    """Run `dvc add` on a single global_step dir, then `dvc push` its .dvc file. Returns True on success."""
    rel_path = step_dir.resolve().relative_to(dvc_root.resolve())
    dvc_file_rel = str(rel_path) + ".dvc"

    logger.info(f"Running: dvc add {rel_path}")
    try:
        result = subprocess.run(
            ["dvc", "add", str(rel_path)],
            capture_output=True,
            text=True,
            cwd=str(dvc_root),
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"dvc add timed out after 1800s for {step_dir.name}")
        return False
    if result.returncode != 0:
        logger.error(f"dvc add failed for {step_dir.name}: {result.stderr.strip()}")
        return False
    logger.debug(f"dvc add stdout: {result.stdout.strip()}")

    logger.info(f"Running: dvc push {dvc_file_rel}")
    try:
        result = subprocess.run(
            ["dvc", "push", dvc_file_rel],
            capture_output=True,
            text=True,
            cwd=str(dvc_root),
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"dvc push timed out after 1800s for {step_dir.name}")
        return False
    if result.returncode != 0:
        logger.error(f"dvc push failed for {step_dir.name}: {result.stderr.strip()}")
        return False
    logger.info(f"dvc push complete for {step_dir.name}: {result.stdout.strip()}")

    return True


def _verify_step_backup(
    step_dir: Path,
    dvc_root: Path,
    logger: logging.Logger,
) -> bool:
    """Check if a single step's .dvc file is fully backed up to remote."""
    rel_path = step_dir.resolve().relative_to(dvc_root.resolve())
    dvc_file_rel = str(rel_path) + ".dvc"
    logger.info(f"Verifying backup: dvc status --cloud {dvc_file_rel}")
    result = subprocess.run(
        ["dvc", "status", "--cloud", dvc_file_rel],
        capture_output=True,
        text=True,
        cwd=str(dvc_root),
    )
    for line in result.stdout.splitlines():
        if "new:" in line.strip():
            logger.debug(f"Not fully backed up: {dvc_file_rel}")
            return False
    return True


def _remove_old_checkpoints_dvc(
    checkpoints_dir: Path,
    dvc_root: Path,
    logger: logging.Logger,
) -> None:
    """Remove old-style checkpoints.dvc that tracked the entire directory (migrating to per-step)."""
    old_dvc = checkpoints_dir.with_suffix(".dvc")
    if old_dvc.exists():
        rel_path = old_dvc.resolve().relative_to(dvc_root.resolve())
        logger.info(f"Removing old-style whole-dir DVC tracking: dvc remove {rel_path}")
        subprocess.run(
            ["dvc", "remove", str(rel_path)],
            capture_output=True,
            text=True,
            cwd=str(dvc_root),
        )


def _clean_checkpoint(
    checkpoints_dir: Path,
    step: int,
    logger: logging.Logger,
) -> None:
    """Remove checkpoint contents but keep the empty directory as a marker."""
    step_dir = checkpoints_dir / f"global_step_{step}"
    if not step_dir.exists():
        return

    # Count contents before removal
    contents = list(step_dir.iterdir())
    if not contents:
        logger.debug(f"global_step_{step} already empty, skipping clean")
        return

    for item in contents:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # Leave a marker file so we know it was cleaned (not just failed)
    (step_dir / ".cleaned_by_daemon").write_text(
        f"Cleaned at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )
    logger.info(f"Cleaned global_step_{step} (kept empty dir as marker)")


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------


def run_daemon(
    run_dir: Path,
    verl_pid: int,
    poll_interval: float,
    clean_after_backup: bool,
) -> None:
    logger = _setup_logging(run_dir=run_dir)
    checkpoints_dir = run_dir / "checkpoints"

    logger.info("=" * 60)
    logger.info(f"Checkpoint daemon started")
    logger.info(f"  run_dir:            {run_dir}")
    logger.info(f"  verl_pid:           {verl_pid}")
    logger.info(f"  poll_interval:      {poll_interval}s")
    logger.info(f"  clean_after_backup: {clean_after_backup}")

    dvc_root = _find_dvc_root(start=run_dir)
    if dvc_root is None:
        logger.error("Could not find DVC root! Daemon exiting.")
        sys.exit(1)
    logger.info(f"  dvc_root:           {dvc_root}")
    logger.info("=" * 60)

    # Migrate from old whole-dir tracking to per-step tracking
    _remove_old_checkpoints_dvc(
        checkpoints_dir=checkpoints_dir,
        dvc_root=dvc_root,
        logger=logger,
    )

    backed_up_steps: set[int] = set()

    while True:
        verl_alive = _is_pid_alive(pid=verl_pid)
        if not verl_alive:
            logger.info(f"Verl process (PID {verl_pid}) has exited. Running final backup pass.")

        _backup_pass(
            checkpoints_dir=checkpoints_dir,
            dvc_root=dvc_root,
            clean_after_backup=clean_after_backup,
            backed_up_steps=backed_up_steps,
            logger=logger,
        )

        if not verl_alive:
            logger.info("Final backup pass complete. Daemon exiting.")
            break

        logger.debug(f"Sleeping {poll_interval}s until next poll")
        time.sleep(poll_interval)


def _backup_pass(
    checkpoints_dir: Path,
    dvc_root: Path,
    clean_after_backup: bool,
    backed_up_steps: set[int],
    logger: logging.Logger,
) -> None:
    """One iteration: discover checkpoints, backup new ones, clean old backed-up ones."""
    complete_step = _get_complete_step(checkpoints_dir=checkpoints_dir)
    all_steps = _list_checkpoint_steps(checkpoints_dir=checkpoints_dir)
    non_empty_steps = [
        s for s in all_steps
        if _is_checkpoint_non_empty(checkpoints_dir=checkpoints_dir, step=s)
    ]

    if not non_empty_steps:
        logger.debug("No non-empty checkpoints found")
        return

    # --- Phase 1: Backup new checkpoints (one DVC file per step) ---
    new_steps = [s for s in non_empty_steps if s not in backed_up_steps]
    if not new_steps:
        logger.debug(f"No new checkpoints to backup (already backed up: {sorted(backed_up_steps)})")
    else:
        # Only backup checkpoints that are confirmed complete
        safe_steps = [
            s for s in new_steps
            if complete_step is not None and s <= complete_step
        ]
        if not safe_steps:
            logger.debug(f"New steps {new_steps} but none confirmed complete (complete_step={complete_step})")
        else:
            logger.info(f"New complete checkpoints to backup: {safe_steps}")

            newly_backed: list[int] = []
            for step in safe_steps:
                step_dir = checkpoints_dir / f"global_step_{step}"

                success = _dvc_add_and_push_step(
                    step_dir=step_dir,
                    dvc_root=dvc_root,
                    logger=logger,
                )
                if not success:
                    logger.warning(f"DVC add/push failed for global_step_{step} — will retry next poll")
                    continue

                if _verify_step_backup(step_dir=step_dir, dvc_root=dvc_root, logger=logger):
                    backed_up_steps.add(step)
                    newly_backed.append(step)
                    logger.info(f"global_step_{step}: backup confirmed")
                else:
                    logger.warning(f"global_step_{step}: backup NOT confirmed on remote")

            # Clean DVC cache once after all pushes
            if newly_backed:
                _clean_dvc_cache(dvc_root=dvc_root, logger=logger)

    # --- Phase 2: Clean old backed-up checkpoints ---
    if clean_after_backup and backed_up_steps:
        # Re-check which steps are currently non-empty (may have changed during backup)
        current_non_empty = [
            s for s in _list_checkpoint_steps(checkpoints_dir=checkpoints_dir)
            if _is_checkpoint_non_empty(checkpoints_dir=checkpoints_dir, step=s)
        ]
        if current_non_empty:
            latest = max(current_non_empty)
            for step in sorted(backed_up_steps):
                if step < latest and _is_checkpoint_non_empty(checkpoints_dir=checkpoints_dir, step=step):
                    _clean_checkpoint(
                        checkpoints_dir=checkpoints_dir,
                        step=step,
                        logger=logger,
                    )
                elif step == latest:
                    logger.debug(f"global_step_{step}: not cleaning (latest non-empty checkpoint)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VFH checkpoint daemon")
    parser.add_argument("--run-dir", required=True, help="Path to the VFH run directory")
    parser.add_argument("--verl-pid", type=int, required=True, help="PID of the verl training process")
    parser.add_argument("--poll-interval", type=float, default=300, help="Seconds between polls (default: 300)")
    parser.add_argument("--clean-after-backup", action="store_true", help="Remove checkpoint contents after confirmed backup")
    args = parser.parse_args()

    run_daemon(
        run_dir=Path(args.run_dir),
        verl_pid=args.verl_pid,
        poll_interval=args.poll_interval,
        clean_after_backup=args.clean_after_backup,
    )


if __name__ == "__main__":
    main()
