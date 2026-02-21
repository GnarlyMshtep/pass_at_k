"""Checkpoint daemon — backs up and optionally cleans checkpoints during training.

Spawned by the orchestrator as a subprocess. Watches the verl PID and:
  1. Periodically polls for new complete checkpoints
  2. Runs `dvc add` + `dvc push` to back them up
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
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(run_dir: Path) -> logging.Logger:
    log_dir = run_dir / "daemon_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "checkpoint_daemon.log"

    logger = logging.getLogger("checkpoint_daemon")
    logger.setLevel(logging.DEBUG)

    # File handler — everything
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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
    # Non-empty = has files inside (not just an empty dir marker)
    return any(step_dir.iterdir())


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
    result = subprocess.run(
        ["dvc", "gc", "--not-in-remote", "-w", "-f", "-v"],
        capture_output=True,
        text=True,
        cwd=str(dvc_root),
    )
    if result.returncode != 0:
        logger.warning(f"dvc gc failed: {result.stderr.strip()}")
    else:
        logger.info(f"dvc gc complete: {result.stdout.strip()}")


def _dvc_add_and_push(
    checkpoints_dir: Path,
    dvc_root: Path,
    logger: logging.Logger,
) -> bool:
    """Run `dvc add` on the checkpoints dir, then `dvc push`. Returns True on success."""
    rel_path = checkpoints_dir.resolve().relative_to(dvc_root.resolve())

    logger.info(f"Running: dvc add {rel_path}")
    result = subprocess.run(
        ["dvc", "add", str(rel_path)],
        capture_output=True,
        text=True,
        cwd=str(dvc_root),
    )
    if result.returncode != 0:
        logger.error(f"dvc add failed: {result.stderr.strip()}")
        return False
    logger.debug(f"dvc add stdout: {result.stdout.strip()}")

    logger.info("Running: dvc push")
    result = subprocess.run(
        ["dvc", "push"],
        capture_output=True,
        text=True,
        cwd=str(dvc_root),
    )
    if result.returncode != 0:
        logger.error(f"dvc push failed: {result.stderr.strip()}")
        return False
    logger.info(f"dvc push complete: {result.stdout.strip()}")

    # Clean DVC cache to reclaim disk space (data is already on remote)
    _clean_dvc_cache(dvc_root=dvc_root, logger=logger)

    return True


def _verify_backup(
    checkpoints_dir: Path,
    dvc_root: Path,
    logger: logging.Logger,
) -> set[str]:
    """Return set of relative paths that are NOT yet backed up."""
    logger.info("Verifying backup status with dvc status --cloud")
    result = subprocess.run(
        ["dvc", "status", "--cloud"],
        capture_output=True,
        text=True,
        cwd=str(dvc_root),
    )
    not_backed_up: set[str] = set()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("new:"):
            not_backed_up.add(stripped.removeprefix("new:").strip())
    logger.debug(f"Not backed up: {len(not_backed_up)} items")
    return not_backed_up


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
    """One iteration: discover checkpoints, backup, optionally clean."""
    complete_step = _get_complete_step(checkpoints_dir=checkpoints_dir)
    all_steps = _list_checkpoint_steps(checkpoints_dir=checkpoints_dir)
    non_empty_steps = [
        s for s in all_steps
        if _is_checkpoint_non_empty(checkpoints_dir=checkpoints_dir, step=s)
    ]

    if not non_empty_steps:
        logger.debug("No non-empty checkpoints found")
        return

    new_steps = [s for s in non_empty_steps if s not in backed_up_steps]
    if not new_steps:
        logger.debug(f"No new checkpoints to backup (already backed up: {sorted(backed_up_steps)})")
        return

    # Only backup checkpoints that are confirmed complete
    safe_steps = [
        s for s in new_steps
        if complete_step is not None and s <= complete_step
    ]
    if not safe_steps:
        logger.debug(f"New steps {new_steps} but none confirmed complete (complete_step={complete_step})")
        return

    logger.info(f"New complete checkpoints to backup: {safe_steps}")

    # DVC add + push
    success = _dvc_add_and_push(
        checkpoints_dir=checkpoints_dir,
        dvc_root=dvc_root,
        logger=logger,
    )
    if not success:
        logger.warning("DVC add/push failed — will retry next poll")
        return

    # Verify backup
    not_backed_up = _verify_backup(
        checkpoints_dir=checkpoints_dir,
        dvc_root=dvc_root,
        logger=logger,
    )

    # Mark backed-up steps and optionally clean
    for step in safe_steps:
        step_rel = str(
            (checkpoints_dir / f"global_step_{step}")
            .resolve()
            .relative_to(dvc_root.resolve())
        )
        # Check if any files in this step dir are NOT backed up
        step_files_not_backed = [
            p for p in not_backed_up
            if p.startswith(step_rel)
        ]
        if step_files_not_backed:
            logger.warning(
                f"global_step_{step}: {len(step_files_not_backed)} files not confirmed backed up — skipping clean"
            )
            continue

        backed_up_steps.add(step)
        logger.info(f"global_step_{step}: backup confirmed")

        if clean_after_backup:
            # Don't clean the latest checkpoint — verl might need it for resume
            if step < max(non_empty_steps):
                _clean_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    step=step,
                    logger=logger,
                )
            else:
                logger.debug(
                    f"global_step_{step}: not cleaning (latest non-empty checkpoint)"
                )


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
