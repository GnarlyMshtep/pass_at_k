"""Logging for DVC backup — append-only log file + stdout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

# Resolved relative to git root at import time.
_LOG_PATH: Path | None = None

# Module-level verbose flag, set by pipeline.main().
_VERBOSE = False


def init_log(logs_root: Path) -> None:
    """Set the log file path. Called once at startup."""
    global _LOG_PATH
    _LOG_PATH = logs_root / "dvc_backup_logs.txt"


def set_verbose(verbose: bool) -> None:
    """Enable/disable verbose output."""
    global _VERBOSE
    _VERBOSE = verbose


def log(msg: str) -> None:
    """Append a timestamped line to the backup log AND print to stdout."""
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a") as f:
            f.write(line + "\n")
    print(line)


def vprint(msg: str) -> None:
    """Print only if verbose mode is on."""
    if _VERBOSE:
        print(f"  [verbose] {msg}")
