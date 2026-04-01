"""Configuration and constants for DVC backup."""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Timeouts (seconds)
# ---------------------------------------------------------------------------

DVC_ADD_TIMEOUT = 3600     # 60 min
DVC_PUSH_TIMEOUT = 7200    # 120 min
DVC_GC_TIMEOUT = 600       # 10 min
DVC_STATUS_TIMEOUT = 120   # 2 min
DVC_PULL_TIMEOUT = 7200    # 120 min

# Space safety multiplier: dvc add doubles data (original + cache),
# plus headroom for round-trip verification temp copy.
SPACE_MULTIPLIER = 2.5


# ---------------------------------------------------------------------------
# Config dataclass (parsed by tyro)
# ---------------------------------------------------------------------------


@dataclass
class DvcBackupConfig:
    """Configuration for DVC backup of VerlRun checkpoints and rollouts."""

    logs_root: str = "logs/VerlRun"
    dry_run: bool = False
    verbose: bool = False
    setup_test_dir: bool = False  # create a fake VerlRun dir for testing

    # Auto-approve: skip confirmation prompts.
    yes: bool = False

    # Batching: max GB to process per batch (required).
    space_budget_gb: float = 500

    # Max batches to process (for testing). None = no limit.
    max_batches: int | None = None

    # Single-dir mode: backup only this run dir (overrides logs_root discovery).
    path: str | None = None

    # Verification: skip the round-trip pull-and-compare step (faster, less safe).
    skip_roundtrip_verify: bool = False

    # Where to move data during round-trip verification.
    # Default: {target_parent}/.dvc_verify_tmp/ (same FS, instant mv).
    verify_temp_dir: str | None = None

    # Move verification temp data to ~/dvc_verify_tmp/ instead of same FS.
    # Slower (cross-FS copy) but doesn't consume shared FS space.
    verify_on_local_fs: bool = False
