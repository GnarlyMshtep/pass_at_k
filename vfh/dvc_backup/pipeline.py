"""Main backup pipeline: discovery → batching → add → push → verify → delete."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import tyro

from vfh.dvc_backup.backup_log import init_log, log, set_verbose
from vfh.dvc_backup.config import SPACE_MULTIPLIER, DvcBackupConfig
from vfh.dvc_backup.discovery import discover_all, discover_single
from vfh.dvc_backup.dvc_ops import (
    clear_cache_fast,
    dvc_add_batch,
    dvc_gc,
    dvc_push_batch,
    git_add_and_commit,
)
from vfh.dvc_backup.presentation import human_size, print_summary
from vfh.dvc_backup.test_setup import setup_test_dir
from vfh.dvc_backup.types import BackupTarget, RunSummary
from vfh.dvc_backup.verification import roundtrip_verify


# ---------------------------------------------------------------------------
# Space check
# ---------------------------------------------------------------------------


def _check_free_space(
    logs_root: Path,
    all_targets: list[BackupTarget],
    space_budget_gb: float | None,
) -> None:
    """Check that we have enough free disk space. Exits on failure."""
    free_bytes = shutil.disk_usage(logs_root).free

    budget_bytes = space_budget_gb * 1e9
    needed = budget_bytes * SPACE_MULTIPLIER
    if free_bytes < needed:
        log(f"ERROR: --space-budget-gb {space_budget_gb} requires "
            f"~{human_size(needed)} free but only {human_size(free_bytes)} available "
            f"({SPACE_MULTIPLIER}x multiplier: original + cache + verification headroom).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def _build_batches(
    targets: list[BackupTarget],
    budget_bytes: float,
) -> list[list[BackupTarget]]:
    """Split targets into batches that fit within the budget.

    Targets are sorted by size ascending so small ones batch together.
    If the smallest remaining target exceeds the budget, raises an error.
    """
    remaining = sorted(targets, key=lambda t: t.size_bytes)
    batches: list[list[BackupTarget]] = []

    while remaining:
        batch: list[BackupTarget] = []
        batch_size = 0.0

        i = 0
        while i < len(remaining):
            t = remaining[i]
            if batch_size + t.size_bytes <= budget_bytes:
                batch.append(t)
                batch_size += t.size_bytes
                remaining.pop(i)
            else:
                i += 1

        if not batch:
            # No target fits — the smallest one exceeds the budget
            smallest = remaining[0]
            log(f"ERROR: Smallest remaining target ({smallest.path.name}, "
                f"{human_size(smallest.size_bytes)}) exceeds the effective budget "
                f"({human_size(budget_bytes)}). Cannot proceed.")
            log(f"  Increase --space-budget-gb or free more disk space.")
            sys.exit(1)

        batches.append(batch)

    return batches


def _process_batch(
    batch: list[BackupTarget],
    all_targets: list[BackupTarget],
    logs_root: Path,
    config: DvcBackupConfig,
) -> tuple[list[BackupTarget], int]:
    """Process a single batch: add → push → verify → delete.

    Returns (verified_targets, freed_bytes).
    """
    needs_add = [t for t in batch if not t.already_added]
    needs_push_only = [t for t in batch if t.already_added]

    if needs_push_only:
        log(f"  {len(needs_push_only)} target(s) already have .dvc files — pushing directly")

    # Step 1: dvc add fresh targets (batched)
    added_targets: list[BackupTarget] = list(needs_push_only)
    if needs_add:
        total_add_size = sum(t.size_bytes for t in needs_add)
        log(f"  --- DVC Add ({len(needs_add)} target(s), {human_size(total_add_size)}) ---")
        newly_added = dvc_add_batch(targets=needs_add)
        added_targets.extend(newly_added)
        if len(newly_added) < len(needs_add):
            log(f"  {len(needs_add) - len(newly_added)} target(s) failed to add")

    if not added_targets:
        log("  All add operations failed in this batch.")
        return [], 0

    # Step 2: git commit .dvc files
    if needs_add:
        log("  --- Git Commit ---")
        git_add_and_commit(logs_root=logs_root)

    # Step 3: dvc push (batched)
    total_push_size = sum(t.size_bytes for t in added_targets)
    log(f"  --- DVC Push ({len(added_targets)} target(s), {human_size(total_push_size)}) ---")
    pushed = dvc_push_batch(targets=added_targets)
    if len(pushed) < len(added_targets):
        log(f"  {len(added_targets) - len(pushed)} target(s) failed to push")

    if not pushed:
        log("  All push operations failed in this batch.")
        return [], 0

    # Step 4: Round-trip verification
    verified: list[BackupTarget] = []
    if config.skip_roundtrip_verify:
        log(f"  --- Skipping round-trip verification (--skip-roundtrip-verify) ---")
        verified = pushed
    else:
        log(f"  --- Round-Trip Verification ({len(pushed)} target(s)) ---")
        # Clear DVC cache ONCE before all verifications so pulls must come from S3
        log(f"  Clearing DVC cache before verification...")
        clear_cache_fast()
        for t in pushed:
            ok = roundtrip_verify(
                target=t,
                verify_temp_dir=config.verify_temp_dir,
                verify_on_local_fs=config.verify_on_local_fs,
            )
            if ok:
                verified.append(t)
            else:
                log(f"    VERIFICATION FAILED for {t.path.name} — keeping local data")

    # Step 5: Report
    log(f"  Verified: {len(verified)}/{len(pushed)}")
    if not verified:
        log("  No targets verified. Skipping deletion.")
        return [], 0

    # Step 6: Delete verified originals (data was already moved away during
    # verification and temp copy deleted on success, so the pulled copy is
    # now the canonical local copy — nothing more to delete for verified targets).
    # For --skip-roundtrip-verify, prompt for deletion.
    freed_bytes = 0
    if config.skip_roundtrip_verify:
        total_verified_size = sum(t.size_bytes for t in verified)
        if config.yes:
            do_delete = True
            log(f"  Auto-approved deletion of {len(verified)} target(s) ({human_size(total_verified_size)})")
        else:
            response = input(
                f"\n  Delete {len(verified)} verified target(s) "
                f"({human_size(total_verified_size)}) from local disk? [y/N] "
            ).strip().lower()
            do_delete = response == "y"
        if do_delete:
            log("  --- Deleting verified targets ---")
            for t in verified:
                if t.path.is_dir():
                    shutil.rmtree(t.path)
                    log(f"  Deleted: {t.path}")
                    freed_bytes += t.size_bytes
        else:
            log("  Keeping local data.")
    else:
        # Round-trip verified: both copies (temp original + pulled) were deleted.
        freed_bytes = sum(t.size_bytes for t in verified)

    # Step 7: Clean DVC cache (fast rm since all data was verified on remote)
    log("  --- Cleaning DVC cache ---")
    clear_cache_fast()

    return verified, freed_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = tyro.cli(DvcBackupConfig)

    set_verbose(verbose=config.verbose)

    # --- Test mode ---
    if config.setup_test_dir:
        setup_test_dir()
        return

    logs_root = Path(config.logs_root)
    if not config.path and not logs_root.is_dir():
        print(f"ERROR: logs_root not found: {logs_root}")
        sys.exit(1)

    # Init logging
    init_log(logs_root=logs_root)
    log("=" * 60)
    log(f"DVC Backup started")
    log(f"  Config: space_budget_gb={config.space_budget_gb}, "
        f"max_batches={config.max_batches}, "
        f"skip_verify={config.skip_roundtrip_verify}, "
        f"path={config.path}, dry_run={config.dry_run}")

    # --- Discovery ---
    if config.path:
        log(f"Single-dir mode: {config.path}")
        summaries = discover_single(run_dir=Path(config.path))
    else:
        log("Discovering targets...")
        summaries = discover_all(logs_root=logs_root)

    if not summaries:
        log("No untracked checkpoints or rollouts found. Nothing to do.")
        return

    # --- Summary ---
    print_summary(summaries=summaries)

    all_targets = [t for s in summaries for t in s.targets]

    if config.dry_run:
        log("\n(dry-run mode — no changes made)")
        return

    # --- Space check ---
    _check_free_space(
        logs_root=logs_root,
        all_targets=all_targets,
        space_budget_gb=config.space_budget_gb,
    )

    # --- Confirmation ---
    total_size = sum(t.size_bytes for t in all_targets)
    if config.yes:
        log(f"\nAuto-approved backup of {len(all_targets)} target(s) ({human_size(total_size)})")
    else:
        response = input(
            f"\nProceed with backup of {len(all_targets)} target(s) "
            f"({human_size(total_size)})? [y/N] "
        ).strip().lower()
        if response != "y":
            log("Aborted.")
            return

    # --- Execute ---
    _run_batched(config=config, all_targets=all_targets, logs_root=logs_root)

    log("\nDone.")


def _run_batched(
    config: DvcBackupConfig,
    all_targets: list[BackupTarget],
    logs_root: Path,
) -> None:
    """Process targets in batches, with freed space rolling forward."""
    budget_bytes = config.space_budget_gb * 1e9
    freed_bytes = 0.0

    # Build initial batches based on starting budget
    # We rebuild after each batch because freed_bytes grows
    remaining = list(all_targets)
    batch_num = 0

    while remaining:
        effective_budget = budget_bytes + freed_bytes
        batch_num += 1

        # Greedily fill a batch up to effective budget (smallest first)
        remaining_sorted = sorted(remaining, key=lambda t: t.size_bytes)
        batch: list[BackupTarget] = []
        batch_size = 0.0

        leftover: list[BackupTarget] = []
        for t in remaining_sorted:
            if batch_size + t.size_bytes <= effective_budget:
                batch.append(t)
                batch_size += t.size_bytes
            else:
                leftover.append(t)

        if not batch:
            smallest = remaining_sorted[0]
            log(f"ERROR: Smallest remaining target ({smallest.path.name}, "
                f"{human_size(smallest.size_bytes)}) exceeds effective budget "
                f"({human_size(effective_budget)}). Cannot proceed.")
            log(f"  Freed so far: {human_size(freed_bytes)}. "
                f"Increase --space-budget-gb or free more disk space.")
            sys.exit(1)

        remaining = leftover

        log(f"\n{'='*60}")
        log(f"Batch {batch_num}: {len(batch)} targets, {human_size(batch_size)} "
            f"(budget: {human_size(effective_budget)}, "
            f"freed so far: {human_size(freed_bytes)})")
        log(f"  Remaining after this batch: {len(remaining)} targets")

        verified, batch_freed = _process_batch(
            batch=batch,
            all_targets=all_targets,
            logs_root=logs_root,
            config=config,
        )

        freed_bytes += batch_freed
        log(f"  Batch freed {human_size(batch_freed)} "
            f"(total freed: {human_size(freed_bytes)})")

        if config.max_batches is not None and batch_num >= config.max_batches:
            log(f"\n  Reached --max-batches {config.max_batches}. "
                f"Stopping with {len(remaining)} targets remaining.")
            break


