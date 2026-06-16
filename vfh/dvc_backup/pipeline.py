"""Main backup pipeline: discovery → batching → add → push → verify → delete."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import tyro

from vfh.dvc_backup.backup_log import init_log, log, set_verbose
from vfh.dvc_backup.config import SPACE_MULTIPLIER, DvcBackupConfig
from vfh.dvc_backup.discovery import (
    AutoDiscovery,
    DiscoveryStrategy,
    VFHDiscovery,
    discover_all,
    discover_single,
    newest_mtime_in_tree,
)
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
from vfh.dvc_backup.verification import roundtrip_verify_batch


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
            log(f"ERROR: Smallest remaining target ({smallest.label}, "
                f"{human_size(smallest.size_bytes)}) exceeds the effective budget "
                f"({human_size(budget_bytes)}). Cannot proceed.")
            log(f"  Increase --space-budget-gb or free more disk space.")
            sys.exit(1)

        batches.append(batch)

    return batches


def _merge_stale_targets(
    targets: list[BackupTarget],
) -> list[BackupTarget]:
    """For targets with stale .dvc files (data dir newer than manifest),
    pull remote contents and merge with current local contents before re-adding.

    Returns list of targets that had merge conflicts (these are skipped).
    Mutates targets in-place: sets already_added=False for successfully merged targets.
    """
    import hashlib

    from vfh.dvc_backup.dvc_ops import dvc_pull

    conflicted: list[BackupTarget] = []

    for t in targets:
        if not t.already_added:
            continue

        try:
            dvc_m = t.dvc_file.stat().st_mtime
        except FileNotFoundError:
            t.already_added = False
            continue

        data_m = newest_mtime_in_tree(path=t.path)
        if data_m <= dvc_m:
            continue

        stale_minutes = (data_m - dvc_m) / 60
        log(f"  Stale .dvc detected for {t.label} "
            f"(data newer than manifest by {stale_minutes:.1f} min)")

        merge_tmp = t.path.parent / f".merge_tmp_{t.path.name}"
        if merge_tmp.exists():
            shutil.rmtree(merge_tmp)
        shutil.move(src=str(t.path), dst=str(merge_tmp))
        log(f"    Moved current local contents to {merge_tmp.name}")

        log(f"    Pulling remote contents for {t.dvc_file.name}...")
        pull_ok = dvc_pull(dvc_file=t.dvc_file)
        if not pull_ok or not t.path.exists():
            log(f"    WARNING: Could not pull remote contents — restoring local and skipping")
            if t.path.exists():
                shutil.rmtree(t.path)
            shutil.move(src=str(merge_tmp), dst=str(t.path))
            conflicted.append(t)
            continue

        n_merged = 0
        n_conflicts = 0
        conflict_files: list[str] = []

        def _md5(p: Path) -> str:
            h = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()

        for local_file in merge_tmp.rglob("*"):
            if not local_file.is_file():
                continue
            rel = local_file.relative_to(merge_tmp)
            remote_file = t.path / rel

            if not remote_file.exists():
                remote_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src=str(local_file), dst=str(remote_file))
                n_merged += 1
            else:
                if _md5(local_file) != _md5(remote_file):
                    n_conflicts += 1
                    conflict_files.append(str(rel))

        if n_conflicts > 0:
            log(f"    MERGE CONFLICT: {n_conflicts} file(s) differ between local and remote:")
            for cf in conflict_files[:10]:
                log(f"      - {cf}")
            if len(conflict_files) > 10:
                log(f"      ... and {len(conflict_files) - 10} more")
            log(f"    Reverting to local contents (not re-adding)")
            shutil.rmtree(t.path)
            shutil.move(src=str(merge_tmp), dst=str(t.path))
            conflicted.append(t)
            continue

        shutil.rmtree(merge_tmp)
        log(f"    Merged: {n_merged} new local file(s) added to {t.label}")
        t.already_added = False

    return conflicted


def _process_batch(
    batch: list[BackupTarget],
    all_targets: list[BackupTarget],
    logs_root: Path,
    config: DvcBackupConfig,
) -> tuple[list[BackupTarget], int]:
    """Process a single batch: add → push → verify → delete.

    Returns (verified_targets, freed_bytes).
    """
    conflicted = _merge_stale_targets(targets=batch)
    if conflicted:
        log(f"  Skipping {len(conflicted)} target(s) with merge conflicts")
        batch = [t for t in batch if t not in conflicted]
        if not batch:
            return [], 0
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
        log(f"  Clearing DVC cache before verification...")
        clear_cache_fast()
        results = roundtrip_verify_batch(
            targets=pushed,
            verify_temp_dir=config.verify_temp_dir,
            verify_on_local_fs=config.verify_on_local_fs,
        )
        for t, ok in zip(pushed, results):
            if ok:
                verified.append(t)
            else:
                log(f"    VERIFICATION FAILED for {t.label} — keeping local data")

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


def run_backup(
    config: DvcBackupConfig,
    strategy: DiscoveryStrategy | None = None,
) -> None:
    """Programmatic entry point. CLI `main()` wraps this."""
    if strategy is None:
        strategy = AutoDiscovery()

    set_verbose(verbose=config.verbose)

    # --- Test mode ---
    if config.setup_test_dir:
        setup_test_dir()
        return

    logs_root = Path(config.logs_root)
    if not config.path and not logs_root.is_dir():
        print(f"ERROR: logs_root not found: {logs_root}")
        sys.exit(1)

    init_log(logs_root=logs_root)
    log("=" * 60)
    log(f"DVC Backup started ({strategy.name})")
    log(f"  Config: space_budget_gb={config.space_budget_gb}, "
        f"max_batch_size_gb={config.max_batch_size_gb}, "
        f"skip_recent_minutes={config.skip_recent_minutes}, "
        f"max_batches={config.max_batches}, "
        f"skip_verify={config.skip_roundtrip_verify}, "
        f"path={config.path}, dry_run={config.dry_run}")

    skip_recent_seconds = float(config.skip_recent_minutes) * 60

    # --- Discovery ---
    if config.path:
        log(f"Single-dir mode: {config.path}")
        summaries = discover_single(
            run_dir=Path(config.path), strategy=strategy,
            skip_recent_seconds=skip_recent_seconds,
        )
    else:
        log("Discovering targets...")
        summaries = discover_all(
            logs_root=logs_root, strategy=strategy,
            skip_recent_seconds=skip_recent_seconds,
        )

    if not summaries:
        log("No untracked targets found. Nothing to do.")
        return

    print_summary(summaries=summaries)
    all_targets = [t for s in summaries for t in s.targets]

    if config.dry_run:
        log("\n(dry-run mode — no changes made)")
        return

    _check_free_space(
        logs_root=logs_root,
        all_targets=all_targets,
        space_budget_gb=config.space_budget_gb,
    )

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

    _run_batched(
        config=config, all_targets=all_targets, logs_root=logs_root,
        strategy=strategy, skip_recent_seconds=skip_recent_seconds,
    )

    log("\nDone.")


def main() -> None:
    run_backup(config=tyro.cli(DvcBackupConfig))


def _rediscover(
    config: DvcBackupConfig,
    strategy: DiscoveryStrategy,
    skip_recent_seconds: float,
    known_paths: set[Path],
) -> list[BackupTarget]:
    """Re-run discovery and return targets not seen before (by path).

    Catches new `global_step_N` dirs / rollouts that appeared during earlier
    batches, or runs that have since gone quiescent past the recency window.
    """
    if config.path:
        summaries = discover_single(
            run_dir=Path(config.path), strategy=strategy,
            skip_recent_seconds=skip_recent_seconds,
        )
    else:
        summaries = discover_all(
            logs_root=Path(config.logs_root), strategy=strategy,
            skip_recent_seconds=skip_recent_seconds,
        )
    new_targets: list[BackupTarget] = []
    for s in summaries:
        for t in s.targets:
            if t.path not in known_paths:
                new_targets.append(t)
    return new_targets


def _run_batched(
    config: DvcBackupConfig,
    all_targets: list[BackupTarget],
    logs_root: Path,
    strategy: DiscoveryStrategy,
    skip_recent_seconds: float,
) -> None:
    """Process targets in batches, with freed space rolling forward."""
    budget_bytes = config.space_budget_gb * 1e9
    max_batch_bytes = (
        config.max_batch_size_gb * 1e9 if config.max_batch_size_gb is not None else None
    )
    freed_bytes = 0.0

    # Build initial batches based on starting budget
    # We rebuild after each batch because freed_bytes grows
    remaining = list(all_targets)
    known_paths: set[Path] = {t.path for t in all_targets}
    batch_num = 0

    while remaining:
        # Re-discover before selecting this batch: pick up new global_step_N
        # dirs that landed during earlier batches, or runs that recently went
        # quiet past the skip_recent_minutes window.
        if batch_num > 0:
            newly_found = _rediscover(
                config=config, strategy=strategy,
                skip_recent_seconds=skip_recent_seconds,
                known_paths=known_paths,
            )
            if newly_found:
                total_new = sum(t.size_bytes for t in newly_found)
                log(f"Mid-backup re-discovery: +{len(newly_found)} new target(s) "
                    f"({human_size(total_new)})")
                remaining.extend(newly_found)
                known_paths |= {t.path for t in newly_found}

        effective_budget = budget_bytes + freed_bytes
        if max_batch_bytes is not None:
            effective_budget = min(effective_budget, max_batch_bytes)
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
            log(f"ERROR: Smallest remaining target ({smallest.label}, "
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


